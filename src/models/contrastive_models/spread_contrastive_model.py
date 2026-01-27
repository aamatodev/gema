#  Copyright (c) 2025.
#  University of West Florida (https://uwf.edu/intelligent-systems-and-robotics/)
#  All rights reserved.


from __future__ import annotations
from typing import Tuple
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_add_pool
from benchmarl.models.gnn import _batch_from_dense_to_ptg, _get_edge_index

# Local utilities ----------------------------------------------------------- #
from src.models.utils.goal_gen_utils import generate_objective_node_features
from src.models.utils.goal_gen_utils import extract_features_from_obs


class MLPEncoder(nn.Module):
    """Two‑layer MLP with ReLU activations.

    Args:
        input_size:  Dimension of the incoming features.
        hidden_size: Hidden layer width (default: 128).
        output_size: Dimension of the output embedding.
    """

    def __init__(
            self, input_size: int, output_size: int, hidden_size: int = 128
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x: Tensor) -> Tensor:  # noqa: D401 – simple signature OK
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class PoolEncoder(nn.Module):
    """Down‑projects a pooled graph representation to a fixed size."""

    def __init__(self, input_size: int, output_size: int = 16) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, output_size)

    def forward(self, x: Tensor) -> Tensor:  # noqa: D401
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# ----------------------------------------------------------------------------
#  Main model
# ----------------------------------------------------------------------------

class SpreadGraphContrastiveModel(nn.Module):
    """Contrastive encoder for a SimpleSpread batch.

    Given a batch of multi‑agent observations, the model builds two graphs:
    1. **Current** state – contains agents & landmark *positions*.
    2. **Objective** state – all agent positions replaced by landmark targets.

    Both graphs are encoded with a shared GAT‑v2 layer. The two pooled
    embeddings are concatenated and pushed through a final MLP to obtain the
    contrastive embedding *f(x)*.

    The forward pass additionally returns the two intermediate pooled graph
    embeddings, useful for contrastive or triplet losses.
    """

    def __init__(self, num_agents: int, device: torch.device | str = "cpu") -> None:
        super().__init__()
        self.num_agents = num_agents
        self.device = torch.device(device)

        # (x, y, type) → 16‑D node embedding
        self.node_encoder = MLPEncoder(input_size=3, output_size=16).to(self.device)

        # GNN: one‑hop attention with edge attributes (dx, dy, distance)
        self.gnn = GATv2Conv(in_channels=16, out_channels=32, heads=1, edge_dim=3).to(
            self.device
        )

        # Graph‑level pooling & projection
        self.pool_projection = PoolEncoder(input_size=32, output_size=16).to(self.device)

        # Final comparison MLP (concatenated [current | objective])
        self.metric_head = MLPEncoder(input_size=32, output_size=32).to(self.device)

    # --------------------------------------------------------------------- #
    #  Forward pass
    # --------------------------------------------------------------------- #

    def forward(
            self, observations: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Encode a batch of observations.

        Args:
            observations: Raw env observations with shape `[B, …]`.

        Returns:
            final_emb:      32‑D embedding for contrastive loss.
            current_pool:   16‑D pooled embedding for current graph.
            objective_pool: 16‑D pooled embedding for objective graph.
        """

        # ------------------------------------------------------------------ #
        # 1.Feature extraction (positions only needed here)
        # ------------------------------------------------------------------ #
        (
            agents_pos,  # (B, N, 2)
            _agents_vel,  # unused for now
            landmark_pos,  # (B, N, 2)
            _rel_landmarks_pos,
            _rel_other_pos,
        ) = extract_features_from_obs(observations)
        batch_size = agents_pos.size(0)
        self.num_agents = agents_pos.size(1)

        # ------------------------------------------------------------------ #
        # 2.Objective state – agents *should* sit on landmarks
        # ------------------------------------------------------------------ #
        (
            obj_pos,  # (B, N, 2)
            _obj_vel,
            _obj_rel_landmarks_pos,
            _obj_rel_other_pos,
        ) = generate_objective_node_features(landmark_pos)

        # ------------------------------------------------------------------ #
        # 3.Prepare graph node features
        #   type = 0 for agents, 1 for landmarks (helps GNN distinguish)
        # ------------------------------------------------------------------ #
        agent_type = torch.zeros((batch_size, self.num_agents, 1), device=self.device)
        lm_type = torch.ones_like(agent_type)

        # Current graph (agents + landmarks)
        cur_nodes = torch.cat([agents_pos, obj_pos], dim=1)  # (B, 2N, 2)
        cur_types = torch.cat([agent_type, lm_type], dim=1)
        cur_feats = torch.cat([cur_nodes, cur_types], dim=-1)

        # Objective graph (all positions set to obj_pos)
        obj_nodes = torch.cat([obj_pos, obj_pos], dim=1)  # same landmarks twice
        obj_feats = torch.cat([obj_nodes, cur_types], dim=-1)

        # ------------------------------------------------------------------ #
        # 4.Encode node features → graphs → pooled graph embeddings
        # ------------------------------------------------------------------ #
        cur_encoded = self.node_encoder(cur_feats)
        obj_encoded = self.node_encoder(obj_feats)

        num_total_nodes = self.num_agents * 2  # agents + landmarks per batch

        cur_graph = _batch_from_dense_to_ptg(
            x=cur_encoded,
            edge_index=_get_edge_index("full", False, num_total_nodes, self.device),
            self_loops=False,
            pos=cur_nodes,
            edge_radius=100
        )
        obj_graph = _batch_from_dense_to_ptg(
            x=obj_encoded,
            edge_index=_get_edge_index("full", False, num_total_nodes, self.device),
            self_loops=False,
            pos=obj_nodes,
        )

        # Shared GAT‑v2 over both graphs
        cur_h = self.gnn(cur_graph.x, cur_graph.edge_index, cur_graph.edge_attr)
        obj_h = self.gnn(obj_graph.x, obj_graph.edge_index, obj_graph.edge_attr)

        # Graph‑level pooling (add over nodes)
        cur_pool = self.pool_projection(global_add_pool(cur_h, cur_graph.batch))
        obj_pool = self.pool_projection(global_add_pool(obj_h, obj_graph.batch))

        # ------------------------------------------------------------------ #
        # 5.Metric head – produce final 32‑D embedding
        # ------------------------------------------------------------------ #
        final_emb = self.metric_head(torch.cat([cur_pool, obj_pool], dim=-1)).squeeze(1)
        final_emb_2 = self.metric_head(torch.cat([obj_pool, obj_pool], dim=-1)).squeeze(1)

        return final_emb, final_emb_2, cur_pool, obj_pool
