from __future__ import annotations
import torch
from benchmarl.models.gnn import _batch_from_dense_to_ptg, _get_edge_index
from torch import nn, Tensor
from torch_geometric.nn import GATv2Conv, global_add_pool
import torch.nn.functional as F

from src.models.utils.goal_gen_balancing_utils import extract_features_from_obs


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


class BalancingGraphContrastiveModel(nn.Module):
    def __init__(self, num_agents: int, device: torch.device | str = "cpu") -> None:
        super().__init__()
        self.num_agents = num_agents
        self.device = torch.device(device)

        # (x, y, type) → 16‑D node embedding
        self.node_encoder = MLPEncoder(input_size=2, output_size=16).to(self.device)

        # GNN: one‑hop attention with edge attributes (dx, dy, distance)
        self.gnn = GATv2Conv(in_channels=16, out_channels=32, heads=3, edge_dim=1).to(
            self.device
        )

        # Graph‑level pooling & projection
        self.pool_projection = PoolEncoder(input_size=96, output_size=32).to(self.device)

        # Final comparison MLP (concatenated [current | objective])
        self.metric_head = MLPEncoder(input_size=64, output_size=32).to(self.device)

    def forward(self, obs):
        current_status, target_status = extract_features_from_obs(obs)
        batch_size = current_status.shape[0]

        agent_type = torch.zeros((batch_size, self.num_agents, 1), device=self.device)
        lm_type = torch.ones_like(agent_type)

        current_status = current_status[..., 0, :]
        cur_nodes_f = current_status.view(batch_size, self.num_agents * 2, 1)

        cur_types = torch.cat([agent_type, lm_type], dim=1)
        cur_feats = torch.cat([cur_nodes_f, cur_types], dim=-1)

        # Objective graph (all positions set to obj_pos)
        target_status = target_status[..., 0, :]
        target_status_f = target_status.view(batch_size, self.num_agents, 1).repeat(1, 2, 1)
        obj_feats = torch.cat([target_status_f, cur_types], dim=-1)

        # Encode node features
        cur_encoded = self.node_encoder(cur_feats)
        obj_encoded = self.node_encoder(obj_feats)

        num_total_nodes = self.num_agents * 2  # agents + landmarks per batch

        cur_graph = _batch_from_dense_to_ptg(
            x=cur_encoded,
            edge_index=_get_edge_index("full", False, num_total_nodes, self.device),
            self_loops=False,
            pos=None,
        )

        obj_graph = _batch_from_dense_to_ptg(
            x=obj_encoded,
            edge_index=_get_edge_index("full", False, num_total_nodes, self.device),
            self_loops=False,
            pos=None,
        )

        # Shared GAT‑v2 over both graphs
        cur_h = self.gnn(cur_graph.x, cur_graph.edge_index, cur_graph.edge_attr)
        obj_h = self.gnn(obj_graph.x, obj_graph.edge_index, obj_graph.edge_attr)

        # Graph‑level pooling (add over nodes)
        cur_pool = self.pool_projection(global_add_pool(cur_h, cur_graph.batch))
        obj_pool = self.pool_projection(global_add_pool(obj_h, obj_graph.batch))

        final_emb = self.metric_head(torch.cat([cur_pool, obj_pool], dim=-1)).squeeze(1)

        return final_emb, cur_pool, obj_pool
