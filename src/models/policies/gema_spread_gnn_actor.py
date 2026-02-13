from __future__ import annotations

import importlib
from dataclasses import dataclass, MISSING
from typing import Type, Sequence, Optional

import numpy as np
import torch
from benchmarl.models.gnn import _batch_from_dense_to_ptg, _get_edge_index
from torch import nn, Tensor
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

from tensordict import TensorDictBase
from benchmarl.models.common import Model, ModelConfig

from src.models.utils.goal_gen_spread_utils import split_spread_observation, create_objective_features


# --------------------------------------------------------------------------- #
#                               Helper Modules                                #
# --------------------------------------------------------------------------- #

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class FinalEncoder(nn.Module):
    """One‑layer MLP used to embed raw node features."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, 256)
        self.linear1 = nn.Linear(256, 256)
        self.linear2 = nn.Linear(256, 256)
        self.linear3 = nn.Linear(256, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, F)
        l1 = torch.relu(self.linear(x))
        l2 = torch.relu(self.linear1(l1))
        l3 = torch.relu(self.linear2(l2))
        l4 = self.linear3(l3)

        return l4


class Encoder(nn.Module):
    """One‑layer MLP used to embed raw node features."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = layer_init(nn.Linear(in_features, out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, F)
        return self.linear(x)


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

    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# --------------------------------------------------------------------------- #
#                            Main Agent‑Level Model                           #
# --------------------------------------------------------------------------- #

class GemaSpreadGnnActor(Model):
    """Actor network that augments each agent’s input with a learned
    representation of the *objective* (all agents sitting on landmarks)."""

    def __init__(
            self,
            activation_class: Type[nn.Module],
            **kwargs,
    ):
        model_path = kwargs.pop("offline_model_path")
        model_class = kwargs.pop("offline_model_class")

        # Initialise BenchMARL base Model
        super().__init__(
            input_spec=kwargs.pop("input_spec"),
            output_spec=kwargs.pop("output_spec"),
            agent_group=kwargs.pop("agent_group"),
            input_has_agent_dim=kwargs.pop("input_has_agent_dim"),
            n_agents=kwargs.pop("n_agents"),
            centralised=kwargs.pop("centralised"),
            share_params=kwargs.pop("share_params"),
            device=kwargs.pop("device"),
            action_spec=kwargs.pop("action_spec"),
            model_index=kwargs.pop("model_index"),
            is_critic=kwargs.pop("is_critic"),
        )

        self.activation_class = activation_class
        self.output_features = self.output_leaf_spec.shape[-1]
        # Remove landmark positions (2 dims per agent) from the raw observation feature count
        self.input_features = (
                sum(spec.shape[-1] for spec in self.input_spec.values(True, True))
                - self.n_agents * 2
        )

        # ----------------------------- Sub‑modules -------------------------- #

        # 1) Graph‑level communication between agents
        self.gnn = GATv2Conv(16, 16, heads=3, edge_dim=3).to(self.device)

        self.final_mlp = FinalEncoder(49, self.output_features).to(self.device)

        # 3) Node encoder shared by agents & landmarks – (x, y, type) → 16‑D
        self.node_encoder = MLPEncoder(input_size=3, output_size=16).to(self.device)

        # 4) Pretrained contrastive model to get global context
        self.sge_model = model_class(num_agents=self.n_agents, device=self.device)
        self.sge_model.load_state_dict(torch.load(model_path, map_location=torch.device(self.device)))
        self.sge_model.eval()

    # ----------------------------- Forward Pass ------------------------------ #

    def _forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        if not self.input_has_agent_dim:
            raise ValueError("Model expects per‑agent dimension in the input.")

        # ---------------- 1. Parse observation ---------------- #
        obs = tensordict.get("agents")["observation"]
        (
            agents_pos,
            _agents_vel,  # Velocities unused for now
            landmark_pos,
            _rel_landmarks_pos,
            _rel_other_pos,
        ) = split_spread_observation(obs)

        batch_size = agents_pos.size(0)

        # ---------------- 2. Build *objective* state ------------ #
        (
            obj_pos,
            _obj_vel,
            _obj_rel_landmarks_pos,
            _obj_rel_other_pos,
        ) = create_objective_features(landmark_pos, agents_pos.shape[1])

        # ---------------- 3. Encode graph nodes ----------------- #
        # type feature helps GNN distinguish agent vs. landmark
        agent_type = torch.zeros((batch_size, agents_pos.shape[1], 1), device=self.device)
        obj_type = torch.ones_like(agent_type)

        cur_nodes = torch.cat([agents_pos, obj_pos], dim=1).to(self.device)  # (B, 2N, 2)
        cur_types = torch.cat([agent_type, obj_type], dim=1).to(self.device)
        cur_feats = torch.cat([cur_nodes, cur_types], dim=-1).to(self.device)  # (B*2N, 3)

        node_embeddings = self.node_encoder(cur_feats)  # (B*2N, 16)

        # ---------------- 4. Global contrastive context ---------- #
        with torch.no_grad():
            final_emb, final_emb_2, current_state_embedding, objective_state_embedding = self.sge_model(obs)

        similarity = F.cosine_similarity(current_state_embedding, objective_state_embedding, dim=-1).unsqueeze(
            1).unsqueeze(1)
        similarity = similarity.repeat(1, agents_pos.shape[1], 1)  # (B, N, 1)

        # ---------------- 5. Build batched graph ---------------- #
        num_total_nodes = agents_pos.shape[1] * 2

        graph_repr = _batch_from_dense_to_ptg(
            x=node_embeddings,
            edge_index=_get_edge_index("full", False, num_total_nodes, self.device),
            self_loops=False,
            pos=cur_nodes,
            edge_radius=100
        )

        cur_h = self.gnn(graph_repr.x, graph_repr.edge_index, graph_repr.edge_attr).view(batch_size, num_total_nodes,
                                                                                         -1)

        # ---------------- 6. Concatenate all features ------------ #
        agent_inputs = torch.cat([cur_h[:, :agents_pos.shape[1], :], similarity], dim=2)  # (B, N, 81)

        # ---------------- 7. Per‑agent policy head --------------- #
        actions = self.final_mlp(agent_inputs.view(batch_size, agents_pos.shape[1], -1))

        # ---------------- 8. Write outputs into TensorDict ------- #
        tensordict.set(self.out_keys[0], actions)

        return tensordict


# --------------------------------------------------------------------------- #
#                           Hydra / YAML Config Glue                          #
# --------------------------------------------------------------------------- #

@dataclass
class GemaSpreadGnnActorConfig(ModelConfig):
    """Hydra config schema for :class:`SimpleSpreadObjectiveSharing`."""

    activation_class: Type[nn.Module] = MISSING

    # Gema Config Params
    offline_model_path: str = MISSING
    offline_model_class: str = MISSING

    activation_kwargs: Optional[dict] = None
    norm_class: Optional[Type[nn.Module]] = None
    norm_kwargs: Optional[dict] = None

    @staticmethod
    def associated_class():
        return GemaSpreadGnnActor
