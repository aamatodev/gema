#  Copyright (c) 2025.
#  University of West Florida (https://uwf.edu/intelligent-systems-and-robotics/)
#  All rights reserved.


from __future__ import annotations

import importlib
from dataclasses import dataclass, MISSING
from pathlib import Path
from typing import Type, Sequence
import torch
from benchmarl.models.gnn import _batch_from_dense_to_ptg, _get_edge_index
from torch_geometric.nn import GATv2Conv
from benchmarl.models.common import Model, ModelConfig
from tensordict import TensorDictBase
from torch import nn, Tensor
import torch.nn.functional as F

from src.models.utils.goal_gen_balancing_utils import layer_init, extract_features_from_obs


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

class GemaBalancingGnnActor(Model):
    def __init__(
            self,
            activation_class: Type[nn.Module],
            **kwargs,
    ):
        model_path = kwargs.pop("offline_model_path")
        model_class = kwargs.pop("offline_model_class")

        # Remember the kwargs to the super() class
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

        self.output_features = self.output_leaf_spec.shape[-1]
        self.input_features = sum(
            [spec.shape[-1] for spec in self.input_spec.values(True, True)]
        ) - self.n_agents * 2  # we remove the "landmark_pos" from the input features
        self.activation_class = activation_class

        # ----------------------------- Sub‑modules -------------------------- #

        # 1) Node encoder shared by agents & landmarks – (x, y, type) → 16‑D
        self.node_encoder = MLPEncoder(input_size=2, output_size=16).to(self.device)

        # 2) Graph‑level communication between agents
        self.gnn = GATv2Conv(16, 16, heads=3, edge_dim=1).to(self.device)

        # 3) Final MLP per‑agent policy head
        self.final_mlp = FinalEncoder(49, self.output_features).to(self.device)

        # 4) Pre‑trained contrastive model providing a *global* context vector
        self.sge_model = model_class(num_agents=self.n_agents, device=self.device)
        self.sge_model.load_state_dict(torch.load(model_path, map_location=torch.device(self.device)))
        self.sge_model.eval()

    def _perform_checks(self):
        super()._perform_checks()

    def _forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        obs = tensordict["agent"]["observation"]
        current_status, target_status = extract_features_from_obs(obs)

        if len(obs.shape) == 2:
            # If the observation is 2D, we need to add a dimension
            obs = obs.unsqueeze(0)
            current_status = current_status.unsqueeze(0)

        batch_size = current_status.shape[0]
        agent_type = torch.zeros((batch_size, self.n_agents, 1), device=self.device)
        lm_type = torch.ones_like(agent_type)

        current_status = current_status[..., 0, :]
        cur_nodes_f = current_status.view(batch_size, self.n_agents * 2, 1)

        cur_types = torch.cat([agent_type, lm_type], dim=1)
        cur_feats = torch.cat([cur_nodes_f, cur_types], dim=-1)

        # Encode node features
        cur_encoded = self.node_encoder(cur_feats)

        num_total_nodes = self.n_agents * 2  # agents + landmarks per batch

        cur_graph = _batch_from_dense_to_ptg(
            x=cur_encoded,
            edge_index=_get_edge_index("full", False, num_total_nodes, self.device),
            self_loops=False,
            pos=None,
        )

        # Shared GAT‑v2 over both graphs
        cur_h = self.gnn(cur_graph.x, cur_graph.edge_index, cur_graph.edge_attr)

        # ---------------- 4. Global contrastive context ---------- #
        with torch.no_grad():
            _, _, current_state_embedding, objective_state_embedding = self.sge_model(obs)

        similarity = F.cosine_similarity(current_state_embedding, objective_state_embedding, dim=-1).unsqueeze(
            1).unsqueeze(1)
        similarity = similarity.repeat(1, self.n_agents, 1)  # (B, N, 1)

        # ---------------- 6. Concatenate all features ------------ #
        agent_inputs = torch.cat([cur_h.view((obs.shape[0], self.n_agents * 2, -1))[:, :self.n_agents, :],
                                  similarity], dim=2)  # (B, N, 81)

        # ---------------- 7. Per‑agent policy head --------------- #
        actions = self.final_mlp(agent_inputs.view(batch_size, self.n_agents, -1))

        # ---------------- 8. Write outputs into TensorDict ------- #
        if len(tensordict["agent"]["observation"].shape) == 2:
            # If the observation is 2D, we need to remove the dimension
            actions = actions.squeeze(0)

        tensordict.set(self.out_keys[0], actions)

        return tensordict


@dataclass
class GemaBalancingGnnActorConfig(ModelConfig):
    # The config parameters for this class, these will be loaded from yaml
    activation_class: Type[nn.Module] = MISSING
    # Gema Config Params
    offline_model_path: str = MISSING
    offline_model_class: str = MISSING


    @staticmethod
    def associated_class():
        # The associated algorithm class
        return GemaBalancingGnnActor

