#  Copyright (c) 2025.
#  University of West Florida (https://uwf.edu/intelligent-systems-and-robotics/)
#  All rights reserved.


from dataclasses import dataclass, MISSING
from typing import Dict, Iterable, Tuple, Type

from benchmarl.algorithms import Mappo
from benchmarl.algorithms.common import Algorithm, AlgorithmConfig

import torch
from tensordict import TensorDictBase
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.data import Composite, Unbounded, Bounded

from src.utilts.graph_utils import generate_graph

from benchmarl.models.gnn import _batch_from_dense_to_ptg


@dataclass
class GemaConfig(AlgorithmConfig):
    """Configuration dataclass for :class:`~benchmarl.algorithms.Mappo`."""

    # Mappo Config Params
    share_param_critic: bool = MISSING
    clip_epsilon: float = MISSING
    entropy_coef: float = MISSING
    critic_coef: float = MISSING
    loss_critic_type: str = MISSING
    lmbda: float = MISSING
    scale_mapping: str = MISSING
    use_tanh_normal: bool = MISSING
    minibatch_advantage: bool = MISSING

    # Gema Config Params
    offline_model_path: str = MISSING

    @staticmethod
    def associated_class() -> Type[Algorithm]:
        return Gema

    @staticmethod
    def supports_continuous_actions() -> bool:
        return True

    @staticmethod
    def supports_discrete_actions() -> bool:
        return True

    @staticmethod
    def on_policy() -> bool:
        return True

    @staticmethod
    def has_centralized_critic() -> bool:
        return True


class Gema(Mappo):
    def __init__(
            self, **kwargs
    ):
        # In the init function you can define the init parameters you need, just make sure
        # to pass the kwargs to the super() class

        model_path = kwargs.pop("offline_model_path")
        super().__init__(**kwargs)
        self.sge_model = SMACV2GraphContrastiveModel(device=self.device,
                                                     enemy_feature_idx=list(range(4, 49)))
        self.sge_model.load_state_dict(torch.load(model_path, map_location=torch.device(self.device)))
        self.sge_model.eval()

    def get_critic(self, group: str) -> TensorDictModule:
        n_agents = len(self.group_map[group])
        if self.share_param_critic:
            critic_output_spec = Composite({"state_value": Unbounded(shape=(1,))})
        else:
            critic_output_spec = Composite(
                {
                    group: Composite(
                        {"state_value": Unbounded(shape=(n_agents, 1))},
                        shape=(n_agents,),
                    )
                }
            )

        if self.state_spec is not None:
            input_has_agent_dim = False
            critic_input_spec = self.state_spec

        else:
            input_has_agent_dim = True
            critic_input_spec = Composite(
                {group: self.observation_spec[group].clone().to(self.device)}
            )
        # modify the state spec to include the current and goal state from the sge model
        new_state = Composite({
            "update_state": Bounded(
                low=-1.0,
                high=1.0,
                shape=torch.Size(((self.state_spec["state"].shape[0] + 65),)),
                device=self.device,
                dtype=torch.float32,
            )})

        value_module = self.critic_model_config.get_model(
            input_spec=new_state,
            output_spec=critic_output_spec,
            n_agents=n_agents,
            centralised=True,
            input_has_agent_dim=input_has_agent_dim,
            agent_group=group,
            share_params=self.share_param_critic,
            device=self.device,
            action_spec=self.action_spec,
        )

        if self.share_param_critic:
            expand_module = TensorDictModule(
                lambda value: value.unsqueeze(-2).expand(
                    *value.shape[:-1], n_agents, 1
                ),
                in_keys=["state_value"],
                out_keys=[(group, "state_value")],
            )
            value_module = TensorDictSequential(value_module, expand_module)

        return value_module

    def process_batch(self, group: str, batch: TensorDictBase) -> TensorDictBase:
        # Here we process to batch to prepare it for the loss computation.
        agents_obs = batch[(group, "observation")]
        agents_obs = agents_obs.reshape(-1, agents_obs.shape[-1])  # Flatten batch dimensions

        og_view = batch[("info", "full_obs")].view(-1, 5, 92)
        graphs_from_batch = _batch_from_dense_to_ptg(batch_size=og_view.shape[0],
                                                     node_features=agents_obs,
                                                     edge_attr=None,
                                                     n_agents=5,
                                                     device=self.device,
                                                     use_radius=True)

        with torch.no_grad():
            embeddings, final_embeddings, current_state, goal_state = self.sge_model(graphs_from_batch)

            similarity = torch.nn.functional.cosine_similarity(embeddings, final_embeddings, dim=-1)
            similarity = (similarity + 1) / 2
            # add the similarity to the rewards
            batch.set(("next", "reward"),
                      batch.get(("next", "reward")) + similarity.to(batch.get(("next", "reward")).device).reshape(
                          batch.get(("next", "reward")).shape),
                      inplace=True)

            self.experiment.logger.log({
                "enhanced_reward": batch[("next", "reward")].mean(-1).sum(-1).sum(-1).mean(),
                "distance": similarity.mean()
            })

        state_shape = batch.get(("state")).shape
        updated_state = torch.cat([batch.get("state"),
                                   embeddings.view(state_shape[0], state_shape[1], -1),
                                   final_embeddings.view(state_shape[0], state_shape[1], -1),
                                   similarity.view(state_shape[0], state_shape[1], 1)
                                   ], dim=-1)
        batch.set("update_state", updated_state)
        batch.set(("next", "update_state"), updated_state)

        # Standard MAPPO processing
        batch = super().process_batch(group, batch)

        return batch

    def process_loss_vals(
            self, group: str, loss_vals: TensorDictBase
    ) -> TensorDictBase:
        # Here you can modify the loss_vals tensordict containing entries loss_name->loss_value
        # For example you can sum two entries in a new entry to optimize them together.
        loss_vals = super().process_loss_vals(group, loss_vals)

        return loss_vals
