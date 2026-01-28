#  Copyright (c) 2025.
#  University of West Florida (https://uwf.edu/intelligent-systems-and-robotics/)
#  All rights reserved.


from dataclasses import dataclass, MISSING
from typing import Type

from benchmarl.algorithms import Mappo
from benchmarl.algorithms.common import Algorithm, AlgorithmConfig

import torch
from tensordict import TensorDictBase
import importlib


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
    offline_model_class: str = MISSING

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

        model_path = kwargs.pop("offline_model_path")
        model_class_name_path = kwargs.pop("offline_model_class")
        module_path, class_name = model_class_name_path.rsplit(".", 1)
        sge_cls = getattr(importlib.import_module(module_path), class_name)

        super().__init__(**kwargs)
        group = list(self.experiment.train_group_map.keys())[0]
        self.sge_model = sge_cls(num_agents=len(self.experiment.train_group_map[group]), device=self.device)
        self.sge_model.load_state_dict(torch.load(model_path, map_location=torch.device(self.device)))
        self.sge_model.eval()

    def process_batch(self, group: str, batch: TensorDictBase) -> TensorDictBase:
        # Here we process to batch to prepare it for the loss computation.
        A = len(self.group_map[group])

        agents_obs = batch.get((group, "observation"))

        E, B = batch.shape

        with torch.no_grad():
            _, _, current_state, goal_state = self.sge_model(agents_obs.view(E * B, A, -1))

            similarity = torch.nn.functional.cosine_similarity(current_state, goal_state, dim=-1).view(E, B, -1).repeat(1, 1, A)

            # add the similarity to the rewards
            batch.set(("next", group, "reward"),
                      batch.get(("next", group, "reward")) + similarity.to(
                          batch.get(("next", group, "reward")).device).reshape(
                          batch.get(("next", group, "reward")).shape),
                      inplace=True)

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
