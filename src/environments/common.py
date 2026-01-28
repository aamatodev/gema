#  Copyright (c) 2025.
#  University of West Florida (https://uwf.edu/intelligent-systems-and-robotics/)
#  All rights reserved.

import copy
from typing import Callable, Dict, List, Optional

import torchrl
from torchrl.data import Composite
from torchrl.envs import EnvBase, PettingZooEnv, PettingZooWrapper
from torchrl.envs.libs.vmas import VmasEnv

from benchmarl.environments.common import Task, TaskClass
from benchmarl.utils import DEVICE_TYPING
from .spread.spread_graph_obs import SpreadGraphObsScenario
from src.environments.balancing.balancing_env import LoadBalancingEnv


class GEMAClass(TaskClass):
    def get_env_fun(
            self,
            num_envs: int,
            continuous_actions: bool,
            seed: Optional[int],
            device: DEVICE_TYPING,
    ) -> Callable[[], EnvBase]:
        config = copy.deepcopy(self.config)

        if self.name in {GEMATask.SPREAD.name}:
            task = SpreadGraphObsScenario()
            env = VmasEnv(
                scenario=task,
                num_envs=num_envs,
                continuous_actions=continuous_actions,
                seed=seed,
                device=device,
                categorical_actions=True,
                clamp_actions=True,
                **config,
            )
            env.scenario_name.viewer_zoom = 1.5

        if self.name in {GEMATask.LOAD_BALANCING.name}:
            env = PettingZooWrapper(
                env=LoadBalancingEnv(),
                return_state=False,
                use_mask=False,
                group_map=None,
                categorical_actions=True,
                device=device,
                seed=seed,
            )

        return lambda: env

    def supports_continuous_actions(self) -> bool:
        if self.name in {
            GEMATask.LOAD_BALANCING.name,
        }:
            return False

        return True

    def supports_discrete_actions(self) -> bool:
        return True

    def has_render(self, env: EnvBase) -> bool:
        return True

    def max_steps(self, env: EnvBase) -> int:
        return self.config["max_steps"]

    def group_map(self, env: EnvBase) -> Dict[str, List[str]]:
        if hasattr(env, "group_map"):
            return env.group_map
        return {"agents": [agent.name for agent in env.agents]}

    def state_spec(self, env: EnvBase) -> Optional[Composite]:
        if "state" in env.observation_spec:
            return Composite({"state": env.observation_spec["state"].clone()})
        return None

    def action_mask_spec(self, env: EnvBase) -> Optional[Composite]:
        if self.name in {GEMATask.LOAD_BALANCING.name}:
            observation_spec = env.observation_spec.clone()
            for group in self.group_map(env):
                group_obs_spec = observation_spec[group]
                for key in list(group_obs_spec.keys()):
                    if key != "action_mask":
                        del group_obs_spec[key]
                if group_obs_spec.is_empty():
                    del observation_spec[group]
            if "state" in observation_spec.keys():
                del observation_spec["state"]
            if observation_spec.is_empty():
                return None
        return None

    def observation_spec(self, env: EnvBase) -> Composite:
        if self.name in {GEMATask.LOAD_BALANCING.name}:
            observation_spec = env.observation_spec.clone()
            for group in self.group_map(env):
                group_obs_spec = observation_spec[group]
                for key in list(group_obs_spec.keys()):
                    if key != "observation":
                        del group_obs_spec[key]
            if "state" in observation_spec.keys():
                del observation_spec["state"]
            return observation_spec
        elif self.name in {GEMATask.SPREAD.name}:
            observation_spec = env.full_observation_spec_unbatched.clone()
            for group in self.group_map(env):
                if "info" in observation_spec[group]:
                    del observation_spec[(group, "info")]
            return observation_spec

    def info_spec(self, env: EnvBase) -> Optional[Composite]:
        if self.name in {GEMATask.LOAD_BALANCING.name}:
            observation_spec = env.observation_spec.clone()
            for group in self.group_map(env):
                group_obs_spec = observation_spec[group]
                for key in list(group_obs_spec.keys()):
                    if key != "info":
                        del group_obs_spec[key]
            if "state" in observation_spec.keys():
                del observation_spec["state"]
            return observation_spec
        elif self.name in {GEMATask.SPREAD.name}:
            info_spec = env.full_observation_spec_unbatched.clone()
            for group in self.group_map(env):
                del info_spec[(group, "observation")]
            for group in self.group_map(env):
                if "info" in info_spec[group]:
                    return info_spec
            return None

    def action_spec(self, env: EnvBase) -> Composite:
        if self.name in {GEMATask.LOAD_BALANCING.name}:
            return env.full_action_spec
        return env.full_action_spec_unbatched

    @staticmethod
    def env_name() -> str:
        return "gema"


class GEMATask(Task):
    """Enum for MAGMA tasks."""

    SPREAD = None
    LOAD_BALANCING = None

    @staticmethod
    def associated_class():
        return GEMAClass
