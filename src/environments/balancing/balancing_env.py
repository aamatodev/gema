#  Copyright (c) 2025.
#  University of West Florida (https://uwf.edu/intelligent-systems-and-robotics/)
#  All rights reserved.

import contextlib
import itertools

from pettingzoo import ParallelEnv
from pettingzoo.utils import wrappers
from gymnasium import spaces
import numpy as np
from pettingzoo.utils.env import AgentID


def env():
    return wrappers.parallel_to_aec(LoadBalancingEnv())


@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


class LoadBalancingEnv(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "load_balancing_parallel_v2"}

    def __init__(self, max_steps=100, target_loads=(0.6, 0.4, 0.8), render_mode="none"):
        super().__init__()
        self.steps = 0
        self.current_seed = 0
        self.norm_load = None
        self.load = None
        self.n_agents = 3
        self.agents = [f"agent_{i}" for i in range(self.n_agents)]
        self.possible_agents = self.agents[:]
        self.agent_idx = {agent: i for i, agent in enumerate(self.agents)}
        self.max_steps = max_steps
        self.target_loads = np.array(target_loads)

        self.single_action_space = spaces.Discrete(3)  # 0: accept, 1: send to agent1, 2: send to agent2
        self.single_observation_spaces = spaces.Box(low=0, high=100, shape=(self.n_agents * 2,), dtype=np.float32)

        self.action_spaces = {agent: self.single_action_space for agent in self.agents}
        self.observation_spaces = {agent: self.single_observation_spaces for agent in self.agents}

        self.used_seeds = set()
        self.seed_list = [self.used_seeds.add(i) for i in range(100000)]

        self.isFirst = True
        # self.reset(self.current_seed)

    def action_space(self, agent: AgentID):
        return self.single_action_space

    def observation_space(self, agent: AgentID):
        return self.single_observation_spaces

    def reset(self, seed=0, options=None):

        if self.isFirst:
            self.current_seed = seed
            self.isFirst = False

        self.current_seed += 1

        with temp_seed(self.current_seed):
            self.load = np.random.randint(low=0, high=100, size=self.n_agents).astype(np.float32)
            self.norm_load = self.load / np.sum(self.load) * 100  # Normalize load
            self.target_loads = self.random_target_loads(min_spread=20)

        self.dones = {agent: False for agent in self.agents}
        self.infos = {agent: {"contrastive_reward": 0, "vanilla_reward": 0, "winning_step": 0} for agent in self.agents}
        self.rewards = {agent: 0.0 for agent in self.agents}
        self.steps = 0
        self.winning_steps = 0

        return self._get_observations(), self.infos

    def _get_observations(self):
        obs = {}

        for i, agent in enumerate(self.agents):
            own = self.norm_load[i]
            others = [self.norm_load[j] for j in range(self.n_agents) if j != i]
            agent_loads = [own] + others

            own_target = self.target_loads[i]
            others_target = [self.target_loads[j] for j in range(self.n_agents) if j != i]
            target_loads = [own_target] + others_target

            full_obs = agent_loads + target_loads  # fixed order: [t0, t1, t2]
            obs[agent] = np.array(full_obs, dtype=np.float32)

        return obs

    def random_target_loads(self, min_spread=0.2):

        counter = 0

        with temp_seed(self.current_seed):
            while True:
                counter += 1
                vals = np.random.randint(low=0, high=100, size=self.n_agents).astype(np.float32)
                norm_vals = vals / np.sum(vals) * 100  # Normalize load

                if np.max(norm_vals) - np.min(norm_vals) > min_spread and not np.allclose(self.norm_load, norm_vals,
                                                                                          atol=min_spread):
                    return norm_vals

                if counter > 600:
                    self.reset(self.current_seed)

    def step(self, actions):
        self.steps += 1
        self.rewards = {agent: 0.0 for agent in self.agents}

        # Each agent has a job to process
        for agent, action in actions.items():
            idx = self.agent_idx[agent]

            if action == 0:
                self.load[idx] += 1
            else:
                destination_idx = (idx + action) % self.n_agents  # either +1 or +2
                self.load[destination_idx] += 1

        total_load = np.sum(self.load)
        if total_load > 0:
            self.norm_load = (self.load / total_load) * 100
            satisfied_targets_np = self.satisfied_targets_np(self.norm_load, self.target_loads, atol=3)

            if satisfied_targets_np == 2:
                self.winning_steps += 1
                self.infos[agent]["winning_step"] = self.winning_steps

            for agent in self.agents:
                self.rewards[agent] = satisfied_targets_np - 1

        self.dones = {agent: self.steps >= self.max_steps for agent in self.agents}
        return self._get_observations(), self.rewards, self.dones, self.dones, self.infos

    def info(self, agent: AgentID):
        return self.infos[agent]

    def render(self):
        print(f"Step {self.steps} - Load: {self.norm_load} - Target: {self.target_loads} - rewards: {self.rewards}")

    @staticmethod
    def satisfied_targets_np(loads, targets, atol: float = 0.0):
        best = 0
        for perm in itertools.permutations(targets):
            cand = sum(abs(l - t) <= atol for l, t in zip(loads, perm))
            best = max(best, cand)
            if best == 3:  # can’t do better than perfect
                break
        label = 0 if best == 0 else 1 if best == 1 else 2
        return label
