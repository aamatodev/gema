#  Copyright (c) 2025.
#  University of West Florida (https://uwf.edu/intelligent-systems-and-robotics/)
#  All rights reserved.
import numpy as np
import torch
from tensordict import TensorDictBase


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def extract_features_from_obs(obs):
    current_obs = obs  # [batch, 3]
    target = obs[..., -3:]  # the last 3
    return current_obs, target


def generate_objective_node_features(targets):
    obj_feature = torch.cat([targets, targets], dim=-1)
    return obj_feature
