#  Copyright (c) 2025.
#  University of West Florida (https://uwf.edu/intelligent-systems-and-robotics/)
#  All rights reserved.


import torch
from tensordict import TensorDictBase


def extract_features_from_obs(obs):
    agents_pos = obs["agent_pos"]
    agents_vel = obs["agent_vel"]
    landmarks_pos = obs["landmark_pos"]
    relative_landmarks_pos = obs["relative_landmark_pos"]
    relative_other_pos = obs["other_pos"]

    return agents_pos, agents_vel, landmarks_pos, relative_landmarks_pos, relative_other_pos


def generate_objective_node_features(landmark_pos):
    """
    Build per‑landmark objective / relative‑position tensors that scale to any
    number of landmarks `N`.

    landmark_pos: [B, N, 6]   –x/y coordinates of each landmark
    returns:
        objective_pos         [B, N, 2]
        objective_vel         [B, N, 2]   (zeros)
        relative_landmarks_pos[B, N, 2N]  (all landmarks wrt landmark‑1)
        relative_other_pos    [B, N, 2(N‑1)] (all *other* landmarks)
    """
    B, n_landmark, _ = landmark_pos.shape
    device = landmark_pos.device

    # Pick landmark‑1 as the “objective” and replicate so every row has a target
    objective_pos = landmark_pos[:, 1, :].view(B, n_landmark, 2)
    objective_vel = torch.zeros_like(objective_pos)

    # [B, N, 2N] – subtract objective position from every landmark pair (x,y)
    relative_landmarks_pos = landmark_pos - objective_pos.repeat(1, 1, n_landmark)

    # ---------- robust, size‑independent index generation ----------
    # Full list 0 … 2N‑1  (x0,y0,x1,y1,…)
    all_idx = torch.arange(2 * n_landmark, device=device)

    # For each landmark i, remove its own xᵢ and yᵢ slots
    idx_rows = [
        all_idx[(all_idx != 2 * i) & (all_idx != 2 * i + 1)]
        for i in range(n_landmark)
    ]  # list of N tensors, each length 2(N‑1)

    # Stack and broadcast to batches: [B, N, 2(N‑1)]
    indices = torch.stack(idx_rows).unsqueeze(0).expand(B, -1, -1)

    # Gather the “other‑landmark” coordinates
    relative_other_pos = torch.gather(relative_landmarks_pos, 2, indices)

    return objective_pos, objective_vel, relative_landmarks_pos, relative_other_pos


def split_spread_observation(obs: TensorDictBase):
    """Extract tensors of interest from a SimpleSpread observation dict.

    Returns:
        Tuple containing:
            agents_pos (B, N, 2)
            agents_vel (B, N, 2)
            landmarks_pos (B, N, 2)
            rel_landmarks_pos (B, N, 2)
            rel_other_pos (B, N, 2)
    """
    return (
        obs["agent_pos"],
        obs["agent_vel"],
        obs["landmark_pos"],
        obs["relative_landmark_pos"],
        obs["other_pos"],
    )


def create_objective_features(
        landmark_pos: torch.Tensor,
        n_agents: int,
):
    """Generate 'ideal' (objective) features where each agent sits on a landmark.

    Args:
        landmark_pos: (B, N, 2) absolute landmark positions.
        n_agents:     Number of agents (== number of landmarks in SimpleSpread).

    Returns:
        Tuple of tensors describing the objective state:
            objective_pos               (B, N, 2)
            objective_vel               (B, N, 2) – always zero
            rel_landmarks_pos_objective (B, N, 2)
            rel_other_pos_objective     (B, N, N‑1, 2)
    """
    bsz, n_landmarks, _ = landmark_pos.shape
    objective_pos = landmark_pos[:, 1, :].reshape(-1, n_landmarks, 2).clone()
    objective_vel = torch.zeros_like(objective_pos)

    # Position of every landmark relative to every other landmark
    rel_landmarks_pos = landmark_pos - objective_pos.repeat(1, 1, n_landmarks)

    # For each agent i remove its self‑distance to produce "other landmark" features
    indices = []
    for i in range(n_agents):
        start = i * 2
        indices.append(
            [j for j in range(2 * n_agents) if j not in (start, start + 1)]
        )

    idx = torch.tensor(indices, device=landmark_pos.device)  # (N, 2N‑2)
    idx = idx.unsqueeze(0).expand(bsz, -1, -1)  # (B, N, 2N‑2)
    rel_other_pos = torch.gather(rel_landmarks_pos, 2, idx)

    return objective_pos, objective_vel, rel_landmarks_pos, rel_other_pos
