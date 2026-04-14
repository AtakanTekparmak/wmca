"""Generic data generation for any AutumnEnv subclass.

Works for environments with or without agents:
- No agent (pure dynamics): generates (state, next_state) pairs.
  in_channels = out_channels = n_cell_types
- With agent: generates (state + action_field, next_state) pairs.
  in_channels = n_cell_types + n_action_channels
"""
from __future__ import annotations

from typing import Type

import numpy as np
import torch

from wmca.benchmarks import BenchmarkData, _split_trajectories, _to_torch
from wmca.envs.autumn.base import AutumnEnv


def generate_autumn_transitions(
    env_class: Type[AutumnEnv],
    n_trajectories: int = 500,
    episode_length: int = 30,
    action_policy: str = "random",
    seed: int = 42,
    grid_size: int | None = None,
    device: str | torch.device = "cpu",
    **env_kwargs,
) -> BenchmarkData:
    """Generate (state+action, next_state) pairs from an AutumnEnv.

    For pure-dynamics envs (no agent): X and Y are both one-hot grids.
    For action-conditioned envs: X = [state_one_hot | action_field].

    Parameters
    ----------
    env_class : subclass of AutumnEnv
    n_trajectories : number of independent episodes
    episode_length : steps per episode
    action_policy : "random" or "noop" (only dynamics, action=0 every step)
    seed : random seed
    grid_size : override env default grid size
    device : torch device for output tensors
    **env_kwargs : forwarded to env_class constructor
    """
    device = torch.device(device)
    rng = np.random.default_rng(seed)

    # Build a probe env to get dimensions
    probe_kwargs = dict(env_kwargs)
    if grid_size is not None:
        probe_kwargs["grid_size"] = grid_size
    probe = env_class(seed=seed, **probe_kwargs)
    probe.reset()
    n_cell_types = probe.n_cell_types
    gs = probe.grid_size
    has_agent = probe.agent_pos is not None

    # Determine action space
    if has_agent:
        n_actions = probe.N_ACTIONS
    else:
        # Pure dynamics: only noop action, no action encoding needed
        n_actions = 0

    # Collect trajectories as one-hot: (n_traj, episode_length+1, C, H, W)
    trajs = np.zeros(
        (n_trajectories, episode_length + 1, n_cell_types, gs, gs),
        dtype=np.float32,
    )
    # Action storage (only if agent-based)
    if has_agent:
        actions = np.zeros(
            (n_trajectories, episode_length), dtype=np.int32,
        )
        agent_positions = np.zeros(
            (n_trajectories, episode_length, 2), dtype=np.int32,
        )

    for i in range(n_trajectories):
        env_seed = int(rng.integers(0, 2**31))
        env = env_class(seed=env_seed, **probe_kwargs)
        obs = env.reset()
        trajs[i, 0] = obs

        for t in range(episode_length):
            if has_agent:
                if action_policy == "random":
                    action = int(rng.integers(0, n_actions))
                else:
                    action = 0  # noop
                actions[i, t] = action
                agent_positions[i, t] = env.agent_pos
            else:
                action = 0  # noop for pure dynamics

            obs = env.step(action)
            trajs[i, t + 1] = obs

    # Split trajectories 70/15/15
    if has_agent:
        # Stack actions alongside for splitting
        train_t, val_t, test_t = _split_trajectories(trajs)
        train_a, val_a, test_a = _split_trajectories(actions)
        train_ap, val_ap, test_ap = _split_trajectories(agent_positions)

        def _make_action_conditioned_pairs(traj_split, act_split, apos_split):
            N, Tp1 = traj_split.shape[:2]
            T = Tp1 - 1
            states = traj_split[:, :-1]  # (N, T, C, H, W)
            next_states = traj_split[:, 1:]  # (N, T, C, H, W)

            # Build action fields: (N, T, n_actions, H, W)
            af = np.zeros((N, T, n_actions, gs, gs), dtype=np.float32)
            for n_idx in range(N):
                for t_idx in range(T):
                    a = act_split[n_idx, t_idx]
                    ar, ac = apos_split[n_idx, t_idx]
                    af[n_idx, t_idx, a, ar, ac] = 1.0

            # Reshape to samples
            X_state = states.reshape(-1, n_cell_types, gs, gs)
            X_action = af.reshape(-1, n_actions, gs, gs)
            X = np.concatenate([X_state, X_action], axis=1)
            Y = next_states.reshape(-1, n_cell_types, gs, gs)
            return X, Y

        X_tr, Y_tr = _make_action_conditioned_pairs(train_t, train_a, train_ap)
        X_v, Y_v = _make_action_conditioned_pairs(val_t, val_a, val_ap)
        X_te, Y_te = _make_action_conditioned_pairs(test_t, test_a, test_ap)

        in_channels = n_cell_types + n_actions
        out_channels = n_cell_types
        action_conditioned = True
    else:
        # Pure dynamics: state -> next_state
        train_t, val_t, test_t = _split_trajectories(trajs)

        def _make_pairs(traj_split):
            N, Tp1 = traj_split.shape[:2]
            rest = traj_split.shape[2:]
            X = traj_split[:, :-1].reshape(-1, *rest)
            Y = traj_split[:, 1:].reshape(-1, *rest)
            return X, Y

        X_tr, Y_tr = _make_pairs(train_t)
        X_v, Y_v = _make_pairs(val_t)
        X_te, Y_te = _make_pairs(test_t)

        in_channels = n_cell_types
        out_channels = n_cell_types
        action_conditioned = False

    data = _to_torch([X_tr, Y_tr, X_v, Y_v, X_te, Y_te], device)

    meta = {
        "name": env_class.__name__,
        "loss_type": "cross_entropy",
        "metric": "accuracy",
        "in_channels": in_channels,
        "out_channels": out_channels,
        "grid_size": gs,
        "n_cell_types": n_cell_types,
        "cell_type_names": env_class.CELL_TYPES,
        "n_trajectories": n_trajectories,
        "episode_length": episode_length,
        "action_conditioned": action_conditioned,
    }

    return BenchmarkData(*data, meta)
