"""DMControl data generation for CML world model training.

Generates (state, action, next_state) transition trajectories from:
  - cartpole-swingup  (5D state, 1D action)
  - reacher-easy      (6D state, 2D action)

Uses random actions uniform in the action space. 500 trajectories x 100 steps
per task, split 70/15/15 train/val/test. Saves to ~/wmca/data/*.npz.
"""

import os
# Force CPU / no rendering — we only need state data.
os.environ.setdefault("MUJOCO_GL", "disable")

import numpy as np
from dm_control import suite


TASKS = [
    # (domain, task, expected_state_dim, save_name)
    ("cartpole", "swingup", 5, "dmcontrol_cartpole.npz"),
    ("reacher", "easy", 6, "dmcontrol_reacher.npz"),
]

N_TRAJ = 500
STEPS_PER_TRAJ = 100
SEED = 0

DATA_DIR = os.path.expanduser("~/wmca/data")


def flatten_obs(obs):
    """Concatenate dm_control OrderedDict observations into a flat vector."""
    return np.concatenate([np.asarray(v, dtype=np.float32).ravel() for v in obs.values()])


def generate_task(domain, task, n_traj, steps, seed):
    env = suite.load(domain, task, task_kwargs={"random": seed})
    aspec = env.action_spec()
    a_dim = int(np.prod(aspec.shape))
    a_min = np.asarray(aspec.minimum, dtype=np.float32)
    a_max = np.asarray(aspec.maximum, dtype=np.float32)
    # Broadcast scalar bounds up to a_dim if needed.
    if a_min.shape == ():
        a_min = np.full((a_dim,), float(a_min), dtype=np.float32)
    if a_max.shape == ():
        a_max = np.full((a_dim,), float(a_max), dtype=np.float32)

    # Determine state_dim from one reset.
    ts = env.reset()
    state_dim = flatten_obs(ts.observation).shape[0]

    rng = np.random.default_rng(seed)

    states = np.zeros((n_traj, steps, state_dim), dtype=np.float32)
    actions = np.zeros((n_traj, steps, a_dim), dtype=np.float32)
    next_states = np.zeros((n_traj, steps, state_dim), dtype=np.float32)

    for t in range(n_traj):
        ts = env.reset()
        s = flatten_obs(ts.observation)
        for k in range(steps):
            a = rng.uniform(a_min, a_max).astype(np.float32)
            ts = env.step(a)
            s_next = flatten_obs(ts.observation)
            states[t, k] = s
            actions[t, k] = a
            next_states[t, k] = s_next
            if ts.last():
                # dm_control default episodes are 1000 control steps; we shouldn't hit
                # this within 100, but handle it defensively by resetting.
                ts = env.reset()
                s_next = flatten_obs(ts.observation)
            s = s_next
        if (t + 1) % 50 == 0:
            print(f"  [{domain}-{task}] {t + 1}/{n_traj} trajectories")

    return states, actions, next_states, a_min, a_max, state_dim


def split_and_save(states, actions, next_states, a_min, a_max, state_dim, out_path):
    n = states.shape[0]
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(n)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]

    np.savez_compressed(
        out_path,
        states=states,
        actions=actions,
        next_states=next_states,
        action_min=a_min,
        action_max=a_max,
        state_dim=np.int64(state_dim),
        train_idx=train_idx.astype(np.int64),
        val_idx=val_idx.astype(np.int64),
        test_idx=test_idx.astype(np.int64),
    )
    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(
        f"  saved {out_path}  "
        f"states={states.shape} actions={actions.shape} next_states={next_states.shape}  "
        f"split={len(train_idx)}/{len(val_idx)}/{len(test_idx)}  ({size_mb:.2f} MB)"
    )


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    for domain, task, expected_state_dim, fname in TASKS:
        print(f"Generating {domain}-{task} ...")
        states, actions, next_states, a_min, a_max, state_dim = generate_task(
            domain, task, N_TRAJ, STEPS_PER_TRAJ, SEED
        )
        assert state_dim == expected_state_dim, (
            f"{domain}-{task}: expected state_dim={expected_state_dim}, got {state_dim}"
        )
        out_path = os.path.join(DATA_DIR, fname)
        split_and_save(states, actions, next_states, a_min, a_max, state_dim, out_path)
    print("Done.")


if __name__ == "__main__":
    main()
