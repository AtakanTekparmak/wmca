"""Crafter-real benchmark: real Crafter frames through a frozen autoencoder."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from wmca.benchmarks import BenchmarkData, _to_torch
from wmca.modules.frame_encoder import FrozenFrameEncoder

_CRAFTER_N_ACTIONS = 17  # noop + 16 actions


@dataclass
class TrajectorySpec:
    """Contiguous-episode rollout input for the Crafter-latent probe.

    encoded_frames: (T+1, 1, grid_size, grid_size) float32
        encoded_frames[0]     = encoder(frames[ep_start])
        encoded_frames[t+1]   = encoder(next_frames[ep_start + t])
    actions:        (T,) int64 in [0, _CRAFTER_N_ACTIONS)
        actions[t] was taken at step t to produce encoded_frames[t+1].
    episode_index:  int (for debugging / provenance)
    """

    encoded_frames: torch.Tensor
    actions: torch.Tensor
    episode_index: int


def _find_episode_boundaries(
    frames: np.ndarray,
    next_frames: np.ndarray,
    data_dir: Path,
    chunk: int = 4096,
) -> np.ndarray:
    """Scan for indices i where frames[i+1] != next_frames[i] (episode break).

    Returns:
        boundaries: (E+1,) int64 array of episode start indices. The episode
        starting at boundaries[e] has length boundaries[e+1] - boundaries[e].
        boundaries[0] == 0 and boundaries[-1] == len(frames).

    Cached on disk at ``data_dir / 'episode_boundaries.npy'``.
    """
    cache = data_dir / "episode_boundaries.npy"
    if cache.exists():
        cached = np.load(str(cache))
        # Only trust cache if it covers exactly the current n_frames
        if cached.size >= 2 and cached[-1] == len(frames):
            return cached

    n = len(frames)
    breaks: list[int] = []
    for start in range(0, n - 1, chunk):
        end = min(start + chunk, n - 1)
        # Compare frames[i+1] vs next_frames[i] for i in [start, end)
        a = next_frames[start:end]
        b = frames[start + 1 : end + 1]
        # Any per-pixel mismatch (use exact inequality since both are float32
        # copies of the same env observations)
        mism = (a != b).reshape(end - start, -1).any(axis=1)
        for j in np.nonzero(mism)[0]:
            breaks.append(int(start + j + 1))
    # Episode starts: 0, then each break index, then n as sentinel end.
    boundaries = np.array([0] + breaks + [n], dtype=np.int64)
    # Persist cache
    try:
        np.save(str(cache), boundaries)
    except Exception:
        pass
    return boundaries


def _collect_frames(
    n_frames: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect (frame, action, next_frame) tuples via random policy.

    Returns:
        frames:      (N, 3, 64, 64) float32 in [0, 1]
        actions:     (N,) int32
        next_frames: (N, 3, 64, 64) float32 in [0, 1]
    """
    import crafter

    env = crafter.Env()
    rng = np.random.RandomState(seed)

    frames, actions, next_frames = [], [], []
    while len(frames) < n_frames:
        obs = env.reset()
        done = False
        while not done and len(frames) < n_frames:
            # obs is (64, 64, 3) uint8
            frame = obs.astype(np.float32) / 255.0
            frame = frame.transpose(2, 0, 1)  # -> (3, 64, 64)

            action = rng.randint(_CRAFTER_N_ACTIONS)
            obs, _, done, _ = env.step(action)

            next_frame = obs.astype(np.float32) / 255.0
            next_frame = next_frame.transpose(2, 0, 1)

            frames.append(frame)
            actions.append(action)
            next_frames.append(next_frame)

    frames = np.stack(frames[:n_frames])
    actions = np.array(actions[:n_frames], dtype=np.int32)
    next_frames = np.stack(next_frames[:n_frames])
    return frames, actions, next_frames


@torch.no_grad()
def _encode_batched(
    encoder: FrozenFrameEncoder,
    frames: np.ndarray,
    batch_size: int = 512,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    """Encode frames in batches to avoid OOM. Returns (N, 1, 16, 16) float32."""
    device = torch.device(device)
    encoded = []
    for i in range(0, len(frames), batch_size):
        batch = torch.from_numpy(frames[i : i + batch_size]).float().to(device)
        z = encoder(batch).cpu().numpy()
        encoded.append(z)
    return np.concatenate(encoded, axis=0)


def generate_crafter_real(
    grid_size: int = 16,
    n_frames: int = 50000,
    seed: int = 42,
    device: str | torch.device = "cpu",
    data_dir: str = "experiments/crafter_data",
    encoder_path: str | None = None,
):
    """Crafter-real benchmark: real Crafter frames encoded through a frozen AE.

    Returns BenchmarkData where:
      X = [encoded_frame, action_field]  (N, 2, 16, 16)
      Y = encoded_next_frame             (N, 1, 16, 16)
    """
    device = torch.device(device)
    data_dir = Path(data_dir)
    if encoder_path is None:
        encoder_path = str(data_dir / "frame_encoder.pt")

    # ------------------------------------------------------------------
    # 1. Load or collect raw frames
    # ------------------------------------------------------------------
    frames_path = data_dir / "frames.npy"
    actions_path = data_dir / "actions.npy"
    next_frames_path = data_dir / "next_frames.npy"

    if frames_path.exists() and actions_path.exists() and next_frames_path.exists():
        frames = np.load(str(frames_path))
        actions = np.load(str(actions_path))
        next_frames = np.load(str(next_frames_path))
        # Truncate / warn if saved data is shorter
        n = min(len(frames), len(actions), len(next_frames), n_frames)
        frames, actions, next_frames = frames[:n], actions[:n], next_frames[:n]
    else:
        data_dir.mkdir(parents=True, exist_ok=True)
        frames, actions, next_frames = _collect_frames(n_frames, seed)
        np.save(str(frames_path), frames)
        np.save(str(actions_path), actions)
        np.save(str(next_frames_path), next_frames)

    n = len(frames)

    # ------------------------------------------------------------------
    # 2. Encode through frozen autoencoder
    # ------------------------------------------------------------------
    encoder = FrozenFrameEncoder(encoder_path, device=str(device))
    encoder.to(device)

    enc_frames = _encode_batched(encoder, frames, device=device)         # (N, 1, 16, 16)
    enc_next_frames = _encode_batched(encoder, next_frames, device=device)  # (N, 1, 16, 16)

    encoder_param_count = sum(p.numel() for p in encoder.parameters())

    # ------------------------------------------------------------------
    # 3. Build action field: single channel filled with (action+1)/17
    # ------------------------------------------------------------------
    action_fields = np.zeros((n, 1, grid_size, grid_size), dtype=np.float32)
    for i in range(n):
        action_fields[i, 0, :, :] = (actions[i] + 1.0) / _CRAFTER_N_ACTIONS

    # ------------------------------------------------------------------
    # 4. Assemble X, Y
    # ------------------------------------------------------------------
    X = np.concatenate([enc_frames, action_fields], axis=1)  # (N, 2, 16, 16)
    Y = enc_next_frames                                       # (N, 1, 16, 16)

    # ------------------------------------------------------------------
    # 5. Split 70/15/15
    # ------------------------------------------------------------------
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)

    X_tr, Y_tr = X[:n_train], Y[:n_train]
    X_v, Y_v = X[n_train : n_train + n_val], Y[n_train : n_train + n_val]
    X_te, Y_te = X[n_train + n_val :], Y[n_train + n_val :]

    data = _to_torch([X_tr, Y_tr, X_v, Y_v, X_te, Y_te], device)

    meta = {
        "name": "crafter_real",
        "loss_type": "mse",
        "metric": "mse",
        "in_channels": 2,
        "out_channels": 1,
        "grid_size": grid_size,
        "encoder_params": encoder_param_count,
        "n_frames": n,
    }
    return BenchmarkData(*data, meta)


def generate_crafter_real_trajectories(
    grid_size: int = 16,
    n_frames: int = 100_000,
    seed: int = 42,
    device: str | torch.device = "cpu",
    data_dir: str = "experiments/crafter_data",
    encoder_path: str | None = None,
    min_traj_len: int = 105,
    max_test_trajectories: Optional[int] = None,
):
    """Crafter-latent benchmark that *also* carries whole test trajectories.

    Same training/val/test pairwise shape as ``generate_crafter_real`` — the
    returned ``BenchmarkData`` is drop-in compatible with ``train_model``.

    Additionally, ``BenchmarkData.meta["test_trajectories"]`` is a list of
    :class:`TrajectorySpec` entries covering episodes fully contained in the
    test region (start index >= ``n_train + n_val``) whose length is at
    least ``min_traj_len``.

    Parameters:
        grid_size: encoder latent side (coupled to the frozen encoder; 16).
        n_frames: number of raw pairs to load from disk.
        seed: only affects training reproducibility once the .npy cache
            exists; does NOT re-collect data.
        device: where tensors live in the returned namedtuple.
        data_dir: where to find frames.npy / actions.npy / next_frames.npy /
            frame_encoder.pt.
        encoder_path: optional override for the encoder checkpoint path.
        min_traj_len: drop test episodes shorter than this many steps.
        max_test_trajectories: optional cap on the number of trajectories
            returned in meta["test_trajectories"] (applied after filtering).
    """
    assert grid_size == 16, (
        "generate_crafter_real_trajectories: the frozen encoder outputs 16x16; "
        "grid_size must be 16."
    )

    device = torch.device(device)
    data_dir_path = Path(data_dir)
    if encoder_path is None:
        encoder_path = str(data_dir_path / "frame_encoder.pt")

    frames_path = data_dir_path / "frames.npy"
    actions_path = data_dir_path / "actions.npy"
    next_frames_path = data_dir_path / "next_frames.npy"

    if not (
        frames_path.exists() and actions_path.exists() and next_frames_path.exists()
    ):
        raise FileNotFoundError(
            f"Raw Crafter data missing under {data_dir_path}. "
            "generate_crafter_real_trajectories does not re-collect — "
            "call generate_crafter_real once first, or populate the .npy files."
        )

    frames = np.load(str(frames_path))
    actions = np.load(str(actions_path))
    next_frames = np.load(str(next_frames_path))
    n = min(len(frames), len(actions), len(next_frames), n_frames)
    frames, actions, next_frames = frames[:n], actions[:n], next_frames[:n]

    # ------------------------------------------------------------------
    # Episode boundary scan (cached)
    # ------------------------------------------------------------------
    boundaries = _find_episode_boundaries(frames, next_frames, data_dir_path)

    # ------------------------------------------------------------------
    # Encode frames + next_frames through the frozen AE
    # ------------------------------------------------------------------
    encoder = FrozenFrameEncoder(encoder_path, device=str(device))
    encoder.to(device)

    enc_frames = _encode_batched(encoder, frames, device=device)          # (N, 1, 16, 16)
    enc_next_frames = _encode_batched(encoder, next_frames, device=device)  # (N, 1, 16, 16)
    encoder_param_count = sum(p.numel() for p in encoder.parameters())

    # ------------------------------------------------------------------
    # Build action field (same convention as generate_crafter_real)
    # ------------------------------------------------------------------
    action_fields = np.zeros((n, 1, grid_size, grid_size), dtype=np.float32)
    for i in range(n):
        action_fields[i, 0, :, :] = (actions[i] + 1.0) / _CRAFTER_N_ACTIONS

    X = np.concatenate([enc_frames, action_fields], axis=1)  # (N, 2, 16, 16)
    Y = enc_next_frames                                       # (N, 1, 16, 16)

    # ------------------------------------------------------------------
    # Pairwise 70/15/15 split (identical to generate_crafter_real)
    # ------------------------------------------------------------------
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    test_start = n_train + n_val

    X_tr, Y_tr = X[:n_train], Y[:n_train]
    X_v, Y_v = X[n_train:test_start], Y[n_train:test_start]
    X_te, Y_te = X[test_start:], Y[test_start:]

    data = _to_torch([X_tr, Y_tr, X_v, Y_v, X_te, Y_te], device)

    # ------------------------------------------------------------------
    # Build test trajectories
    # ------------------------------------------------------------------
    enc_frames_t = torch.from_numpy(enc_frames)           # keep on CPU for probe
    enc_next_frames_t = torch.from_numpy(enc_next_frames)
    actions_t = torch.from_numpy(actions.astype(np.int64))

    trajectories: List[TrajectorySpec] = []
    for ep_idx in range(len(boundaries) - 1):
        ep_start = int(boundaries[ep_idx])
        ep_end = int(boundaries[ep_idx + 1])      # exclusive
        ep_len = ep_end - ep_start
        if ep_start < test_start:
            continue
        if ep_len < min_traj_len:
            continue
        T = ep_len  # number of transitions in the episode

        # encoded_frames[0] = enc_frames[ep_start]
        # encoded_frames[t+1] = enc_next_frames[ep_start + t] for t in [0, T)
        head = enc_frames_t[ep_start : ep_start + 1]          # (1, 1, 16, 16)
        tail = enc_next_frames_t[ep_start : ep_start + T]     # (T, 1, 16, 16)
        ef = torch.cat([head, tail], dim=0).contiguous()      # (T+1, 1, 16, 16)
        acts = actions_t[ep_start : ep_start + T].contiguous()  # (T,)

        trajectories.append(
            TrajectorySpec(
                encoded_frames=ef,
                actions=acts,
                episode_index=ep_idx,
            )
        )

    if max_test_trajectories is not None:
        trajectories = trajectories[:max_test_trajectories]

    meta = {
        "name": "crafter_real_traj",
        "loss_type": "mse",
        "metric": "mse",
        "in_channels": 2,
        "out_channels": 1,
        "grid_size": grid_size,
        "encoder_params": encoder_param_count,
        "n_frames": n,
        "n_episodes_total": int(len(boundaries) - 1),
        "n_test_trajectories": len(trajectories),
        "min_traj_len": min_traj_len,
        "test_trajectories": trajectories,
        "n_crafter_actions": _CRAFTER_N_ACTIONS,
    }
    return BenchmarkData(*data, meta)
