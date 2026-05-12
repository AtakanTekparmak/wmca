"""DOOM-real benchmark: real DOOM frames through a frozen autoencoder.

Mirrors crafter_real.py but for DOOM (ViZDoom) frames.
Frames are resized to 64x64 so the same FrameAutoencoder architecture is reused.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from wmca.benchmarks import BenchmarkData, _to_torch
from wmca.modules.frame_encoder import FrozenFrameEncoder

# Default for basic scenario; callers can override via n_actions kwarg.
_DOOM_N_ACTIONS = 3


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
        # Resize to 64x64 if needed (encoder expects 64x64)
        _, _, h, w = batch.shape
        if h != 64 or w != 64:
            batch = F.interpolate(batch, size=(64, 64), mode="bilinear",
                                  align_corners=False)
        z = encoder(batch).cpu().numpy()
        encoded.append(z)
    return np.concatenate(encoded, axis=0)


def generate_doom_real(
    grid_size: int = 16,
    n_frames: int = 50000,
    n_actions: int = _DOOM_N_ACTIONS,
    seed: int = 42,
    device: str | torch.device = "cpu",
    data_dir: str = "experiments/doom_data",
    encoder_path: str | None = None,
) -> BenchmarkData:
    """DOOM-real benchmark: real DOOM frames encoded through a frozen AE.

    Expects pre-collected data at data_dir:
        frames.npy      (N, 3, H, W) float32 [0,1]
        actions.npy     (N,) int
        next_frames.npy (N, 3, H, W) float32 [0,1]

    Returns BenchmarkData where:
      X = [encoded_frame, action_field]  (N, 2, 16, 16)
      Y = encoded_next_frame             (N, 1, 16, 16)
    """
    device = torch.device(device)
    data_dir = Path(data_dir)
    if encoder_path is None:
        encoder_path = str(data_dir / "frame_encoder.pt")

    # ------------------------------------------------------------------
    # 1. Load pre-collected frames
    # ------------------------------------------------------------------
    frames_path = data_dir / "frames.npy"
    actions_path = data_dir / "actions.npy"
    next_frames_path = data_dir / "next_frames.npy"

    if not all(p.exists() for p in (frames_path, actions_path, next_frames_path)):
        raise FileNotFoundError(
            f"DOOM data not found in {data_dir}. "
            "Run a ViZDoom collection script first to generate "
            "frames.npy, actions.npy, and next_frames.npy."
        )

    frames = np.load(str(frames_path))
    actions = np.load(str(actions_path))
    next_frames = np.load(str(next_frames_path))

    # Truncate to requested size
    n = min(len(frames), len(actions), len(next_frames), n_frames)
    frames, actions, next_frames = frames[:n], actions[:n], next_frames[:n]
    print(f"  loaded {n} DOOM frames from {data_dir}")

    # ------------------------------------------------------------------
    # 2. Encode through frozen autoencoder (resizes to 64x64 internally)
    # ------------------------------------------------------------------
    encoder = FrozenFrameEncoder(encoder_path, device=str(device))
    encoder.to(device)

    enc_frames = _encode_batched(encoder, frames, device=device)          # (N, 1, 16, 16)
    enc_next_frames = _encode_batched(encoder, next_frames, device=device)  # (N, 1, 16, 16)

    encoder_param_count = sum(p.numel() for p in encoder.parameters())

    # ------------------------------------------------------------------
    # 3. Build action field: single channel filled with (action+1)/n_actions
    # ------------------------------------------------------------------
    action_fields = np.zeros((n, 1, grid_size, grid_size), dtype=np.float32)
    for i in range(n):
        action_fields[i, 0, :, :] = (actions[i] + 1.0) / n_actions

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
        "name": "doom_real",
        "loss_type": "mse",
        "metric": "mse",
        "in_channels": 2,
        "out_channels": 1,
        "grid_size": grid_size,
        "encoder_params": encoder_param_count,
        "n_frames": n,
        "n_actions": n_actions,
        "game": "doom",
    }
    return BenchmarkData(*data, meta)
