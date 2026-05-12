"""Generate Atari training data for Pong + Breakout.

Collects frames via random policy, trains grid-native autoencoder on MPS,
encodes all frames to latent grids. Reports PSNR and shapes.

Key constraints: total memory < 3GB, batch_size=8, MPS device.

Output per game (in experiments/atari_data/):
    {game}_frames.npy       — raw one-hot frames (N, C, H, W)
    {game}_actions.npy      — action sequences (N,)
    {game}_next_frames.npy  — next-frame data (N, C, H, W)
    {game}_encoder.pt       — trained encoder weights
    {game}_latents.npy      — encoded latent grids (N, 1, H, W)
    {game}_next_latents.npy — next-frame latent grids (N, 1, H, W)
    {game}_boundaries.npy   — episode boundaries

Usage:
    PYTHONPATH=src uv run python experiments/generate_atari_data.py
"""
from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


# ═══════════════════════════════════════════════════════════════════════════════
#  Config
# ═══════════════════════════════════════════════════════════════════════════════

GAMES = ["pong", "breakout"]
N_FRAMES = 75_000
DEVICE = "mps"
ENCODER_BATCH_SIZE = 8
ENCODER_EPOCHS = 50
ENCODER_LR = 1e-3
DATA_DIR = Path("experiments/atari_data")

GAME_CONFIGS = {
    "pong": {"grid_h": 16, "grid_w": 32, "n_channels": 4, "n_actions": 3},
    "breakout": {"grid_h": 20, "grid_w": 16, "n_channels": 4, "n_actions": 3},
}


# ═══════════════════════════════════════════════════════════════════════════════
#  Grid-native autoencoder
# ═══════════════════════════════════════════════════════════════════════════════

class GridNativeEncoder(nn.Module):
    """Grid-native autoencoder: (C, H, W) → (1, H, W) → (C, H, W)."""

    def __init__(self, in_channels: int = 4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, 3, padding=1),
            nn.Sigmoid(),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, in_channels, 3, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    @property
    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ═══════════════════════════════════════════════════════════════════════════════
#  Env helpers
# ═══════════════════════════════════════════════════════════════════════════════

def make_env(game: str, config: dict, seed: int = 42):
    if game == "pong":
        from wmca.envs.atari_pong import PongEnv
        return PongEnv(grid_h=config["grid_h"], grid_w=config["grid_w"], seed=seed)
    elif game == "breakout":
        from wmca.envs.atari_pong import BreakoutEnv
        return BreakoutEnv(grid_h=config["grid_h"], grid_w=config["grid_w"], seed=seed)
    else:
        raise ValueError(f"Unknown game: {game}")


def collect_frames(game: str, config: dict, n_frames: int, seed: int = 42):
    """Collect (frame, action, next_frame) via random policy."""
    env = make_env(game, config, seed)
    rng = np.random.default_rng(seed)
    n_actions = config["n_actions"]

    frames, actions, next_frames = [], [], []
    while len(frames) < n_frames:
        obs = env.reset()
        for _ in range(200):
            if len(frames) >= n_frames:
                break
            frame = obs.copy()
            action = int(rng.integers(n_actions))
            obs = env.step(action)
            next_frame = obs.copy()
            frames.append(frame)
            actions.append(action)
            next_frames.append(next_frame)

    frames = np.stack(frames[:n_frames], axis=0).astype(np.float32)
    actions = np.array(actions[:n_frames], dtype=np.int32)
    next_frames = np.stack(next_frames[:n_frames], axis=0).astype(np.float32)
    return frames, actions, next_frames


# ═══════════════════════════════════════════════════════════════════════════════
#  Encoder training
# ═══════════════════════════════════════════════════════════════════════════════

def train_encoder(
    frames: np.ndarray,
    n_channels: int,
    batch_size: int,
    epochs: int,
    lr: float,
) -> GridNativeEncoder:
    dev = torch.device(DEVICE)
    model = GridNativeEncoder(in_channels=n_channels).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    data = torch.from_numpy(frames).float()
    n = len(data)

    best_loss = float("inf")
    best_state = None
    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, device=torch.device("cpu"))
        total_loss = 0.0
        n_batches = 0

        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            xb = data[idx].to(dev)
            recon = model(xb)
            loss = criterion(recon, xb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"    epoch {epoch + 1:3d}/{epochs:3d}  "
                  f"loss={avg_loss:.6f}  best={best_loss:.6f}  "
                  f"elapsed={elapsed:.0f}s")

    if best_state:
        model.load_state_dict(best_state)
    return model.cpu()


# ═══════════════════════════════════════════════════════════════════════════════
#  Encoding
# ═══════════════════════════════════════════════════════════════════════════════

def encode_all(
    encoder: GridNativeEncoder,
    frames: np.ndarray,
    next_frames: np.ndarray,
    encode_batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode all frames to latents. encode_batch_size can be larger (inference)."""
    dev = torch.device(DEVICE)
    encoder = encoder.to(dev)
    encoder.eval()

    latents_list, next_latents_list = [], []
    with torch.no_grad():
        for i in range(0, len(frames), encode_batch_size):
            xb = torch.from_numpy(frames[i : i + encode_batch_size]).float().to(dev)
            nx = torch.from_numpy(next_frames[i : i + encode_batch_size]).float().to(dev)
            latents_list.append(encoder.encode(xb).cpu().numpy())
            next_latents_list.append(encoder.encode(nx).cpu().numpy())

    latents = np.concatenate(latents_list, axis=0)
    next_latents = np.concatenate(next_latents_list, axis=0)
    return latents, next_latents


def compute_psnr(original: np.ndarray, recon: np.ndarray) -> float:
    mse = np.mean((original - recon) ** 2)
    if mse < 1e-10:
        return 100.0
    return float(10.0 * math.log10(1.0 / mse))


def find_boundaries(frames: np.ndarray, next_frames: np.ndarray) -> np.ndarray:
    """Find episode breaks where next_frames[i] != frames[i+1]."""
    n = len(frames)
    chunk = 4096
    breaks = []
    for start in range(0, n - 1, chunk):
        end = min(start + chunk, n - 1)
        a = next_frames[start:end]
        b = frames[start + 1 : end + 1]
        mism = (a != b).reshape(end - start, -1).any(axis=1)
        for j in np.nonzero(mism)[0]:
            breaks.append(int(start + j + 1))
    return np.array([0] + breaks + [n], dtype=np.int64)


def estimate_memory_mb(game: str, config: dict, n_frames: int) -> float:
    """Estimate bytes for frames + next_frames + latents + next_latents."""
    c = config["n_channels"]
    h = config["grid_h"]
    w = config["grid_w"]
    # frames(float32, c ch) + next_frames + latents(float32, 1 ch) + next_latents
    per_frame_bytes = (c * 2 + 2) * h * w * 4
    return (n_frames * per_frame_bytes) / (1024 * 1024)


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Atari Training Data Generation — Pong + Breakout")
    print(f"  Device: {DEVICE}")
    print(f"  Encoder batch size: {ENCODER_BATCH_SIZE}")
    print(f"  Encoder epochs: {ENCODER_EPOCHS}")
    print(f"  Frames per game: {N_FRAMES:,}")
    print("=" * 70)

    total_est_mb = sum(estimate_memory_mb(g, GAME_CONFIGS[g], N_FRAMES) for g in GAMES)
    print(f"\n  Estimated disk usage (raw): {total_est_mb:.0f} MB")
    print(f"  Budget: < 3,072 MB — {'OK' if total_est_mb < 3072 else 'EXCEEDED!'}\n")

    results = {}

    for game in GAMES:
        cfg = GAME_CONFIGS[game]
        print(f"\n{'─' * 70}")
        print(f"  {game.upper()}")
        print(f"  Grid: {cfg['grid_h']}×{cfg['grid_w']}  Channels: {cfg['n_channels']}")
        print(f"{'─' * 70}")

        frames_path = DATA_DIR / f"{game}_frames.npy"
        actions_path = DATA_DIR / f"{game}_actions.npy"
        next_frames_path = DATA_DIR / f"{game}_next_frames.npy"
        encoder_path = DATA_DIR / f"{game}_encoder.pt"
        latents_path = DATA_DIR / f"{game}_latents.npy"
        next_latents_path = DATA_DIR / f"{game}_next_latents.npy"
        boundaries_path = DATA_DIR / f"{game}_boundaries.npy"

        # ── Step 1: Collect frames ──
        if frames_path.exists():
            print(f"\n  [1/4] Loading existing frames...")
            frames = np.load(str(frames_path))
            actions = np.load(str(actions_path))
            next_frames = np.load(str(next_frames_path))
            # Truncate if more than requested
            if len(frames) > N_FRAMES:
                frames = frames[:N_FRAMES]
                actions = actions[:N_FRAMES]
                next_frames = next_frames[:N_FRAMES]
        else:
            print(f"\n  [1/4] Collecting {N_FRAMES:,} frames via random policy...")
            t0 = time.time()
            frames, actions, next_frames = collect_frames(game, cfg, N_FRAMES)
            elapsed = time.time() - t0
            print(f"    Collected {len(frames):,} frames in {elapsed:.1f}s")
            np.save(str(frames_path), frames)
            np.save(str(actions_path), actions)
            np.save(str(next_frames_path), next_frames)

        mem_mb = frames.nbytes / (1024 * 1024)
        print(f"    frames:      {frames.shape}  {frames.dtype}  {mem_mb:.0f} MB")
        print(f"    actions:     {actions.shape}  {actions.dtype}")
        print(f"    next_frames: {next_frames.shape}  {next_frames.dtype}")

        # ── Step 2: Train encoder ──
        if encoder_path.exists() and latents_path.exists() and next_latents_path.exists():
            print(f"\n  [2/4] Encoder + latents already exist, loading...")
            encoder = GridNativeEncoder(in_channels=cfg["n_channels"])
            encoder.load_state_dict(
                torch.load(str(encoder_path), map_location="cpu", weights_only=True)
            )
            latents = np.load(str(latents_path))
            next_latents = np.load(str(next_latents_path))
        else:
            print(f"\n  [2/4] Training encoder ({cfg['n_channels']}→1→{cfg['n_channels']})...")
            print(f"    Params: {GridNativeEncoder(in_channels=cfg['n_channels']).param_count:,}")
            encoder = train_encoder(
                frames, cfg["n_channels"],
                batch_size=ENCODER_BATCH_SIZE,
                epochs=ENCODER_EPOCHS,
                lr=ENCODER_LR,
            )
            torch.save(encoder.state_dict(), str(encoder_path))
            print(f"    Saved encoder → {encoder_path}")

            # ── Step 3: Encode all frames ──
            print(f"\n  [3/4] Encoding all {N_FRAMES:,} frames...")
            t0 = time.time()
            latents, next_latents = encode_all(encoder, frames, next_frames)
            elapsed = time.time() - t0
            np.save(str(latents_path), latents)
            np.save(str(next_latents_path), next_latents)
            print(f"    Encoded in {elapsed:.1f}s")

        print(f"    latents:      {latents.shape}  {latents.dtype}")
        print(f"    next_latents: {next_latents.shape}  {next_latents.dtype}")

        # ── Step 4: PSNR / Quality check ──
        print(f"\n  [4/4] Computing reconstruction PSNR...")
        dev = torch.device(DEVICE)
        encoder = encoder.to(dev)
        encoder.eval()

        # Compute PSNR on a batch of 128 frames
        sample_n = min(128, len(frames))
        with torch.no_grad():
            xb = torch.from_numpy(frames[:sample_n]).float().to(dev)
            recon = encoder(xb).cpu().numpy()

        psnr = compute_psnr(frames[:sample_n], recon)
        print(f"    PSNR ({sample_n} samples): {psnr:.1f} dB")

        # ── Episodes ──
        if not boundaries_path.exists():
            boundaries = find_boundaries(frames, next_frames)
            np.save(str(boundaries_path), boundaries)
        else:
            boundaries = np.load(str(boundaries_path))

        n_episodes = len(boundaries) - 1
        episode_lengths = np.diff(boundaries)
        print(f"    Episodes: {n_episodes}  "
              f"(min={episode_lengths.min()}, "
              f"mean={episode_lengths.mean():.0f}, "
              f"max={episode_lengths.max()})")

        # ── Disk usage ──
        total_disk_mb = sum(
            f.stat().st_size for f in [frames_path, actions_path, next_frames_path,
                                       encoder_path, latents_path, next_latents_path]
            if f.exists()
        ) / (1024 * 1024)
        print(f"    Disk usage: {total_disk_mb:.0f} MB")

        results[game] = {
            "frames_shape": frames.shape,
            "latents_shape": latents.shape,
            "next_latents_shape": next_latents.shape,
            "psnr": psnr,
            "n_episodes": n_episodes,
            "encoder_params": encoder.param_count,
            "disk_mb": total_disk_mb,
        }

    # ═══════════════════════════════════════════════════════════════════════════
    #  Summary
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    grand_total_mb = 0
    for game in GAMES:
        r = results[game]
        print(f"\n  {game.upper()}")
        print(f"    Frames:          {r['frames_shape']}")
        print(f"    Latents:         {r['latents_shape']}")
        print(f"    Next latents:    {r['next_latents_shape']}")
        print(f"    PSNR:            {r['psnr']:.1f} dB")
        print(f"    Encoder params:  {r['encoder_params']:,}")
        print(f"    Episodes:        {r['n_episodes']}")
        print(f"    Disk:            {r['disk_mb']:.0f} MB")
        grand_total_mb += r["disk_mb"]

    print(f"\n  Total disk: {grand_total_mb:.0f} MB")

    # Verify the combined dataset fits <3GB
    total_gb = grand_total_mb / 1024
    status = "OK ✓" if total_gb < 3.0 else "EXCEEDED ✗"
    print(f"  Total disk: {total_gb:.2f} GB  —  Budget <3 GB  [{status}]")
    print("=" * 70)


if __name__ == "__main__":
    main()
