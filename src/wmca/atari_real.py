"""Atari-latent benchmark: Atari frames through a grid-native autoencoder.

Path A of Plan 0 (A.2). Sibling of crafter_real.py.

Collects Pong/Breakout frames via random policy, encodes through a trained
grid-native autoencoder, and produces action-conditioned (X, Y) pairs for
training rescor on Atari latent dynamics.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

# Game-specific constants
GAME_CONFIGS = {
    "pong": {
        "n_actions": 3,
        "grid_h": 16,
        "grid_w": 32,
        "n_channels": 4,  # ball, left_paddle, right_paddle, walls
        "env_class": "PongEnv",
    },
    "breakout": {
        "n_actions": 3,
        "grid_h": 20,
        "grid_w": 16,
        "n_channels": 4,
        "env_class": "BreakoutEnv",
    },
}


class GridNativeEncoder(nn.Module):
    """Grid-native autoencoder for Atari one-hot grids.

    Encoder: (C, H, W) → (1, H, W) latent  (no spatial compression).
    Decoder: (1, H, W) → (C, H, W) reconstruction.
    """

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

    def load_state_dict(self, state_dict, strict=True):
        # Support checkpoints saved with 'enc.'/'dec.' prefix (from v2 training scripts)
        remapped = {}
        for k, v in state_dict.items():
            if k.startswith('enc.'):
                remapped['encoder.' + k[4:]] = v
            elif k.startswith('dec.'):
                remapped['decoder.' + k[4:]] = v
            else:
                remapped[k] = v
        return super().load_state_dict(remapped, strict)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    def param_count(self) -> dict[str, int]:
        enc = sum(p.numel() for p in self.encoder.parameters())
        dec = sum(p.numel() for p in self.decoder.parameters())
        return {"encoder": enc, "decoder": dec, "total": enc + dec}


@dataclass
class TrajectorySpec:
    """Contiguous-episode rollout input for the Atari latent probe.

    encoded_frames: (T+1, C, H, W) float32
        encoded_frames[0]     = encoder(frames[ep_start])
        encoded_frames[t+1]   = encoder(next_frames[ep_start + t])
    actions:        (T,) int64 in [0, n_actions)
    episode_index:  int
    """

    encoded_frames: np.ndarray
    actions: np.ndarray
    episode_index: int


class AtariLatentBenchmark:
    """Loads/collects Atari frames, trains encoder, produces latent data.

    Data is cached under experiments/atari_data/ after first collection.
    """

    def __init__(
        self,
        game: str = "pong",
        n_frames: int = 200000,
        data_dir: str = "experiments/atari_data",
        seed: int = 42,
        device: str = "cpu",
    ):
        self.game = game
        self.n_frames = n_frames
        self.data_dir = Path(data_dir)
        self.seed = seed
        self.device_str = device

        config = GAME_CONFIGS[game]
        self.n_actions = config["n_actions"]
        self.grid_h = config["grid_h"]
        self.grid_w = config["grid_w"]
        self.n_channels = config["n_channels"]

        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Load or collect data
        self._ensure_data()

    def _get_env(self, seed: int | None = None):
        """Create a new Atari env instance."""
        if self.game == "pong":
            from wmca.envs.atari_pong import PongEnv
            return PongEnv(grid_h=self.grid_h, grid_w=self.grid_w, seed=seed or self.seed)
        elif self.game == "breakout":
            from wmca.envs.atari_pong import BreakoutEnv
            return BreakoutEnv(grid_h=self.grid_h, grid_w=self.grid_w, seed=seed or self.seed)
        else:
            raise ValueError(f"Unknown game '{self.game}'")

    def _collect_frames(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Collect (frame, action, next_frame) via random policy.

        Returns:
            frames:      (N, C, H, W) float32 one-hot
            actions:     (N,) int32
            next_frames: (N, C, H, W) float32 one-hot
        """
        env = self._get_env()
        rng = np.random.default_rng(self.seed)

        frames, actions, next_frames = [], [], []
        while len(frames) < self.n_frames:
            obs = env.reset()
            for _ in range(200):  # max episode steps
                if len(frames) >= self.n_frames:
                    break

                frame = obs.copy()  # (C, H, W) float32 one-hot
                action = int(rng.integers(self.n_actions))
                obs = env.step(action)
                next_frame = obs.copy()

                frames.append(frame)
                actions.append(action)
                next_frames.append(next_frame)

        # Truncate to exact count
        frames = np.stack(frames[: self.n_frames], axis=0)
        actions = np.array(actions[: self.n_frames], dtype=np.int32)
        next_frames = np.stack(next_frames[: self.n_frames], axis=0)
        return frames, actions, next_frames

    def _train_encoder(
        self,
        frames: np.ndarray,
        epochs: int = 50,
        batch_size: int = 128,
        lr: float = 1e-3,
    ) -> GridNativeEncoder:
        """Train grid-native autoencoder on Atari one-hot frames."""
        dev = torch.device(self.device_str)
        model = GridNativeEncoder(in_channels=self.n_channels).to(dev)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        data = torch.from_numpy(frames).float()
        n = len(data)

        best_loss = float("inf")
        best_state = None

        for epoch in range(epochs):
            model.train()
            perm = torch.randperm(n)
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

        if best_state:
            model.load_state_dict(best_state)
        return model.cpu()

    def _find_episode_boundaries(self, frames: np.ndarray, next_frames: np.ndarray) -> np.ndarray:
        """Identify episode breaks where frames[i+1] != next_frames[i]."""
        cache = self.data_dir / f"{self.game}_boundaries.npy"
        if cache.exists():
            b = np.load(str(cache))
            if b.size >= 2 and b[-1] == len(frames):
                return b

        n = len(frames)
        chunk = 4096
        breaks: list[int] = []
        for start in range(0, n - 1, chunk):
            end = min(start + chunk, n - 1)
            a = next_frames[start:end]
            b = frames[start + 1 : end + 1]
            mism = (a != b).reshape(end - start, -1).any(axis=1)
            for j in np.nonzero(mism)[0]:
                breaks.append(int(start + j + 1))

        boundaries = np.array([0] + breaks + [n], dtype=np.int64)
        try:
            np.save(str(cache), boundaries)
        except Exception:
            pass
        return boundaries

    def _ensure_data(self):
        """Load cached data or collect + train encoder + encode."""
        frames_path = self.data_dir / f"{self.game}_frames.npy"
        actions_path = self.data_dir / f"{self.game}_actions.npy"
        next_frames_path = self.data_dir / f"{self.game}_next_frames.npy"
        encoder_path = self.data_dir / f"{self.game}_encoder.pt"
        latents_path = self.data_dir / f"{self.game}_frames.npy".replace("_frames", "_latents")

        # Use latents path as canonical
        latent_file = self.data_dir / f"{self.game}_latents.npy"
        next_latent_file = self.data_dir / f"{self.game}_next_latents.npy"

        # If latents already exist, we're done
        if latent_file.exists() and next_latent_file.exists():
            return

        # Collect raw frames if needed
        if not frames_path.exists():
            print(f"[AtariLatentBenchmark] Collecting {self.n_frames} {self.game} frames...")
            frames, actions, next_frames = self._collect_frames()
            np.save(str(frames_path), frames)
            np.save(str(actions_path), actions)
            np.save(str(next_frames_path), next_frames)
        else:
            frames = np.load(str(frames_path))
            actions = np.load(str(actions_path))
            next_frames = np.load(str(next_frames_path))
            # Truncate if more frames than requested
            if len(frames) > self.n_frames:
                frames = frames[: self.n_frames]
                actions = actions[: self.n_frames]
                next_frames = next_frames[: self.n_frames]

        # Train encoder if checkpoint doesn't exist
        if not encoder_path.exists():
            print(f"[AtariLatentBenchmark] Training {self.game} grid-native encoder...")
            encoder = self._train_encoder(frames)
            torch.save(encoder.state_dict(), str(encoder_path))
        else:
            encoder = GridNativeEncoder(in_channels=self.n_channels)
            encoder.load_state_dict(torch.load(str(encoder_path), map_location="cpu",
                                               weights_only=True))
        encoder.eval()

        # Encode all frames
        print(f"[AtariLatentBenchmark] Encoding {self.game} frames...")
        dev = torch.device(self.device_str)
        encoder = encoder.to(dev)

        batch_size = 512
        latents_list, next_latents_list = [], []
        with torch.no_grad():
            for i in range(0, len(frames), batch_size):
                xb = torch.from_numpy(frames[i : i + batch_size]).float().to(dev)
                nx = torch.from_numpy(next_frames[i : i + batch_size]).float().to(dev)
                latents_list.append(encoder.encode(xb).cpu().numpy())
                next_latents_list.append(encoder.encode(nx).cpu().numpy())

        latents = np.concatenate(latents_list, axis=0)
        next_latents = np.concatenate(next_latents_list, axis=0)

        np.save(str(latent_file), latents)
        np.save(str(next_latent_file), next_latents)

        # Save boundaries
        self._find_episode_boundaries(latents, next_latents)

    def get_training_data(self, val_split: float = 0.15) -> tuple[np.ndarray, np.ndarray]:
        """Return (X_train, Y_train) for training.

        X: (N, 2, H, W) = [latent(1ch), action_field(1ch)]
        Y: (N, 1, H, W) = next_latent
        """
        latents = np.load(str(self.data_dir / f"{self.game}_latents.npy"))
        actions = np.load(str(self.data_dir / f"{self.game}_actions.npy"))
        next_latents = np.load(str(self.data_dir / f"{self.game}_next_latents.npy"))

        n = len(latents)
        n_val = int(n * val_split)
        n_train = n - n_val

        # Build X with action field
        X = self._build_action_conditioned(latents, actions)
        Y = next_latents[:, np.newaxis] if next_latents.ndim == 3 else next_latents

        return X[:n_train], Y[:n_train]

    def get_validation_data(self, val_split: float = 0.15) -> tuple[np.ndarray, np.ndarray]:
        """Return (X_val, Y_val)."""
        latents = np.load(str(self.data_dir / f"{self.game}_latents.npy"))
        actions = np.load(str(self.data_dir / f"{self.game}_actions.npy"))
        next_latents = np.load(str(self.data_dir / f"{self.game}_next_latents.npy"))

        n = len(latents)
        n_val = int(n * val_split)
        n_train = n - n_val

        X = self._build_action_conditioned(latents, actions)
        Y = next_latents[:, np.newaxis] if next_latents.ndim == 3 else next_latents

        return X[n_train:], Y[n_train:]

    def _build_action_conditioned(self, latents: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """Build X = concat[latent(1ch), action_field(1ch)].

        Action field: (a+1)/n_actions broadcast to grid.
        """
        n = len(latents)
        h, w = self.grid_h, self.grid_w

        # Ensure latents has shape (N, 1, H, W)
        if latents.ndim == 3:
            latents = latents[:, np.newaxis, :, :]

        action_vals = (actions.astype(np.float32) + 1.0) / self.n_actions
        action_field = np.zeros((n, 1, h, w), dtype=np.float32)
        action_field[:, 0, :, :] = action_vals[:, np.newaxis, np.newaxis]

        return np.concatenate([latents, action_field], axis=1)

    def get_test_trajectories(self, n_trajectories: int | None = None) -> list[dict]:
        """Return list of test trajectories for rollout evaluation.

        Each trajectory:
          - frames: (T+1, 1, H, W) — latent frames including extra for t+1 GT
          - actions: (T,) int — actions

        Uses last 15% of frames as held-out test data (separate from val split which
        uses the preceding 15% in get_validation_data).
        """
        frames = np.load(str(self.data_dir / f"{self.game}_frames.npy"))
        next_frames = np.load(str(self.data_dir / f"{self.game}_next_frames.npy"))
        actions = np.load(str(self.data_dir / f"{self.game}_actions.npy"))
        boundaries = self._find_episode_boundaries(frames, next_frames)

        # Ensure (N, 1, H, W)
        if frames.ndim == 3:
            frames = frames[:, np.newaxis, :, :]

        # Use last 15% as held-out test (val uses preceding 15%)
        n = len(frames)
        n_test = int(n * 0.15)
        test_start = n - n_test

        test_frames = frames[test_start:]
        test_actions = actions[test_start:]

        # Shift boundaries to test-relative indices
        rel_boundaries = boundaries[boundaries >= test_start] - test_start
        rel_boundaries = np.unique(np.concatenate([[0], rel_boundaries, [n_test]]))

        trajectories = []
        min_len = 10

        for i in range(len(rel_boundaries) - 1):
            start = int(rel_boundaries[i])
            end = int(rel_boundaries[i + 1])
            length = end - start
            # Need at least min_len+1 frames to have min_len rollout steps
            if length < min_len + 1:
                continue

            # frames: start..end (L frames) — last one is GT for final action
            # actions: start..end-1 (L-1 actions)
            traj = {
                "frames": test_frames[start:end],
                "actions": test_actions[start:end - 1],
                "length": length - 1,  # number of rollout steps
            }
            trajectories.append(traj)

        trajectories.sort(key=lambda t: t["length"], reverse=True)

        if n_trajectories is not None:
            trajectories = trajectories[:n_trajectories]

        return trajectories
