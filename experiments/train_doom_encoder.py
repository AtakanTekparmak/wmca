"""DOOM frame autoencoder training.

Trains a FrameAutoencoder on pre-collected DOOM frames (resized to 64x64).
Frames must already exist at <data-dir>/frames.npy as (N, 3, H, W) float32 [0,1]
(any resolution -- they are resized to 64x64 on load).

Usage:
    uv run --with scikit-learn python experiments/train_doom_encoder.py
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from wmca.modules.frame_encoder import FrameAutoencoder


# -- Data loading ------------------------------------------------------------


def load_and_resize_frames(data_dir: str, target_size: int = 64) -> torch.Tensor:
    """Load DOOM frames and resize to (target_size, target_size).

    Expects frames.npy with shape (N, 3, H, W) float32 in [0, 1].
    Returns (N, 3, target_size, target_size) float32 tensor.
    """
    path = os.path.join(data_dir, "frames.npy")
    frames = np.load(path)
    print(f"  loaded {len(frames)} frames, raw shape {tuple(frames.shape)}")

    frames_t = torch.from_numpy(frames)
    _, _, h, w = frames_t.shape
    if h != target_size or w != target_size:
        print(f"  resizing from {h}x{w} to {target_size}x{target_size}")
        frames_t = F.interpolate(
            frames_t, size=(target_size, target_size), mode="bilinear",
            align_corners=False,
        )
    return frames_t


# -- Training ----------------------------------------------------------------


def train_autoencoder(
    data_dir: str,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> None:
    """Train FrameAutoencoder on pre-collected DOOM frames (resized to 64x64)."""
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  device: {device}")

    frames = load_and_resize_frames(data_dir)
    n = len(frames)

    # 90/10 split
    from sklearn.model_selection import train_test_split

    train_idx, val_idx = train_test_split(
        np.arange(n), test_size=0.1, random_state=seed
    )
    train_ds = TensorDataset(frames[train_idx])
    val_ds = TensorDataset(frames[val_idx])
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    print(f"  train: {len(train_idx)}, val: {len(val_idx)}")

    # Model
    model = FrameAutoencoder().to(device)
    counts = model.param_count()
    print(f"  params: {counts}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    patience_counter = 0
    patience = 10
    save_path = os.path.join(data_dir, "frame_encoder.pt")

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss_sum = 0.0
        train_batches = 0
        for (batch,) in train_dl:
            batch = batch.to(device)
            recon = model(batch)
            loss = criterion(recon, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()
            train_batches += 1
        train_loss = train_loss_sum / train_batches

        # Val
        model.eval()
        val_loss_sum = 0.0
        val_batches = 0
        with torch.no_grad():
            for (batch,) in val_dl:
                batch = batch.to(device)
                recon = model(batch)
                loss = criterion(recon, batch)
                val_loss_sum += loss.item()
                val_batches += 1
        val_loss = val_loss_sum / val_batches

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"  epoch {epoch:3d}/{epochs}  "
                f"train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  "
                f"best={best_val_loss:.6f}  patience={patience_counter}/{patience}"
            )

        if patience_counter >= patience:
            print(f"  early stopping at epoch {epoch}")
            break

    # Reload best model and report
    model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
    model.eval()
    print(f"\n  best val loss: {best_val_loss:.6f}")
    print(f"  saved to {save_path}")

    # Per-sample reconstruction MSE on a few validation samples
    print("\n  sample reconstructions (val set):")
    sample_batch = frames[val_idx[:8]].to(device)
    with torch.no_grad():
        recons = model(sample_batch)
    for i in range(len(sample_batch)):
        mse = ((sample_batch[i] - recons[i]) ** 2).mean().item()
        print(f"    sample {i}: MSE = {mse:.6f}")


# -- CLI ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a frame autoencoder on pre-collected DOOM frames."
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=str, default="experiments/doom_data")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    frames_path = os.path.join(args.data_dir, "frames.npy")
    if not os.path.exists(frames_path):
        print(f"ERROR: {frames_path} not found.")
        print("Collect DOOM frames first (e.g. via a separate ViZDoom collection script).")
        return

    print("Training autoencoder on DOOM frames ...")
    train_autoencoder(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
