"""Crafter frame collection + autoencoder training.

1. Collects N frames from Crafter using a random policy.
2. Trains a FrameAutoencoder on the collected frames with MSE loss.

Usage:
    uv run --with crafter,scikit-learn python experiments/train_frame_encoder.py
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from wmca.modules.frame_encoder import FrameAutoencoder


# ── Frame collection ────────────────────────────────────────────────────────


def collect_frames(n_frames: int, seed: int, data_dir: str) -> None:
    """Collect frames from Crafter with a random policy and save to disk."""
    import crafter

    os.makedirs(data_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    env = crafter.Env()

    frames = np.empty((n_frames, 3, 64, 64), dtype=np.float32)
    actions = np.empty(n_frames, dtype=np.int64)
    next_frames = np.empty((n_frames, 3, 64, 64), dtype=np.float32)

    idx = 0
    episodes = 0
    while idx < n_frames:
        obs = env.reset()
        done = False
        while not done and idx < n_frames:
            action = int(rng.integers(0, 17))
            next_obs, _reward, done, _info = env.step(action)

            # uint8 HWC -> float32 CHW normalised to [0, 1]
            frames[idx] = np.transpose(obs, (2, 0, 1)).astype(np.float32) / 255.0
            actions[idx] = action
            next_frames[idx] = np.transpose(next_obs, (2, 0, 1)).astype(np.float32) / 255.0

            obs = next_obs
            idx += 1

            if idx % 10_000 == 0:
                print(f"  collected {idx}/{n_frames} frames")
        episodes += 1

    print(f"  done: {n_frames} frames from {episodes} episodes")

    np.save(os.path.join(data_dir, "frames.npy"), frames)
    np.save(os.path.join(data_dir, "actions.npy"), actions)
    np.save(os.path.join(data_dir, "next_frames.npy"), next_frames)
    print(f"  saved to {data_dir}/")


# ── Training ────────────────────────────────────────────────────────────────


def train_autoencoder(
    data_dir: str,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> None:
    """Train FrameAutoencoder on collected frames."""
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  device: {device}")

    # Load frames
    frames = np.load(os.path.join(data_dir, "frames.npy"))
    frames = torch.from_numpy(frames)
    n = len(frames)
    print(f"  loaded {n} frames, shape {tuple(frames.shape)}")

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


# ── CLI ─────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect Crafter frames and train a frame autoencoder."
    )
    parser.add_argument("--n-frames", type=int, default=100_000)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip-collection",
        action="store_true",
        help="Skip frame collection if data already exists.",
    )
    parser.add_argument("--data-dir", type=str, default="experiments/crafter_data")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Frame collection
    data_exists = all(
        os.path.exists(os.path.join(args.data_dir, f))
        for f in ("frames.npy", "actions.npy", "next_frames.npy")
    )

    if args.skip_collection and data_exists:
        print("Skipping frame collection (data already exists).")
    else:
        print(f"Collecting {args.n_frames} frames from Crafter ...")
        collect_frames(args.n_frames, args.seed, args.data_dir)

    # Training
    print("Training autoencoder ...")
    train_autoencoder(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
