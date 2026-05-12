"""Train Atari frame encoder — Path A.1 of Plan 0.

Collects Pong/Breakout frames via random play, trains a grid-native
autoencoder, encodes all frames to latent grids.

The AtariLatentBenchmark class handles collection, training, and encoding.
This script is a CLI wrapper with configurable parameters.

Output:
    experiments/atari_data/{game}_frames.npy         — raw one-hot frames
    experiments/atari_data/{game}_actions.npy        — action sequences
    experiments/atari_data/{game}_next_frames.npy    — next-frame data
    experiments/atari_data/{game}_encoder.pt         — trained encoder weights
    experiments/atari_data/{game}_latents.npy        — encoded latent grids
    experiments/atari_data/{game}_next_latents.npy   — next-frame latent grids

Usage:
    PYTHONPATH=src uv run python experiments/train_atari_encoder.py
    PYTHONPATH=src GAME=breakout N_FRAMES=200000 uv run \\
        python experiments/train_atari_encoder.py
"""
from __future__ import annotations

import os

from wmca.atari_real import AtariLatentBenchmark


def main():
    game = os.environ.get("GAME", "pong")
    n_frames = int(os.environ.get("N_FRAMES", "200000"))
    device = os.environ.get("DEVICE", "cuda" if __import__("torch").cuda.is_available() else "cpu")

    print(f"Path A.1 — Atari Frame Encoder")
    print(f"  Game: {game}")
    print(f"  N frames: {n_frames}")
    print(f"  Device: {device}")

    benchmark = AtariLatentBenchmark(
        game=game,
        n_frames=n_frames,
        device=device,
    )

    # Data collection and encoding happens in __init__
    X_train, Y_train = benchmark.get_training_data()
    X_val, Y_val = benchmark.get_validation_data()

    print(f"\nDone. Data shapes:")
    print(f"  X_train: {X_train.shape} (in_channels=2: latent + action_field)")
    print(f"  Y_train: {Y_train.shape} (out_channels=1)")
    print(f"  X_val: {X_val.shape}, Y_val: {Y_val.shape}")

    # Quick sanity: reconstruction quality via the loaded encoder
    import numpy as np
    import torch
    from wmca.atari_real import GridNativeEncoder

    frames = np.load(f"experiments/atari_data/{game}_frames.npy")
    encoder = GridNativeEncoder(in_channels=benchmark.n_channels)
    encoder.load_state_dict(torch.load(
        f"experiments/atari_data/{game}_encoder.pt",
        map_location="cpu", weights_only=True,
    ))
    encoder.eval()

    # PSNR on a batch
    with torch.no_grad():
        xb = torch.from_numpy(frames[:128]).float()
        recon = encoder(xb)
        mse = torch.nn.functional.mse_loss(recon, xb).item()
        psnr = 10.0 * __import__("math").log10(1.0 / max(mse, 1e-10))
        print(f"  Reconstruction PSNR (sample batch): {psnr:.1f} dB")


if __name__ == "__main__":
    main()
