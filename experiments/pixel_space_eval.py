"""Pixel-space prediction quality: encode -> predict in latent -> decode back.

Compares our models' pixel-space L2 against DELTA-IRIS's autoencoder L2 (0.000185).

Usage:
    uv run --with crafter,scikit-learn python experiments/pixel_space_eval.py
"""
import numpy as np
import torch

from wmca.modules.frame_encoder import FrameAutoencoder
from wmca.crafter_real import generate_crafter_real
from wmca.model_registry import create_model, train_model


def main():
    print("=== Pixel-space prediction quality ===\n")

    # Load autoencoder
    autoencoder = FrameAutoencoder()
    autoencoder.load_state_dict(
        torch.load("experiments/crafter_data/frame_encoder.pt",
                    map_location="cpu", weights_only=True)
    )
    autoencoder.eval()

    # Load raw frames
    frames = np.load("experiments/crafter_data/frames.npy", mmap_mode="r")
    next_frames = np.load("experiments/crafter_data/next_frames.npy", mmap_mode="r")
    actions = np.load("experiments/crafter_data/actions.npy", mmap_mode="r")

    # Test set: frames 90K-91K (outside the 50K used for benchmark training)
    test_start = 90000
    n_test = 1000
    test_frames = torch.from_numpy(frames[test_start:test_start + n_test].copy())
    test_next = torch.from_numpy(next_frames[test_start:test_start + n_test].copy())
    test_actions = actions[test_start:test_start + n_test].copy()
    print(f"Test set: {n_test} frames from index {test_start}\n")

    # 1. Autoencoder-only baseline (encode -> decode, no world model)
    with torch.no_grad():
        ae_recon = autoencoder(test_next)
        ae_l2 = ((test_next - ae_recon) ** 2).mean().item()
    print(f"Autoencoder reconstruction L2 (next frame): {ae_l2:.6f}")
    print(f"  (DELTA-IRIS autoencoder L2: 0.000185)\n")

    # 2. Load benchmark data and train models
    print("Loading crafter_real benchmark...")
    data = generate_crafter_real(n_frames=50000)

    # Encode test frames
    with torch.no_grad():
        enc_test = autoencoder.encode(test_frames)
        enc_test_next_gt = autoencoder.encode(test_next)

    # Build action fields
    action_fields = torch.zeros(n_test, 1, 16, 16)
    for i in range(n_test):
        action_fields[i, 0, :, :] = (test_actions[i] + 1.0) / 17.0
    test_X = torch.cat([enc_test, action_fields], dim=1)

    results = {}
    for model_name in ["rescor", "pure_nca", "conv2d", "rescor_e3c", "rescor_mp_gate"]:
        print(f"\nTraining {model_name}...")
        model = create_model(model_name, in_channels=2, out_channels=1, grid_size=16)
        model = train_model(
            model, data.X_train, data.Y_train,
            X_val=data.X_val, Y_val=data.Y_val,
            loss_type="mse", epochs=30, batch_size=64, lr=1e-3,
        )

        model.eval()
        with torch.no_grad():
            pred_latent = model(test_X)
            pred_pixels = autoencoder.decode(pred_latent)

        pixel_l2 = ((test_next - pred_pixels) ** 2).mean().item()
        latent_mse = ((enc_test_next_gt - pred_latent) ** 2).mean().item()

        pc = model.param_count()
        results[model_name] = {
            "pixel_l2": pixel_l2,
            "latent_mse": latent_mse,
            "params": pc,
        }
        print(f"  pixel L2: {pixel_l2:.6f}, latent MSE: {latent_mse:.6f}")

    print("\n===== PIXEL-SPACE RESULTS =====")
    print(f"{'Model':20s}  {'Pixel L2':>12s}  {'Latent MSE':>12s}  {'Params':>20s}")
    print("-" * 70)
    print(f"{'AE only (no WM)':20s}  {ae_l2:12.6f}  {'n/a':>12s}  {'34K frozen':>20s}")
    for name, r in sorted(results.items(), key=lambda x: x[1]["pixel_l2"]):
        trained = r["params"]["trained"]
        print(f"{name:20s}  {r['pixel_l2']:12.6f}  {r['latent_mse']:12.6f}  {str(trained)+' + 34K frozen':>20s}")
    print("-" * 70)
    print(f"{'DELTA-IRIS (paper)':20s}  {0.000185:12.6f}  {'n/a':>12s}  {'25M':>20s}")


if __name__ == "__main__":
    main()
