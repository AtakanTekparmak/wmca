"""Pixel-space prediction quality for DOOM: encode -> predict in latent -> decode.

Computes PSNR, SSIM, and optionally LPIPS on decoded predictions vs ground truth
at 64x64 resolution.  Prints a comparison table including GameNGen reference numbers.

Usage:
    uv run --with scikit-learn,scikit-image python experiments/doom_pixel_eval.py
    # With LPIPS (optional):
    uv run --with scikit-learn,scikit-image,lpips python experiments/doom_pixel_eval.py
"""
import math

import numpy as np
import torch
import torch.nn.functional as F

from wmca.modules.frame_encoder import FrameAutoencoder
from wmca.doom_real import generate_doom_real
from wmca.model_registry import create_model, train_model


# -- Metrics -----------------------------------------------------------------


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Per-image PSNR averaged over batch. Inputs in [0, 1]."""
    mse = ((pred - target) ** 2).mean(dim=(1, 2, 3))  # (B,)
    # Clamp to avoid log(0)
    mse = mse.clamp(min=1e-10)
    psnr_vals = 10.0 * torch.log10(1.0 / mse)
    return psnr_vals.mean().item()


def ssim_batch(pred: torch.Tensor, target: torch.Tensor,
               window_size: int = 7) -> float:
    """Simple SSIM averaged over batch. Inputs: (B, C, H, W) in [0, 1].

    Uses skimage for per-image computation to match standard implementations.
    """
    try:
        from skimage.metrics import structural_similarity
    except ImportError:
        return float("nan")

    pred_np = pred.cpu().numpy()
    target_np = target.cpu().numpy()
    vals = []
    for i in range(len(pred_np)):
        # channel_axis=0 because shape is (C, H, W)
        s = structural_similarity(
            pred_np[i], target_np[i],
            data_range=1.0, channel_axis=0,
            win_size=window_size,
        )
        vals.append(s)
    return float(np.mean(vals))


def lpips_batch(pred: torch.Tensor, target: torch.Tensor,
                lpips_fn) -> float:
    """LPIPS averaged over batch. lpips_fn is a pre-created lpips.LPIPS model."""
    # LPIPS expects inputs in [-1, 1]
    pred_scaled = pred * 2.0 - 1.0
    target_scaled = target * 2.0 - 1.0
    with torch.no_grad():
        d = lpips_fn(pred_scaled, target_scaled)  # (B, 1, 1, 1)
    return d.mean().item()


# -- Main --------------------------------------------------------------------


def main():
    print("=== DOOM Pixel-space Prediction Quality ===\n")

    data_dir = "experiments/doom_data"

    # Load autoencoder
    autoencoder = FrameAutoencoder()
    autoencoder.load_state_dict(
        torch.load(f"{data_dir}/frame_encoder.pt",
                    map_location="cpu", weights_only=True)
    )
    autoencoder.eval()

    # Load raw frames and resize to 64x64
    frames_raw = np.load(f"{data_dir}/frames.npy", mmap_mode="r")
    next_frames_raw = np.load(f"{data_dir}/next_frames.npy", mmap_mode="r")
    actions = np.load(f"{data_dir}/actions.npy", mmap_mode="r")

    # Infer n_actions from data
    n_actions = int(actions.max()) + 1
    print(f"Detected {n_actions} actions in dataset\n")

    # Test set: use frames beyond the 50K training window
    test_start = min(90000, len(frames_raw) - 1001)
    n_test = min(1000, len(frames_raw) - test_start)
    test_frames = torch.from_numpy(frames_raw[test_start:test_start + n_test].copy())
    test_next = torch.from_numpy(next_frames_raw[test_start:test_start + n_test].copy())
    test_actions = actions[test_start:test_start + n_test].copy()

    # Resize to 64x64 if needed
    _, _, h, w = test_frames.shape
    if h != 64 or w != 64:
        test_frames = F.interpolate(test_frames, size=(64, 64), mode="bilinear",
                                    align_corners=False)
        test_next = F.interpolate(test_next, size=(64, 64), mode="bilinear",
                                  align_corners=False)

    print(f"Test set: {n_test} frames from index {test_start}")
    print(f"Test frame shape: {tuple(test_frames.shape)}\n")

    # Try to load LPIPS
    lpips_fn = None
    try:
        import lpips
        lpips_fn = lpips.LPIPS(net="alex", verbose=False)
        lpips_fn.eval()
        print("LPIPS (AlexNet) loaded\n")
    except ImportError:
        print("LPIPS not available (install with: pip install lpips)\n")

    # 1. Autoencoder-only baseline (encode -> decode, no world model)
    with torch.no_grad():
        ae_recon = autoencoder(test_next)
        ae_l2 = ((test_next - ae_recon) ** 2).mean().item()
        ae_psnr = psnr(ae_recon, test_next)
        ae_ssim = ssim_batch(ae_recon, test_next)
        ae_lpips = lpips_batch(ae_recon, test_next, lpips_fn) if lpips_fn else float("nan")

    print(f"Autoencoder reconstruction (next frame):")
    print(f"  L2={ae_l2:.6f}  PSNR={ae_psnr:.2f}  SSIM={ae_ssim:.4f}  LPIPS={ae_lpips:.4f}\n")

    # 2. Load benchmark and train models
    print("Loading doom_real benchmark ...")
    data = generate_doom_real(n_frames=50000, n_actions=n_actions, data_dir=data_dir)

    # Encode test frames
    with torch.no_grad():
        enc_test = autoencoder.encode(test_frames)
        enc_test_next_gt = autoencoder.encode(test_next)

    # Build action fields
    action_fields = torch.zeros(n_test, 1, 16, 16)
    for i in range(n_test):
        action_fields[i, 0, :, :] = (test_actions[i] + 1.0) / n_actions
    test_X = torch.cat([enc_test, action_fields], dim=1)

    results = {}
    for model_name in ["rescor", "pure_nca", "conv2d", "rescor_mp_gate"]:
        print(f"\nTraining {model_name} ...")
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
        p = psnr(pred_pixels, test_next)
        s = ssim_batch(pred_pixels, test_next)
        lp = lpips_batch(pred_pixels, test_next, lpips_fn) if lpips_fn else float("nan")

        pc = model.param_count()
        results[model_name] = {
            "pixel_l2": pixel_l2,
            "latent_mse": latent_mse,
            "psnr": p,
            "ssim": s,
            "lpips": lp,
            "params": pc,
        }
        print(f"  pixel L2={pixel_l2:.6f}  PSNR={p:.2f}  SSIM={s:.4f}  LPIPS={lp:.4f}")

    # 3. Results table
    print("\n" + "=" * 95)
    print("DOOM PIXEL-SPACE RESULTS")
    print("=" * 95)
    header = (f"{'Model':20s}  {'Pixel L2':>10s}  {'PSNR':>7s}  "
              f"{'SSIM':>7s}  {'LPIPS':>7s}  {'Latent MSE':>12s}  {'Params':>20s}")
    print(header)
    print("-" * 95)

    # AE-only row
    print(f"{'AE only (no WM)':20s}  {ae_l2:10.6f}  {ae_psnr:7.2f}  "
          f"{ae_ssim:7.4f}  {ae_lpips:7.4f}  {'n/a':>12s}  {'34K frozen':>20s}")

    # Model rows
    for name, r in sorted(results.items(), key=lambda x: -x[1]["psnr"]):
        trained = r["params"]["trained"]
        lp_str = f"{r['lpips']:7.4f}" if not math.isnan(r["lpips"]) else "   n/a "
        print(f"{name:20s}  {r['pixel_l2']:10.6f}  {r['psnr']:7.2f}  "
              f"{r['ssim']:7.4f}  {lp_str}  {r['latent_mse']:12.6f}  "
              f"{str(trained) + ' + 34K frozen':>20s}")

    print("-" * 95)
    # GameNGen reference: PSNR ~29.43, SSIM ~0.85, LPIPS ~0.049 (from paper)
    print(f"{'GameNGen (paper)':20s}  {'n/a':>10s}  {'29.43':>7s}  "
          f"{'0.8500':>7s}  {'0.0490':>7s}  {'n/a':>12s}  {'~500M':>20s}")
    print("=" * 95)
    print("\nNote: GameNGen operates at 320x240; our eval is at 64x64.")
    print("Direct PSNR/SSIM comparison requires matching resolution and scenario.")


if __name__ == "__main__":
    main()
