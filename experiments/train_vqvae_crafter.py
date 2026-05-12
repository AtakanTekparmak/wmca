"""Train VQ-VAE on Crafter frames — proper multi-epoch run.

Memory budget: < 3GB total (mmap frames, tiny model, batch=8).
Device: MPS, dtype: float32, no torch.compile.
"""
from __future__ import annotations

import gc
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from wmca.modules.vqvae import VQVAE


def get_memory_mb() -> dict[str, float]:
    """Get process memory in MB (RSS). MPS doesn't give device mem easily."""
    import resource
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS returns KB
    return {"rss_mb": rss / 1024}


def main():
    # ── Config ──────────────────────────────────────────────────────────
    DATA_PATH = Path(__file__).parent / "crafter_data" / "frames.npy"
    OUT_DIR = Path(__file__).parent / "crafter_data"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    BATCH_SIZE = 8
    NUM_EPOCHS = 50
    LR = 3e-4
    NUM_EMBEDDINGS = 256
    EMBED_DIM = 64
    COMMITMENT_COST = 0.25
    VAL_SPLIT = 0.10
    LOG_EVERY_N_BATCHES = 100
    SAVE_EVERY_N_EPOCHS = 10

    device = torch.device("mps")
    print(f"Device: {device}")
    print(f"Memory before load: {get_memory_mb()}")

    # ── Data (mmap — never loads fully into RAM) ────────────────────────
    frames = np.load(DATA_PATH, mmap_mode="r")  # (100000, 3, 64, 64) float32 [0,1]
    N = len(frames)
    n_val = int(N * VAL_SPLIT)
    n_train = N - n_val

    # Shuffle indices once
    rng = np.random.default_rng(42)
    all_idx = rng.permutation(N)
    train_idx = all_idx[:n_train]
    val_idx = all_idx[n_train:]

    print(f"Total frames: {N} (~{frames.nbytes / 1024**2:.0f} MB on disk)")
    print(f"Train: {n_train}, Val: {n_val}")
    print(f"Batches/epoch: {n_train // BATCH_SIZE}")

    # ── Model ───────────────────────────────────────────────────────────
    model = VQVAE(
        num_embeddings=NUM_EMBEDDINGS,
        embed_dim=EMBED_DIM,
        in_channels=3,
        commitment_cost=COMMITMENT_COST,
    ).to(device)
    model = model.float()  # ensure float32

    param_counts = model.param_count()
    print(f"Model params: {param_counts} (~{param_counts['total']*4/1024**2:.1f} MB float32)")

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=NUM_EPOCHS, eta_min=1e-5)

    print(f"Memory after model load: {get_memory_mb()}")

    # ── Training Loop ───────────────────────────────────────────────────
    history: list[dict] = []
    best_val_loss = float("inf")
    t_start = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        # --- Train ---
        model.train()
        train_recon = 0.0
        train_vq = 0.0
        train_codebook_used = 0.0
        n_batches = 0

        # Shuffle train indices each epoch
        epoch_train_idx = rng.permutation(train_idx)
        batches = (len(epoch_train_idx) + BATCH_SIZE - 1) // BATCH_SIZE

        for b in range(batches):
            start = b * BATCH_SIZE
            end = min(start + BATCH_SIZE, len(epoch_train_idx))
            batch_indices = epoch_train_idx[start:end]

            # Read from mmap only what we need
            xb = torch.from_numpy(frames[batch_indices].copy()).float().to(device)

            recon, indices, vq_loss = model(xb)
            recon_loss = nn.functional.mse_loss(recon, xb)

            loss = recon_loss + vq_loss

            opt.zero_grad()
            loss.backward()
            opt.step()

            train_recon += recon_loss.item()
            train_vq += vq_loss.item()
            train_codebook_used += model.codebook_utilization()
            n_batches += 1

            if b > 0 and b % LOG_EVERY_N_BATCHES == 0:
                mem = get_memory_mb()
                elapsed = time.time() - t_start
                print(
                    f"  [e{epoch}/{NUM_EPOCHS} b{b}/{batches}] "
                    f"r={train_recon/n_batches:.5f} vq={train_vq/n_batches:.5f} "
                    f"cb={model.codebook_utilization():.2f} "
                    f"mem={mem['rss_mb']:.0f}MB "
                    f"elapsed={elapsed:.0f}s"
                )

        train_recon /= n_batches
        train_vq /= n_batches
        train_codebook_used /= n_batches
        scheduler.step()
        lr_now = scheduler.get_last_lr()[0]

        # --- Validation ---
        model.eval()
        val_recon = 0.0
        val_vq = 0.0
        val_batches = 0
        val_batch_indices = [val_idx[i:i + BATCH_SIZE] for i in range(0, n_val, BATCH_SIZE)]

        with torch.no_grad():
            for batch_indices in val_batch_indices:
                xb = torch.from_numpy(frames[batch_indices].copy()).float().to(device)
                recon, indices, vq_loss = model(xb)
                val_recon += nn.functional.mse_loss(recon, xb).item()
                val_vq += vq_loss.item()
                val_batches += 1

        val_recon /= val_batches
        val_vq /= val_batches
        cb_now = model.codebook_utilization()
        elapsed = time.time() - t_start

        epoch_dict = {
            "epoch": epoch,
            "train_recon": train_recon,
            "train_vq": train_vq,
            "val_recon": val_recon,
            "val_vq": val_vq,
            "train_cb": train_codebook_used,
            "val_cb": cb_now,
            "lr": lr_now,
            "elapsed_s": elapsed,
            "mem_mb": get_memory_mb()["rss_mb"],
        }
        history.append(epoch_dict)

        print(
            f"EPOCH {epoch:3d}/{NUM_EPOCHS} | "
            f"train r={train_recon:.6f} vq={train_vq:.6f} | "
            f"val r={val_recon:.6f} vq={val_vq:.6f} | "
            f"cb={cb_now:.3f} | lr={lr_now:.2e} | "
            f"mem={epoch_dict['mem_mb']:.0f}MB | {elapsed:.0f}s"
        )

        # --- Checkpoint ---
        if val_recon < best_val_loss:
            best_val_loss = val_recon
            torch.save(
                {"epoch": epoch, "model_state": model.state_dict(),
                 "opt_state": opt.state_dict(), "history": history},
                OUT_DIR / "vqvae_best.pt",
            )
            print(f"  → Best checkpoint (val_rec={val_recon:.6f})")

        if epoch % SAVE_EVERY_N_EPOCHS == 0:
            torch.save(
                {"epoch": epoch, "model_state": model.state_dict(),
                 "opt_state": opt.state_dict(), "history": history},
                OUT_DIR / f"vqvae_e{epoch:03d}.pt",
            )

    # ── Final ───────────────────────────────────────────────────────────
    torch.save(
        {"epoch": NUM_EPOCHS, "model_state": model.state_dict(),
         "opt_state": opt.state_dict(), "history": history},
        OUT_DIR / "vqvae_final.pt",
    )

    t_total = time.time() - t_start
    print(f"\nDone in {t_total:.0f}s ({t_total/60:.1f}min).")
    print(f"Best val recon: {best_val_loss:.6f}")
    print(f"Final codebook utilization: {model.codebook_utilization():.3f}")
    print(f"Peak memory: {get_memory_mb()}")

    # Quick sanity: show a reconstruction sample
    model.eval()
    with torch.no_grad():
        sample_idx = val_idx[:4]
        x_sample = torch.from_numpy(frames[sample_idx].copy()).float().to(device)
        recon_sample, _, _ = model(x_sample)
        mse = nn.functional.mse_loss(recon_sample, x_sample).item()
        print(f"Sample reconstruction MSE: {mse:.6f}")

    return history


if __name__ == "__main__":
    main()
