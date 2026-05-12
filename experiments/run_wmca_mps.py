#!/usr/bin/env python3
"""WMCA Experiments — M4 MPS local runner (3GB cap).

Orchestrates Path A (Atari encoder → rescor → rollout) and Path C
(VQ-VAE → DiscreteRescor → rollout) locally on Apple Silicon MPS.

Usage:
    PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.094 \\
    PYTHONPATH=src uv run python experiments/run_wmca_mps.py

    # Run only Path A or C:
    PYTHONPATH=src uv run python experiments/run_wmca_mps.py --path A
    PYTHONPATH=src uv run python experiments/run_wmca_mps.py --path C

    # Quick smoke test (fewer frames, fewer epochs):
    PYTHONPATH=src uv run python experiments/run_wmca_mps.py --quick

Stages:
  A.1 — Atari frame collection + grid-native encoder training
  A.2 — Atari latent rescor training (rescor_rens, 3 seeds)
  A.3 — Atari latent rollout probe (H={15,50,100}, 20 trajs)
  C.1 — VQ-VAE training on Crafter frames
  C.2 — Discrete token encoding
  C.3 — DiscreteRescor training (3 seeds)
  C.4 — Discrete token rollout probe (H={15,50,100})
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

# ── ensure wmca is importable ──────────────────────────────────────────────
_SRC = str(Path(__file__).resolve().parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# ── MPS memory cap ─────────────────────────────────────────────────────────
_MPS_RATIO = float(os.environ.get("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.094"))
_has_mps = torch.backends.mps.is_available()


def get_device() -> torch.device:
    if _has_mps:
        return torch.device("mps")
    return torch.device("cpu")


def print_header(title: str):
    print(f"\n{'='*70}\n  {title}\n{'='*70}")


def timestamp() -> str:
    return datetime.now().strftime("%H:%M:%S")


def param_count_str(model) -> str:
    try:
        pc = model.param_count()
        return f"{pc['trained']:,} trained + {pc.get('frozen', 0):,} frozen"
    except Exception:
        return f"{sum(p.numel() for p in model.parameters()):,}"


# ═══════════════════════════════════════════════════════════════════════════
#  Path A — Atari Latent Rollout Probe
# ═══════════════════════════════════════════════════════════════════════════

def run_atari_stage(args):
    """Run Path A: Atari encoder + rescor training + rollout."""
    from wmca.atari_real import AtariLatentBenchmark

    print_header("Path A — Atari Latent Rollout Probe")
    dev = get_device()
    print(f"  Device: {dev}  (MPS ratio: {_MPS_RATIO})")

    # ═══ A.1 — Atari encoder ══════════════════════════════════════════════
    print_header("A.1 — Atari Frame Encoder")
    t0 = time.time()

    game = args.game
    n_frames = args.n_frames // 10 if args.quick else args.n_frames

    bench = AtariLatentBenchmark(
        game=game,
        n_frames=n_frames,
        device=dev.type,
        seed=42,
    )
    X_train, Y_train = bench.get_training_data()
    X_val, Y_val = bench.get_validation_data()

    _, _, H, W = X_train.shape
    print(f"  Game: {game}  Grid: {H}x{W}")
    print(f"  N_frames: {n_frames}  Train: {len(X_train)}  Val: {len(X_val)}")
    print(f"  Time: {time.time() - t0:.0f}s")

    # ═══ A.2 — Train rescor on Atari latents ═════════════════════════════
    print_header("A.2 — Atari Latent Rescor Training")

    from wmca.modules.hybrid import ResidualCorrectionWM
    from wmca.model_registry import train_model

    seeds = [42] if args.quick else [42, 43, 44]
    epochs = args.epochs_atari // 5 if args.quick else args.epochs_atari
    in_ch, out_ch = 2, 1  # latent + action_field → latent

    trained_models = {}
    for seed in seeds:
        print(f"\n  [{timestamp()}] Seed {seed} — rescor_rens K=32")
        model = ResidualCorrectionWM(
            in_channels=in_ch,
            out_channels=out_ch,
            hidden_ch=16 if not args.quick else 8,
            cml_gate="multi_r_uniform",
            cml_K=32,
            seed=seed,
            use_sigmoid=True,
        )
        print(f"    Params: {param_count_str(model)}")

        t_seed = time.time()
        trained = train_model(
            model,
            X_train, Y_train,
            X_val, Y_val,
            loss_type="mse",
            epochs=epochs,
            batch_size=args.batch_size_small,
            lr=args.lr,
            device=dev,
            compile=False,   # MPS: no compile
            bf16=False,      # MPS: no bf16
        )
        trained_models[seed] = trained
        print(f"    Done in {time.time() - t_seed:.0f}s")

    # ═══ A.3 — Atari Rollout Probe ═══════════════════════════════════════
    print_header("A.3 — Atari Rollout Probe")

    from dreamerv3_scaffolding.rollout_stability_probe_atari import (
        rollout_mse, make_action_field,
    )

    test_trajs = bench.get_test_trajectories(n_trajectories=args.n_trajs)
    print(f"  Test trajectories: {len(test_trajs)}")

    all_results = {}
    for seed in seeds:
        model = trained_models[seed].to(torch.device("cpu")).eval()
        per_traj = []
        for tidx, traj in enumerate(test_trajs):
            r = rollout_mse(
                model, traj, horizon=100,
                device=torch.device("cpu"),
                n_actions=bench.n_actions,
            )
            r["traj_idx"] = tidx
            per_traj.append(r)

        h15_mse = np.nanmedian([r["abs_mse"].get("H=15", float("nan")) for r in per_traj])
        h50_mse = np.nanmedian([r["abs_mse"].get("H=50", float("nan")) for r in per_traj])
        h100_mse = np.nanmedian([r["abs_mse"].get("H=100", float("nan")) for r in per_traj])
        h15_ratio = np.nanmedian([r["ratios"].get("H=15", float("nan")) for r in per_traj])
        h100_cos = np.nanmedian([r["cos_div_at"].get("H=100", float("nan")) for r in per_traj])

        all_results[f"seed_{seed}"] = {
            "model": "rescor_rens",
            "game": game,
            "seed": seed,
            "n_trajectories": len(per_traj),
            "step1_mse_median": float(np.nanmedian([r["step1_mse"] for r in per_traj])),
            "H=15_abs_mse": float(h15_mse),
            "H=50_abs_mse": float(h50_mse),
            "H=100_abs_mse": float(h100_mse),
            "H=15_ratio": float(h15_ratio),
            "H=100_cos_div": float(h100_cos),
        }
        print(f"  Seed {seed}: H=15 ratio={h15_ratio:.2f}  H=100 cos_div={h100_cos:.4f}")

    # Save results
    out_path = Path("experiments/results/atari_rollout_probe_mps.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "config": {"game": game, "seeds": seeds, "epochs": epochs,
                       "device": dev.type, "grid": f"{H}x{W}"},
            "per_seed": all_results,
        }, f, indent=2)
    print(f"\n  Results → {out_path}")


# ═══════════════════════════════════════════════════════════════════════════
#  Path C — VQ-VAE + DiscreteRescor
# ═══════════════════════════════════════════════════════════════════════════

def run_vqvae_stage(args):
    """Run Path C: VQ-VAE + DiscreteRescor + rollout."""
    from wmca.modules.vqvae import VQVAE

    print_header("Path C — VQ-VAE + DiscreteRescor")
    dev = get_device()
    print(f"  Device: {dev}  (MPS ratio: {_MPS_RATIO})")

    # ═══ C.1 — VQ-VAE training ════════════════════════════════════════════
    print_header("C.1 — VQ-VAE Training")

    data_dir = Path("experiments/crafter_data")

    # Load Crafter frames
    frames_path = data_dir / "frames.npy"
    if not frames_path.exists():
        print("  ERROR: No Crafter frames found at experiments/crafter_data/frames.npy")
        print("  Run crafter frame collection first.")
        return

    frames = np.load(frames_path).astype(np.float32)
    # frames should be in [0,1]; clamp just in case
    frames = np.clip(frames, 0.0, 1.0)

    if args.quick:
        frames = frames[:5000]
        vq_epochs = min(args.epochs_vqvae, 5)
    else:
        vq_epochs = args.epochs_vqvae

    N = len(frames)
    n_train = int(N * 0.85)
    X_train = torch.from_numpy(frames[:n_train]).float().to(dev)
    X_val = torch.from_numpy(frames[n_train:]).float().to(dev)

    print(f"  Frames: {frames.shape}  Train: {n_train}  Val: {N - n_train}")
    print(f"  VQ config: V={args.vocab_size}  embed_dim={args.embed_dim}")
    print(f"  Epochs: {vq_epochs}  Batch: {args.batch_size_small}")

    model_vqvae = VQVAE(
        num_embeddings=args.vocab_size,
        embed_dim=args.embed_dim,
        in_channels=3,
        commitment_cost=0.25,
    ).to(dev)
    print(f"  Params: {param_count_str(model_vqvae)}")

    optimizer = torch.optim.Adam(model_vqvae.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss()

    best_psnr = -float("inf")
    best_state_vq = None
    t_vq = time.time()

    for epoch in range(vq_epochs):
        model_vqvae.train()
        total_recon = 0.0
        total_vq = 0.0
        n_batches = 0

        perm = torch.randperm(len(X_train), device=dev)
        for i in range(0, len(perm), args.batch_size_small):
            idx = perm[i : i + args.batch_size_small]
            xb = X_train[idx]
            recon, indices, vq_loss = model_vqvae(xb)
            recon_loss = criterion(recon, xb)
            loss = recon_loss + vq_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_recon += recon_loss.item()
            total_vq += vq_loss.item()
            n_batches += 1

        # Validation
        model_vqvae.eval()
        with torch.no_grad():
            val_batch = X_val[:min(512, len(X_val))]
            val_recon, val_indices, _ = model_vqvae(val_batch)
            mse_v = criterion(val_recon, val_batch).item()
            psnr = 10.0 * np.log10(1.0 / max(mse_v, 1e-10)) if mse_v > 0 else 100.0
            usage = len(val_indices.unique()) / args.vocab_size

        avg_r = total_recon / max(n_batches, 1)
        avg_v = total_vq / max(n_batches, 1)
        print(f"  [{timestamp()}] Epoch {epoch+1:3d}/{vq_epochs}  "
              f"recon={avg_r:.6f}  vq={avg_v:.6f}  "
              f"PSNR={psnr:.1f}dB  usage={usage:.1%}")

        if psnr > best_psnr:
            best_psnr = psnr
            best_state_vq = {k: v.cpu().clone() for k, v in model_vqvae.state_dict().items()}

    vq_time = time.time() - t_vq
    print(f"\n  VQ-VAE done in {vq_time:.0f}s. Best PSNR: {best_psnr:.1f} dB")

    if best_state_vq:
        model_vqvae.load_state_dict(best_state_vq)
    model_vqvae.eval()

    # Save checkpoint
    ckpt_path = data_dir / "vqvae_checkpoint.pt"
    torch.save(model_vqvae.state_dict(), ckpt_path)
    print(f"  Checkpoint → {ckpt_path}")

    # ═══ C.2 — Encode to tokens ═══════════════════════════════════════════
    print_header("C.2 — Discrete Token Encoding")

    all_frames_t = torch.from_numpy(frames).float().to(dev)
    all_indices = []
    with torch.no_grad():
        for i in range(0, len(all_frames_t), args.batch_size_small * 4):
            xb = all_frames_t[i : i + args.batch_size_small * 4]
            _, tok_idx, _ = model_vqvae(xb)
            all_indices.append(tok_idx.cpu())
    all_indices = torch.cat(all_indices, dim=0).numpy()  # (N, 16, 16)

    tokens = all_indices[:-1]
    next_tokens = all_indices[1:]
    # Build actions: if we have them, use; otherwise zeros
    actions_path = data_dir / "actions.npy"
    if actions_path.exists():
        actions = np.load(actions_path).astype(np.int64)[:len(tokens)]
    else:
        actions = np.zeros(len(tokens), dtype=np.int64)

    np.save(data_dir / "tokens.npy", tokens)
    np.save(data_dir / "next_tokens.npy", next_tokens)
    np.save(data_dir / "actions.npy", actions)
    print(f"  tokens: {tokens.shape}  next_tokens: {next_tokens.shape}  actions: {actions.shape}")
    usage_post = len(np.unique(tokens)) / args.vocab_size
    print(f"  Codebook usage (encoded): {usage_post:.1%}")

    # ═══ C.3 — DiscreteRescor training ════════════════════════════════════
    print_header("C.3 — DiscreteRescor Training")

    from wmca.model_registry import train_discrete_rescor

    seeds = [42] if args.quick else [42, 43, 44]
    epochs_dr = args.epochs_disc // 5 if args.quick else args.epochs_disc

    n_val = int(len(tokens) * 0.15)
    t_tr, t_v = tokens[:-n_val], tokens[-n_val:]
    n_tr, n_v = next_tokens[:-n_val], next_tokens[-n_val:]
    a_tr, a_v = actions[:-n_val], actions[-n_val:]

    n_actions = int(actions.max()) + 1
    print(f"  N_actions: {n_actions}  Train: {len(t_tr)}  Val: {len(t_v)}")

    dr_models = {}
    for seed in seeds:
        print(f"\n  [{timestamp()}] Seed {seed} — discrete_rescor V={args.vocab_size}")

        from wmca.modules.discrete_rescor import DiscreteRescor
        dr = DiscreteRescor(
            vocab_size=args.vocab_size,
            n_actions=n_actions,
            embed_dim=args.embed_dim,
            hidden_ch=8 if args.quick else 16,
            cml_K=32,
            seed=seed,
            use_sigmoid=False,
        )
        print(f"    Params: {param_count_str(dr)}")

        t_dr = time.time()
        trained_dr = train_discrete_rescor(
            dr, t_tr, n_tr, a_tr,
            tokens_val=t_v, next_tokens_val=n_v, actions_val=a_v,
            epochs=epochs_dr,
            batch_size=args.batch_size_small,
            lr=args.lr,
            device=dev,
        )
        dr_models[seed] = trained_dr
        print(f"    Done in {time.time() - t_dr:.0f}s")

    # ═══ C.4 — Discrete Token Rollout Probe ═══════════════════════════════
    print_header("C.4 — Discrete Token Rollout Probe")

    rollout_results = {}
    test_start = int(len(tokens) * 0.7)
    t_test = tokens[test_start:]
    n_test = next_tokens[test_start:]
    a_test = actions[test_start:]

    for seed in seeds:
        model_dr = dr_models[seed].to(torch.device("cpu")).eval()
        per_traj = []

        # Use first 20 sequences as test rollouts (each up to 50 steps)
        n_seqs = min(20, len(t_test) // 50)
        for seq_idx in range(n_seqs):
            start = seq_idx * 50
            h_max = min(50, len(t_test) - start - 1)
            if h_max < 10:
                continue

            init = t_test[start]  # (H, W)
            acts = a_test[start : start + h_max]
            gts = n_test[start : start + h_max]

            current = torch.from_numpy(init).long().unsqueeze(0)  # (1, H, W)
            accs = []
            with torch.no_grad():
                for h in range(h_max):
                    a_t = torch.tensor([acts[h]], dtype=torch.long)
                    logits = model_dr(current, a_t)
                    pred = logits.argmax(dim=1).squeeze(0)  # (H, W)
                    gt = torch.from_numpy(gts[h]).long()
                    acc = (pred == gt).float().mean().item()
                    accs.append(acc)
                    current = pred.unsqueeze(0)  # feed back

            per_traj.append({
                "seq_idx": seq_idx,
                "steps": h_max,
                "per_step_acc": accs,
                "H=15_acc": accs[14] if len(accs) > 14 else None,
                "H=50_acc": accs[-1] if len(accs) >= 50 else None,
            })

        h15_acc = np.nanmean([t["H=15_acc"] for t in per_traj if t["H=15_acc"] is not None])
        h50_acc = np.nanmean([t["H=50_acc"] for t in per_traj if t["H=50_acc"] is not None])
        step1_acc = np.nanmean([t["per_step_acc"][0] for t in per_traj if t["per_step_acc"]])

        rollout_results[f"seed_{seed}"] = {
            "model": "discrete_rescor",
            "seed": seed,
            "n_sequences": len(per_traj),
            "step1_acc": float(step1_acc),
            "H=15_acc": float(h15_acc),
            "H=50_acc": float(h50_acc),
        }
        print(f"  Seed {seed}: step1_acc={step1_acc:.3f}  H=15_acc={h15_acc:.3f}  H=50_acc={h50_acc:.3f}")

    # Save
    out_path = Path("experiments/results/discrete_rollout_probe_mps.json")
    with open(out_path, "w") as f:
        json.dump({
            "config": {
                "vocab_size": args.vocab_size,
                "embed_dim": args.embed_dim,
                "seeds": seeds,
                "epochs": epochs_dr,
                "device": dev.type,
                "vq_psnr": best_psnr,
            },
            "per_seed": rollout_results,
        }, f, indent=2)
    print(f"\n  Results → {out_path}")

    del X_train, X_val, all_frames_t
    import gc; gc.collect()


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="WMCA Experiments — M4 MPS runner (3GB cap)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--path", default="AC", choices=["A", "C", "AC"],
                        help="Which path(s) to run: A (Atari), C (VQ-VAE+DiscreteRescor), AC (both)")
    parser.add_argument("--quick", action="store_true",
                        help="Smoke-test mode: fewer frames/seeds/epochs")
    parser.add_argument("--game", default="pong", choices=["pong", "breakout"],
                        help="Atari game for Path A")
    parser.add_argument("--n-frames", type=int, default=200000,
                        help="Atari frames to collect")
    parser.add_argument("--epochs-atari", type=int, default=100,
                        help="Epochs for Atari rescor training")
    parser.add_argument("--epochs-vqvae", type=int, default=50,
                        help="Epochs for VQ-VAE training")
    parser.add_argument("--epochs-disc", type=int, default=100,
                        help="Epochs for DiscreteRescor training")
    parser.add_argument("--vocab-size", type=int, default=512,
                        help="VQ-VAE codebook size")
    parser.add_argument("--embed-dim", type=int, default=64,
                        help="VQ-VAE / DiscreteRescor embedding dimension")
    parser.add_argument("--lr", type=float, default=1.4e-3,
                        help="Learning rate (sqrt-rule adjusted)")
    parser.add_argument("--batch-size-small", type=int, default=32,
                        help="Batch size for MPS (smaller for 3GB cap)")
    parser.add_argument("--n-traj", type=int, default=20,
                        help="Number of test trajectories for rollout")
    args = parser.parse_args()

    total_t0 = time.time()

    print("═" * 70)
    print("  WMCA Experiments — M4 MPS Runner")
    print(f"  Started: {datetime.now().isoformat()}")
    print(f"  Paths: {args.path}  Quick: {args.quick}")
    print(f"  MPS ratio: {_MPS_RATIO}  (≈ {_MPS_RATIO * 32:.0f}GB cap on 32GB unified)")
    print("═" * 70)

    try:
        if "A" in args.path:
            run_atari_stage(args)
            import gc; gc.collect()

        if "C" in args.path:
            run_vqvae_stage(args)
            import gc; gc.collect()

    except KeyboardInterrupt:
        print("\n\n  Interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n  ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    total_time = time.time() - total_t0
    print_header(f"Done! Total wall time: {total_time/60:.1f} min ({total_time:.0f}s)")
    print(f"  Results in experiments/results/")


if __name__ == "__main__":
    main()
