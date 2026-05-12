"""Action-conditioned autoregressive rollout probe for Atari latents.

Sibling of dreamerv3_scaffolding/rollout_stability_probe_crafter.py.
Trains rescor on Atari autoencoder latents with action conditioning,
then measures autoregressive rollout stability at H={15, 50, 100}.

Usage:
    python dreamerv3_scaffolding/rollout_stability_probe_atari.py \
        --game pong --model rescor_rens --seeds 42 43 44 \
        --epochs 100 --device cuda
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.wmca.atari_real import AtariLatentBenchmark
from src.wmca.model_registry import create_model, train_model


def make_action_field(action: int, n_actions: int, H: int, W: int) -> np.ndarray:
    """Create a single-channel action field: (1, H, W) with uniform value (a+1)/n_actions."""
    value = (action + 1) / n_actions
    return np.full((1, H, W), value, dtype=np.float32)


def rollout_mse(
    model: torch.nn.Module,
    trajectory: dict,
    horizon: int,
    device: torch.device,
    n_actions: int = 3,
) -> dict:
    """Autoregressive rollout with closed-loop state, open-loop actions.

    Args:
        trajectory: dict with "frames" (T+1, 1, H, W) and "actions" (T,) keys.
        horizon: rollout length.
        device: torch device.
        n_actions: number of discrete actions.

    Returns:
        dict with per_step_mse, per_step_cos_div, summary stats.
    """
    model.eval()
    traj_frames = trajectory["frames"]  # (T+1, 1, H, W)
    traj_actions = trajectory["actions"]  # (T,)
    T = min(len(traj_actions), len(traj_frames) - 1, horizon)

    current_frame = traj_frames[0].copy()  # (1, H, W)
    H_grid, W_grid = current_frame.shape[1:3]

    per_step_mse = []
    per_step_cos_div = []

    with torch.no_grad():
        for t in range(T):
            action = int(traj_actions[t])
            gt_next = traj_frames[t + 1]  # (1, H, W)

            # Build input: [current_frame (1ch), action_field (1ch)]
            act_field = make_action_field(action, n_actions, H_grid, W_grid)
            x = np.concatenate([current_frame, act_field], axis=0)  # (2, H, W)
            x_t = torch.from_numpy(x).float().unsqueeze(0).to(device)

            pred = model(x_t).squeeze(0).cpu().numpy()  # (1, H, W)

            # Step MSE
            mse = float(np.mean((pred - gt_next) ** 2))
            per_step_mse.append(mse)

            # Cosine divergence
            pred_flat = pred.reshape(-1)
            gt_flat = gt_next.reshape(-1)
            cos_sim = float(np.dot(pred_flat, gt_flat) / (
                np.linalg.norm(pred_flat) * np.linalg.norm(gt_flat) + 1e-12
            ))
            per_step_cos_div.append(1.0 - cos_sim)

            # Advance state: use clamped prediction as next input
            current_frame = np.clip(pred, 0.0, 1.0).astype(np.float32)

    step1_mse = per_step_mse[0] if per_step_mse else float("nan")
    abs_mse = {}
    ratios = {}
    cos_div_at = {}
    for h in [14, 49, 99]:
        if h < len(per_step_mse):
            abs_mse[f"H={h+1}"] = per_step_mse[h]
            ratios[f"H={h+1}"] = per_step_mse[h] / step1_mse if step1_mse > 0 else float("nan")
            cos_div_at[f"H={h+1}"] = per_step_cos_div[h]

    return {
        "per_step_mse": per_step_mse,
        "per_step_cos_div": per_step_cos_div,
        "step1_mse": step1_mse,
        "abs_mse": abs_mse,
        "ratios": ratios,
        "cos_div_at": cos_div_at,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="pong", choices=["pong", "breakout"])
    parser.add_argument("--model", default="rescor_rens",
                        choices=["rescor_rens", "rescor_mamba_rand"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1.4e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--output", default=None,
                        help="Output JSON path (default: experiments/results/atari_rollout_probe.json)")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Load data via AtariLatentBenchmark
    benchmark = AtariLatentBenchmark(game=args.game, device=args.device)
    X_train, Y_train = benchmark.get_training_data(val_split=0.15)
    X_val, Y_val = benchmark.get_validation_data(val_split=0.15)

    # Determine grid size from data
    _, _, H, W = X_train.shape  # (N, 2, H, W)
    in_ch, out_ch = 2, 1
    n_actions = benchmark.n_actions

    print(f"Atari Rollout Probe: game={args.game}, model={args.model}, grid={H}x{W}")
    print(f"  in_ch={in_ch}, out_ch={out_ch}, n_train={len(X_train)}, n_val={len(X_val)}")

    # Train + rollout per seed
    results = {}
    all_rollouts = []

    for seed in args.seeds:
        print(f"\n--- Seed {seed} ---")
        # Instantiate model based on --model flag
        from src.wmca.modules.hybrid import ResidualCorrectionWM

        if args.model == "rescor_mamba_rand":
            # rescor_mamba_rand needs multi-frame input; not yet wired for action-conditioned rollout.
            # Fall back to rescor_rens for now and warn.
            print(f"  WARNING: rescor_mamba_rand not yet supported for action-conditioned rollout.")
            print(f"  Falling back to rescor_rens.")

        model = ResidualCorrectionWM(
            in_channels=in_ch,
            out_channels=out_ch,
            hidden_ch=16,
            cml_gate="multi_r_uniform",
            cml_K=32,
            seed=seed,
            use_sigmoid=True,
        )

        trained = train_model(
            model,
            X_train, Y_train,
            X_val, Y_val,
            loss_type="mse",
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device,
            compile=args.compile,
            bf16=args.bf16,
        )

        # Get test trajectories
        test_trajs = benchmark.get_test_trajectories(n_trajectories=20)

        seed_rollouts = []
        for traj_idx, traj in enumerate(test_trajs):
            rollout = rollout_mse(
                trained, traj, horizon=100,
                device=torch.device("cpu"), n_actions=n_actions,
            )
            rollout["traj_idx"] = traj_idx
            seed_rollouts.append(rollout)

        # Median across trajectories
        h15_mse = np.nanmedian([r["abs_mse"].get("H=15", float("nan")) for r in seed_rollouts])
        h50_mse = np.nanmedian([r["abs_mse"].get("H=50", float("nan")) for r in seed_rollouts])
        h100_mse = np.nanmedian([r["abs_mse"].get("H=100", float("nan")) for r in seed_rollouts])
        h15_ratio = np.nanmedian([r["ratios"].get("H=15", float("nan")) for r in seed_rollouts])
        h100_cos = np.nanmedian([r["cos_div_at"].get("H=100", float("nan")) for r in seed_rollouts])

        results[f"seed_{seed}"] = {
            "model": args.model,
            "game": args.game,
            "seed": seed,
            "n_trajectories": len(seed_rollouts),
            "step1_mse_median": float(np.nanmedian([r["step1_mse"] for r in seed_rollouts])),
            "H=15_abs_mse": float(h15_mse),
            "H=50_abs_mse": float(h50_mse),
            "H=100_abs_mse": float(h100_mse),
            "H=15_ratio": float(h15_ratio),
            "H=100_cos_div": float(h100_cos),
        }
        all_rollouts.extend(seed_rollouts)

        print(f"  Step1 MSE median: {results[f'seed_{seed}']['step1_mse_median']:.2e}")
        print(f"  H=15 ratio: {h15_ratio:.2f}  H=100 cos_div: {h100_cos:.4f}")

    # Save
    out_path = Path(args.output) if args.output else Path("experiments/results/atari_rollout_probe.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "config": {
            "game": args.game,
            "model": args.model,
            "seeds": args.seeds,
            "epochs": args.epochs,
            "grid": f"{H}x{W}",
        },
        "per_seed": results,
        "rollouts": all_rollouts,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
