"""Rollout-stability probe for rescor_rens K=32 on Crafter-latent dynamics.

Task #31. Sibling of `rollout_stability_probe.py` (Task #27). Same rescor_rens
K=32 recipe, same decision gate, but the target distribution is encoded
Crafter frames (via the frozen autoencoder at
`experiments/crafter_data/frame_encoder.pt`) rather than synthetic heat/gs/ks.

Decision gate (per `crafter_probe_plan.md` §5):
  * median_seed(MSE_15 / MSE_1) < 2.0   -> DROP Dreamer posterior
  * 2.0 .. 5.0                          -> KEEP posterior (safety net)
  * > 5.0                               -> KEEP posterior AND promote rescor_mamba

Protocol: 3 seeds x 100 epochs x rescor_rens K=32. Rollout input at step t
is reassembled as [pred_frame, action_field(traj.actions[t])] (2 channels),
matching training. Latents are NOT clamped (unlike the synthetic probe —
encoded features are unbounded).

Usage:
    PYTHONPATH=src PYTHONUNBUFFERED=1 uv run --with scikit-learn,scipy \\
        python dreamerv3_scaffolding/rollout_stability_probe_crafter.py
"""
from __future__ import annotations

import gc
import json
import os
import time
from pathlib import Path
from statistics import median

import numpy as np
import torch

from wmca.crafter_real import generate_crafter_real_trajectories
from wmca.model_registry import create_model, train_model


# ---- Defaults (can be overridden via env vars for smoke testing) ------------
SEEDS = [int(s) for s in os.environ.get("SEEDS", "42,43,44").split(",")]
K = int(os.environ.get("K", "32"))
MODEL = os.environ.get("MODEL", "rescor_rens")
EPOCHS = int(os.environ.get("EPOCHS", "100"))
GRID = int(os.environ.get("GRID", "16"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "64"))
LR = float(os.environ.get("LR", "1e-3"))

# Data sizing
N_FRAMES = int(os.environ.get("N_FRAMES", "100000"))
MIN_TRAJ_LEN = int(os.environ.get("MIN_TRAJ_LEN", "105"))

# Rollout budget
N_ROLLOUT_TRAJS = int(os.environ.get("N_ROLLOUT_TRAJS", "20"))
N_TEST_TRAJECTORIES = int(os.environ.get("N_TEST_TRAJECTORIES", "0"))  # 0 -> all qualifying

HORIZONS = [int(h) for h in os.environ.get("HORIZONS", "15,50,100").split(",")]

# Output path override (mostly useful for smoke)
OUT_PATH = Path(
    os.environ.get(
        "OUT_PATH",
        "experiments/results/rollout_stability_probe_crafter.json",
    )
)

_CRAFTER_N_ACTIONS = 17


def cosine_div(pred: np.ndarray, true: np.ndarray) -> float:
    p = pred.reshape(-1)
    t = true.reshape(-1)
    num = float((p * t).sum())
    den = float(np.linalg.norm(p) * np.linalg.norm(t)) + 1e-8
    return 1.0 - num / den


def build_action_field(action: int, grid: int, device: torch.device) -> torch.Tensor:
    """(1, 1, grid, grid) float32 plane filled with (a+1)/17."""
    val = float((action + 1.0) / _CRAFTER_N_ACTIONS)
    return torch.full((1, 1, grid, grid), val, dtype=torch.float32, device=device)


def train_and_probe(seed: int, data):
    t0 = time.time()
    meta = data.meta
    model = create_model(
        MODEL,
        in_channels=meta["in_channels"],
        out_channels=meta["out_channels"],
        grid_size=GRID,
        seed=seed,
        cml_K=K,
    )
    pc = model.param_count() if hasattr(model, "param_count") else sum(
        p.numel() for p in model.parameters()
    )
    model = train_model(
        model,
        data.X_train, data.Y_train,
        X_val=data.X_val, Y_val=data.Y_val,
        loss_type=meta["loss_type"],
        epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR,
    )
    train_time = time.time() - t0

    # Select trajectories to roll.
    trajectories = meta["test_trajectories"]
    effective_min_len = MIN_TRAJ_LEN
    max_h = max(HORIZONS)

    # Gather qualifying trajectories. Apply automatic horizon fallback per plan
    # §6 to be defensive, even though the measured data has 80 trajs >= 105.
    qualifying = [t for t in trajectories if t.actions.numel() >= max_h]
    if len(qualifying) < N_ROLLOUT_TRAJS:
        # Try 50 then 15
        for fallback_h in [50, 15]:
            qualifying = [t for t in trajectories if t.actions.numel() >= fallback_h]
            if len(qualifying) >= N_ROLLOUT_TRAJS:
                max_h = fallback_h
                print(
                    f"  [warn] only {len([t for t in trajectories if t.actions.numel() >= max(HORIZONS)])} "
                    f"trajs reach H={max(HORIZONS)}; falling back to H={fallback_h}."
                )
                break
        else:
            # Not enough even at H=15: use whatever we have
            qualifying = [t for t in trajectories if t.actions.numel() >= 15]
            max_h = 15
            print(f"  [warn] fewer than {N_ROLLOUT_TRAJS} trajs reach even H=15.")

    n_use = min(N_ROLLOUT_TRAJS, len(qualifying))
    use_trajs = qualifying[:n_use]

    # Per-step accumulators with per-step denominator (trajs that reached step s).
    per_step_mse_sum = np.zeros(max_h, dtype=np.float64)
    per_step_cos_sum = np.zeros(max_h, dtype=np.float64)
    per_step_count = np.zeros(max_h, dtype=np.int64)

    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        for traj in use_trajs:
            T = int(traj.actions.numel())
            x_frame = traj.encoded_frames[0:1].to(device)   # (1, 1, 16, 16)
            for s in range(max_h):
                if s >= T:
                    break
                a = int(traj.actions[s].item())
                af = build_action_field(a, GRID, device)      # (1, 1, 16, 16)
                x = torch.cat([x_frame, af], dim=1)          # (1, 2, 16, 16)
                pred = model(x)                               # (1, 1, 16, 16)
                gt = traj.encoded_frames[s + 1 : s + 2].to(device)  # (1, 1, 16, 16)
                mse = float(((pred - gt) ** 2).mean().item())
                per_step_mse_sum[s] += mse
                per_step_cos_sum[s] += cosine_div(
                    pred.detach().cpu().numpy(), gt.detach().cpu().numpy()
                )
                per_step_count[s] += 1
                # NO clamp: encoded latents are unbounded
                x_frame = pred

    # Mean per step (safe divide).
    per_step_mse = np.where(
        per_step_count > 0, per_step_mse_sum / np.maximum(per_step_count, 1), np.nan
    )
    per_step_cos = np.where(
        per_step_count > 0, per_step_cos_sum / np.maximum(per_step_count, 1), np.nan
    )

    horizons_out = {}
    mse_step1 = float(per_step_mse[0]) if per_step_count[0] > 0 else float("nan")
    for h in HORIZONS:
        if h > max_h or h - 1 >= len(per_step_mse) or per_step_count[h - 1] == 0:
            horizons_out[f"H={h}"] = {
                "mse_final": None,
                "cosine_div_final": None,
                "mse_ratio_vs_step1": None,
                "n_trajs_reached": int(per_step_count[h - 1]) if h - 1 < len(per_step_count) else 0,
            }
            continue
        mse_h = float(per_step_mse[h - 1])
        horizons_out[f"H={h}"] = {
            "mse_final": mse_h,
            "cosine_div_final": float(per_step_cos[h - 1]),
            "mse_ratio_vs_step1": (mse_h / mse_step1) if (mse_step1 and mse_step1 > 0) else float("inf"),
            "n_trajs_reached": int(per_step_count[h - 1]),
        }

    return {
        "seed": seed,
        "params": pc,
        "train_time_s": train_time,
        "mse_per_step": [None if np.isnan(v) else float(v) for v in per_step_mse.tolist()],
        "cosine_div_per_step": [None if np.isnan(v) else float(v) for v in per_step_cos.tolist()],
        "n_trajectories_per_step": per_step_count.tolist(),
        "horizons": horizons_out,
        "n_rollout_trajectories": n_use,
        "effective_max_h": int(max_h),
        "effective_min_traj_len": int(effective_min_len),
    }


def _load_data():
    kwargs = dict(
        grid_size=GRID,
        n_frames=N_FRAMES,
        seed=42,
        device="cpu",
        min_traj_len=MIN_TRAJ_LEN,
    )
    if N_TEST_TRAJECTORIES > 0:
        kwargs["max_test_trajectories"] = N_TEST_TRAJECTORIES
    print(f"[data] generate_crafter_real_trajectories kwargs={kwargs}")
    data = generate_crafter_real_trajectories(**kwargs)
    print(
        f"[data] n_pairs={data.meta['n_frames']}  "
        f"n_episodes={data.meta['n_episodes_total']}  "
        f"n_test_trajectories={data.meta['n_test_trajectories']}"
    )
    return data


def main():
    out_path = OUT_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume-from-JSON: same pattern as rollout_stability_probe.py.
    results: dict[str, dict] = {}
    if out_path.exists():
        try:
            with open(out_path, "r") as f:
                prior = json.load(f)
            src = prior.get("per_cell", prior)
            n_loaded = 0
            for sk, cell in src.items():
                if cell and cell.get("mse_per_step"):
                    results[sk] = cell
                    n_loaded += 1
            print(f"[resume] loaded {n_loaded} completed cells from {out_path}")
        except Exception as e:
            print(f"[resume] failed ({e}); starting fresh")

    need_data = any(str(s) not in results for s in SEEDS)
    data = _load_data() if need_data else None

    for seed in SEEDS:
        sk = str(seed)
        if results.get(sk):
            r = results[sk]
            h15 = r.get("horizons", {}).get("H=15", {})
            ratio = h15.get("mse_ratio_vs_step1")
            print(
                f"[seed={seed}] resumed  H15_ratio="
                f"{ratio if ratio is None else f'{ratio:.2f}'}"
            )
            continue
        print("=" * 78)
        print(f"seed={seed}  model={MODEL}  K={K}  epochs={EPOCHS}  grid={GRID}")
        print("=" * 78)
        try:
            r = train_and_probe(seed, data)
            results[sk] = r
            h = r["horizons"]
            print(f"  trained in {r['train_time_s']:.0f}s  params={r['params']}")
            step1 = r["mse_per_step"][0]
            print(f"  step 1 MSE = {step1 if step1 is None else f'{step1:.4e}'}")
            for hh in HORIZONS:
                cell = h[f"H={hh}"]
                if cell["mse_final"] is None:
                    print(f"  H={hh:3d}  (not reached; {cell['n_trajs_reached']} trajs)")
                    continue
                print(
                    f"  H={hh:3d}  MSE={cell['mse_final']:.4e}  "
                    f"ratio_vs_step1={cell['mse_ratio_vs_step1']:.2f}  "
                    f"cos_div={cell['cosine_div_final']:.4f}  "
                    f"(n={cell['n_trajs_reached']})"
                )
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            results[sk] = {"error": str(e), "seed": seed}

        with open(out_path, "w") as f:
            json.dump(
                {
                    "per_cell": results,
                    "protocol": {
                        "model": MODEL, "K": K, "epochs": EPOCHS,
                        "grid": GRID, "seeds": SEEDS,
                        "horizons": HORIZONS,
                        "n_rollout_trajectories": N_ROLLOUT_TRAJS,
                        "min_traj_len": MIN_TRAJ_LEN,
                        "batch_size": BATCH_SIZE,
                        "lr": LR,
                        "n_frames": N_FRAMES,
                        "benchmark": "crafter_real_traj",
                    },
                },
                f,
                indent=2,
            )
        gc.collect()

    # Summary and decision gate across seeds.
    print()
    print("=" * 90)
    print("SUMMARY — rescor_rens K=32 on Crafter-latent (autoregressive rollout)")
    print("=" * 90)
    hdr = (
        f"{'H':>3s}  {'MSE median':>12s}  {'ratio median':>14s}  "
        f"{'cos_div med':>12s}  {'n_seeds':>7s}"
    )
    print(hdr)
    print("-" * len(hdr))
    summary = {}
    for hh in HORIZONS:
        ratios, mses, cosds = [], [], []
        for sk, cell in results.items():
            if "horizons" not in cell:
                continue
            c = cell["horizons"].get(f"H={hh}")
            if not c or c.get("mse_final") is None:
                continue
            ratios.append(c["mse_ratio_vs_step1"])
            mses.append(c["mse_final"])
            cosds.append(c["cosine_div_final"])
        if ratios:
            s = {
                "mse_median": median(mses),
                "ratio_median": median(ratios),
                "cos_div_median": median(cosds),
                "n_seeds": len(ratios),
            }
            summary[f"H={hh}"] = s
            print(
                f"{hh:>3d}  {s['mse_median']:>12.4e}  "
                f"{s['ratio_median']:>14.2f}  {s['cos_div_median']:>12.4f}  "
                f"{s['n_seeds']:>7d}"
            )

    # Decision gate: rescor_rens K=32 "stable on Crafter-latent" iff
    # median(MSE_15 / MSE_1) < 2.0 (with 2..5 = marginal, >5 = unstable).
    print()
    print("=" * 90)
    print("DECISION GATE — median(MSE_15 / MSE_1)")
    print("=" * 90)
    s15 = summary.get("H=15", {})
    r15 = s15.get("ratio_median")
    if r15 is None:
        verdict = "UNKNOWN (no H=15 cell completed)"
        bucket = "unknown"
    elif r15 < 2.0:
        verdict = "STABLE — DROP Dreamer posterior; resume rescor-only DreamerV3 fork"
        bucket = "stable"
    elif r15 < 5.0:
        verdict = (
            "MARGINAL — KEEP posterior as safety net; de-prioritise rescor_mamba"
        )
        bucket = "marginal"
    else:
        verdict = (
            "UNSTABLE — KEEP posterior AND promote rescor_mamba ablation"
        )
        bucket = "unstable"
    gate = {
        "ratio_median_H15": r15,
        "bucket": bucket,
        "verdict": verdict,
    }
    print(
        f"  median_ratio_H15={r15 if r15 is None else f'{r15:.2f}':>6}   "
        f"-> {verdict}"
    )

    with open(out_path, "w") as f:
        json.dump(
            {
                "per_cell": results,
                "summary_median": summary,
                "decision_gate": gate,
                "protocol": {
                    "model": MODEL, "K": K, "epochs": EPOCHS,
                    "grid": GRID, "seeds": SEEDS,
                    "horizons": HORIZONS,
                    "n_rollout_trajectories": N_ROLLOUT_TRAJS,
                    "min_traj_len": MIN_TRAJ_LEN,
                    "batch_size": BATCH_SIZE,
                    "lr": LR,
                    "n_frames": N_FRAMES,
                    "benchmark": "crafter_real_traj",
                },
            },
            f,
            indent=2,
        )
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
