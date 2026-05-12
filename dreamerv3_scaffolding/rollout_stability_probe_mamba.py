"""Rollout-stability probe for rescor_mamba variants.

Sibling of `rollout_stability_probe.py` (Task #27). Tests whether adding
per-cell Mamba temporal context (K=4 past frames) stabilizes autoregressive
rollouts on the chaotic targets where plain rescor_rens K=32 failed
(gs 17×, ks 127× MSE blowup at H=15).

Key difference from the Task #27 probe: rank-5 model input (B, K, C, H, W)
instead of rank-4. Maintains a rolling K-frame buffer seeded from K
ground-truth test frames, then advances with predictions.

Data-gen policy (per plan §6.2): generate the benchmark with context_k=1
for the probe's ground-truth trajectory buffer, and context_k=4 for training.
Two benchmark regenerations per cell, but no index-math bugs.

Pre-registered gate (same as Task #27): median MSE_15 / MSE_1 < 2 across
seeds ⇒ stable on that benchmark.

Usage:
    PYTHONPATH=src PYTHONUNBUFFERED=1 uv run --with scikit-learn,scipy \\
        python dreamerv3_scaffolding/rollout_stability_probe_mamba.py

Env overrides (smoke mode):
    SMOKE=1         → SEEDS=[42], EPOCHS=2, one bench only, 2 rollout trajs
    VARIANTS=csv    → subset of rescor_mamba{,_rand,_stat,_stat_rand}
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

from wmca.benchmarks import generate_gray_scott, generate_heat, generate_ks
from wmca.model_registry import create_model, train_model


BENCHMARKS = {
    "heat": generate_heat,
    "gs": generate_gray_scott,
    "ks": generate_ks,
}

ALL_VARIANTS = [
    "rescor_mamba",
    "rescor_mamba_rand",
    "rescor_mamba_stat",
    "rescor_mamba_stat_rand",
]

SEEDS = [42, 43, 44]
EPOCHS = 100
N_STEPS = 105
N_TRAJECTORIES = 200
GRID = 16
CONTEXT_K = 4
N_ROLLOUT_TRAJS = 20
HORIZONS = [15, 50, 100]

SMOKE = os.environ.get("SMOKE", "0") == "1"
if SMOKE:
    SEEDS = [42]
    EPOCHS = 2
    BENCHMARKS = {"heat": generate_heat}
    N_ROLLOUT_TRAJS = 2
    N_TRAJECTORIES = 40

VARIANTS = os.environ.get("VARIANTS", ",".join(ALL_VARIANTS)).split(",")


def cosine_div(pred: np.ndarray, true: np.ndarray) -> float:
    p = pred.reshape(-1)
    t = true.reshape(-1)
    num = float((p * t).sum())
    den = float(np.linalg.norm(p) * np.linalg.norm(t)) + 1e-8
    return 1.0 - num / den


def train_and_probe(variant, bench_name, gen_fn, seed):
    # --- Train on context_k=4 multi-frame pairs ---
    t0 = time.time()
    data_train = gen_fn(grid_size=GRID, seed=seed, n_steps=N_STEPS,
                        n_trajectories=N_TRAJECTORIES, context_k=CONTEXT_K)
    meta = data_train.meta
    assert meta["context_k"] == CONTEXT_K, meta
    model = create_model(
        variant,
        in_channels=meta["in_channels"],
        out_channels=meta["out_channels"],
        grid_size=GRID,
        seed=seed,
    )
    pc = model.param_count() if hasattr(model, "param_count") else {
        "trained": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "frozen": sum(p.numel() for p in model.parameters() if not p.requires_grad),
    }
    model = train_model(
        model, data_train.X_train, data_train.Y_train,
        X_val=data_train.X_val, Y_val=data_train.Y_val,
        loss_type=meta["loss_type"],
        epochs=EPOCHS, batch_size=64, lr=1e-3,
    )
    train_time = time.time() - t0

    # --- Rebuild single-frame ground-truth trajectories for the rollout ---
    data_eval = gen_fn(grid_size=GRID, seed=seed, n_steps=N_STEPS,
                       n_trajectories=N_TRAJECTORIES, context_k=1)
    T = N_STEPS
    X_test = data_eval.X_test    # (n_traj*T, C, H, W)
    Y_test = data_eval.Y_test
    n_test_traj = X_test.shape[0] // T
    n_use = min(N_ROLLOUT_TRAJS, n_test_traj)

    max_h = max(HORIZONS)
    per_step_mse = np.zeros(max_h, dtype=np.float64)
    per_step_cos = np.zeros(max_h, dtype=np.float64)

    model.eval()
    device = next(model.parameters()).device
    K = CONTEXT_K
    with torch.no_grad():
        for ti in range(n_use):
            # Seed buffer with K ground-truth frames: [X[ti*T], Y[ti*T], ..., Y[ti*T+K-2]]
            seed_frames = [X_test[ti * T]]
            for k in range(K - 1):
                seed_frames.append(Y_test[ti * T + k])
            buf = torch.stack(seed_frames, dim=0).unsqueeze(0).to(device)  # (1,K,C,H,W)

            for s in range(max_h):
                pred = model(buf)                                          # (1,C,H,W)
                gt_idx = ti * T + K - 1 + s
                if gt_idx >= Y_test.shape[0] or (gt_idx // T) != ti:
                    # Ran out of ground truth for this trajectory
                    break
                gt = Y_test[gt_idx].unsqueeze(0).to(device)
                per_step_mse[s] += float(((pred - gt) ** 2).mean().item()) / n_use
                per_step_cos[s] += cosine_div(pred.cpu().numpy(),
                                              gt.cpu().numpy()) / n_use
                pred_clamped = pred.clamp(0.0, 1.0).unsqueeze(1)            # (1,1,C,H,W)
                buf = torch.cat([buf[:, 1:], pred_clamped], dim=1)

    horizons_out = {}
    mse_step1 = float(per_step_mse[0])
    for h in HORIZONS:
        mse_h = float(per_step_mse[h - 1])
        horizons_out[f"H={h}"] = {
            "mse_final": mse_h,
            "cosine_div_final": float(per_step_cos[h - 1]),
            "mse_ratio_vs_step1": (mse_h / mse_step1) if mse_step1 > 0 else float("inf"),
        }

    return {
        "variant": variant,
        "bench": bench_name,
        "seed": seed,
        "params": pc,
        "train_time_s": train_time,
        "mse_per_step": per_step_mse.tolist(),
        "cosine_div_per_step": per_step_cos.tolist(),
        "horizons": horizons_out,
        "n_rollout_trajectories": n_use,
    }


def main():
    suffix = "_smoke" if SMOKE else ""
    out_path = Path(f"experiments/results/rollout_stability_probe_mamba{suffix}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = {v: {bn: {} for bn in BENCHMARKS} for v in VARIANTS}
    # Resume from prior partial run
    if out_path.exists():
        try:
            with open(out_path, "r") as f:
                prior = json.load(f)
            src = prior.get("per_cell", prior)
            n_loaded = 0
            for v in VARIANTS:
                if v not in src:
                    continue
                for bn in BENCHMARKS:
                    if bn not in src[v]:
                        continue
                    for sk, cell in src[v][bn].items():
                        if cell and cell.get("mse_per_step"):
                            results[v][bn][sk] = cell
                            n_loaded += 1
            print(f"[resume] loaded {n_loaded} completed cells from {out_path}")
        except Exception as e:
            print(f"[resume] failed ({e}); starting fresh")

    for variant in VARIANTS:
        for bn, gf in BENCHMARKS.items():
            for seed in SEEDS:
                sk = str(seed)
                if results[variant][bn].get(sk):
                    r = results[variant][bn][sk]
                    h15 = r["horizons"]["H=15"]["mse_ratio_vs_step1"]
                    print(f"[{variant} {bn} seed={seed}] resumed  H15_ratio={h15:.2f}")
                    continue
                print("=" * 82)
                print(f"{variant}  {bn}  seed={seed}  context_k={CONTEXT_K}  epochs={EPOCHS}")
                print("=" * 82)
                try:
                    r = train_and_probe(variant, bn, gf, seed)
                    results[variant][bn][sk] = r
                    h = r["horizons"]
                    print(f"  trained in {r['train_time_s']:.0f}s  params={r['params']}")
                    print(f"  step 1 MSE = {r['mse_per_step'][0]:.4e}")
                    for hh in HORIZONS:
                        cell = h[f"H={hh}"]
                        print(f"  H={hh:3d}  MSE={cell['mse_final']:.4e}  "
                              f"ratio_vs_step1={cell['mse_ratio_vs_step1']:.2f}  "
                              f"cos_div={cell['cosine_div_final']:.4f}")
                except Exception as e:
                    print(f"  FAILED: {type(e).__name__}: {e}")
                    import traceback; traceback.print_exc()
                    results[variant][bn][sk] = {"error": str(e), "seed": seed,
                                                "variant": variant, "bench": bn}
                with open(out_path, "w") as f:
                    json.dump({"per_cell": results, "protocol": {
                        "variants": VARIANTS, "epochs": EPOCHS, "grid": GRID,
                        "n_steps": N_STEPS, "context_k": CONTEXT_K,
                        "seeds": SEEDS, "horizons": HORIZONS,
                        "n_rollout_trajectories": N_ROLLOUT_TRAJS,
                    }}, f, indent=2)
                gc.collect()

    # Summary & decision gate (median across seeds, per variant × benchmark)
    print()
    print("=" * 100)
    print(f"SUMMARY — rescor_mamba rollout stability probe  (context_k={CONTEXT_K})")
    print("=" * 100)
    hdr = f"{'variant':24s}  {'bench':5s}  {'H':>3s}  {'MSE median':>12s}  {'ratio median':>14s}  {'cos_div med':>12s}"
    print(hdr); print("-" * len(hdr))
    summary = {v: {} for v in VARIANTS}
    for variant in VARIANTS:
        for bn in BENCHMARKS:
            summary[variant][bn] = {}
            for hh in HORIZONS:
                ratios, mses, cosds = [], [], []
                for sk, cell in results[variant][bn].items():
                    if "horizons" not in cell:
                        continue
                    c = cell["horizons"].get(f"H={hh}")
                    if not c:
                        continue
                    ratios.append(c["mse_ratio_vs_step1"])
                    mses.append(c["mse_final"])
                    cosds.append(c["cosine_div_final"])
                if ratios:
                    s = {"mse_median": median(mses),
                         "ratio_median": median(ratios),
                         "cos_div_median": median(cosds),
                         "n_seeds": len(ratios)}
                    summary[variant][bn][f"H={hh}"] = s
                    print(f"{variant:24s}  {bn:5s}  {hh:>3d}  "
                          f"{s['mse_median']:>12.4e}  {s['ratio_median']:>14.2f}  "
                          f"{s['cos_div_median']:>12.4f}")

    print()
    print("=" * 100)
    print("DECISION GATE — stable to H=15 iff median(MSE_15/MSE_1) < 2.0")
    print("=" * 100)
    gate = {}
    for variant in VARIANTS:
        gate[variant] = {}
        for bn in BENCHMARKS:
            s15 = summary[variant][bn].get("H=15", {})
            r = s15.get("ratio_median")
            stable = (r is not None) and (r < 2.0)
            gate[variant][bn] = {"stable_to_H15": bool(stable), "ratio_median": r}
            rs = f"{r:.2f}" if r is not None else "n/a"
            print(f"{variant:24s}  {bn:5s}  median_ratio_H15={rs:>6s}   "
                  f"{'STABLE' if stable else 'unstable'}")

    with open(out_path, "w") as f:
        json.dump({
            "per_cell": results,
            "summary_median": summary,
            "decision_gate": gate,
            "protocol": {
                "variants": VARIANTS, "epochs": EPOCHS, "grid": GRID,
                "n_steps": N_STEPS, "context_k": CONTEXT_K,
                "seeds": SEEDS, "horizons": HORIZONS,
                "n_rollout_trajectories": N_ROLLOUT_TRAJS,
            },
        }, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
