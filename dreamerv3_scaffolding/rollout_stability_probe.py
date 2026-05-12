"""Rollout-stability probe for rescor_rens K=32.

Task #27 / next_steps.md §1 / dreamerv3_fork_plan.md M2.

Trains rescor_rens K=32 on heat / gs / ks at grid_size=16 with
n_steps=105 so test trajectories are long enough to roll for H=100.
Then rolls each model autoregressively over many test trajectories,
averaging per-step MSE against ground truth.

Decision gate (per dreamerv3_fork_plan.md §4):
  * stable to H=15 (MSE_15 < 2 * MSE_1)  -> DROP Dreamer posterior.
  * diverges                              -> KEEP posterior.

Protocol: 3 benchmarks x 3 seeds x 100 epochs. Pairs (X,Y) from the
test split are contiguous within each trajectory, so trajectory i's
first frame is X_test[i*T] and its ground-truth successors are
Y_test[i*T + 0 .. T-1].

Usage:
    PYTHONPATH=src PYTHONUNBUFFERED=1 uv run --with scikit-learn,scipy \\
        python dreamerv3_scaffolding/rollout_stability_probe.py
"""
from __future__ import annotations

import gc
import json
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

SEEDS = [42, 43, 44]
K = 32
MODEL = "rescor_rens"
EPOCHS = 100
N_STEPS = 105          # long enough for H=100
N_TRAJECTORIES = 200   # keeps training-pair count near phase1 budget
GRID = 16
N_ROLLOUT_TRAJS = 20   # avg per-step MSE over this many test trajectories
HORIZONS = [15, 50, 100]


def cosine_div(pred: np.ndarray, true: np.ndarray) -> float:
    p = pred.reshape(-1)
    t = true.reshape(-1)
    num = float((p * t).sum())
    den = float(np.linalg.norm(p) * np.linalg.norm(t)) + 1e-8
    return 1.0 - num / den


def train_and_probe(bench_name, gen_fn, seed):
    t0 = time.time()
    data = gen_fn(grid_size=GRID, seed=seed, n_steps=N_STEPS,
                  n_trajectories=N_TRAJECTORIES)
    meta = data.meta
    model = create_model(
        MODEL,
        in_channels=meta["in_channels"],
        out_channels=meta["out_channels"],
        grid_size=GRID,
        seed=seed,
        cml_K=K,
    )
    pc = model.param_count()
    model = train_model(
        model, data.X_train, data.Y_train,
        X_val=data.X_val, Y_val=data.Y_val,
        loss_type=meta["loss_type"],
        epochs=EPOCHS, batch_size=64, lr=1e-3,
    )
    train_time = time.time() - t0

    # Pairs come out of _make_pairs() as (N_traj * T, ...) with trajectory
    # blocks contiguous. T = N_STEPS.
    T = N_STEPS
    X_test = data.X_test
    Y_test = data.Y_test
    n_test_traj = X_test.shape[0] // T
    n_use = min(N_ROLLOUT_TRAJS, n_test_traj)

    # Per-step MSE / cosine-divergence averaged across trajectories.
    max_h = max(HORIZONS)
    per_step_mse = np.zeros(max_h, dtype=np.float64)
    per_step_cos = np.zeros(max_h, dtype=np.float64)

    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        for ti in range(n_use):
            x = X_test[ti * T].unsqueeze(0).to(device)          # (1, C, H, W)
            for s in range(max_h):
                pred = model(x)                                  # (1, C, H, W)
                gt = Y_test[ti * T + s].unsqueeze(0).to(device)  # (1, C, H, W)
                mse = float(((pred - gt) ** 2).mean().item())
                per_step_mse[s] += mse / n_use
                p_np = pred.cpu().numpy()
                g_np = gt.cpu().numpy()
                per_step_cos[s] += cosine_div(p_np, g_np) / n_use
                x = pred.clamp(0.0, 1.0)

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
    out_path = Path("experiments/results/rollout_stability_probe.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = {bn: {} for bn in BENCHMARKS}
    # Resume from prior partial run
    if out_path.exists():
        try:
            with open(out_path, "r") as f:
                prior = json.load(f)
            src = prior.get("per_cell", prior)
            n_loaded = 0
            for bn in BENCHMARKS:
                if bn not in src:
                    continue
                for sk, cell in src[bn].items():
                    if cell and cell.get("mse_per_step"):
                        results[bn][sk] = cell
                        n_loaded += 1
            print(f"[resume] loaded {n_loaded} completed cells from {out_path}")
        except Exception as e:
            print(f"[resume] failed ({e}); starting fresh")

    for bn, gf in BENCHMARKS.items():
        for seed in SEEDS:
            sk = str(seed)
            if results[bn].get(sk):
                r = results[bn][sk]
                print(f"[{bn:4s} seed={seed}] resumed  "
                      f"H15_ratio={r['horizons']['H=15']['mse_ratio_vs_step1']:.2f}")
                continue
            print("=" * 78)
            print(f"{bn}  seed={seed}  K={K}  epochs={EPOCHS}  n_steps={N_STEPS}")
            print("=" * 78)
            try:
                r = train_and_probe(bn, gf, seed)
                results[bn][sk] = r
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
                results[bn][sk] = {"error": str(e), "seed": seed}
            with open(out_path, "w") as f:
                json.dump({"per_cell": results, "protocol": {
                    "model": MODEL, "K": K, "epochs": EPOCHS,
                    "grid": GRID, "n_steps": N_STEPS, "seeds": SEEDS,
                    "horizons": HORIZONS,
                    "n_rollout_trajectories": N_ROLLOUT_TRAJS,
                }}, f, indent=2)
            gc.collect()

    # Summary and decision gate (median across seeds per benchmark)
    print()
    print("=" * 90)
    print("SUMMARY — rescor_rens K=32 autoregressive rollout stability")
    print("=" * 90)
    hdr = f"{'bench':6s}  {'H':>3s}  {'MSE median':>12s}  {'ratio median':>14s}  {'cos_div med':>12s}"
    print(hdr)
    print("-" * len(hdr))
    summary = {}
    for bn in BENCHMARKS:
        summary[bn] = {}
        for hh in HORIZONS:
            ratios, mses, cosds = [], [], []
            for sk, cell in results[bn].items():
                if "horizons" not in cell:
                    continue
                c = cell["horizons"].get(f"H={hh}")
                if not c:
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
                summary[bn][f"H={hh}"] = s
                print(f"{bn:6s}  {hh:>3d}  {s['mse_median']:>12.4e}  "
                      f"{s['ratio_median']:>14.2f}  {s['cos_div_median']:>12.4f}")

    # Decision gate: per-benchmark "stable to H=15" = median ratio < 2
    print()
    print("=" * 90)
    print("DECISION GATE — stable to H=15 iff median(MSE_15 / MSE_1) < 2.0")
    print("=" * 90)
    gate = {}
    for bn in BENCHMARKS:
        s15 = summary[bn].get("H=15", {})
        r = s15.get("ratio_median")
        stable = (r is not None) and (r < 2.0)
        gate[bn] = {"stable_to_H15": bool(stable), "ratio_median": r}
        print(f"{bn:6s}  median_ratio_H15={r if r is None else f'{r:.2f}':>6}   "
              f"{'STABLE (drop posterior)' if stable else 'UNSTABLE (keep posterior)'}")

    agree = all(g["stable_to_H15"] for g in gate.values())
    any_stable = any(g["stable_to_H15"] for g in gate.values())
    overall = (
        "DROP posterior — all benchmarks stable"
        if agree else
        ("MIXED — keep posterior by default, note which benchmarks are stable"
         if any_stable else
         "KEEP posterior — no benchmark is stable")
    )
    print()
    print(f"Overall: {overall}")

    with open(out_path, "w") as f:
        json.dump({
            "per_cell": results,
            "summary_median": summary,
            "decision_gate": gate,
            "overall": overall,
            "protocol": {
                "model": MODEL, "K": K, "epochs": EPOCHS,
                "grid": GRID, "n_steps": N_STEPS, "seeds": SEEDS,
                "horizons": HORIZONS,
                "n_rollout_trajectories": N_ROLLOUT_TRAJS,
            },
        }, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
