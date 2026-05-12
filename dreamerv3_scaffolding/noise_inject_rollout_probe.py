"""Noise-injection rollout-stability probe.

Task #32 / noise_inject_plan.md §4.

Clone of ``dreamerv3_scaffolding/rollout_stability_probe.py`` that loads
the checkpoints produced by ``experiments/noise_inject_ablation.py``
instead of retraining, and iterates over the sigma variants.

Expects ``experiments/results/noise_inject_ckpts/rens_K32_sigma{sigma}_{bench}_seed{seed}.pt``
to exist for every (sigma, bench, seed) cell.

Reports per-step MSE + ratio at H={15, 50, 100}.

Decision gate (per dreamerv3_fork_plan.md §4):
  * stable to H=15 (MSE_15 < 2 * MSE_1)  -> DROP Dreamer posterior.
  * diverges                              -> KEEP posterior.

Saves:
  - experiments/results/noise_inject_rollout_probe.json

Usage:
    PYTHONPATH=src PYTHONUNBUFFERED=1 uv run --with scikit-learn,scipy \\
        python dreamerv3_scaffolding/noise_inject_rollout_probe.py
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
from wmca.model_registry import create_model


BENCHMARKS = {
    "heat": generate_heat,
    "gs": generate_gray_scott,
    "ks": generate_ks,
}

SIGMAS = [0.0, 0.02]
SEEDS = [42, 43, 44]
K = 32
MODEL = "rescor_rens"
N_STEPS = 105
N_TRAJECTORIES = 200
GRID = 16
N_ROLLOUT_TRAJS = 20
HORIZONS = [15, 50, 100]

CKPT_DIR = Path("experiments/results/noise_inject_ckpts")


def cosine_div(pred: np.ndarray, true: np.ndarray) -> float:
    p = pred.reshape(-1)
    t = true.reshape(-1)
    num = float((p * t).sum())
    den = float(np.linalg.norm(p) * np.linalg.norm(t)) + 1e-8
    return 1.0 - num / den


def ckpt_path_for(sigma, bench_name, seed):
    return CKPT_DIR / f"rens_K32_sigma{sigma}_{bench_name}_seed{seed}.pt"


def probe_one(sigma, bench_name, gen_fn, seed, max_h: int | None = None):
    """Load checkpoint, regen dataset, roll autoregressively, report stats."""
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

    ckpt = ckpt_path_for(sigma, bench_name, seed)
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(state)

    T = N_STEPS
    X_test = data.X_test
    Y_test = data.Y_test
    n_test_traj = X_test.shape[0] // T
    n_use = min(N_ROLLOUT_TRAJS, n_test_traj)

    probe_max_h = max_h if max_h is not None else max(HORIZONS)
    per_step_mse = np.zeros(probe_max_h, dtype=np.float64)
    per_step_cos = np.zeros(probe_max_h, dtype=np.float64)

    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        for ti in range(n_use):
            x = X_test[ti * T].unsqueeze(0).to(device)
            for s in range(probe_max_h):
                pred = model(x)
                gt = Y_test[ti * T + s].unsqueeze(0).to(device)
                mse = float(((pred - gt) ** 2).mean().item())
                per_step_mse[s] += mse / n_use
                p_np = pred.cpu().numpy()
                g_np = gt.cpu().numpy()
                per_step_cos[s] += cosine_div(p_np, g_np) / n_use
                x = pred.clamp(0.0, 1.0)

    horizons_out = {}
    mse_step1 = float(per_step_mse[0])
    for h in HORIZONS:
        if h > probe_max_h:
            continue
        mse_h = float(per_step_mse[h - 1])
        horizons_out[f"H={h}"] = {
            "mse_final": mse_h,
            "cosine_div_final": float(per_step_cos[h - 1]),
            "mse_ratio_vs_step1": (mse_h / mse_step1) if mse_step1 > 0 else float("inf"),
        }

    return {
        "sigma": sigma,
        "bench": bench_name,
        "seed": seed,
        "params": pc,
        "probe_time_s": time.time() - t0,
        "mse_per_step": per_step_mse.tolist(),
        "cosine_div_per_step": per_step_cos.tolist(),
        "horizons": horizons_out,
        "n_rollout_trajectories": n_use,
        "ckpt": str(ckpt),
        "max_h": probe_max_h,
    }


def sigma_key(sigma: float) -> str:
    return f"sigma={sigma}"


def main(smoke: bool = False, smoke_max_h: int = 15):
    out_path = Path("experiments/results/noise_inject_rollout_probe.json")
    if smoke:
        out_path = Path("experiments/results/noise_inject_rollout_probe_smoke.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # results[sigma_key][bench][seed_str] = cell
    results = {sigma_key(s): {bn: {} for bn in BENCHMARKS} for s in SIGMAS}

    # In smoke mode, restrict to available checkpoints.
    sigmas_to_run = SIGMAS
    seeds_to_run = SEEDS
    benches_to_run = dict(BENCHMARKS)
    if smoke:
        # Only sigma=0.02, heat, seed=42 (matching the ablation smoke test).
        sigmas_to_run = [0.02]
        seeds_to_run = [42]
        benches_to_run = {"heat": generate_heat}

    if out_path.exists():
        try:
            with open(out_path, "r") as f:
                prior = json.load(f)
            src = prior.get("per_cell", prior)
            n_loaded = 0
            for sk in results:
                if sk not in src:
                    continue
                for bn in BENCHMARKS:
                    if bn not in src[sk]:
                        continue
                    for seed_str, cell in src[sk][bn].items():
                        if cell and cell.get("mse_per_step"):
                            results[sk][bn][seed_str] = cell
                            n_loaded += 1
            print(f"[resume] loaded {n_loaded} completed cells from {out_path}")
        except Exception as e:
            print(f"[resume] failed ({e}); starting fresh")

    for sigma in sigmas_to_run:
        sk = sigma_key(sigma)
        for bn, gf in benches_to_run.items():
            for seed in seeds_to_run:
                seed_str = str(seed)
                if results[sk][bn].get(seed_str) and results[sk][bn][seed_str].get("mse_per_step"):
                    r = results[sk][bn][seed_str]
                    h15 = r.get("horizons", {}).get("H=15", {})
                    print(f"[sigma={sigma} {bn:4s} seed={seed}] resumed  "
                          f"H15_ratio={h15.get('mse_ratio_vs_step1', float('nan')):.2f}")
                    continue
                ckpt = ckpt_path_for(sigma, bn, seed)
                if not ckpt.exists():
                    print(f"[sigma={sigma} {bn:4s} seed={seed}] MISSING checkpoint {ckpt}; skipping")
                    results[sk][bn][seed_str] = {
                        "error": "checkpoint missing", "ckpt": str(ckpt),
                        "sigma": sigma, "bench": bn, "seed": seed,
                    }
                    continue
                print("=" * 78)
                print(f"probe  sigma={sigma}  {bn}  seed={seed}  K={K}")
                print("=" * 78)
                try:
                    r = probe_one(sigma, bn, gf, seed,
                                  max_h=smoke_max_h if smoke else None)
                    results[sk][bn][seed_str] = r
                    print(f"  probe time: {r['probe_time_s']:.1f}s  params={r['params']}")
                    print(f"  step 1 MSE = {r['mse_per_step'][0]:.4e}")
                    for hh in HORIZONS:
                        if hh > r["max_h"]:
                            continue
                        c = r["horizons"][f"H={hh}"]
                        print(f"  H={hh:3d}  MSE={c['mse_final']:.4e}  "
                              f"ratio_vs_step1={c['mse_ratio_vs_step1']:.2f}  "
                              f"cos_div={c['cosine_div_final']:.4f}")
                except Exception as e:
                    print(f"  FAILED: {type(e).__name__}: {e}")
                    results[sk][bn][seed_str] = {
                        "error": str(e), "sigma": sigma, "bench": bn, "seed": seed,
                    }
                with open(out_path, "w") as f:
                    json.dump({
                        "per_cell": results,
                        "protocol": {
                            "model": MODEL, "K": K, "grid": GRID,
                            "n_steps": N_STEPS,
                            "n_trajectories": N_TRAJECTORIES,
                            "sigmas": SIGMAS, "seeds": SEEDS,
                            "horizons": HORIZONS,
                            "n_rollout_trajectories": N_ROLLOUT_TRAJS,
                            "smoke": smoke,
                        },
                    }, f, indent=2)
                gc.collect()

    # ---- Summary ---------------------------------------------------------
    print()
    print("=" * 90)
    print("SUMMARY — noise-injection rollout stability (median across seeds)")
    print("=" * 90)
    hdr = f"{'sigma':>8s}  {'bench':6s}  {'H':>3s}  {'MSE median':>12s}  {'ratio median':>14s}  {'cos_div med':>12s}"
    print(hdr)
    print("-" * len(hdr))
    summary = {sigma_key(s): {bn: {} for bn in BENCHMARKS} for s in SIGMAS}
    for sigma in SIGMAS:
        sk = sigma_key(sigma)
        for bn in BENCHMARKS:
            for hh in HORIZONS:
                ratios, mses, cosds = [], [], []
                for seed_str, cell in results[sk][bn].items():
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
                    summary[sk][bn][f"H={hh}"] = s
                    print(f"{sigma:>8.3f}  {bn:6s}  {hh:>3d}  "
                          f"{s['mse_median']:>12.4e}  "
                          f"{s['ratio_median']:>14.2f}  "
                          f"{s['cos_div_median']:>12.4f}")

    # Decision gate (sigma=0.02 is the one that matters)
    print()
    print("=" * 90)
    print("DECISION GATE — stable to H=15 iff median(MSE_15 / MSE_1) < 2.0")
    print("=" * 90)
    gate = {sigma_key(s): {} for s in SIGMAS}
    for sigma in SIGMAS:
        sk = sigma_key(sigma)
        for bn in BENCHMARKS:
            s15 = summary[sk][bn].get("H=15", {})
            r = s15.get("ratio_median")
            stable = (r is not None) and (r < 2.0)
            gate[sk][bn] = {"stable_to_H15": bool(stable), "ratio_median": r}
            r_str = f"{r:.2f}" if r is not None else "n/a"
            print(f"sigma={sigma:<5} {bn:6s}  median_ratio_H15={r_str:>6s}   "
                  f"{'STABLE (drop posterior)' if stable else 'UNSTABLE (keep posterior)'}")

    with open(out_path, "w") as f:
        json.dump({
            "per_cell": results,
            "summary_median": summary,
            "decision_gate": gate,
            "protocol": {
                "model": MODEL, "K": K, "grid": GRID,
                "n_steps": N_STEPS,
                "n_trajectories": N_TRAJECTORIES,
                "sigmas": SIGMAS, "seeds": SEEDS,
                "horizons": HORIZONS,
                "n_rollout_trajectories": N_ROLLOUT_TRAJS,
                "smoke": smoke,
            },
        }, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true",
                   help="Only sigma=0.02, heat, seed=42, H=15 (uses ablation smoke ckpt).")
    p.add_argument("--smoke-max-h", type=int, default=15)
    args = p.parse_args()
    main(smoke=args.smoke, smoke_max_h=args.smoke_max_h)
