"""Noise-injection training ablation for rescor_rens K=32.

Task #32 / noise_inject_plan.md.

Tests whether training with Gaussian noise (sigma=0.02) injected on the
input x stabilizes autoregressive rollouts for rescor_rens K=32. This is
the cheapest architecture-free attempt to revive the DROP-posterior
decision from Task #27.

Protocol:
  - model: rescor_rens (K=32)
  - noise sigma: {0.0, 0.02}
  - seeds: {42, 43, 44}
  - benchmarks: {heat, gs, ks}  (same subset as rollout_stability_probe)
  - epochs: 100
  - grid_size: 16, n_steps: 105, n_trajectories: 200
  - batch_size: 64, lr: 1e-3

Saves:
  - experiments/results/noise_inject_ablation.json
  - experiments/results/noise_inject_ckpts/rens_K32_sigma{sigma}_{bench}_seed{seed}.pt

Usage:
    PYTHONPATH=src PYTHONUNBUFFERED=1 uv run --with scikit-learn,scipy \\
        python experiments/noise_inject_ablation.py
"""
from __future__ import annotations

import gc
import json
import math
import time
from pathlib import Path
from statistics import median

import torch

from wmca.benchmarks import (
    generate_gray_scott,
    generate_heat,
    generate_ks,
)
from wmca.model_registry import create_model, train_model


BENCHMARKS = {
    "heat": generate_heat,
    "gs": generate_gray_scott,
    "ks": generate_ks,
}

SIGMAS = [0.0, 0.02]
SEEDS = [42, 43, 44]
K = 32
MODEL = "rescor_rens"
EPOCHS = 100
N_STEPS = 105
N_TRAJECTORIES = 200
GRID = 16

CKPT_DIR = Path("experiments/results/noise_inject_ckpts")


def eval_mse(model, X_test, Y_test):
    model.eval()
    device = next(model.parameters()).device
    total = 0.0
    n = 0
    batch_size = 256
    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            xb = X_test[i : i + batch_size].to(device)
            yb = Y_test[i : i + batch_size].to(device)
            pb = model(xb)
            total += float(((pb - yb) ** 2).mean().item()) * len(xb)
            n += len(xb)
    return total / max(n, 1)


def run_one(sigma, bench_name, gen_fn, seed):
    t0 = time.time()
    data = gen_fn(grid_size=GRID, seed=seed, n_steps=N_STEPS,
                  n_trajectories=N_TRAJECTORIES)
    meta = data.meta
    m = create_model(
        MODEL,
        in_channels=meta["in_channels"],
        out_channels=meta["out_channels"],
        grid_size=GRID,
        seed=seed,
        cml_K=K,
    )
    pc = m.param_count()
    m = train_model(
        m, data.X_train, data.Y_train,
        X_val=data.X_val, Y_val=data.Y_val,
        loss_type=meta["loss_type"],
        epochs=EPOCHS, batch_size=64, lr=1e-3,
        train_noise_sigma=sigma,
    )
    mse_1step = eval_mse(m, data.X_test, data.Y_test)
    elapsed = time.time() - t0

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CKPT_DIR / f"rens_K32_sigma{sigma}_{bench_name}_seed{seed}.pt"
    # state_dict on CPU (train_model already moves the model back to CPU
    # at the end, but be explicit for safety).
    cpu_state = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
    torch.save(cpu_state, ckpt_path)

    return {
        "score": mse_1step,
        "metric": "mse",
        "sigma": sigma,
        "bench": bench_name,
        "seed": seed,
        "params": pc,
        "train_time_s": elapsed,
        "ckpt": str(ckpt_path),
    }


def stats_summary(values):
    if not values:
        return {"mean": None, "median": None, "std": None, "min": None, "max": None, "n": 0}
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    std = math.sqrt(var)
    return {
        "mean": mean,
        "median": median(values),
        "std": std,
        "min": min(values),
        "max": max(values),
        "n": n,
    }


def sigma_key(sigma: float) -> str:
    return f"sigma={sigma}"


def main():
    out_path = Path("experiments/results/noise_inject_ablation.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    # results[sigma_key][bench_name][seed_str] = cell
    results = {sigma_key(s): {bn: {} for bn in BENCHMARKS} for s in SIGMAS}

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
                        if cell and cell.get("score") is not None:
                            results[sk][bn][seed_str] = cell
                            n_loaded += 1
            print(f"[resume] loaded {n_loaded} completed cells from {out_path}")
        except Exception as e:
            print(f"[resume] failed to parse prior JSON ({e}); starting fresh")

    for sigma in SIGMAS:
        sk = sigma_key(sigma)
        for seed in SEEDS:
            print("=" * 78)
            print(f"{MODEL}  K={K}  sigma={sigma}  seed={seed}  epochs={EPOCHS}")
            print("=" * 78)
            for bn, gf in BENCHMARKS.items():
                existing = results[sk][bn].get(str(seed), {})
                if existing.get("score") is not None:
                    print(f"  {bn:6s}  {existing['score']:.4e}  [resumed]  "
                          f"params={existing.get('params', '?')}")
                    continue
                try:
                    r = run_one(sigma, bn, gf, seed)
                    results[sk][bn][str(seed)] = r
                    print(f"  {bn:6s}  {r['score']:.4e}  "
                          f"[{r['train_time_s']:.0f}s]  params={r['params']}")
                except Exception as e:
                    print(f"  {bn:6s}  FAILED: {type(e).__name__}: {e}")
                    results[sk][bn][str(seed)] = {
                        "score": None, "metric": "mse", "error": str(e),
                        "sigma": sigma, "bench": bn, "seed": seed,
                    }
                with open(out_path, "w") as f:
                    json.dump({
                        "per_cell": results,
                        "protocol": {
                            "model": MODEL, "K": K, "epochs": EPOCHS,
                            "grid": GRID, "n_steps": N_STEPS,
                            "n_trajectories": N_TRAJECTORIES,
                            "sigmas": SIGMAS, "seeds": SEEDS,
                            "benchmarks": list(BENCHMARKS),
                        },
                    }, f, indent=2)
                gc.collect()
            print()

    # ---- Summary ---------------------------------------------------------
    print("=" * 90)
    print(f"SUMMARY — noise-injection ablation (1-step MSE medians, "
          f"{len(SEEDS)} seeds × {EPOCHS} epochs)")
    print("=" * 90)

    summary = {sigma_key(s): {} for s in SIGMAS}
    for sigma in SIGMAS:
        sk = sigma_key(sigma)
        for bn in BENCHMARKS:
            scores = [r["score"] for r in results[sk][bn].values()
                      if r.get("score") is not None]
            summary[sk][bn] = stats_summary(scores)

    hdr = f"{'sigma':>8s}  {'bench':6s}  {'median':>14s}  {'mean':>14s}  {'std':>12s}  n"
    print(hdr)
    print("-" * len(hdr))
    for sigma in SIGMAS:
        sk = sigma_key(sigma)
        for bn in BENCHMARKS:
            s = summary[sk][bn]
            med = f"{s['median']:.4e}" if s['median'] is not None else "n/a"
            mn = f"{s['mean']:.4e}" if s['mean'] is not None else "n/a"
            sd = f"{s['std']:.2e}" if s['std'] is not None else "n/a"
            print(f"{sigma:>8.3f}  {bn:6s}  {med:>14s}  {mn:>14s}  {sd:>12s}  {s['n']}")

    final = {
        "per_cell": results,
        "summary_1step": summary,
        "protocol": {
            "model": MODEL, "K": K, "epochs": EPOCHS,
            "grid": GRID, "n_steps": N_STEPS,
            "n_trajectories": N_TRAJECTORIES,
            "sigmas": SIGMAS, "seeds": SEEDS,
            "benchmarks": list(BENCHMARKS),
        },
    }
    with open(out_path, "w") as f:
        json.dump(final, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
