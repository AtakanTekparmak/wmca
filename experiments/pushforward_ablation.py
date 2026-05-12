"""Pushforward-trick training ablation for rescor_rens K=32.

Pushforward trick (Brandstetter et al., "Message Passing Neural PDE
Solvers", ICLR 2022): with probability p (default 0.5) per training
step, replace the standard single-step loss with a two-step pushforward
loss that supervises the model's own one-step prediction against the
ground-truth t+2 target. ~30 LOC change, no architecture change. Near-
universally a win in the neural-PDE literature since 2022.

This ablation tests whether the pushforward trick stabilizes
autoregressive rollouts for rescor_rens K=32 on the heat / gs / ks
benchmarks (the same subset as the noise-injection / rollout-stability
probes).

Protocol:
  - model: rescor_rens (K=32)
  - pushforward: {False, True}
  - seeds: {42, 43, 44}
  - benchmarks: {heat, gs, ks}
  - epochs: 100
  - grid_size: 16, n_steps: 105, n_trajectories: 200
  - batch_size: 64, lr: 1e-3
  - pushforward_prob: 0.5

Saves:
  - experiments/results/pushforward_ablation.json
  - experiments/results/pushforward_ckpts/rens_K32_pf{0|1}_{bench}_seed{seed}.pt

Usage:
    PYTHONPATH=src PYTHONUNBUFFERED=1 uv run --with scikit-learn,scipy \\
        python experiments/pushforward_ablation.py

Smoke mode (1 seed, 2 epochs, heat only):
    PYTHONPATH=src PYTHONUNBUFFERED=1 uv run --with scikit-learn,scipy \\
        python experiments/pushforward_ablation.py --smoke
"""
from __future__ import annotations

import argparse
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

PUSHFORWARDS = [False, True]
SEEDS = [42, 43, 44]
K = 32
MODEL = "rescor_rens"
EPOCHS = 100
N_STEPS = 105
N_TRAJECTORIES = 200
GRID = 16
PUSHFORWARD_PROB = 0.5

# ---- GPU-utilization tuning constants ---------------------------------------
# RTX Pro 6000 96GB: kernel-launch bound at batch 64.
# Smoke benchmark (1-cell heat, 100 epochs):
#   - linear LR scaling (256/4e-3) -> 8.7x speedup but mse_ratio>1e6 (broken)
#   - sqrt LR scaling   (128/1.4e-3) -> 4.9x speedup, mse_ratio=24 (acceptable;
#     baseline mse=8.9e-8 is at bf16 floor — absolute mse 2.2e-6 stays useful)
# Picked batch=128/lr=1.4e-3 per spec fallback ("if loss gets meaningfully
# worse, fall back to batch 128 with sqrt-LR scaling"). Compile + bf16 give
# additional headroom on top of the batch-size win.
BATCH_SIZE = 128
LR = 1.4e-3  # = 1e-3 * sqrt(128 / 64), sqrt scaling from baseline
USE_COMPILE = True
USE_BF16 = True
TRAIN_DEVICE = "cuda"
DATA_DEVICE = "cuda"

CKPT_DIR = Path("experiments/results/pushforward_ckpts")


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


def run_one(pf, bench_name, gen_fn, seed, epochs):
    t0 = time.time()
    data = gen_fn(grid_size=GRID, seed=seed, n_steps=N_STEPS,
                  n_trajectories=N_TRAJECTORIES, device=DATA_DEVICE)
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
        epochs=epochs, batch_size=BATCH_SIZE, lr=LR,
        pushforward=pf,
        pushforward_n_steps=N_STEPS,
        pushforward_prob=PUSHFORWARD_PROB,
        device=TRAIN_DEVICE,
        compile=USE_COMPILE,
        bf16=USE_BF16,
    )
    mse_1step = eval_mse(m, data.X_test, data.Y_test)
    elapsed = time.time() - t0

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    pf_tag = 1 if pf else 0
    ckpt_path = CKPT_DIR / f"rens_K32_pf{pf_tag}_{bench_name}_seed{seed}.pt"
    cpu_state = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
    torch.save(cpu_state, ckpt_path)

    return {
        "score": mse_1step,
        "metric": "mse",
        "pushforward": pf,
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


def pf_key(pf: bool) -> str:
    return f"pushforward={pf}"


def main(smoke: bool = False):
    out_path = Path("experiments/results/pushforward_ablation.json")
    if smoke:
        out_path = Path("experiments/results/pushforward_ablation_smoke.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    if smoke:
        pfs = [True]
        seeds = [42]
        benches = {"heat": generate_heat}
        epochs = 2
    else:
        pfs = PUSHFORWARDS
        seeds = SEEDS
        benches = BENCHMARKS
        epochs = EPOCHS

    # results[pf_key][bench_name][seed_str] = cell
    results = {pf_key(p): {bn: {} for bn in benches} for p in pfs}

    if out_path.exists():
        try:
            with open(out_path, "r") as f:
                prior = json.load(f)
            src = prior.get("per_cell", prior)
            n_loaded = 0
            for pk in results:
                if pk not in src:
                    continue
                for bn in benches:
                    if bn not in src[pk]:
                        continue
                    for seed_str, cell in src[pk][bn].items():
                        if cell and cell.get("score") is not None:
                            results[pk][bn][seed_str] = cell
                            n_loaded += 1
            print(f"[resume] loaded {n_loaded} completed cells from {out_path}")
        except Exception as e:
            print(f"[resume] failed to parse prior JSON ({e}); starting fresh")

    for pf in pfs:
        pk = pf_key(pf)
        for seed in seeds:
            print("=" * 78)
            print(f"{MODEL}  K={K}  pushforward={pf}  seed={seed}  epochs={epochs}")
            print("=" * 78)
            for bn, gf in benches.items():
                existing = results[pk][bn].get(str(seed), {})
                if existing.get("score") is not None:
                    print(f"  {bn:6s}  {existing['score']:.4e}  [resumed]  "
                          f"params={existing.get('params', '?')}")
                    continue
                try:
                    r = run_one(pf, bn, gf, seed, epochs)
                    results[pk][bn][str(seed)] = r
                    print(f"  {bn:6s}  {r['score']:.4e}  "
                          f"[{r['train_time_s']:.0f}s]  params={r['params']}")
                except Exception as e:
                    print(f"  {bn:6s}  FAILED: {type(e).__name__}: {e}")
                    results[pk][bn][str(seed)] = {
                        "score": None, "metric": "mse", "error": str(e),
                        "pushforward": pf, "bench": bn, "seed": seed,
                    }
                with open(out_path, "w") as f:
                    json.dump({
                        "per_cell": results,
                        "protocol": {
                            "model": MODEL, "K": K, "epochs": epochs,
                            "grid": GRID, "n_steps": N_STEPS,
                            "n_trajectories": N_TRAJECTORIES,
                            "pushforwards": pfs, "seeds": seeds,
                            "benchmarks": list(benches),
                            "pushforward_prob": PUSHFORWARD_PROB,
                            "smoke": smoke,
                        },
                    }, f, indent=2)
                gc.collect()
            print()

    # ---- Summary ---------------------------------------------------------
    print("=" * 90)
    print(f"SUMMARY — pushforward ablation (1-step MSE medians, "
          f"{len(seeds)} seeds × {epochs} epochs)")
    print("=" * 90)

    summary = {pf_key(p): {} for p in pfs}
    for pf in pfs:
        pk = pf_key(pf)
        for bn in benches:
            scores = [r["score"] for r in results[pk][bn].values()
                      if r.get("score") is not None]
            summary[pk][bn] = stats_summary(scores)

    hdr = f"{'pushfwd':>8s}  {'bench':6s}  {'median':>14s}  {'mean':>14s}  {'std':>12s}  n"
    print(hdr)
    print("-" * len(hdr))
    for pf in pfs:
        pk = pf_key(pf)
        for bn in benches:
            s = summary[pk][bn]
            med = f"{s['median']:.4e}" if s['median'] is not None else "n/a"
            mn = f"{s['mean']:.4e}" if s['mean'] is not None else "n/a"
            sd = f"{s['std']:.2e}" if s['std'] is not None else "n/a"
            print(f"{str(pf):>8s}  {bn:6s}  {med:>14s}  {mn:>14s}  {sd:>12s}  {s['n']}")

    final = {
        "per_cell": results,
        "summary_1step": summary,
        "protocol": {
            "model": MODEL, "K": K, "epochs": epochs,
            "grid": GRID, "n_steps": N_STEPS,
            "n_trajectories": N_TRAJECTORIES,
            "pushforwards": pfs, "seeds": seeds,
            "benchmarks": list(benches),
            "pushforward_prob": PUSHFORWARD_PROB,
            "smoke": smoke,
        },
    }
    with open(out_path, "w") as f:
        json.dump(final, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true",
                   help="1 seed × 2 epochs × heat only (smoke test).")
    args = p.parse_args()
    main(smoke=args.smoke)
