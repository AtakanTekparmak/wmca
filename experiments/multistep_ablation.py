"""Multistep penalty NODE loss ablation for rescor_mamba_rand.

Sibling of `pushforward_ablation_mamba.py`. Trains rescor_mamba_rand
with the multistep-penalty training scheme (Chakraborty et al. 2024,
arXiv 2407.00568 / 2410.05572): roll H steps, supervise per-step
against ground truth, BPTT through the last K_bptt steps only.

Hypothesis test: does the multistep penalty fix mamba_rand's H=100
catastrophe on gs while preserving its H=15 absolute-MSE win
over rens?

Variants: multistep_horizon ∈ {1, 4, 8} × 3 seeds × 3 benchmarks (heat, gs, ks).
- H=1 is the no-multistep baseline (bit-identical to current train_model).
- H=4 with K_bptt=4: full BPTT.
- H=8 with K_bptt=4: half no_grad rollout, half BPTT.

Saves:
  - experiments/results/multistep_ablation.json
  - experiments/results/multistep_ckpts/mamba_rand_h{H}_{bench}_seed{seed}.pt

Usage:
    PYTHONPATH=src PYTHONUNBUFFERED=1 uv run --with scikit-learn,scipy \\
        python experiments/multistep_ablation.py            # full
    PYTHONPATH=src PYTHONUNBUFFERED=1 uv run --with scikit-learn,scipy \\
        python experiments/multistep_ablation.py --smoke    # smoke
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

# (multistep_horizon, multistep_bptt) variants. H=1 is the no-multistep
# baseline; train_model treats H=1 as a no-op (bit-identical to legacy).
HORIZONS = [1, 4, 8]
BPTT_FOR_H = {1: 1, 4: 4, 8: 4}
SEEDS = [42, 43, 44]
MODEL = "rescor_mamba_rand"
EPOCHS = 100
N_STEPS = 105
N_TRAJECTORIES = 200
GRID = 16
CONTEXT_K = 4
WEIGHT_SCHEDULE = "uniform"

# ---- GPU-utilization tuning constants ---------------------------------------
# RTX Pro 6000 96GB; same stack as Day 1 pushforward ablation.
BATCH_SIZE = 128
LR = 1.4e-3
USE_COMPILE = True
USE_BF16 = True   # If multistep training NaNs (bf16 underflow on chaotic
                  # gradients), fall back to USE_BF16 = False.
TRAIN_DEVICE = "cuda"
DATA_DEVICE = "cuda"

CKPT_DIR = Path("experiments/results/multistep_ckpts")


def n_pair_steps(n_steps: int, context_k: int) -> int:
    """Number of (X_window, Y) pairs per trajectory in the K-frame layout.

    _make_k_frame_pairs builds (T+1 - K) windows per trajectory; here
    T+1 = n_steps + 1, so n_windows = n_steps + 1 - K.
    """
    return n_steps + 1 - context_k


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


def run_one(H, bench_name, gen_fn, seed, epochs, use_bf16=USE_BF16):
    t0 = time.time()
    data = gen_fn(grid_size=GRID, seed=seed, n_steps=N_STEPS,
                  n_trajectories=N_TRAJECTORIES, context_k=CONTEXT_K,
                  device=DATA_DEVICE)
    meta = data.meta
    m = create_model(
        MODEL,
        in_channels=meta["in_channels"],
        out_channels=meta["out_channels"],
        grid_size=GRID,
        seed=seed,
    )
    pc = m.param_count() if hasattr(m, "param_count") else {
        "trained": sum(p.numel() for p in m.parameters() if p.requires_grad),
        "frozen": sum(p.numel() for p in m.parameters() if not p.requires_grad),
    }

    K_bptt = BPTT_FOR_H[H]
    n_pairs = n_pair_steps(N_STEPS, CONTEXT_K)

    m = train_model(
        m, data.X_train, data.Y_train,
        X_val=data.X_val, Y_val=data.Y_val,
        loss_type=meta["loss_type"],
        epochs=epochs, batch_size=BATCH_SIZE, lr=LR,
        multistep_horizon=H,
        multistep_bptt=K_bptt,
        multistep_weight_schedule=WEIGHT_SCHEDULE,
        multistep_n_steps=n_pairs,
        device=TRAIN_DEVICE,
        compile=USE_COMPILE,
        bf16=use_bf16,
    )
    mse_1step = eval_mse(m, data.X_test, data.Y_test)
    elapsed = time.time() - t0

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CKPT_DIR / f"mamba_rand_h{H}_{bench_name}_seed{seed}.pt"
    cpu_state = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
    torch.save(cpu_state, ckpt_path)

    return {
        "score": mse_1step,
        "metric": "mse",
        "multistep_horizon": H,
        "multistep_bptt": K_bptt,
        "weight_schedule": WEIGHT_SCHEDULE,
        "bench": bench_name,
        "seed": seed,
        "params": pc,
        "train_time_s": elapsed,
        "ckpt": str(ckpt_path),
        "bf16": bool(use_bf16),
    }


def stats_summary(values):
    if not values:
        return {"mean": None, "median": None, "std": None,
                "min": None, "max": None, "n": 0}
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    std = math.sqrt(var)
    return {
        "mean": mean, "median": median(values), "std": std,
        "min": min(values), "max": max(values), "n": n,
    }


def h_key(H: int) -> str:
    return f"H={H}"


def main(smoke: bool = False, bf16: bool = USE_BF16):
    out_path = Path("experiments/results/multistep_ablation.json")
    if smoke:
        out_path = Path("experiments/results/multistep_ablation_smoke.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    if smoke:
        horizons = [4]
        seeds = [42]
        benches = {"heat": generate_heat}
        epochs = 2
    else:
        horizons = HORIZONS
        seeds = SEEDS
        benches = BENCHMARKS
        epochs = EPOCHS

    results = {h_key(H): {bn: {} for bn in benches} for H in horizons}

    if out_path.exists():
        try:
            with open(out_path, "r") as f:
                prior = json.load(f)
            src = prior.get("per_cell", prior)
            n_loaded = 0
            for hk in results:
                if hk not in src:
                    continue
                for bn in benches:
                    if bn not in src[hk]:
                        continue
                    for seed_str, cell in src[hk][bn].items():
                        if cell and cell.get("score") is not None:
                            results[hk][bn][seed_str] = cell
                            n_loaded += 1
            print(f"[resume] loaded {n_loaded} completed cells from {out_path}")
        except Exception as e:
            print(f"[resume] failed to parse prior JSON ({e}); starting fresh")

    for H in horizons:
        hk = h_key(H)
        for seed in seeds:
            print("=" * 78)
            print(f"{MODEL}  context_k={CONTEXT_K}  multistep_H={H}  "
                  f"K_bptt={BPTT_FOR_H[H]}  seed={seed}  epochs={epochs}")
            print("=" * 78)
            for bn, gf in benches.items():
                existing = results[hk][bn].get(str(seed), {})
                if existing.get("score") is not None:
                    print(f"  {bn:6s}  {existing['score']:.4e}  [resumed]  "
                          f"params={existing.get('params', '?')}")
                    continue
                try:
                    r = run_one(H, bn, gf, seed, epochs, use_bf16=bf16)
                    results[hk][bn][str(seed)] = r
                    print(f"  {bn:6s}  {r['score']:.4e}  "
                          f"[{r['train_time_s']:.0f}s]  params={r['params']}")
                except Exception as e:
                    print(f"  {bn:6s}  FAILED: {type(e).__name__}: {e}")
                    import traceback; traceback.print_exc()
                    results[hk][bn][str(seed)] = {
                        "score": None, "metric": "mse", "error": str(e),
                        "multistep_horizon": H, "bench": bn, "seed": seed,
                    }
                with open(out_path, "w") as f:
                    json.dump({
                        "per_cell": results,
                        "protocol": {
                            "model": MODEL, "context_k": CONTEXT_K,
                            "epochs": epochs, "grid": GRID,
                            "n_steps": N_STEPS,
                            "n_trajectories": N_TRAJECTORIES,
                            "horizons": horizons, "seeds": seeds,
                            "benchmarks": list(benches),
                            "bptt_for_h": BPTT_FOR_H,
                            "weight_schedule": WEIGHT_SCHEDULE,
                            "batch_size": BATCH_SIZE, "lr": LR,
                            "compile": USE_COMPILE, "bf16": bool(bf16),
                            "smoke": smoke,
                        },
                    }, f, indent=2)
                gc.collect()
            print()

    print("=" * 90)
    print(f"SUMMARY — multistep mamba_rand ablation (1-step MSE medians, "
          f"{len(seeds)} seeds × {epochs} epochs)")
    print("=" * 90)

    summary = {h_key(H): {} for H in horizons}
    for H in horizons:
        hk = h_key(H)
        for bn in benches:
            scores = [r["score"] for r in results[hk][bn].values()
                      if r.get("score") is not None]
            summary[hk][bn] = stats_summary(scores)

    hdr = (f"{'H':>3s}  {'bench':6s}  {'median':>14s}  {'mean':>14s}  "
           f"{'std':>12s}  n")
    print(hdr)
    print("-" * len(hdr))
    for H in horizons:
        hk = h_key(H)
        for bn in benches:
            s = summary[hk][bn]
            med = f"{s['median']:.4e}" if s['median'] is not None else "n/a"
            mn = f"{s['mean']:.4e}" if s['mean'] is not None else "n/a"
            sd = f"{s['std']:.2e}" if s['std'] is not None else "n/a"
            print(f"{H:>3d}  {bn:6s}  {med:>14s}  {mn:>14s}  "
                  f"{sd:>12s}  {s['n']}")

    final = {
        "per_cell": results,
        "summary_1step": summary,
        "protocol": {
            "model": MODEL, "context_k": CONTEXT_K, "epochs": epochs,
            "grid": GRID, "n_steps": N_STEPS,
            "n_trajectories": N_TRAJECTORIES,
            "horizons": horizons, "seeds": seeds,
            "benchmarks": list(benches),
            "bptt_for_h": BPTT_FOR_H,
            "weight_schedule": WEIGHT_SCHEDULE,
            "batch_size": BATCH_SIZE, "lr": LR,
            "compile": USE_COMPILE, "bf16": bool(bf16),
            "smoke": smoke,
        },
    }
    with open(out_path, "w") as f:
        json.dump(final, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true",
                   help="1 seed × 2 epochs × heat × H=4 only (smoke test).")
    p.add_argument("--no-bf16", action="store_true",
                   help="Disable bf16 autocast (fp32 training). Use if "
                        "multistep gradients underflow / NaN.")
    args = p.parse_args()
    main(smoke=args.smoke, bf16=(not args.no_bf16))
