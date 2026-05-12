"""Combo experiment: drift-gated hybrid + multistep penalty training.

The trivial stacking of Day 4-5 architectural change (`rescor_mamba_gated_rand`)
with Day 2-3's training-time change (multistep_horizon=8, K_bptt=4).

Single mitigation per `brainstorm_combo_minimal.md`: bump `gate_bias_init`
from 0.5 → 1.0 (initial gate ≈ sigmoid(1.0) ≈ 0.73, more open by default)
to keep the gate from collapsing closed under multistep training's
poisoned drift signal.

Saves:
  - experiments/results/drift_gated_multistep_ablation.json
  - experiments/results/drift_gated_multistep_ckpts/{bench}_seed{seed}.pt

Usage:
    PYTHONPATH=src PYTHONUNBUFFERED=1 uv run --with scikit-learn,scipy \\
        python experiments/drift_gated_multistep_ablation.py
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

from wmca.benchmarks import generate_gray_scott, generate_ks
from wmca.model_registry import create_model, train_model


BENCHMARKS = {
    "gs": generate_gray_scott,
    "ks": generate_ks,
}

MODEL = "rescor_mamba_gated_rand"
SEEDS = [42, 43, 44]
EPOCHS = 100
N_STEPS = 105
N_TRAJECTORIES = 200
GRID = 16
CONTEXT_K = 4

BATCH_SIZE = 128
LR = 1.4e-3
USE_COMPILE = True
USE_BF16 = True
TRAIN_DEVICE = "cuda"
DATA_DEVICE = "cuda"

# Combo-specific:
MULTISTEP_HORIZON = 8
MULTISTEP_BPTT = 4
GATE_BIAS_INIT = 1.0  # bumped from default 0.5

CKPT_DIR = Path("experiments/results/drift_gated_multistep_ckpts")


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


def run_one(bench_name, gen_fn, seed, epochs):
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
    # Override gate_bias_init = 1.0 (mitigation per brainstorm_combo_minimal.md)
    with torch.no_grad():
        m.gate_bias.data.fill_(GATE_BIAS_INIT)
    pc = m.param_count() if hasattr(m, "param_count") else {
        "trained": sum(p.numel() for p in m.parameters() if p.requires_grad),
        "frozen": sum(p.numel() for p in m.parameters() if not p.requires_grad),
    }
    gate_scale_init = float(m.gate_scale.detach().cpu().item())
    gate_bias_init_actual = float(m.gate_bias.detach().cpu().item())

    m = train_model(
        m, data.X_train, data.Y_train,
        X_val=data.X_val, Y_val=data.Y_val,
        loss_type=meta["loss_type"],
        epochs=epochs, batch_size=BATCH_SIZE, lr=LR,
        device=TRAIN_DEVICE,
        compile=USE_COMPILE,
        bf16=USE_BF16,
        multistep_horizon=MULTISTEP_HORIZON,
        multistep_bptt=MULTISTEP_BPTT,
        multistep_n_steps=N_STEPS,
    )
    mse_1step = eval_mse(m, data.X_test, data.Y_test)
    elapsed = time.time() - t0

    gate_scale_final = float(m.gate_scale.detach().cpu().item())
    gate_bias_final = float(m.gate_bias.detach().cpu().item())

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CKPT_DIR / f"{bench_name}_seed{seed}.pt"
    cpu_state = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
    torch.save(cpu_state, ckpt_path)

    return {
        "score": mse_1step,
        "metric": "mse",
        "bench": bench_name,
        "seed": seed,
        "params": pc,
        "train_time_s": elapsed,
        "ckpt": str(ckpt_path),
        "gate_scale_init": gate_scale_init,
        "gate_scale_final": gate_scale_final,
        "gate_bias_init": gate_bias_init_actual,
        "gate_bias_final": gate_bias_final,
        "multistep_horizon": MULTISTEP_HORIZON,
        "multistep_bptt": MULTISTEP_BPTT,
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


def main():
    out_path = Path("experiments/results/drift_gated_multistep_ablation.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    results = {bn: {} for bn in BENCHMARKS}
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
                    if cell and cell.get("score") is not None:
                        results[bn][sk] = cell
                        n_loaded += 1
            print(f"[resume] loaded {n_loaded} completed cells from {out_path}")
        except Exception as e:
            print(f"[resume] failed ({e}); starting fresh")

    for seed in SEEDS:
        print("=" * 78)
        print(f"{MODEL}  H={MULTISTEP_HORIZON}  K_bptt={MULTISTEP_BPTT}  "
              f"gate_bias_init={GATE_BIAS_INIT}  seed={seed}  epochs={EPOCHS}")
        print("=" * 78)
        for bn, gf in BENCHMARKS.items():
            existing = results[bn].get(str(seed), {})
            if existing.get("score") is not None:
                print(f"  {bn:6s}  {existing['score']:.4e}  [resumed]  "
                      f"params={existing.get('params', '?')}")
                continue
            try:
                r = run_one(bn, gf, seed, EPOCHS)
                results[bn][str(seed)] = r
                print(f"  {bn:6s}  {r['score']:.4e}  "
                      f"[{r['train_time_s']:.0f}s]  "
                      f"gate_scale {r['gate_scale_init']:.3f}→{r['gate_scale_final']:.3f}  "
                      f"gate_bias {r['gate_bias_init']:.3f}→{r['gate_bias_final']:.3f}")
            except Exception as e:
                print(f"  {bn:6s}  FAILED: {type(e).__name__}: {e}")
                import traceback; traceback.print_exc()
                results[bn][str(seed)] = {
                    "score": None, "metric": "mse", "error": str(e),
                    "bench": bn, "seed": seed,
                }
            with open(out_path, "w") as f:
                json.dump({
                    "per_cell": results,
                    "protocol": {
                        "model": MODEL, "context_k": CONTEXT_K,
                        "epochs": EPOCHS, "grid": GRID,
                        "n_steps": N_STEPS,
                        "n_trajectories": N_TRAJECTORIES,
                        "multistep_horizon": MULTISTEP_HORIZON,
                        "multistep_bptt": MULTISTEP_BPTT,
                        "gate_bias_init": GATE_BIAS_INIT,
                        "seeds": SEEDS,
                        "benchmarks": list(BENCHMARKS),
                    },
                }, f, indent=2)
            gc.collect()
        print()

    print("=" * 90)
    print("SUMMARY — drift-gated + multistep H=8 (1-step MSE medians)")
    print("=" * 90)
    summary = {}
    for bn in BENCHMARKS:
        scores = [r["score"] for r in results[bn].values()
                  if r.get("score") is not None]
        summary[bn] = stats_summary(scores)
        s = summary[bn]
        med = f"{s['median']:.4e}" if s['median'] is not None else "n/a"
        mn = f"{s['mean']:.4e}" if s['mean'] is not None else "n/a"
        sd = f"{s['std']:.2e}" if s['std'] is not None else "n/a"
        print(f"{bn:6s}  median={med}  mean={mn}  std={sd}  n={s['n']}")

    final = {
        "per_cell": results,
        "summary_1step": summary,
        "protocol": {
            "model": MODEL, "context_k": CONTEXT_K, "epochs": EPOCHS,
            "grid": GRID, "n_steps": N_STEPS,
            "n_trajectories": N_TRAJECTORIES,
            "multistep_horizon": MULTISTEP_HORIZON,
            "multistep_bptt": MULTISTEP_BPTT,
            "gate_bias_init": GATE_BIAS_INIT,
            "seeds": SEEDS,
            "benchmarks": list(BENCHMARKS),
        },
    }
    with open(out_path, "w") as f:
        json.dump(final, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
