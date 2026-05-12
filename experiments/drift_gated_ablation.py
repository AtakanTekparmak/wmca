"""Drift-gated hybrid (rescor_mamba_gated_rand) ablation.

Day 4-5 of the all-horizon-stability sprint. Sibling of
``pushforward_ablation_mamba.py`` but tests the **architectural** fix
(drift-gated sigmoid attenuator on the Mamba+NCA correction) rather
than a training-time fix (pushforward / multistep penalty).

Standard single-step MSE training — no pushforward, no multistep
horizon. The point is to test the gated architecture in isolation.

Hypothesis: the drift gate auto-attenuates the K=4-frame error
amplifier at high drift. Near-manifold predictions retain mamba_rand's
H=15 win; off-manifold predictions fall back toward pure rens K=32
(mediocre, but not catastrophic).

Saves:
  - experiments/results/drift_gated_ablation.json
  - experiments/results/drift_gated_ckpts/mamba_gated_rand_{bench}_seed{seed}.pt

Usage:
    PYTHONPATH=src PYTHONUNBUFFERED=1 uv run --with scikit-learn,scipy \\
        python experiments/drift_gated_ablation.py
    PYTHONPATH=src uv run python experiments/drift_gated_ablation.py --smoke
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

MODELS = ["rescor_mamba_gated_rand"]
SEEDS = [42, 43, 44]
EPOCHS = 100
N_STEPS = 105
N_TRAJECTORIES = 200
GRID = 16
CONTEXT_K = 4

# ---- GPU-utilization tuning constants ---------------------------------------
# Same operating point as pushforward_ablation_mamba.py. Compile + bf16 stack
# applied; the train_model wrapper logs and falls back to eager if compile
# fails on the Mamba block.
BATCH_SIZE = 128
LR = 1.4e-3
USE_COMPILE = True
USE_BF16 = True
TRAIN_DEVICE = "cuda"
DATA_DEVICE = "cuda"

CKPT_DIR = Path("experiments/results/drift_gated_ckpts")


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


def run_one(model_name, bench_name, gen_fn, seed, epochs,
            train_device: str = TRAIN_DEVICE,
            data_device: str = DATA_DEVICE,
            use_compile: bool = USE_COMPILE,
            use_bf16: bool = USE_BF16):
    t0 = time.time()
    data = gen_fn(grid_size=GRID, seed=seed, n_steps=N_STEPS,
                  n_trajectories=N_TRAJECTORIES, context_k=CONTEXT_K,
                  device=data_device)
    meta = data.meta
    m = create_model(
        model_name,
        in_channels=meta["in_channels"],
        out_channels=meta["out_channels"],
        grid_size=GRID,
        seed=seed,
    )
    pc = m.param_count() if hasattr(m, "param_count") else {
        "trained": sum(p.numel() for p in m.parameters() if p.requires_grad),
        "frozen": sum(p.numel() for p in m.parameters() if not p.requires_grad),
    }

    # Snapshot pre-train gate scalars for diagnosis.
    gate_scale_init = float(m.gate_scale.detach().cpu().item())
    gate_bias_init = float(m.gate_bias.detach().cpu().item())

    m = train_model(
        m, data.X_train, data.Y_train,
        X_val=data.X_val, Y_val=data.Y_val,
        loss_type=meta["loss_type"],
        epochs=epochs, batch_size=BATCH_SIZE, lr=LR,
        device=train_device,
        compile=use_compile,
        bf16=use_bf16,
    )
    mse_1step = eval_mse(m, data.X_test, data.Y_test)
    elapsed = time.time() - t0

    gate_scale_final = float(m.gate_scale.detach().cpu().item())
    gate_bias_final = float(m.gate_bias.detach().cpu().item())

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CKPT_DIR / f"{model_name}_{bench_name}_seed{seed}.pt"
    cpu_state = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
    torch.save(cpu_state, ckpt_path)

    return {
        "score": mse_1step,
        "metric": "mse",
        "model": model_name,
        "bench": bench_name,
        "seed": seed,
        "params": pc,
        "train_time_s": elapsed,
        "ckpt": str(ckpt_path),
        "gate_scale_init": gate_scale_init,
        "gate_bias_init": gate_bias_init,
        "gate_scale_final": gate_scale_final,
        "gate_bias_final": gate_bias_final,
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


def main(smoke: bool = False):
    out_path = Path("experiments/results/drift_gated_ablation.json")
    if smoke:
        out_path = Path("experiments/results/drift_gated_ablation_smoke.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    if smoke:
        models = ["rescor_mamba_gated_rand"]
        seeds = [42]
        benches = {"heat": generate_heat}
        epochs = 2
        # CPU smoke: avoid competing with GPU pod, avoid compile/bf16 paths.
        train_device = "cpu"
        data_device = "cpu"
        use_compile = False
        use_bf16 = False
    else:
        models = MODELS
        seeds = SEEDS
        benches = BENCHMARKS
        epochs = EPOCHS
        train_device = TRAIN_DEVICE
        data_device = DATA_DEVICE
        use_compile = USE_COMPILE
        use_bf16 = USE_BF16

    results = {mn: {bn: {} for bn in benches} for mn in models}

    if out_path.exists():
        try:
            with open(out_path, "r") as f:
                prior = json.load(f)
            src = prior.get("per_cell", prior)
            n_loaded = 0
            for mn in results:
                if mn not in src:
                    continue
                for bn in benches:
                    if bn not in src[mn]:
                        continue
                    for seed_str, cell in src[mn][bn].items():
                        if cell and cell.get("score") is not None:
                            results[mn][bn][seed_str] = cell
                            n_loaded += 1
            print(f"[resume] loaded {n_loaded} completed cells from {out_path}")
        except Exception as e:
            print(f"[resume] failed to parse prior JSON ({e}); starting fresh")

    for mn in models:
        for seed in seeds:
            print("=" * 78)
            print(f"{mn}  context_k={CONTEXT_K}  seed={seed}  epochs={epochs}")
            print("=" * 78)
            for bn, gf in benches.items():
                existing = results[mn][bn].get(str(seed), {})
                if existing.get("score") is not None:
                    print(f"  {bn:6s}  {existing['score']:.4e}  [resumed]  "
                          f"params={existing.get('params', '?')}")
                    continue
                try:
                    r = run_one(mn, bn, gf, seed, epochs,
                                train_device=train_device,
                                data_device=data_device,
                                use_compile=use_compile,
                                use_bf16=use_bf16)
                    results[mn][bn][str(seed)] = r
                    print(f"  {bn:6s}  {r['score']:.4e}  "
                          f"[{r['train_time_s']:.0f}s]  params={r['params']}")
                    print(f"          gate_scale: {r['gate_scale_init']:.4f} "
                          f"-> {r['gate_scale_final']:.4f}   "
                          f"gate_bias: {r['gate_bias_init']:.4f} "
                          f"-> {r['gate_bias_final']:.4f}")
                except Exception as e:
                    print(f"  {bn:6s}  FAILED: {type(e).__name__}: {e}")
                    import traceback; traceback.print_exc()
                    results[mn][bn][str(seed)] = {
                        "score": None, "metric": "mse", "error": str(e),
                        "model": mn, "bench": bn, "seed": seed,
                    }
                with open(out_path, "w") as f:
                    json.dump({
                        "per_cell": results,
                        "protocol": {
                            "models": models, "context_k": CONTEXT_K,
                            "epochs": epochs, "grid": GRID,
                            "n_steps": N_STEPS,
                            "n_trajectories": N_TRAJECTORIES,
                            "seeds": seeds,
                            "benchmarks": list(benches),
                            "smoke": smoke,
                            "batch_size": BATCH_SIZE, "lr": LR,
                            "compile": use_compile, "bf16": use_bf16,
                            "train_device": train_device,
                        },
                    }, f, indent=2)
                gc.collect()
            print()

    print("=" * 90)
    print(f"SUMMARY — drift-gated mamba_rand ablation (1-step MSE medians, "
          f"{len(seeds)} seeds × {epochs} epochs)")
    print("=" * 90)

    summary = {mn: {} for mn in models}
    for mn in models:
        for bn in benches:
            scores = [r["score"] for r in results[mn][bn].values()
                      if r.get("score") is not None]
            summary[mn][bn] = stats_summary(scores)

    hdr = (f"{'model':>26s}  {'bench':6s}  {'median':>14s}  {'mean':>14s}  "
           f"{'std':>12s}  n")
    print(hdr)
    print("-" * len(hdr))
    for mn in models:
        for bn in benches:
            s = summary[mn][bn]
            med = f"{s['median']:.4e}" if s['median'] is not None else "n/a"
            mn_v = f"{s['mean']:.4e}" if s['mean'] is not None else "n/a"
            sd = f"{s['std']:.2e}" if s['std'] is not None else "n/a"
            print(f"{mn:>26s}  {bn:6s}  {med:>14s}  {mn_v:>14s}  "
                  f"{sd:>12s}  {s['n']}")

    final = {
        "per_cell": results,
        "summary_1step": summary,
        "protocol": {
            "models": models, "context_k": CONTEXT_K, "epochs": epochs,
            "grid": GRID, "n_steps": N_STEPS,
            "n_trajectories": N_TRAJECTORIES,
            "seeds": seeds,
            "benchmarks": list(benches),
            "smoke": smoke,
            "batch_size": BATCH_SIZE, "lr": LR,
            "compile": use_compile, "bf16": use_bf16,
            "train_device": train_device,
        },
    }
    with open(out_path, "w") as f:
        json.dump(final, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true",
                   help="1 seed × 2 epochs × heat only on CPU (smoke test).")
    args = p.parse_args()
    main(smoke=args.smoke)
