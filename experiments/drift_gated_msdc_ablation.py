"""Combo experiment: drift-gated hybrid + MSDC multistep loss.

Sibling of `drift_gated_multistep_ablation.py`. Same recipe, but the
multistep penalty is *drift-conditioned* (MSDC, brainstorm B §2.1):
each step h's MSE contribution is multiplied by ``(1 - alpha * gate_h)``,
where ``gate_h`` is the per-cell drift gate evaluated on the rolled
state. This gives the gate a coherence-discrimination gradient signal.

Hyperparams (matched to combo A so results are directly comparable):
  - rescor_mamba_gated_rand
  - H = 8, K_bptt = 4, msdc_alpha = 0.5
  - gate_bias_init = 1.0 (same combo-A mitigation)
  - 3 seeds × {gs, ks} × 100 epochs
  - bf16 + compile + cuda + batch=128 + lr=1.4e-3

Saves:
  - experiments/results/drift_gated_msdc_ablation.json
  - experiments/results/drift_gated_msdc_ckpts/{bench}_seed{seed}.pt

Usage:
    PYTHONPATH=src PYTHONUNBUFFERED=1 uv run --with scikit-learn,scipy \\
        python experiments/drift_gated_msdc_ablation.py
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
GATE_BIAS_INIT = 1.0  # same mitigation as combo A
MSDC_ALPHA = 0.5      # half-strength drift conditioning per brainstorm §2.1

CKPT_DIR = Path("experiments/results/drift_gated_msdc_ckpts")


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


def run_one(bench_name, gen_fn, seed, epochs, *,
            train_device=TRAIN_DEVICE, data_device=DATA_DEVICE,
            use_compile=USE_COMPILE, use_bf16=USE_BF16,
            batch_size=BATCH_SIZE):
    t0 = time.time()
    data = gen_fn(grid_size=GRID, seed=seed, n_steps=N_STEPS,
                  n_trajectories=N_TRAJECTORIES, context_k=CONTEXT_K,
                  device=data_device)
    meta = data.meta
    m = create_model(
        MODEL,
        in_channels=meta["in_channels"],
        out_channels=meta["out_channels"],
        grid_size=GRID,
        seed=seed,
    )
    # Combo-A mitigation: gate_bias_init = 1.0
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
        epochs=epochs, batch_size=batch_size, lr=LR,
        device=train_device,
        compile=use_compile,
        bf16=use_bf16,
        multistep_horizon=MULTISTEP_HORIZON,
        multistep_bptt=MULTISTEP_BPTT,
        multistep_n_steps=N_STEPS,
        msdc_alpha=MSDC_ALPHA,
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
        "msdc_alpha": MSDC_ALPHA,
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="Single seed (42) × single bench × 2 epochs on CPU")
    parser.add_argument("--bench", default=None,
                        help="Restrict to one benchmark (gs|ks|heat|...)")
    args = parser.parse_args()

    if args.smoke:
        seeds = [42]
        epochs = 2
        train_device = "cpu"
        data_device = "cpu"
        use_compile = False
        use_bf16 = False
        batch_size = 32
        if args.bench:
            from wmca import benchmarks as wb
            gen_fn = getattr(wb, f"generate_{args.bench}")
            benchmarks = {args.bench: gen_fn}
        else:
            benchmarks = {"gs": generate_gray_scott}
        out_path = Path("experiments/results/drift_gated_msdc_ablation_smoke.json")
    else:
        seeds = SEEDS
        epochs = EPOCHS
        train_device = TRAIN_DEVICE
        data_device = DATA_DEVICE
        use_compile = USE_COMPILE
        use_bf16 = USE_BF16
        batch_size = BATCH_SIZE
        benchmarks = BENCHMARKS
        out_path = Path("experiments/results/drift_gated_msdc_ablation.json")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    results = {bn: {} for bn in benchmarks}
    if out_path.exists():
        try:
            with open(out_path, "r") as f:
                prior = json.load(f)
            src = prior.get("per_cell", prior)
            n_loaded = 0
            for bn in benchmarks:
                if bn not in src:
                    continue
                for sk, cell in src[bn].items():
                    if cell and cell.get("score") is not None:
                        results[bn][sk] = cell
                        n_loaded += 1
            print(f"[resume] loaded {n_loaded} completed cells from {out_path}")
        except Exception as e:
            print(f"[resume] failed ({e}); starting fresh")

    for seed in seeds:
        print("=" * 78)
        print(f"{MODEL}  H={MULTISTEP_HORIZON}  K_bptt={MULTISTEP_BPTT}  "
              f"alpha={MSDC_ALPHA}  gate_bias_init={GATE_BIAS_INIT}  "
              f"seed={seed}  epochs={epochs}")
        print("=" * 78)
        for bn, gf in benchmarks.items():
            existing = results[bn].get(str(seed), {})
            if existing.get("score") is not None:
                print(f"  {bn:6s}  {existing['score']:.4e}  [resumed]  "
                      f"params={existing.get('params', '?')}")
                continue
            try:
                r = run_one(bn, gf, seed, epochs,
                            train_device=train_device, data_device=data_device,
                            use_compile=use_compile, use_bf16=use_bf16,
                            batch_size=batch_size)
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
                        "epochs": epochs, "grid": GRID,
                        "n_steps": N_STEPS,
                        "n_trajectories": N_TRAJECTORIES,
                        "multistep_horizon": MULTISTEP_HORIZON,
                        "multistep_bptt": MULTISTEP_BPTT,
                        "msdc_alpha": MSDC_ALPHA,
                        "gate_bias_init": GATE_BIAS_INIT,
                        "seeds": seeds,
                        "benchmarks": list(benchmarks),
                    },
                }, f, indent=2)
            gc.collect()
        print()

    print("=" * 90)
    print("SUMMARY — drift-gated + MSDC H=8 alpha=0.5 (1-step MSE medians)")
    print("=" * 90)
    summary = {}
    for bn in benchmarks:
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
            "model": MODEL, "context_k": CONTEXT_K, "epochs": epochs,
            "grid": GRID, "n_steps": N_STEPS,
            "n_trajectories": N_TRAJECTORIES,
            "multistep_horizon": MULTISTEP_HORIZON,
            "multistep_bptt": MULTISTEP_BPTT,
            "msdc_alpha": MSDC_ALPHA,
            "gate_bias_init": GATE_BIAS_INIT,
            "seeds": seeds,
            "benchmarks": list(benchmarks),
        },
    }
    with open(out_path, "w") as f:
        json.dump(final, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
