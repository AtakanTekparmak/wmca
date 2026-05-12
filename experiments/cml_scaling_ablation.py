"""CML Scaling Ablation — sweep CML hyperparameters with fixed NCA.

Varies one CML axis at a time (steps, r, eps, beta, kernel_size) while
keeping the NCA correction fixed (vanilla rescor architecture). All CML
changes cost zero extra trained params.

Usage:
    uv run --with scikit-learn,scipy python experiments/cml_scaling_ablation.py
    uv run --with scikit-learn,scipy python experiments/cml_scaling_ablation.py --sweep steps
    uv run --with scikit-learn,scipy python experiments/cml_scaling_ablation.py --benchmarks heat gol gray_scott
"""
import argparse
import json
import time
from pathlib import Path

import torch

from wmca.benchmarks import (
    generate_gol,
    generate_gray_scott,
    generate_heat,
    generate_ks,
    generate_rule110,
    generate_wireworld,
)
from wmca.model_registry import create_model, train_model


# -- Benchmark registry -------------------------------------------------------

BENCHMARKS = {
    "heat": generate_heat,
    "gol": generate_gol,
    "gray_scott": generate_gray_scott,
    "ks": generate_ks,
    "rule110": generate_rule110,
    "wireworld": generate_wireworld,
}

DEFAULT_BENCHMARKS = ["heat", "gol", "gray_scott", "ks", "rule110", "wireworld"]

# -- Sweep configs -------------------------------------------------------------

DEFAULTS = {"r": 3.90, "eps": 0.30, "beta": 0.15, "cml_steps": 15, "kernel_size": 3}

SWEEPS = {
    "steps": {
        "param": "cml_steps",
        "values": [5, 10, 15, 30, 50, 100],
    },
    "r": {
        "param": "r",
        "values": [3.57, 3.70, 3.85, 3.90, 3.95, 3.99],
    },
    "eps": {
        "param": "eps",
        "values": [0.05, 0.15, 0.30, 0.50, 0.70],
    },
    "beta": {
        "param": "beta",
        "values": [0.01, 0.05, 0.15, 0.30, 0.50],
    },
    "kernel": {
        "param": "kernel_size",
        "values": [3, 5, 7],
    },
    "channels": {
        "param": "cml_channels",
        "values": [1, 4, 8, 16],
    },
}


# -- Evaluation ----------------------------------------------------------------

def evaluate_model(model, X_test, Y_test, meta):
    model.eval()
    with torch.no_grad():
        preds = model(X_test)

    if meta["metric"] == "mse":
        return ((preds - Y_test) ** 2).mean().item()
    elif meta["metric"] == "accuracy":
        if meta["loss_type"] in ("ce", "cross_entropy"):
            pred_classes = preds.argmax(dim=1)
            true_classes = Y_test.argmax(dim=1) if Y_test.dim() == 4 and Y_test.shape[1] > 1 else Y_test.long().squeeze(1)
            return (pred_classes == true_classes).float().mean().item()
        else:
            pred_binary = (preds > 0.5).float()
            return (pred_binary == Y_test).float().mean().item()
    return float("nan")


def run_single(benchmark_name, gen_fn, cml_kwargs, seed=42):
    data = gen_fn(grid_size=16, seed=seed)
    meta = data.meta

    model = create_model(
        "rescor",
        in_channels=meta["in_channels"],
        out_channels=meta["out_channels"],
        grid_size=16,
        seed=seed,
        **cml_kwargs,
    )

    model = train_model(
        model, data.X_train, data.Y_train,
        X_val=data.X_val, Y_val=data.Y_val,
        loss_type=meta["loss_type"],
        epochs=30, batch_size=64, lr=1e-3,
    )

    score = evaluate_model(model, data.X_test, data.Y_test, meta)
    pc = model.param_count()
    return {
        "score": score,
        "metric": meta["metric"],
        "trained_params": pc["trained"],
        "frozen_params": pc["frozen"],
    }


# -- Main ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="CML Scaling Ablation")
    parser.add_argument("--sweep", choices=list(SWEEPS.keys()) + ["all"],
                        default="all", help="Which CML axis to sweep")
    parser.add_argument("--benchmarks", nargs="+", default=DEFAULT_BENCHMARKS,
                        choices=list(BENCHMARKS.keys()))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="experiments/results/cml_scaling_ablation.json")
    args = parser.parse_args()

    sweeps_to_run = list(SWEEPS.keys()) if args.sweep == "all" else [args.sweep]
    results = {}

    for sweep_name in sweeps_to_run:
        sweep = SWEEPS[sweep_name]
        param_name = sweep["param"]
        values = sweep["values"]
        results[sweep_name] = {}

        print(f"\n{'='*60}")
        print(f"SWEEP: {sweep_name} ({param_name})")
        print(f"Values: {values}")
        print(f"{'='*60}")

        for val in values:
            cml_kwargs = dict(DEFAULTS)
            cml_kwargs[param_name] = val
            results[sweep_name][str(val)] = {}

            print(f"\n  {param_name}={val}")

            for bench_name in args.benchmarks:
                gen_fn = BENCHMARKS[bench_name]
                t0 = time.time()
                result = run_single(bench_name, gen_fn, cml_kwargs, seed=args.seed)
                elapsed = time.time() - t0

                results[sweep_name][str(val)][bench_name] = result
                direction = "lower" if result["metric"] == "mse" else "higher"
                print(f"    {bench_name:15s}  {result['metric']}={result['score']:.6e}  "
                      f"({direction} is better)  [{elapsed:.1f}s]")

        # Print summary table for this sweep
        print(f"\n--- {sweep_name} summary ---")
        header = f"{'Value':>8s}"
        for b in args.benchmarks:
            header += f"  {b:>15s}"
        print(header)
        print("-" * len(header))

        for val in values:
            row = f"{val:>8}"
            for b in args.benchmarks:
                r = results[sweep_name][str(val)].get(b, {})
                s = r.get("score", float("nan"))
                row += f"  {s:>15.6e}"
            print(row)

    # Save results
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
