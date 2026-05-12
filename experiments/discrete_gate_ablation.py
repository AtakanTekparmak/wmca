"""Discrete selection gate ablation — global vs per-cell vs baseline.

Compares rescor (frozen) vs rescor_discrete_global (5 softmax logits) vs
rescor_discrete_percell (per-cell softmax over K=5 candidates).

Usage:
    PYTHONUNBUFFERED=1 uv run --with scikit-learn,scipy python experiments/discrete_gate_ablation.py
"""
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


BENCHMARKS = {
    "heat": generate_heat,
    "gol": generate_gol,
    "gray_scott": generate_gray_scott,
    "ks": generate_ks,
    "rule110": generate_rule110,
    "wireworld": generate_wireworld,
}

MODELS = ["rescor", "rescor_discrete_global", "rescor_discrete_percell"]


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


def main():
    seed = 42
    results = {}

    for model_name in MODELS:
        results[model_name] = {}
        print(f"\n{'='*50}")
        print(f"MODEL: {model_name}")
        print(f"{'='*50}")

        for bench_name, gen_fn in BENCHMARKS.items():
            data = gen_fn(grid_size=16, seed=seed)
            meta = data.meta

            model = create_model(
                model_name,
                in_channels=meta["in_channels"],
                out_channels=meta["out_channels"],
                grid_size=16, seed=seed,
            )
            pc = model.param_count()

            t0 = time.time()
            model = train_model(
                model, data.X_train, data.Y_train,
                X_val=data.X_val, Y_val=data.Y_val,
                loss_type=meta["loss_type"],
                epochs=30, batch_size=64, lr=1e-3,
            )
            elapsed = time.time() - t0

            score = evaluate_model(model, data.X_test, data.Y_test, meta)

            # Get selection info for discrete models
            select_info = ""
            if hasattr(model.cml_2d, "get_selection_info"):
                si = model.cml_2d.get_selection_info()
                if "effective_eps" in si:
                    select_info = f"  eps={si['effective_eps']:.3f} beta={si['effective_beta']:.3f}"
                    # Show top-2 candidates
                    sorted_w = sorted(si["weights"].items(), key=lambda x: -x[1])
                    top2 = ", ".join(f"{k}:{v:.2f}" for k, v in sorted_w[:2])
                    select_info += f"  top2=[{top2}]"

            results[model_name][bench_name] = {
                "score": score,
                "metric": meta["metric"],
                "params": pc,
                "select_info": select_info,
            }

            direction = "lower" if meta["metric"] == "mse" else "higher"
            print(f"  {bench_name:15s}  {meta['metric']}={score:.6e}  "
                  f"({direction} better)  [{elapsed:.1f}s]  {pc}{select_info}")

    # Summary table
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    header = f"{'Model':30s}"
    for b in BENCHMARKS:
        header += f"  {b:>15s}"
    print(header)
    print("-" * len(header))
    for model_name in MODELS:
        row = f"{model_name:30s}"
        for b in BENCHMARKS:
            r = results[model_name].get(b, {})
            s = r.get("score", float("nan"))
            row += f"  {s:>15.6e}"
        print(row)

    # Selection analysis
    print(f"\n{'='*80}")
    print("DISCRETE GATE SELECTIONS")
    print(f"{'='*80}")
    for model_name in MODELS:
        if model_name == "rescor":
            continue
        print(f"\n{model_name}:")
        for b in BENCHMARKS:
            si = results[model_name].get(b, {}).get("select_info", "")
            print(f"  {b:15s}{si}")

    out_path = Path("experiments/results/discrete_gate_ablation.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
