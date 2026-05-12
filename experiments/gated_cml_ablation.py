"""Learned (eps, beta) gate ablation — static vs dynamic vs baseline.

Compares rescor (frozen eps/beta) vs rescor_gate_static (gate computed once
from input) vs rescor_gate_dynamic (gate recomputed each CML step).

Usage:
    uv run --with scikit-learn,scipy python experiments/gated_cml_ablation.py
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

MODELS = ["rescor", "rescor_gate_static", "rescor_gate_dynamic"]


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

            # For gated models, check what eps/beta the gate learned
            gate_info = ""
            if hasattr(model.cml_2d, "gate"):
                with torch.no_grad():
                    sample = data.X_test[:10, :meta["out_channels"]]
                    eb = torch.sigmoid(model.cml_2d.gate(sample))
                    eps_mean = (eb[:, 0] * model.cml_2d.eps_max).mean().item()
                    beta_mean = (eb[:, 1] * model.cml_2d.beta_max).mean().item()
                    eps_std = (eb[:, 0] * model.cml_2d.eps_max).std().item()
                    beta_std = (eb[:, 1] * model.cml_2d.beta_max).std().item()
                gate_info = f"  eps={eps_mean:.3f}±{eps_std:.3f}, beta={beta_mean:.3f}±{beta_std:.3f}"

            results[model_name][bench_name] = {
                "score": score,
                "metric": meta["metric"],
                "params": pc,
                "gate_info": gate_info,
            }

            direction = "lower" if meta["metric"] == "mse" else "higher"
            print(f"  {bench_name:15s}  {meta['metric']}={score:.6e}  "
                  f"({direction} better)  [{elapsed:.1f}s]  {pc}{gate_info}")

    # Summary table
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    header = f"{'Model':25s}"
    for b in BENCHMARKS:
        header += f"  {b:>15s}"
    print(header)
    print("-" * len(header))
    for model_name in MODELS:
        row = f"{model_name:25s}"
        for b in BENCHMARKS:
            r = results[model_name].get(b, {})
            s = r.get("score", float("nan"))
            row += f"  {s:>15.6e}"
        print(row)

    # Gate analysis
    print(f"\n{'='*80}")
    print("LEARNED GATE VALUES (mean ± std over test samples)")
    print(f"{'='*80}")
    for model_name in MODELS:
        if model_name == "rescor":
            continue
        print(f"\n{model_name}:")
        for b in BENCHMARKS:
            gi = results[model_name].get(b, {}).get("gate_info", "")
            print(f"  {b:15s}{gi}")

    # Save
    out_path = Path("experiments/results/gated_cml_ablation.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
