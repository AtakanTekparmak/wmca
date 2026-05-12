"""Uniform-ESN + multi-r (vanilla rescor scaling) ablation.

Tests two hypotheses:

1. Is the ESN gate doing any work? Run rescor_esn with strict 1/K averaging
   (zero trainable gate params) at K=8/16/32. If uniform ~= learned, the gate
   is useless; if uniform < learned, the gate matters even at high K.

2. Does "vanilla rescor + scaling" work without ESN? rescor_mr = K vanilla CMLs
   (logistic + hand-designed coupling, same eps/beta) with K different r values
   log-spaced over [3.57, 3.99]. Diversity axis = chaos depth, not random
   spatial structure. Tests whether the ESN framing was necessary at all.

Runs:
  - rescor_esn_uniform at K=8, 16, 32
  - rescor_mr         at K=8, 16, 32 (learned gate)
  - rescor_mr_uniform at K=8, 16, 32

Compare against existing baselines:
  - rescor      (vanilla, 1 CML, 321 params)
  - rescor_esn  at K=8 (learned gate, prior result)

Usage:
    PYTHONUNBUFFERED=1 uv run --with scikit-learn,scipy python experiments/uniform_and_mr_ablation.py
"""
import gc
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

CONFIGS = [
    ("rescor_esn_uniform", 8),
    ("rescor_esn_uniform", 16),
    ("rescor_esn_uniform", 32),
    ("rescor_mr", 8),
    ("rescor_mr", 16),
    ("rescor_mr", 32),
    ("rescor_mr_uniform", 8),
    ("rescor_mr_uniform", 16),
    ("rescor_mr_uniform", 32),
]

SEED = 42


def evaluate_model(model, X_test, Y_test, meta):
    model.eval()
    with torch.no_grad():
        preds = model(X_test)
    if meta["metric"] == "mse":
        return ((preds - Y_test) ** 2).mean().item()
    if meta["loss_type"] in ("ce", "cross_entropy"):
        pred_classes = preds.argmax(dim=1)
        if Y_test.dim() == 4 and Y_test.shape[1] > 1:
            true_classes = Y_test.argmax(dim=1)
        else:
            true_classes = Y_test.long().squeeze(1)
        return (pred_classes == true_classes).float().mean().item()
    pred_binary = (preds > 0.5).float()
    return (pred_binary == Y_test).float().mean().item()


def fmt_score(s, metric):
    if s is None:
        return "FAILED"
    return f"{s:.6e}" if metric == "mse" else f"{s*100:.2f}%"


def run_one(model_name, bench_name, gen_fn, seed, K):
    data = gen_fn(grid_size=16, seed=seed)
    meta = data.meta
    m = create_model(
        model_name,
        in_channels=meta["in_channels"],
        out_channels=meta["out_channels"],
        grid_size=16, seed=seed,
        cml_K=K,
    )
    pc = m.param_count()
    t0 = time.time()
    m = train_model(
        m, data.X_train, data.Y_train,
        X_val=data.X_val, Y_val=data.Y_val,
        loss_type=meta["loss_type"],
        epochs=30, batch_size=64, lr=1e-3,
    )
    s = evaluate_model(m, data.X_test, data.Y_test, meta)
    elapsed = time.time() - t0

    info = m.cml_2d.get_selection_info()
    return {
        "score": s, "metric": meta["metric"],
        "params": pc, "time": elapsed, "K": K,
        "gate_mode": info.get("gate_mode", "?"),
        "gate_top_idx": info.get("top_idx", 0),
        "gate_top_weight": info.get("top_weight", 0.0),
        "gate_entropy": info.get("entropy", 0.0),
        "gate_weights": info.get("weights"),
        "r_values": info.get("r_values"),
        "top_r": info.get("top_r"),
    }


def main():
    results = {}
    for model_name, K in CONFIGS:
        label = f"{model_name}_K{K}"
        print("=" * 80)
        print(f"{label}")
        print("=" * 80)
        results[label] = {}
        for bn, gf in BENCHMARKS.items():
            try:
                r = run_one(model_name, bn, gf, SEED, K)
                results[label][bn] = r
                gate_detail = f"top=k{r['gate_top_idx']}:{r['gate_top_weight']:.2f}"
                if r.get("top_r") is not None:
                    gate_detail += f" r={r['top_r']:.3f}"
                print(f"  {bn:15s}  {fmt_score(r['score'], r['metric']):>15s}  "
                      f"[{r['time']:.0f}s]  {gate_detail}  H={r['gate_entropy']:.2f}  "
                      f"params={r['params']}")
            except Exception as e:
                print(f"  {bn:15s}  FAILED: {type(e).__name__}: {e}")
                results[label][bn] = {"score": None, "metric": "?", "error": str(e)}
            gc.collect()
        print()

    # Baselines from prior runs
    rescor_ref = {
        "heat": 5.351980e-07, "gol": 0.9532, "gray_scott": 7.107806e-06,
        "ks": 6.016547e-06, "rule110": 0.9693, "wireworld": 0.9826,
    }
    esn_k8_ref = {
        "heat": 1.294373e-06, "gol": 0.9488, "gray_scott": 4.515140e-06,
        "ks": 1.139536e-06, "rule110": 0.9693, "wireworld": 0.9913,
    }

    # Wide comparison table
    print("=" * 160)
    print("SUMMARY — uniform-ESN + multi-r vs rescor / rescor_esn K=8")
    print("=" * 160)
    header = f"{'Benchmark':12s}  {'rescor':>12s}  {'esn(lrn) K=8':>12s}"
    for model_name, K in CONFIGS:
        col = f"{model_name.replace('rescor_','')}_K{K}"
        header += f"  {col:>16s}"
    print(header)
    print("-" * len(header))
    for bn in BENCHMARKS:
        metric = "mse" if bn in ("heat", "gray_scott", "ks") else "bce"
        row = f"{bn:12s}  {fmt_score(rescor_ref[bn], metric):>12s}  {fmt_score(esn_k8_ref[bn], metric):>12s}"
        for model_name, K in CONFIGS:
            label = f"{model_name}_K{K}"
            rb = results.get(label, {}).get(bn, {})
            row += f"  {fmt_score(rb.get('score'), rb.get('metric', metric)):>16s}"
        print(row)

    out_path = Path("experiments/results/uniform_and_mr_ablation.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
