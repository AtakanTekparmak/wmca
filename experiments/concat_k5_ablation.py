"""Concat K=5 ablation: multi_config with CONCATENATION instead of softmax blend.

Tests the diversity hypothesis cleanly. Same 5 CMLs as multi_config_k5, but the
NCA sees all 5 outputs as input channels (no gating). If concat beats the
softmax-blend version, the wins come from feature diversity, not learned
selection. If concat loses, gating is doing something useful.

Baseline comparison:
  rescor                        (single CML, 321 params)
  rescor_multi_config_k5        (softmax blend, K=5 candidates, 326 params)
  rescor_multi_config_concat_k5 (concat, K=5 candidates, ~947 params)

Usage:
    PYTHONUNBUFFERED=1 uv run --with scikit-learn,scipy python experiments/concat_k5_ablation.py
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

K5_CANDIDATES = [
    (0.05, 0.15),   # heat-optimal
    (0.15, 0.01),   # KS-optimal
    (0.15, 0.15),   # Gray-Scott-optimal
    (0.30, 0.15),   # default
    (0.50, 0.30),   # discrete-CA-optimal
]


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
    return f"{s:.6e}" if metric == "mse" else f"{s*100:.2f}%"


def run_one(model_name, bench_name, gen_fn, seed, **extra_kwargs):
    data = gen_fn(grid_size=16, seed=seed)
    meta = data.meta
    m = create_model(model_name,
                     in_channels=meta["in_channels"],
                     out_channels=meta["out_channels"],
                     grid_size=16, seed=seed,
                     **extra_kwargs)
    pc = m.param_count()
    t0 = time.time()
    m = train_model(m, data.X_train, data.Y_train,
                    X_val=data.X_val, Y_val=data.Y_val,
                    loss_type=meta["loss_type"],
                    epochs=30, batch_size=64, lr=1e-3)
    s = evaluate_model(m, data.X_test, data.Y_test, meta)
    elapsed = time.time() - t0
    return {"score": s, "metric": meta["metric"], "params": pc, "time": elapsed}


def main():
    seed = 42
    results = {}

    print("=" * 60)
    print("rescor (baseline)")
    print("=" * 60)
    results["rescor"] = {}
    for bn, gf in BENCHMARKS.items():
        r = run_one("rescor", bn, gf, seed)
        results["rescor"][bn] = r
        print(f"  {bn:15s}  {fmt_score(r['score'], r['metric']):>15s}  "
              f"[{r['time']:.0f}s]  {r['params']}")
        gc.collect()

    print("\n" + "=" * 60)
    print("rescor_multi_config_concat_k5 (concat, no gate)")
    print("=" * 60)
    results["concat_k5"] = {}
    for bn, gf in BENCHMARKS.items():
        r = run_one("rescor", bn, gf, seed,
                    cml_gate="multi_config_concat",
                    cml_candidates=K5_CANDIDATES)
        results["concat_k5"][bn] = r
        print(f"  {bn:15s}  {fmt_score(r['score'], r['metric']):>15s}  "
              f"[{r['time']:.0f}s]  {r['params']}")
        gc.collect()

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Benchmark':15s}  {'rescor':>15s}  {'concat_k5':>15s}  {'winner':>20s}")
    print("-" * 75)
    for bn in BENCHMARKS:
        rb = results["rescor"][bn]
        rc = results["concat_k5"][bn]
        base_s = fmt_score(rb["score"], rb["metric"])
        c_s = fmt_score(rc["score"], rc["metric"])
        if rb["metric"] == "mse":
            ratio = rb["score"] / max(rc["score"], 1e-15)
            winner = "concat" if rc["score"] < rb["score"] else "base"
        else:
            ratio = rc["score"] / max(rb["score"], 1e-15)
            winner = "concat" if rc["score"] > rb["score"] else "base"
        print(f"{bn:15s}  {base_s:>15s}  {c_s:>15s}  {winner} ({ratio:.2f}x)")

    out_path = Path("experiments/results/concat_k5_ablation.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
