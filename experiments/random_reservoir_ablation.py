"""Random-Reservoir ablation: K frozen CMLs with random coupling kernels.

Tests whether oracle knowledge (hand-picked (eps, beta) candidates + warm-start)
in multi_config_k5 can be fully replaced by random-coupling diversity.

Two variants:
  preserved: logistic f(x)=r*x*(1-x) kept; only coupling kernel randomized per K.
  full:      logistic dropped; ESN-style tanh recurrence on [-1,1]-centered grid.

Both use K=8 frozen reservoirs with distinct RNG seeds for their coupling kernels.
Softmax gate on top, NO warm-start (uniform init). Gradient isolation via no_grad.

Runs only the two new variants on all 6 benchmarks. Compare against existing
rescor + multi_config_k5 numbers (stored/printed separately).

Usage:
    PYTHONUNBUFFERED=1 uv run --with scikit-learn,scipy python experiments/random_reservoir_ablation.py
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

K = 8
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
    return f"{s:.6e}" if metric == "mse" else f"{s*100:.2f}%"


def run_one(model_name, bench_name, gen_fn, seed):
    data = gen_fn(grid_size=16, seed=seed)
    meta = data.meta
    m = create_model(model_name,
                     in_channels=meta["in_channels"],
                     out_channels=meta["out_channels"],
                     grid_size=16, seed=seed,
                     cml_K=K)
    pc = m.param_count()
    t0 = time.time()
    m = train_model(m, data.X_train, data.Y_train,
                    X_val=data.X_val, Y_val=data.Y_val,
                    loss_type=meta["loss_type"],
                    epochs=30, batch_size=64, lr=1e-3)
    s = evaluate_model(m, data.X_test, data.Y_test, meta)
    elapsed = time.time() - t0

    # Gate info
    info = m.cml_2d.get_selection_info()
    weights = info["weights"]
    top_idx = info["top_idx"]
    top_w = info["top_weight"]
    entropy = info["entropy"]

    return {
        "score": s, "metric": meta["metric"],
        "params": pc, "time": elapsed,
        "gate_top_idx": top_idx, "gate_top_weight": top_w,
        "gate_entropy": entropy, "gate_weights": weights,
    }


def main():
    results = {}

    for variant in ("rescor_random_reservoir_preserved",
                    "rescor_random_reservoir_full"):
        label = variant.replace("rescor_random_reservoir_", "")
        print("=" * 70)
        print(f"MODEL: {variant} (K={K}, no warm-start)")
        print("=" * 70)
        results[label] = {}
        for bn, gf in BENCHMARKS.items():
            r = run_one(variant, bn, gf, SEED)
            results[label][bn] = r
            print(f"  {bn:15s}  {fmt_score(r['score'], r['metric']):>15s}  "
                  f"[{r['time']:.0f}s]  top=k{r['gate_top_idx']}:{r['gate_top_weight']:.2f}  "
                  f"H={r['gate_entropy']:.2f}  params={r['params']}")
            gc.collect()
        print()

    # Summary table
    print("=" * 100)
    print("SUMMARY — Random Reservoir K=8 (no oracle candidates, no warm-start)")
    print("=" * 100)
    header = f"{'Benchmark':15s}  {'preserved':>15s}  {'full':>15s}  {'p.top':>8s}  {'p.H':>6s}  {'f.top':>8s}  {'f.H':>6s}"
    print(header)
    print("-" * 100)
    for bn in BENCHMARKS:
        rp = results["preserved"][bn]
        rf = results["full"][bn]
        sp = fmt_score(rp["score"], rp["metric"])
        sf = fmt_score(rf["score"], rf["metric"])
        p_top = f"k{rp['gate_top_idx']}:{rp['gate_top_weight']:.2f}"
        f_top = f"k{rf['gate_top_idx']}:{rf['gate_top_weight']:.2f}"
        print(f"{bn:15s}  {sp:>15s}  {sf:>15s}  {p_top:>8s}  {rp['gate_entropy']:>6.2f}  "
              f"{f_top:>8s}  {rf['gate_entropy']:>6.2f}")

    out_path = Path("experiments/results/random_reservoir_ablation.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
