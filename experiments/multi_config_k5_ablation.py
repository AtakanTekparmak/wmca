"""Multi-config CML ablation with K=5 candidates + warm-start init.

The definitive test: given a candidate set that COVERS the sweep optima for
all benchmarks, can the learned gate pick the right candidate per task?

Candidates (K=5) cover the sweep optima:
  0: (0.05, 0.15) -- heat-optimal (weak coupling + default drive)
  1: (0.15, 0.01) -- KS-optimal (weak coupling + free-running)
  2: (0.15, 0.15) -- Gray-Scott-optimal
  3: (0.30, 0.15) -- default (warm-start favored)
  4: (0.50, 0.30) -- discrete-CA-optimal

Warm-start: init logits with +2.0 on the default (idx 3), 0 on others.
softmax(2, 0, 0, 0, 0) puts ~74% on default and ~6.5% on each other.

Three possible outcomes per benchmark:
  (i)  Gate picks correct candidate -> learnable CML works
  (ii) Gate picks wrong candidate even when available -> kill the direction
  (iii) Gate picks correct but regresses elsewhere -> motivates meta-policy

Usage:
    PYTHONUNBUFFERED=1 uv run --with scikit-learn,scipy python experiments/multi_config_k5_ablation.py
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

# K=5 candidates covering all sweep optima
K5_CANDIDATES = [
    (0.05, 0.15),   # heat-optimal
    (0.15, 0.01),   # KS-optimal
    (0.15, 0.15),   # Gray-Scott-optimal
    (0.30, 0.15),   # default (warm-start)
    (0.50, 0.30),   # discrete-CA-optimal
]
WARM_START_IDX = 3  # default candidate

# Sweep-optimal (eps, beta) per benchmark for diagnostic comparison
SWEEP_OPTIMAL = {
    "heat":       (0.05, 0.15),
    "gol":        (0.30, 0.15),
    "gray_scott": (0.15, 0.15),
    "ks":         (0.15, 0.01),
    "rule110":    (0.30, 0.15),
    "wireworld":  (0.30, 0.15),
}


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


def main():
    seed = 42
    results = {}

    # Baseline: vanilla rescor
    print("=" * 60)
    print("MODEL: rescor (baseline)")
    print("=" * 60)
    results["rescor"] = {}
    for bn, gf in BENCHMARKS.items():
        data = gf(grid_size=16, seed=seed)
        meta = data.meta
        m = create_model("rescor",
                         in_channels=meta["in_channels"],
                         out_channels=meta["out_channels"],
                         grid_size=16, seed=seed)
        t0 = time.time()
        m = train_model(m, data.X_train, data.Y_train,
                        X_val=data.X_val, Y_val=data.Y_val,
                        loss_type=meta["loss_type"],
                        epochs=30, batch_size=64, lr=1e-3)
        s = evaluate_model(m, data.X_test, data.Y_test, meta)
        results["rescor"][bn] = {"score": s, "metric": meta["metric"]}
        print(f"  {bn:15s}  {meta['metric']}={fmt_score(s, meta['metric'])}  [{time.time()-t0:.0f}s]")
        del m, data
        gc.collect()

    # K=5 multi_config with warm-start
    print("\n" + "=" * 60)
    print(f"MODEL: rescor_multi_config_k5 (K=5 + warm-start idx {WARM_START_IDX})")
    print(f"Candidates: {K5_CANDIDATES}")
    print("=" * 60)
    results["multi_config_k5"] = {}
    for bn, gf in BENCHMARKS.items():
        data = gf(grid_size=16, seed=seed)
        meta = data.meta
        m = create_model("rescor",
                         in_channels=meta["in_channels"],
                         out_channels=meta["out_channels"],
                         grid_size=16, seed=seed,
                         cml_gate="multi_config",
                         cml_candidates=K5_CANDIDATES,
                         cml_warm_start_idx=WARM_START_IDX,
                         cml_warm_start_logit=2.0)
        pc = m.param_count()
        t0 = time.time()
        m = train_model(m, data.X_train, data.Y_train,
                        X_val=data.X_val, Y_val=data.Y_val,
                        loss_type=meta["loss_type"],
                        epochs=30, batch_size=64, lr=1e-3)
        s = evaluate_model(m, data.X_test, data.Y_test, meta)

        # Extract learned selection
        si = m.cml_2d.get_selection_info() if hasattr(m.cml_2d, "get_selection_info") else {}
        weights = si.get("weights", {})
        # Argmax candidate
        if weights:
            top_cand, top_w = max(weights.items(), key=lambda x: x[1])
        else:
            top_cand, top_w = "?", 0.0

        opt = SWEEP_OPTIMAL.get(bn, ("?", "?"))
        opt_str = f"({opt[0]:.2f},{opt[1]:.2f})"
        correct = top_cand == opt_str
        marker = "OK" if correct else "MISS"

        results["multi_config_k5"][bn] = {
            "score": s,
            "metric": meta["metric"],
            "weights": weights,
            "top_cand": top_cand,
            "top_weight": top_w,
            "sweep_optimal": opt_str,
            "correct": correct,
            "params": pc,
        }
        print(f"  {bn:15s}  {meta['metric']}={fmt_score(s, meta['metric'])}  "
              f"[{time.time()-t0:.0f}s]  argmax={top_cand}:{top_w:.2f}  "
              f"opt={opt_str}  [{marker}]")
        del m, data
        gc.collect()

    # Summary
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print(f"{'Benchmark':15s}  {'rescor':>15s}  {'multi_k5':>15s}  "
          f"{'argmax':>15s}  {'sweep-opt':>15s}  verdict")
    print("-" * 95)
    for bn in BENCHMARKS:
        r_base = results["rescor"][bn]
        r_k5 = results["multi_config_k5"][bn]
        base_s = fmt_score(r_base["score"], r_base["metric"])
        k5_s = fmt_score(r_k5["score"], r_k5["metric"])
        correct = "OK" if r_k5["correct"] else "MISS"
        # Winner
        if r_base["metric"] == "mse":
            winner = "k5" if r_k5["score"] < r_base["score"] else "base"
            ratio = r_base["score"] / max(r_k5["score"], 1e-15)
        else:
            winner = "k5" if r_k5["score"] > r_base["score"] else "base"
            ratio = r_k5["score"] / max(r_base["score"], 1e-15)
        argmax_str = f"{r_k5['top_cand']}:{r_k5['top_weight']:.2f}"
        print(f"{bn:15s}  {base_s:>15s}  {k5_s:>15s}  "
              f"{argmax_str:>15s}  "
              f"{r_k5['sweep_optimal']:>15s}  {winner} ({ratio:.2f}x) [{correct}]")

    out_path = Path("experiments/results/multi_config_k5_ablation.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
