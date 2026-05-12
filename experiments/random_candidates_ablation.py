"""Variant B (random-k5): K=8 random (eps, beta) pairs, fixed coupling, logistic kept.

This isolates the oracle variable in multi_config_k5. Same architecture as k5
(logistic + fixed 3x3 coupling kernel + detached CML passes + softmax gate),
but the K=8 candidate (eps, beta) pairs are sampled uniformly from
[0, 0.8] x [0, 0.5] instead of being placed at per-benchmark sweep optima.
NO warm-start (uniform gate init).

  - If B matches k5: oracle doesn't matter; random sampling on the same axis works.
  - If B matches A-preserved: oracle is load-bearing; random (eps, beta) is as bad
    as random coupling with the logistic map.

Usage:
    PYTHONUNBUFFERED=1 uv run --with scikit-learn,scipy python experiments/random_candidates_ablation.py
"""
import gc
import json
import time
from pathlib import Path

import numpy as np
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
SAMPLE_SEED = 12345  # fixed seed for candidate sampling — reproducible
EPS_RANGE = (0.0, 0.8)
BETA_RANGE = (0.0, 0.5)

rng = np.random.default_rng(SAMPLE_SEED)
RANDOM_CANDIDATES = [
    (float(rng.uniform(*EPS_RANGE)), float(rng.uniform(*BETA_RANGE)))
    for _ in range(K)
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


def run_one(bench_name, gen_fn, seed):
    data = gen_fn(grid_size=16, seed=seed)
    meta = data.meta
    m = create_model(
        "rescor_multi_config",
        in_channels=meta["in_channels"],
        out_channels=meta["out_channels"],
        grid_size=16, seed=seed,
        cml_candidates=RANDOM_CANDIDATES,
        cml_warm_start_idx=None,
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
    # info["weights"] is dict of "(eps,beta)" -> weight
    if isinstance(info.get("weights"), dict):
        ws = list(info["weights"].items())
        ws.sort(key=lambda kv: -kv[1])
        top_label, top_w = ws[0]
        entropy = -sum(w * np.log(w + 1e-12) for _, w in ws)
    else:
        top_label, top_w, entropy = "?", 0.0, 0.0

    return {
        "score": s, "metric": meta["metric"],
        "params": pc, "time": elapsed,
        "gate_top_label": top_label, "gate_top_weight": top_w,
        "gate_entropy": entropy,
        "gate_weights": info.get("weights"),
        "eff_eps": info.get("effective_eps"),
        "eff_beta": info.get("effective_beta"),
    }


def main():
    print("=" * 75)
    print("Variant B — random-k5 (K=8 random (eps, beta), logistic + fixed coupling)")
    print("=" * 75)
    print(f"Random candidates (seed={SAMPLE_SEED}):")
    for i, (e, b) in enumerate(RANDOM_CANDIDATES):
        print(f"  k{i}: (eps={e:.3f}, beta={b:.3f})")
    print()

    seed = 42
    results = {}
    for bn, gf in BENCHMARKS.items():
        r = run_one(bn, gf, seed)
        results[bn] = r
        print(f"  {bn:15s}  {fmt_score(r['score'], r['metric']):>15s}  "
              f"[{r['time']:.0f}s]  top={r['gate_top_label']}:{r['gate_top_weight']:.2f}  "
              f"H={r['gate_entropy']:.2f}  params={r['params']}")
        gc.collect()

    # Comparison table (rescor, k5, A-full baselines from prior runs)
    rescor_ref = {
        "heat": 5.351980e-07, "gol": 0.9532, "gray_scott": 7.107806e-06,
        "ks": 6.016547e-06, "rule110": 0.9693, "wireworld": 0.9826,
    }
    k5_ref = {
        "heat": 8.844725e-08, "gol": 0.9484, "gray_scott": 2.772944e-06,
        "ks": 2.825210e-07, "rule110": 0.9693, "wireworld": 0.9902,
    }
    a_full_ref = {
        "heat": 1.294373e-06, "gol": 0.9488, "gray_scott": 4.515140e-06,
        "ks": 1.139536e-06, "rule110": 0.9693, "wireworld": 0.9913,
    }

    print("\n" + "=" * 95)
    print("SUMMARY — Variant B vs rescor / k5 / A-full")
    print("=" * 95)
    print(f"{'Benchmark':15s}  {'rescor':>15s}  {'k5':>15s}  {'A-full':>15s}  {'B':>15s}")
    print("-" * 95)
    for bn in BENCHMARKS:
        rb = results[bn]
        metric = rb["metric"]
        s_rescor = fmt_score(rescor_ref[bn], metric)
        s_k5 = fmt_score(k5_ref[bn], metric)
        s_afull = fmt_score(a_full_ref[bn], metric)
        s_b = fmt_score(rb["score"], metric)
        print(f"{bn:15s}  {s_rescor:>15s}  {s_k5:>15s}  {s_afull:>15s}  {s_b:>15s}")

    out_path = Path("experiments/results/random_candidates_ablation.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"candidates": RANDOM_CANDIDATES, "results": results}, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
