"""Stat-bank NCA variance ablation on top of rescor_rens K=32.

Phase 1 of the deeper-NCA plan: does ensemble variance across K=32 reservoirs
carry task-relevant information beyond the mean?

Variants (both built on rescor_rens K=32 frozen reservoir bank):
  - rescor_rens_stat_full:   NCA sees [x, cml_mean, cml_var, cml_min, cml_max]
                             (753 trained params)
  - rescor_rens_stat_no_var: NCA sees [x, cml_mean, cml_min, cml_max]
                             (609 trained params)

Both residual against cml_mean (same as vanilla rescor_rens).

Compare against rescor and rescor_rens K=32 (from prior runs).

Usage:
    PYTHONUNBUFFERED=1 uv run --with scikit-learn,scipy python experiments/stat_bank_ablation.py
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

MODELS = ["rescor_rens_stat_full", "rescor_rens_stat_no_var"]
K = 32
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


def run_one(model_name, bench_name, gen_fn, seed):
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
    return {"score": s, "metric": meta["metric"], "params": pc, "time": elapsed}


def main():
    results = {}
    for model_name in MODELS:
        label = model_name.replace("rescor_rens_stat_", "C_")
        print("=" * 75)
        print(f"{model_name}  (K={K})")
        print("=" * 75)
        results[label] = {}
        for bn, gf in BENCHMARKS.items():
            try:
                r = run_one(model_name, bn, gf, SEED)
                results[label][bn] = r
                print(f"  {bn:15s}  {fmt_score(r['score'], r['metric']):>15s}  "
                      f"[{r['time']:.0f}s]  params={r['params']}")
            except Exception as e:
                print(f"  {bn:15s}  FAILED: {type(e).__name__}: {e}")
                results[label][bn] = {"score": None, "metric": "?", "error": str(e)}
            gc.collect()
        print()

    # Baselines for comparison
    rescor_ref = {
        "heat": 5.351980e-07, "gol": 0.9532, "gray_scott": 7.107806e-06,
        "ks": 6.016547e-06, "rule110": 0.9693, "wireworld": 0.9826,
    }
    k5_ref = {
        "heat": 8.844725e-08, "gol": 0.9484, "gray_scott": 2.772944e-06,
        "ks": 2.825210e-07, "rule110": 0.9693, "wireworld": 0.9902,
    }
    rens_ref = {  # rescor_mr_uniform / rescor_rens K=32
        "heat": 4.868374e-09, "gol": 0.9598, "gray_scott": 4.343400e-06,
        "ks": 2.416965e-06, "rule110": 0.9693, "wireworld": 0.9911,
    }

    print("=" * 115)
    print("SUMMARY — stat-bank variance ablation vs rescor / k5 / rescor_rens K=32")
    print("=" * 115)
    header = f"{'Benchmark':14s}  {'rescor':>14s}  {'k5':>14s}  {'rens K=32':>14s}  {'C_full':>14s}  {'C_no_var':>14s}"
    print(header)
    print("-" * len(header))
    for bn in BENCHMARKS:
        metric = "mse" if bn in ("heat", "gray_scott", "ks") else "bce"
        rb_full = results.get("C_full", {}).get(bn, {})
        rb_nv = results.get("C_no_var", {}).get(bn, {})
        s_rescor = fmt_score(rescor_ref[bn], metric)
        s_k5 = fmt_score(k5_ref[bn], metric)
        s_rens = fmt_score(rens_ref[bn], metric)
        s_full = fmt_score(rb_full.get("score"), rb_full.get("metric", metric))
        s_nv = fmt_score(rb_nv.get("score"), rb_nv.get("metric", metric))
        print(f"{bn:14s}  {s_rescor:>14s}  {s_k5:>14s}  {s_rens:>14s}  {s_full:>14s}  {s_nv:>14s}")

    out_path = Path("experiments/results/stat_bank_ablation.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
