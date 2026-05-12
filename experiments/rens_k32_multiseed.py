"""Multi-seed confirmation of rescor_rens K=32 hero across all 6 benchmarks.

Tests whether the 18× heat-over-oracle and 99.11% wireworld results hold up
across seeds {42, 43, 44}. Single-seed wins on chaotic benchmarks are suspect;
this is the insurance run before any paper/fork claims.

For each benchmark, reports: mean, std, min, max, individual seeds.

Usage:
    PYTHONUNBUFFERED=1 uv run --with scikit-learn,scipy python experiments/rens_k32_multiseed.py
"""
import gc
import json
import time
from pathlib import Path
import math

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

SEEDS = [42, 43, 44]
K = 32
MODEL = "rescor_rens"  # alias for rescor_mr_uniform


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


def run_one(bench_name, gen_fn, seed):
    data = gen_fn(grid_size=16, seed=seed)
    meta = data.meta
    m = create_model(
        MODEL,
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
    return {"score": s, "metric": meta["metric"], "params": pc, "time": elapsed, "seed": seed}


def mean_std(values):
    if not values:
        return float("nan"), float("nan")
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return mean, math.sqrt(var)


def main():
    results = {bn: {} for bn in BENCHMARKS}

    for seed in SEEDS:
        print("=" * 75)
        print(f"rescor_rens K={K}  seed={seed}")
        print("=" * 75)
        for bn, gf in BENCHMARKS.items():
            try:
                r = run_one(bn, gf, seed)
                results[bn][str(seed)] = r
                print(f"  {bn:15s}  {fmt_score(r['score'], r['metric']):>15s}  "
                      f"[{r['time']:.0f}s]  params={r['params']}")
            except Exception as e:
                print(f"  {bn:15s}  FAILED: {type(e).__name__}: {e}")
                results[bn][str(seed)] = {"score": None, "metric": "?", "error": str(e)}
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

    print("=" * 115)
    print(f"SUMMARY — rescor_rens K={K} multi-seed confirmation (3 seeds)")
    print("=" * 115)
    header = f"{'Benchmark':14s}  {'rescor (s42)':>14s}  {'k5 (s42)':>14s}  {'s42':>14s}  {'s43':>14s}  {'s44':>14s}  {'mean':>14s}  {'std':>12s}"
    print(header)
    print("-" * len(header))
    summary = {}
    for bn in BENCHMARKS:
        metric = "mse" if bn in ("heat", "gray_scott", "ks") else "bce"
        seed_scores = []
        row_cells = []
        for seed in SEEDS:
            rb = results[bn].get(str(seed), {})
            score = rb.get("score")
            if score is not None:
                seed_scores.append(score)
            row_cells.append(fmt_score(score, metric))
        mean, std = mean_std(seed_scores)
        summary[bn] = {"scores": seed_scores, "mean": mean, "std": std, "metric": metric}
        mean_str = fmt_score(mean, metric) if not math.isnan(mean) else "n/a"
        if metric == "mse":
            std_str = f"{std:.2e}"
        else:
            std_str = f"{std*100:.3f}pp"
        print(f"{bn:14s}  {fmt_score(rescor_ref[bn], metric):>14s}  "
              f"{fmt_score(k5_ref[bn], metric):>14s}  "
              f"{row_cells[0]:>14s}  {row_cells[1]:>14s}  {row_cells[2]:>14s}  "
              f"{mean_str:>14s}  {std_str:>12s}")

    out_path = Path("experiments/results/rens_k32_multiseed.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"per_seed": results, "summary": summary,
                   "baselines": {"rescor_s42": rescor_ref, "k5_s42": k5_ref}}, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
