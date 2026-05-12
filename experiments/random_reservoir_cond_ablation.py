"""A-full + input-conditioned gate (D): does per-sample routing help?

Same as rescor_random_reservoir_full (K=8 random tanh reservoirs, no logistic)
but the softmax gate is now input-conditioned: a tiny 2-layer MLP maps
per-sample (mean, var, grad-norm) statistics to K perturbation logits which
are added to a learnable global bias.

Hypernet params: 3 -> 8 hidden -> K = ~104 extra params.
Total: 329 + ~104 = 433 trained.

Tests: does per-sample routing close the 14.5x heat gap vs k5? If yes, the
"gate never commits" story from the non-conditioned variants was a routing
problem, not a diversity problem.

Usage:
    PYTHONUNBUFFERED=1 uv run --with scikit-learn,scipy python experiments/random_reservoir_cond_ablation.py
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


def run_one(bench_name, gen_fn, seed):
    data = gen_fn(grid_size=16, seed=seed)
    meta = data.meta
    m = create_model(
        "rescor_random_reservoir_full_cond",
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
        "params": pc, "time": elapsed,
        "gate_global_top_idx": info["top_idx"],
        "gate_global_top_weight": info["top_weight"],
        "gate_global_entropy": info["entropy"],
        "gate_global_weights": info["weights"],
    }


def main():
    print("=" * 75)
    print("A-full + conditioned gate (D) — K=8 tanh reservoirs, hypernet routing")
    print("=" * 75)

    results = {}
    for bn, gf in BENCHMARKS.items():
        try:
            r = run_one(bn, gf, SEED)
            results[bn] = r
            print(f"  {bn:15s}  {fmt_score(r['score'], r['metric']):>15s}  "
                  f"[{r['time']:.0f}s]  global_top=k{r['gate_global_top_idx']}:{r['gate_global_top_weight']:.2f}  "
                  f"H={r['gate_global_entropy']:.2f}  params={r['params']}")
        except Exception as e:
            print(f"  {bn:15s}  FAILED: {type(e).__name__}: {e}")
            results[bn] = {"score": None, "metric": "?", "error": str(e)}
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
    print("SUMMARY — A-full + D (input-conditioned gate) vs rescor / k5 / A-full")
    print("=" * 95)
    print(f"{'Benchmark':15s}  {'rescor':>15s}  {'k5':>15s}  {'A-full':>15s}  {'A-full+D':>15s}")
    print("-" * 95)
    for bn in BENCHMARKS:
        rb = results.get(bn, {})
        metric = rb.get("metric", "?")
        if rb.get("score") is None:
            s_d = "FAILED"
        else:
            s_d = fmt_score(rb["score"], metric)
        # Fall back to a valid metric for pretty-printing the baselines
        bench_metric = "mse" if bn in ("heat", "gray_scott", "ks") else "bce"
        s_rescor = fmt_score(rescor_ref[bn], bench_metric)
        s_k5 = fmt_score(k5_ref[bn], bench_metric)
        s_afull = fmt_score(a_full_ref[bn], bench_metric)
        print(f"{bn:15s}  {s_rescor:>15s}  {s_k5:>15s}  {s_afull:>15s}  {s_d:>15s}")

    out_path = Path("experiments/results/random_reservoir_cond_ablation.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
