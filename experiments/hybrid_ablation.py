"""Hybrid MR+ESN ablation: does combining chaos-depth + random-coupling close the GS gap?

Current hero rescor_mr_uniform K=32 beats k5 oracle on heat/gol/ks/wireworld but
loses on gray_scott (which needs spatial-coupling diversity). Hybrid tries to
capture both axes: half K CMLs are r-variants (chaos depth), half are tanh
reservoirs with random coupling (spatial). Uniform 1/K averaging — zero gate.

Runs: rescor_hybrid at K=16, K=32. Compare to rescor, rescor_mr_uniform K=32,
and rescor_esn_uniform K=32 (from prior results).

Usage:
    PYTHONUNBUFFERED=1 uv run --with scikit-learn,scipy python experiments/hybrid_ablation.py
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

K_VALUES = [16, 32]
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


def run_one(bench_name, gen_fn, seed, K):
    data = gen_fn(grid_size=16, seed=seed)
    meta = data.meta
    m = create_model(
        "rescor_hybrid",
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
        "K_mr": info["K_mr"], "K_esn": info["K_esn"],
    }


def main():
    results = {}
    for K in K_VALUES:
        label = f"hybrid_K{K}"
        print("=" * 75)
        print(f"rescor_hybrid K={K}  (K_mr={K//2} + K_esn={K - K//2})")
        print("=" * 75)
        results[label] = {}
        for bn, gf in BENCHMARKS.items():
            try:
                r = run_one(bn, gf, SEED, K)
                results[label][bn] = r
                print(f"  {bn:15s}  {fmt_score(r['score'], r['metric']):>15s}  "
                      f"[{r['time']:.0f}s]  params={r['params']}")
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
    k5_ref = {
        "heat": 8.844725e-08, "gol": 0.9484, "gray_scott": 2.772944e-06,
        "ks": 2.825210e-07, "rule110": 0.9693, "wireworld": 0.9902,
    }
    mr_u_k32_ref = {
        "heat": 4.868374e-09, "gol": 0.9598, "gray_scott": 4.343400e-06,
        "ks": 2.416965e-06, "rule110": 0.9693, "wireworld": 0.9911,
    }
    esn_u_k32_ref = {
        "heat": 2.547518e-06, "gol": 0.9579, "gray_scott": 3.866991e-06,
        "ks": 1.700466e-06, "rule110": 0.9693, "wireworld": 0.9825,
    }

    print("=" * 130)
    print("SUMMARY — rescor_hybrid vs rescor / k5 / mr_uniform K=32 / esn_uniform K=32")
    print("=" * 130)
    header = f"{'Benchmark':14s}  {'rescor':>14s}  {'k5':>14s}  {'mr_u K=32':>14s}  {'esn_u K=32':>14s}  {'hybrid K=16':>14s}  {'hybrid K=32':>14s}"
    print(header)
    print("-" * len(header))
    for bn in BENCHMARKS:
        metric = "mse" if bn in ("heat", "gray_scott", "ks") else "bce"
        rb_h16 = results.get("hybrid_K16", {}).get(bn, {})
        rb_h32 = results.get("hybrid_K32", {}).get(bn, {})
        s_rescor = fmt_score(rescor_ref[bn], metric)
        s_k5 = fmt_score(k5_ref[bn], metric)
        s_mru = fmt_score(mr_u_k32_ref[bn], metric)
        s_esnu = fmt_score(esn_u_k32_ref[bn], metric)
        s_h16 = fmt_score(rb_h16.get("score"), rb_h16.get("metric", metric))
        s_h32 = fmt_score(rb_h32.get("score"), rb_h32.get("metric", metric))
        print(f"{bn:14s}  {s_rescor:>14s}  {s_k5:>14s}  {s_mru:>14s}  {s_esnu:>14s}  {s_h16:>14s}  {s_h32:>14s}")

    out_path = Path("experiments/results/hybrid_ablation.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
