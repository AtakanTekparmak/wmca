"""rescor_esn K-scaling ablation: does frozen random-reservoir diversity scale?

Tests the scale-pilled thesis for rescor_esn (A-full):
  - Add more frozen tanh reservoirs (K=16, K=32) with distinct random couplings
  - Same NCA (~321 params) + K softmax logits
  - No oracle, no warm-start, no per-sample routing (learned from the D negative result)

Baselines (from prior runs, do NOT re-run):
  - rescor:    vanilla CML (321 params, 12 frozen)
  - esn K=8:   existing A-full result (329 params, 75 frozen)

New this run:
  - esn K=16:  337 trained params (321 NCA + 16 logits), 147 frozen
  - esn K=32:  353 trained params (321 NCA + 32 logits), 291 frozen

Usage:
    PYTHONUNBUFFERED=1 uv run --with scikit-learn,scipy python experiments/esn_k_scaling_ablation.py
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
        "rescor_esn",
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
        "gate_top_idx": info["top_idx"],
        "gate_top_weight": info["top_weight"],
        "gate_entropy": info["entropy"],
        "gate_weights": info["weights"],
    }


def main():
    results = {}
    for K in K_VALUES:
        label = f"K{K}"
        print("=" * 75)
        print(f"rescor_esn — K={K} (tanh reservoirs, random coupling, no oracle)")
        print("=" * 75)
        results[label] = {}
        for bn, gf in BENCHMARKS.items():
            try:
                r = run_one(bn, gf, SEED, K)
                results[label][bn] = r
                print(f"  {bn:15s}  {fmt_score(r['score'], r['metric']):>15s}  "
                      f"[{r['time']:.0f}s]  top=k{r['gate_top_idx']}:{r['gate_top_weight']:.2f}  "
                      f"H={r['gate_entropy']:.2f}  params={r['params']}")
            except Exception as e:
                print(f"  {bn:15s}  FAILED: {type(e).__name__}: {e}")
                results[label][bn] = {"score": None, "metric": "?", "error": str(e)}
            gc.collect()
        print()

    # Comparison table (rescor, k5 oracle, esn K=8, K=16, K=32)
    rescor_ref = {
        "heat": 5.351980e-07, "gol": 0.9532, "gray_scott": 7.107806e-06,
        "ks": 6.016547e-06, "rule110": 0.9693, "wireworld": 0.9826,
    }
    k5_ref = {
        "heat": 8.844725e-08, "gol": 0.9484, "gray_scott": 2.772944e-06,
        "ks": 2.825210e-07, "rule110": 0.9693, "wireworld": 0.9902,
    }
    esn_k8_ref = {
        "heat": 1.294373e-06, "gol": 0.9488, "gray_scott": 4.515140e-06,
        "ks": 1.139536e-06, "rule110": 0.9693, "wireworld": 0.9913,
    }

    print("=" * 115)
    print("SCALE-PILLED SUMMARY — rescor_esn at K=8, 16, 32 vs rescor / k5")
    print("=" * 115)
    print(f"{'Benchmark':15s}  {'rescor':>15s}  {'k5':>15s}  {'esn K=8':>15s}  {'esn K=16':>15s}  {'esn K=32':>15s}")
    print("-" * 115)
    for bn in BENCHMARKS:
        metric = "mse" if bn in ("heat", "gray_scott", "ks") else "bce"
        s_rescor = fmt_score(rescor_ref[bn], metric)
        s_k5 = fmt_score(k5_ref[bn], metric)
        s_k8 = fmt_score(esn_k8_ref[bn], metric)
        k16_r = results.get("K16", {}).get(bn, {})
        k32_r = results.get("K32", {}).get(bn, {})
        s_k16 = fmt_score(k16_r.get("score"), k16_r.get("metric", metric))
        s_k32 = fmt_score(k32_r.get("score"), k32_r.get("metric", metric))
        print(f"{bn:15s}  {s_rescor:>15s}  {s_k5:>15s}  {s_k8:>15s}  {s_k16:>15s}  {s_k32:>15s}")

    out_path = Path("experiments/results/esn_k_scaling_ablation.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
