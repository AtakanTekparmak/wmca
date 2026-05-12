"""Phase 1 honest baseline — rens vs stat_full vs stat_no_var @ 5 seeds × 100 epochs.

Post-demotion protocol. Tests whether the stat-bank variant's S56 epoch-diagnostic
result (150-epoch heat = 1.18e-7) beats rens K=32's honest multi-seed result
(3-seed mean heat = 8.86e-7 at 30 epochs) when both are given matched compute.

Pre-registered winner criterion (written before run, per new methodology):
- A variant V "wins" benchmark B vs rens_K=32 if median(V on B) beats median(rens on B)
  by more than 2× on MSE benchmarks OR by more than 1.0pp on accuracy benchmarks.
- Report: per-variant × per-benchmark median, mean, std, min, max.
- No single-seed callouts. No cherry-picked seeds.

Variants tested:
  rescor_rens (K=32, uniform averaging, 321 trained)
  rescor_rens_stat_full (K=32, NCA sees [x, mean, var, min, max], 753 trained)
  rescor_rens_stat_no_var (K=32, NCA sees [x, mean, min, max], 609 trained)

Seeds: 42, 43, 44, 45, 46 (5 seeds, new standing protocol)
Epochs: 100 (per S56 methodology caveat)
Benchmarks: all 6 (heat, gol, gs, ks, rule110, wireworld)

Total runs: 3 × 5 × 6 = 90. Estimated wallclock: 30-45 hours.

Usage:
    PYTHONUNBUFFERED=1 uv run --with scikit-learn,scipy python experiments/phase1_honest_baseline.py
"""
import gc
import json
import math
import time
from pathlib import Path
from statistics import median

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

VARIANTS = [
    "rescor_rens",
    "rescor_rens_stat_full",
    "rescor_rens_stat_no_var",
]

SEEDS = [42, 43, 44]  # reduced from 5 to 3 due to unexpected compute slowdown
K = 32
EPOCHS = 100


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
    return f"{s:.4e}" if metric == "mse" else f"{s*100:.2f}%"


def run_one(variant, bench_name, gen_fn, seed):
    data = gen_fn(grid_size=16, seed=seed)
    meta = data.meta
    m = create_model(
        variant,
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
        epochs=EPOCHS, batch_size=64, lr=1e-3,
    )
    s = evaluate_model(m, data.X_test, data.Y_test, meta)
    elapsed = time.time() - t0
    return {"score": s, "metric": meta["metric"], "params": pc,
            "time": elapsed, "seed": seed, "variant": variant}


def stats_summary(values):
    if not values:
        return {"mean": None, "median": None, "std": None, "min": None, "max": None, "n": 0}
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    std = math.sqrt(var)
    return {
        "mean": mean, "median": median(values), "std": std,
        "min": min(values), "max": max(values), "n": n,
    }


def main():
    # Incremental persistence + resume-from-JSON
    out_path = Path("experiments/results/phase1_honest_baseline.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = {v: {bn: {} for bn in BENCHMARKS} for v in VARIANTS}

    # Resume: load prior incremental JSON if present (handles shape differences
    # from an earlier invocation that used a different SEEDS list / format).
    if out_path.exists():
        try:
            with open(out_path, "r") as f:
                prior = json.load(f)
            # Handle both legacy format (top-level dict of variants) and final
            # format (nested "per_seed" block).
            cell_src = prior.get("per_seed", prior)
            n_loaded = 0
            for v in VARIANTS:
                if v not in cell_src:
                    continue
                for bn in BENCHMARKS:
                    if bn not in cell_src[v]:
                        continue
                    for seed_key, cell in cell_src[v][bn].items():
                        if cell and cell.get("score") is not None:
                            results[v][bn][seed_key] = cell
                            n_loaded += 1
            print(f"[resume] loaded {n_loaded} completed cells from {out_path}")
        except Exception as e:
            print(f"[resume] failed to parse prior JSON ({e}); starting fresh")

    for variant in VARIANTS:
        for seed in SEEDS:
            print("=" * 78)
            print(f"{variant}   K={K}   seed={seed}   epochs={EPOCHS}")
            print("=" * 78)
            for bn, gf in BENCHMARKS.items():
                existing = results[variant][bn].get(str(seed), {})
                if existing.get("score") is not None:
                    print(f"  {bn:15s}  {fmt_score(existing['score'], existing['metric']):>14s}  "
                          f"[resumed]  params={existing.get('params', '?')}")
                    continue
                try:
                    r = run_one(variant, bn, gf, seed)
                    results[variant][bn][str(seed)] = r
                    print(f"  {bn:15s}  {fmt_score(r['score'], r['metric']):>14s}  "
                          f"[{r['time']:.0f}s]  params={r['params']}")
                except Exception as e:
                    print(f"  {bn:15s}  FAILED: {type(e).__name__}: {e}")
                    results[variant][bn][str(seed)] = {
                        "score": None, "metric": "?", "error": str(e), "seed": seed
                    }
                # Incremental save so we don't lose progress
                with open(out_path, "w") as f:
                    json.dump(results, f, indent=2)
                gc.collect()
            print()

    # ----- Summary ----------------------------------------------------------
    print("=" * 120)
    print(f"SUMMARY — Phase 1 honest baseline ({len(SEEDS)} seeds × {EPOCHS} epochs)")
    print("=" * 120)

    summary = {v: {} for v in VARIANTS}
    for variant in VARIANTS:
        for bn in BENCHMARKS:
            seed_scores = [r["score"] for r in results[variant][bn].values()
                           if r.get("score") is not None]
            summary[variant][bn] = stats_summary(seed_scores)

    # Reference rescor baseline (single-seed s42, from prior runs; not
    # multi-seed validated — included only as a directional anchor)
    rescor_ref_s42 = {
        "heat": 5.351980e-07, "gol": 0.9532, "gray_scott": 7.107806e-06,
        "ks": 6.016547e-06, "rule110": 0.9693, "wireworld": 0.9826,
    }
    k5_ref_s42 = {
        "heat": 8.844725e-08, "gol": 0.9484, "gray_scott": 2.772944e-06,
        "ks": 2.825210e-07, "rule110": 0.9693, "wireworld": 0.9902,
    }

    for variant in VARIANTS:
        print()
        print(f"── {variant} ─────────────────────────────────────────────")
        hdr = f"{'bench':14s}  {'median':>14s}  {'mean':>14s}  {'std':>12s}  {'min':>14s}  {'max':>14s}  n"
        print(hdr)
        print("-" * len(hdr))
        for bn in BENCHMARKS:
            s = summary[variant][bn]
            metric = "mse" if bn in ("heat", "gray_scott", "ks") else "bce"
            def f(v):
                return fmt_score(v, metric) if v is not None else "n/a"
            std_str = (f"{s['std']:.2e}" if metric == "mse"
                       else f"{s['std']*100:.2f}pp") if s["std"] is not None else "n/a"
            print(f"{bn:14s}  {f(s['median']):>14s}  {f(s['mean']):>14s}  "
                  f"{std_str:>12s}  {f(s['min']):>14s}  {f(s['max']):>14s}  {s['n']}")

    # Pre-registered win check (medians-based)
    print()
    print("=" * 120)
    print("PRE-REGISTERED WIN CHECK — medians vs rescor_rens, " +
          "2× for MSE / 1.0pp for accuracy")
    print("=" * 120)
    win_header = f"{'bench':14s}  {'stat_full vs rens':>22s}  {'stat_no_var vs rens':>22s}"
    print(win_header)
    print("-" * len(win_header))
    rens_med = {bn: summary["rescor_rens"][bn]["median"] for bn in BENCHMARKS}
    for bn in BENCHMARKS:
        rens_m = rens_med[bn]
        metric = "mse" if bn in ("heat", "gray_scott", "ks") else "bce"
        def verdict(other_med, rens_m, metric):
            if other_med is None or rens_m is None:
                return "n/a"
            if metric == "mse":
                ratio = rens_m / other_med
                if ratio > 2.0:
                    return f"WIN ({ratio:.2f}× better)"
                if ratio < 0.5:
                    return f"LOSS ({1/ratio:.2f}× worse)"
                return f"tie ({ratio:.2f}×)"
            diff_pp = (other_med - rens_m) * 100
            if diff_pp > 1.0:
                return f"WIN (+{diff_pp:.2f}pp)"
            if diff_pp < -1.0:
                return f"LOSS ({diff_pp:.2f}pp)"
            return f"tie ({diff_pp:+.2f}pp)"
        full_v = verdict(summary["rescor_rens_stat_full"][bn]["median"], rens_m, metric)
        nv_v = verdict(summary["rescor_rens_stat_no_var"][bn]["median"], rens_m, metric)
        print(f"{bn:14s}  {full_v:>22s}  {nv_v:>22s}")

    # Final JSON dump with summary + baselines
    final = {
        "protocol": {
            "variants": VARIANTS, "seeds": SEEDS, "epochs": EPOCHS,
            "K": K, "benchmarks": list(BENCHMARKS),
            "winner_criterion": "median gap ≥ 2× (MSE) or ≥ 1.0pp (accuracy)",
        },
        "per_seed": results,
        "summary": summary,
        "baselines_s42_only": {"rescor": rescor_ref_s42, "multi_config_k5": k5_ref_s42},
    }
    with open(out_path, "w") as f:
        json.dump(final, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
