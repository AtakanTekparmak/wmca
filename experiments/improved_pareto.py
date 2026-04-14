"""Improved Pareto plots from unified ablation results.

Creates two plots:
  1. Aggregated Pareto: one big plot, normalised performance averaged
     across all benchmarks, per model.
  2. Per-benchmark Pareto: 2x4 grid (7 benchmarks + 1 legend panel), one
     scatter per benchmark, with Pareto frontier.

Run with:
    FORCE_CPU=1 uv run --with matplotlib python experiments/improved_pareto.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = PROJECT_ROOT / "experiments" / "results" / "unified_ablation.json"
PLOTS_DIR = PROJECT_ROOT / "experiments" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Benchmarks where the primary metric is accuracy (higher = better) vs MSE (lower = better)
ACCURACY_BENCHMARKS = {"gol", "rule110", "wireworld", "grid_world"}

# Expected canonical order for the 2x4 grid layout
CANONICAL_BENCHMARKS = [
    "heat", "gol", "ks", "gray_scott",
    "rule110", "wireworld", "grid_world",
]

# Colorblind-friendly palette (Wong 2011, https://www.nature.com/articles/nmeth.1618)
CB_PALETTE = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#D55E00",  # vermillion
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
]

MARKERS = ["o", "s", "D", "^", "v", "P", "X", "*"]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_results() -> tuple[dict, dict]:
    with open(RESULTS_PATH, "r") as f:
        data = json.load(f)
    return data.get("results", {}), data.get("config", {})


def primary_metric(benchmark: str, one_step: dict) -> float | None:
    """Return the primary one-step metric for a benchmark.

    For accuracy benchmarks -> 'accuracy' (higher is better).
    For MSE benchmarks      -> 'MSE' / first key (lower is better).
    """
    if not one_step:
        return None
    if benchmark in ACCURACY_BENCHMARKS:
        if "accuracy" in one_step:
            return float(one_step["accuracy"])
        # fall back to first numeric value
    # deterministic: use first key that is a number
    for v in one_step.values():
        if isinstance(v, (int, float)):
            return float(v)
    return None


def is_higher_better(benchmark: str) -> bool:
    return benchmark in ACCURACY_BENCHMARKS


def trained_params(model_entry: dict) -> int | None:
    pc = model_entry.get("params", {})
    t = pc.get("trained")
    if t is None:
        return None
    return int(t)


def pareto_frontier(xs, ys, maximize_y: bool):
    """Indices of Pareto-optimal points.

    X is always minimised (fewer params = better).
    Y is minimised if maximize_y=False, maximised if maximize_y=True.
    Returns indices sorted by x ascending.
    """
    order = sorted(range(len(xs)), key=lambda i: (xs[i], -ys[i] if maximize_y else ys[i]))
    frontier = []
    if maximize_y:
        best = -math.inf
        for i in order:
            if ys[i] > best:
                frontier.append(i)
                best = ys[i]
    else:
        best = math.inf
        for i in order:
            if ys[i] < best:
                frontier.append(i)
                best = ys[i]
    return frontier


def normalize_score(value: float, worst: float, best: float) -> float:
    """Map raw metric to [0, 1] where 1 = best, 0 = worst."""
    if best == worst:
        return 1.0
    return (value - worst) / (best - worst)


def collect_all_models(results: dict) -> list[str]:
    seen: list[str] = []
    seen_set: set[str] = set()
    for model_map in results.values():
        for m in model_map:
            if m not in seen_set:
                seen.append(m)
                seen_set.add(m)
    return seen


def style_for(idx: int) -> dict:
    return dict(
        color=CB_PALETTE[idx % len(CB_PALETTE)],
        marker=MARKERS[idx % len(MARKERS)],
    )


# ---------------------------------------------------------------------------
# Plot 1: aggregated Pareto (normalised score averaged across benchmarks)
# ---------------------------------------------------------------------------
def plot_aggregated(results: dict, out_path: Path) -> None:
    models = collect_all_models(results)
    if not models:
        print("No models found in results; skipping aggregated plot.")
        return

    # For each benchmark, find best/worst raw metric to normalise against
    bench_bounds: dict[str, tuple[float, float]] = {}  # name -> (worst, best)
    bench_higher_better: dict[str, bool] = {}
    for bname, mr in results.items():
        vals = []
        for m, entry in mr.items():
            v = primary_metric(bname, entry.get("one_step", {}))
            if v is not None and not math.isnan(v):
                vals.append(v)
        if not vals:
            continue
        higher = is_higher_better(bname)
        if higher:
            worst = min(vals)
            best = max(vals)
        else:
            worst = max(vals)
            best = min(vals)
        bench_bounds[bname] = (worst, best)
        bench_higher_better[bname] = higher

    # For each model, average normalised score and average trained params
    agg_scores: dict[str, float] = {}
    agg_params: dict[str, int] = {}
    for model in models:
        norm_scores = []
        param_vals = []
        for bname, (worst, best) in bench_bounds.items():
            entry = results[bname].get(model)
            if entry is None:
                continue
            v = primary_metric(bname, entry.get("one_step", {}))
            if v is None or math.isnan(v):
                continue
            norm_scores.append(normalize_score(v, worst, best))
            t = trained_params(entry)
            if t is not None and t > 0:
                param_vals.append(t)
        if not norm_scores or not param_vals:
            continue
        agg_scores[model] = float(np.mean(norm_scores))
        # Use mean trained params across benchmarks (usually constant per model)
        agg_params[model] = int(round(float(np.mean(param_vals))))

    if not agg_scores:
        print("No aggregated points to plot.")
        return

    fig, ax = plt.subplots(figsize=(11, 7.5))

    xs, ys, names = [], [], []
    for i, model in enumerate(models):
        if model not in agg_scores:
            continue
        p = agg_params[model]
        s = agg_scores[model]
        st = style_for(i)
        ax.scatter(
            p, s,
            s=220, zorder=4,
            color=st["color"], marker=st["marker"],
            edgecolors="black", linewidths=1.2,
            label=model,
        )
        # Offset label slightly above & right
        ax.annotate(
            model, (p, s),
            xytext=(9, 6), textcoords="offset points",
            fontsize=11, fontweight="bold", zorder=5,
        )
        xs.append(p)
        ys.append(s)
        names.append(model)

    # Pareto frontier: minimise params, maximise score
    if len(xs) >= 2:
        pf = pareto_frontier(xs, ys, maximize_y=True)
        pf_sorted = sorted(pf, key=lambda i: xs[i])
        ax.plot(
            [xs[i] for i in pf_sorted],
            [ys[i] for i in pf_sorted],
            color="black", linestyle="--", linewidth=1.6,
            alpha=0.55, zorder=2, label="Pareto frontier",
        )

    ax.set_xscale("log")
    ax.set_xlabel("Trainable parameters (log scale)", fontsize=13)
    ax.set_ylabel("Normalised performance  (1 = best, 0 = worst)", fontsize=13)
    ax.set_title(
        f"Parameter Efficiency (averaged across {len(bench_bounds)} benchmark"
        f"{'s' if len(bench_bounds) != 1 else ''})",
        fontsize=15,
    )
    ax.set_ylim(-0.08, 1.12)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="lower right", fontsize=10, frameon=True, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Plot 2: per-benchmark Pareto (2x4 grid)
# ---------------------------------------------------------------------------
def plot_per_benchmark(results: dict, out_path: Path) -> None:
    models = collect_all_models(results)
    if not models:
        print("No models found in results; skipping per-benchmark plot.")
        return

    model_style = {m: style_for(i) for i, m in enumerate(models)}

    # Preserve canonical order first, then append any extras
    benchmark_order = [b for b in CANONICAL_BENCHMARKS if b in results]
    for b in results:
        if b not in benchmark_order:
            benchmark_order.append(b)

    n_rows, n_cols = 2, 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 10))
    axes_flat = axes.flatten()

    for slot in range(n_rows * n_cols):
        ax = axes_flat[slot]
        # Last cell reserved for legend
        if slot >= len(benchmark_order) or slot == n_rows * n_cols - 1:
            ax.axis("off")
            continue

        bname = benchmark_order[slot]
        mr = results.get(bname, {})
        higher = is_higher_better(bname)

        xs, ys = [], []
        for model in models:
            entry = mr.get(model)
            if entry is None:
                continue
            v = primary_metric(bname, entry.get("one_step", {}))
            t = trained_params(entry)
            if v is None or t is None or t <= 0 or math.isnan(v):
                continue
            st = model_style[model]
            ax.scatter(
                t, v,
                s=150, zorder=4,
                color=st["color"], marker=st["marker"],
                edgecolors="black", linewidths=1.0,
            )
            xs.append(t)
            ys.append(v)

        if not xs:
            ax.set_title(f"{bname} (no data)", fontsize=12)
            ax.set_xscale("log")
            continue

        # Pareto frontier
        if len(xs) >= 2:
            pf = pareto_frontier(xs, ys, maximize_y=higher)
            pf_sorted = sorted(pf, key=lambda i: xs[i])
            ax.plot(
                [xs[i] for i in pf_sorted],
                [ys[i] for i in pf_sorted],
                color="black", linestyle="--", linewidth=1.4,
                alpha=0.55, zorder=2,
            )

        ax.set_xscale("log")
        if not higher and min(ys) > 0:
            ax.set_yscale("log")

        metric_label = "Accuracy (higher = better)" if higher else "MSE / CE (lower = better)"
        ax.set_xlabel("Trainable params (log)", fontsize=11)
        ax.set_ylabel(metric_label, fontsize=11)
        ax.set_title(bname, fontsize=13)
        ax.grid(True, which="both", alpha=0.25)

    # Shared legend in the bottom-right panel
    legend_ax = axes_flat[-1]
    handles = []
    for model in models:
        st = model_style[model]
        handles.append(
            Line2D(
                [0], [0],
                marker=st["marker"], color="white",
                markerfacecolor=st["color"], markeredgecolor="black",
                markersize=13, linewidth=0, label=model,
            )
        )
    handles.append(
        Line2D(
            [0], [0],
            color="black", linestyle="--", linewidth=1.6,
            alpha=0.6, label="Pareto frontier",
        )
    )
    legend_ax.legend(
        handles=handles,
        loc="center",
        fontsize=12,
        frameon=True,
        framealpha=0.9,
        title="Models",
        title_fontsize=13,
    )

    fig.suptitle("Parameter Efficiency per Benchmark", fontsize=17, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    if not RESULTS_PATH.exists():
        raise SystemExit(f"Results file not found: {RESULTS_PATH}")
    results, _ = load_results()
    if not results:
        raise SystemExit(f"No results in {RESULTS_PATH}")

    print(f"Loaded {len(results)} benchmark(s): {', '.join(results.keys())}")
    models = collect_all_models(results)
    print(f"Found {len(models)} model(s): {', '.join(models)}")

    plot_aggregated(results, PLOTS_DIR / "pareto_aggregated.png")
    plot_per_benchmark(results, PLOTS_DIR / "pareto_per_benchmark.png")


if __name__ == "__main__":
    main()
