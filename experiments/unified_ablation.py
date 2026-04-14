"""Unified Ablation Runner: all benchmarks x all models + evaluation.

Runs the full (benchmark x model) grid, collects 1-step and rollout metrics,
prints summary tables, saves JSON results and summary plots.

Imports from two modules being created in parallel:
  - wmca.benchmarks   (BENCHMARKS dict, BenchmarkData dataclass)
  - wmca.model_registry (create_model, train_model, evaluate_model,
                          evaluate_rollout, param_count, MODEL_REGISTRY)

Usage:
    # Run everything
    uv run --with scikit-learn,matplotlib,scipy python experiments/unified_ablation.py --no-wandb

    # Run specific benchmarks
    uv run --with scikit-learn,matplotlib,scipy python experiments/unified_ablation.py --benchmarks heat gol ks --no-wandb

    # Run specific models
    uv run --with scikit-learn,matplotlib,scipy python experiments/unified_ablation.py --models rescor pure_nca conv2d --no-wandb

    # Quick test (fewer epochs, smaller data)
    uv run --with scikit-learn,matplotlib,scipy python experiments/unified_ablation.py --quick --no-wandb

    # Specific grid size
    uv run --with scikit-learn,matplotlib,scipy python experiments/unified_ablation.py --grid-size 64 --no-wandb

    # List available benchmarks and models
    uv run python experiments/unified_ablation.py --list
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Project root setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"
PLOTS_DIR = PROJECT_ROOT / "experiments" / "plots"
SEED = 42

# ---------------------------------------------------------------------------
# Canonical lists — these are the ground truth for what this runner supports.
# The actual availability depends on whether the modules are importable.
# ---------------------------------------------------------------------------
ALL_BENCHMARKS = ["heat", "gol", "ks", "gray_scott", "rule110", "wireworld", "grid_world", "dmcontrol", "crafter_lite", "minigrid", "autumn_disease", "autumn_gravity", "autumn_water", "atari_pong", "atari_breakout"]
ALL_MODELS = ["rescor", "rescor_e2", "rescor_e3", "rescor_e3b", "rescor_e3c", "rescor_e4", "rescor_e6", "rescor_traj_attn", "rescor_mp_gate", "pure_nca", "conv2d", "mlp", "cml_ridge", "nca_inside_cml", "gated_blend", "cml_reg"]

# Benchmarks where the primary metric is accuracy (higher=better) vs MSE (lower=better)
ACCURACY_BENCHMARKS = {"gol", "rule110", "wireworld", "grid_world", "crafter_lite", "autumn_disease", "autumn_gravity", "autumn_water", "atari_pong", "atari_breakout"}

# Rollout horizons to evaluate
ROLLOUT_HORIZONS = [1, 3, 5, 10]

# ---------------------------------------------------------------------------
# Graceful imports of parallel modules
# ---------------------------------------------------------------------------
BENCHMARKS = None
run_cem_evaluation = None
_benchmarks_available = False
_benchmarks_import_error = ""

try:
    from wmca.benchmarks import BENCHMARKS
    _benchmarks_available = True
    try:
        from wmca.benchmarks import run_cem_evaluation  # type: ignore
    except ImportError:
        run_cem_evaluation = None
except ImportError as e:
    _benchmarks_import_error = str(e)
except Exception as e:
    _benchmarks_import_error = f"Unexpected error: {e}"

MODEL_REGISTRY = None
create_model = None
train_model = None
train_ridge_model = None
evaluate_model = None
evaluate_rollout = None
param_count = None
CML2DRidge = None
_models_available = False
_models_import_error = ""

try:
    from wmca.model_registry import (
        MODEL_REGISTRY,
        CML2DRidge,
        create_model,
        train_model,
        train_ridge_model,
        evaluate_model,
        evaluate_rollout,
        param_count,
    )
    _models_available = True
except ImportError as e:
    _models_import_error = str(e)
except Exception as e:
    _models_import_error = f"Unexpected error: {e}"

# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------

def _get_plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _get_torch():
    import torch
    return torch


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified Ablation: benchmarks x models with evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--benchmarks", nargs="+", default=None, metavar="NAME",
        help=f"Benchmarks to run (default: all). Choices: {', '.join(ALL_BENCHMARKS)}",
    )
    parser.add_argument(
        "--models", nargs="+", default=None, metavar="NAME",
        help=f"Models to run (default: all). Choices: {', '.join(ALL_MODELS)}",
    )
    parser.add_argument(
        "--grid-size", type=int, default=None,
        help="Grid size (default: 16, or 8 with --quick)",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode: 10 epochs, 100 trajectories, grid_size=8",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override number of training epochs",
    )
    parser.add_argument(
        "--trajectories", type=int, default=None,
        help="Override number of trajectories for data generation",
    )
    parser.add_argument(
        "--no-wandb", action="store_true",
        help="Disable wandb logging",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List available benchmarks and models, then exit",
    )
    parser.add_argument(
        "--seed", type=int, default=SEED,
        help=f"Random seed (default: {SEED})",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Resolve effective config from args + defaults
# ---------------------------------------------------------------------------

def resolve_config(args: argparse.Namespace) -> dict:
    """Compute effective run configuration from CLI args and defaults."""
    if args.quick:
        epochs = args.epochs or 10
        trajectories = args.trajectories or 100
        grid_size = args.grid_size or 8
    else:
        epochs = args.epochs or 30
        trajectories = args.trajectories or 300
        grid_size = args.grid_size or 16

    selected_benchmarks = args.benchmarks if args.benchmarks else list(ALL_BENCHMARKS)
    selected_models = args.models if args.models else list(ALL_MODELS)

    # Validate selections
    for b in selected_benchmarks:
        if b not in ALL_BENCHMARKS:
            print(f"WARNING: Unknown benchmark '{b}'. Known: {ALL_BENCHMARKS}")
    for m in selected_models:
        if m not in ALL_MODELS:
            print(f"WARNING: Unknown model '{m}'. Known: {ALL_MODELS}")

    return {
        "benchmarks": selected_benchmarks,
        "models": selected_models,
        "grid_size": grid_size,
        "epochs": epochs,
        "trajectories": trajectories,
        "seed": args.seed,
        "quick": args.quick,
        "no_wandb": args.no_wandb,
    }


# ---------------------------------------------------------------------------
# List mode
# ---------------------------------------------------------------------------

def print_available():
    """Print available benchmarks and models, then exit."""
    print("=" * 60)
    print("AVAILABLE BENCHMARKS")
    print("=" * 60)
    for b in ALL_BENCHMARKS:
        avail = "OK" if (_benchmarks_available and BENCHMARKS and b in BENCHMARKS) else "not loaded"
        metric = "accuracy" if b in ACCURACY_BENCHMARKS else "MSE"
        print(f"  {b:<16} [{metric}]  ({avail})")

    if not _benchmarks_available:
        print(f"\n  wmca.benchmarks import failed: {_benchmarks_import_error}")

    print()
    print("=" * 60)
    print("AVAILABLE MODELS")
    print("=" * 60)
    for m in ALL_MODELS:
        avail = "OK" if (_models_available and MODEL_REGISTRY and m in MODEL_REGISTRY) else "not loaded"
        print(f"  {m:<20} ({avail})")

    if not _models_available:
        print(f"\n  wmca.model_registry import failed: {_models_import_error}")

    print()
    print("Rollout horizons:", ROLLOUT_HORIZONS)


# ---------------------------------------------------------------------------
# CEM planning evaluation for grid_world benchmark
# ---------------------------------------------------------------------------

def run_cem_planning_eval(model, data, config: dict, device) -> dict:
    """Run CEM planning evaluation for the grid_world benchmark.

    Returns a dict with planning-specific metrics (success_rate, avg_steps,
    avg_reward).
    """
    if run_cem_evaluation is None:
        print("    [SKIP] CEM planning eval — wmca.benchmarks.run_cem_evaluation not available")
        return {
            "success_rate": float("nan"),
            "avg_steps": float("nan"),
            "avg_reward": float("nan"),
        }

    # Use fewer episodes in quick mode to keep runtime reasonable
    n_episodes = 20 if config.get("quick") else 200
    try:
        t0 = time.time()
        metrics = run_cem_evaluation(
            model,
            data,
            n_episodes=n_episodes,
            device=device,
            seed=config["seed"],
        )
        elapsed = time.time() - t0
        sr = metrics.get("success_rate", float("nan"))
        avg_s = metrics.get("avg_steps", float("nan"))
        method = metrics.get("planning_method", "CEM")
        print(f"    {method} planning: success={sr * 100:.1f}%  "
              f"avg_steps={avg_s:.1f}  time={elapsed:.1f}s")
        return metrics
    except Exception as e:
        print(f"    [SKIP] CEM planning eval — error: {e}")
        import traceback; traceback.print_exc()
        return {
            "success_rate": float("nan"),
            "avg_steps": float("nan"),
            "avg_reward": float("nan"),
        }


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def run_ablation(config: dict) -> dict:
    """Run the full (benchmark x model) ablation.

    Returns nested dict:  results[benchmark_name][model_name] = { metrics }
    """
    torch = _get_torch()
    from wmca.utils import pick_device
    device = pick_device()

    results: dict[str, dict[str, dict]] = defaultdict(dict)
    selected_benchmarks = config["benchmarks"]
    selected_models = config["models"]

    print("=" * 72)
    print("UNIFIED ABLATION")
    print("=" * 72)
    print(f"  Benchmarks : {selected_benchmarks}")
    print(f"  Models     : {selected_models}")
    print(f"  Grid size  : {config['grid_size']}")
    print(f"  Epochs     : {config['epochs']}")
    print(f"  Trajectories: {config['trajectories']}")
    print(f"  Seed       : {config['seed']}")
    print(f"  Device     : {device}")
    print(f"  Quick mode : {config['quick']}")
    print()

    n_benchmarks = len(selected_benchmarks)
    for bi, benchmark_name in enumerate(selected_benchmarks, 1):
        print(f"\n{'=' * 72}")
        print(f"  [{bi}/{n_benchmarks}] BENCHMARK: {benchmark_name} ({config['grid_size']}x{config['grid_size']})")
        print(f"{'=' * 72}")

        # ---- Generate data ----
        if benchmark_name not in BENCHMARKS:
            print(f"  [SKIP] Benchmark '{benchmark_name}' not found in BENCHMARKS dict")
            continue

        t_data = time.time()
        try:
            data = BENCHMARKS[benchmark_name](
                grid_size=config["grid_size"],
                n_trajectories=config["trajectories"],
                seed=config["seed"],
                device=device,
            )
        except Exception as e:
            print(f"  [SKIP] Data generation failed for '{benchmark_name}': {e}")
            continue
        print(f"  Data generated in {time.time() - t_data:.1f}s")
        print(f"    X_train: {data.X_train.shape}, Y_train: {data.Y_train.shape}")
        print(f"    X_test:  {data.X_test.shape},  Y_test:  {data.Y_test.shape}")
        if hasattr(data, "meta") and data.meta:
            print(f"    Meta: {data.meta}")

        meta = data.meta if hasattr(data, "meta") and data.meta else {}
        in_channels = meta.get("in_channels", 1)
        out_channels = meta.get("out_channels", in_channels)
        loss_type = meta.get("loss_type", "mse")

        # Infer actual spatial dims from data tensor shape (N, C, H, W)
        data_shape = data.X_train.shape
        actual_grid_h = data_shape[2] if len(data_shape) >= 3 else config["grid_size"]
        actual_grid_w = data_shape[3] if len(data_shape) >= 4 else actual_grid_h

        # ---- Run each model ----
        n_models = len(selected_models)
        for mi, model_name in enumerate(selected_models, 1):
            print(f"\n  --- [{mi}/{n_models}] {model_name} on {benchmark_name} ---")

            if model_name not in MODEL_REGISTRY:
                print(f"    [SKIP] Model '{model_name}' not found in MODEL_REGISTRY")
                continue

            try:
                model = create_model(
                    model_name,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    grid_size=config["grid_size"],
                    grid_h=actual_grid_h,
                    grid_w=actual_grid_w,
                    seed=config["seed"],
                )
            except Exception as e:
                print(f"    [SKIP] Model creation failed for '{model_name}': {e}")
                continue

            # ---- Train ----
            t_train = time.time()
            try:
                if CML2DRidge is not None and isinstance(model, CML2DRidge):
                    # Ridge model — needs numpy arrays
                    X_tr_np = data.X_train.cpu().numpy() if hasattr(data.X_train, 'cpu') else np.asarray(data.X_train)
                    Y_tr_np = data.Y_train.cpu().numpy() if hasattr(data.Y_train, 'cpu') else np.asarray(data.Y_train)
                    model, _ridge_stats = train_ridge_model(model, X_tr_np, Y_tr_np)
                else:
                    model = train_model(
                        model,
                        data.X_train, data.Y_train,
                        X_val=getattr(data, "X_val", None),
                        Y_val=getattr(data, "Y_val", None),
                        loss_type=loss_type,
                        epochs=config["epochs"],
                        benchmark_name=benchmark_name,
                        model_name=model_name,
                        device=device,
                    )
            except Exception as e:
                print(f"    [SKIP] Training failed for '{model_name}': {e}")
                import traceback; traceback.print_exc()
                continue
            elapsed = time.time() - t_train
            print(f"    Training time: {elapsed:.1f}s")

            # ---- Evaluate 1-step ----
            try:
                if CML2DRidge is not None and isinstance(model, CML2DRidge):
                    X_te_np = data.X_test.cpu().numpy() if hasattr(data.X_test, 'cpu') else np.asarray(data.X_test)
                    Y_te_np = data.Y_test.cpu().numpy() if hasattr(data.Y_test, 'cpu') else np.asarray(data.Y_test)
                    preds = model.predict(X_te_np)
                    mse_val = float(np.mean((preds - Y_te_np) ** 2))
                    one_step_metrics = {"mse": mse_val}
                else:
                    one_step_metrics = evaluate_model(
                        model, data.X_test, data.Y_test,
                        loss_type=loss_type,
                        benchmark_name=benchmark_name,
                        device=device,
                    )
            except Exception as e:
                print(f"    [SKIP] 1-step evaluation failed: {e}")
                one_step_metrics = {}

            # ---- Evaluate rollout ----
            rollout_metrics = {}
            try:
                if CML2DRidge is not None and isinstance(model, CML2DRidge):
                    print(f"    [SKIP] Rollout not supported for CML2DRidge")
                else:
                    rollout_metrics = evaluate_rollout(
                        model, data,
                        horizons=ROLLOUT_HORIZONS,
                        benchmark_name=benchmark_name,
                        device=device,
                    )
                    if isinstance(rollout_metrics, dict) and (
                        "action_conditioned" in rollout_metrics
                        or "skipped_reason" in rollout_metrics
                    ):
                        reason = rollout_metrics.get(
                            "skipped_reason", "action-conditioned data"
                        )
                        print(f"    [SKIP] Rollout: {reason}")
            except Exception as e:
                print(f"    [SKIP] Rollout evaluation failed: {e}")

            # ---- Param count ----
            try:
                pc = param_count(model)
            except Exception as e:
                print(f"    [SKIP] Param count failed: {e}")
                pc = {"trained": 0, "frozen": 0}

            # ---- CEM planning for grid_world ----
            cem_metrics = {}
            if benchmark_name == "grid_world":
                cem_metrics = run_cem_planning_eval(model, data, config, device)

            # ---- Store results ----
            result_entry = {
                "one_step": one_step_metrics,
                "rollout": rollout_metrics,
                "params": pc,
                "train_time": elapsed,
            }
            if cem_metrics:
                result_entry["cem_planning"] = cem_metrics

            results[benchmark_name][model_name] = result_entry

            # Print summary line
            _print_model_summary(model_name, result_entry, benchmark_name)

            # Free memory
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Free benchmark data
        del data
        gc.collect()

    return dict(results)


def _print_model_summary(model_name: str, entry: dict, benchmark_name: str):
    """Print a concise one-line summary for a model run."""
    one_step = entry.get("one_step", {})
    pc = entry.get("params", {})
    t = entry.get("train_time", 0)

    # Pick the primary metric from one_step
    if one_step:
        primary_key = list(one_step.keys())[0]
        primary_val = one_step[primary_key]
        metric_str = f"{primary_key}={primary_val:.6f}"
    else:
        metric_str = "N/A"

    # Rollout summary: pick h=10 if available
    rollout = entry.get("rollout", {})
    if isinstance(rollout, dict) and (
        "action_conditioned" in rollout or "skipped_reason" in rollout
    ):
        rollout_str = "skipped(action-cond)"
    elif rollout and "10" in rollout:
        rollout_str = f"h10={rollout['10']:.6f}"
    elif rollout and 10 in rollout:
        rollout_str = f"h10={rollout[10]:.6f}"
    elif rollout:
        # Take the last horizon available, filtering out non-numeric marker keys
        numeric_keys = [k for k in rollout.keys()
                        if isinstance(k, int)
                        or (isinstance(k, str) and k.isdigit())]
        if numeric_keys:
            last_key = numeric_keys[-1]
            rollout_str = f"h{last_key}={rollout[last_key]:.6f}"
        else:
            rollout_str = "N/A"
    else:
        rollout_str = "N/A"

    params_str = f"{pc.get('trained', '?')}/{pc.get('frozen', '?')}"

    print(f"    >> {model_name}: {metric_str}  rollout:{rollout_str}  "
          f"params(T/F):{params_str}  time:{t:.1f}s")


# ---------------------------------------------------------------------------
# Summary table printing
# ---------------------------------------------------------------------------

def print_summary_tables(results: dict, config: dict):
    """Print comprehensive summary tables to stdout."""
    grid_size = config["grid_size"]

    print()
    print("=" * 80)
    print("UNIFIED ABLATION RESULTS")
    print("=" * 80)

    # Per-benchmark tables
    for benchmark_name, model_results in results.items():
        if not model_results:
            continue

        is_acc = benchmark_name in ACCURACY_BENCHMARKS
        metric_type = "Acc" if is_acc else "MSE"

        print(f"\nBenchmark: {benchmark_name} ({grid_size}x{grid_size})")
        print("-" * 78)

        # Determine available rollout horizons from results
        available_horizons = set()
        for mr in model_results.values():
            rollout = mr.get("rollout", {})
            if isinstance(rollout, dict) and (
                "action_conditioned" in rollout or "skipped_reason" in rollout
            ):
                continue
            for k in rollout.keys():
                k_str = str(k)
                if k_str.isdigit():
                    available_horizons.add(k_str)
        # Sort horizons numerically
        sorted_horizons = sorted(available_horizons, key=lambda x: int(x))

        # Pick one representative horizon for the compact table
        repr_horizon = None
        for h_candidate in ["10", "5", "3", "1"]:
            if h_candidate in sorted_horizons:
                repr_horizon = h_candidate
                break

        # Header
        header = f"{'Model':<18} {'1-step ' + metric_type:>14}"
        if repr_horizon:
            header += f"   {'h=' + repr_horizon + ' ' + metric_type:>12}"
        header += f"   {'Params(T/F)':>14}   {'Time':>7}"
        print(header)
        print("-" * len(header))

        # Rows
        for model_name, mr in model_results.items():
            one_step = mr.get("one_step", {})
            rollout = mr.get("rollout", {})
            pc = mr.get("params", {})
            t = mr.get("train_time", 0)

            # Primary metric
            if one_step:
                primary_val = list(one_step.values())[0]
                one_step_str = f"{primary_val:.6f}"
            else:
                one_step_str = "N/A"

            # Rollout metric
            if isinstance(rollout, dict) and (
                "action_conditioned" in rollout or "skipped_reason" in rollout
            ):
                rollout_str = "skipped"
            elif repr_horizon and repr_horizon in rollout:
                rollout_val = rollout[repr_horizon]
                rollout_str = f"{rollout_val:.6f}"
            elif repr_horizon and int(repr_horizon) in rollout:
                rollout_val = rollout[int(repr_horizon)]
                rollout_str = f"{rollout_val:.6f}"
            else:
                rollout_str = "N/A"

            params_str = f"{pc.get('trained', '?')}/{pc.get('frozen', '?')}"
            time_str = f"{t:.1f}s"

            row = f"{model_name:<18} {one_step_str:>14}"
            if repr_horizon:
                row += f"   {rollout_str:>12}"
            row += f"   {params_str:>14}   {time_str:>7}"
            print(row)

        # CEM planning sub-table for grid_world
        has_cem = any("cem_planning" in mr for mr in model_results.values())
        if has_cem:
            print(f"\n  CEM Planning ({benchmark_name}):")
            print(f"  {'Model':<18} {'Success%':>10} {'AvgSteps':>10}")
            print(f"  {'-' * 40}")
            for model_name, mr in model_results.items():
                cem = mr.get("cem_planning", {})
                if cem:
                    sr = cem.get("success_rate", float("nan"))
                    avg_s = cem.get("avg_steps", float("nan"))
                    print(f"  {model_name:<18} {sr * 100:>9.1f}% {avg_s:>10.1f}")

    # Cross-benchmark scoring + ranking
    _print_cross_benchmark_ranking(results)


# ---------------------------------------------------------------------------
# Cross-benchmark scoring helpers
# ---------------------------------------------------------------------------

def _normalized_score(value, all_values, higher_is_better: bool) -> float:
    """Normalize a metric value to [0, 1] across all models on a benchmark.

    1.0 = best model, 0.0 = worst model. Returns 0.0 if no valid values,
    1.0 if all values are tied.
    """
    valid = [v for v in all_values if v is not None and not math.isnan(v)]
    if not valid:
        return 0.0
    if value is None or math.isnan(value):
        return 0.0
    v_min, v_max = min(valid), max(valid)
    if v_max == v_min:
        return 1.0
    if higher_is_better:
        return (value - v_min) / (v_max - v_min)
    else:
        return (v_max - value) / (v_max - v_min)


def _raw_score_per_benchmark(metric_value, higher_is_better: bool):
    """Non-normalized score in [0, 1]. Doesn't depend on other models.

    - higher_is_better (accuracy): clamp to [0, 1].
    - lower_is_better  (MSE):     map [0, inf] -> [0, 1] via 1/(1+x).

    Returns None if the metric is missing / NaN so callers can skip it.
    """
    if metric_value is None:
        return None
    try:
        fv = float(metric_value)
    except (TypeError, ValueError):
        return None
    if math.isnan(fv):
        return None
    if higher_is_better:
        return max(0.0, min(1.0, fv))
    else:
        return 1.0 / (1.0 + max(0.0, fv))


def _pareto_score_per_benchmark(
    model_perf: dict[str, float],
    model_params: dict[str, float],
    higher_is_better: bool,
) -> dict[str, float]:
    """Per-benchmark Pareto-efficiency score in (perf, params) space.

    For each model, 1.0 means on the Pareto frontier (no other model
    strictly dominates it on BOTH perf and param count). Off-frontier
    models get a score based on (normalized) Euclidean distance to the
    nearest frontier point: 1 - dist / max_dist, clipped to [0, 1].
    """
    names = [n for n, v in model_perf.items()
             if v is not None and not math.isnan(v)]
    if not names:
        return {n: 0.0 for n in model_perf}

    # Normalize perf to [0,1] where 1=best (so Pareto frontier is always
    # "maximize perf_norm, minimize params"), then also normalize params
    # to [0,1] for distance computation.
    perf_vals = [model_perf[n] for n in names]
    perf_norms = {
        n: _normalized_score(model_perf[n], perf_vals, higher_is_better)
        for n in names
    }

    param_vals = [float(model_params.get(n, 0.0)) for n in names]
    p_min, p_max = min(param_vals), max(param_vals)
    if p_max > p_min:
        param_norms = {
            n: (float(model_params.get(n, 0.0)) - p_min) / (p_max - p_min)
            for n in names
        }
    else:
        param_norms = {n: 0.0 for n in names}

    # Find Pareto frontier: model A dominates B iff
    # perf_norm[A] >= perf_norm[B] AND param_norm[A] <= param_norm[B]
    # with at least one strict inequality.
    frontier: set[str] = set()
    for n in names:
        dominated = False
        for m in names:
            if m == n:
                continue
            better_or_eq = (
                perf_norms[m] >= perf_norms[n]
                and param_norms[m] <= param_norms[n]
            )
            strictly_better = (
                perf_norms[m] > perf_norms[n]
                or param_norms[m] < param_norms[n]
            )
            if better_or_eq and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.add(n)

    # Max possible distance in the unit square is sqrt(2)
    max_dist = math.sqrt(2.0)
    scores: dict[str, float] = {}
    for n in names:
        if n in frontier:
            scores[n] = 1.0
            continue
        # Distance to nearest frontier point
        best = float("inf")
        for f in frontier:
            d = math.hypot(
                perf_norms[n] - perf_norms[f],
                param_norms[n] - param_norms[f],
            )
            if d < best:
                best = d
        if best == float("inf"):
            scores[n] = 0.0
        else:
            s = 1.0 - (best / max_dist)
            scores[n] = max(0.0, min(1.0, s))

    # Models that were invalid get 0
    for n in model_perf:
        if n not in scores:
            scores[n] = 0.0
    return scores


def compute_cross_benchmark_scores(results: dict) -> dict:
    """Compute per-model cross-benchmark scores.

    Returns a dict keyed by model name with the following fields:
        norm_score:      mean normalized score across benchmarks [0-1]
        raw_score:       mean non-normalized score across benchmarks [0-1]
                         (accuracy clamped; MSE mapped via 1/(1+x))
        param_eff_score: mean (norm_score / log10(trained_params + 1))
                         per benchmark, then averaged
        pareto_score:    mean per-benchmark Pareto-efficiency score [0-1]
        avg_rank:        legacy average rank (1=best)
        best_on:         list of benchmark names where model is #1
        worst_on:        list of benchmark names where model is last
        n_benchmarks:    number of benchmarks contributing to this model
    """
    all_model_names: set[str] = set()
    for model_results in results.values():
        all_model_names.update(model_results.keys())
    if not all_model_names:
        return {}

    per_model_norm: dict[str, list[float]] = defaultdict(list)
    per_model_raw: dict[str, list[float]] = defaultdict(list)
    per_model_pareto: dict[str, list[float]] = defaultdict(list)
    per_model_param_eff: dict[str, list[float]] = defaultdict(list)
    per_model_ranks: dict[str, list[int]] = defaultdict(list)
    best_on: dict[str, list[str]] = defaultdict(list)
    worst_on: dict[str, list[str]] = defaultdict(list)

    for benchmark_name, model_results in results.items():
        if not model_results:
            continue

        is_acc = benchmark_name in ACCURACY_BENCHMARKS
        higher_is_better = is_acc

        # Extract primary metric and trainable param count per model
        model_perf: dict[str, float] = {}
        model_params: dict[str, float] = {}
        for model_name, mr in model_results.items():
            one_step = mr.get("one_step", {})
            if not one_step:
                continue
            val = list(one_step.values())[0]
            if val is None:
                continue
            try:
                fval = float(val)
            except (TypeError, ValueError):
                continue
            if math.isnan(fval):
                continue
            model_perf[model_name] = fval
            pc = mr.get("params", {}) or {}
            model_params[model_name] = float(pc.get("trained", 0) or 0)

        if not model_perf:
            continue

        perf_values = list(model_perf.values())

        # Normalized + raw score
        for model_name, val in model_perf.items():
            norm = _normalized_score(val, perf_values, higher_is_better)
            per_model_norm[model_name].append(norm)
            raw = _raw_score_per_benchmark(val, higher_is_better)
            if raw is not None:
                per_model_raw[model_name].append(raw)
            # Param-weighted: norm / log10(params + 1)
            tp = model_params.get(model_name, 0.0)
            denom = math.log10(tp + 1.0)
            if denom <= 0:
                # Models with 0 trainable params: give them the raw norm
                # so they aren't penalized infinitely. (log10(1)=0.)
                per_model_param_eff[model_name].append(norm)
            else:
                per_model_param_eff[model_name].append(norm / denom)

        # Pareto score
        pareto_scores = _pareto_score_per_benchmark(
            model_perf, model_params, higher_is_better
        )
        for model_name, ps in pareto_scores.items():
            per_model_pareto[model_name].append(ps)

        # Legacy rank (1 = best)
        sorted_models = sorted(
            model_perf.items(),
            key=lambda x: x[1],
            reverse=higher_is_better,
        )
        for rank_idx, (model_name, _) in enumerate(sorted_models):
            per_model_ranks[model_name].append(rank_idx + 1)
        best_on[sorted_models[0][0]].append(benchmark_name)
        worst_on[sorted_models[-1][0]].append(benchmark_name)

    summary: dict[str, dict] = {}
    for model_name in all_model_names:
        norms = per_model_norm.get(model_name, [])
        raws = per_model_raw.get(model_name, [])
        paretos = per_model_pareto.get(model_name, [])
        peffs = per_model_param_eff.get(model_name, [])
        ranks = per_model_ranks.get(model_name, [])
        summary[model_name] = {
            "norm_score": float(np.mean(norms)) if norms else 0.0,
            "raw_score": float(np.mean(raws)) if raws else 0.0,
            "param_eff_score": float(np.mean(peffs)) if peffs else 0.0,
            "pareto_score": float(np.mean(paretos)) if paretos else 0.0,
            "avg_rank": float(np.mean(ranks)) if ranks else float("inf"),
            "best_on": list(best_on.get(model_name, [])),
            "worst_on": list(worst_on.get(model_name, [])),
            "n_benchmarks": len(norms),
        }
    return summary


def _print_cross_benchmark_ranking(results: dict):
    """Print cross-benchmark scoring table using multiple metrics.

    Reports four complementary scores (normalized, raw/non-normalized,
    param-efficiency, and Pareto-efficiency) plus the legacy average
    rank for backward compatibility. Sorted by raw score (descending)
    since it's the primary absolute measure.
    """
    if len(results) < 1:
        return

    summary = compute_cross_benchmark_scores(results)
    if not summary:
        return

    rows = sorted(
        summary.items(),
        key=lambda kv: kv[1]["raw_score"],
        reverse=True,
    )

    print(f"\n{'=' * 104}")
    print("CROSS-BENCHMARK COMPARISON")
    print(f"{'=' * 104}")
    header = (
        f"{'Model':<16} {'NormScore':>10}   {'RawScore':>9}   "
        f"{'ParamEffScore':>13}   {'ParetoScore':>12}   "
        f"{'AvgRank':>8}   {'Best On':<20}"
    )
    print(header)
    print("-" * len(header))
    for model_name, s in rows:
        best = ",".join(s["best_on"]) if s["best_on"] else "-"
        avg_rank_str = (
            f"{s['avg_rank']:>8.1f}"
            if math.isfinite(s["avg_rank"]) else f"{'inf':>8}"
        )
        print(
            f"{model_name:<16} "
            f"{s['norm_score']:>10.3f}   "
            f"{s['raw_score']:>9.3f}   "
            f"{s['param_eff_score']:>13.3f}   "
            f"{s['pareto_score']:>12.3f}   "
            f"{avg_rank_str}   "
            f"{best:<20}"
        )


# ---------------------------------------------------------------------------
# Save results to JSON
# ---------------------------------------------------------------------------

def save_results_json(results: dict, config: dict):
    """Save results to experiments/results/unified_ablation.json."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "unified_ablation.json"

    # Make everything JSON-serializable
    def _to_serializable(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
            return str(obj)
        return obj

    def _deep_convert(d):
        if isinstance(d, dict):
            return {str(k): _deep_convert(v) for k, v in d.items()}
        if isinstance(d, (list, tuple)):
            return [_deep_convert(x) for x in d]
        return _to_serializable(d)

    # Compute cross-benchmark scores (norm / param-eff / pareto + legacy rank)
    try:
        cross_scores = compute_cross_benchmark_scores(results)
    except Exception as e:
        print(f"  [WARN] Failed to compute cross-benchmark scores: {e}")
        cross_scores = {}

    payload = {
        "config": config,
        "results": _deep_convert(results),
        "cross_benchmark_scores": _deep_convert(cross_scores),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)

    print(f"\nResults saved to {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def generate_plots(results: dict, config: dict):
    """Generate summary plots:
      1. Pareto plot (params vs performance)
      2. Heatmap (models x benchmarks, colored by rank)
    """
    try:
        plt = _get_plt()
    except ImportError:
        print("\n[SKIP] Plotting — matplotlib not available")
        return

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    _plot_pareto(results, config, plt)
    _plot_heatmap(results, config, plt)


def _plot_pareto(results: dict, config: dict, plt):
    """Pareto plot: total params (trained) vs primary 1-step metric per benchmark."""
    n_benchmarks = len(results)
    if n_benchmarks == 0:
        return

    fig, axes = plt.subplots(1, n_benchmarks, figsize=(6 * n_benchmarks, 5), squeeze=False)

    colors = plt.cm.tab10(np.linspace(0, 1, len(ALL_MODELS)))
    model_color_map = {name: colors[i] for i, name in enumerate(ALL_MODELS)}

    for col_idx, (benchmark_name, model_results) in enumerate(results.items()):
        ax = axes[0, col_idx]
        if not model_results:
            ax.set_title(f"{benchmark_name} (no data)")
            continue

        is_acc = benchmark_name in ACCURACY_BENCHMARKS

        xs, ys, labels = [], [], []
        for model_name, mr in model_results.items():
            pc = mr.get("params", {})
            one_step = mr.get("one_step", {})
            if not one_step:
                continue

            total_params = pc.get("trained", 0) + pc.get("frozen", 0)
            primary_val = list(one_step.values())[0]

            xs.append(total_params)
            ys.append(primary_val)
            labels.append(model_name)

            color = model_color_map.get(model_name, "gray")
            ax.scatter(total_params, primary_val, color=color, s=80, zorder=3)
            ax.annotate(model_name, (total_params, primary_val),
                        textcoords="offset points", xytext=(5, 5),
                        fontsize=7, ha="left")

        metric_label = "Accuracy" if is_acc else "MSE"
        ax.set_xlabel("Total Parameters")
        ax.set_ylabel(f"1-step {metric_label}")
        ax.set_title(f"{benchmark_name} ({config['grid_size']}x{config['grid_size']})")
        ax.grid(True, alpha=0.3)
        if not is_acc and ys:
            ax.set_yscale("log")

    fig.suptitle("Pareto: Parameters vs Performance", fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = PLOTS_DIR / "unified_pareto.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved plot: {path}")


def _plot_heatmap(results: dict, config: dict, plt):
    """Heatmap: models (rows) x benchmarks (cols), colored by within-benchmark rank."""
    benchmark_names = list(results.keys())
    if not benchmark_names:
        return

    # Collect all models across benchmarks
    all_model_names_ordered: list[str] = []
    seen = set()
    for mr in results.values():
        for mn in mr:
            if mn not in seen:
                all_model_names_ordered.append(mn)
                seen.add(mn)

    if not all_model_names_ordered:
        return

    n_models = len(all_model_names_ordered)
    n_bench = len(benchmark_names)

    # Build rank matrix
    rank_matrix = np.full((n_models, n_bench), np.nan)

    for col_idx, benchmark_name in enumerate(benchmark_names):
        model_results = results.get(benchmark_name, {})
        is_acc = benchmark_name in ACCURACY_BENCHMARKS

        # Get primary metric per model
        scores: list[tuple[str, float]] = []
        for model_name, mr in model_results.items():
            one_step = mr.get("one_step", {})
            if one_step:
                scores.append((model_name, list(one_step.values())[0]))

        # Sort to get ranks
        scores.sort(key=lambda x: x[1], reverse=is_acc)
        name_to_rank = {name: rank + 1 for rank, (name, _) in enumerate(scores)}

        for row_idx, model_name in enumerate(all_model_names_ordered):
            if model_name in name_to_rank:
                rank_matrix[row_idx, col_idx] = name_to_rank[model_name]

    fig, ax = plt.subplots(figsize=(max(6, n_bench * 1.5), max(4, n_models * 0.6)))

    # Use a diverging colormap: low rank (good) = green, high rank (bad) = red
    cmap = plt.cm.RdYlGn_r
    im = ax.imshow(rank_matrix, cmap=cmap, aspect="auto", vmin=1, vmax=n_models)

    # Annotate cells with rank number
    for i in range(n_models):
        for j in range(n_bench):
            val = rank_matrix[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{int(val)}", ha="center", va="center",
                        fontsize=10, fontweight="bold",
                        color="white" if val > n_models / 2 else "black")

    ax.set_xticks(range(n_bench))
    ax.set_xticklabels(benchmark_names, rotation=30, ha="right")
    ax.set_yticks(range(n_models))
    ax.set_yticklabels(all_model_names_ordered)
    ax.set_title(f"Model Rank by Benchmark ({config['grid_size']}x{config['grid_size']})",
                 fontsize=12, fontweight="bold")

    fig.colorbar(im, ax=ax, label="Rank (1=best)", shrink=0.8)
    fig.tight_layout()
    path = PLOTS_DIR / "unified_heatmap.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved plot: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # --list mode: print and exit
    if args.list:
        print_available()
        return

    # Check module availability
    if not _benchmarks_available:
        print(f"ERROR: Cannot import wmca.benchmarks: {_benchmarks_import_error}")
        print("       This module is being created in parallel. Run again once it is ready.")
        sys.exit(1)

    if not _models_available:
        print(f"ERROR: Cannot import wmca.model_registry: {_models_import_error}")
        print("       This module is being created in parallel. Run again once it is ready.")
        sys.exit(1)

    # Disable wandb if requested
    if args.no_wandb:
        os.environ["WANDB_DISABLED"] = "true"
        os.environ["WANDB_MODE"] = "disabled"

    config = resolve_config(args)

    # Set seeds
    np.random.seed(config["seed"])
    torch = _get_torch()
    torch.manual_seed(config["seed"])

    t_total = time.time()

    # ---- Run ablation ----
    results = run_ablation(config)

    total_time = time.time() - t_total
    print(f"\nTotal wall time: {total_time:.1f}s")

    # ---- Print summary tables ----
    print_summary_tables(results, config)

    # ---- Save JSON ----
    save_results_json(results, config)

    # ---- Generate plots ----
    print("\nGenerating plots...")
    generate_plots(results, config)

    print("\nDone.")


if __name__ == "__main__":
    main()
