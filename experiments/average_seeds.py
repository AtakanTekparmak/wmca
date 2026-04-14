"""Average results from multiple seed runs and output a clean summary table.

Usage:
    uv run python experiments/average_seeds.py experiments/results/unified_ablation_seed*.json
"""
import json
import sys
from collections import defaultdict
from pathlib import Path


def load_results(path: str) -> dict:
    with open(path) as f:
        data = json.load(f)
    return data["results"]


def main():
    if len(sys.argv) < 2:
        print("Usage: python average_seeds.py <result1.json> [result2.json] ...")
        sys.exit(1)

    files = sys.argv[1:]
    print(f"Averaging {len(files)} seed files: {[Path(f).name for f in files]}")

    # Collect per-(benchmark, model) metrics across seeds
    metrics = defaultdict(lambda: defaultdict(list))  # {(bench, model): {metric: [values]}}

    for f in files:
        results = load_results(f)
        for bench, bench_data in results.items():
            if not isinstance(bench_data, dict):
                continue
            for model, model_data in bench_data.items():
                if not isinstance(model_data, dict):
                    continue
                for key, val in model_data.items():
                    if isinstance(val, (int, float)) and key in (
                        "one_step_mse", "one_step_accuracy", "param_count_trained",
                        "param_count_frozen", "train_time_s",
                    ):
                        metrics[(bench, model)][key].append(val)

    # Print summary table
    print("\n" + "=" * 100)
    print(f"{'Benchmark':<20} {'Model':<20} {'Params':>8} {'1-step MSE':>14} {'1-step Acc':>12} {'Seeds':>6}")
    print("=" * 100)

    prev_bench = ""
    for (bench, model), vals in sorted(metrics.items()):
        if bench != prev_bench:
            if prev_bench:
                print("-" * 100)
            prev_bench = bench

        params = int(vals["param_count_trained"][0]) if vals["param_count_trained"] else "?"

        mse_vals = vals.get("one_step_mse", [])
        acc_vals = vals.get("one_step_accuracy", [])
        n_seeds = max(len(mse_vals), len(acc_vals))

        mse_str = f"{sum(mse_vals)/len(mse_vals):.2e}" if mse_vals else "—"
        acc_str = f"{100*sum(acc_vals)/len(acc_vals):.2f}%" if acc_vals else "—"

        print(f"{bench:<20} {model:<20} {params:>8} {mse_str:>14} {acc_str:>12} {n_seeds:>6}")

    print("=" * 100)

    # Also dump a clean JSON for easy consumption
    averaged = {}
    for (bench, model), vals in sorted(metrics.items()):
        if bench not in averaged:
            averaged[bench] = {}
        entry = {}
        for key, vlist in vals.items():
            entry[key] = sum(vlist) / len(vlist)
            entry[f"{key}_std"] = (sum((v - entry[key])**2 for v in vlist) / len(vlist)) ** 0.5
            entry[f"{key}_n"] = len(vlist)
        averaged[bench][model] = entry

    out_path = Path(files[0]).parent / "averaged_results.json"
    with open(out_path, "w") as f:
        json.dump(averaged, f, indent=2)
    print(f"\nAveraged results saved to {out_path}")


if __name__ == "__main__":
    main()
