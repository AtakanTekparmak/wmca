"""Int8 CML ablation for rescor_e3c.

Tests whether quantizing the CML dynamics to int8 (128 levels) hurts
rescor_e3c performance. Three precision levels:

  - float32 (baseline)
  - bfloat16 (intermediate)
  - int8 (128-level quantization after each CML step)

Benchmarks: KS (strongest PDE result), heat (simplest PDE), GoL (discrete).
3 seeds each.

Usage:
    PYTHONPATH=src python3 experiments/int8_ablation.py --no-wandb
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wmca.benchmarks import BENCHMARKS
from wmca.modules.hybrid import CML2DWithStats, ResidualCorrectionWMv9
from wmca.model_registry import (
    create_model,
    train_model,
    evaluate_model,
    evaluate_rollout,
    param_count,
)

RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Quantized CML2DWithStats
# ---------------------------------------------------------------------------

class CML2DWithStatsQuantized(CML2DWithStats):
    """CML with quantized states during the iteration loop.

    precision='float32' : no quantization (baseline)
    precision='bfloat16': cast to bfloat16 and back each step
    precision='int8'    : round to 128 levels in [0, 1] each step
    """

    def __init__(self, *args, precision: str = "float32", **kwargs):
        super().__init__(*args, **kwargs)
        self.precision = precision

    def forward(self, drive: torch.Tensor) -> dict[str, torch.Tensor]:
        grid = drive
        first = drive
        r, eps, beta = self.r, self.eps, self.beta
        states: list[torch.Tensor] = []

        for _ in range(self.steps):
            mapped = r * grid * (1.0 - grid)
            local = F.conv2d(mapped, self.K_local, padding=1,
                             groups=self.in_channels)
            physics = (1 - eps) * mapped + eps * local
            grid = (1 - beta) * physics + beta * drive
            grid = grid.clamp(1e-4, 1 - 1e-4)

            # Apply quantization
            if self.precision == "int8":
                grid = (grid * 127.0).round() / 127.0
            elif self.precision == "bfloat16":
                grid = grid.to(torch.bfloat16).to(torch.float32)

            states.append(grid)

        last = grid
        stacked = torch.stack(states, dim=0)
        mean = stacked.mean(dim=0)
        var = stacked.var(dim=0, unbiased=False)
        delta = last - first
        last_drive = last - drive

        return {
            "last": last,
            "mean": mean,
            "var": var,
            "delta": delta,
            "last_drive": last_drive,
        }


class ResidualCorrectionWMv9Quantized(nn.Module):
    """rescor_e3c with quantized CML dynamics.

    Drop-in replacement: same architecture, but uses CML2DWithStatsQuantized
    so the CML loop runs at the specified precision.
    """

    def __init__(self, in_channels: int = 1, hidden_ch: int = 32,
                 cml_steps: int = 15,
                 r: float = 3.90, eps: float = 0.3, beta: float = 0.15,
                 seed: int = 42, out_channels: int | None = None,
                 use_sigmoid: bool = True,
                 precision: str = "float32"):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_sigmoid = use_sigmoid

        # Quantized CML
        self.cml_2d = CML2DWithStatsQuantized(
            out_channels, cml_steps, r, eps, beta, seed,
            precision=precision,
        )

        nca_in = in_channels + 5 * out_channels

        self.perceive_d1 = nn.Conv2d(nca_in, hidden_ch, 3, padding=1, dilation=1)
        self.perceive_d2 = nn.Conv2d(nca_in, hidden_ch, 3, padding=2, dilation=2)

        self.dilation_alpha = nn.Parameter(torch.zeros(1, hidden_ch, 1, 1))

        self.update = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(hidden_ch, hidden_ch, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_ch, out_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        state = x[:, : self.out_channels]
        stats = self.cml_2d(state)

        nca_input = torch.cat(
            [
                x,
                stats["last"],
                stats["mean"],
                stats["var"],
                stats["delta"],
                stats["last_drive"],
            ],
            dim=1,
        )

        h1 = self.perceive_d1(nca_input)
        h2 = self.perceive_d2(nca_input) * self.dilation_alpha

        feat = h1 + h2
        correction = self.update(feat)

        out = stats["last"] + correction
        if self.use_sigmoid:
            out = torch.clamp(out, 0, 1)
        return out

    def get_alpha_params(self) -> list[nn.Parameter]:
        return [self.dilation_alpha]

    def param_count(self) -> dict[str, int]:
        trained = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(b.numel() for b in self.cml_2d.buffers())
        return {"trained": trained, "frozen": frozen}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

PRECISIONS = ["float32", "bfloat16", "int8"]
BENCHMARK_NAMES = ["ks", "heat", "gol"]
SEEDS = [42, 123, 777]
ROLLOUT_HORIZONS = [1, 3, 5, 10]


def create_quantized_model(precision: str, in_channels: int = 1,
                           out_channels: int = 1, seed: int = 42):
    """Create a rescor_e3c model with the given CML precision."""
    return ResidualCorrectionWMv9Quantized(
        in_channels=in_channels,
        out_channels=out_channels,
        seed=seed,
        precision=precision,
        use_sigmoid=(out_channels == in_channels),
    )


def run_ablation(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()

    results = defaultdict(dict)

    for bench_name in BENCHMARK_NAMES:
        print(f"{'='*70}")
        print(f"  Benchmark: {bench_name}")
        print(f"{'='*70}")

        gen_fn = BENCHMARKS[bench_name]
        # Use smaller data for faster runs
        gen_kwargs = {"device": device}
        if bench_name == "ks":
            gen_kwargs["n_trajectories"] = 200
            gen_kwargs["n_steps"] = 100
            gen_kwargs["grid_size"] = 64
        elif bench_name == "heat":
            gen_kwargs["n_trajectories"] = 500
            gen_kwargs["n_steps"] = 50
            gen_kwargs["grid_size"] = 32
        elif bench_name == "gol":
            gen_kwargs["n_trajectories"] = 1000
            gen_kwargs["n_steps"] = 20
            gen_kwargs["grid_size"] = 32

        data = gen_fn(**gen_kwargs)
        meta = data.meta
        in_ch = meta["in_channels"]
        out_ch = meta["out_channels"]
        loss_type = meta["loss_type"]

        for prec in PRECISIONS:
            seed_results = []
            for seed in SEEDS:
                label = f"{bench_name}/{prec}/seed={seed}"
                print(f"\n--- {label} ---")
                t0 = time.time()

                model = create_quantized_model(
                    precision=prec,
                    in_channels=in_ch,
                    out_channels=out_ch,
                    seed=seed,
                )
                pc = model.param_count()
                print(f"  Params: trained={pc['trained']}, frozen={pc['frozen']}")

                # Train
                epochs = 30
                model = train_model(
                    model, data.X_train, data.Y_train,
                    X_val=data.X_val, Y_val=data.Y_val,
                    loss_type=loss_type, epochs=epochs,
                    batch_size=64, lr=1e-3, device=device,
                )

                # 1-step eval
                metrics_1step = evaluate_model(
                    model, data.X_test, data.Y_test,
                    loss_type=loss_type, device=device,
                )
                print(f"  1-step: {metrics_1step}")

                # Rollout eval
                model_dev = model.to(device)
                rollout = evaluate_rollout(
                    model_dev, data,
                    horizons=ROLLOUT_HORIZONS,
                    benchmark_name=bench_name,
                    device=device,
                )
                print(f"  Rollout: {rollout}")

                elapsed = time.time() - t0
                print(f"  Time: {elapsed:.1f}s")

                seed_results.append({
                    "seed": seed,
                    "1step": metrics_1step,
                    "rollout": {str(k): v for k, v in rollout.items()},
                    "time_s": elapsed,
                    "param_count": pc,
                })

                # Cleanup
                del model, model_dev
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            results[bench_name][prec] = seed_results

        # Free benchmark data
        del data
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("  INT8 ABLATION SUMMARY")
    print(f"{'='*70}\n")

    for bench_name in BENCHMARK_NAMES:
        print(f"\n--- {bench_name} ---")
        primary_metric = "mse" if bench_name in ("ks", "heat") else "accuracy"

        header = f"{'Precision':<12}"
        header += f"{'1-step (mean+-std)':<25}"
        for h in ROLLOUT_HORIZONS:
            header += f"{'h='+str(h):<15}"
        print(header)
        print("-" * len(header))

        for prec in PRECISIONS:
            seed_data = results[bench_name][prec]
            vals_1step = [s["1step"][primary_metric] for s in seed_data]
            mean_1 = np.mean(vals_1step)
            std_1 = np.std(vals_1step)

            row = f"{prec:<12}"
            row += f"{mean_1:.6f} +- {std_1:.6f}  "

            for h in ROLLOUT_HORIZONS:
                h_key = str(h)
                h_vals = [s["rollout"].get(h_key, float("nan")) for s in seed_data]
                h_mean = np.mean(h_vals)
                row += f"{h_mean:.6f}     "

            print(row)

    # Save JSON
    out_path = RESULTS_DIR / "int8_ablation.json"
    # Convert for JSON serialization
    serializable = {}
    for bname, prec_dict in results.items():
        serializable[bname] = {}
        for prec, slist in prec_dict.items():
            serializable[bname][prec] = slist
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    # -----------------------------------------------------------------------
    # Key finding check
    # -----------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("  KEY FINDINGS")
    print(f"{'='*70}")

    for bench_name in BENCHMARK_NAMES:
        primary_metric = "mse" if bench_name in ("ks", "heat") else "accuracy"
        f32_vals = [s["1step"][primary_metric] for s in results[bench_name]["float32"]]
        int8_vals = [s["1step"][primary_metric] for s in results[bench_name]["int8"]]
        bf16_vals = [s["1step"][primary_metric] for s in results[bench_name]["bfloat16"]]

        f32_mean = np.mean(f32_vals)
        int8_mean = np.mean(int8_vals)
        bf16_mean = np.mean(bf16_vals)

        if primary_metric == "mse":
            degradation_int8 = (int8_mean - f32_mean) / (f32_mean + 1e-10) * 100
            degradation_bf16 = (bf16_mean - f32_mean) / (f32_mean + 1e-10) * 100
            print(f"\n{bench_name} ({primary_metric}):")
            print(f"  float32:  {f32_mean:.6f}")
            print(f"  bfloat16: {bf16_mean:.6f} ({degradation_bf16:+.1f}%)")
            print(f"  int8:     {int8_mean:.6f} ({degradation_int8:+.1f}%)")
            viable = abs(degradation_int8) < 10
            print(f"  -> Int8 viable: {'YES' if viable else 'NO'} (threshold: <10% degradation)")
        else:
            degradation_int8 = (f32_mean - int8_mean) / (f32_mean + 1e-10) * 100
            degradation_bf16 = (f32_mean - bf16_mean) / (f32_mean + 1e-10) * 100
            print(f"\n{bench_name} ({primary_metric}):")
            print(f"  float32:  {f32_mean:.6f}")
            print(f"  bfloat16: {bf16_mean:.6f} ({-degradation_bf16:+.1f}%)")
            print(f"  int8:     {int8_mean:.6f} ({-degradation_int8:+.1f}%)")
            viable = abs(degradation_int8) < 5
            print(f"  -> Int8 viable: {'YES' if viable else 'NO'} (threshold: <5% accuracy drop)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Int8 CML ablation for rescor_e3c")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb")
    args = parser.parse_args()

    run_ablation(args)
