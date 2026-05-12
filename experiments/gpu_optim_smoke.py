"""GPU-optimization smoke benchmark: compare baseline vs optimized config.

Trains ONE rens K=32 cell (heat, seed=42, pushforward=False) for 5 epochs
under two configurations:

  1. Baseline:   batch=64,  lr=1e-3, no compile, fp32
  2. Optimized:  batch=256, lr=4e-3, torch.compile, bf16

Reports wallclock + final test MSE for each. Used to validate the
3-5x speedup goal before launching the full Day 1 ablation.

Usage:
    PYTHONPATH=src PYTHONUNBUFFERED=1 uv run --with scikit-learn,scipy \\
        python experiments/gpu_optim_smoke.py
"""
from __future__ import annotations

import argparse
import gc
import time

import torch

from wmca.benchmarks import generate_heat
from wmca.model_registry import create_model, train_model


GRID = 16
N_STEPS = 105
N_TRAJECTORIES = 200
SEED = 42
EPOCHS = 20  # 5 epochs is too short to amortize torch.compile warmup


def eval_mse(model, X_test, Y_test, device="cuda"):
    model = model.to(device).eval()
    total = 0.0
    n = 0
    bs = 256
    with torch.no_grad():
        for i in range(0, len(X_test), bs):
            xb = X_test[i : i + bs].to(device)
            yb = Y_test[i : i + bs].to(device)
            pb = model(xb)
            total += float(((pb - yb) ** 2).mean().item()) * len(xb)
            n += len(xb)
    return total / max(n, 1)


def run_config(
    label: str,
    *,
    batch_size: int,
    lr: float,
    compile_flag: bool,
    bf16_flag: bool,
):
    print(f"\n{'=' * 78}\n{label}\n{'=' * 78}")
    print(f"  batch={batch_size}  lr={lr}  compile={compile_flag}  bf16={bf16_flag}")

    data = generate_heat(
        grid_size=GRID, seed=SEED, n_steps=N_STEPS,
        n_trajectories=N_TRAJECTORIES, device="cuda",
    )
    meta = data.meta
    m = create_model(
        "rescor_rens",
        in_channels=meta["in_channels"],
        out_channels=meta["out_channels"],
        grid_size=GRID,
        seed=SEED,
        cml_K=32,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    m = train_model(
        m, data.X_train, data.Y_train,
        X_val=data.X_val, Y_val=data.Y_val,
        loss_type=meta["loss_type"],
        epochs=EPOCHS,
        batch_size=batch_size, lr=lr,
        device="cuda",
        compile=compile_flag,
        bf16=bf16_flag,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.time() - t0

    mse = eval_mse(m, data.X_test, data.Y_test, device="cuda")
    peak_mem_gb = (torch.cuda.max_memory_allocated() / (1024 ** 3)
                   if torch.cuda.is_available() else 0.0)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    print(f"  -> wallclock: {elapsed:.2f}s   test_mse: {mse:.4e}   "
          f"peak_mem: {peak_mem_gb:.2f} GB")

    del m, data
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {"label": label, "wallclock_s": elapsed, "test_mse": mse,
            "peak_mem_gb": peak_mem_gb}


def main():
    global EPOCHS
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["baseline", "optimized", "both",
                                       "all", "batch_only", "compile_only",
                                       "bf16_only"],
                    default="both")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    args = ap.parse_args()
    EPOCHS = args.epochs

    print(f"PyTorch {torch.__version__}  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")

    results = []
    if args.only in ("baseline", "both", "all", "batch_only",
                     "compile_only", "bf16_only"):
        results.append(run_config(
            "BASELINE (batch=64, fp32, no compile)",
            batch_size=64, lr=1e-3, compile_flag=False, bf16_flag=False,
        ))
    if args.only in ("batch_only", "all"):
        results.append(run_config(
            "BATCH-ONLY (batch=256, lr=4e-3, fp32, no compile)",
            batch_size=256, lr=4e-3, compile_flag=False, bf16_flag=False,
        ))
    if args.only in ("compile_only", "all"):
        results.append(run_config(
            "COMPILE-ONLY (batch=64, fp32, compile=reduce-overhead)",
            batch_size=64, lr=1e-3, compile_flag=True, bf16_flag=False,
        ))
    if args.only in ("bf16_only", "all"):
        results.append(run_config(
            "BF16-ONLY (batch=64, bf16, no compile)",
            batch_size=64, lr=1e-3, compile_flag=False, bf16_flag=True,
        ))
    if args.only in ("optimized", "both", "all"):
        results.append(run_config(
            "OPTIMIZED (batch=128, lr=1.4e-3, bf16, compile=default)",
            batch_size=128, lr=1.4e-3, compile_flag=True, bf16_flag=True,
        ))

    if len(results) >= 2:
        b = results[0]
        print(f"\n{'=' * 78}\nSUMMARY (vs BASELINE @ {b['wallclock_s']:.1f}s, "
              f"mse={b['test_mse']:.4e})\n{'=' * 78}")
        for r in results[1:]:
            speedup = b["wallclock_s"] / max(r["wallclock_s"], 1e-9)
            mse_ratio = r["test_mse"] / max(b["test_mse"], 1e-12)
            print(f"  {r['label'][:44]:44s}  "
                  f"{r['wallclock_s']:7.2f}s  "
                  f"speedup={speedup:5.2f}x  "
                  f"mse={r['test_mse']:.3e}  "
                  f"mse_ratio={mse_ratio:.2f}")


if __name__ == "__main__":
    main()
