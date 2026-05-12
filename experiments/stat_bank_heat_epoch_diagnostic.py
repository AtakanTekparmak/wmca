"""Epoch diagnostic: is stat-bank's catastrophic heat regression an optimization issue?

Hypothesis: with 5 input channels (3 of which are near-zero-but-noisy on heat's
smooth dynamics), the NCA needs more epochs to drive extraneous-feature weights
to zero. At 30 epochs it's under-converged.

Run: rescor_rens_stat_full on HEAT only, at epochs ∈ {30, 60, 100, 150}.
Baseline (not re-run, from prior data): rescor_rens K=32 heat at 30 epochs = 4.87e-9.

If MSE monotonically decreases toward ~5e-9 with more epochs, stat-bank just
needs more training. If it plateaus above rens K=32 even at 150, structural.

Usage:
    PYTHONUNBUFFERED=1 uv run --with scikit-learn,scipy python experiments/stat_bank_heat_epoch_diagnostic.py
"""
import gc
import json
import time
from pathlib import Path

import torch

from wmca.benchmarks import generate_heat
from wmca.model_registry import create_model, train_model


EPOCHS_LIST = [30, 60, 100, 150]
SEED = 42


def evaluate_model(model, X_test, Y_test):
    model.eval()
    with torch.no_grad():
        preds = model(X_test)
    return ((preds - Y_test) ** 2).mean().item()


def run_one(epochs):
    data = generate_heat(grid_size=16, seed=SEED)
    meta = data.meta
    m = create_model(
        "rescor_rens_stat_full",
        in_channels=meta["in_channels"],
        out_channels=meta["out_channels"],
        grid_size=16, seed=SEED, cml_K=32,
    )
    pc = m.param_count()
    t0 = time.time()
    m = train_model(
        m, data.X_train, data.Y_train,
        X_val=data.X_val, Y_val=data.Y_val,
        loss_type="mse", epochs=epochs, batch_size=64, lr=1e-3,
    )
    mse = evaluate_model(m, data.X_test, data.Y_test)
    elapsed = time.time() - t0
    return {"epochs": epochs, "mse": mse, "time": elapsed, "params": pc}


def main():
    print("=" * 75)
    print("rescor_rens_stat_full on HEAT — epoch convergence diagnostic")
    print("Reference: rescor_rens K=32 at 30 epochs = 4.87e-9")
    print("=" * 75)
    results = []
    for ep in EPOCHS_LIST:
        r = run_one(ep)
        results.append(r)
        print(f"  epochs={ep:>3d}  MSE={r['mse']:.6e}  [{r['time']:.0f}s]  params={r['params']}")
        gc.collect()

    print()
    print("=" * 75)
    print("Convergence trajectory")
    print("=" * 75)
    baseline = 4.868374e-09
    print(f"{'epochs':>8s}  {'MSE':>15s}  {'ratio vs rens K=32':>22s}")
    for r in results:
        ratio = r["mse"] / baseline
        print(f"{r['epochs']:>8d}  {r['mse']:>15.6e}  {ratio:>18.1f}×")

    out_path = Path("experiments/results/stat_bank_heat_epoch_diagnostic.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"baseline_rens_K32_heat_30ep": baseline, "results": results}, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
