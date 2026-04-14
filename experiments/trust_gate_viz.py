"""Trust Gate Visualization: Shows the Matching Principle is learnable.

For each benchmark, trains a rescor_mp_gate model and visualizes where
the trust gate opens (trusts CML physics) vs closes (falls back to NCA).

Generates three figures:
  1. Per-benchmark gate histogram grid
  2. Spatial gate heatmap grid (one test example per benchmark)
  3. Bar chart of mean trust gate value per benchmark

Usage:
    uv run --with scikit-learn,matplotlib,scipy python experiments/trust_gate_viz.py
"""

from __future__ import annotations

import gc
import os
import sys
from pathlib import Path

# Force unbuffered output
os.environ["PYTHONUNBUFFERED"] = "1"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch
import torch.nn.functional as F

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wmca.benchmarks import BENCHMARKS, BenchmarkData
from wmca.model_registry import create_model, train_model

# ── Config ────────────────────────────────────────────────────────────────────

BENCHMARKS_TO_VIZ = [
    "heat", "ks", "gray_scott", "gol",
    "rule110", "wireworld", "crafter_lite", "minigrid",
]

SEED = 42
EPOCHS = 30
BATCH_SIZE = 64
LR = 1e-3
DEVICE = "cpu"

PLOT_DIR = PROJECT_ROOT / "experiments" / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)


# ── Trust gate extraction ────────────────────────────────────────────────────

def extract_trust_gate(model, X_batch: torch.Tensor) -> torch.Tensor:
    """Extract per-cell trust gate values from a trained MP-Gate model.

    Args:
        model: A trained MatchingPrincipleGateWM instance.
        X_batch: Input tensor (B, C_in, H, W).

    Returns:
        trust: Tensor of shape (B, 1, H, W) with values in [0, 1].
    """
    model.eval()
    with torch.no_grad():
        state = X_batch[:, :model.out_channels]
        stats = model.cml_2d(state)
        gate_input = torch.cat([stats["var"], stats["last_drive"]], dim=1)
        trust = torch.sigmoid(model.trust_gate(gate_input))
    return trust  # (B, 1, H, W)


# ── Per-benchmark pipeline ───────────────────────────────────────────────────

def run_benchmark(name: str) -> dict:
    """Generate data, train model, extract trust gate values for one benchmark.

    Returns dict with keys: name, trust_all, trust_spatial, input_spatial, meta.
    """
    print(f"\n{'='*60}")
    print(f"  Benchmark: {name}")
    print(f"{'='*60}")

    # Generate data
    print(f"  Generating data...")
    gen_fn = BENCHMARKS[name]
    data: BenchmarkData = gen_fn(seed=SEED, device=DEVICE)
    meta = data.meta

    in_ch = meta["in_channels"]
    out_ch = meta["out_channels"]
    loss_type = meta["loss_type"]
    grid_size = meta.get("grid_size", meta.get("grid_width", 16))

    print(f"  in_ch={in_ch}, out_ch={out_ch}, loss={loss_type}, grid={grid_size}")
    print(f"  Train: {data.X_train.shape}, Test: {data.X_test.shape}")

    # Create model
    model = create_model(
        "rescor_mp_gate",
        in_channels=in_ch,
        out_channels=out_ch,
        grid_size=grid_size,
        seed=SEED,
    )
    n_params = model.param_count()
    print(f"  Model params: {n_params}")

    # Train
    print(f"  Training ({EPOCHS} epochs)...")
    model = train_model(
        model,
        data.X_train, data.Y_train,
        X_val=data.X_val, Y_val=data.Y_val,
        loss_type=loss_type,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR,
        device=DEVICE,
    )
    print(f"  Training complete.")

    # Extract trust gate on test set (limit to 256 samples for speed)
    X_test = data.X_test
    if isinstance(X_test, np.ndarray):
        X_test = torch.from_numpy(X_test).float()
    X_test = X_test[:256]

    trust_all = extract_trust_gate(model, X_test)  # (B, 1, H, W)
    trust_np = trust_all.numpy()

    mean_trust = trust_np.mean()
    std_trust = trust_np.std()
    print(f"  Trust gate: mean={mean_trust:.4f}, std={std_trust:.4f}")
    print(f"  Trust range: [{trust_np.min():.4f}, {trust_np.max():.4f}]")

    # Pick one example for spatial heatmap (first test sample)
    trust_spatial = trust_np[0, 0]  # (H, W)
    input_spatial = X_test[0].numpy()  # (C_in, H, W)

    # Cleanup
    del model, data, X_test, trust_all
    gc.collect()

    return {
        "name": name,
        "trust_all": trust_np,   # (B, 1, H, W)
        "trust_spatial": trust_spatial,  # (H, W)
        "input_spatial": input_spatial,  # (C_in, H, W)
        "meta": meta,
        "mean_trust": mean_trust,
        "std_trust": std_trust,
    }


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_histograms(results: list[dict], save_prefix: str = "trust_gate"):
    """Figure 1: Per-benchmark gate histogram grid."""
    n = len(results)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.0 * nrows))
    axes = np.atleast_2d(axes)

    for idx, res in enumerate(results):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]

        vals = res["trust_all"].ravel()
        ax.hist(vals, bins=50, range=(0, 1), density=True,
                color="#4C72B0", edgecolor="white", linewidth=0.3, alpha=0.85)

        # Mark mean
        mean_v = res["mean_trust"]
        ax.axvline(mean_v, color="#C44E52", linestyle="--", linewidth=1.5,
                   label=f"mean={mean_v:.3f}")

        ax.set_xlim(0, 1)
        ax.set_title(res["name"], fontsize=11, fontweight="bold")
        ax.set_xlabel("Trust gate value g", fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.legend(fontsize=8, loc="upper left")
        ax.tick_params(labelsize=8)

    # Hide unused subplots
    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    fig.suptitle(
        "Trust Gate Distribution: g=1 trusts CML, g=0 trusts NCA",
        fontsize=13, fontweight="bold", y=1.02,
    )
    fig.tight_layout()

    for ext in ("png", "pdf"):
        path = PLOT_DIR / f"{save_prefix}_histograms.{ext}"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close(fig)


def plot_spatial_heatmaps(results: list[dict], save_prefix: str = "trust_gate"):
    """Figure 2: Spatial gate heatmap for one test example per benchmark."""
    n = len(results)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols * 2, figsize=(3.0 * ncols * 2, 3.0 * nrows))
    axes = np.atleast_2d(axes)

    cmap = plt.cm.RdBu  # blue=trust NCA (low), red=trust CML (high)
    norm = mcolors.Normalize(vmin=0, vmax=1)

    for idx, res in enumerate(results):
        row = idx // ncols
        col_base = (idx % ncols) * 2

        trust_map = res["trust_spatial"]  # (H, W)
        input_arr = res["input_spatial"]  # (C_in, H, W)
        is_1d = trust_map.shape[0] == 1

        # Left panel: input state (first channel or sum)
        ax_inp = axes[row, col_base]
        if input_arr.shape[0] == 1:
            inp_vis = input_arr[0]
        elif input_arr.shape[0] <= 3:
            # For 2-channel: show channel 0
            inp_vis = input_arr[0]
        else:
            # Multi-channel (wireworld, crafter, minigrid): show argmax
            inp_vis = input_arr.argmax(axis=0).astype(np.float32)

        if is_1d:
            ax_inp.imshow(np.tile(inp_vis, (8, 1)), cmap="viridis", aspect="auto")
        else:
            ax_inp.imshow(inp_vis, cmap="viridis", aspect="equal")
        ax_inp.set_title(f"{res['name']}\n(input)", fontsize=9, fontweight="bold")
        ax_inp.set_xticks([])
        ax_inp.set_yticks([])

        # Right panel: trust gate heatmap
        ax_gate = axes[row, col_base + 1]
        if is_1d:
            im = ax_gate.imshow(np.tile(trust_map, (8, 1)), cmap=cmap,
                                norm=norm, aspect="auto")
        else:
            im = ax_gate.imshow(trust_map, cmap=cmap, norm=norm, aspect="equal")
        ax_gate.set_title(f"{res['name']}\n(trust gate)", fontsize=9, fontweight="bold")
        ax_gate.set_xticks([])
        ax_gate.set_yticks([])

    # Hide unused subplots
    for idx in range(n, nrows * ncols):
        row = idx // ncols
        col_base = (idx % ncols) * 2
        axes[row, col_base].set_visible(False)
        axes[row, col_base + 1].set_visible(False)

    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)
    cb.set_label("Trust gate g", fontsize=10)
    cb.ax.tick_params(labelsize=8)

    fig.suptitle(
        "Spatial Trust Gate Maps: Blue=NCA, Red=CML",
        fontsize=13, fontweight="bold", y=1.02,
    )
    fig.tight_layout(rect=[0, 0, 0.91, 1.0])

    for ext in ("png", "pdf"):
        path = PLOT_DIR / f"{save_prefix}_spatial.{ext}"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close(fig)


def plot_bar_chart(results: list[dict], save_prefix: str = "trust_gate"):
    """Figure 3: Mean trust gate value per benchmark, sorted."""
    # Sort by mean trust descending (CML-trusting first)
    sorted_res = sorted(results, key=lambda r: r["mean_trust"], reverse=True)

    names = [r["name"] for r in sorted_res]
    means = [r["mean_trust"] for r in sorted_res]
    stds = [r["std_trust"] for r in sorted_res]

    fig, ax = plt.subplots(figsize=(8, 4.5))

    # Color bars by trust level: high=red (CML), low=blue (NCA)
    cmap = plt.cm.RdBu
    colors = [cmap(m) for m in means]

    bars = ax.barh(range(len(names)), means, xerr=stds,
                   color=colors, edgecolor="black", linewidth=0.5,
                   capsize=3, error_kw={"linewidth": 1.0})

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=11, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Mean Trust Gate Value g", fontsize=12)
    ax.set_title(
        "Matching Principle: Per-Benchmark CML Trust",
        fontsize=13, fontweight="bold",
    )

    # Add value labels
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(min(m + s + 0.02, 0.95), i, f"{m:.3f}",
                va="center", fontsize=10, fontweight="bold")

    # Reference lines
    ax.axvline(0.5, color="gray", linestyle=":", linewidth=1.0, alpha=0.7)
    ax.text(0.51, len(names) - 0.5, "g=0.5\n(equal blend)",
            fontsize=8, color="gray", va="top")

    # Annotations
    ax.annotate("Trust CML (physics)", xy=(0.85, -0.08),
                xycoords="axes fraction", fontsize=9, color="#C44E52",
                fontweight="bold", ha="center")
    ax.annotate("Trust NCA (learned)", xy=(0.15, -0.08),
                xycoords="axes fraction", fontsize=9, color="#4C72B0",
                fontweight="bold", ha="center")

    ax.invert_yaxis()
    fig.tight_layout()

    for ext in ("png", "pdf"):
        path = PLOT_DIR / f"{save_prefix}_bar.{ext}"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("Trust Gate Visualization")
    print("========================")
    print(f"Benchmarks: {BENCHMARKS_TO_VIZ}")
    print(f"Seed: {SEED}, Epochs: {EPOCHS}, Device: {DEVICE}")
    print()

    results = []
    for bm_name in BENCHMARKS_TO_VIZ:
        if bm_name not in BENCHMARKS:
            print(f"  WARNING: benchmark '{bm_name}' not in registry, skipping.")
            continue
        res = run_benchmark(bm_name)
        results.append(res)

    if not results:
        print("No benchmarks ran successfully.")
        return

    # Summary
    print(f"\n{'='*60}")
    print("  Summary")
    print(f"{'='*60}")
    for r in sorted(results, key=lambda x: x["mean_trust"], reverse=True):
        print(f"  {r['name']:15s}  mean_trust={r['mean_trust']:.4f}  std={r['std_trust']:.4f}")

    # Generate figures
    print(f"\nGenerating figures...")
    plot_histograms(results)
    plot_spatial_heatmaps(results)
    plot_bar_chart(results)

    print(f"\nDone. All figures saved to {PLOT_DIR}")


if __name__ == "__main__":
    main()
