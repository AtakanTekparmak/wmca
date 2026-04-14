"""Phase 2.5a: Chaotic Map Generalization - Does the Matching Principle hold
across different chaotic maps?

Tests 4 chaotic maps as the fixed reservoir in Variant D (ResidualCorrection):
  1. Logistic map (baseline): f(x) = r*x*(1-x), r=3.90
  2. Sine map: f(x) = (r/4)*sin(pi*x) + 0.5
  3. Tent map: f(x) = r*min(x, 1-x), r=1.95
  4. Bernoulli map: f(x) = (2*x) mod 1.0

For each map, 3 models on 2 benchmarks:
  - ResidualCorrection(D) with that map's CML2D
  - PureNCA (map-independent baseline)
  - Conv2D (map-independent baseline)

Benchmarks: Heat equation + GoL at 16x16.

Key question: If the Matching Principle holds, ALL chaotic maps should work
similarly for continuous targets (heat) and ALL should fail similarly for
discrete targets (GoL).

Usage:
    FORCE_CPU=1 uv run --with scikit-learn,matplotlib,scipy python experiments/phase25a_chaotic_maps.py --no-wandb
"""
from __future__ import annotations

import argparse
import gc
import math
import os
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wmca.modules.hybrid import PureNCA
from wmca.utils import pick_device


def _get_plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


# ===== Constants ===========================================================
PLOTS_DIR = PROJECT_ROOT / "experiments" / "plots"
SEED = 42

GRID_H, GRID_W = 16, 16

# Heat PDE parameters
ALPHA_HEAT = 0.1
DT = 0.01
DX = 1.0 / GRID_H
_HEAT_COEFF = ALPHA_HEAT * DT / (DX * DX)
_LAP_KERNEL_NP = np.array([[0., 1., 0.],
                            [1., -4., 1.],
                            [0., 1., 0.]], dtype=np.float32)

# GoL
GOL_DENSITY = 0.3

# Dataset sizes
HEAT_N_TRAJ = 300
HEAT_TRAJ_LEN = 30
GOL_N_TRAJ = 500
GOL_TRAJ_LEN = 20

# Training
LR = 1e-3
EPOCHS = 30
BATCH_SIZE = 64

# CML defaults
CML_STEPS = 15
CML_EPS = 0.3
CML_BETA = 0.15

# Rollout
ROLLOUT_HORIZONS = [1, 3, 5, 10]


# ===== Chaotic Map Definitions =============================================

def logistic_map(grid: torch.Tensor, r: float = 3.90) -> torch.Tensor:
    """f(x) = r*x*(1-x)"""
    return r * grid * (1.0 - grid)


def sine_map(grid: torch.Tensor, r: float = 3.90) -> torch.Tensor:
    """f(x) = (r/4)*sin(pi*x) + 0.5, keeps output in [0,1] for x in [0,1]."""
    # sin(pi*x) peaks at 1 when x=0.5, so (r/4)*sin(pi*x) in [0, r/4]
    # For r=3.90, max = 0.975; adding 0.5 would exceed 1.
    # Instead: f(x) = (r/4)*sin(pi*x) which maps [0,1]->[0,r/4]~[0,0.975]
    # This is the standard sine map analog of logistic.
    return (r / 4.0) * torch.sin(math.pi * grid)


def tent_map(grid: torch.Tensor, r: float = 1.95) -> torch.Tensor:
    """f(x) = r*min(x, 1-x), r in [0,2], chaotic near r=2."""
    return r * torch.min(grid, 1.0 - grid)


def bernoulli_map(grid: torch.Tensor, **kwargs) -> torch.Tensor:
    """f(x) = (2*x) mod 1.0. Fully chaotic, uniform invariant measure."""
    return torch.fmod(2.0 * grid, 1.0)


CHAOTIC_MAPS = {
    "Logistic":  logistic_map,
    "Sine":      sine_map,
    "Tent":      tent_map,
    "Bernoulli": bernoulli_map,
}


# ===== CML2D Variant with pluggable map ====================================

class CML2DVariant(nn.Module):
    """2D Coupled Map Lattice with a pluggable chaotic map function."""

    def __init__(self, map_fn: Callable, in_channels: int = 1,
                 steps: int = 15, eps: float = 0.3, beta: float = 0.15,
                 seed: int = 42):
        super().__init__()
        self.map_fn = map_fn
        self.in_channels = in_channels
        self.steps = steps

        self.register_buffer("eps", torch.tensor(eps))
        self.register_buffer("beta", torch.tensor(beta))

        rng = torch.Generator().manual_seed(seed)
        K_raw = torch.rand(in_channels, 1, 3, 3, generator=rng).abs()
        K_norm = K_raw / K_raw.sum(dim=(-1, -2), keepdim=True)
        self.register_buffer("K_local", K_norm)

    def forward(self, drive: torch.Tensor) -> torch.Tensor:
        grid = drive
        eps, beta = self.eps, self.beta
        for _ in range(self.steps):
            mapped = self.map_fn(grid)
            local = F.conv2d(mapped, self.K_local, padding=1,
                             groups=self.in_channels)
            physics = (1 - eps) * mapped + eps * local
            grid = (1 - beta) * physics + beta * drive
        return grid.clamp(1e-4, 1 - 1e-4)

    def param_count(self) -> dict[str, int]:
        return {"trained": 0, "frozen": sum(b.numel() for b in self.buffers())}


class ResidualCorrectionVariant(nn.Module):
    """CML (with pluggable map) provides base prediction, NCA learns correction."""

    def __init__(self, map_fn: Callable, in_channels: int = 1,
                 hidden_ch: int = 16, cml_steps: int = 15,
                 eps: float = 0.3, beta: float = 0.15, seed: int = 42):
        super().__init__()

        self.cml_2d = CML2DVariant(map_fn, in_channels, cml_steps,
                                    eps, beta, seed)

        self.nca = nn.Sequential(
            nn.Conv2d(in_channels * 2, hidden_ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_ch, in_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cml_out = self.cml_2d(x)
        correction = self.nca(torch.cat([x, cml_out], dim=1))
        return torch.clamp(cml_out + correction, 0, 1)

    def param_count(self) -> dict[str, int]:
        trained = sum(p.numel() for p in self.parameters())
        frozen = sum(b.numel() for b in self.cml_2d.buffers())
        return {"trained": trained, "frozen": frozen}


class Conv2DBaseline(nn.Module):
    """3-layer CNN: 1->16->16->1."""

    def __init__(self, use_sigmoid: bool = False):
        super().__init__()
        layers = [
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, 3, padding=1),
        ]
        if use_sigmoid:
            layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def param_count(self) -> dict[str, int]:
        return {
            "trained": sum(p.numel() for p in self.parameters()),
            "frozen": 0,
        }


# ===== Data Generation (copied from phase2_ablation.py) ====================

def heat_step(u: np.ndarray) -> np.ndarray:
    from scipy.signal import convolve2d
    lap = convolve2d(u, _LAP_KERNEL_NP, mode="same", boundary="fill", fillvalue=0.0)
    u_new = u + _HEAT_COEFF * lap
    return np.clip(u_new, 0.0, 1.0)


def _random_gaussian_blob(h: int, w: int, rng: np.random.RandomState) -> np.ndarray:
    cy, cx = rng.uniform(0.2, 0.8) * h, rng.uniform(0.2, 0.8) * w
    sigma = rng.uniform(1.5, 4.0)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    return np.exp(-((yy - cy)**2 + (xx - cx)**2) / (2 * sigma**2))


def generate_heat_ic(h: int, w: int, rng: np.random.RandomState) -> np.ndarray:
    n_blobs = rng.randint(2, 6)
    u0 = np.zeros((h, w), dtype=np.float32)
    for _ in range(n_blobs):
        u0 += _random_gaussian_blob(h, w, rng) * rng.uniform(0.3, 1.0)
    u_max = u0.max()
    if u_max > 1e-8:
        u0 = u0 / u_max
    return u0


def generate_heat_trajectories() -> np.ndarray:
    rng = np.random.RandomState(SEED)
    trajs = np.zeros((HEAT_N_TRAJ, HEAT_TRAJ_LEN + 1, GRID_H, GRID_W), dtype=np.float32)
    for i in range(HEAT_N_TRAJ):
        u = generate_heat_ic(GRID_H, GRID_W, rng)
        trajs[i, 0] = u
        for t in range(HEAT_TRAJ_LEN):
            u = heat_step(u)
            trajs[i, t + 1] = u
    return trajs


def gol_step(grid: np.ndarray) -> np.ndarray:
    from scipy.signal import convolve2d
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.float32)
    neighbors = convolve2d(grid.astype(np.float32), kernel, mode="same", boundary="wrap")
    born = (grid == 0) & (neighbors == 3)
    survive = (grid == 1) & ((neighbors == 2) | (neighbors == 3))
    return (born | survive).astype(np.float32)


def generate_gol_trajectories() -> np.ndarray:
    rng = np.random.RandomState(SEED)
    trajs = np.zeros((GOL_N_TRAJ, GOL_TRAJ_LEN + 1, GRID_H, GRID_W), dtype=np.float32)
    for i in range(GOL_N_TRAJ):
        grid = (rng.rand(GRID_H, GRID_W) < GOL_DENSITY).astype(np.float32)
        trajs[i, 0] = grid
        for t in range(GOL_TRAJ_LEN):
            grid = gol_step(grid)
            trajs[i, t + 1] = grid
    return trajs


def make_pairs(trajs: np.ndarray):
    X = trajs[:, :-1].reshape(-1, GRID_H, GRID_W)
    Y = trajs[:, 1:].reshape(-1, GRID_H, GRID_W)
    return X, Y


def split_trajectories(trajs: np.ndarray):
    n = len(trajs)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    return trajs[:n_train], trajs[n_train:n_train + n_val], trajs[n_train + n_val:]


# ===== Training =============================================================

def train_model(model: nn.Module, X_train: np.ndarray, Y_train: np.ndarray,
                X_val: np.ndarray, Y_val: np.ndarray,
                loss_type: str = "mse",
                epochs: int = EPOCHS, lr: float = LR,
                batch_size: int = BATCH_SIZE,
                model_name: str = "",
                benchmark_name: str = "") -> tuple[nn.Module, float]:
    device = torch.device("cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if loss_type == "mse":
        criterion = nn.MSELoss()
    else:
        criterion = nn.BCELoss()

    X_tr = torch.from_numpy(X_train).float().unsqueeze(1).to(device)
    Y_tr = torch.from_numpy(Y_train).float().unsqueeze(1).to(device)
    X_v = torch.from_numpy(X_val).float().unsqueeze(1).to(device)
    Y_v = torch.from_numpy(Y_val).float().unsqueeze(1).to(device)

    best_val_loss = float("inf")
    best_state: dict | None = None

    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(X_tr), device=device)
        total_loss = 0.0
        n_batches = 0
        for i in range(0, len(perm), batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = X_tr[idx], Y_tr[idx]
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss_sum = 0.0
            val_n = 0
            for vi in range(0, len(X_v), batch_size):
                vx = X_v[vi:vi + batch_size]
                vy = Y_v[vi:vi + batch_size]
                vp = model(vx)
                val_loss_sum += criterion(vp, vy).item() * len(vx)
                val_n += len(vx)
            val_loss = val_loss_sum / val_n

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0:
            print(f"    [{benchmark_name}|{model_name}] Epoch {epoch+1:3d}/{epochs}  "
                  f"train={total_loss / n_batches:.6f}  val={val_loss:.6f}")

    train_time = time.time() - t0
    model.load_state_dict(best_state)
    model = model.to(torch.device("cpu"))
    print(f"    Best val_loss: {best_val_loss:.6f}  ({train_time:.1f}s)")

    del X_tr, Y_tr, X_v, Y_v
    gc.collect()
    return model, train_time


# ===== Evaluation ===========================================================

def mse_metric(Y_true: np.ndarray, Y_pred: np.ndarray) -> float:
    return float(np.mean((Y_true - Y_pred) ** 2))


def cell_accuracy(Y_true: np.ndarray, Y_pred: np.ndarray) -> float:
    pred_binary = (Y_pred >= 0.5).astype(np.float32)
    return float(np.mean(pred_binary == Y_true))


def evaluate_spatial(model: nn.Module, X_test: np.ndarray, Y_test: np.ndarray):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X_test), BATCH_SIZE):
            xb = torch.from_numpy(X_test[i:i+BATCH_SIZE]).float().unsqueeze(1)
            pb = model(xb).squeeze(1).numpy()
            preds.append(pb)
    return np.concatenate(preds, axis=0)


def make_predict_fn(model: nn.Module):
    model.eval()
    def _predict(x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x_t = torch.from_numpy(x).float().reshape(1, 1, GRID_H, GRID_W)
            out = model(x_t).squeeze().numpy()
        return np.clip(out, 0, 1)
    return _predict


def multistep_rollout(predict_fn, x0: np.ndarray, n_steps: int,
                      binarize: bool = False) -> np.ndarray:
    preds = np.zeros((n_steps, GRID_H, GRID_W), dtype=np.float32)
    x = x0.copy()
    for t in range(n_steps):
        raw = predict_fn(x)
        preds[t] = raw
        if binarize:
            x = (np.clip(raw, 0, 1) >= 0.5).astype(np.float32)
        else:
            x = np.clip(raw, 0, 1).astype(np.float32)
    return preds


# ===== Per-Benchmark Runner =================================================

def run_benchmark(benchmark_name: str, trajs: np.ndarray,
                  loss_type: str, binarize_rollout: bool):
    print(f"\n{'=' * 72}")
    print(f"  BENCHMARK: {benchmark_name}")
    print(f"{'=' * 72}")

    train_t, val_t, test_t = split_trajectories(trajs)
    X_train, Y_train = make_pairs(train_t)
    X_val, Y_val = make_pairs(val_t)
    X_test, Y_test = make_pairs(test_t)
    print(f"  Train pairs: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    use_sigmoid = (loss_type == "bce")

    results = {}
    predict_fns = {}

    # --- Map-independent baselines (train once) ---
    for base_name, base_factory in [
        ("PureNCA", lambda: PureNCA(in_channels=1, hidden_ch=16, steps=1)),
        ("Conv2D", lambda: Conv2DBaseline(use_sigmoid=use_sigmoid)),
    ]:
        print(f"\n  --- {base_name} (map-independent baseline) ---")
        model = base_factory()
        model, train_time = train_model(
            model, X_train, Y_train, X_val, Y_val,
            loss_type=loss_type, model_name=base_name,
            benchmark_name=benchmark_name,
        )
        preds = evaluate_spatial(model, X_test, Y_test)
        pc = model.param_count()

        if loss_type == "mse":
            one_step = mse_metric(Y_test, preds)
        else:
            one_step = cell_accuracy(Y_test, preds)

        print(f"    1-step metric: {one_step:.6f}")
        print(f"    Params: trained={pc['trained']}, frozen={pc['frozen']}")

        results[base_name] = {
            "one_step": one_step,
            "params_trained": pc["trained"],
            "params_frozen": pc["frozen"],
            "train_time": train_time,
        }
        predict_fns[base_name] = make_predict_fn(model)

    # --- ResidualCorrection with each chaotic map ---
    for map_name, map_fn in CHAOTIC_MAPS.items():
        model_label = f"ResCor({map_name})"
        print(f"\n  --- {model_label} ---")

        model = ResidualCorrectionVariant(
            map_fn=map_fn, in_channels=1, hidden_ch=16,
            cml_steps=CML_STEPS, eps=CML_EPS, beta=CML_BETA, seed=SEED,
        )
        model, train_time = train_model(
            model, X_train, Y_train, X_val, Y_val,
            loss_type=loss_type, model_name=model_label,
            benchmark_name=benchmark_name,
        )
        preds = evaluate_spatial(model, X_test, Y_test)
        pc = model.param_count()

        if loss_type == "mse":
            one_step = mse_metric(Y_test, preds)
        else:
            one_step = cell_accuracy(Y_test, preds)

        print(f"    1-step metric: {one_step:.6f}")
        print(f"    Params: trained={pc['trained']}, frozen={pc['frozen']}")

        results[model_label] = {
            "one_step": one_step,
            "params_trained": pc["trained"],
            "params_frozen": pc["frozen"],
            "train_time": train_time,
        }
        predict_fns[model_label] = make_predict_fn(model)

    # --- Multi-step rollout ---
    print(f"\n  --- Multi-step rollout (horizons {ROLLOUT_HORIZONS}) ---")
    n_rollout = min(20, len(test_t))
    rollout_results = {h: {} for h in ROLLOUT_HORIZONS}

    for name, pfn in predict_fns.items():
        for h in ROLLOUT_HORIZONS:
            scores = []
            for ti in range(n_rollout):
                x0 = test_t[ti, 0]
                true_future = test_t[ti, 1:h + 1]
                if len(true_future) < h:
                    continue
                pred_future = multistep_rollout(pfn, x0, h, binarize=binarize_rollout)
                if loss_type == "mse":
                    scores.append(mse_metric(true_future, pred_future))
                else:
                    pred_bin = (pred_future >= 0.5).astype(np.float32)
                    scores.append(float(np.mean(pred_bin == true_future)))
            rollout_results[h][name] = float(np.mean(scores)) if scores else float("nan")

        vals = [f"h={h}: {rollout_results[h][name]:.4f}" for h in ROLLOUT_HORIZONS]
        print(f"    {name}: {', '.join(vals)}")

    return results, rollout_results, loss_type


# ===== Plotting =============================================================

MAP_COLORS = {
    "ResCor(Logistic)":  "tab:red",
    "ResCor(Sine)":      "tab:green",
    "ResCor(Tent)":      "tab:purple",
    "ResCor(Bernoulli)": "tab:orange",
    "PureNCA":           "tab:blue",
    "Conv2D":            "tab:brown",
}

MAP_MARKERS = {
    "ResCor(Logistic)":  "o",
    "ResCor(Sine)":      "s",
    "ResCor(Tent)":      "^",
    "ResCor(Bernoulli)": "D",
    "PureNCA":           "P",
    "Conv2D":            "X",
}


def make_combined_plot(heat_results, heat_rollout, gol_results, gol_rollout):
    plt = _get_plt()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # --- Top-left: Heat 1-step bar chart ---
    ax = axes[0, 0]
    names = list(heat_results.keys())
    vals = [heat_results[n]["one_step"] for n in names]
    colors = [MAP_COLORS.get(n, "tab:gray") for n in names]
    short_names = [n.replace("ResCor(", "RC(").replace(")", ")") for n in names]
    bars = ax.bar(range(len(names)), vals, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(short_names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("MSE (lower = better)")
    ax.set_title("Heat Equation: 1-step MSE")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, axis="y")
    # Annotate bars
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.4f}", ha="center", va="bottom", fontsize=7)

    # --- Top-right: Heat rollout ---
    ax = axes[0, 1]
    horizons = sorted(heat_rollout.keys())
    for name in names:
        rvals = [heat_rollout[h][name] for h in horizons]
        ax.plot(horizons, rvals,
                f"{MAP_MARKERS.get(name, 'o')}-",
                color=MAP_COLORS.get(name, "tab:gray"),
                label=name, markersize=7, linewidth=1.5)
    ax.set_xlabel("Rollout horizon (steps)")
    ax.set_ylabel("MSE (log scale)")
    ax.set_yscale("log")
    ax.set_title("Heat Equation: Rollout MSE")
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.3)

    # --- Bottom-left: GoL 1-step bar chart ---
    ax = axes[1, 0]
    names_g = list(gol_results.keys())
    vals_g = [gol_results[n]["one_step"] for n in names_g]
    colors_g = [MAP_COLORS.get(n, "tab:gray") for n in names_g]
    short_names_g = [n.replace("ResCor(", "RC(").replace(")", ")") for n in names_g]
    bars = ax.bar(range(len(names_g)), vals_g, color=colors_g, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(names_g)))
    ax.set_xticklabels(short_names_g, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Cell Accuracy (higher = better)")
    ax.set_title("Game of Life: 1-step Cell Accuracy")
    ax.set_ylim(0.5, 1.0)
    ax.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars, vals_g):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.4f}", ha="center", va="bottom", fontsize=7)

    # --- Bottom-right: GoL rollout ---
    ax = axes[1, 1]
    horizons_g = sorted(gol_rollout.keys())
    for name in names_g:
        rvals = [gol_rollout[h][name] for h in horizons_g]
        ax.plot(horizons_g, rvals,
                f"{MAP_MARKERS.get(name, 'o')}-",
                color=MAP_COLORS.get(name, "tab:gray"),
                label=name, markersize=7, linewidth=1.5)
    ax.set_xlabel("Rollout horizon (steps)")
    ax.set_ylabel("Cell Accuracy")
    ax.set_title("Game of Life: Rollout Accuracy")
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.3)

    fig.suptitle("Phase 2.5a: Chaotic Map Generalization\n"
                 "Does the Matching Principle hold across different chaotic maps?",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    path = PLOTS_DIR / "phase25a_chaotic_maps.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\n  Saved plot: {path}")


# ===== Summary Tables =======================================================

def print_summary_table(heat_results, heat_rollout, gol_results, gol_rollout):
    print(f"\n{'=' * 90}")
    print(f"  PHASE 2.5a SUMMARY: Chaotic Map Generalization")
    print(f"{'=' * 90}")

    all_names = list(heat_results.keys())

    # Header
    print(f"\n  {'Model':<22} {'Heat MSE':>10} {'GoL Acc':>10} "
          f"{'Heat h=10':>10} {'GoL h=10':>10} {'Trained':>8} {'Frozen':>7}")
    print(f"  {'-' * 22} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 8} {'-' * 7}")

    for name in all_names:
        h_mse = heat_results[name]["one_step"]
        g_acc = gol_results[name]["one_step"]
        h_r10 = heat_rollout[10].get(name, float("nan"))
        g_r10 = gol_rollout[10].get(name, float("nan"))
        trained = heat_results[name]["params_trained"]
        frozen = heat_results[name]["params_frozen"]
        print(f"  {name:<22} {h_mse:>10.6f} {g_acc:>10.4f} "
              f"{h_r10:>10.6f} {g_r10:>10.4f} {trained:>8d} {frozen:>7d}")

    # --- Analysis ---
    print(f"\n  {'=' * 70}")
    print(f"  MATCHING PRINCIPLE ANALYSIS")
    print(f"  {'=' * 70}")

    rc_names = [n for n in all_names if n.startswith("ResCor")]
    if rc_names:
        heat_mses = [heat_results[n]["one_step"] for n in rc_names]
        gol_accs = [gol_results[n]["one_step"] for n in rc_names]

        h_mean, h_std = np.mean(heat_mses), np.std(heat_mses)
        g_mean, g_std = np.mean(gol_accs), np.std(gol_accs)
        h_cv = h_std / h_mean if h_mean > 0 else 0
        g_cv = g_std / g_mean if g_mean > 0 else 0

        print(f"\n  ResidualCorrection across maps:")
        print(f"    Heat MSE:  mean={h_mean:.6f}, std={h_std:.6f}, CV={h_cv:.2%}")
        print(f"    GoL Acc:   mean={g_mean:.4f}, std={g_std:.4f}, CV={g_cv:.2%}")

        # Check: do Conv2D/PureNCA baselines beat ResCor on GoL?
        conv_gol = gol_results.get("Conv2D", {}).get("one_step", 0)
        nca_gol = gol_results.get("PureNCA", {}).get("one_step", 0)
        best_rc_gol = max(gol_accs)

        print(f"\n  GoL comparison:")
        print(f"    Best ResCor:  {best_rc_gol:.4f}")
        print(f"    Conv2D:       {conv_gol:.4f}")
        print(f"    PureNCA:      {nca_gol:.4f}")

        if conv_gol > best_rc_gol or nca_gol > best_rc_gol:
            print(f"    -> Baselines outperform ResCor on GoL (as expected by Matching Principle)")
        else:
            print(f"    -> WARNING: ResCor competitive on GoL -- may need investigation")

        # Check: does ResCor beat baselines on Heat?
        conv_heat = heat_results.get("Conv2D", {}).get("one_step", float("inf"))
        nca_heat = heat_results.get("PureNCA", {}).get("one_step", float("inf"))
        best_rc_heat = min(heat_mses)

        print(f"\n  Heat comparison:")
        print(f"    Best ResCor:  {best_rc_heat:.6f}")
        print(f"    Conv2D:       {conv_heat:.6f}")
        print(f"    PureNCA:      {nca_heat:.6f}")

        if best_rc_heat < conv_heat and best_rc_heat < nca_heat:
            print(f"    -> ResCor beats baselines on Heat (chaotic reservoir helps)")
        elif best_rc_heat < conv_heat or best_rc_heat < nca_heat:
            print(f"    -> ResCor beats at least one baseline on Heat")
        else:
            print(f"    -> Baselines match or beat ResCor on Heat")

        # Verdict
        print(f"\n  VERDICT:")
        all_maps_similar_heat = h_cv < 0.30  # coefficient of variation < 30%
        all_maps_similar_gol = g_cv < 0.10   # GoL accuracy should be tightly clustered

        if all_maps_similar_heat:
            print(f"    [PASS] All chaotic maps give similar Heat MSE (CV={h_cv:.2%} < 30%)")
        else:
            print(f"    [FAIL] Chaotic maps diverge on Heat MSE (CV={h_cv:.2%} >= 30%)")

        if all_maps_similar_gol:
            print(f"    [PASS] All chaotic maps give similar GoL accuracy (CV={g_cv:.2%} < 10%)")
        else:
            print(f"    [WARN] Chaotic maps diverge on GoL accuracy (CV={g_cv:.2%} >= 10%)")

        if all_maps_similar_heat and all_maps_similar_gol:
            print(f"    => MATCHING PRINCIPLE GENERALIZES: map choice doesn't matter,")
            print(f"       only continuous vs discrete dynamics mismatch matters.")
        else:
            print(f"    => PARTIAL/INCONCLUSIVE: need deeper investigation.")


# ===== Main =================================================================

def run_experiment(args):
    os.environ.setdefault("FORCE_CPU", "1")
    device = pick_device()

    print("=" * 72)
    print("PHASE 2.5a: Chaotic Map Generalization")
    print("Does the Matching Principle hold across different chaotic maps?")
    print("=" * 72)
    print(f"  Grid: {GRID_H}x{GRID_W}")
    print(f"  Epochs: {EPOCHS}, LR: {LR}, Batch: {BATCH_SIZE}")
    print(f"  CML steps: {CML_STEPS}, eps: {CML_EPS}, beta: {CML_BETA}")
    print(f"  Rollout horizons: {ROLLOUT_HORIZONS}")
    print(f"  Device: {device}")
    print(f"  Maps: {list(CHAOTIC_MAPS.keys())}")

    # ---- Benchmark 1: Heat Equation ----
    print("\n[1/2] Generating heat equation trajectories ...")
    t0 = time.time()
    heat_trajs = generate_heat_trajectories()
    print(f"  Generated {HEAT_N_TRAJ} trajectories x {HEAT_TRAJ_LEN} steps ({time.time()-t0:.1f}s)")

    heat_results, heat_rollout, _ = run_benchmark(
        "Heat Equation", heat_trajs, loss_type="mse", binarize_rollout=False)

    del heat_trajs
    gc.collect()

    # ---- Benchmark 2: Game of Life ----
    print("\n[2/2] Generating Game of Life trajectories ...")
    t0 = time.time()
    gol_trajs = generate_gol_trajectories()
    print(f"  Generated {GOL_N_TRAJ} trajectories x {GOL_TRAJ_LEN} steps ({time.time()-t0:.1f}s)")

    gol_results, gol_rollout, _ = run_benchmark(
        "Game of Life", gol_trajs, loss_type="bce", binarize_rollout=True)

    del gol_trajs
    gc.collect()

    # ---- Summary ----
    print_summary_table(heat_results, heat_rollout, gol_results, gol_rollout)

    # ---- Plot ----
    make_combined_plot(heat_results, heat_rollout, gol_results, gol_rollout)

    print("\nPhase 2.5a complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()
    run_experiment(args)
