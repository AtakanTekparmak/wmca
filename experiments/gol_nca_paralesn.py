"""NCA + ParalESN Hybrid for Game of Life Prediction.

Architecture: Learned NCA (spatial dynamics) + ParalESN reservoir (temporal
context). The ParalESN provides temporal memory across GoL timesteps; the NCA
uses its features as a second input channel to improve spatial predictions.

Compares three models:
  1. NCA + ParalESN (ours): learned NCA with ParalESN temporal context
  2. Pure NCA (ablation): same NCA architecture, no temporal context
  3. Conv2D baseline: reused from gol_prediction.py

Usage:
    FORCE_CPU=1 uv run --with scikit-learn,matplotlib,scipy python experiments/gol_nca_paralesn.py --no-wandb
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Ensure the project root is importable
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wmca.modules.paralesn import ParalESNLayer
from wmca.utils import pick_device


def _get_plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


# ===== Constants ===========================================================
PLOTS_DIR = PROJECT_ROOT / "experiments" / "plots"
SEED = 42

# Grid
GRID_H, GRID_W = 32, 32
GRID_SIZE = GRID_H * GRID_W  # 1024
DENSITY = 0.3

# Dataset
N_TRAJECTORIES = 1000
TRAJ_LENGTH = 20  # steps per trajectory

# NCA + ParalESN defaults
NCA_HIDDEN_CH = 16
PARALESN_HIDDEN = 128  # keep params reasonable
WINDOW_SIZE = 5        # sequence window length for training

# Conv2D baseline defaults (same as gol_prediction.py)
CNN_LR = 1e-3
CNN_EPOCHS = 50
CNN_BATCH = 64

# Shared training defaults
LR = 1e-3
EPOCHS = 50
BATCH = 32

# Rollout horizons
ROLLOUT_HORIZONS = [1, 2, 3, 5, 10]


# ===== ParalESN config ======================================================
@dataclass
class ParalESNCfg:
    """Minimal config object for ParalESNLayer."""
    hidden_size: int = PARALESN_HIDDEN
    rho_min: float = 0.95
    rho_max: float = 0.999
    theta_min: float = 0.0
    theta_max: float = 6.2832
    tau: float = 0.5
    mix_kernel_size: int = 5
    omega_in: float = 1.0
    omega_b: float = 0.1
    use_fft: bool = True


# ===== Data Generation (reused from gol_prediction.py) ====================
def gol_step(grid: np.ndarray) -> np.ndarray:
    """Single Game of Life step. grid: (H, W) binary {0, 1}."""
    from scipy.signal import convolve2d
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.float32)
    neighbors = convolve2d(grid.astype(np.float32), kernel,
                           mode="same", boundary="wrap")
    born = (grid == 0) & (neighbors == 3)
    survive = (grid == 1) & ((neighbors == 2) | (neighbors == 3))
    return (born | survive).astype(np.float32)


def generate_trajectories(n_traj: int = N_TRAJECTORIES,
                           traj_len: int = TRAJ_LENGTH,
                           h: int = GRID_H, w: int = GRID_W,
                           density: float = DENSITY,
                           seed: int = SEED) -> np.ndarray:
    """Generate GoL trajectories.

    Returns: (n_traj, traj_len+1, H, W) float32 array.
    Each trajectory has traj_len+1 frames (initial + traj_len steps).
    """
    rng = np.random.RandomState(seed)
    trajs = np.zeros((n_traj, traj_len + 1, h, w), dtype=np.float32)
    for i in range(n_traj):
        grid = (rng.rand(h, w) < density).astype(np.float32)
        trajs[i, 0] = grid
        for t in range(traj_len):
            grid = gol_step(grid)
            trajs[i, t + 1] = grid
    return trajs


def split_trajectories(trajs: np.ndarray):
    """Split trajectories into train/val/test (70/15/15 by trajectory)."""
    n = len(trajs)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    return trajs[:n_train], trajs[n_train:n_train + n_val], trajs[n_train + n_val:]


def make_pairs(trajs: np.ndarray):
    """Convert trajectories to (input, target) pairs for 1-step prediction.

    Returns:
        X: (N_traj * T, H, W)  — grid at time t
        Y: (N_traj * T, H, W)  — grid at time t+1
    """
    X = trajs[:, :-1].reshape(-1, GRID_H, GRID_W)
    Y = trajs[:, 1:].reshape(-1, GRID_H, GRID_W)
    return X, Y


def make_sequence_windows(trajs: np.ndarray, window: int = WINDOW_SIZE):
    """Extract sliding windows of length `window` from trajectories.

    For each trajectory of length T+1 and each valid start position s:
        input_seq:  trajs[s : s+window]        — (window, H, W)
        target_seq: trajs[s+1 : s+window+1]    — (window, H, W)

    Returns:
        X_seq: (N, window, H, W)  — input sequences
        Y_seq: (N, window, H, W)  — target sequences (next frame at each step)
    """
    n_traj, t_plus_1, h, w = trajs.shape
    T = t_plus_1 - 1  # number of transitions
    xs, ys = [], []
    for i in range(n_traj):
        for s in range(T - window + 1):
            xs.append(trajs[i, s:s + window])
            ys.append(trajs[i, s + 1:s + window + 1])
    X_seq = np.array(xs, dtype=np.float32)
    Y_seq = np.array(ys, dtype=np.float32)
    return X_seq, Y_seq


# ===== Model 1: NCA + ParalESN =============================================
class NCAParalESNModel(nn.Module):
    """Learned NCA with ParalESN temporal context.

    The ParalESN reservoir (frozen) processes the flattened grid sequence to
    produce temporal context features. These are projected back to spatial
    dimensions and fused with the current grid as a second input channel for
    the NCA convolutions.
    """

    def __init__(self, grid_size: int = GRID_SIZE,
                 hidden_ch: int = NCA_HIDDEN_CH,
                 paralesn_hidden: int = PARALESN_HIDDEN,
                 seed: int = SEED):
        super().__init__()
        self.grid_h = GRID_H
        self.grid_w = GRID_W
        self.paralesn_hidden = paralesn_hidden
        self.name = "NCA+ParalESN"

        # 1. ParalESN reservoir (frozen) — temporal memory
        cfg = ParalESNCfg(hidden_size=paralesn_hidden)
        rng = torch.Generator().manual_seed(seed)
        self.paralesn = ParalESNLayer(cfg, layer_idx=0,
                                      input_size=grid_size, rng=rng)
        for p in self.paralesn.parameters():
            p.requires_grad_(False)

        # 2. Learned projection: ParalESN features → spatial grid
        self.feature_to_grid = nn.Linear(paralesn_hidden, grid_size)

        # 3. Learned NCA — spatial dynamics
        #    2 input channels: current grid + ParalESN feature map
        self.perceive = nn.Conv2d(2, hidden_ch, 3, padding=1)
        self.update = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(hidden_ch, hidden_ch, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_ch, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, grids_sequence: torch.Tensor) -> torch.Tensor:
        """Predict next grid for each timestep in a sequence.

        Args:
            grids_sequence: (B, T, H, W) — sequence of T grids

        Returns:
            pred: (B, T, H, W) — predicted next grids
        """
        B, T, H, W = grids_sequence.shape

        # 1. ParalESN processes flattened grid sequence
        flat = grids_sequence.reshape(B, T, H * W)  # (B, T, 1024)
        with torch.no_grad():
            h, _z = self.paralesn(flat)   # _z is zeroed (out_proj zero-init)
        mixed = self.paralesn._mix(h)     # (B, T, paralesn_hidden) — bypass zero out_proj

        # 2. Project ParalESN features back to spatial grid
        feature_grids = self.feature_to_grid(mixed)            # (B, T, H*W)
        feature_grids = torch.sigmoid(feature_grids)
        feature_grids = feature_grids.reshape(B, T, 1, H, W)  # (B, T, 1, H, W)

        # 3. NCA: process each timestep with 2-channel input
        grids_in = grids_sequence.unsqueeze(2)                 # (B, T, 1, H, W)
        combined = torch.cat([grids_in, feature_grids], dim=2) # (B, T, 2, H, W)

        combined_flat = combined.reshape(B * T, 2, H, W)
        features = self.perceive(combined_flat)                # (B*T, hidden_ch, H, W)
        pred = self.update(features)                           # (B*T, 1, H, W)
        return pred.reshape(B, T, H, W)

    def predict_one(self, x: np.ndarray) -> np.ndarray:
        """Single-step prediction without temporal context. x: (H, W) -> (H, W)."""
        self.eval()
        with torch.no_grad():
            # Use a 1-step sequence (no real temporal context, but valid)
            x_t = torch.from_numpy(x).float().reshape(1, 1, GRID_H, GRID_W)
            pred = self.forward(x_t)  # (1, 1, H, W)
        return pred.squeeze().numpy()

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ===== Model 2: Pure NCA (ablation) ========================================
class PureNCAModel(nn.Module):
    """NCA without ParalESN — single-channel input, no temporal context.

    Same parameter budget as NCAParalESNModel minus the temporal pathway,
    to isolate the contribution of the ParalESN.
    """

    def __init__(self, hidden_ch: int = NCA_HIDDEN_CH):
        super().__init__()
        self.name = "NCA(pure)"

        self.perceive = nn.Conv2d(1, hidden_ch, 3, padding=1)
        self.update = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(hidden_ch, hidden_ch, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_ch, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, H, W) -> (B, 1, H, W)."""
        features = self.perceive(x)
        return self.update(features)

    def predict_one(self, x: np.ndarray) -> np.ndarray:
        """x: (H, W) -> (H, W)."""
        self.eval()
        with torch.no_grad():
            x_t = torch.from_numpy(x).float().reshape(1, 1, GRID_H, GRID_W)
            pred = self.forward(x_t)
        return pred.squeeze().numpy()

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ===== Model 3: Conv2D Baseline (from gol_prediction.py) ==================
class Conv2DModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "Conv2D"
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def predict_one(self, x: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            x_t = torch.from_numpy(x).float().reshape(1, 1, GRID_H, GRID_W)
            return self.forward(x_t).squeeze().numpy()

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ===== Training =============================================================
def train_nca_paralesn(model: NCAParalESNModel,
                       X_train_seq: np.ndarray, Y_train_seq: np.ndarray,
                       X_val_seq: np.ndarray, Y_val_seq: np.ndarray,
                       epochs: int = EPOCHS, lr: float = LR,
                       batch_size: int = BATCH) -> NCAParalESNModel:
    """Train NCA+ParalESN on sequence windows with BCE loss."""
    # Only train NCA layers + feature_to_grid; ParalESN is frozen
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=lr)
    criterion = nn.BCELoss()

    X_tr = torch.from_numpy(X_train_seq)  # (N, T, H, W)
    Y_tr = torch.from_numpy(Y_train_seq)
    X_v = torch.from_numpy(X_val_seq)
    Y_v = torch.from_numpy(Y_val_seq)

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(X_tr))
        total_loss = 0.0
        n_batches = 0
        for i in range(0, len(perm), batch_size):
            idx = perm[i:i + batch_size]
            xb = X_tr[idx]   # (B, T, H, W)
            yb = Y_tr[idx]   # (B, T, H, W)
            pred = model(xb)  # (B, T, H, W)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        model.eval()
        with torch.no_grad():
            val_pred = model(X_v)
            val_loss = criterion(val_pred, Y_v).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1:3d}/{epochs}  "
                  f"train_loss={total_loss / n_batches:.6f}  "
                  f"val_loss={val_loss:.6f}")

    model.load_state_dict(best_state)
    print(f"    Best val_loss: {best_val_loss:.6f}")
    return model


def train_pure_nca(model: PureNCAModel,
                   X_train: np.ndarray, Y_train: np.ndarray,
                   X_val: np.ndarray, Y_val: np.ndarray,
                   epochs: int = EPOCHS, lr: float = LR,
                   batch_size: int = CNN_BATCH) -> PureNCAModel:
    """Train Pure NCA on single-step pairs with BCE loss."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    X_tr = torch.from_numpy(X_train).float().unsqueeze(1)  # (N, 1, H, W)
    Y_tr = torch.from_numpy(Y_train).float().unsqueeze(1)
    X_v = torch.from_numpy(X_val).float().unsqueeze(1)
    Y_v = torch.from_numpy(Y_val).float().unsqueeze(1)

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(X_tr))
        total_loss = 0.0
        n_batches = 0
        for i in range(0, len(perm), batch_size):
            idx = perm[i:i + batch_size]
            pred = model(X_tr[idx])
            loss = criterion(pred, Y_tr[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        model.eval()
        with torch.no_grad():
            val_pred = model(X_v)
            val_loss = criterion(val_pred, Y_v).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1:3d}/{epochs}  "
                  f"train_loss={total_loss / n_batches:.6f}  "
                  f"val_loss={val_loss:.6f}")

    model.load_state_dict(best_state)
    print(f"    Best val_loss: {best_val_loss:.6f}")
    return model


def train_cnn(model: Conv2DModel,
              X_train: np.ndarray, Y_train: np.ndarray,
              X_val: np.ndarray, Y_val: np.ndarray,
              epochs: int = CNN_EPOCHS, lr: float = CNN_LR,
              batch_size: int = CNN_BATCH) -> Conv2DModel:
    """Train Conv2D with BCE loss."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    X_tr = torch.from_numpy(X_train).float().unsqueeze(1)
    Y_tr = torch.from_numpy(Y_train).float().unsqueeze(1)
    X_v = torch.from_numpy(X_val).float().unsqueeze(1)
    Y_v = torch.from_numpy(Y_val).float().unsqueeze(1)

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(X_tr))
        total_loss = 0.0
        n_batches = 0
        for i in range(0, len(perm), batch_size):
            idx = perm[i:i + batch_size]
            pred = model(X_tr[idx])
            loss = criterion(pred, Y_tr[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        model.eval()
        with torch.no_grad():
            val_pred = model(X_v)
            val_loss = criterion(val_pred, Y_v).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1:3d}/{epochs}  "
                  f"train_loss={total_loss / n_batches:.6f}  "
                  f"val_loss={val_loss:.6f}")

    model.load_state_dict(best_state)
    print(f"    Best val_loss: {best_val_loss:.6f}")
    return model


# ===== Evaluation ===========================================================
def cell_accuracy(Y_true: np.ndarray, Y_pred: np.ndarray,
                  threshold: float = 0.5) -> float:
    """Fraction of cells correctly predicted."""
    pred_binary = (Y_pred >= threshold).astype(np.float32)
    return float(np.mean(pred_binary == Y_true))


def grid_perfect_accuracy(Y_true: np.ndarray, Y_pred: np.ndarray,
                           threshold: float = 0.5) -> float:
    """Fraction of grids predicted with zero cell errors."""
    pred_binary = (Y_pred >= threshold).astype(np.float32)
    per_grid = np.all(pred_binary.reshape(len(Y_true), -1) ==
                      Y_true.reshape(len(Y_true), -1), axis=1)
    return float(np.mean(per_grid))


def multistep_rollout(predict_fn, x0: np.ndarray,
                      n_steps: int) -> np.ndarray:
    """Roll out a model for n_steps. predict_fn: (H, W) -> (H, W).

    Feed-back is binarized (threshold 0.5) since GoL inputs are binary.
    The raw continuous output is stored for accuracy measurement.

    Returns: (n_steps, H, W) predicted grids (continuous).
    """
    preds = np.zeros((n_steps, GRID_H, GRID_W), dtype=np.float32)
    x = x0.copy()
    for t in range(n_steps):
        raw = predict_fn(x)
        preds[t] = raw
        x = (np.clip(raw, 0, 1) >= 0.5).astype(np.float32)
    return preds


def multistep_cell_accuracy(true_grids: np.ndarray,
                             pred_grids: np.ndarray) -> float:
    """Cell accuracy over a sequence of grids."""
    pred_binary = (pred_grids >= 0.5).astype(np.float32)
    return float(np.mean(pred_binary == true_grids))


# ===== Plotting =============================================================
def make_plots(results: dict, rollout_results: dict,
               true_future: np.ndarray,
               rollouts: dict,
               x0: np.ndarray,
               log_wandb: bool):
    plt = _get_plt()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    wandb = None
    if log_wandb:
        import wandb as _wandb
        wandb = _wandb

    model_colors = {
        "NCA+ParalESN": "tab:purple",
        "NCA(pure)":    "tab:orange",
        "Conv2D":       "tab:green",
    }
    model_markers = {
        "NCA+ParalESN": "D",
        "NCA(pure)":    "o",
        "Conv2D":       "s",
    }
    model_names = list(results.keys())

    # --- Figure 1: Multi-step cell accuracy vs horizon ---
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    horizons = sorted(rollout_results.keys())
    for name in model_names:
        vals = [rollout_results[h][name] for h in horizons]
        ax1.plot(horizons, vals,
                 f"{model_markers[name]}-",
                 color=model_colors[name],
                 label=name, markersize=7)
    ax1.set_xlabel("Rollout horizon (steps)")
    ax1.set_ylabel("Cell accuracy")
    ax1.set_title("GoL Multi-step Cell Accuracy vs Horizon\n(NCA+ParalESN vs baselines)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.4, 1.02)
    fig1.tight_layout()
    path1 = PLOTS_DIR / "gol_nca_paralesn_rollout_accuracy.png"
    fig1.savefig(path1, dpi=150)
    if wandb:
        wandb.log({"plots/rollout_accuracy": wandb.Image(fig1)})
    plt.close(fig1)
    print(f"  Saved: {path1}")

    # --- Figure 2: Example predictions (3 timesteps) ---
    show_steps = [0, 2, min(4, len(true_future) - 1)]
    n_cols = 1 + len(model_names)  # true + each model
    fig2, axes = plt.subplots(len(show_steps), n_cols,
                              figsize=(3.5 * n_cols, 3 * len(show_steps)))
    col_titles = ["True"] + model_names
    col_data = [true_future] + [rollouts[n] for n in model_names]

    for row_i, t in enumerate(show_steps):
        for col_i, (title, data) in enumerate(zip(col_titles, col_data)):
            ax = axes[row_i, col_i]
            grid = data[t] if col_i == 0 else (data[t] >= 0.5).astype(np.float32)
            ax.imshow(grid, cmap="binary", vmin=0, vmax=1, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if row_i == 0:
                ax.set_title(title, fontsize=10)
            if col_i == 0:
                ax.set_ylabel(f"t+{t + 1}", fontsize=10)

    fig2.suptitle("GoL: True vs Predicted Grids", fontsize=13, y=1.01)
    fig2.tight_layout()
    path2 = PLOTS_DIR / "gol_nca_paralesn_example_predictions.png"
    fig2.savefig(path2, dpi=150, bbox_inches="tight")
    if wandb:
        wandb.log({"plots/example_predictions": wandb.Image(fig2)})
    plt.close(fig2)
    print(f"  Saved: {path2}")

    # --- Figure 3: Bar chart — 1-step accuracy metrics ---
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(12, 5))
    names = model_names
    cell_accs = [results[n]["cell_acc"] for n in names]
    grid_accs = [results[n]["grid_acc"] for n in names]
    colors = [model_colors[n] for n in names]
    x_pos = np.arange(len(names))

    ax3a.bar(x_pos, cell_accs, color=colors, alpha=0.85)
    ax3a.set_xticks(x_pos)
    ax3a.set_xticklabels(names, fontsize=10)
    ax3a.set_ylabel("Cell Accuracy")
    ax3a.set_title("1-Step Cell Accuracy")
    ax3a.set_ylim(0, 1.05)
    ax3a.grid(True, alpha=0.3, axis="y")
    for i, v in enumerate(cell_accs):
        ax3a.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)

    max_grid = max(grid_accs) if max(grid_accs) > 0 else 0.01
    ax3b.bar(x_pos, grid_accs, color=colors, alpha=0.85)
    ax3b.set_xticks(x_pos)
    ax3b.set_xticklabels(names, fontsize=10)
    ax3b.set_ylabel("Grid-Perfect Accuracy")
    ax3b.set_title("1-Step Grid-Perfect Accuracy")
    ax3b.set_ylim(0, max_grid * 1.3 + 0.01)
    ax3b.grid(True, alpha=0.3, axis="y")
    for i, v in enumerate(grid_accs):
        ax3b.text(i, v + 0.002, f"{v:.4f}", ha="center", fontsize=9)

    fig3.suptitle("GoL NCA+ParalESN: 1-Step Prediction Accuracy", fontsize=13)
    fig3.tight_layout()
    path3 = PLOTS_DIR / "gol_nca_paralesn_accuracy_bars.png"
    fig3.savefig(path3, dpi=150)
    if wandb:
        wandb.log({"plots/accuracy_bars": wandb.Image(fig3)})
    plt.close(fig3)
    print(f"  Saved: {path3}")

    # --- Figure 4: Param count comparison ---
    fig4, ax4 = plt.subplots(figsize=(7, 4))
    params = [results[n]["params"] for n in names]
    bars = ax4.bar(x_pos, params, color=colors, alpha=0.85)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(names, fontsize=10)
    ax4.set_ylabel("Trainable Parameters")
    ax4.set_title("Model Size Comparison (trainable params)")
    ax4.grid(True, alpha=0.3, axis="y")
    for i, v in enumerate(params):
        ax4.text(i, v * 1.02, f"{v:,}", ha="center", fontsize=9)
    fig4.tight_layout()
    path4 = PLOTS_DIR / "gol_nca_paralesn_params.png"
    fig4.savefig(path4, dpi=150)
    if wandb:
        wandb.log({"plots/param_counts": wandb.Image(fig4)})
    plt.close(fig4)
    print(f"  Saved: {path4}")


# ===== Summary ==============================================================
def print_summary(results: dict, rollout_results: dict):
    model_names = list(results.keys())

    print("\n" + "=" * 80)
    print("NCA + ParalESN HYBRID — GAME OF LIFE PREDICTION SUMMARY")
    print("=" * 80)

    col_w = 16
    print(f"\n{'Model':<{col_w}}  {'Cell Acc':>10s}  {'Grid-Perfect':>13s}  "
          f"{'Params (train)':>15s}")
    print("-" * 65)
    for name in model_names:
        r = results[name]
        print(f"{name:<{col_w}}  {r['cell_acc']:10.4f}  {r['grid_acc']:13.4f}  "
              f"{r['params']:15,d}")

    print(f"\n--- Multi-step Rollout Cell Accuracy ---")
    header = f"{'Horizon':>8s}"
    for name in model_names:
        header += f"  {name:>{col_w}s}"
    print(header)
    print("-" * (10 + len(model_names) * (col_w + 2)))
    for h in sorted(rollout_results.keys()):
        row = f"{h:8d}"
        for name in model_names:
            row += f"  {rollout_results[h][name]:{col_w}.4f}"
        print(row)

    print("=" * 80)


# ===== Main Experiment ======================================================
def run_experiment(args):
    os.environ.setdefault("FORCE_CPU", "1")
    pick_device()

    log_wandb = not args.no_wandb
    config = dict(
        grid_h=GRID_H, grid_w=GRID_W, density=DENSITY,
        n_trajectories=N_TRAJECTORIES, traj_length=TRAJ_LENGTH,
        nca_hidden_ch=NCA_HIDDEN_CH, paralesn_hidden=PARALESN_HIDDEN,
        window_size=WINDOW_SIZE, lr=LR, epochs=EPOCHS, batch=BATCH,
    )

    if log_wandb:
        from wmca.training import init_wandb
        init_wandb("gol-nca-paralesn", config=config,
                   tags=["gol", "nca", "paralesn", "hybrid"])

    print("=" * 72)
    print("GoL NCA + ParalESN HYBRID vs BASELINES")
    print("=" * 72)

    # --- Data ---
    print("\n[1/5] Generating Game of Life trajectories ...")
    t0 = time.time()
    trajs = generate_trajectories()
    train_trajs, val_trajs, test_trajs = split_trajectories(trajs)

    # Sequence windows for NCA+ParalESN training
    X_train_seq, Y_train_seq = make_sequence_windows(train_trajs, WINDOW_SIZE)
    X_val_seq, Y_val_seq = make_sequence_windows(val_trajs, WINDOW_SIZE)

    # Single-step pairs for Pure NCA and Conv2D
    X_train, Y_train = make_pairs(train_trajs)
    X_val, Y_val = make_pairs(val_trajs)
    X_test, Y_test = make_pairs(test_trajs)

    data_time = time.time() - t0
    print(f"  Trajectories: {len(trajs)} x {TRAJ_LENGTH} steps on "
          f"{GRID_H}x{GRID_W} grid  ({data_time:.1f}s)")
    print(f"  Train seq windows: {len(X_train_seq)}, Val: {len(X_val_seq)}")
    print(f"  Train pairs: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    results = {}

    # --- Model 1: NCA + ParalESN ---
    print(f"\n[2/5] Training NCA + ParalESN  "
          f"(paralesn_hidden={PARALESN_HIDDEN}, nca_hidden={NCA_HIDDEN_CH}) ...")
    t0 = time.time()
    nca_pe = NCAParalESNModel()
    nca_pe = train_nca_paralesn(nca_pe, X_train_seq, Y_train_seq,
                                 X_val_seq, Y_val_seq)
    nca_pe.eval()
    # Evaluate on test set (single-step, no temporal context)
    with torch.no_grad():
        X_test_t = torch.from_numpy(X_test).float().unsqueeze(1)  # (N, 1, H, W)
        # Feed as 1-step sequences
        X_test_seq_t = X_test_t.unsqueeze(1).squeeze(2)           # already (N, 1, H, W)
        nca_pe_pred_list = []
        for i in range(0, len(X_test_t), 256):
            batch = X_test_t[i:i + 256].squeeze(1)                # (B, H, W)
            seq = batch.unsqueeze(1)                               # (B, 1, H, W)
            p = nca_pe(seq)                                        # (B, 1, H, W)
            nca_pe_pred_list.append(p.squeeze(1).numpy())
    nca_pe_pred = np.concatenate(nca_pe_pred_list, axis=0)        # (N, H, W)
    nca_pe_cell_acc = cell_accuracy(Y_test, nca_pe_pred)
    nca_pe_grid_acc = grid_perfect_accuracy(Y_test, nca_pe_pred)
    nca_pe_time = time.time() - t0
    nca_pe_params = nca_pe.param_count()
    print(f"  Cell accuracy:    {nca_pe_cell_acc:.4f}")
    print(f"  Grid-perfect acc: {nca_pe_grid_acc:.4f}")
    print(f"  Trainable params: {nca_pe_params:,}  ({nca_pe_time:.1f}s)")
    results["NCA+ParalESN"] = {
        "cell_acc": nca_pe_cell_acc,
        "grid_acc": nca_pe_grid_acc,
        "params": nca_pe_params,
    }

    # --- Model 2: Pure NCA ---
    print(f"\n[3/5] Training Pure NCA  (hidden_ch={NCA_HIDDEN_CH}) ...")
    t0 = time.time()
    pure_nca = PureNCAModel()
    pure_nca = train_pure_nca(pure_nca, X_train, Y_train, X_val, Y_val)
    pure_nca.eval()
    with torch.no_grad():
        X_test_t = torch.from_numpy(X_test).float().unsqueeze(1)
        pure_nca_pred = pure_nca(X_test_t).squeeze(1).numpy()
    pure_nca_cell_acc = cell_accuracy(Y_test, pure_nca_pred)
    pure_nca_grid_acc = grid_perfect_accuracy(Y_test, pure_nca_pred)
    pure_nca_time = time.time() - t0
    pure_nca_params = pure_nca.param_count()
    print(f"  Cell accuracy:    {pure_nca_cell_acc:.4f}")
    print(f"  Grid-perfect acc: {pure_nca_grid_acc:.4f}")
    print(f"  Trainable params: {pure_nca_params:,}  ({pure_nca_time:.1f}s)")
    results["NCA(pure)"] = {
        "cell_acc": pure_nca_cell_acc,
        "grid_acc": pure_nca_grid_acc,
        "params": pure_nca_params,
    }

    # --- Model 3: Conv2D Baseline ---
    print(f"\n[4/5] Training Conv2D baseline ...")
    t0 = time.time()
    cnn = Conv2DModel()
    cnn = train_cnn(cnn, X_train, Y_train, X_val, Y_val)
    cnn.eval()
    with torch.no_grad():
        X_test_t = torch.from_numpy(X_test).float().unsqueeze(1)
        cnn_pred = cnn(X_test_t).squeeze(1).numpy()
    cnn_cell_acc = cell_accuracy(Y_test, cnn_pred)
    cnn_grid_acc = grid_perfect_accuracy(Y_test, cnn_pred)
    cnn_time = time.time() - t0
    cnn_params = cnn.param_count()
    print(f"  Cell accuracy:    {cnn_cell_acc:.4f}")
    print(f"  Grid-perfect acc: {cnn_grid_acc:.4f}")
    print(f"  Trainable params: {cnn_params:,}  ({cnn_time:.1f}s)")
    results["Conv2D"] = {
        "cell_acc": cnn_cell_acc,
        "grid_acc": cnn_grid_acc,
        "params": cnn_params,
    }

    # --- Multi-step rollout ---
    print("\n[5/5] Multi-step rollout evaluation ...")
    test_traj = test_trajs[0]       # (T+1, H, W)
    x0 = test_traj[0]               # initial grid
    max_horizon = min(max(ROLLOUT_HORIZONS), TRAJ_LENGTH)
    true_future = test_traj[1:max_horizon + 1]  # ground truth

    nca_pe_rollout = multistep_rollout(nca_pe.predict_one, x0, max_horizon)
    pure_nca_rollout = multistep_rollout(pure_nca.predict_one, x0, max_horizon)
    cnn_rollout = multistep_rollout(cnn.predict_one, x0, max_horizon)

    rollout_results = {}
    for h in ROLLOUT_HORIZONS:
        if h > max_horizon:
            break
        true_h = true_future[:h]
        r = {
            "NCA+ParalESN": multistep_cell_accuracy(true_h, nca_pe_rollout[:h]),
            "NCA(pure)":    multistep_cell_accuracy(true_h, pure_nca_rollout[:h]),
            "Conv2D":       multistep_cell_accuracy(true_h, cnn_rollout[:h]),
        }
        rollout_results[h] = r
        print(f"  Horizon {h:>2d}:  "
              f"NCA+ParalESN={r['NCA+ParalESN']:.4f}  "
              f"NCA(pure)={r['NCA(pure)']:.4f}  "
              f"Conv2D={r['Conv2D']:.4f}")

    # --- Plots ---
    print("\nGenerating plots ...")
    rollout_dict = {
        "NCA+ParalESN": nca_pe_rollout,
        "NCA(pure)":    pure_nca_rollout,
        "Conv2D":       cnn_rollout,
    }
    make_plots(results, rollout_results, true_future, rollout_dict, x0, log_wandb)

    # --- wandb logging ---
    if log_wandb:
        import wandb
        for name, r in results.items():
            tag = name.lower().replace("+", "_plus_").replace("(", "").replace(")", "")
            wandb.log({
                f"one_step/{tag}_cell_acc": r["cell_acc"],
                f"one_step/{tag}_grid_acc": r["grid_acc"],
                f"params/{tag}": r["params"],
            })
        for h, vals in rollout_results.items():
            for name, acc in vals.items():
                tag = name.lower().replace("+", "_plus_").replace("(", "").replace(")", "")
                wandb.log({f"rollout/horizon_{h}_{tag}": acc})
        wandb.finish()

    print_summary(results, rollout_results)


# ===== Entry Point ==========================================================
def main():
    parser = argparse.ArgumentParser(
        description="GoL NCA + ParalESN Hybrid vs baselines")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable Weights & Biases logging")
    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
