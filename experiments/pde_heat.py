"""Phase 1c: Heat Equation Prediction — CML Reservoir vs Learned NCA vs Baselines.

The heat equation du/dt = alpha * laplacian(u) is the simplest PDE: linear,
diffusive, smooth. The CML's local coupling IS diffusion, so this is
the CML's sweet spot.

Compares six models on one-step and multi-step heat equation prediction:
  1. CML-2D (fixed):       2D CML reservoir + Ridge readout
  2. CML-2D + ParalESN:    ParalESN temporal backbone + CML-2D features + Ridge
  3. NCA-2D (1 step):      Learned NCA, single step
  4. NCA-2D (3 step, res): Learned NCA, 3 residual steps
  5. Conv2D baseline:      3-layer CNN
  6. MLP baseline:         Flatten + MLP

Usage:
    FORCE_CPU=1 uv run --with scikit-learn,matplotlib,scipy python experiments/pde_heat.py --no-wandb
"""
from __future__ import annotations

import argparse
import gc
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
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wmca.modules.cml import CML
from wmca.modules.paralesn import ParalESNLayer
from wmca.utils import pick_device

def _get_ridge():
    from sklearn.linear_model import Ridge
    return Ridge

def _get_plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt

# ===== Constants ===========================================================
PLOTS_DIR = PROJECT_ROOT / "experiments" / "plots"
SEED = 42

GRID_H, GRID_W = 32, 32
GRID_SIZE = GRID_H * GRID_W  # 1024

# PDE parameters
ALPHA_HEAT = 0.1      # thermal diffusivity
DT = 0.01             # time step
DX = 1.0 / GRID_H     # spatial step

# Dataset
N_TRAJECTORIES = 500
TRAJ_LENGTH = 50

# Training
LR = 1e-3
EPOCHS = 50
BATCH_SIZE = 64

# CML 2D defaults
M_2D = 15
R_2D = 3.90
EPS_2D = 0.3
BETA_2D = 0.15

# Ridge
RIDGE_ALPHA = 1.0

# ParalESN
PARALESN_HIDDEN = 256

# Rollout horizons
ROLLOUT_HORIZONS = [1, 5, 10, 20, 50]


@dataclass
class ParalESNCfg:
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


# ===== Heat Equation Data ===================================================

# Pre-compute the Laplacian kernel (numpy) and coefficient once
_LAP_KERNEL_NP = np.array([[0., 1., 0.],
                            [1., -4., 1.],
                            [0., 1., 0.]], dtype=np.float32)
_HEAT_COEFF = ALPHA_HEAT * DT / (DX * DX)


def heat_step(u: np.ndarray) -> np.ndarray:
    """Single heat equation step with finite differences + zero BCs (pure numpy).

    u: (H, W) float32 in [0, 1].
    u_new = u + alpha * dt / dx^2 * laplacian(u)
    Zero Dirichlet BCs: edges fixed at 0.
    """
    from scipy.signal import convolve2d
    lap = convolve2d(u, _LAP_KERNEL_NP, mode="same", boundary="fill", fillvalue=0.0)
    u_new = u + _HEAT_COEFF * lap
    return np.clip(u_new, 0.0, 1.0)


def _random_gaussian_blob(h: int, w: int, rng: np.random.RandomState) -> np.ndarray:
    """Generate a single Gaussian blob on an (h, w) grid."""
    cy, cx = rng.uniform(0.2, 0.8) * h, rng.uniform(0.2, 0.8) * w
    sigma = rng.uniform(1.5, 4.0)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    blob = np.exp(-((yy - cy)**2 + (xx - cx)**2) / (2 * sigma**2))
    return blob


def generate_initial_condition(h: int, w: int, rng: np.random.RandomState) -> np.ndarray:
    """Random sum of 2-5 Gaussian blobs, normalized to [0, 1]."""
    n_blobs = rng.randint(2, 6)
    u0 = np.zeros((h, w), dtype=np.float32)
    for _ in range(n_blobs):
        u0 += _random_gaussian_blob(h, w, rng) * rng.uniform(0.3, 1.0)
    # Normalize to [0, 1]
    u_max = u0.max()
    if u_max > 1e-8:
        u0 = u0 / u_max
    return u0


def generate_trajectories(n_traj: int = N_TRAJECTORIES,
                          traj_len: int = TRAJ_LENGTH,
                          h: int = GRID_H, w: int = GRID_W,
                          seed: int = SEED) -> np.ndarray:
    """Generate heat equation trajectories.

    Returns: (n_traj, traj_len+1, H, W) float32.
    """
    rng = np.random.RandomState(seed)
    trajs = np.zeros((n_traj, traj_len + 1, h, w), dtype=np.float32)
    for i in range(n_traj):
        u0 = generate_initial_condition(h, w, rng)
        trajs[i, 0] = u0
        u = u0
        for t in range(traj_len):
            u = heat_step(u)
            trajs[i, t + 1] = u
    return trajs


def make_pairs(trajs: np.ndarray):
    """(N_traj, T+1, H, W) -> X:(N*T, H, W), Y:(N*T, H, W)."""
    X = trajs[:, :-1].reshape(-1, GRID_H, GRID_W)
    Y = trajs[:, 1:].reshape(-1, GRID_H, GRID_W)
    return X, Y


def split_trajectories(trajs: np.ndarray):
    """70/15/15 split by trajectory."""
    n = len(trajs)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    return trajs[:n_train], trajs[n_train:n_train + n_val], trajs[n_train + n_val:]


# ===== Model 1: CML 2D Reservoir (fixed) ===================================
class CML2D:
    """2D CML with conv2d coupling — fixed reservoir."""

    def __init__(self, H: int, W: int, steps: int, r: float, eps: float,
                 beta: float, rng: torch.Generator):
        self.H, self.W = H, W
        self.steps = steps
        self.r, self.eps, self.beta = r, eps, beta
        K_raw = torch.rand(1, 1, 3, 3, generator=rng)
        self.K_local = K_raw / K_raw.sum()

    @torch.no_grad()
    def forward(self, drive: torch.Tensor) -> torch.Tensor:
        """drive: (B, 1, H, W) in [0,1] -> (B, 1, H, W)."""
        grid = drive
        r, eps, beta = self.r, self.eps, self.beta
        K = self.K_local
        for _ in range(self.steps):
            mapped = r * grid * (1.0 - grid)
            local = F.conv2d(mapped, K, padding=1)
            physics = (1.0 - eps) * mapped + eps * local
            grid = (1.0 - beta) * physics + beta * drive
        return grid.clamp(1e-4, 1.0 - 1e-4)


class CML2DReservoir:
    """CML-2D + Ridge readout."""

    def __init__(self, H: int = GRID_H, W: int = GRID_W,
                 M: int = M_2D, r: float = R_2D,
                 eps: float = EPS_2D, beta: float = BETA_2D,
                 alpha: float = RIDGE_ALPHA, seed: int = SEED):
        self.name = "CML-2D(fixed)"
        rng = torch.Generator().manual_seed(seed)
        self.cml2d = CML2D(H=H, W=W, steps=M, r=r, eps=eps, beta=beta, rng=rng)
        self.alpha = alpha
        self.ridge = None

    def _features(self, X: np.ndarray) -> np.ndarray:
        """X: (N, H, W) -> (N, H*W) CML features."""
        X_t = torch.from_numpy(X).float().unsqueeze(1)
        out = self.cml2d.forward(X_t)
        return out.squeeze(1).reshape(len(X), -1).numpy()

    def fit(self, X: np.ndarray, Y: np.ndarray):
        Ridge = _get_ridge()
        Y_flat = Y.reshape(len(Y), -1)
        feats = self._features(X)
        self.ridge = Ridge(alpha=self.alpha)
        self.ridge.fit(feats, Y_flat)

    def predict(self, X: np.ndarray) -> np.ndarray:
        feats = self._features(X)
        pred_flat = self.ridge.predict(feats)
        return pred_flat.reshape(-1, GRID_H, GRID_W).clip(0, 1)

    def predict_one(self, x: np.ndarray) -> np.ndarray:
        return self.predict(x[np.newaxis])[0]

    def param_count(self) -> int:
        if self.ridge is None:
            return 0
        return self.ridge.coef_.size + self.ridge.intercept_.size


# ===== Model 2: CML-2D + ParalESN ==========================================
class CML2DParalESNHybrid:
    """ParalESN temporal features + CML-2D spatial features + Ridge."""

    def __init__(self, hidden_size: int = PARALESN_HIDDEN,
                 M: int = M_2D, r: float = R_2D,
                 eps: float = EPS_2D, beta: float = BETA_2D,
                 alpha: float = RIDGE_ALPHA, seed: int = SEED):
        self.name = "CML-2D+ParalESN"
        self.hidden_size = hidden_size

        cfg = ParalESNCfg(hidden_size=hidden_size)
        rng = torch.Generator().manual_seed(seed)
        self.paralesn = ParalESNLayer(cfg, layer_idx=0, input_size=GRID_SIZE, rng=rng)
        self.paralesn.eval()
        for p in self.paralesn.parameters():
            p.requires_grad_(False)

        rng_cml = torch.Generator().manual_seed(seed + 10)
        self.cml2d = CML2D(H=GRID_H, W=GRID_W, steps=M, r=r, eps=eps,
                           beta=beta, rng=rng_cml)

        self.alpha = alpha
        self.ridge = None

    def _cml_features(self, X: np.ndarray) -> np.ndarray:
        """X: (N, H, W) -> (N, H*W)."""
        X_t = torch.from_numpy(X).float().unsqueeze(1)
        out = self.cml2d.forward(X_t)
        return out.squeeze(1).reshape(len(X), -1).numpy()

    def _paralesn_features(self, trajs_flat: np.ndarray) -> np.ndarray:
        """trajs_flat: (N_traj, T, 1024) -> (N_traj*T, hidden_size)."""
        X_t = torch.from_numpy(trajs_flat).float()
        with torch.no_grad():
            h, _z = self.paralesn(X_t)
            mixed = self.paralesn._mix(h)  # (N_traj, T, hidden_size)
        N, T, H = mixed.shape
        return mixed.reshape(N * T, H).numpy()

    def fit(self, X: np.ndarray, Y: np.ndarray, trajs: np.ndarray):
        Ridge = _get_ridge()
        n_traj, t_plus_1, h, w = trajs.shape
        T = t_plus_1 - 1
        seqs = trajs[:, :-1].reshape(n_traj, T, -1)
        targets = trajs[:, 1:].reshape(n_traj * T, -1)
        X_flat = trajs[:, :-1].reshape(-1, h, w)

        # CML features
        cml_feats = self._cml_features(X_flat)
        # ParalESN features
        pesn_feats = self._paralesn_features(seqs)
        # Concatenate
        feats = np.concatenate([cml_feats, pesn_feats], axis=1)

        self.ridge = Ridge(alpha=self.alpha)
        self.ridge.fit(feats, targets)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Single-frame prediction (no temporal context)."""
        cml_feats = self._cml_features(X)
        X_flat = X.reshape(len(X), -1)
        X_seq = X_flat[:, np.newaxis, :]  # (N, 1, 1024)
        pesn_feats = self._paralesn_features(X_seq)
        feats = np.concatenate([cml_feats, pesn_feats], axis=1)
        pred_flat = self.ridge.predict(feats)
        return pred_flat.reshape(-1, GRID_H, GRID_W).clip(0, 1)

    def predict_one(self, x: np.ndarray) -> np.ndarray:
        return self.predict(x[np.newaxis])[0]

    def param_count(self) -> int:
        if self.ridge is None:
            return 0
        return self.ridge.coef_.size + self.ridge.intercept_.size


# ===== Model 3: NCA-2D (learned, 1 step) ===================================
class NCA2D(nn.Module):
    """Learned NCA: Conv2d perception + 1x1 update."""

    def __init__(self, hidden_ch: int = 16, steps: int = 1,
                 residual: bool = False):
        super().__init__()
        self.steps = steps
        self.residual = residual
        self.perceive = nn.Conv2d(1, hidden_ch, 3, padding=1)
        self.update = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(hidden_ch, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, H, W) -> (B, 1, H, W)."""
        for _ in range(self.steps):
            features = self.perceive(x)
            delta = self.update(features)
            if self.residual:
                x = x + delta
            else:
                x = delta
        return x

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ===== Model 5: Conv2D Baseline ============================================
class Conv2DBaseline(nn.Module):
    """3-layer CNN: 1->16->16->1, 3x3 kernels."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ===== Model 6: MLP Baseline ===============================================
class MLPBaseline(nn.Module):
    """Flatten 1024 -> 512 -> 1024."""

    def __init__(self, input_size: int = GRID_SIZE, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, input_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ===== Training =============================================================

def train_nn_model(model: nn.Module,
                   X_train: np.ndarray, Y_train: np.ndarray,
                   X_val: np.ndarray, Y_val: np.ndarray,
                   is_spatial: bool = True,
                   epochs: int = EPOCHS, lr: float = LR,
                   batch_size: int = BATCH_SIZE,
                   device: torch.device | None = None) -> nn.Module:
    """Train any model with MSE loss + Adam.

    is_spatial: if True, input/output are (B,1,H,W); if False, (B,1024).
    """
    if device is None:
        device = torch.device("cpu")

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    if is_spatial:
        X_tr = torch.from_numpy(X_train).float().unsqueeze(1).to(device)
        Y_tr = torch.from_numpy(Y_train).float().unsqueeze(1).to(device)
        X_v = torch.from_numpy(X_val).float().unsqueeze(1).to(device)
        Y_v = torch.from_numpy(Y_val).float().unsqueeze(1).to(device)
    else:
        X_tr = torch.from_numpy(X_train.reshape(len(X_train), -1)).float().to(device)
        Y_tr = torch.from_numpy(Y_train.reshape(len(Y_train), -1)).float().to(device)
        X_v = torch.from_numpy(X_val.reshape(len(X_val), -1)).float().to(device)
        Y_v = torch.from_numpy(Y_val.reshape(len(Y_val), -1)).float().to(device)

    best_val_loss = float("inf")
    best_state: dict | None = None

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(X_tr), device=device)
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
            # Evaluate val in batches to save memory
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
            print(f"    Epoch {epoch+1:3d}/{epochs}  "
                  f"train={total_loss / n_batches:.6f}  val={val_loss:.6f}")

    model.load_state_dict(best_state)
    model = model.to(torch.device("cpu"))
    print(f"    Best val_loss: {best_val_loss:.6f}")

    # Free GPU memory
    del X_tr, Y_tr, X_v, Y_v
    gc.collect()
    return model


# ===== Evaluation ===========================================================

def mse(Y_true: np.ndarray, Y_pred: np.ndarray) -> float:
    return float(np.mean((Y_true - Y_pred) ** 2))


def multistep_rollout(predict_fn, x0: np.ndarray,
                      n_steps: int) -> np.ndarray:
    """Roll out predict_fn for n_steps. No binarization — continuous PDE.

    Returns (n_steps, H, W).
    """
    preds = np.zeros((n_steps, GRID_H, GRID_W), dtype=np.float32)
    x = x0.copy()
    for t in range(n_steps):
        raw = predict_fn(x)
        preds[t] = raw
        x = np.clip(raw, 0, 1).astype(np.float32)
    return preds


def multistep_mse(true_grids: np.ndarray, pred_grids: np.ndarray) -> float:
    return float(np.mean((true_grids - pred_grids) ** 2))


def make_predict_fn_spatial(model: nn.Module):
    """Wrap a (B,1,H,W)->(B,1,H,W) model."""
    model.eval()
    def _predict(x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x_t = torch.from_numpy(x).float().reshape(1, 1, GRID_H, GRID_W)
            out = model(x_t).squeeze().numpy()
        return np.clip(out, 0, 1)
    return _predict


def make_predict_fn_flat(model: nn.Module):
    """Wrap a (B,1024)->(B,1024) model."""
    model.eval()
    def _predict(x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x_t = torch.from_numpy(x.reshape(1, -1)).float()
            out = model(x_t).numpy().reshape(GRID_H, GRID_W)
        return np.clip(out, 0, 1)
    return _predict


def evaluate_model_spatial(model: nn.Module,
                           X_test: np.ndarray, Y_test: np.ndarray):
    """Return (mse, pred_array)."""
    model.eval()
    preds = []
    bs = BATCH_SIZE
    with torch.no_grad():
        for i in range(0, len(X_test), bs):
            xb = torch.from_numpy(X_test[i:i+bs]).float().unsqueeze(1)
            pb = model(xb).squeeze(1).numpy()
            preds.append(pb)
    pred = np.concatenate(preds, axis=0)
    return mse(Y_test, pred), pred


def evaluate_model_flat(model: nn.Module,
                        X_test: np.ndarray, Y_test: np.ndarray):
    """Return (mse, pred_array)."""
    model.eval()
    preds = []
    bs = BATCH_SIZE
    with torch.no_grad():
        for i in range(0, len(X_test), bs):
            xb = torch.from_numpy(X_test[i:i+bs].reshape(-1, GRID_SIZE)).float()
            pb = model(xb).numpy().reshape(-1, GRID_H, GRID_W)
            preds.append(pb)
    pred = np.concatenate(preds, axis=0)
    return mse(Y_test, pred), pred


# ===== Plotting =============================================================

MODEL_COLORS = {
    "CML-2D(fixed)":     "tab:red",
    "CML-2D+ParalESN":   "tab:purple",
    "NCA-2D(1step)":     "tab:blue",
    "NCA-2D(3step-res)": "tab:orange",
    "Conv2D":            "tab:green",
    "MLP":               "tab:brown",
}


def make_plots(results: dict, rollout_results: dict,
               true_future: np.ndarray, rollouts: dict,
               x0: np.ndarray, log_wandb: bool):
    plt = _get_plt()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    model_names = list(results.keys())
    horizons = sorted(rollout_results.keys())

    # ---- Plot 1: Rollout MSE vs horizon (log scale y) ----
    fig1, ax1 = plt.subplots(figsize=(9, 5))
    markers = ["s", "P", "o", "^", "D", "v"]
    for mi, name in enumerate(model_names):
        vals = [rollout_results[h][name] for h in horizons]
        ax1.plot(horizons, vals, f"{markers[mi % len(markers)]}-",
                 color=MODEL_COLORS.get(name, "tab:gray"),
                 label=name, markersize=7)
    ax1.set_xlabel("Rollout horizon (steps)")
    ax1.set_ylabel("MSE (log scale)")
    ax1.set_yscale("log")
    ax1.set_title("Heat Equation: Rollout MSE vs Horizon")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    path1 = PLOTS_DIR / "heat_rollout_mse.png"
    fig1.savefig(path1, dpi=150)
    plt.close(fig1)

    # ---- Plot 2: Example predictions for best 2 models at steps 1, 10, 25 ----
    # Find best 2 models by 1-step MSE
    sorted_models = sorted(model_names, key=lambda n: results[n]["mse"])
    best2 = sorted_models[:2]
    show_steps_raw = [1, 10, 25]
    max_avail = len(true_future)
    show_steps = [s for s in show_steps_raw if s <= max_avail]

    n_rows = len(show_steps)
    n_cols = 1 + len(best2)  # True + best2
    fig2, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["True"] + best2
    for row_i, step in enumerate(show_steps):
        idx = step - 1  # 0-indexed
        for col_i, title in enumerate(col_titles):
            ax = axes[row_i, col_i]
            if col_i == 0:
                grid = true_future[idx]
            else:
                grid = rollouts[title][idx]
            im = ax.imshow(grid, cmap="hot", vmin=0, vmax=1, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if row_i == 0:
                ax.set_title(title, fontsize=10)
            if col_i == 0:
                ax.set_ylabel(f"step {step}", fontsize=10)

    fig2.suptitle(f"Heat Eq: True vs Best 2 Models", fontsize=12, y=1.01)
    fig2.tight_layout()
    path2 = PLOTS_DIR / "heat_example_predictions.png"
    fig2.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close(fig2)

    print(f"  Plots saved -> {path1.name}, {path2.name}")


# ===== Summary ==============================================================

def print_summary(results: dict, rollout_results: dict):
    model_names = list(results.keys())
    horizons = sorted(rollout_results.keys())

    print("\n" + "=" * 90)
    print("PHASE 1c SUMMARY: HEAT EQUATION PREDICTION")
    print("=" * 90)

    col_w = 14
    header = f"{'Model':<22s}  {'1-Step MSE':>{col_w}}  {'Params':>{col_w}}"
    print(f"\n{header}")
    print("-" * (22 + 2 * (col_w + 2) + 4))
    for name in model_names:
        r = results[name]
        print(f"{name:<22s}  {r['mse']:{col_w}.6f}  {r['params']:{col_w}d}")

    print(f"\n--- Multi-step Rollout MSE ---")
    hdr = f"{'Horizon':>8s}"
    for name in model_names:
        hdr += f"  {name:>22s}"
    print(hdr)
    for h in horizons:
        row = f"{h:8d}"
        for name in model_names:
            row += f"  {rollout_results[h][name]:22.6f}"
        print(row)

    print("=" * 90)


# ===== Main Experiment ======================================================

def run_experiment(args):
    os.environ.setdefault("FORCE_CPU", "1")
    device = pick_device()

    log_wandb = not args.no_wandb
    config = dict(
        grid_h=GRID_H, grid_w=GRID_W,
        alpha_heat=ALPHA_HEAT, dt=DT, dx=DX,
        n_trajectories=N_TRAJECTORIES, traj_length=TRAJ_LENGTH,
        lr=LR, epochs=EPOCHS, batch_size=BATCH_SIZE,
        m_2d=M_2D, r_2d=R_2D, eps_2d=EPS_2D, beta_2d=BETA_2D,
        ridge_alpha=RIDGE_ALPHA,
    )

    if log_wandb:
        from wmca.training import init_wandb
        init_wandb("pde-heat-1c", config=config,
                   tags=["heat", "phase-1c", "pde"])

    print("=" * 72)
    print("PHASE 1c: HEAT EQUATION PREDICTION — CML vs NCA vs BASELINES")
    print("=" * 72)

    # ---- Data ----
    print("\n[1/8] Generating heat equation trajectories ...")
    t0 = time.time()
    trajs = generate_trajectories()
    train_trajs, val_trajs, test_trajs = split_trajectories(trajs)
    X_train, Y_train = make_pairs(train_trajs)
    X_val, Y_val = make_pairs(val_trajs)
    X_test, Y_test = make_pairs(test_trajs)
    print(f"  {len(trajs)} trajectories x {TRAJ_LENGTH} steps on "
          f"{GRID_H}x{GRID_W}  ({time.time()-t0:.1f}s)")
    print(f"  Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")
    print(f"  PDE params: alpha={ALPHA_HEAT}, dt={DT}, dx={DX:.4f}")

    results: dict[str, dict] = {}
    predict_fns: dict[str, object] = {}

    # ---- Model 1: CML-2D (fixed) ----
    print("\n[2/8] CML-2D (fixed reservoir + Ridge) ...")
    t0 = time.time()
    cml2d = CML2DReservoir()
    cml2d.fit(X_train, Y_train)
    cml2d_pred = cml2d.predict(X_test)
    cml2d_mse = mse(Y_test, cml2d_pred)
    elapsed = time.time() - t0
    cml2d_params = cml2d.param_count()
    print(f"  MSE: {cml2d_mse:.6f}  Params: {cml2d_params}  ({elapsed:.1f}s)")
    results["CML-2D(fixed)"] = {"mse": cml2d_mse, "params": cml2d_params}
    predict_fns["CML-2D(fixed)"] = cml2d.predict_one
    del cml2d_pred
    gc.collect()

    # ---- Model 2: CML-2D + ParalESN ----
    print("\n[3/8] CML-2D + ParalESN hybrid ...")
    t0 = time.time()
    hybrid = CML2DParalESNHybrid()
    hybrid.fit(X_train, Y_train, train_trajs)
    hybrid_pred = hybrid.predict(X_test)
    hybrid_mse = mse(Y_test, hybrid_pred)
    elapsed = time.time() - t0
    hybrid_params = hybrid.param_count()
    print(f"  MSE: {hybrid_mse:.6f}  Params: {hybrid_params}  ({elapsed:.1f}s)")
    results["CML-2D+ParalESN"] = {"mse": hybrid_mse, "params": hybrid_params}
    predict_fns["CML-2D+ParalESN"] = hybrid.predict_one
    del hybrid_pred
    gc.collect()

    # ---- Model 3: NCA-2D (1 step) ----
    print("\n[4/8] NCA-2D (learned, 1 step) ...")
    t0 = time.time()
    nca1 = NCA2D(hidden_ch=16, steps=1, residual=False)
    nca1 = train_nn_model(nca1, X_train, Y_train, X_val, Y_val,
                          is_spatial=True, device=device)
    nca1_mse, _ = evaluate_model_spatial(nca1, X_test, Y_test)
    elapsed = time.time() - t0
    nca1_params = nca1.param_count()
    print(f"  MSE: {nca1_mse:.6f}  Params: {nca1_params}  ({elapsed:.1f}s)")
    results["NCA-2D(1step)"] = {"mse": nca1_mse, "params": nca1_params}
    predict_fns["NCA-2D(1step)"] = make_predict_fn_spatial(nca1)

    # ---- Model 4: NCA-2D (3 step, residual) ----
    print("\n[5/8] NCA-2D (learned, 3 steps, residual) ...")
    t0 = time.time()
    nca3 = NCA2D(hidden_ch=16, steps=3, residual=True)
    nca3 = train_nn_model(nca3, X_train, Y_train, X_val, Y_val,
                          is_spatial=True, device=device)
    nca3_mse, _ = evaluate_model_spatial(nca3, X_test, Y_test)
    elapsed = time.time() - t0
    nca3_params = nca3.param_count()
    print(f"  MSE: {nca3_mse:.6f}  Params: {nca3_params}  ({elapsed:.1f}s)")
    results["NCA-2D(3step-res)"] = {"mse": nca3_mse, "params": nca3_params}
    predict_fns["NCA-2D(3step-res)"] = make_predict_fn_spatial(nca3)

    # ---- Model 5: Conv2D baseline ----
    print("\n[6/8] Conv2D baseline ...")
    t0 = time.time()
    cnn = Conv2DBaseline()
    cnn = train_nn_model(cnn, X_train, Y_train, X_val, Y_val,
                         is_spatial=True, device=device)
    cnn_mse, _ = evaluate_model_spatial(cnn, X_test, Y_test)
    elapsed = time.time() - t0
    cnn_params = cnn.param_count()
    print(f"  MSE: {cnn_mse:.6f}  Params: {cnn_params}  ({elapsed:.1f}s)")
    results["Conv2D"] = {"mse": cnn_mse, "params": cnn_params}
    predict_fns["Conv2D"] = make_predict_fn_spatial(cnn)

    # ---- Model 6: MLP baseline ----
    print("\n[7/8] MLP baseline ...")
    t0 = time.time()
    mlp = MLPBaseline()
    mlp = train_nn_model(mlp, X_train, Y_train, X_val, Y_val,
                         is_spatial=False, device=device)
    mlp_mse, _ = evaluate_model_flat(mlp, X_test, Y_test)
    elapsed = time.time() - t0
    mlp_params = mlp.param_count()
    print(f"  MSE: {mlp_mse:.6f}  Params: {mlp_params}  ({elapsed:.1f}s)")
    results["MLP"] = {"mse": mlp_mse, "params": mlp_params}
    predict_fns["MLP"] = make_predict_fn_flat(mlp)

    # ---- Multi-step rollout ----
    print("\n[8/8] Multi-step rollout evaluation ...")
    test_traj = test_trajs[0]
    x0 = test_traj[0]
    max_horizon = min(max(ROLLOUT_HORIZONS), TRAJ_LENGTH)
    true_future = test_traj[1:max_horizon + 1]

    rollouts: dict[str, np.ndarray] = {}
    model_names = list(results.keys())
    for name in model_names:
        rollouts[name] = multistep_rollout(predict_fns[name], x0, max_horizon)

    rollout_results: dict[int, dict] = {}
    for h in ROLLOUT_HORIZONS:
        if h > max_horizon:
            break
        true_h = true_future[:h]
        row = {name: multistep_mse(true_h, rollouts[name][:h])
               for name in model_names}
        rollout_results[h] = row
        parts = "  ".join(f"{n}={row[n]:.6f}" for n in model_names)
        print(f"  Horizon {h:>2d}:  {parts}")

    # ---- Plots ----
    print("\n  Generating plots ...")
    make_plots(results, rollout_results, true_future, rollouts, x0, log_wandb)

    # ---- wandb logging ----
    if log_wandb:
        import wandb
        for name, r in results.items():
            tag = name.lower().replace("-", "_").replace("(", "").replace(")", "").replace("+", "_")
            wandb.log({
                f"one_step/{tag}_mse": r["mse"],
                f"params/{tag}": r["params"],
            })
        for h, vals in rollout_results.items():
            for name, v in vals.items():
                tag = name.lower().replace("-", "_").replace("(", "").replace(")", "").replace("+", "_")
                wandb.log({f"rollout/horizon_{h}_{tag}": v})
        wandb.finish()

    # ---- Summary ----
    print_summary(results, rollout_results)


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1c: Heat equation prediction — CML vs NCA vs baselines")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Skip wandb logging")
    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
