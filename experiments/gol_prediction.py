"""Phase 1b: Game of Life Next-State Prediction — CML Reservoir vs Baselines.

The GoL IS a cellular automaton on a 2D grid. The CML lattice maps directly
to the GoL grid — no encoding mismatch. This tests whether a CML can
learn/predict another CA's dynamics.

Compares four models on one-step and multi-step GoL prediction:
  1. CML 1D Reservoir (flattened): wrong topology — ablation baseline
  2. CML 2D Reservoir (spatial):   correct topology — CML should shine
  3. MLP baseline:                 standard neural net, no spatial bias
  4. Conv2D baseline:              correct inductive bias (local receptive fields)

Usage:
    uv run --with scikit-learn,matplotlib,scipy python experiments/gol_prediction.py --no-wandb
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

from wmca.modules.cml import CML
from wmca.modules.paralesn import ParalESNLayer
from wmca.utils import pick_device

# Lazy imports
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

# Grid
GRID_H, GRID_W = 32, 32
GRID_SIZE = GRID_H * GRID_W  # 1024
DENSITY = 0.3

# Dataset
N_TRAJECTORIES = 1000
TRAJ_LENGTH = 20  # steps per trajectory

# CML 1D defaults
C_1D = 256
M_1D = 15
KERNEL_SIZE_1D = 3
R_1D = 3.90
EPS_1D = 0.3
BETA_1D = 0.15

# CML 2D defaults
M_2D = 15
R_2D = 3.90
EPS_2D = 0.3
BETA_2D = 0.15

# MLP defaults
MLP_HIDDEN = 512
MLP_LR = 1e-3
MLP_EPOCHS = 50
MLP_BATCH = 64

# Conv2D defaults
CNN_LR = 1e-3
CNN_EPOCHS = 50
CNN_BATCH = 64

# Ridge alpha
RIDGE_ALPHA = 1.0

# Rollout horizons
ROLLOUT_HORIZONS = [1, 2, 3, 5, 10]

# ParalESN+CML hybrid defaults
PARALESN_HIDDEN = 256


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


# ===== Data Generation =====================================================
def gol_step(grid: np.ndarray) -> np.ndarray:
    """Single Game of Life step. grid: (H, W) binary {0, 1}."""
    from scipy.signal import convolve2d
    # Count alive neighbors (8-connected) via convolution
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.float32)
    neighbors = convolve2d(grid.astype(np.float32), kernel,
                           mode="same", boundary="wrap")
    # GoL rules
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


def make_pairs(trajs: np.ndarray):
    """Convert trajectories to (input, target) pairs for 1-step prediction.

    Args:
        trajs: (N_traj, T+1, H, W)
    Returns:
        X: (N_traj * T, H, W)  — grid at time t
        Y: (N_traj * T, H, W)  — grid at time t+1
    """
    n_traj, t_plus_1, h, w = trajs.shape
    t = t_plus_1 - 1
    X = trajs[:, :-1].reshape(-1, h, w)  # all but last frame
    Y = trajs[:, 1:].reshape(-1, h, w)   # all but first frame
    return X, Y


def split_trajectories(trajs: np.ndarray):
    """Split trajectories into train/val/test (70/15/15 by trajectory).

    Returns: (train_trajs, val_trajs, test_trajs)
    """
    n = len(trajs)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    return trajs[:n_train], trajs[n_train:n_train + n_val], trajs[n_train + n_val:]


# ===== CML 2D Reservoir (inline) ==========================================
class CML2D:
    """2D Coupled Map Lattice that preserves spatial structure.

    Each cell has its own CML state. Coupling is via 2D convolution
    with a 3x3 kernel, matching GoL's neighbor topology.
    """

    def __init__(self, H: int, W: int, steps: int, r: float, eps: float,
                 beta: float, rng: torch.Generator):
        self.H = H
        self.W = W
        self.steps = steps
        self.r = r
        self.eps = eps
        self.beta = beta

        # Positive normalized 3x3 coupling kernel
        K_raw = torch.rand(1, 1, 3, 3, generator=rng)
        self.K_local = K_raw / K_raw.sum()

    @torch.no_grad()
    def forward(self, drive: torch.Tensor) -> torch.Tensor:
        """Run CML 2D dynamics.

        Args:
            drive: (B, 1, H, W) in [0, 1]
        Returns: (B, 1, H, W) in [0, 1]
        """
        grid = drive
        r = self.r
        eps = self.eps
        beta = self.beta
        K = self.K_local

        for _ in range(self.steps):
            mapped = r * grid * (1.0 - grid)
            local = F.conv2d(mapped, K, padding=1)
            physics = (1.0 - eps) * mapped + eps * local
            grid = (1.0 - beta) * physics + beta * drive

        return grid.clamp(1e-4, 1.0 - 1e-4)


# ===== Model 1: CML 1D Reservoir (flattened) ==============================
class CML1DReservoir:
    """Flatten 32x32 grid to 1024, project to C=256, run 1D CML, Ridge out."""

    def __init__(self, C: int = C_1D, M: int = M_1D,
                 kernel_size: int = KERNEL_SIZE_1D, r: float = R_1D,
                 eps: float = EPS_1D, beta: float = BETA_1D,
                 alpha: float = RIDGE_ALPHA, seed: int = SEED):
        self.C = C
        self.name = "CML-1D(flat)"
        rng = torch.Generator().manual_seed(seed)
        self.cml = CML(C=C, steps=M, kernel_size=kernel_size,
                       r=r, eps=eps, beta=beta, rng=rng)
        self.cml.eval()

        # Fixed random input projection: 1024 -> C
        self.W_in = torch.randn(GRID_SIZE, C,
                                generator=torch.Generator().manual_seed(seed + 1))
        self.W_in *= 0.3

        self.alpha = alpha
        self.ridge = None

    def _features(self, X_flat: np.ndarray) -> np.ndarray:
        """Map (N, 1024) -> (N, C) CML features."""
        X_t = torch.from_numpy(X_flat).float()
        drive = torch.sigmoid(X_t @ self.W_in)  # (N, C)
        with torch.no_grad():
            out = self.cml(drive)  # (N, C)
        return out.numpy()

    def fit(self, X: np.ndarray, Y: np.ndarray):
        """X, Y: (N, H, W)."""
        Ridge = _get_ridge()
        X_flat = X.reshape(len(X), -1)
        Y_flat = Y.reshape(len(Y), -1)
        feats = self._features(X_flat)
        self.ridge = Ridge(alpha=self.alpha)
        self.ridge.fit(feats, Y_flat)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """X: (N, H, W) -> (N, H, W) predictions."""
        X_flat = X.reshape(len(X), -1)
        feats = self._features(X_flat)
        pred_flat = self.ridge.predict(feats)  # (N, 1024)
        return pred_flat.reshape(-1, GRID_H, GRID_W)

    def predict_one(self, x: np.ndarray) -> np.ndarray:
        """x: (H, W) -> (H, W)."""
        return self.predict(x[np.newaxis])[0]

    def param_count(self) -> int:
        if self.ridge is None:
            return 0
        return self.ridge.coef_.size + self.ridge.intercept_.size


# ===== Model 2: CML 2D Reservoir (spatial) ================================
class CML2DReservoir:
    """Keep 32x32 spatial structure. CML 2D dynamics, Ridge readout."""

    def __init__(self, H: int = GRID_H, W: int = GRID_W,
                 M: int = M_2D, r: float = R_2D,
                 eps: float = EPS_2D, beta: float = BETA_2D,
                 alpha: float = RIDGE_ALPHA, seed: int = SEED):
        self.name = "CML-2D(spatial)"
        rng = torch.Generator().manual_seed(seed)
        self.cml2d = CML2D(H=H, W=W, steps=M, r=r, eps=eps, beta=beta, rng=rng)
        self.alpha = alpha
        self.ridge = None

    def _features(self, X: np.ndarray) -> np.ndarray:
        """X: (N, H, W) -> (N, H*W) CML 2D features."""
        X_t = torch.from_numpy(X).float().unsqueeze(1)  # (N, 1, H, W)
        out = self.cml2d.forward(X_t)  # (N, 1, H, W)
        return out.squeeze(1).reshape(len(X), -1).numpy()  # (N, H*W)

    def fit(self, X: np.ndarray, Y: np.ndarray):
        Ridge = _get_ridge()
        Y_flat = Y.reshape(len(Y), -1)
        feats = self._features(X)
        self.ridge = Ridge(alpha=self.alpha)
        self.ridge.fit(feats, Y_flat)

    def predict(self, X: np.ndarray) -> np.ndarray:
        feats = self._features(X)
        pred_flat = self.ridge.predict(feats)
        return pred_flat.reshape(-1, GRID_H, GRID_W)

    def predict_one(self, x: np.ndarray) -> np.ndarray:
        return self.predict(x[np.newaxis])[0]

    def param_count(self) -> int:
        if self.ridge is None:
            return 0
        return self.ridge.coef_.size + self.ridge.intercept_.size


# ===== Model 3: MLP Baseline ==============================================
class MLPModel(nn.Module):
    def __init__(self, input_size: int = GRID_SIZE, hidden: int = MLP_HIDDEN):
        super().__init__()
        self.name = "MLP"
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, input_size),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (B, 1024)
        return self.net(x)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def train_mlp(model: MLPModel, X_train: np.ndarray, Y_train: np.ndarray,
              X_val: np.ndarray, Y_val: np.ndarray,
              epochs: int = MLP_EPOCHS, lr: float = MLP_LR,
              batch_size: int = MLP_BATCH) -> MLPModel:
    """Train MLP with BCE loss."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    X_tr = torch.from_numpy(X_train.reshape(len(X_train), -1)).float()
    Y_tr = torch.from_numpy(Y_train.reshape(len(Y_train), -1)).float()
    X_v = torch.from_numpy(X_val.reshape(len(X_val), -1)).float()
    Y_v = torch.from_numpy(Y_val.reshape(len(Y_val), -1)).float()

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


# ===== Model 4: Conv2D Baseline ============================================
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

    def forward(self, x):
        # x: (B, 1, H, W)
        return self.net(x)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def train_cnn(model: Conv2DModel, X_train: np.ndarray, Y_train: np.ndarray,
              X_val: np.ndarray, Y_val: np.ndarray,
              epochs: int = CNN_EPOCHS, lr: float = CNN_LR,
              batch_size: int = CNN_BATCH) -> Conv2DModel:
    """Train Conv2D with BCE loss."""
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


# ===== Model 5: ParalESN+CML Hybrid ========================================
class ParalESNCMLHybrid:
    """ParalESN temporal context + CML nonlinear expansion + Ridge readout.

    Treats each GoL trajectory as a sequence:
      - Flatten each 32x32 grid → 1024-dim
      - ParalESN (input=1024, hidden=256) over the sequence for temporal context
      - Use _mix(h) output directly (bypasses zero-init out_proj)
      - Map mix output to [0,1] via (mix+1)/2, feed as drive to 1D CML
      - Ridge readout from CML output (256) → 1024 binary predictions
    """

    def __init__(self, hidden_size: int = PARALESN_HIDDEN,
                 C: int = C_1D, M: int = M_1D,
                 r: float = R_1D, eps: float = EPS_1D, beta: float = BETA_1D,
                 alpha: float = RIDGE_ALPHA, seed: int = SEED):
        self.hidden_size = hidden_size
        self.name = "ParalESN+CML"

        cfg = ParalESNCfg(hidden_size=hidden_size)
        rng = torch.Generator().manual_seed(seed)
        self.paralesn = ParalESNLayer(cfg, layer_idx=0, input_size=GRID_SIZE, rng=rng)
        self.paralesn.eval()
        for p in self.paralesn.parameters():
            p.requires_grad_(False)

        rng_cml = torch.Generator().manual_seed(seed + 10)
        self.cml = CML(C=hidden_size, steps=M, kernel_size=KERNEL_SIZE_1D,
                       r=r, eps=eps, beta=beta, rng=rng_cml)
        self.cml.eval()

        self.alpha = alpha
        self.ridge = None

    def _traj_features(self, trajs_flat: np.ndarray) -> np.ndarray:
        """Process trajectory sequences through ParalESN then CML.

        Args:
            trajs_flat: (N_traj, T, 1024)
        Returns: (N_traj * T, hidden_size) features
        """
        X_t = torch.from_numpy(trajs_flat).float()  # (N_traj, T, 1024)
        with torch.no_grad():
            h, _z = self.paralesn(X_t)  # h: (N_traj, T, hidden_size) complex
            mixed = self.paralesn._mix(h)  # (N_traj, T, hidden_size), tanh, bypasses zero out_proj
            N, T, H = mixed.shape
            mixed_flat = mixed.reshape(N * T, H)  # (N*T, hidden_size)
            drive = (mixed_flat + 1.0) / 2.0  # tanh [-1,1] -> [0,1] for CML
            cml_out = self.cml(drive)  # (N*T, hidden_size)
        return cml_out.numpy()

    def fit(self, X: np.ndarray, Y: np.ndarray, trajs: np.ndarray):
        """Fit on full trajectories to use temporal context.

        Args:
            X: (N_pairs, H, W) — not used directly; trajs provides sequencing
            Y: (N_pairs, H, W) — targets
            trajs: (N_traj, T+1, H, W) — full trajectories for sequencing
        """
        Ridge = _get_ridge()
        # Build input sequences: (N_traj, T, 1024)
        n_traj, t_plus_1, h, w = trajs.shape
        T = t_plus_1 - 1
        seqs = trajs[:, :-1].reshape(n_traj, T, -1)   # inputs
        targets = trajs[:, 1:].reshape(n_traj * T, -1)  # targets, flattened

        feats = self._traj_features(seqs)  # (N_traj*T, hidden_size)
        self.ridge = Ridge(alpha=self.alpha)
        self.ridge.fit(feats, targets)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict one-step without trajectory context (single frames).

        For evaluation compatibility: process as batch of length-1 sequences.
        """
        X_flat = X.reshape(len(X), -1)  # (N, 1024)
        X_seq = X_flat[:, np.newaxis, :]  # (N, 1, 1024) — each as length-1 sequence
        feats = self._traj_features(X_seq)  # (N, hidden_size)
        pred_flat = self.ridge.predict(feats)  # (N, 1024)
        return pred_flat.reshape(-1, GRID_H, GRID_W)

    def predict_one(self, x: np.ndarray) -> np.ndarray:
        """x: (H, W) -> (H, W)."""
        return self.predict(x[np.newaxis])[0]

    def param_count(self) -> int:
        if self.ridge is None:
            return 0
        return self.ridge.coef_.size + self.ridge.intercept_.size


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
    # Compare each grid: all cells must match
    per_grid = np.all(pred_binary.reshape(len(Y_true), -1) ==
                      Y_true.reshape(len(Y_true), -1), axis=1)
    return float(np.mean(per_grid))


def multistep_rollout(predict_fn, x0: np.ndarray,
                      n_steps: int) -> np.ndarray:
    """Roll out a model for n_steps. predict_fn: (H,W) -> (H,W).

    Feed-back is binarized (threshold 0.5) since GoL inputs are binary.
    The raw continuous output is stored for accuracy measurement.

    Returns: (n_steps, H, W) predicted grids (continuous).
    """
    preds = np.zeros((n_steps, GRID_H, GRID_W), dtype=np.float32)
    x = x0.copy()
    for t in range(n_steps):
        raw = predict_fn(x)
        preds[t] = raw
        # Binarize before feeding back (GoL grids are binary)
        x = (np.clip(raw, 0, 1) >= 0.5).astype(np.float32)
    return preds


def multistep_cell_accuracy(true_grids: np.ndarray,
                            pred_grids: np.ndarray) -> float:
    """Cell accuracy over a sequence of grids."""
    pred_binary = (pred_grids >= 0.5).astype(np.float32)
    return float(np.mean(pred_binary == true_grids))


# ===== Main Experiment ======================================================
def run_experiment(args):
    os.environ.setdefault("FORCE_CPU", "1")
    device = pick_device()

    log_wandb = not args.no_wandb
    config = dict(
        grid_h=GRID_H, grid_w=GRID_W, density=DENSITY,
        n_trajectories=N_TRAJECTORIES, traj_length=TRAJ_LENGTH,
        c_1d=C_1D, m_1d=M_1D, r_1d=R_1D, eps_1d=EPS_1D, beta_1d=BETA_1D,
        m_2d=M_2D, r_2d=R_2D, eps_2d=EPS_2D, beta_2d=BETA_2D,
        mlp_hidden=MLP_HIDDEN, mlp_lr=MLP_LR, mlp_epochs=MLP_EPOCHS,
        cnn_lr=CNN_LR, cnn_epochs=CNN_EPOCHS,
        ridge_alpha=RIDGE_ALPHA,
    )

    if log_wandb:
        from wmca.training import init_wandb
        init_wandb("gol-prediction-1b", config=config,
                   tags=["gol", "phase-1b", "reservoir", "2d-cml"])

    print("=" * 72)
    print("PHASE 1b: GAME OF LIFE PREDICTION — CML vs BASELINES")
    print("=" * 72)

    # --- Data ---
    print("\n[1/8] Generating Game of Life trajectories ...")
    t0 = time.time()
    trajs = generate_trajectories()
    train_trajs, val_trajs, test_trajs = split_trajectories(trajs)
    X_train, Y_train = make_pairs(train_trajs)
    X_val, Y_val = make_pairs(val_trajs)
    X_test, Y_test = make_pairs(test_trajs)
    data_time = time.time() - t0
    print(f"  Trajectories: {len(trajs)} x {TRAJ_LENGTH} steps on "
          f"{GRID_H}x{GRID_W} grid  ({data_time:.1f}s)")
    print(f"  Train pairs: {len(X_train)}, Val: {len(X_val)}, "
          f"Test: {len(X_test)}")

    results = {}  # model_name -> dict of metrics

    # --- Model 1: CML 1D Reservoir (flattened) ---
    print("\n[2/8] CML 1D Reservoir (flattened — wrong topology) ...")
    t0 = time.time()
    cml1d = CML1DReservoir()
    cml1d.fit(X_train, Y_train)
    cml1d_pred = cml1d.predict(X_test)
    cml1d_cell_acc = cell_accuracy(Y_test, cml1d_pred)
    cml1d_grid_acc = grid_perfect_accuracy(Y_test, cml1d_pred)
    cml1d_time = time.time() - t0
    cml1d_params = cml1d.param_count()
    print(f"  Cell accuracy:    {cml1d_cell_acc:.4f}")
    print(f"  Grid-perfect acc: {cml1d_grid_acc:.4f}")
    print(f"  Params: {cml1d_params}  ({cml1d_time:.1f}s)")
    results["CML-1D"] = {"cell_acc": cml1d_cell_acc,
                         "grid_acc": cml1d_grid_acc,
                         "params": cml1d_params}

    # --- Model 2: CML 2D Reservoir (spatial) ---
    print("\n[3/8] CML 2D Reservoir (spatial — correct topology) ...")
    t0 = time.time()
    cml2d = CML2DReservoir()
    cml2d.fit(X_train, Y_train)
    cml2d_pred = cml2d.predict(X_test)
    cml2d_cell_acc = cell_accuracy(Y_test, cml2d_pred)
    cml2d_grid_acc = grid_perfect_accuracy(Y_test, cml2d_pred)
    cml2d_time = time.time() - t0
    cml2d_params = cml2d.param_count()
    print(f"  Cell accuracy:    {cml2d_cell_acc:.4f}")
    print(f"  Grid-perfect acc: {cml2d_grid_acc:.4f}")
    print(f"  Params: {cml2d_params}  ({cml2d_time:.1f}s)")
    results["CML-2D"] = {"cell_acc": cml2d_cell_acc,
                         "grid_acc": cml2d_grid_acc,
                         "params": cml2d_params}

    # --- Model 3: MLP Baseline ---
    print("\n[4/8] MLP baseline ...")
    t0 = time.time()
    mlp = MLPModel()
    mlp = train_mlp(mlp, X_train, Y_train, X_val, Y_val)
    mlp.eval()
    with torch.no_grad():
        X_test_t = torch.from_numpy(X_test.reshape(len(X_test), -1)).float()
        mlp_pred_flat = mlp(X_test_t).numpy()
    mlp_pred = mlp_pred_flat.reshape(-1, GRID_H, GRID_W)
    mlp_cell_acc = cell_accuracy(Y_test, mlp_pred)
    mlp_grid_acc = grid_perfect_accuracy(Y_test, mlp_pred)
    mlp_time = time.time() - t0
    mlp_params = mlp.param_count()
    print(f"  Cell accuracy:    {mlp_cell_acc:.4f}")
    print(f"  Grid-perfect acc: {mlp_grid_acc:.4f}")
    print(f"  Params: {mlp_params}  ({mlp_time:.1f}s)")
    results["MLP"] = {"cell_acc": mlp_cell_acc,
                      "grid_acc": mlp_grid_acc,
                      "params": mlp_params}

    # --- Model 4: Conv2D Baseline ---
    print("\n[5/8] Conv2D baseline ...")
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
    print(f"  Params: {cnn_params}  ({cnn_time:.1f}s)")
    results["Conv2D"] = {"cell_acc": cnn_cell_acc,
                         "grid_acc": cnn_grid_acc,
                         "params": cnn_params}

    # --- Model 5: ParalESN+CML Hybrid ---
    print("\n[6/8] ParalESN+CML hybrid (temporal context + CML expansion) ...")
    t0 = time.time()
    hybrid = ParalESNCMLHybrid()
    hybrid.fit(X_train, Y_train, train_trajs)
    hybrid_pred = hybrid.predict(X_test)
    hybrid_cell_acc = cell_accuracy(Y_test, hybrid_pred)
    hybrid_grid_acc = grid_perfect_accuracy(Y_test, hybrid_pred)
    hybrid_time = time.time() - t0
    hybrid_params = hybrid.param_count()
    print(f"  Cell accuracy:    {hybrid_cell_acc:.4f}")
    print(f"  Grid-perfect acc: {hybrid_grid_acc:.4f}")
    print(f"  Params: {hybrid_params}  ({hybrid_time:.1f}s)")
    results["ParalESN+CML"] = {"cell_acc": hybrid_cell_acc,
                               "grid_acc": hybrid_grid_acc,
                               "params": hybrid_params}

    # --- Multi-step rollout ---
    print("\n[7/8] Multi-step rollout evaluation ...")
    # Pick a test trajectory and roll out from its initial state
    test_traj = test_trajs[0]  # (T+1, H, W)
    x0 = test_traj[0]          # initial grid
    max_horizon = min(max(ROLLOUT_HORIZONS), TRAJ_LENGTH)
    true_future = test_traj[1:max_horizon + 1]  # ground truth

    # Define predict functions for each model
    def cml1d_predict(x):
        return cml1d.predict_one(x)

    def cml2d_predict(x):
        return cml2d.predict_one(x)

    def mlp_predict(x):
        mlp.eval()
        with torch.no_grad():
            x_t = torch.from_numpy(x.reshape(1, -1)).float()
            return mlp(x_t).numpy().reshape(GRID_H, GRID_W)

    def cnn_predict(x):
        cnn.eval()
        with torch.no_grad():
            x_t = torch.from_numpy(x).float().reshape(1, 1, GRID_H, GRID_W)
            return cnn(x_t).squeeze().numpy()

    def hybrid_predict(x):
        return hybrid.predict_one(x)

    cml1d_rollout = multistep_rollout(cml1d_predict, x0, max_horizon)
    cml2d_rollout = multistep_rollout(cml2d_predict, x0, max_horizon)
    mlp_rollout = multistep_rollout(mlp_predict, x0, max_horizon)
    cnn_rollout = multistep_rollout(cnn_predict, x0, max_horizon)
    hybrid_rollout = multistep_rollout(hybrid_predict, x0, max_horizon)

    rollout_results = {}
    for h in ROLLOUT_HORIZONS:
        if h > max_horizon:
            break
        true_h = true_future[:h]
        r = {
            "CML-1D": multistep_cell_accuracy(true_h, cml1d_rollout[:h]),
            "CML-2D": multistep_cell_accuracy(true_h, cml2d_rollout[:h]),
            "MLP": multistep_cell_accuracy(true_h, mlp_rollout[:h]),
            "Conv2D": multistep_cell_accuracy(true_h, cnn_rollout[:h]),
            "ParalESN+CML": multistep_cell_accuracy(true_h, hybrid_rollout[:h]),
        }
        rollout_results[h] = r
        print(f"  Horizon {h:>2d}:  "
              f"CML-1D={r['CML-1D']:.4f}  CML-2D={r['CML-2D']:.4f}  "
              f"MLP={r['MLP']:.4f}  Conv2D={r['Conv2D']:.4f}  "
              f"ParalESN+CML={r['ParalESN+CML']:.4f}")

    # --- Plots ---
    print("\n[8/8] Generating plots ...")
    make_plots(results, rollout_results, true_future,
               cml1d_rollout, cml2d_rollout, mlp_rollout, cnn_rollout,
               hybrid_rollout, x0, log_wandb)

    # --- wandb logging ---
    if log_wandb:
        import wandb
        for name, r in results.items():
            tag = name.lower().replace("-", "_").replace("(", "").replace(")", "")
            wandb.log({
                f"one_step/{tag}_cell_acc": r["cell_acc"],
                f"one_step/{tag}_grid_acc": r["grid_acc"],
                f"params/{tag}": r["params"],
            })
        for h, vals in rollout_results.items():
            for name, acc in vals.items():
                tag = name.lower().replace("-", "_")
                wandb.log({f"rollout/horizon_{h}_{tag}": acc})
        wandb.finish()

    # --- Summary ---
    print_summary(results, rollout_results)


# ===== Plotting =============================================================
def make_plots(results, rollout_results, true_future,
               cml1d_rollout, cml2d_rollout, mlp_rollout, cnn_rollout,
               hybrid_rollout, x0, log_wandb):
    plt = _get_plt()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    wandb = None
    if log_wandb:
        import wandb as _wandb
        wandb = _wandb

    model_colors = {
        "CML-1D": "tab:orange",
        "CML-2D": "tab:red",
        "MLP": "tab:blue",
        "Conv2D": "tab:green",
        "ParalESN+CML": "tab:purple",
    }

    # --- Figure 1: Multi-step cell accuracy vs horizon ---
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    horizons = sorted(rollout_results.keys())
    for label, marker in [("CML-1D", "o"), ("CML-2D", "s"),
                           ("MLP", "^"), ("Conv2D", "D"),
                           ("ParalESN+CML", "P")]:
        vals = [rollout_results[h][label] for h in horizons]
        ax1.plot(horizons, vals, f"{marker}-", color=model_colors[label],
                 label=label, markersize=7)
    ax1.set_xlabel("Rollout horizon (steps)")
    ax1.set_ylabel("Cell accuracy")
    ax1.set_title("GoL Multi-step Cell Accuracy vs Horizon")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.4, 1.02)
    fig1.tight_layout()
    fig1.savefig(PLOTS_DIR / "gol_rollout_accuracy.png", dpi=150)
    if wandb:
        wandb.log({"plots/rollout_accuracy": wandb.Image(fig1)})
    plt.close(fig1)

    # --- Figure 2: Example predictions (5 models side-by-side, 3 timesteps) ---
    show_steps = [0, 2, min(4, len(true_future) - 1)]
    fig2, axes = plt.subplots(len(show_steps), 6, figsize=(19, 3 * len(show_steps)))
    col_titles = ["True", "CML-1D", "CML-2D", "MLP", "Conv2D", "ParalESN+CML"]
    rollouts = [true_future, cml1d_rollout, cml2d_rollout,
                mlp_rollout, cnn_rollout, hybrid_rollout]

    for row_i, t in enumerate(show_steps):
        for col_i, (title, data) in enumerate(zip(col_titles, rollouts)):
            ax = axes[row_i, col_i]
            if col_i == 0:
                # True grid is binary
                grid = data[t]
            else:
                # Threshold predictions for display
                grid = (data[t] >= 0.5).astype(np.float32)
            ax.imshow(grid, cmap="binary", vmin=0, vmax=1,
                      interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if row_i == 0:
                ax.set_title(title, fontsize=11)
            if col_i == 0:
                ax.set_ylabel(f"t+{t+1}", fontsize=11)

    fig2.suptitle("GoL: True vs Predicted Grids", fontsize=13, y=1.01)
    fig2.tight_layout()
    fig2.savefig(PLOTS_DIR / "gol_example_predictions.png", dpi=150,
                 bbox_inches="tight")
    if wandb:
        wandb.log({"plots/example_predictions": wandb.Image(fig2)})
    plt.close(fig2)

    # --- Figure 3: Bar chart — 1-step cell accuracy and grid-perfect accuracy ---
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(12, 5))
    names = list(results.keys())
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

    ax3b.bar(x_pos, grid_accs, color=colors, alpha=0.85)
    ax3b.set_xticks(x_pos)
    ax3b.set_xticklabels(names, fontsize=10)
    ax3b.set_ylabel("Grid-Perfect Accuracy")
    ax3b.set_title("1-Step Grid-Perfect Accuracy")
    ax3b.set_ylim(0, max(grid_accs) * 1.3 + 0.01 if max(grid_accs) > 0 else 0.1)
    ax3b.grid(True, alpha=0.3, axis="y")
    for i, v in enumerate(grid_accs):
        ax3b.text(i, v + 0.002, f"{v:.4f}", ha="center", fontsize=9)

    fig3.suptitle("GoL: 1-Step Prediction Accuracy", fontsize=13)
    fig3.tight_layout()
    fig3.savefig(PLOTS_DIR / "gol_accuracy_bars.png", dpi=150)
    if wandb:
        wandb.log({"plots/accuracy_bars": wandb.Image(fig3)})
    plt.close(fig3)

    print(f"  Plots saved to {PLOTS_DIR}/")


# ===== Summary ==============================================================
def print_summary(results, rollout_results):
    print("\n" + "=" * 80)
    print("PHASE 1b SUMMARY: GAME OF LIFE PREDICTION")
    print("=" * 80)

    model_names = ["CML-1D", "CML-2D", "MLP", "Conv2D", "ParalESN+CML"]
    print(f"\n{'Model':<20s}  {'Cell Acc':>10s}  {'Grid-Perfect':>13s}  "
          f"{'Params':>10s}")
    print("-" * 60)
    for name in model_names:
        r = results[name]
        print(f"{name:<20s}  {r['cell_acc']:10.4f}  {r['grid_acc']:13.4f}  "
              f"{r['params']:10d}")

    print(f"\n--- Multi-step Rollout Cell Accuracy ---")
    header = f"{'Horizon':>8s}"
    for name in model_names:
        header += f"  {name:>13s}"
    print(header)
    for h in sorted(rollout_results.keys()):
        row = f"{h:8d}"
        for name in model_names:
            row += f"  {rollout_results[h][name]:13.4f}"
        print(row)

    print("=" * 80)


# ===== Entry Point ==========================================================
def main():
    parser = argparse.ArgumentParser(
        description="Phase 1b: GoL prediction — CML vs baselines")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Skip wandb logging")
    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
