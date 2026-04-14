"""Phase 1a: Lorenz Attractor Prediction — CML Reservoir vs Baselines.

Compares six models on one-step and multi-step Lorenz prediction:
  1. CML Reservoir (ours): fixed CML + Ridge readout
  2. Classic ESN: sparse random recurrent net + Ridge readout
  3. GRU baseline: trained end-to-end with Adam
  4. ParalESN+CML Hybrid: ParalESN temporal context + fixed CML + Ridge
  5. Learned CML: learned MLP replaces fixed logistic map (no temporal memory)
  6. Learned CML + ParalESN: ParalESN temporal backbone + learned CML

Usage:
    uv run --with scikit-learn,matplotlib,scipy python experiments/lorenz_prediction.py --no-wandb
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

def _get_solve_ivp():
    from scipy.integrate import solve_ivp
    return solve_ivp

# ===== Constants ===========================================================
PLOTS_DIR = PROJECT_ROOT / "experiments" / "plots"
SEED = 42
DT = 0.02
N_STEPS = 10000
SIGMA, RHO, BETA_LORENZ = 10.0, 28.0, 8.0 / 3.0

# CML defaults
C = 256
M = 15
KERNEL_SIZE = 3
R_DEFAULT = 3.90
EPS = 0.3
BETA_CML = 0.15

# ESN defaults
ESN_HIDDEN = 256
ESN_SPECTRAL_RADIUS = 0.95
ESN_DENSITY = 0.1
ESN_INPUT_SCALE = 0.1

# GRU defaults
GRU_HIDDEN = 256
GRU_LR = 1e-3
GRU_EPOCHS = 100

# Ridge alpha
RIDGE_ALPHA = 1.0

# Rollout horizons
ROLLOUT_HORIZONS = [1, 5, 10, 25, 50, 100, 200]

# r sweep for CML
R_SWEEP = [3.69, 3.80, 3.90, 3.99]

# ParalESN+CML hybrid defaults
PARALESN_HIDDEN = 256

# Learned CML defaults
LCML_HIDDEN = 64     # MLP bottleneck
LCML_LR = 1e-3
LCML_EPOCHS = 100
LCML_M = 15          # iterative steps (same as fixed CML)


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


# Lyapunov time
LYAPUNOV_EXPONENT = 0.91
LYAPUNOV_TIME_STEPS = int(1.0 / (LYAPUNOV_EXPONENT * DT))  # ~55 steps

# VPT threshold
VPT_THRESHOLD = 0.4


# ===== Data Generation =====================================================
def generate_lorenz(n_steps: int = N_STEPS, dt: float = DT,
                    seed: int = SEED) -> np.ndarray:
    """Generate Lorenz attractor trajectory. Returns (n_steps, 3)."""
    solve_ivp = _get_solve_ivp()
    rng = np.random.RandomState(seed)
    y0 = rng.randn(3) * 0.1 + np.array([1.0, 1.0, 1.0])

    def lorenz(t, state):
        x, y, z = state
        return [
            SIGMA * (y - x),
            x * (RHO - z) - y,
            x * y - BETA_LORENZ * z,
        ]

    t_span = (0.0, n_steps * dt)
    t_eval = np.linspace(0, n_steps * dt, n_steps + 1)
    sol = solve_ivp(lorenz, t_span, y0, method="RK45", t_eval=t_eval,
                    max_step=dt)
    # Drop initial point, keep n_steps points
    return sol.y.T[1:]  # (n_steps, 3)


def normalize_data(data: np.ndarray):
    """Normalize each dim to [0, 1]. Returns (normalized, mins, maxs)."""
    mins = data.min(axis=0)
    maxs = data.max(axis=0)
    ranges = maxs - mins
    ranges[ranges < 1e-8] = 1.0
    normalized = (data - mins) / ranges
    return normalized, mins, maxs


def denormalize(data: np.ndarray, mins: np.ndarray, maxs: np.ndarray):
    return data * (maxs - mins) + mins


def make_dataset(data: np.ndarray):
    """Split into train/val/test. Returns (X, Y) pairs for each split."""
    n = len(data) - 1  # last point has no target
    X = data[:-1]  # state at t
    Y = data[1:]   # state at t+1

    n_train = int(0.70 * n)
    n_val = int(0.15 * n)

    X_train, Y_train = X[:n_train], Y[:n_train]
    X_val, Y_val = X[n_train:n_train + n_val], Y[n_train:n_train + n_val]
    X_test, Y_test = X[n_train + n_val:], Y[n_train + n_val:]

    return (X_train, Y_train), (X_val, Y_val), (X_test, Y_test)


# ===== Model 1: CML Reservoir ==============================================
class CMLReservoir:
    """Fixed CML reservoir + Ridge readout."""

    def __init__(self, C: int, M: int, kernel_size: int, r: float,
                 eps: float, beta: float, alpha: float = RIDGE_ALPHA,
                 seed: int = SEED):
        self.C = C
        self.r = r
        rng = torch.Generator().manual_seed(seed)
        self.cml = CML(C=C, steps=M, kernel_size=kernel_size,
                       r=r, eps=eps, beta=beta, rng=rng)
        self.cml.eval()

        # Fixed random input projection: 3 -> C
        self.W_in = torch.randn(3, C, generator=torch.Generator().manual_seed(seed + 1))
        self.W_in *= 0.5  # scale so sigmoid doesn't saturate too much

        self.alpha = alpha
        self.ridge = None
        self.name = f"CML(r={r})"

    def _features(self, X: np.ndarray) -> np.ndarray:
        """Map (N, 3) -> (N, C) CML features."""
        X_t = torch.from_numpy(X).float()
        # Project to C dims, sigmoid to [0,1]
        drive = torch.sigmoid(X_t @ self.W_in)  # (N, C)
        with torch.no_grad():
            out = self.cml(drive)  # (N, C)
        return out.numpy()

    def fit(self, X: np.ndarray, Y: np.ndarray):
        Ridge = _get_ridge()
        feats = self._features(X)
        self.ridge = Ridge(alpha=self.alpha)
        self.ridge.fit(feats, Y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        feats = self._features(X)
        return self.ridge.predict(feats)

    def predict_one(self, x: np.ndarray) -> np.ndarray:
        """Predict single step: (3,) -> (3,)."""
        return self.predict(x.reshape(1, -1)).flatten()

    def param_count(self) -> int:
        """Trainable params = Ridge readout only."""
        if self.ridge is None:
            return 0
        return self.ridge.coef_.size + self.ridge.intercept_.size


# ===== Model 2: Classic ESN ================================================
class ClassicESN:
    """Echo State Network with sparse reservoir + Ridge readout."""

    def __init__(self, hidden_size: int = ESN_HIDDEN,
                 spectral_radius: float = ESN_SPECTRAL_RADIUS,
                 density: float = ESN_DENSITY,
                 input_scale: float = ESN_INPUT_SCALE,
                 alpha: float = RIDGE_ALPHA,
                 seed: int = SEED):
        rng = np.random.RandomState(seed)
        self.hidden_size = hidden_size
        self.name = "ESN"

        # Sparse reservoir matrix
        W_res = rng.randn(hidden_size, hidden_size)
        mask = (rng.rand(hidden_size, hidden_size) < density).astype(np.float64)
        W_res *= mask
        # Scale to desired spectral radius
        eigenvalues = np.linalg.eigvals(W_res)
        sr = np.max(np.abs(eigenvalues))
        if sr > 1e-10:
            W_res = W_res * (spectral_radius / sr)
        self.W_res = W_res.astype(np.float32)

        # Input weights
        self.W_in = (rng.randn(hidden_size, 3) * input_scale).astype(np.float32)

        self.alpha = alpha
        self.ridge = None

    def _collect_states(self, X: np.ndarray) -> np.ndarray:
        """Run ESN sequentially, collect hidden states. X: (T, 3)."""
        T = len(X)
        H = np.zeros((T, self.hidden_size), dtype=np.float32)
        h = np.zeros(self.hidden_size, dtype=np.float32)
        for t in range(T):
            h = np.tanh(self.W_res @ h + self.W_in @ X[t])
            H[t] = h
        return H

    def fit(self, X: np.ndarray, Y: np.ndarray):
        Ridge = _get_ridge()
        H = self._collect_states(X)
        self.ridge = Ridge(alpha=self.alpha)
        self.ridge.fit(H, Y)
        # Save final hidden state for rollout continuity
        self._last_h = H[-1].copy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        H = self._collect_states(X)
        return self.ridge.predict(H)

    def predict_one_step(self, x: np.ndarray, h: np.ndarray):
        """Single step: returns (prediction, new_h)."""
        h_new = np.tanh(self.W_res @ h + self.W_in @ x)
        pred = self.ridge.predict(h_new.reshape(1, -1)).flatten()
        return pred, h_new

    def warmup_hidden(self, X: np.ndarray) -> np.ndarray:
        """Run through X to get the hidden state at the end."""
        H = self._collect_states(X)
        return H[-1].copy()

    def param_count(self) -> int:
        if self.ridge is None:
            return 0
        return self.ridge.coef_.size + self.ridge.intercept_.size


# ===== Model 3: GRU ========================================================
class GRUModel(nn.Module):
    def __init__(self, hidden_size: int = GRU_HIDDEN):
        super().__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size=3, hidden_size=hidden_size,
                          num_layers=1, batch_first=True)
        self.readout = nn.Linear(hidden_size, 3)
        self.name = "GRU"

    def forward(self, x, h=None):
        # x: (B, T, 3)
        out, h_n = self.gru(x, h)
        pred = self.readout(out)  # (B, T, 3)
        return pred, h_n

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def train_gru(model: GRUModel, X_train: np.ndarray, Y_train: np.ndarray,
              X_val: np.ndarray, Y_val: np.ndarray,
              epochs: int = GRU_EPOCHS, lr: float = GRU_LR,
              seq_len: int = 50):
    """Train GRU on sequences."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Create sequences
    def make_sequences(X, Y, seq_len):
        n = len(X) - seq_len
        xs, ys = [], []
        for i in range(0, n, seq_len // 2):  # overlapping windows
            xs.append(X[i:i + seq_len])
            ys.append(Y[i:i + seq_len])
        return (torch.from_numpy(np.array(xs)).float(),
                torch.from_numpy(np.array(ys)).float())

    X_seq, Y_seq = make_sequences(X_train, Y_train, seq_len)
    X_val_seq, Y_val_seq = make_sequences(X_val, Y_val, seq_len)

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        model.train()
        # Mini-batch
        perm = torch.randperm(len(X_seq))
        batch_size = 64
        total_loss = 0.0
        n_batches = 0
        for i in range(0, len(perm), batch_size):
            idx = perm[i:i + batch_size]
            xb = X_seq[idx]
            yb = Y_seq[idx]
            pred, _ = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred, _ = model(X_val_seq)
            val_loss = criterion(val_pred, Y_val_seq).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1:3d}/{epochs}  "
                  f"train_loss={total_loss / n_batches:.6f}  "
                  f"val_loss={val_loss:.6f}")

    model.load_state_dict(best_state)
    print(f"    Best val_loss: {best_val_loss:.6f}")
    return model


# ===== Model 4: ParalESN+CML Hybrid ========================================
class ParalESNCMLHybrid:
    """ParalESN temporal context + CML nonlinear expansion + Ridge readout."""

    def __init__(self, hidden_size: int = PARALESN_HIDDEN,
                 C: int = C, M: int = M, kernel_size: int = KERNEL_SIZE,
                 r: float = R_DEFAULT, eps: float = EPS, beta: float = BETA_CML,
                 alpha: float = RIDGE_ALPHA, seed: int = SEED):
        self.hidden_size = hidden_size
        self.name = "ParalESN+CML"

        cfg = ParalESNCfg(hidden_size=hidden_size)
        rng = torch.Generator().manual_seed(seed)
        self.paralesn = ParalESNLayer(cfg, layer_idx=0, input_size=3, rng=rng)
        self.paralesn.eval()
        # Freeze the learned out_proj (we don't train anything)
        for p in self.paralesn.parameters():
            p.requires_grad_(False)

        rng_cml = torch.Generator().manual_seed(seed + 10)
        self.cml = CML(C=hidden_size, steps=M, kernel_size=kernel_size,
                       r=r, eps=eps, beta=beta, rng=rng_cml)
        self.cml.eval()

        self.alpha = alpha
        self.ridge = None

    def _features(self, X: np.ndarray) -> np.ndarray:
        """Map (T, 3) -> (T, hidden_size) hybrid features."""
        X_t = torch.from_numpy(X).float().unsqueeze(0)  # (1, T, 3)
        with torch.no_grad():
            h, _z = self.paralesn(X_t)  # _z is zero (out_proj is zero-init)
            mixed = self.paralesn._mix(h)  # (1, T, hidden_size), bypasses zero out_proj
            mixed = mixed.squeeze(0)  # (T, hidden_size)
            drive = (mixed + 1.0) / 2.0  # tanh output [-1,1] -> [0,1] for CML
            cml_out = self.cml(drive)  # (T, hidden_size)
        return cml_out.numpy()

    def fit(self, X: np.ndarray, Y: np.ndarray):
        Ridge = _get_ridge()
        feats = self._features(X)
        self.ridge = Ridge(alpha=self.alpha)
        self.ridge.fit(feats, Y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        feats = self._features(X)
        return self.ridge.predict(feats)

    def param_count(self) -> int:
        if self.ridge is None:
            return 0
        return self.ridge.coef_.size + self.ridge.intercept_.size


def multistep_rollout_hybrid(model: ParalESNCMLHybrid, x0: np.ndarray,
                             warmup_data: np.ndarray,
                             n_steps: int) -> np.ndarray:
    """Roll out ParalESN+CML hybrid for n_steps.

    Maintains ParalESN hidden state across rollout steps.
    warmup_data: (T_warmup, 3) data before test set, used to initialize
                 ParalESN hidden state.
    """
    # Warmup: run ParalESN over warmup_data to get initial hidden state
    warmup_t = torch.from_numpy(warmup_data).float().unsqueeze(0)  # (1, T, 3)
    with torch.no_grad():
        h_warmup, _ = model.paralesn(warmup_t)
        h_prev = h_warmup[:, -1, :]  # (1, hidden_size) complex

    trajectory = np.zeros((n_steps, 3))
    x = torch.from_numpy(x0).float().unsqueeze(0)  # (1, 3)
    with torch.no_grad():
        for t in range(n_steps):
            # ParalESN single token step
            h_prev, _z_t = model.paralesn.forward_token(x, h_prev)
            # _z_t is zero (out_proj zero-init), use _mix_single instead
            mixed_t = model.paralesn._mix_single(h_prev)  # (1, hidden_size)
            drive = (mixed_t + 1.0) / 2.0  # tanh [-1,1] -> [0,1]
            cml_out = model.cml(drive)  # (1, hidden_size)
            # Ridge prediction
            pred = model.ridge.predict(cml_out.numpy()).flatten()
            trajectory[t] = pred
            x = torch.from_numpy(pred).float().unsqueeze(0)
    return trajectory


# ===== Model 5: Learned CML (no temporal memory) ============================
class LearnedCMLUpdate(nn.Module):
    """Small MLP that replaces the fixed logistic map dynamics."""

    def __init__(self, C: int = C, bottleneck: int = LCML_HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(C, bottleneck),
            nn.ReLU(),
            nn.Linear(bottleneck, C),
            nn.Sigmoid(),  # output stays in [0,1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LearnedCML(nn.Module):
    """Replace fixed logistic map with a learned MLP feature expansion.

    Per-timestep: state(3) -> W_in(3->C) -> sigmoid -> LearnedUpdate(C->C, M steps)
                  -> Linear -> prediction(3)
    The LearnedUpdate is a small MLP applied iteratively, replacing r*x*(1-x) + coupling.
    """

    def __init__(self, C: int = C, M: int = LCML_M,
                 bottleneck: int = LCML_HIDDEN, beta: float = BETA_CML,
                 seed: int = SEED):
        super().__init__()
        self.C = C
        self.M = M
        self.beta = beta
        self.name = "LearnedCML"

        # Fixed random input projection (frozen)
        rng = torch.Generator().manual_seed(seed + 1)
        W_in = torch.randn(3, C, generator=rng) * 0.5
        self.register_buffer("W_in", W_in)

        # Learned update MLP
        self.update_mlp = LearnedCMLUpdate(C, bottleneck)

        # Learned output projection
        self.readout = nn.Linear(C, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, 3) -> (N, 3) prediction."""
        drive = torch.sigmoid(x @ self.W_in)  # (N, C) in [0,1]
        grid = drive
        for _ in range(self.M):
            grid = (1 - self.beta) * self.update_mlp(grid) + self.beta * drive
        return self.readout(grid)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Return learned CML features (N, C) for external use."""
        drive = torch.sigmoid(x @ self.W_in)
        grid = drive
        for _ in range(self.M):
            grid = (1 - self.beta) * self.update_mlp(grid) + self.beta * drive
        return grid

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def train_learned_cml(model: LearnedCML, X_train: np.ndarray, Y_train: np.ndarray,
                      X_val: np.ndarray, Y_val: np.ndarray,
                      epochs: int = LCML_EPOCHS, lr: float = LCML_LR):
    """Train LearnedCML end-to-end with Adam on MSE."""
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=lr)
    criterion = nn.MSELoss()

    X_t = torch.from_numpy(X_train).float()
    Y_t = torch.from_numpy(Y_train).float()
    X_v = torch.from_numpy(X_val).float()
    Y_v = torch.from_numpy(Y_val).float()

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        model.train()
        # Mini-batch training
        batch_size = 1024
        perm = torch.randperm(len(X_t))
        total_loss = 0.0
        n_batches = 0
        for i in range(0, len(perm), batch_size):
            idx = perm[i:i + batch_size]
            pred = model(X_t[idx])
            loss = criterion(pred, Y_t[idx])
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

        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1:3d}/{epochs}  "
                  f"train_loss={total_loss / n_batches:.6f}  "
                  f"val_loss={val_loss:.6f}")

    model.load_state_dict(best_state)
    print(f"    Best val_loss: {best_val_loss:.6f}")
    return model


def multistep_rollout_learned_cml(model: LearnedCML, x0: np.ndarray,
                                  n_steps: int) -> np.ndarray:
    """Roll out LearnedCML for n_steps (no temporal memory). x0: (3,)."""
    model.eval()
    trajectory = np.zeros((n_steps, 3))
    x = torch.from_numpy(x0).float().unsqueeze(0)  # (1, 3)
    with torch.no_grad():
        for t in range(n_steps):
            pred = model(x)
            trajectory[t] = pred.squeeze().numpy()
            x = pred
    return trajectory


# ===== Model 6: Learned CML + ParalESN (temporal memory) ====================
class LearnedCMLParalESN(nn.Module):
    """ParalESN temporal backbone + learned CML feature expansion.

    ParalESN processes input sequence in parallel (frozen reservoir).
    Its output drives a learned CML update MLP for M steps.
    Final output goes through a learned linear readout.
    """

    def __init__(self, hidden_size: int = PARALESN_HIDDEN,
                 M: int = LCML_M, bottleneck: int = LCML_HIDDEN,
                 beta: float = BETA_CML, seed: int = SEED):
        super().__init__()
        self.hidden_size = hidden_size
        self.M = M
        self.beta = beta
        self.name = "LearnedCML+ParalESN"

        # Frozen ParalESN
        cfg = ParalESNCfg(hidden_size=hidden_size)
        rng = torch.Generator().manual_seed(seed)
        self.paralesn = ParalESNLayer(cfg, layer_idx=0, input_size=3, rng=rng)
        self.paralesn.eval()
        for p in self.paralesn.parameters():
            p.requires_grad_(False)

        # Learned update MLP (same architecture as LearnedCML)
        self.update_mlp = LearnedCMLUpdate(hidden_size, bottleneck)

        # Learned output projection
        self.readout = nn.Linear(hidden_size, 3)

    def _drive_from_paralesn(self, X_t: torch.Tensor) -> torch.Tensor:
        """Run ParalESN and convert to drive in [0,1].

        X_t: (1, T, 3) -> drive: (T, hidden_size) in [0,1]
        """
        with torch.no_grad():
            h, _z = self.paralesn(X_t)
            mixed = self.paralesn._mix(h)  # (1, T, hidden_size)
            mixed = mixed.squeeze(0)       # (T, hidden_size)
        drive = (mixed + 1.0) / 2.0        # tanh [-1,1] -> [0,1]
        return drive

    def forward(self, X_t: torch.Tensor) -> torch.Tensor:
        """X_t: (1, T, 3) -> predictions (T, 3)."""
        drive = self._drive_from_paralesn(X_t)  # (T, hidden_size)
        grid = drive
        for _ in range(self.M):
            grid = (1 - self.beta) * self.update_mlp(grid) + self.beta * drive
        return self.readout(grid)

    def features(self, X_t: torch.Tensor) -> torch.Tensor:
        """Return learned CML features (T, hidden_size)."""
        drive = self._drive_from_paralesn(X_t)
        grid = drive
        for _ in range(self.M):
            grid = (1 - self.beta) * self.update_mlp(grid) + self.beta * drive
        return grid

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def train_learned_cml_paralesn(model: LearnedCMLParalESN,
                               X_train: np.ndarray, Y_train: np.ndarray,
                               X_val: np.ndarray, Y_val: np.ndarray,
                               epochs: int = LCML_EPOCHS, lr: float = LCML_LR):
    """Train LearnedCML+ParalESN end-to-end (only MLP + readout, ParalESN frozen)."""
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=lr)
    criterion = nn.MSELoss()

    # Pre-compute ParalESN drives (frozen, so we can cache)
    X_train_t = torch.from_numpy(X_train).float().unsqueeze(0)  # (1, T, 3)
    X_val_t = torch.from_numpy(X_val).float().unsqueeze(0)
    Y_t = torch.from_numpy(Y_train).float()
    Y_v = torch.from_numpy(Y_val).float()

    with torch.no_grad():
        train_drive = model._drive_from_paralesn(X_train_t)  # (T_train, hidden)
        val_drive = model._drive_from_paralesn(X_val_t)      # (T_val, hidden)

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        model.train()
        batch_size = 1024
        perm = torch.randperm(len(train_drive))
        total_loss = 0.0
        n_batches = 0
        for i in range(0, len(perm), batch_size):
            idx = perm[i:i + batch_size]
            drive_batch = train_drive[idx]
            grid = drive_batch
            for _ in range(model.M):
                grid = (1 - model.beta) * model.update_mlp(grid) + model.beta * drive_batch
            pred = model.readout(grid)
            loss = criterion(pred, Y_t[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        model.eval()
        with torch.no_grad():
            grid = val_drive
            for _ in range(model.M):
                grid = (1 - model.beta) * model.update_mlp(grid) + model.beta * val_drive
            val_pred = model.readout(grid)
            val_loss = criterion(val_pred, Y_v).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1:3d}/{epochs}  "
                  f"train_loss={total_loss / n_batches:.6f}  "
                  f"val_loss={val_loss:.6f}")

    model.load_state_dict(best_state)
    print(f"    Best val_loss: {best_val_loss:.6f}")
    return model


def multistep_rollout_learned_cml_paralesn(model: LearnedCMLParalESN,
                                           x0: np.ndarray,
                                           warmup_data: np.ndarray,
                                           n_steps: int) -> np.ndarray:
    """Roll out LearnedCML+ParalESN for n_steps.

    Maintains ParalESN hidden state across rollout steps.
    """
    # Warmup: run ParalESN over warmup_data to get initial hidden state
    warmup_t = torch.from_numpy(warmup_data).float().unsqueeze(0)
    with torch.no_grad():
        h_warmup, _ = model.paralesn(warmup_t)
        h_prev = h_warmup[:, -1, :]  # (1, hidden_size) complex

    model.eval()
    trajectory = np.zeros((n_steps, 3))
    x = torch.from_numpy(x0).float().unsqueeze(0)  # (1, 3)
    with torch.no_grad():
        for t in range(n_steps):
            # ParalESN single token step
            h_prev, _z_t = model.paralesn.forward_token(x, h_prev)
            mixed_t = model.paralesn._mix_single(h_prev)  # (1, hidden_size)
            drive = (mixed_t + 1.0) / 2.0  # tanh [-1,1] -> [0,1]

            # Learned CML update
            grid = drive
            for _ in range(model.M):
                grid = (1 - model.beta) * model.update_mlp(grid) + model.beta * drive
            pred = model.readout(grid)

            trajectory[t] = pred.squeeze().numpy()
            x = pred  # feed prediction back
    return trajectory


# ===== Evaluation ===========================================================
def one_step_mse(Y_true: np.ndarray, Y_pred: np.ndarray) -> float:
    return float(np.mean((Y_true - Y_pred) ** 2))


def multistep_rollout_cml(model: CMLReservoir, x0: np.ndarray,
                          n_steps: int) -> np.ndarray:
    """Roll out CML reservoir for n_steps. x0: (3,)."""
    trajectory = np.zeros((n_steps, 3))
    x = x0.copy()
    for t in range(n_steps):
        x = model.predict_one(x)
        trajectory[t] = x
    return trajectory


def multistep_rollout_esn(model: ClassicESN, x0: np.ndarray,
                          h0: np.ndarray, n_steps: int) -> np.ndarray:
    """Roll out ESN for n_steps."""
    trajectory = np.zeros((n_steps, 3))
    x, h = x0.copy(), h0.copy()
    for t in range(n_steps):
        x, h = model.predict_one_step(x, h)
        trajectory[t] = x
    return trajectory


def multistep_rollout_gru(model: GRUModel, x0: np.ndarray,
                          h0: torch.Tensor, n_steps: int) -> np.ndarray:
    """Roll out GRU for n_steps."""
    model.eval()
    trajectory = np.zeros((n_steps, 3))
    x = torch.from_numpy(x0).float().reshape(1, 1, 3)
    h = h0
    with torch.no_grad():
        for t in range(n_steps):
            pred, h = model(x, h)
            x_np = pred.squeeze().numpy()
            trajectory[t] = x_np
            x = pred  # feed prediction back
    return trajectory


def compute_vpt(true_traj: np.ndarray, pred_traj: np.ndarray,
                threshold: float = VPT_THRESHOLD) -> int:
    """Valid prediction time: steps before normalized error > threshold."""
    # Normalize error by the std of the true trajectory
    std = true_traj.std(axis=0)
    std[std < 1e-8] = 1.0
    for t in range(len(pred_traj)):
        err = np.sqrt(np.mean(((true_traj[t] - pred_traj[t]) / std) ** 2))
        if err > threshold:
            return t
    return len(pred_traj)


# ===== Main Experiment ======================================================
def run_experiment(args):
    os.environ.setdefault("FORCE_CPU", "1")
    device = pick_device()

    log_wandb = not args.no_wandb
    config = dict(
        dt=DT, n_steps=N_STEPS, sigma=SIGMA, rho=RHO,
        beta_lorenz=BETA_LORENZ, C=C, M=M, kernel_size=KERNEL_SIZE,
        r_default=R_DEFAULT, eps=EPS, beta_cml=BETA_CML,
        esn_hidden=ESN_HIDDEN, esn_spectral_radius=ESN_SPECTRAL_RADIUS,
        esn_density=ESN_DENSITY, gru_hidden=GRU_HIDDEN,
        gru_lr=GRU_LR, gru_epochs=GRU_EPOCHS,
        ridge_alpha=RIDGE_ALPHA, r_sweep=R_SWEEP,
    )

    if log_wandb:
        from wmca.training import init_wandb
        init_wandb("lorenz-prediction-1a", config=config,
                   tags=["lorenz", "phase-1a", "reservoir"])

    print("=" * 72)
    print("PHASE 1a: LORENZ PREDICTION — CML RESERVOIR vs BASELINES")
    print("=" * 72)

    # --- Data ---
    print("\n[1/8] Generating Lorenz attractor ...")
    raw = generate_lorenz()
    data, mins, maxs = normalize_data(raw)
    (X_train, Y_train), (X_val, Y_val), (X_test, Y_test) = make_dataset(data)
    print(f"  Trajectory: {raw.shape}, dt={DT}")
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # --- CML Reservoir ---
    print("\n[2/8] CML Reservoir ...")
    t0 = time.time()
    cml_model = CMLReservoir(C=C, M=M, kernel_size=KERNEL_SIZE,
                             r=R_DEFAULT, eps=EPS, beta=BETA_CML)
    cml_model.fit(X_train, Y_train)
    cml_pred = cml_model.predict(X_test)
    cml_mse = one_step_mse(Y_test, cml_pred)
    cml_time = time.time() - t0
    print(f"  One-step MSE: {cml_mse:.8f}  ({cml_time:.1f}s)")
    print(f"  Trainable params: {cml_model.param_count()}")

    # --- Classic ESN ---
    print("\n[3/8] Classic ESN ...")
    t0 = time.time()
    esn_model = ClassicESN()
    esn_model.fit(X_train, Y_train)
    # For test one-step MSE: run ESN sequentially over test data
    esn_pred = esn_model.predict(X_test)
    esn_mse = one_step_mse(Y_test, esn_pred)
    esn_time = time.time() - t0
    print(f"  One-step MSE: {esn_mse:.8f}  ({esn_time:.1f}s)")
    print(f"  Trainable params: {esn_model.param_count()}")

    # --- GRU ---
    print("\n[4/8] GRU baseline ...")
    t0 = time.time()
    gru_model = GRUModel()
    gru_model = train_gru(gru_model, X_train, Y_train, X_val, Y_val)
    # One-step MSE on test set
    gru_model.eval()
    with torch.no_grad():
        X_test_t = torch.from_numpy(X_test).float().unsqueeze(0)  # (1, T, 3)
        gru_out, _ = gru_model(X_test_t)
        gru_pred = gru_out.squeeze(0).numpy()
    gru_mse = one_step_mse(Y_test, gru_pred)
    gru_time = time.time() - t0
    print(f"  One-step MSE: {gru_mse:.8f}  ({gru_time:.1f}s)")
    print(f"  Trainable params: {gru_model.param_count()}")

    # --- ParalESN+CML Hybrid ---
    print("\n[5/8] ParalESN+CML Hybrid ...")
    t0 = time.time()
    hybrid_model = ParalESNCMLHybrid()
    hybrid_model.fit(X_train, Y_train)
    hybrid_pred = hybrid_model.predict(X_test)
    hybrid_mse = one_step_mse(Y_test, hybrid_pred)
    hybrid_time = time.time() - t0
    print(f"  One-step MSE: {hybrid_mse:.8f}  ({hybrid_time:.1f}s)")
    print(f"  Trainable params: {hybrid_model.param_count()}")

    # --- Learned CML (no temporal memory) ---
    print("\n[6/8] Learned CML ...")
    t0 = time.time()
    torch.manual_seed(SEED)
    lcml_model = LearnedCML()
    lcml_model = train_learned_cml(lcml_model, X_train, Y_train, X_val, Y_val)
    lcml_model.eval()
    with torch.no_grad():
        X_test_t = torch.from_numpy(X_test).float()
        lcml_pred = lcml_model(X_test_t).numpy()
    lcml_mse = one_step_mse(Y_test, lcml_pred)
    lcml_time = time.time() - t0
    print(f"  One-step MSE: {lcml_mse:.8f}  ({lcml_time:.1f}s)")
    print(f"  Trainable params: {lcml_model.param_count()}")

    # --- Learned CML + ParalESN ---
    print("\n[7/8] Learned CML + ParalESN ...")
    t0 = time.time()
    torch.manual_seed(SEED)
    lcml_paralesn_model = LearnedCMLParalESN()
    lcml_paralesn_model = train_learned_cml_paralesn(
        lcml_paralesn_model, X_train, Y_train, X_val, Y_val)
    lcml_paralesn_model.eval()
    with torch.no_grad():
        X_test_seq = torch.from_numpy(X_test).float().unsqueeze(0)  # (1, T, 3)
        lcml_paralesn_pred = lcml_paralesn_model(X_test_seq).numpy()
    lcml_paralesn_mse = one_step_mse(Y_test, lcml_paralesn_pred)
    lcml_paralesn_time = time.time() - t0
    print(f"  One-step MSE: {lcml_paralesn_mse:.8f}  ({lcml_paralesn_time:.1f}s)")
    print(f"  Trainable params: {lcml_paralesn_model.param_count()}")

    # --- Multi-step rollout ---
    print("\n[8/8] Multi-step rollout evaluation ...")
    # Use the start of test set as initial condition
    # We need a warmup for ESN to get its hidden state
    # Warmup ESN on the data leading up to test set
    n_train = int(0.70 * (len(data) - 1))
    n_val = int(0.15 * (len(data) - 1))
    warmup_data = data[:n_train + n_val]
    esn_h0 = esn_model.warmup_hidden(warmup_data)

    # Warmup GRU on same data
    gru_model.eval()
    with torch.no_grad():
        warmup_t = torch.from_numpy(warmup_data).float().unsqueeze(0)
        _, gru_h0 = gru_model(warmup_t)

    x0 = X_test[0]
    true_future = data[n_train + n_val + 1:]  # ground truth future states
    max_horizon = min(max(ROLLOUT_HORIZONS), len(true_future))

    # Rollout each model
    cml_rollout = multistep_rollout_cml(cml_model, x0, max_horizon)
    esn_rollout = multistep_rollout_esn(esn_model, x0, esn_h0, max_horizon)
    gru_rollout = multistep_rollout_gru(gru_model, x0, gru_h0, max_horizon)
    hybrid_rollout = multistep_rollout_hybrid(hybrid_model, x0, warmup_data,
                                              max_horizon)
    lcml_rollout = multistep_rollout_learned_cml(lcml_model, x0, max_horizon)
    lcml_paralesn_rollout = multistep_rollout_learned_cml_paralesn(
        lcml_paralesn_model, x0, warmup_data, max_horizon)

    true_traj = true_future[:max_horizon]

    # MSE at each horizon
    rollout_results = {}
    for h in ROLLOUT_HORIZONS:
        if h > max_horizon:
            break
        cml_h_mse = one_step_mse(true_traj[:h], cml_rollout[:h])
        esn_h_mse = one_step_mse(true_traj[:h], esn_rollout[:h])
        gru_h_mse = one_step_mse(true_traj[:h], gru_rollout[:h])
        hybrid_h_mse = one_step_mse(true_traj[:h], hybrid_rollout[:h])
        lcml_h_mse = one_step_mse(true_traj[:h], lcml_rollout[:h])
        lcml_pe_h_mse = one_step_mse(true_traj[:h], lcml_paralesn_rollout[:h])
        rollout_results[h] = {"CML": cml_h_mse, "ESN": esn_h_mse,
                              "GRU": gru_h_mse, "Hybrid": hybrid_h_mse,
                              "LearnedCML": lcml_h_mse,
                              "LCML+PE": lcml_pe_h_mse}
        print(f"  Horizon {h:>3d}:  CML={cml_h_mse:.6f}  "
              f"ESN={esn_h_mse:.6f}  GRU={gru_h_mse:.6f}  "
              f"Hybrid={hybrid_h_mse:.6f}  "
              f"LCML={lcml_h_mse:.6f}  LCML+PE={lcml_pe_h_mse:.6f}")

    # VPT
    cml_vpt = compute_vpt(true_traj, cml_rollout)
    esn_vpt = compute_vpt(true_traj, esn_rollout)
    gru_vpt = compute_vpt(true_traj, gru_rollout)
    hybrid_vpt = compute_vpt(true_traj, hybrid_rollout)
    lcml_vpt = compute_vpt(true_traj, lcml_rollout)
    lcml_pe_vpt = compute_vpt(true_traj, lcml_paralesn_rollout)
    cml_vpt_lt = cml_vpt * DT * LYAPUNOV_EXPONENT
    esn_vpt_lt = esn_vpt * DT * LYAPUNOV_EXPONENT
    gru_vpt_lt = gru_vpt * DT * LYAPUNOV_EXPONENT
    hybrid_vpt_lt = hybrid_vpt * DT * LYAPUNOV_EXPONENT
    lcml_vpt_lt = lcml_vpt * DT * LYAPUNOV_EXPONENT
    lcml_pe_vpt_lt = lcml_pe_vpt * DT * LYAPUNOV_EXPONENT
    print(f"\n  VPT (steps):     CML={cml_vpt}  ESN={esn_vpt}  "
          f"GRU={gru_vpt}  Hybrid={hybrid_vpt}  "
          f"LCML={lcml_vpt}  LCML+PE={lcml_pe_vpt}")
    print(f"  VPT (Lyap times): CML={cml_vpt_lt:.2f}  ESN={esn_vpt_lt:.2f}  "
          f"GRU={gru_vpt_lt:.2f}  Hybrid={hybrid_vpt_lt:.2f}  "
          f"LCML={lcml_vpt_lt:.2f}  LCML+PE={lcml_pe_vpt_lt:.2f}")

    # --- r sweep ---
    print("\n--- r sweep (CML reservoir) ---")
    r_sweep_results = {}
    for r in R_SWEEP:
        m = CMLReservoir(C=C, M=M, kernel_size=KERNEL_SIZE,
                         r=r, eps=EPS, beta=BETA_CML)
        m.fit(X_train, Y_train)
        pred = m.predict(X_test)
        mse = one_step_mse(Y_test, pred)
        rollout = multistep_rollout_cml(m, x0, max_horizon)
        vpt = compute_vpt(true_traj, rollout)
        vpt_lt = vpt * DT * LYAPUNOV_EXPONENT
        r_sweep_results[r] = {"mse": mse, "vpt": vpt, "vpt_lt": vpt_lt}
        print(f"  r={r:.2f}  1-step MSE={mse:.8f}  VPT={vpt} steps "
              f"({vpt_lt:.2f} Lyap times)")

    # --- Plots ---
    print("\nGenerating plots ...")
    make_plots(rollout_results, true_traj, cml_rollout, esn_rollout,
               gru_rollout, hybrid_rollout, lcml_rollout,
               lcml_paralesn_rollout, r_sweep_results, log_wandb)

    # --- wandb logging ---
    if log_wandb:
        import wandb
        wandb.log({
            "one_step/cml_mse": cml_mse,
            "one_step/esn_mse": esn_mse,
            "one_step/gru_mse": gru_mse,
            "one_step/hybrid_mse": hybrid_mse,
            "one_step/lcml_mse": lcml_mse,
            "one_step/lcml_paralesn_mse": lcml_paralesn_mse,
            "vpt/cml_steps": cml_vpt,
            "vpt/esn_steps": esn_vpt,
            "vpt/gru_steps": gru_vpt,
            "vpt/hybrid_steps": hybrid_vpt,
            "vpt/lcml_steps": lcml_vpt,
            "vpt/lcml_paralesn_steps": lcml_pe_vpt,
            "vpt/cml_lyap": cml_vpt_lt,
            "vpt/esn_lyap": esn_vpt_lt,
            "vpt/gru_lyap": gru_vpt_lt,
            "vpt/hybrid_lyap": hybrid_vpt_lt,
            "vpt/lcml_lyap": lcml_vpt_lt,
            "vpt/lcml_paralesn_lyap": lcml_pe_vpt_lt,
            "params/cml": cml_model.param_count(),
            "params/esn": esn_model.param_count(),
            "params/gru": gru_model.param_count(),
            "params/hybrid": hybrid_model.param_count(),
            "params/lcml": lcml_model.param_count(),
            "params/lcml_paralesn": lcml_paralesn_model.param_count(),
        })
        for h, vals in rollout_results.items():
            wandb.log({f"rollout/horizon_{h}_cml": vals["CML"],
                       f"rollout/horizon_{h}_esn": vals["ESN"],
                       f"rollout/horizon_{h}_gru": vals["GRU"],
                       f"rollout/horizon_{h}_hybrid": vals["Hybrid"],
                       f"rollout/horizon_{h}_lcml": vals["LearnedCML"],
                       f"rollout/horizon_{h}_lcml_paralesn": vals["LCML+PE"]})
        for r, vals in r_sweep_results.items():
            wandb.log({f"r_sweep/r_{r:.2f}_mse": vals["mse"],
                       f"r_sweep/r_{r:.2f}_vpt": vals["vpt"]})
        wandb.finish()

    # --- Summary table ---
    print_summary(cml_mse, esn_mse, gru_mse, hybrid_mse,
                  lcml_mse, lcml_paralesn_mse,
                  cml_vpt, esn_vpt, gru_vpt, hybrid_vpt,
                  lcml_vpt, lcml_pe_vpt,
                  cml_vpt_lt, esn_vpt_lt, gru_vpt_lt, hybrid_vpt_lt,
                  lcml_vpt_lt, lcml_pe_vpt_lt,
                  cml_model.param_count(), esn_model.param_count(),
                  gru_model.param_count(), hybrid_model.param_count(),
                  lcml_model.param_count(), lcml_paralesn_model.param_count(),
                  rollout_results, r_sweep_results)


# ===== Plotting =============================================================
def make_plots(rollout_results, true_traj, cml_rollout, esn_rollout,
               gru_rollout, hybrid_rollout, lcml_rollout,
               lcml_paralesn_rollout, r_sweep_results, log_wandb):
    plt = _get_plt()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    wandb = None
    if log_wandb:
        import wandb as _wandb
        wandb = _wandb

    # --- Figure 1: Rollout MSE vs horizon (log scale) ---
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    horizons = sorted(rollout_results.keys())
    for label, color, marker in [("CML", "tab:red", "o"),
                                  ("ESN", "tab:blue", "s"),
                                  ("GRU", "tab:green", "^"),
                                  ("Hybrid", "tab:purple", "D"),
                                  ("LearnedCML", "tab:orange", "v"),
                                  ("LCML+PE", "tab:brown", "P")]:
        vals = [rollout_results[h][label] for h in horizons]
        ax1.plot(horizons, vals, f"{marker}-", color=color, label=label,
                 markersize=6)
    ax1.set_xlabel("Rollout horizon (steps)")
    ax1.set_ylabel("MSE (log scale)")
    ax1.set_yscale("log")
    ax1.set_title("Multi-step Rollout MSE vs Horizon")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    fig1.savefig(PLOTS_DIR / "lorenz_rollout_mse.png", dpi=150)
    if wandb:
        wandb.log({"plots/rollout_mse": wandb.Image(fig1)})
    plt.close(fig1)

    # --- Figure 2: Example rollout trajectory (first 200 steps) ---
    n_show = min(200, len(true_traj))
    fig2, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    dim_names = ["x", "y", "z"]
    for d, (ax, name) in enumerate(zip(axes, dim_names)):
        ax.plot(range(n_show), true_traj[:n_show, d], "k-", lw=1.5,
                label="True", alpha=0.8)
        ax.plot(range(n_show), cml_rollout[:n_show, d], "--", color="tab:red",
                lw=1.0, label="CML")
        ax.plot(range(n_show), esn_rollout[:n_show, d], "--", color="tab:blue",
                lw=1.0, label="ESN")
        ax.plot(range(n_show), gru_rollout[:n_show, d], "--",
                color="tab:green", lw=1.0, label="GRU")
        ax.plot(range(n_show), hybrid_rollout[:n_show, d], "--",
                color="tab:purple", lw=1.0, label="Hybrid")
        ax.plot(range(n_show), lcml_rollout[:n_show, d], "--",
                color="tab:orange", lw=1.0, label="LearnedCML")
        ax.plot(range(n_show), lcml_paralesn_rollout[:n_show, d], "--",
                color="tab:brown", lw=1.0, label="LCML+PE")
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.3)
        if d == 0:
            ax.legend(loc="upper right", fontsize=7, ncol=2)
    axes[-1].set_xlabel("Rollout step")
    fig2.suptitle("Lorenz Rollout: True vs Predicted (first 200 steps)")
    fig2.tight_layout()
    fig2.savefig(PLOTS_DIR / "lorenz_rollout_trajectory.png", dpi=150)
    if wandb:
        wandb.log({"plots/rollout_trajectory": wandb.Image(fig2)})
    plt.close(fig2)

    # --- Figure 3: r-sweep VPT ---
    fig3, ax3 = plt.subplots(figsize=(7, 5))
    rs = sorted(r_sweep_results.keys())
    vpts = [r_sweep_results[r]["vpt_lt"] for r in rs]
    ax3.bar([f"{r:.2f}" for r in rs], vpts, color="tab:red", alpha=0.8)
    ax3.set_xlabel("r (logistic map parameter)")
    ax3.set_ylabel("Valid Prediction Time (Lyapunov times)")
    ax3.set_title("CML Reservoir: VPT vs r")
    ax3.grid(True, alpha=0.3, axis="y")
    fig3.tight_layout()
    fig3.savefig(PLOTS_DIR / "lorenz_r_sweep_vpt.png", dpi=150)
    if wandb:
        wandb.log({"plots/r_sweep_vpt": wandb.Image(fig3)})
    plt.close(fig3)

    print(f"  Plots saved to {PLOTS_DIR}/")


# ===== Summary ==============================================================
def print_summary(cml_mse, esn_mse, gru_mse, hybrid_mse,
                  lcml_mse, lcml_paralesn_mse,
                  cml_vpt, esn_vpt, gru_vpt, hybrid_vpt,
                  lcml_vpt, lcml_pe_vpt,
                  cml_vpt_lt, esn_vpt_lt, gru_vpt_lt, hybrid_vpt_lt,
                  lcml_vpt_lt, lcml_pe_vpt_lt,
                  cml_params, esn_params, gru_params, hybrid_params,
                  lcml_params, lcml_pe_params,
                  rollout_results, r_sweep_results):
    print("\n" + "=" * 90)
    print("PHASE 1a SUMMARY: LORENZ PREDICTION")
    print("=" * 90)

    print(f"\n{'Model':<20s}  {'1-step MSE':>12s}  {'VPT (steps)':>12s}  "
          f"{'VPT (Lyap)':>12s}  {'Params':>10s}")
    print("-" * 90)
    for name, mse, vpt, vpt_lt, params in [
        ("CML", cml_mse, cml_vpt, cml_vpt_lt, cml_params),
        ("ESN", esn_mse, esn_vpt, esn_vpt_lt, esn_params),
        ("GRU", gru_mse, gru_vpt, gru_vpt_lt, gru_params),
        ("ParalESN+CML", hybrid_mse, hybrid_vpt, hybrid_vpt_lt, hybrid_params),
        ("LearnedCML", lcml_mse, lcml_vpt, lcml_vpt_lt, lcml_params),
        ("LCML+ParalESN", lcml_paralesn_mse, lcml_pe_vpt, lcml_pe_vpt_lt, lcml_pe_params),
    ]:
        print(f"{name:<20s}  {mse:12.8f}  {vpt:12d}  "
              f"{vpt_lt:12.2f}  {params:10d}")

    print(f"\n--- Multi-step Rollout MSE ---")
    all_models = ["CML", "ESN", "GRU", "Hybrid", "LearnedCML", "LCML+PE"]
    header = f"{'Horizon':>8s}"
    for name in all_models:
        header += f"  {name:>12s}"
    print(header)
    for h in sorted(rollout_results.keys()):
        row = f"{h:8d}"
        for name in all_models:
            row += f"  {rollout_results[h][name]:12.8f}"
        print(row)

    print(f"\n--- r Sweep (CML Reservoir) ---")
    print(f"{'r':>6s}  {'1-step MSE':>12s}  {'VPT (steps)':>12s}  "
          f"{'VPT (Lyap)':>12s}")
    for r in sorted(r_sweep_results.keys()):
        v = r_sweep_results[r]
        print(f"{r:6.2f}  {v['mse']:12.8f}  {v['vpt']:12d}  "
              f"{v['vpt_lt']:12.2f}")

    print("=" * 90)


# ===== Entry Point ==========================================================
def main():
    parser = argparse.ArgumentParser(
        description="Phase 1a: Lorenz prediction — CML vs baselines")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Skip wandb logging")
    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
