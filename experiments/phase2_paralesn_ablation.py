"""Phase 2 ParalESN Ablation: 3 injection modes x 4 architecture variants.

Ablates how temporal context from a frozen ParalESN reservoir improves
(or not) the 4 hybrid CML+NCA architectures from Phase 2.

Injection modes:
  Mode 0: No ParalESN  (baseline, same as Phase 2)
  Mode 1: Input injection  — ParalESN features concatenated to model input
  Mode 2: Output injection — ParalESN correction added after model output

Models (4 variants + 2 baselines):
  GatedBlend(A), CMLReg(B), NCAInCML(C), ResCor(D)
  PureNCA (mode 0 only), Conv2D (mode 0 only)

Benchmarks:
  Heat equation (16x16, MSE)
  Game of Life  (16x16, cell accuracy)

Usage:
    FORCE_CPU=1 uv run --with scikit-learn,matplotlib,scipy python experiments/phase2_paralesn_ablation.py --no-wandb
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

from wmca.modules.hybrid import (
    CML2D,
    PureNCA,
    GatedBlendWM,
    CMLRegularizedNCA,
    NCAInsideCML,
    ResidualCorrectionWM,
)
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

GRID_H, GRID_W = 16, 16
GRID_SIZE = GRID_H * GRID_W

# Heat PDE
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
CML_R = 3.90
CML_EPS = 0.3
CML_BETA = 0.15

# Variant B regularization
REG_LAMBDA = 0.1

# ParalESN config
PARALESN_HIDDEN = 64
N_CTX_CH = 2          # number of context channels for input injection
HISTORY_LEN = 5       # past grids fed to ParalESN


# ===== ParalESN Config Dataclass ===========================================

@dataclass
class ParalESNCfg:
    """Minimal config for ParalESNLayer."""
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


# ===== Data Generation (reused from Phase 2) ===============================

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


# ===== Windowed pair extraction =============================================

def make_pairs_mode0(trajs: np.ndarray):
    """Mode 0: simple (x_t, y_{t+1}) pairs. Returns X:(N, H, W), Y:(N, H, W)."""
    X = trajs[:, :-1].reshape(-1, GRID_H, GRID_W)
    Y = trajs[:, 1:].reshape(-1, GRID_H, GRID_W)
    return X, Y


def make_pairs_with_history(trajs: np.ndarray, history_len: int = HISTORY_LEN):
    """Modes 1 & 2: extract (x_current, x_history, y_target) windows.

    For each trajectory, we need at least history_len+1 steps to form one sample.
    x_history: (history_len, H*W) flattened past grids
    x_current: (H, W)
    y_target:  (H, W)

    Returns X_current: (N, H, W), X_history: (N, history_len, H*W), Y: (N, H, W).
    """
    N_traj, T_plus1, H, W = trajs.shape
    T = T_plus1 - 1  # number of transitions

    currents = []
    histories = []
    targets = []

    for i in range(N_traj):
        for t in range(history_len, T):
            # history: grids at t-history_len .. t-1
            hist = trajs[i, t - history_len:t].reshape(history_len, H * W)
            curr = trajs[i, t]
            tgt = trajs[i, t + 1]
            histories.append(hist)
            currents.append(curr)
            targets.append(tgt)

    X_current = np.array(currents, dtype=np.float32)
    X_history = np.array(histories, dtype=np.float32)
    Y = np.array(targets, dtype=np.float32)
    return X_current, X_history, Y


def split_trajectories(trajs: np.ndarray):
    n = len(trajs)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    return trajs[:n_train], trajs[n_train:n_train + n_val], trajs[n_train + n_val:]


# ===== Conv2D Baseline =====================================================

class Conv2DBaseline(nn.Module):
    def __init__(self, in_channels: int = 1, use_sigmoid: bool = False):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, 16, 3, padding=1),
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
        return {"trained": sum(p.numel() for p in self.parameters()), "frozen": 0}


# ===== ParalESN Injection Wrappers ==========================================

class ParalESNInputInjection(nn.Module):
    """Mode 1: ParalESN temporal features concatenated to model input.

    Uses a small learned adapter conv to fuse (1 + n_ctx_ch) channels
    down to 1 channel, so the base model stays at in_channels=1 and
    its output shape is unaffected.
    """

    def __init__(self, base_model: nn.Module, grid_size: int = 16,
                 paralesn_hidden: int = PARALESN_HIDDEN,
                 n_context_channels: int = N_CTX_CH, seed: int = SEED):
        super().__init__()
        self.base_model = base_model
        self.grid_size = grid_size
        self.n_ctx_ch = n_context_channels

        cfg = ParalESNCfg(hidden_size=paralesn_hidden)
        rng = torch.Generator().manual_seed(seed)
        self.paralesn = ParalESNLayer(cfg, layer_idx=0,
                                      input_size=grid_size * grid_size, rng=rng)
        # Freeze reservoir
        for p in self.paralesn.parameters():
            p.requires_grad = False

        self.proj = nn.Linear(paralesn_hidden,
                              n_context_channels * grid_size * grid_size)

        # Adapter: fuse (1 + n_ctx_ch) channels -> 1 channel
        self.adapter = nn.Conv2d(1 + n_context_channels, 1, 1)

    def forward(self, x_current: torch.Tensor, x_history: torch.Tensor) -> torch.Tensor:
        """
        x_current: (B, 1, H, W)
        x_history: (B, T, H*W) flattened past grids
        """
        B = x_current.shape[0]
        H, W = self.grid_size, self.grid_size

        h, _ = self.paralesn(x_history)
        mixed = self.paralesn._mix(h)  # (B, T, hidden) — bypass zero out_proj
        z_last = mixed[:, -1, :]       # (B, hidden)

        ctx = self.proj(z_last)        # (B, n_ctx_ch * H * W)
        ctx = ctx.reshape(B, self.n_ctx_ch, H, W)

        combined = torch.cat([x_current, ctx], dim=1)  # (B, 1+n_ctx_ch, H, W)
        fused = torch.sigmoid(self.adapter(combined))   # (B, 1, H, W) clamped to [0,1]
        out = self.base_model(fused)
        # Handle CMLRegularizedNCA which returns (nca_out, cml_ref) in training
        if isinstance(out, tuple):
            out = out[0]
        return out

    def param_count(self) -> dict[str, int]:
        base_pc = self.base_model.param_count()
        proj_params = sum(p.numel() for p in self.proj.parameters())
        adapter_params = sum(p.numel() for p in self.adapter.parameters())
        paralesn_frozen = sum(b.numel() for b in self.paralesn.buffers())
        return {
            "trained": base_pc["trained"] + proj_params + adapter_params,
            "frozen": base_pc["frozen"] + paralesn_frozen,
        }


class ParalESNOutputInjection(nn.Module):
    """Mode 2: ParalESN correction added after spatial model output."""

    def __init__(self, base_model: nn.Module, grid_size: int = 16,
                 paralesn_hidden: int = PARALESN_HIDDEN, seed: int = SEED):
        super().__init__()
        self.base_model = base_model
        self.grid_size = grid_size

        cfg = ParalESNCfg(hidden_size=paralesn_hidden)
        rng = torch.Generator().manual_seed(seed)
        self.paralesn = ParalESNLayer(cfg, layer_idx=0,
                                      input_size=grid_size * grid_size, rng=rng)
        # Freeze reservoir
        for p in self.paralesn.parameters():
            p.requires_grad = False

        self.proj = nn.Linear(paralesn_hidden, grid_size * grid_size)
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, x_current: torch.Tensor, x_history: torch.Tensor) -> torch.Tensor:
        """
        x_current: (B, 1, H, W)
        x_history: (B, T, H*W)
        """
        B = x_current.shape[0]
        H, W = self.grid_size, self.grid_size

        base_out = self.base_model(x_current)  # (B, 1, H, W) or tuple
        # Handle CMLRegularizedNCA which returns (nca_out, cml_ref) in training
        if isinstance(base_out, tuple):
            base_out = base_out[0]

        h, _ = self.paralesn(x_history)
        mixed = self.paralesn._mix(h)
        z_last = mixed[:, -1, :]

        correction = self.proj(z_last).reshape(B, 1, H, W)
        return torch.clamp(base_out + self.alpha * torch.sigmoid(correction), 0, 1)

    def param_count(self) -> dict[str, int]:
        base_pc = self.base_model.param_count()
        proj_params = sum(p.numel() for p in self.proj.parameters())
        alpha_params = 1
        paralesn_frozen = sum(b.numel() for b in self.paralesn.buffers())
        return {
            "trained": base_pc["trained"] + proj_params + alpha_params,
            "frozen": base_pc["frozen"] + paralesn_frozen,
        }


# ===== Training Utilities ===================================================

def train_model_mode0(model: nn.Module, X_train: np.ndarray, Y_train: np.ndarray,
                      X_val: np.ndarray, Y_val: np.ndarray,
                      loss_type: str = "mse", model_name: str = "",
                      benchmark_name: str = "", is_variant_b: bool = False,
                      epochs: int = EPOCHS, lr: float = LR,
                      batch_size: int = BATCH_SIZE) -> tuple[nn.Module, float]:
    """Train a spatial-only model (mode 0): (B,1,H,W) -> (B,1,H,W)."""
    model = model.to(torch.device("cpu"))
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=lr)

    criterion = nn.MSELoss() if loss_type == "mse" else nn.BCELoss()

    X_tr = torch.from_numpy(X_train).float().unsqueeze(1)
    Y_tr = torch.from_numpy(Y_train).float().unsqueeze(1)
    X_v = torch.from_numpy(X_val).float().unsqueeze(1)
    Y_v = torch.from_numpy(Y_val).float().unsqueeze(1)

    best_val = float("inf")
    best_state = None

    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(X_tr))
        total_loss, n_b = 0.0, 0
        for i in range(0, len(perm), batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = X_tr[idx], Y_tr[idx]
            if is_variant_b:
                nca_out, cml_ref = model(xb)
                pred_loss = criterion(nca_out, yb)
                reg_loss = F.mse_loss(nca_out, cml_ref.detach())
                loss = pred_loss + REG_LAMBDA * reg_loss
            else:
                loss = criterion(model(xb), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_b += 1

        model.eval()
        with torch.no_grad():
            vs, vn = 0.0, 0
            for vi in range(0, len(X_v), batch_size):
                vx, vy = X_v[vi:vi + batch_size], Y_v[vi:vi + batch_size]
                vp = model(vx) if not is_variant_b else model(vx)
                if is_variant_b and isinstance(vp, tuple):
                    vp = vp[0]
                vs += criterion(vp, vy).item() * len(vx)
                vn += len(vx)
            val_loss = vs / vn

        if not np.isnan(val_loss) and val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0:
            print(f"    [{benchmark_name}|{model_name}] ep {epoch+1:3d}/{epochs} "
                  f"train={total_loss/n_b:.6f} val={val_loss:.6f}")

    train_time = time.time() - t0
    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        print(f"    WARNING: no valid checkpoint (all NaN), using last state")
    print(f"    Best val: {best_val:.6f} ({train_time:.1f}s)")

    del X_tr, Y_tr, X_v, Y_v
    gc.collect()
    return model, train_time


def train_model_temporal(model: nn.Module,
                         Xc_train: np.ndarray, Xh_train: np.ndarray, Y_train: np.ndarray,
                         Xc_val: np.ndarray, Xh_val: np.ndarray, Y_val: np.ndarray,
                         loss_type: str = "mse", model_name: str = "",
                         benchmark_name: str = "",
                         epochs: int = EPOCHS, lr: float = LR,
                         batch_size: int = BATCH_SIZE) -> tuple[nn.Module, float]:
    """Train a temporal-injection model (modes 1 & 2).

    forward(x_current, x_history) -> prediction.
    """
    model = model.to(torch.device("cpu"))
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=lr)

    criterion = nn.MSELoss() if loss_type == "mse" else nn.BCELoss()

    Xc_tr = torch.from_numpy(Xc_train).float().unsqueeze(1)   # (N,1,H,W)
    Xh_tr = torch.from_numpy(Xh_train).float()                # (N,T,H*W)
    Y_tr = torch.from_numpy(Y_train).float().unsqueeze(1)     # (N,1,H,W)
    Xc_v = torch.from_numpy(Xc_val).float().unsqueeze(1)
    Xh_v = torch.from_numpy(Xh_val).float()
    Y_v = torch.from_numpy(Y_val).float().unsqueeze(1)

    best_val = float("inf")
    best_state = None

    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(Xc_tr))
        total_loss, n_b = 0.0, 0
        for i in range(0, len(perm), batch_size):
            idx = perm[i:i + batch_size]
            xc, xh, yb = Xc_tr[idx], Xh_tr[idx], Y_tr[idx]
            pred = model(xc, xh)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_b += 1

        model.eval()
        with torch.no_grad():
            vs, vn = 0.0, 0
            for vi in range(0, len(Xc_v), batch_size):
                xc = Xc_v[vi:vi + batch_size]
                xh = Xh_v[vi:vi + batch_size]
                vy = Y_v[vi:vi + batch_size]
                vp = model(xc, xh)
                vs += criterion(vp, vy).item() * len(xc)
                vn += len(xc)
            val_loss = vs / vn

        if not np.isnan(val_loss) and val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0:
            print(f"    [{benchmark_name}|{model_name}] ep {epoch+1:3d}/{epochs} "
                  f"train={total_loss/n_b:.6f} val={val_loss:.6f}")

    train_time = time.time() - t0
    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        print(f"    WARNING: no valid checkpoint (all NaN), using last state")
    print(f"    Best val: {best_val:.6f} ({train_time:.1f}s)")

    del Xc_tr, Xh_tr, Y_tr, Xc_v, Xh_v, Y_v
    gc.collect()
    return model, train_time


# ===== Evaluation ===========================================================

def mse_metric(Y_true: np.ndarray, Y_pred: np.ndarray) -> float:
    return float(np.mean((Y_true - Y_pred) ** 2))


def cell_accuracy(Y_true: np.ndarray, Y_pred: np.ndarray) -> float:
    return float(np.mean((Y_pred >= 0.5).astype(np.float32) == Y_true))


def evaluate_mode0(model: nn.Module, X_test: np.ndarray) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X_test), BATCH_SIZE):
            xb = torch.from_numpy(X_test[i:i + BATCH_SIZE]).float().unsqueeze(1)
            pb = model(xb).squeeze(1).numpy()
            preds.append(pb)
    return np.concatenate(preds, axis=0)


def evaluate_temporal(model: nn.Module, Xc: np.ndarray, Xh: np.ndarray) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(Xc), BATCH_SIZE):
            xc = torch.from_numpy(Xc[i:i + BATCH_SIZE]).float().unsqueeze(1)
            xh = torch.from_numpy(Xh[i:i + BATCH_SIZE]).float()
            pb = model(xc, xh).squeeze(1).numpy()
            preds.append(pb)
    return np.concatenate(preds, axis=0)


# ===== Model Factory ========================================================

def _build_base_model(name: str, in_channels: int = 1, use_sigmoid: bool = False):
    """Build a base spatial model with configurable in_channels."""
    if name == "PureNCA":
        return PureNCA(in_channels=in_channels, hidden_ch=16, steps=1)
    elif name == "GatedBlend(A)":
        return GatedBlendWM(in_channels=in_channels, hidden_ch=16,
                            cml_steps=CML_STEPS, nca_steps=1,
                            r=CML_R, eps=CML_EPS, beta=CML_BETA, seed=SEED)
    elif name == "CMLReg(B)":
        return CMLRegularizedNCA(in_channels=in_channels, hidden_ch=16,
                                 nca_steps=1, r=CML_R, eps=CML_EPS,
                                 beta=CML_BETA, seed=SEED)
    elif name == "NCAInCML(C)":
        return NCAInsideCML(in_channels=in_channels, hidden_ch=16, steps=5,
                            eps=CML_EPS, beta=CML_BETA, seed=SEED)
    elif name == "ResCor(D)":
        return ResidualCorrectionWM(in_channels=in_channels, hidden_ch=16,
                                    cml_steps=CML_STEPS, r=CML_R,
                                    eps=CML_EPS, beta=CML_BETA, seed=SEED)
    elif name == "Conv2D":
        return Conv2DBaseline(in_channels=in_channels, use_sigmoid=use_sigmoid)
    else:
        raise ValueError(f"Unknown model: {name}")


# The 4 variants that get all 3 modes
VARIANT_NAMES = ["GatedBlend(A)", "CMLReg(B)", "NCAInCML(C)", "ResCor(D)"]
# Baselines only get mode 0
BASELINE_NAMES = ["PureNCA", "Conv2D"]

# Which variants use variant_b training
VARIANT_B_SET = {"CMLReg(B)"}


def build_all_configs(use_sigmoid: bool = False):
    """Build all 14 model configs.

    Returns list of dicts:
        name, mode, model, is_variant_b, is_temporal
    """
    configs = []

    # --- Mode 0: all 4 variants + 2 baselines ---
    for vname in VARIANT_NAMES:
        m = _build_base_model(vname, in_channels=1, use_sigmoid=use_sigmoid)
        configs.append({
            "name": vname,
            "mode": 0,
            "label": f"{vname} m0",
            "model": m,
            "is_variant_b": vname in VARIANT_B_SET,
            "is_temporal": False,
        })

    for bname in BASELINE_NAMES:
        m = _build_base_model(bname, in_channels=1, use_sigmoid=use_sigmoid)
        configs.append({
            "name": bname,
            "mode": 0,
            "label": f"{bname} m0",
            "model": m,
            "is_variant_b": False,
            "is_temporal": False,
        })

    # --- Mode 1: input injection (4 variants only) ---
    for vname in VARIANT_NAMES:
        # Base model stays at in_channels=1; the ParalESNInputInjection adapter
        # fuses (1 + n_ctx_ch) channels down to 1 before the base model.
        # CMLReg(B): wrapper hides dual-return, no variant-B reg loss needed.
        base = _build_base_model(vname, in_channels=1, use_sigmoid=use_sigmoid)
        wrapped = ParalESNInputInjection(base, grid_size=GRID_H,
                                         paralesn_hidden=PARALESN_HIDDEN,
                                         n_context_channels=N_CTX_CH, seed=SEED)
        configs.append({
            "name": vname,
            "mode": 1,
            "label": f"{vname} m1",
            "model": wrapped,
            "is_variant_b": False,  # wrapper handles forward
            "is_temporal": True,
        })

    # --- Mode 2: output injection (4 variants only) ---
    for vname in VARIANT_NAMES:
        base = _build_base_model(vname, in_channels=1, use_sigmoid=use_sigmoid)
        wrapped = ParalESNOutputInjection(base, grid_size=GRID_H,
                                          paralesn_hidden=PARALESN_HIDDEN,
                                          seed=SEED)
        configs.append({
            "name": vname,
            "mode": 2,
            "label": f"{vname} m2",
            "model": wrapped,
            "is_variant_b": False,
            "is_temporal": True,
        })

    return configs


# ===== Per-Benchmark Runner =================================================

def run_benchmark(benchmark_name: str, trajs: np.ndarray,
                  loss_type: str):
    """Run all 14 configs on one benchmark. Returns results dict."""
    print(f"\n{'=' * 72}")
    print(f"  BENCHMARK: {benchmark_name}")
    print(f"{'=' * 72}")

    train_t, val_t, test_t = split_trajectories(trajs)

    # Mode 0 data
    X_train_0, Y_train_0 = make_pairs_mode0(train_t)
    X_val_0, Y_val_0 = make_pairs_mode0(val_t)
    X_test_0, Y_test_0 = make_pairs_mode0(test_t)
    print(f"  Mode 0 pairs — train: {len(X_train_0)}, val: {len(X_val_0)}, test: {len(X_test_0)}")

    # Mode 1/2 data (with history)
    Xc_train, Xh_train, Y_train_h = make_pairs_with_history(train_t)
    Xc_val, Xh_val, Y_val_h = make_pairs_with_history(val_t)
    Xc_test, Xh_test, Y_test_h = make_pairs_with_history(test_t)
    print(f"  Mode 1/2 pairs — train: {len(Xc_train)}, val: {len(Xc_val)}, test: {len(Xc_test)}")

    use_sigmoid = (loss_type == "bce")
    configs = build_all_configs(use_sigmoid=use_sigmoid)

    results = {}

    for cfg in configs:
        label = cfg["label"]
        print(f"\n  --- {label} (mode {cfg['mode']}) ---")

        model = cfg["model"]

        if cfg["is_temporal"]:
            model, train_time = train_model_temporal(
                model, Xc_train, Xh_train, Y_train_h,
                Xc_val, Xh_val, Y_val_h,
                loss_type=loss_type, model_name=label,
                benchmark_name=benchmark_name,
            )
            preds = evaluate_temporal(model, Xc_test, Xh_test)
            Y_test_eval = Y_test_h
        else:
            model, train_time = train_model_mode0(
                model, X_train_0, Y_train_0, X_val_0, Y_val_0,
                loss_type=loss_type, model_name=label,
                benchmark_name=benchmark_name,
                is_variant_b=cfg["is_variant_b"],
            )
            preds = evaluate_mode0(model, X_test_0)
            Y_test_eval = Y_test_0

        if loss_type == "mse":
            metric_val = mse_metric(Y_test_eval, preds)
            metric_name = "MSE"
        else:
            metric_val = cell_accuracy(Y_test_eval, preds)
            metric_name = "CellAcc"

        pc = model.param_count()
        print(f"    1-step {metric_name}: {metric_val:.6f}")
        print(f"    Params: trained={pc['trained']}, frozen={pc['frozen']}")
        print(f"    Train time: {train_time:.1f}s")

        results[label] = {
            "name": cfg["name"],
            "mode": cfg["mode"],
            "metric": metric_val,
            "params_trained": pc["trained"],
            "params_frozen": pc["frozen"],
            "train_time": train_time,
        }

        del model
        gc.collect()

    del X_train_0, Y_train_0, X_val_0, Y_val_0, X_test_0, Y_test_0
    del Xc_train, Xh_train, Y_train_h, Xc_val, Xh_val, Y_val_h, Xc_test, Xh_test, Y_test_h
    gc.collect()

    return results


# ===== Summary Tables =======================================================

def print_summary_table(benchmark_name: str, results: dict, metric_name: str):
    """Print a grid: rows = variants, cols = modes."""
    print(f"\n{'=' * 80}")
    print(f"  {benchmark_name} — Variant x Mode Grid")
    print(f"{'=' * 80}")

    # Build grid
    mode_labels = {0: "No ParalESN", 1: "Input Inj.", 2: "Output Inj."}
    print(f"  {'Variant':<16} {'Mode 0':>12} {'Mode 1':>12} {'Mode 2':>12}")
    print(f"  {'-'*16} {'-'*12} {'-'*12} {'-'*12}")

    for vname in VARIANT_NAMES:
        vals = []
        for m in [0, 1, 2]:
            label = f"{vname} m{m}"
            if label in results:
                vals.append(f"{results[label]['metric']:.6f}")
            else:
                vals.append("   ---   ")
        print(f"  {vname:<16} {vals[0]:>12} {vals[1]:>12} {vals[2]:>12}")

    # Baselines
    print(f"  {'-'*16} {'-'*12} {'-'*12} {'-'*12}")
    for bname in BASELINE_NAMES:
        label = f"{bname} m0"
        if label in results:
            val = f"{results[label]['metric']:.6f}"
        else:
            val = "   ---   "
        print(f"  {bname:<16} {val:>12} {'---':>12} {'---':>12}")

    # Param counts
    print(f"\n  Param counts (trained / frozen):")
    for label, r in results.items():
        print(f"    {label:<22} trained={r['params_trained']:>8d}  frozen={r['params_frozen']:>7d}")


# ===== Plotting =============================================================

def plot_heatmap(benchmark_name: str, results: dict, metric_name: str):
    plt = _get_plt()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Build matrix: rows=variants, cols=modes
    n_variants = len(VARIANT_NAMES)
    grid = np.full((n_variants, 3), np.nan)

    for vi, vname in enumerate(VARIANT_NAMES):
        for m in [0, 1, 2]:
            label = f"{vname} m{m}"
            if label in results:
                grid[vi, m] = results[label]["metric"]

    fig, ax = plt.subplots(figsize=(7, 4))
    # For MSE lower is better -> use reversed colormap
    cmap = "RdYlGn_r" if metric_name == "MSE" else "RdYlGn"
    im = ax.imshow(grid, cmap=cmap, aspect="auto")

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Mode 0\n(No ParalESN)", "Mode 1\n(Input Inj.)", "Mode 2\n(Output Inj.)"])
    ax.set_yticks(range(n_variants))
    ax.set_yticklabels(VARIANT_NAMES)

    # Annotate cells
    for vi in range(n_variants):
        for m in range(3):
            val = grid[vi, m]
            if not np.isnan(val):
                ax.text(m, vi, f"{val:.4f}", ha="center", va="center",
                        fontsize=9, fontweight="bold")

    plt.colorbar(im, ax=ax, label=metric_name)
    ax.set_title(f"{benchmark_name}: {metric_name} by Variant x ParalESN Mode")
    fig.tight_layout()

    safe = benchmark_name.lower().replace(" ", "_")
    path = PLOTS_DIR / f"phase2_paralesn_heatmap_{safe}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_bar_chart(benchmark_name: str, results: dict, metric_name: str):
    plt = _get_plt()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    labels = list(results.keys())
    values = [results[l]["metric"] for l in labels]

    # Sort: for MSE ascending (lower=better), for acc descending (higher=better)
    if metric_name == "MSE":
        order = sorted(range(len(values)), key=lambda i: values[i])
    else:
        order = sorted(range(len(values)), key=lambda i: -values[i])

    sorted_labels = [labels[i] for i in order]
    sorted_values = [values[i] for i in order]

    # Color by mode
    mode_colors = {0: "tab:blue", 1: "tab:orange", 2: "tab:green"}
    colors = []
    for l in sorted_labels:
        r = results[l]
        colors.append(mode_colors.get(r["mode"], "tab:gray"))

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(range(len(sorted_labels)), sorted_values, color=colors)
    ax.set_yticks(range(len(sorted_labels)))
    ax.set_yticklabels(sorted_labels, fontsize=8)
    ax.set_xlabel(metric_name)
    ax.set_title(f"{benchmark_name}: All 14 Configs Ranked by {metric_name}")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="tab:blue", label="Mode 0 (no ParalESN)"),
        Patch(facecolor="tab:orange", label="Mode 1 (input inj.)"),
        Patch(facecolor="tab:green", label="Mode 2 (output inj.)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)

    ax.invert_yaxis()
    fig.tight_layout()

    safe = benchmark_name.lower().replace(" ", "_")
    path = PLOTS_DIR / f"phase2_paralesn_bars_{safe}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ===== Main =================================================================

def run_experiment(args):
    os.environ.setdefault("FORCE_CPU", "1")
    device = pick_device()

    print("=" * 72)
    print("PHASE 2 ParalESN ABLATION: 3 Modes x 4 Variants")
    print("=" * 72)
    print(f"  Grid: {GRID_H}x{GRID_W}")
    print(f"  Epochs: {EPOCHS}, LR: {LR}, Batch: {BATCH_SIZE}")
    print(f"  ParalESN hidden: {PARALESN_HIDDEN}, context channels: {N_CTX_CH}")
    print(f"  History length: {HISTORY_LEN}")
    print(f"  Device: {device}")

    all_results = {}

    # ---- Heat Equation ----
    print("\n[1/2] Generating heat equation trajectories ...")
    t0 = time.time()
    heat_trajs = generate_heat_trajectories()
    print(f"  Generated {HEAT_N_TRAJ} trajectories x {HEAT_TRAJ_LEN} steps ({time.time()-t0:.1f}s)")

    results_h = run_benchmark("Heat Equation", heat_trajs, loss_type="mse")
    all_results["Heat Equation"] = ("MSE", results_h)

    del heat_trajs
    gc.collect()

    # ---- Game of Life ----
    print("\n[2/2] Generating Game of Life trajectories ...")
    t0 = time.time()
    gol_trajs = generate_gol_trajectories()
    print(f"  Generated {GOL_N_TRAJ} trajectories x {GOL_TRAJ_LEN} steps ({time.time()-t0:.1f}s)")

    results_g = run_benchmark("Game of Life", gol_trajs, loss_type="bce")
    all_results["Game of Life"] = ("CellAcc", results_g)

    del gol_trajs
    gc.collect()

    # ---- Summary ----
    for bname, (mname, res) in all_results.items():
        print_summary_table(bname, res, mname)

    # ---- Plots ----
    for bname, (mname, res) in all_results.items():
        plot_heatmap(bname, res, mname)
        plot_bar_chart(bname, res, mname)

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()
    run_experiment(args)
