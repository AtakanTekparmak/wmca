"""Phase 1c: Gray-Scott Reaction-Diffusion Prediction — CML/NCA vs Baselines.

Gray-Scott equations (2-channel PDE):
  du/dt = D_u * laplacian(u) - u*v^2 + F*(1-u)
  dv/dt = D_v * laplacian(v) + u*v^2 - (F+k)*v

Parameters: D_u=0.16, D_v=0.08, F=0.035, k=0.065 (mitosis pattern)
Grid: 32x32, dt=1.0 with 4 sub-steps (dt_sub=0.25), periodic BCs
Task: given (u, v) at time t, predict (u, v) at t+1 — 2 channels

Models:
  1. CML-2D (fixed):     Flatten 2 channels, project to 256, CML, Ridge readout
  2. NCA-2D (learned):   Conv2d(2,16,3,pad=1) -> ReLU -> Conv2d(16,2,1)
  3. Conv2D baseline:    3-layer CNN, 2 channels in/out
  4. MLP baseline:       Flatten 2048 -> 512 -> 2048

Usage:
    FORCE_CPU=1 uv run --with scikit-learn,matplotlib,scipy python experiments/pde_gray_scott.py --no-wandb
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wmca.modules.cml import CML
from wmca.utils import pick_device


def _get_plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _get_ridge():
    from sklearn.linear_model import Ridge
    return Ridge


# ===== Constants ===========================================================
PLOTS_DIR = PROJECT_ROOT / "experiments" / "plots"
SEED = 42

GRID_H, GRID_W = 32, 32
GRID_SIZE = GRID_H * GRID_W  # 1024

# Gray-Scott parameters (mitosis pattern)
D_U = 0.16
D_V = 0.08
F_FEED = 0.035
K_KILL = 0.065
DT = 1.0
DX = 1.0
N_SUBSTEPS = 4
DT_SUB = DT / N_SUBSTEPS  # 0.25

# Dataset
N_TRAJECTORIES = 200
TRAJ_LENGTH = 100

# Training
LR = 1e-3
EPOCHS = 50
BATCH_SIZE = 32

# CML
C_CML = 256
M_CML = 15
KERNEL_SIZE_CML = 3
R_CML = 3.90
EPS_CML = 0.3
BETA_CML = 0.15
RIDGE_ALPHA = 1.0

# NCA
HIDDEN_CH = 16

# MLP
MLP_HIDDEN = 512

# Rollout
ROLLOUT_HORIZONS = [1, 5, 10, 25, 50]


# ===== Gray-Scott Simulation ================================================

def _build_laplacian_kernel():
    """Standard 5-point 2D Laplacian stencil as conv kernel."""
    k = np.array([[0, 1, 0],
                  [1, -4, 1],
                  [0, 1, 0]], dtype=np.float32)
    return torch.from_numpy(k).reshape(1, 1, 3, 3)


LAPLACIAN_KERNEL = _build_laplacian_kernel()


def gray_scott_step_torch(u: torch.Tensor, v: torch.Tensor,
                          lap_kernel: torch.Tensor,
                          d_u: float = D_U, d_v: float = D_V,
                          f: float = F_FEED, k: float = K_KILL,
                          dt_sub: float = DT_SUB, dx: float = DX,
                          n_sub: int = N_SUBSTEPS) -> tuple[torch.Tensor, torch.Tensor]:
    """One macro-step of Gray-Scott (n_sub sub-steps).

    u, v: (B, 1, H, W)
    Returns updated (u, v).
    Uses periodic padding for Laplacian.
    """
    coeff = dt_sub / (dx * dx)
    for _ in range(n_sub):
        # Periodic padding (1 pixel each side)
        u_pad = F.pad(u, (1, 1, 1, 1), mode='circular')
        v_pad = F.pad(v, (1, 1, 1, 1), mode='circular')

        lap_u = F.conv2d(u_pad, lap_kernel)  # (B, 1, H, W)
        lap_v = F.conv2d(v_pad, lap_kernel)

        uvv = u * v * v
        du = d_u * coeff * lap_u - uvv + f * (1.0 - u)
        dv = d_v * coeff * lap_v + uvv - (f + k) * v

        u = u + dt_sub * du
        v = v + dt_sub * dv

        u = u.clamp(0.0, 1.0)
        v = v.clamp(0.0, 1.0)

    return u, v


def generate_trajectories(n_traj: int = N_TRAJECTORIES,
                          traj_len: int = TRAJ_LENGTH,
                          h: int = GRID_H, w: int = GRID_W,
                          seed: int = SEED) -> np.ndarray:
    """Generate Gray-Scott trajectories — fully batched over all trajectories.

    Returns: (n_traj, traj_len+1, 2, H, W) float32.
    Channel 0 = u, Channel 1 = v.
    """
    rng = np.random.RandomState(seed)
    lap_k = _build_laplacian_kernel()

    trajs = np.zeros((n_traj, traj_len + 1, 2, h, w), dtype=np.float32)

    # Batched initial conditions: all trajectories at once
    u = torch.ones(n_traj, 1, h, w)
    v = torch.zeros(n_traj, 1, h, w)

    # Each trajectory gets different noise in the 4x4 seed
    ch, cw = h // 2, w // 2
    noise = torch.from_numpy(
        rng.uniform(-0.01, 0.01, (n_traj, 1, 4, 4)).astype(np.float32)
    )
    v[:, :, ch - 2:ch + 2, cw - 2:cw + 2] = 0.25 + noise
    u[:, :, ch - 2:ch + 2, cw - 2:cw + 2] = 0.5

    trajs[:, 0, 0] = u.squeeze(1).numpy()
    trajs[:, 0, 1] = v.squeeze(1).numpy()

    for t in range(traj_len):
        u, v = gray_scott_step_torch(u, v, lap_k)
        trajs[:, t + 1, 0] = u.squeeze(1).numpy()
        trajs[:, t + 1, 1] = v.squeeze(1).numpy()
        if (t + 1) % 25 == 0:
            print(f"    Step {t + 1}/{traj_len}", flush=True)

    return trajs


def normalize_channels(trajs: np.ndarray) -> tuple[np.ndarray, dict]:
    """Normalize u and v channels to [0, 1] separately.

    Returns normalized trajs and stats dict for denormalization.
    """
    stats = {}
    out = trajs.copy()
    for ch, name in enumerate(["u", "v"]):
        ch_data = trajs[:, :, ch]
        cmin = ch_data.min()
        cmax = ch_data.max()
        rng = cmax - cmin
        if rng < 1e-8:
            rng = 1.0
        out[:, :, ch] = (ch_data - cmin) / rng
        stats[name] = {"min": float(cmin), "max": float(cmax), "range": float(rng)}
    return out, stats


def make_pairs(trajs: np.ndarray):
    """(N_traj, T+1, 2, H, W) -> X:(N*T, 2, H, W), Y:(N*T, 2, H, W)."""
    X = trajs[:, :-1].reshape(-1, 2, GRID_H, GRID_W)
    Y = trajs[:, 1:].reshape(-1, 2, GRID_H, GRID_W)
    return X, Y


def split_trajectories(trajs: np.ndarray):
    """70/15/15 split by trajectory."""
    n = len(trajs)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    return trajs[:n_train], trajs[n_train:n_train + n_val], trajs[n_train + n_val:]


# ===== Model 1: CML-2D (fixed reservoir) ====================================

class CML2DReservoir:
    """Flatten 2-channel grid to 2048, project to 256, run 1D CML, Ridge out."""

    def __init__(self, C: int = C_CML, M: int = M_CML,
                 kernel_size: int = KERNEL_SIZE_CML, r: float = R_CML,
                 eps: float = EPS_CML, beta: float = BETA_CML,
                 alpha: float = RIDGE_ALPHA, seed: int = SEED):
        self.C = C
        self.name = "CML-2D"
        self.input_dim = 2 * GRID_SIZE  # 2048

        rng = torch.Generator().manual_seed(seed)
        self.cml = CML(C=C, steps=M, kernel_size=kernel_size,
                       r=r, eps=eps, beta=beta, rng=rng)
        self.cml.eval()

        # Fixed random input projection: 2048 -> C
        self.W_in = torch.randn(self.input_dim, C,
                                generator=torch.Generator().manual_seed(seed + 1))
        self.W_in *= 0.3

        self.alpha = alpha
        self.ridge = None

    def _features(self, X_flat: np.ndarray) -> np.ndarray:
        """(N, 2048) -> (N, C) CML features."""
        X_t = torch.from_numpy(X_flat).float()
        drive = torch.sigmoid(X_t @ self.W_in)
        with torch.no_grad():
            out = self.cml(drive)
        return out.numpy()

    def fit(self, X: np.ndarray, Y: np.ndarray):
        """X, Y: (N, 2, H, W)."""
        Ridge = _get_ridge()
        X_flat = X.reshape(len(X), -1)
        Y_flat = Y.reshape(len(Y), -1)
        feats = self._features(X_flat)
        self.ridge = Ridge(alpha=self.alpha)
        self.ridge.fit(feats, Y_flat)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """X: (N, 2, H, W) -> (N, 2, H, W)."""
        X_flat = X.reshape(len(X), -1)
        feats = self._features(X_flat)
        pred_flat = self.ridge.predict(feats)
        return pred_flat.reshape(-1, 2, GRID_H, GRID_W).astype(np.float32)

    def predict_one(self, x: np.ndarray) -> np.ndarray:
        """x: (2, H, W) -> (2, H, W)."""
        return self.predict(x[np.newaxis])[0]

    def param_count(self) -> int:
        if self.ridge is None:
            return 0
        return self.ridge.coef_.size + self.ridge.intercept_.size


# ===== Model 2: NCA-2D (learned, 1 step) ====================================

class NCA2D(nn.Module):
    """Neural Cellular Automaton for 2-channel Gray-Scott.

    Conv2d(2, hidden, 3, pad=1) -> ReLU -> Conv2d(hidden, 2, 1).
    """

    def __init__(self, hidden_ch: int = HIDDEN_CH):
        super().__init__()
        self.perceive = nn.Conv2d(2, hidden_ch, 3, padding=1)
        self.update = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(hidden_ch, 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 2, H, W) -> (B, 2, H, W)."""
        features = self.perceive(x)
        out = self.update(features)
        return out

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ===== Model 3: Conv2D baseline ==============================================

class Conv2DBaseline(nn.Module):
    """3-layer CNN, 2 channels in/out."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 2, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ===== Model 4: MLP baseline =================================================

class MLPBaseline(nn.Module):
    """Flatten 2048 -> 512 -> 2048."""

    def __init__(self, input_size: int = 2 * GRID_SIZE,
                 hidden: int = MLP_HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, input_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 2048) -> (B, 2048)."""
        return self.net(x)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ===== Training =============================================================

def train_nn_model(model: nn.Module,
                   X_train: np.ndarray, Y_train: np.ndarray,
                   X_val: np.ndarray, Y_val: np.ndarray,
                   epochs: int = EPOCHS, lr: float = LR,
                   batch_size: int = BATCH_SIZE,
                   device: torch.device | None = None,
                   is_mlp: bool = False) -> nn.Module:
    """Train a neural network model with MSE loss + Adam.

    For MLP: flatten to (B, 2048). For conv models: keep (B, 2, H, W).
    """
    if device is None:
        device = torch.device("cpu")

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    if is_mlp:
        X_tr = torch.from_numpy(X_train.reshape(len(X_train), -1)).float().to(device)
        Y_tr = torch.from_numpy(Y_train.reshape(len(Y_train), -1)).float().to(device)
        X_v = torch.from_numpy(X_val.reshape(len(X_val), -1)).float().to(device)
        Y_v = torch.from_numpy(Y_val.reshape(len(Y_val), -1)).float().to(device)
    else:
        X_tr = torch.from_numpy(X_train).float().to(device)
        Y_tr = torch.from_numpy(Y_train).float().to(device)
        X_v = torch.from_numpy(X_val).float().to(device)
        Y_v = torch.from_numpy(Y_val).float().to(device)

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
            val_pred = model(X_v)
            val_loss = criterion(val_pred, Y_v).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1:3d}/{epochs}  "
                  f"train={total_loss / n_batches:.6f}  val={val_loss:.6f}")

    model.load_state_dict(best_state)
    model = model.to(torch.device("cpu"))
    print(f"    Best val_loss: {best_val_loss:.6f}")
    return model


# ===== Evaluation ===========================================================

def compute_mse(Y_true: np.ndarray, Y_pred: np.ndarray) -> float:
    """MSE over all elements."""
    return float(np.mean((Y_true - Y_pred) ** 2))


def multistep_rollout(predict_fn, x0: np.ndarray,
                      n_steps: int) -> np.ndarray:
    """Roll out predict_fn for n_steps.

    predict_fn: (2, H, W) -> (2, H, W)
    Returns: (n_steps, 2, H, W) predictions (continuous).
    No binarization — this is a continuous PDE, feed raw predictions back.
    """
    preds = np.zeros((n_steps, 2, GRID_H, GRID_W), dtype=np.float32)
    x = x0.copy()
    for t in range(n_steps):
        raw = predict_fn(x)
        preds[t] = raw
        x = np.clip(raw, 0, 1)  # keep in valid range but don't binarize
    return preds


def multistep_mse(true_grids: np.ndarray,
                  pred_grids: np.ndarray) -> float:
    """MSE over a sequence of grids."""
    return float(np.mean((true_grids - pred_grids) ** 2))


def make_predict_fn_conv(model: nn.Module):
    """Wrap a (B,2,H,W) model into a (2,H,W) -> (2,H,W) callable."""
    model.eval()

    def _predict(x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x_t = torch.from_numpy(x).float().unsqueeze(0)  # (1, 2, H, W)
            out = model(x_t).squeeze(0).numpy()
        return out

    return _predict


def make_predict_fn_mlp(model: nn.Module):
    """Wrap an MLP into a (2,H,W) -> (2,H,W) callable."""
    model.eval()

    def _predict(x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x_t = torch.from_numpy(x.reshape(1, -1)).float()
            out = model(x_t).numpy().reshape(2, GRID_H, GRID_W)
        return out

    return _predict


# ===== Plotting =============================================================

def make_plots(results: dict, rollout_results: dict,
               true_future: np.ndarray, rollouts: dict,
               x0: np.ndarray, log_wandb: bool):
    plt = _get_plt()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    wandb = None
    if log_wandb:
        import wandb as _wandb
        wandb = _wandb

    model_names = list(results.keys())
    horizons = sorted(rollout_results.keys())

    colors = {
        "CML-2D": "tab:blue",
        "NCA-2D": "tab:orange",
        "Conv2D": "tab:green",
        "MLP": "tab:red",
    }

    # ---- Figure 1: Rollout MSE vs horizon ----
    fig1, ax1 = plt.subplots(figsize=(9, 5))
    markers = ["o", "s", "^", "D"]
    for mi, name in enumerate(model_names):
        vals = [rollout_results[h][name] for h in horizons]
        ax1.plot(horizons, vals, f"{markers[mi % len(markers)]}-",
                 color=colors.get(name, "tab:gray"),
                 label=name, markersize=7)
    ax1.set_xlabel("Rollout horizon (steps)")
    ax1.set_ylabel("MSE")
    ax1.set_title("Gray-Scott: Multi-step Rollout MSE")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale("log")
    fig1.tight_layout()
    path1 = PLOTS_DIR / "gs_rollout_mse.png"
    fig1.savefig(path1, dpi=150)
    if wandb:
        wandb.log({"plots/rollout_mse": wandb.Image(fig1)})
    plt.close(fig1)

    # ---- Figure 2: Example — true vs predicted (u field) at steps 1, 10, 25 ----
    show_steps = [0, 9, 24]  # indices into true_future (0-indexed)
    show_steps = [s for s in show_steps if s < len(true_future)]

    show_models = [n for n in model_names if n in rollouts]

    fig2, axes = plt.subplots(len(show_steps), 1 + len(show_models),
                              figsize=(3 * (1 + len(show_models)),
                                       3 * len(show_steps)))
    if len(show_steps) == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["True"] + show_models
    for row_i, t in enumerate(show_steps):
        # True u-field
        ax = axes[row_i, 0]
        ax.imshow(true_future[t, 0], cmap="viridis", vmin=0, vmax=1,
                  interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        if row_i == 0:
            ax.set_title("True", fontsize=10)
        ax.set_ylabel(f"t+{t+1}", fontsize=10)

        # Model predictions
        for col_i, name in enumerate(show_models, start=1):
            ax = axes[row_i, col_i]
            pred_u = rollouts[name][t, 0]
            ax.imshow(pred_u, cmap="viridis", vmin=0, vmax=1,
                      interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if row_i == 0:
                ax.set_title(name, fontsize=10)

    fig2.suptitle("Gray-Scott: u-field (True vs Predicted)", fontsize=12, y=1.01)
    fig2.tight_layout()
    path2 = PLOTS_DIR / "gs_examples.png"
    fig2.savefig(path2, dpi=150, bbox_inches="tight")
    if wandb:
        wandb.log({"plots/examples": wandb.Image(fig2)})
    plt.close(fig2)

    print(f"  Plots saved -> {path1.name}, {path2.name}")


# ===== Summary ==============================================================

def print_summary(results: dict, rollout_results: dict):
    model_names = list(results.keys())
    horizons = sorted(rollout_results.keys())

    print("\n" + "=" * 90)
    print("SUMMARY: GRAY-SCOTT REACTION-DIFFUSION PREDICTION (PHASE 1c)")
    print("=" * 90)

    # One-step metrics table
    col_w = 14
    header = f"{'Model':<16s}  {'Test MSE':>{col_w}}  {'Params':>{col_w}}"
    print(f"\n{header}")
    print("-" * (16 + 2 * (col_w + 2) + 4))
    for name in model_names:
        r = results[name]
        print(f"{name:<16s}  {r['mse']:{col_w}.6f}  {r['params']:{col_w}d}")

    # Multi-step rollout table
    print(f"\n--- Multi-step Rollout MSE ---")
    hdr = f"{'Horizon':>8s}"
    for name in model_names:
        hdr += f"  {name:>16s}"
    print(hdr)
    for h in horizons:
        row = f"{h:8d}"
        for name in model_names:
            row += f"  {rollout_results[h][name]:16.6f}"
        print(row)

    print("=" * 90)


# ===== Main Experiment ======================================================

def run_experiment(args):
    os.environ.setdefault("FORCE_CPU", "1")
    device = pick_device()

    log_wandb = not args.no_wandb
    config = dict(
        grid_h=GRID_H, grid_w=GRID_W,
        d_u=D_U, d_v=D_V, f_feed=F_FEED, k_kill=K_KILL,
        dt=DT, dx=DX, n_substeps=N_SUBSTEPS,
        n_trajectories=N_TRAJECTORIES, traj_length=TRAJ_LENGTH,
        lr=LR, epochs=EPOCHS, batch_size=BATCH_SIZE,
        c_cml=C_CML, m_cml=M_CML,
        hidden_ch=HIDDEN_CH, mlp_hidden=MLP_HIDDEN,
    )

    if log_wandb:
        from wmca.training import init_wandb
        init_wandb("gs-prediction-1c", config=config,
                   tags=["gray-scott", "phase-1c", "pde", "nca"])

    print("=" * 72, flush=True)
    print("PHASE 1c: GRAY-SCOTT REACTION-DIFFUSION PREDICTION", flush=True)
    print("=" * 72, flush=True)

    # ---- Data ----
    print("\n[1/7] Generating Gray-Scott trajectories ...")
    t0 = time.time()
    trajs_raw = generate_trajectories()
    trajs, norm_stats = normalize_channels(trajs_raw)
    print(f"  Normalization stats: {norm_stats}")

    train_trajs, val_trajs, test_trajs = split_trajectories(trajs)
    X_train, Y_train = make_pairs(train_trajs)
    X_val, Y_val = make_pairs(val_trajs)
    X_test, Y_test = make_pairs(test_trajs)
    data_time = time.time() - t0
    print(f"  {len(trajs)} trajectories x {TRAJ_LENGTH} steps on "
          f"{GRID_H}x{GRID_W} grid (2-ch)  ({data_time:.1f}s)")
    print(f"  Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

    results: dict[str, dict] = {}
    trained_models: dict[str, tuple] = {}  # name -> (model_or_obj, predict_fn)

    # ---- Model 1: CML-2D (fixed reservoir) ----
    print("\n[2/7] CML-2D (fixed reservoir + Ridge readout) ...")
    t0 = time.time()
    cml = CML2DReservoir()
    cml.fit(X_train, Y_train)
    cml_pred = cml.predict(X_test)
    cml_mse = compute_mse(Y_test, cml_pred)
    cml_params = cml.param_count()
    elapsed = time.time() - t0
    print(f"  Test MSE: {cml_mse:.6f}")
    print(f"  Params: {cml_params}  ({elapsed:.1f}s)")
    results["CML-2D"] = {"mse": cml_mse, "params": cml_params}
    trained_models["CML-2D"] = (cml, cml.predict_one)

    # Clean up large arrays
    del cml_pred

    # ---- Model 2: NCA-2D (learned, 1 step) ----
    print("\n[3/7] NCA-2D (learned, 1 step) ...")
    t0 = time.time()
    nca = NCA2D(hidden_ch=HIDDEN_CH)
    nca = train_nn_model(nca, X_train, Y_train, X_val, Y_val,
                         device=device, is_mlp=False)
    nca.eval()
    with torch.no_grad():
        X_t = torch.from_numpy(X_test).float()
        nca_pred = nca(X_t).numpy()
    nca_mse = compute_mse(Y_test, nca_pred)
    nca_params = nca.param_count()
    elapsed = time.time() - t0
    print(f"  Test MSE: {nca_mse:.6f}")
    print(f"  Params: {nca_params}  ({elapsed:.1f}s)")
    results["NCA-2D"] = {"mse": nca_mse, "params": nca_params}
    trained_models["NCA-2D"] = (nca, make_predict_fn_conv(nca))
    del nca_pred

    # ---- Model 3: Conv2D baseline ----
    print("\n[4/7] Conv2D baseline ...")
    t0 = time.time()
    cnn = Conv2DBaseline()
    cnn = train_nn_model(cnn, X_train, Y_train, X_val, Y_val,
                         device=device, is_mlp=False)
    cnn.eval()
    with torch.no_grad():
        X_t = torch.from_numpy(X_test).float()
        cnn_pred = cnn(X_t).numpy()
    cnn_mse = compute_mse(Y_test, cnn_pred)
    cnn_params = cnn.param_count()
    elapsed = time.time() - t0
    print(f"  Test MSE: {cnn_mse:.6f}")
    print(f"  Params: {cnn_params}  ({elapsed:.1f}s)")
    results["Conv2D"] = {"mse": cnn_mse, "params": cnn_params}
    trained_models["Conv2D"] = (cnn, make_predict_fn_conv(cnn))
    del cnn_pred

    # ---- Model 4: MLP baseline ----
    print("\n[5/7] MLP baseline ...")
    t0 = time.time()
    mlp = MLPBaseline()
    mlp = train_nn_model(mlp, X_train, Y_train, X_val, Y_val,
                         device=device, is_mlp=True)
    mlp.eval()
    with torch.no_grad():
        X_t = torch.from_numpy(X_test.reshape(len(X_test), -1)).float()
        mlp_pred = mlp(X_t).numpy().reshape(-1, 2, GRID_H, GRID_W)
    mlp_mse = compute_mse(Y_test, mlp_pred)
    mlp_params = mlp.param_count()
    elapsed = time.time() - t0
    print(f"  Test MSE: {mlp_mse:.6f}")
    print(f"  Params: {mlp_params}  ({elapsed:.1f}s)")
    results["MLP"] = {"mse": mlp_mse, "params": mlp_params}
    trained_models["MLP"] = (mlp, make_predict_fn_mlp(mlp))
    del mlp_pred

    # Free training data
    del X_train, Y_train, X_val, Y_val

    # ---- Multi-step rollout ----
    print("\n[6/7] Multi-step rollout evaluation ...")
    test_traj = test_trajs[0]  # (T+1, 2, H, W)
    x0 = test_traj[0]  # (2, H, W)
    max_horizon = min(max(ROLLOUT_HORIZONS), TRAJ_LENGTH)
    true_future = test_traj[1:max_horizon + 1]  # (max_horizon, 2, H, W)

    rollouts: dict[str, np.ndarray] = {}
    model_names = list(results.keys())

    for name in model_names:
        _, predict_fn = trained_models[name]
        rollouts[name] = multistep_rollout(predict_fn, x0, max_horizon)

    rollout_results: dict[int, dict] = {}
    for h in ROLLOUT_HORIZONS:
        if h > max_horizon:
            break
        true_h = true_future[:h]
        row = {}
        for name in model_names:
            row[name] = multistep_mse(true_h, rollouts[name][:h])
        rollout_results[h] = row
        parts = "  ".join(f"{n}={row[n]:.6f}" for n in model_names)
        print(f"  Horizon {h:>2d}:  {parts}")

    # ---- Plots ----
    print("\n[7/7] Generating plots ...")
    make_plots(results, rollout_results, true_future, rollouts, x0, log_wandb)

    # ---- wandb logging ----
    if log_wandb:
        import wandb
        for name, r in results.items():
            tag = name.lower().replace("-", "_")
            wandb.log({
                f"one_step/{tag}_mse": r["mse"],
                f"params/{tag}": r["params"],
            })
        for h, vals in rollout_results.items():
            for name, mse_val in vals.items():
                tag = name.lower().replace("-", "_")
                wandb.log({f"rollout/horizon_{h}_{tag}": mse_val})
        wandb.finish()

    # ---- Summary ----
    print_summary(results, rollout_results)


# ===== Entry Point ==========================================================

def main():
    parser = argparse.ArgumentParser(
        description="Phase 1c: Gray-Scott reaction-diffusion prediction")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Skip wandb logging")
    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
