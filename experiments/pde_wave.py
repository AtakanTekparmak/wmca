"""Phase 1c: Wave Equation Prediction — CML/NCA vs Baselines.

The 2D wave equation:  d^2u/dt^2 = c^2 * (d^2u/dx^2 + d^2u/dy^2)
is a linear PDE but oscillatory (unlike heat which is diffusive/damping).

Two fields: u (displacement) and v (velocity) since it is 2nd order in time.
Discretisation:
    u_new = u + dt * v
    v_new = v + c^2 * dt * laplacian(u)

Models compared
---------------
1. CML-2D (fixed):  Flatten 2 channels, project to 256 via W_in, CML, Ridge out.
2. NCA-2D (learned, 1 step):  Conv2d(2,16,3,pad=1) -> ReLU -> Conv2d(16,2,1).
3. Conv2D baseline:  3-layer CNN, 2 channels in/out.
4. MLP baseline:     Flatten 2*32*32 -> 512 -> 2*32*32.

Usage
-----
    FORCE_CPU=1 uv run --with scikit-learn,matplotlib,scipy python experiments/pde_wave.py --no-wandb
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

DT = 0.01
C_WAVE = 1.0
DX = 1.0 / GRID_H

N_TRAJECTORIES = 500
TRAJ_LENGTH = 50

# CML reservoir
CML_C = 256
CML_M = 15
CML_KS = 3
CML_R = 3.90
CML_EPS = 0.3
CML_BETA = 0.15
RIDGE_ALPHA = 1.0

# Learned models
LR = 1e-3
EPOCHS = 50
BATCH_SIZE = 64
HIDDEN_CH = 16

ROLLOUT_HORIZONS = [1, 5, 10, 20, 50]


# ===== Data Generation =====================================================

def _laplacian_np(u: np.ndarray) -> np.ndarray:
    """Compute discrete Laplacian via array slicing (zero boundary).

    Much faster than scipy.convolve2d in a per-step loop.
    """
    lap = np.zeros_like(u)
    # Interior points only; boundary stays 0
    lap[1:-1, 1:-1] = (
        u[0:-2, 1:-1] + u[2:, 1:-1] + u[1:-1, 0:-2] + u[1:-1, 2:]
        - 4.0 * u[1:-1, 1:-1]
    ) / (DX ** 2)
    return lap


def wave_step(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Single wave equation step with zero boundary conditions.

    u_new = u + dt * v
    v_new = v + c^2 * dt * laplacian(u)
    """
    lap_u = _laplacian_np(u)
    u_new = u + DT * v
    v_new = v + (C_WAVE ** 2) * DT * lap_u
    # Zero boundary
    u_new[0, :] = u_new[-1, :] = u_new[:, 0] = u_new[:, -1] = 0.0
    v_new[0, :] = v_new[-1, :] = v_new[:, 0] = v_new[:, -1] = 0.0
    return u_new, v_new


def _random_gaussian_blobs(rng: np.random.RandomState,
                           h: int, w: int,
                           n_blobs: int) -> np.ndarray:
    """Create sum of Gaussian blobs on (h, w) grid."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    field = np.zeros((h, w), dtype=np.float32)
    for _ in range(n_blobs):
        cy = rng.uniform(h * 0.2, h * 0.8)
        cx = rng.uniform(w * 0.2, w * 0.8)
        sigma = rng.uniform(1.5, 4.0)
        amp = rng.uniform(0.3, 1.0)
        field += amp * np.exp(-((yy - cy)**2 + (xx - cx)**2) / (2 * sigma**2))
    # Zero boundary
    field[0, :] = field[-1, :] = field[:, 0] = field[:, -1] = 0.0
    return field


def generate_trajectories(n_traj: int = N_TRAJECTORIES,
                          traj_len: int = TRAJ_LENGTH,
                          seed: int = SEED):
    """Generate wave equation trajectories.

    Returns:
        u_trajs: (n_traj, traj_len+1, H, W) displacement
        v_trajs: (n_traj, traj_len+1, H, W) velocity
    """
    rng = np.random.RandomState(seed)

    u_trajs = np.zeros((n_traj, traj_len + 1, GRID_H, GRID_W), dtype=np.float32)
    v_trajs = np.zeros((n_traj, traj_len + 1, GRID_H, GRID_W), dtype=np.float32)

    for i in range(n_traj):
        n_blobs = rng.randint(2, 5)  # 2-4 blobs
        u = _random_gaussian_blobs(rng, GRID_H, GRID_W, n_blobs)
        v = np.zeros((GRID_H, GRID_W), dtype=np.float32)  # zero initial velocity

        u_trajs[i, 0] = u
        v_trajs[i, 0] = v

        for t in range(traj_len):
            u, v = wave_step(u, v)
            u_trajs[i, t + 1] = u
            v_trajs[i, t + 1] = v

    return u_trajs, v_trajs


def normalize_field(field: np.ndarray):
    """Normalize to [0, 1]. Returns (normalized, min, max)."""
    fmin = field.min()
    fmax = field.max()
    rng = fmax - fmin
    if rng < 1e-12:
        return np.zeros_like(field), fmin, fmax
    return (field - fmin) / rng, fmin, fmax


def denormalize_field(field: np.ndarray, fmin: float, fmax: float):
    """Undo normalization."""
    return field * (fmax - fmin) + fmin


def make_pairs(u_trajs: np.ndarray, v_trajs: np.ndarray):
    """Convert to (X, Y) pairs.  X: (N, 2, H, W), Y: (N, 2, H, W)."""
    # Stack channels: (n_traj, T+1, 2, H, W)
    stacked = np.stack([u_trajs, v_trajs], axis=2)
    X = stacked[:, :-1].reshape(-1, 2, GRID_H, GRID_W)
    Y = stacked[:, 1:].reshape(-1, 2, GRID_H, GRID_W)
    return X, Y


def split_trajectories(u_trajs, v_trajs):
    """70/15/15 split by trajectory."""
    n = len(u_trajs)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    return (
        (u_trajs[:n_train], v_trajs[:n_train]),
        (u_trajs[n_train:n_train + n_val], v_trajs[n_train:n_train + n_val]),
        (u_trajs[n_train + n_val:], v_trajs[n_train + n_val:]),
    )


# ===== Model 1: CML-2D (fixed reservoir) ===================================

class CML2DReservoir:
    """Flatten both channels, project to CML_C via W_in, run 1D CML, Ridge out."""

    def __init__(self, seed: int = SEED):
        self.name = "CML-2D"
        rng = torch.Generator().manual_seed(seed)
        self.cml = CML(C=CML_C, steps=CML_M, kernel_size=CML_KS,
                       r=CML_R, eps=CML_EPS, beta=CML_BETA, rng=rng)
        self.cml.eval()
        # Fixed random projection: 2*1024 -> CML_C
        self.W_in = torch.randn(2 * GRID_SIZE, CML_C,
                                generator=torch.Generator().manual_seed(seed + 1))
        self.W_in *= 0.3
        self.ridge = None

    def _features(self, X: np.ndarray) -> np.ndarray:
        """X: (N, 2, H, W) -> (N, CML_C)."""
        X_flat = X.reshape(len(X), -1)  # (N, 2048)
        X_t = torch.from_numpy(X_flat).float()
        drive = torch.sigmoid(X_t @ self.W_in)  # (N, CML_C)
        with torch.no_grad():
            out = self.cml(drive)
        return out.numpy()

    def fit(self, X: np.ndarray, Y: np.ndarray):
        Ridge = _get_ridge()
        Y_flat = Y.reshape(len(Y), -1)  # (N, 2048)
        feats = self._features(X)
        self.ridge = Ridge(alpha=RIDGE_ALPHA)
        self.ridge.fit(feats, Y_flat)

    def predict(self, X: np.ndarray) -> np.ndarray:
        feats = self._features(X)
        pred_flat = self.ridge.predict(feats)
        return pred_flat.reshape(-1, 2, GRID_H, GRID_W)

    def predict_one(self, x: np.ndarray) -> np.ndarray:
        return self.predict(x[np.newaxis])[0]

    def param_count(self) -> int:
        if self.ridge is None:
            return 0
        return self.ridge.coef_.size + self.ridge.intercept_.size


# ===== Model 2: NCA-2D (learned, 1 step) ===================================

class NCA2D(nn.Module):
    """Neural Cellular Automaton: Conv2d(2,16,3) -> ReLU -> Conv2d(16,2,1)."""

    def __init__(self, hidden_ch: int = HIDDEN_CH):
        super().__init__()
        self.perceive = nn.Conv2d(2, hidden_ch, 3, padding=1)
        self.update = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(hidden_ch, 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 2, H, W) -> (B, 2, H, W)."""
        return self.update(self.perceive(x))

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ===== Model 3: Conv2D baseline ============================================

class Conv2DBaseline(nn.Module):
    """3-layer CNN: Conv2d(2,16,3) -> ReLU -> Conv2d(16,16,3) -> ReLU -> Conv2d(16,2,3)."""

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


# ===== Model 4: MLP baseline ===============================================

class MLPBaseline(nn.Module):
    """Flatten 2*32*32=2048 -> 512 -> 2048."""

    def __init__(self, input_size: int = 2 * GRID_SIZE, hidden: int = 512):
        super().__init__()
        self.input_size = input_size
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

def train_model(model: nn.Module,
                X_train: np.ndarray, Y_train: np.ndarray,
                X_val: np.ndarray, Y_val: np.ndarray,
                is_mlp: bool = False,
                epochs: int = EPOCHS, lr: float = LR,
                batch_size: int = BATCH_SIZE,
                device: torch.device | None = None) -> nn.Module:
    """Train a model with MSE loss + Adam.

    For Conv/NCA: input (B, 2, H, W).  For MLP: input (B, 2048).
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
            val_loss = criterion(model(X_v), Y_v).item()

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

def mse_metric(Y_true: np.ndarray, Y_pred: np.ndarray) -> float:
    """MSE over all elements."""
    return float(np.mean((Y_true - Y_pred) ** 2))


def multistep_rollout(predict_fn, x0: np.ndarray,
                      n_steps: int) -> np.ndarray:
    """Roll out predict_fn for n_steps.

    x0: (2, H, W).  predict_fn: (2, H, W) -> (2, H, W).
    Returns: (n_steps, 2, H, W).
    """
    preds = np.zeros((n_steps, 2, GRID_H, GRID_W), dtype=np.float32)
    x = x0.copy()
    for t in range(n_steps):
        x = predict_fn(x)
        preds[t] = x
    return preds


def multistep_mse(true_seq: np.ndarray, pred_seq: np.ndarray) -> float:
    """MSE over a rollout sequence."""
    return float(np.mean((true_seq - pred_seq) ** 2))


def make_predict_fn_conv(model: nn.Module):
    """Wrap (B,2,H,W) model -> (2,H,W)->(2,H,W) callable."""
    model.eval()
    def _predict(x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x_t = torch.from_numpy(x).float().unsqueeze(0)
            return model(x_t).squeeze(0).numpy()
    return _predict


def make_predict_fn_mlp(model: nn.Module):
    """Wrap MLP model -> (2,H,W)->(2,H,W) callable."""
    model.eval()
    def _predict(x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x_t = torch.from_numpy(x.reshape(1, -1)).float()
            return model(x_t).numpy().reshape(2, GRID_H, GRID_W)
    return _predict


def make_predict_fn_cml(reservoir: CML2DReservoir):
    """Wrap CML reservoir -> (2,H,W)->(2,H,W) callable."""
    def _predict(x: np.ndarray) -> np.ndarray:
        return reservoir.predict_one(x)
    return _predict


# ===== Plotting =============================================================

MODEL_COLORS = {
    "CML-2D": "tab:red",
    "NCA-2D": "tab:blue",
    "Conv2D": "tab:green",
    "MLP": "tab:orange",
}


def make_plots(results: dict, rollout_results: dict,
               true_future: np.ndarray, rollouts: dict,
               log_wandb: bool):
    plt = _get_plt()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    wandb = None
    if log_wandb:
        import wandb as _wandb
        wandb = _wandb

    model_names = list(results.keys())
    horizons = sorted(rollout_results.keys())

    # ---- Figure 1: Rollout MSE vs horizon ----
    fig1, ax1 = plt.subplots(figsize=(9, 5))
    markers = {"CML-2D": "s", "NCA-2D": "o", "Conv2D": "D", "MLP": "^"}
    for name in model_names:
        vals = [rollout_results[h][name] for h in horizons]
        ax1.plot(horizons, vals,
                 f"{markers.get(name, 'o')}-",
                 color=MODEL_COLORS.get(name, "tab:gray"),
                 label=name, markersize=7)
    ax1.set_xlabel("Rollout horizon (steps)")
    ax1.set_ylabel("MSE")
    ax1.set_title("Wave Equation: Rollout MSE vs Horizon")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale("log")
    fig1.tight_layout()
    path1 = PLOTS_DIR / "wave_rollout_mse.png"
    fig1.savefig(path1, dpi=150)
    if wandb:
        wandb.log({"plots/wave_rollout_mse": wandb.Image(fig1)})
    plt.close(fig1)

    # ---- Figure 2: Example displacement fields ----
    show_steps = [0, 9, 24]  # t+1, t+10, t+25
    show_steps = [s for s in show_steps if s < len(true_future)]
    n_cols = 1 + len(model_names)  # True + models
    fig2, axes = plt.subplots(len(show_steps), n_cols,
                              figsize=(3.5 * n_cols, 3 * len(show_steps)))
    if len(show_steps) == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["True"] + model_names
    for row_i, t in enumerate(show_steps):
        # True displacement (channel 0 = u)
        ax = axes[row_i, 0]
        im = ax.imshow(true_future[t, 0], cmap="RdBu_r", interpolation="bilinear")
        ax.set_xticks([]); ax.set_yticks([])
        if row_i == 0:
            ax.set_title("True", fontsize=10)
        ax.set_ylabel(f"t+{t+1}", fontsize=10)

        for col_i, name in enumerate(model_names, start=1):
            ax = axes[row_i, col_i]
            pred_u = rollouts[name][t, 0]  # displacement channel
            ax.imshow(pred_u, cmap="RdBu_r", interpolation="bilinear",
                      vmin=true_future[t, 0].min(), vmax=true_future[t, 0].max())
            ax.set_xticks([]); ax.set_yticks([])
            if row_i == 0:
                ax.set_title(name, fontsize=10)

    fig2.suptitle("Wave Eq: True vs Predicted Displacement (u)", fontsize=13, y=1.01)
    fig2.tight_layout()
    path2 = PLOTS_DIR / "wave_example_predictions.png"
    fig2.savefig(path2, dpi=150, bbox_inches="tight")
    if wandb:
        wandb.log({"plots/wave_example_predictions": wandb.Image(fig2)})
    plt.close(fig2)

    print(f"  Plots saved -> {path1.name}, {path2.name}")


# ===== Summary ==============================================================

def print_summary(results: dict, rollout_results: dict):
    model_names = list(results.keys())
    horizons = sorted(rollout_results.keys())

    print("\n" + "=" * 90)
    print("SUMMARY: WAVE EQUATION PREDICTION — PHASE 1c")
    print("=" * 90)

    col_w = 14
    header = f"{'Model':<14s}  {'Test MSE':>{col_w}}  {'Params':>{col_w}}"
    print(f"\n{header}")
    print("-" * (14 + 2 * (col_w + 2) + 4))
    for name in model_names:
        r = results[name]
        print(f"{name:<14s}  {r['mse']:{col_w}.6f}  {r['params']:{col_w}d}")

    print(f"\n--- Multi-step Rollout MSE ---")
    hdr = f"{'Horizon':>8s}"
    for name in model_names:
        hdr += f"  {name:>14s}"
    print(hdr)
    for h in horizons:
        row = f"{h:8d}"
        for name in model_names:
            row += f"  {rollout_results[h][name]:14.6f}"
        print(row)

    print("=" * 90)


# ===== Main Experiment ======================================================

def run_experiment(args):
    os.environ.setdefault("FORCE_CPU", "1")
    # Ensure unbuffered output for real-time monitoring
    sys.stdout.reconfigure(line_buffering=True)
    device = pick_device()

    log_wandb = not args.no_wandb
    config = dict(
        grid_h=GRID_H, grid_w=GRID_W,
        dt=DT, c_wave=C_WAVE, dx=DX,
        n_trajectories=N_TRAJECTORIES, traj_length=TRAJ_LENGTH,
        cml_c=CML_C, cml_m=CML_M, cml_r=CML_R, cml_eps=CML_EPS, cml_beta=CML_BETA,
        lr=LR, epochs=EPOCHS, batch_size=BATCH_SIZE,
        hidden_ch=HIDDEN_CH,
    )

    if log_wandb:
        from wmca.training import init_wandb
        init_wandb("wave-prediction-1c", config=config,
                   tags=["wave", "phase-1c", "pde"])

    print("=" * 72)
    print("PHASE 1c: WAVE EQUATION PREDICTION — CML/NCA vs BASELINES")
    print("=" * 72)

    # ---- Data ----
    print("\n[1/7] Generating wave equation trajectories ...")
    t0 = time.time()
    u_trajs, v_trajs = generate_trajectories()

    # Normalize u and v separately to [0, 1]
    u_all_min, u_all_max = u_trajs.min(), u_trajs.max()
    v_all_min, v_all_max = v_trajs.min(), v_trajs.max()
    u_rng = u_all_max - u_all_min
    v_rng = v_all_max - v_all_min
    if u_rng < 1e-12:
        u_rng = 1.0
    if v_rng < 1e-12:
        v_rng = 1.0
    u_trajs_n = (u_trajs - u_all_min) / u_rng
    v_trajs_n = (v_trajs - v_all_min) / v_rng

    (u_train, v_train), (u_val, v_val), (u_test, v_test) = split_trajectories(
        u_trajs_n, v_trajs_n
    )

    X_train, Y_train = make_pairs(u_train, v_train)
    X_val, Y_val = make_pairs(u_val, v_val)
    X_test, Y_test = make_pairs(u_test, v_test)

    print(f"  {N_TRAJECTORIES} trajectories x {TRAJ_LENGTH} steps on "
          f"{GRID_H}x{GRID_W}  ({time.time()-t0:.1f}s)")
    print(f"  u range: [{u_all_min:.4f}, {u_all_max:.4f}]  "
          f"v range: [{v_all_min:.4f}, {v_all_max:.4f}]")
    print(f"  Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

    results: dict[str, dict] = {}
    trained_models: dict[str, object] = {}

    # ---- Model 1: CML-2D (fixed reservoir) ----
    print("\n[2/7] CML-2D (fixed reservoir + Ridge) ...")
    t0 = time.time()
    cml_res = CML2DReservoir()
    cml_res.fit(X_train, Y_train)
    cml_pred = cml_res.predict(X_test)
    cml_mse = mse_metric(Y_test, cml_pred)
    cml_params = cml_res.param_count()
    elapsed = time.time() - t0
    print(f"  Test MSE: {cml_mse:.6f}")
    print(f"  Params: {cml_params}  ({elapsed:.1f}s)")
    results["CML-2D"] = {"mse": cml_mse, "params": cml_params}
    trained_models["CML-2D"] = cml_res
    del cml_pred

    # ---- Model 2: NCA-2D (learned, 1 step) ----
    print("\n[3/7] NCA-2D (learned, 1 step) ...")
    t0 = time.time()
    nca = NCA2D(hidden_ch=HIDDEN_CH)
    nca = train_model(nca, X_train, Y_train, X_val, Y_val, device=device)
    nca.eval()
    with torch.no_grad():
        nca_pred = nca(torch.from_numpy(X_test).float()).numpy()
    nca_mse = mse_metric(Y_test, nca_pred)
    nca_params = nca.param_count()
    elapsed = time.time() - t0
    print(f"  Test MSE: {nca_mse:.6f}")
    print(f"  Params: {nca_params}  ({elapsed:.1f}s)")
    results["NCA-2D"] = {"mse": nca_mse, "params": nca_params}
    trained_models["NCA-2D"] = nca
    del nca_pred

    # ---- Model 3: Conv2D baseline ----
    print("\n[4/7] Conv2D baseline ...")
    t0 = time.time()
    cnn = Conv2DBaseline()
    cnn = train_model(cnn, X_train, Y_train, X_val, Y_val, device=device)
    cnn.eval()
    with torch.no_grad():
        cnn_pred = cnn(torch.from_numpy(X_test).float()).numpy()
    cnn_mse = mse_metric(Y_test, cnn_pred)
    cnn_params = cnn.param_count()
    elapsed = time.time() - t0
    print(f"  Test MSE: {cnn_mse:.6f}")
    print(f"  Params: {cnn_params}  ({elapsed:.1f}s)")
    results["Conv2D"] = {"mse": cnn_mse, "params": cnn_params}
    trained_models["Conv2D"] = cnn
    del cnn_pred

    # ---- Model 4: MLP baseline ----
    print("\n[5/7] MLP baseline ...")
    t0 = time.time()
    mlp = MLPBaseline()
    mlp = train_model(mlp, X_train, Y_train, X_val, Y_val,
                      is_mlp=True, device=device)
    mlp.eval()
    with torch.no_grad():
        X_test_flat = torch.from_numpy(X_test.reshape(len(X_test), -1)).float()
        mlp_pred = mlp(X_test_flat).numpy().reshape(-1, 2, GRID_H, GRID_W)
    mlp_mse = mse_metric(Y_test, mlp_pred)
    mlp_params = mlp.param_count()
    elapsed = time.time() - t0
    print(f"  Test MSE: {mlp_mse:.6f}")
    print(f"  Params: {mlp_params}  ({elapsed:.1f}s)")
    results["MLP"] = {"mse": mlp_mse, "params": mlp_params}
    trained_models["MLP"] = mlp
    del mlp_pred, X_test_flat

    # ---- Multi-step rollout ----
    print("\n[6/7] Multi-step rollout evaluation ...")
    # Build true future from first test trajectory (normalized)
    stacked_test = np.stack([u_test, v_test], axis=2)  # (n, T+1, 2, H, W)
    test_traj = stacked_test[0]  # (T+1, 2, H, W)
    x0 = test_traj[0]  # (2, H, W)
    max_horizon = min(max(ROLLOUT_HORIZONS), TRAJ_LENGTH)
    true_future = test_traj[1:max_horizon + 1]  # (max_horizon, 2, H, W)

    predict_fns = {
        "CML-2D": make_predict_fn_cml(cml_res),
        "NCA-2D": make_predict_fn_conv(nca),
        "Conv2D": make_predict_fn_conv(cnn),
        "MLP": make_predict_fn_mlp(mlp),
    }

    rollouts: dict[str, np.ndarray] = {}
    for name, fn in predict_fns.items():
        rollouts[name] = multistep_rollout(fn, x0, max_horizon)

    model_names = list(results.keys())
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
    make_plots(results, rollout_results, true_future, rollouts, log_wandb)

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
            for name, v in vals.items():
                tag = name.lower().replace("-", "_")
                wandb.log({f"rollout/horizon_{h}_{tag}": v})
        wandb.finish()

    # ---- Summary ----
    print_summary(results, rollout_results)


# ===== Entry Point ==========================================================

def main():
    parser = argparse.ArgumentParser(
        description="Phase 1c: Wave equation prediction — CML/NCA vs baselines")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Skip wandb logging")
    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
