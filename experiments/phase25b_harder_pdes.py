"""Phase 2.5b: Harder PDEs — Burgers & Kuramoto-Sivashinsky.

Tests ResCor(D) generalization beyond simple linear PDEs.
  1. ResCor(D)  — CML base + NCA correction  (our method)
  2. PureNCA   — learned-only, no CML
  3. Conv2D    — neural baseline
  4. MLP       — no spatial bias

Usage:
    FORCE_CPU=1 uv run --with scikit-learn,matplotlib,scipy \
        python experiments/phase25b_harder_pdes.py --no-wandb
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wmca.modules.hybrid import ResidualCorrectionWM, PureNCA
from wmca.utils import pick_device

# ===== Helpers ==============================================================

def _get_plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt

PLOTS_DIR = PROJECT_ROOT / "experiments" / "plots"
SEED = 42

# Training hyper-params
LR = 1e-3
EPOCHS = 50
BATCH_SIZE = 64

# Rollout horizons
ROLLOUT_HORIZONS = [1, 5, 10, 25, 50]

# =============================================================================
# PDE 1: Burgers Equation  du/dt + u*du/dx = nu * d2u/dx2
# =============================================================================

BURGERS_NU = 0.01
BURGERS_N = 64
BURGERS_DX = 1.0 / BURGERS_N
BURGERS_DT = 0.001
BURGERS_SUB = 10        # sub-steps per trajectory step for stability
BURGERS_N_TRAJ = 300
BURGERS_TLEN = 50


def burgers_step(u: np.ndarray, nu: float, dx: float, dt: float) -> np.ndarray:
    """One explicit step of viscous Burgers (periodic BCs, upwind advection)."""
    N = len(u)
    u_new = u.copy()
    for _ in range(BURGERS_SUB):
        # Advection — upwind
        adv = np.zeros(N, dtype=np.float32)
        for i in range(N):
            if u_new[i] >= 0:
                adv[i] = u_new[i] * (u_new[i] - u_new[(i - 1) % N]) / dx
            else:
                adv[i] = u_new[i] * (u_new[(i + 1) % N] - u_new[i]) / dx
        # Diffusion — central differences
        diff = np.zeros(N, dtype=np.float32)
        for i in range(N):
            diff[i] = (u_new[(i + 1) % N] - 2 * u_new[i] + u_new[(i - 1) % N]) / (dx * dx)
        u_new = u_new - dt * adv + nu * dt * diff
    return u_new


def _burgers_step_vectorized(u: np.ndarray, nu: float, dx: float, dt: float) -> np.ndarray:
    """Vectorized Burgers step (periodic BCs, upwind + central diff)."""
    for _ in range(BURGERS_SUB):
        u_left = np.roll(u, 1)    # u[i-1]
        u_right = np.roll(u, -1)  # u[i+1]
        # Upwind advection
        du_minus = (u - u_left) / dx
        du_plus = (u_right - u) / dx
        adv = np.where(u >= 0, u * du_minus, u * du_plus)
        # Central diffusion
        diff = (u_right - 2.0 * u + u_left) / (dx * dx)
        u = u - dt * adv + nu * dt * diff
    return u


def generate_burgers_ic(N: int, rng: np.random.RandomState) -> np.ndarray:
    """Sum of 2-4 random sine waves on [0, 1]."""
    x = np.linspace(0, 2 * np.pi, N, endpoint=False).astype(np.float32)
    n_modes = rng.randint(2, 5)
    u = np.zeros(N, dtype=np.float32)
    for _ in range(n_modes):
        k = rng.randint(1, 6)
        amp = rng.uniform(0.3, 1.0)
        phase = rng.uniform(0, 2 * np.pi)
        u += amp * np.sin(k * x + phase)
    # Normalize to [0, 1]
    u_min, u_max = u.min(), u.max()
    if u_max - u_min > 1e-8:
        u = (u - u_min) / (u_max - u_min)
    else:
        u = np.full(N, 0.5, dtype=np.float32)
    return u


def generate_burgers_trajectories(n_traj: int = BURGERS_N_TRAJ,
                                  traj_len: int = BURGERS_TLEN,
                                  seed: int = SEED) -> np.ndarray:
    """Returns (n_traj, traj_len+1, 1, N) float32 — shape (B, T, H=1, W=64)."""
    rng = np.random.RandomState(seed)
    N = BURGERS_N
    trajs = np.zeros((n_traj, traj_len + 1, 1, N), dtype=np.float32)
    for i in range(n_traj):
        u0 = generate_burgers_ic(N, rng)
        trajs[i, 0, 0] = u0
        u = u0.copy()
        for t in range(traj_len):
            u = _burgers_step_vectorized(u, BURGERS_NU, BURGERS_DX, BURGERS_DT)
            # Re-normalize to [0, 1] to keep model-friendly range
            u_min, u_max = u.min(), u.max()
            if u_max - u_min > 1e-8:
                u = (u - u_min) / (u_max - u_min)
            trajs[i, t + 1, 0] = u
    return trajs


# =============================================================================
# PDE 2: Kuramoto-Sivashinsky  du/dt = -u*du/dx - d2u/dx2 - d4u/dx4
# =============================================================================

KS_N = 64
KS_L = 22.0
KS_DX = KS_L / KS_N
KS_DT = 0.05
KS_SUB = 5   # sub-steps per trajectory step
KS_N_TRAJ = 200
KS_TLEN = 100


def ks_step_spectral(u: np.ndarray, L: float, dt: float) -> np.ndarray:
    """Semi-implicit spectral step for KS equation (ETDRK1-like).

    du/dt = -u*du/dx - d2u/dx2 - d4u/dx4
    Linear part L_op = -k^2 - k^4  (implicit)
    Nonlinear part N(u) = -u * du/dx  (explicit)
    """
    N = len(u)
    dt_sub = dt / KS_SUB
    k = np.fft.fftfreq(N, d=L / (2.0 * np.pi * N))  # wavenumbers
    # Correct: for domain [0, L], wavenumbers are 2*pi*n/L
    k = 2.0 * np.pi * np.fft.fftfreq(N, d=L / N)

    k2 = k ** 2
    k4 = k ** 4
    L_op = -k2 - k4  # linear operator in Fourier space

    for _ in range(KS_SUB):
        u_hat = np.fft.fft(u)
        # Nonlinear term: -u * du/dx computed in physical space
        du_dx = np.real(np.fft.ifft(1j * k * u_hat))
        nl = -u * du_dx
        nl_hat = np.fft.fft(nl)
        # Semi-implicit Euler: (u_hat_new - u_hat) / dt = L_op * u_hat_new + nl_hat
        # u_hat_new = (u_hat + dt * nl_hat) / (1 - dt * L_op)
        u_hat_new = (u_hat + dt_sub * nl_hat) / (1.0 - dt_sub * L_op)
        u = np.real(np.fft.ifft(u_hat_new)).astype(np.float32)

    return u


def generate_ks_ic(N: int, L: float, rng: np.random.RandomState) -> np.ndarray:
    """Small random perturbation — KS generates its own chaos."""
    x = np.linspace(0, L, N, endpoint=False).astype(np.float32)
    u = np.zeros(N, dtype=np.float32)
    n_modes = rng.randint(3, 7)
    for _ in range(n_modes):
        kk = rng.randint(1, 6)
        amp = rng.uniform(0.01, 0.1)
        phase = rng.uniform(0, 2 * np.pi)
        u += amp * np.sin(2 * np.pi * kk * x / L + phase)
    return u


def generate_ks_trajectories(n_traj: int = KS_N_TRAJ,
                             traj_len: int = KS_TLEN,
                             seed: int = SEED) -> np.ndarray:
    """Returns (n_traj, traj_len+1, 1, N) — shape suitable for Conv2d with H=1."""
    rng = np.random.RandomState(seed)
    N = KS_N
    # First, generate raw trajectories to find global normalization range
    raw_trajs = []
    for i in range(n_traj):
        u0 = generate_ks_ic(N, KS_L, rng)
        traj = [u0.copy()]
        u = u0.copy()
        # Warm-up: let transients die out
        for _ in range(200):
            u = ks_step_spectral(u, KS_L, KS_DT)
        traj = [u.copy()]
        for t in range(traj_len):
            u = ks_step_spectral(u, KS_L, KS_DT)
            traj.append(u.copy())
        raw_trajs.append(np.array(traj, dtype=np.float32))

    # Global normalization to [0, 1]
    all_vals = np.concatenate([t.ravel() for t in raw_trajs])
    g_min, g_max = float(all_vals.min()), float(all_vals.max())
    print(f"  KS raw range: [{g_min:.3f}, {g_max:.3f}]")

    trajs = np.zeros((n_traj, traj_len + 1, 1, N), dtype=np.float32)
    for i, raw in enumerate(raw_trajs):
        normed = (raw - g_min) / (g_max - g_min + 1e-8)
        trajs[i, :, 0, :] = normed

    return trajs


# =============================================================================
# Models
# =============================================================================

class Conv2DBaseline(nn.Module):
    """3-layer CNN baseline. Works for H=1 1D PDEs via Conv2d."""

    def __init__(self, in_ch: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, in_ch, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def param_count(self) -> dict[str, int]:
        return {"trained": sum(p.numel() for p in self.parameters()), "frozen": 0}


class MLPBaseline(nn.Module):
    """Flatten -> MLP -> reshape. No spatial inductive bias."""

    def __init__(self, input_size: int = 64, hidden: int = 256):
        super().__init__()
        self.input_size = input_size
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, input_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, 1, W)
        B = x.shape[0]
        flat = x.reshape(B, -1)
        out = self.net(flat)
        return out.reshape(B, 1, 1, self.input_size)

    def param_count(self) -> dict[str, int]:
        return {"trained": sum(p.numel() for p in self.parameters()), "frozen": 0}


# =============================================================================
# Training & Evaluation (generic)
# =============================================================================

def train_model(model: nn.Module,
                X_train: np.ndarray, Y_train: np.ndarray,
                X_val: np.ndarray, Y_val: np.ndarray,
                epochs: int = EPOCHS, lr: float = LR,
                batch_size: int = BATCH_SIZE,
                device: torch.device | None = None) -> nn.Module:
    """Train with MSE + Adam. X,Y shapes: (N, 1, H, W)."""
    if device is None:
        device = torch.device("cpu")

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    X_tr = torch.from_numpy(X_train).float().to(device)
    Y_tr = torch.from_numpy(Y_train).float().to(device)
    X_v = torch.from_numpy(X_val).float().to(device)
    Y_v = torch.from_numpy(Y_val).float().to(device)

    best_val = float("inf")
    best_state = None

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(X_tr), device=device)
        total_loss, n_b = 0.0, 0
        for i in range(0, len(perm), batch_size):
            idx = perm[i:i + batch_size]
            pred = model(X_tr[idx])
            loss = criterion(pred, Y_tr[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_b += 1

        model.eval()
        with torch.no_grad():
            vl_sum, vl_n = 0.0, 0
            for vi in range(0, len(X_v), batch_size):
                vx = X_v[vi:vi + batch_size]
                vy = Y_v[vi:vi + batch_size]
                vp = model(vx)
                vl_sum += criterion(vp, vy).item() * len(vx)
                vl_n += len(vx)
            val_loss = vl_sum / vl_n

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1:3d}/{epochs}  "
                  f"train={total_loss/n_b:.6f}  val={val_loss:.6f}")

    model.load_state_dict(best_state)
    model = model.to(torch.device("cpu"))
    print(f"    Best val_loss: {best_val:.6f}")

    del X_tr, Y_tr, X_v, Y_v
    gc.collect()
    return model


def evaluate_1step(model: nn.Module, X_test: np.ndarray,
                   Y_test: np.ndarray, batch_size: int = BATCH_SIZE):
    """Returns MSE float."""
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            xb = torch.from_numpy(X_test[i:i+batch_size]).float()
            pb = model(xb).numpy()
            preds.append(pb)
    pred = np.concatenate(preds, axis=0)
    return float(np.mean((Y_test - pred) ** 2))


def make_predict_fn(model: nn.Module, H: int, W: int):
    """Wrap model for single-sample rollout. x: (H, W) -> (H, W)."""
    model.eval()
    def _predict(x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            xt = torch.from_numpy(x).float().reshape(1, 1, H, W)
            out = model(xt).squeeze().numpy().reshape(H, W)
        return np.clip(out, 0, 1)
    return _predict


def multistep_rollout(predict_fn, x0: np.ndarray, n_steps: int,
                      H: int, W: int) -> np.ndarray:
    """Roll out for n_steps. Returns (n_steps, H, W)."""
    preds = np.zeros((n_steps, H, W), dtype=np.float32)
    x = x0.copy()
    for t in range(n_steps):
        raw = predict_fn(x)
        preds[t] = raw
        x = np.clip(raw, 0, 1).astype(np.float32)
    return preds


def rollout_mse(true_grids: np.ndarray, pred_grids: np.ndarray) -> float:
    return float(np.mean((true_grids - pred_grids) ** 2))


# =============================================================================
# Plotting
# =============================================================================

MODEL_COLORS = {
    "ResCor(D)": "tab:red",
    "PureNCA":   "tab:blue",
    "Conv2D":    "tab:green",
    "MLP":       "tab:brown",
}


def plot_rollout(pde_name: str, results: dict, rollout_results: dict,
                 true_future: np.ndarray, rollouts: dict,
                 horizons: list[int]):
    """Save rollout MSE plot + example predictions for a single PDE."""
    plt = _get_plt()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    model_names = list(results.keys())

    # --- Rollout MSE vs horizon ---
    fig1, ax1 = plt.subplots(figsize=(9, 5))
    markers = ["s", "o", "D", "^"]
    for mi, name in enumerate(model_names):
        vals = [rollout_results[h][name] for h in horizons]
        ax1.plot(horizons, vals, f"{markers[mi % 4]}-",
                 color=MODEL_COLORS.get(name, "tab:gray"),
                 label=name, markersize=7)
    ax1.set_xlabel("Rollout horizon (steps)")
    ax1.set_ylabel("MSE (log scale)")
    ax1.set_yscale("log")
    ax1.set_title(f"{pde_name}: Rollout MSE vs Horizon")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    tag = pde_name.lower().replace("-", "_").replace(" ", "_")
    path1 = PLOTS_DIR / f"phase25b_{tag}_rollout.png"
    fig1.savefig(path1, dpi=150)
    plt.close(fig1)

    # --- Example predictions (1D line plots at selected horizons) ---
    show_steps = [h for h in [1, 10, 25] if h <= len(true_future)]
    n_rows = len(show_steps)
    fig2, axes = plt.subplots(n_rows, 1, figsize=(10, 3 * n_rows), sharex=True)
    if n_rows == 1:
        axes = [axes]
    W = true_future.shape[-1]
    x_axis = np.arange(W)
    for ri, step in enumerate(show_steps):
        ax = axes[ri]
        idx = step - 1
        # True is always shape (H, W) where H=1
        true_line = true_future[idx].ravel()
        ax.plot(x_axis, true_line, "k-", linewidth=2, label="True")
        for name in model_names:
            pred_line = rollouts[name][idx].ravel()
            ax.plot(x_axis, pred_line, "--",
                    color=MODEL_COLORS.get(name, "tab:gray"),
                    label=name, linewidth=1.2)
        ax.set_ylabel(f"u (step {step})")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.2)
    axes[-1].set_xlabel("x (grid index)")
    fig2.suptitle(f"{pde_name}: Predictions at Selected Horizons", fontsize=12)
    fig2.tight_layout()
    path2 = PLOTS_DIR / f"phase25b_{tag}_examples.png"
    fig2.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close(fig2)

    print(f"  Plots saved -> {path1.name}, {path2.name}")


# =============================================================================
# Single-PDE experiment runner
# =============================================================================

def run_pde_experiment(pde_name: str, trajs: np.ndarray,
                       grid_H: int, grid_W: int,
                       device: torch.device,
                       traj_len: int):
    """Train 4 models, evaluate, return results + rollout data."""
    print(f"\n{'='*72}")
    print(f"  {pde_name}")
    print(f"{'='*72}")

    n = len(trajs)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    train_t, val_t, test_t = trajs[:n_train], trajs[n_train:n_train+n_val], trajs[n_train+n_val:]

    # Make (input, target) pairs — shape (N, 1, H, W)
    def make_pairs(tr):
        X = tr[:, :-1, :, :]  # (n, T, 1, W)
        Y = tr[:, 1:,  :, :]
        X = X.reshape(-1, 1, grid_H, grid_W)
        Y = Y.reshape(-1, 1, grid_H, grid_W)
        return X, Y

    X_tr, Y_tr = make_pairs(train_t)
    X_v, Y_v = make_pairs(val_t)
    X_te, Y_te = make_pairs(test_t)
    print(f"  Train: {len(X_tr)}  Val: {len(X_v)}  Test: {len(X_te)}")

    results: dict[str, dict] = {}
    predict_fns: dict[str, object] = {}

    # --- 1. ResCor(D) ---
    print(f"\n  [1/4] ResCor(D) ...")
    t0 = time.time()
    rescor = ResidualCorrectionWM(in_channels=1, hidden_ch=16, cml_steps=15)
    rescor = train_model(rescor, X_tr, Y_tr, X_v, Y_v, device=device)
    mse_val = evaluate_1step(rescor, X_te, Y_te)
    pc = rescor.param_count()
    print(f"  MSE: {mse_val:.6f}  Params: {pc}  ({time.time()-t0:.1f}s)")
    results["ResCor(D)"] = {"mse": mse_val, "params": pc["trained"]}
    predict_fns["ResCor(D)"] = make_predict_fn(rescor, grid_H, grid_W)

    # --- 2. PureNCA ---
    print(f"\n  [2/4] PureNCA ...")
    t0 = time.time()
    pnca = PureNCA(in_channels=1, hidden_ch=16, steps=1)
    pnca = train_model(pnca, X_tr, Y_tr, X_v, Y_v, device=device)
    mse_val = evaluate_1step(pnca, X_te, Y_te)
    pc = pnca.param_count()
    print(f"  MSE: {mse_val:.6f}  Params: {pc}  ({time.time()-t0:.1f}s)")
    results["PureNCA"] = {"mse": mse_val, "params": pc["trained"]}
    predict_fns["PureNCA"] = make_predict_fn(pnca, grid_H, grid_W)

    # --- 3. Conv2D ---
    print(f"\n  [3/4] Conv2D ...")
    t0 = time.time()
    cnn = Conv2DBaseline(in_ch=1)
    cnn = train_model(cnn, X_tr, Y_tr, X_v, Y_v, device=device)
    mse_val = evaluate_1step(cnn, X_te, Y_te)
    pc = cnn.param_count()
    print(f"  MSE: {mse_val:.6f}  Params: {pc}  ({time.time()-t0:.1f}s)")
    results["Conv2D"] = {"mse": mse_val, "params": pc["trained"]}
    predict_fns["Conv2D"] = make_predict_fn(cnn, grid_H, grid_W)

    # --- 4. MLP ---
    print(f"\n  [4/4] MLP ...")
    t0 = time.time()
    mlp = MLPBaseline(input_size=grid_H * grid_W, hidden=256)
    mlp = train_model(mlp, X_tr, Y_tr, X_v, Y_v, device=device)
    mse_val = evaluate_1step(mlp, X_te, Y_te)
    pc = mlp.param_count()
    print(f"  MSE: {mse_val:.6f}  Params: {pc}  ({time.time()-t0:.1f}s)")
    results["MLP"] = {"mse": mse_val, "params": pc["trained"]}
    predict_fns["MLP"] = make_predict_fn(mlp, grid_H, grid_W)

    # --- Rollout evaluation ---
    print(f"\n  Multi-step rollout ...")
    test_traj = test_t[0]  # (T+1, 1, W)
    x0 = test_traj[0]       # (1, W)
    max_h = min(max(ROLLOUT_HORIZONS), traj_len)
    true_future = test_traj[1:max_h + 1]  # (max_h, 1, W)

    rollouts = {}
    model_names = list(results.keys())
    for name in model_names:
        rollouts[name] = multistep_rollout(predict_fns[name], x0, max_h,
                                           grid_H, grid_W)

    rollout_results: dict[int, dict] = {}
    valid_horizons = []
    for h in ROLLOUT_HORIZONS:
        if h > max_h:
            break
        valid_horizons.append(h)
        true_h = true_future[:h]
        row = {name: rollout_mse(true_h, rollouts[name][:h]) for name in model_names}
        rollout_results[h] = row
        parts = "  ".join(f"{n}={row[n]:.6f}" for n in model_names)
        print(f"    Horizon {h:>3d}:  {parts}")

    # --- Plots ---
    plot_rollout(pde_name, results, rollout_results, true_future, rollouts,
                 valid_horizons)

    # Cleanup
    del X_tr, Y_tr, X_v, Y_v, X_te, Y_te
    gc.collect()

    return results, rollout_results, valid_horizons


# =============================================================================
# Summary
# =============================================================================

def print_summary(pde_name: str, results: dict, rollout_results: dict,
                  horizons: list[int]):
    model_names = list(results.keys())
    print(f"\n{'='*80}")
    print(f"  {pde_name} SUMMARY")
    print(f"{'='*80}")

    col_w = 14
    header = f"  {'Model':<14s}  {'1-Step MSE':>{col_w}}  {'Params':>{col_w}}"
    print(f"\n{header}")
    print("  " + "-" * (14 + 2 * (col_w + 2) + 4))
    for name in model_names:
        r = results[name]
        print(f"  {name:<14s}  {r['mse']:{col_w}.6f}  {r['params']:{col_w}d}")

    print(f"\n  --- Multi-step Rollout MSE ---")
    hdr = f"  {'Horizon':>8s}"
    for name in model_names:
        hdr += f"  {name:>14s}"
    print(hdr)
    for h in horizons:
        row = f"  {h:8d}"
        for name in model_names:
            row += f"  {rollout_results[h][name]:14.6f}"
        print(row)
    print(f"{'='*80}")


# =============================================================================
# Main
# =============================================================================

def run_experiment(args):
    os.environ.setdefault("FORCE_CPU", "1")
    device = pick_device()

    print("=" * 72)
    print("PHASE 2.5b: HARDER PDEs — Burgers & Kuramoto-Sivashinsky")
    print("=" * 72)

    # ---- PDE 1: Burgers ----
    print("\n[Data] Generating Burgers equation trajectories ...")
    t0 = time.time()
    burgers_trajs = generate_burgers_trajectories()
    print(f"  {BURGERS_N_TRAJ} trajectories x {BURGERS_TLEN} steps, N={BURGERS_N}  "
          f"({time.time()-t0:.1f}s)")
    print(f"  PDE params: nu={BURGERS_NU}, dx={BURGERS_DX:.4f}, "
          f"dt={BURGERS_DT}, sub_steps={BURGERS_SUB}")

    b_res, b_roll, b_h = run_pde_experiment(
        "Burgers Equation", burgers_trajs,
        grid_H=1, grid_W=BURGERS_N,
        device=device, traj_len=BURGERS_TLEN)
    del burgers_trajs
    gc.collect()

    # ---- PDE 2: KS ----
    print("\n[Data] Generating Kuramoto-Sivashinsky trajectories ...")
    t0 = time.time()
    ks_trajs = generate_ks_trajectories()
    print(f"  {KS_N_TRAJ} trajectories x {KS_TLEN} steps, N={KS_N}  "
          f"({time.time()-t0:.1f}s)")
    print(f"  PDE params: L={KS_L}, dx={KS_DX:.4f}, dt={KS_DT}, "
          f"sub_steps={KS_SUB}")

    ks_res, ks_roll, ks_h = run_pde_experiment(
        "Kuramoto-Sivashinsky", ks_trajs,
        grid_H=1, grid_W=KS_N,
        device=device, traj_len=KS_TLEN)
    del ks_trajs
    gc.collect()

    # ---- Final summary ----
    print_summary("Burgers Equation", b_res, b_roll, b_h)
    print_summary("Kuramoto-Sivashinsky", ks_res, ks_roll, ks_h)

    # --- Combined comparison table ---
    print(f"\n{'='*80}")
    print("  COMBINED: ResCor(D) vs Best Baseline")
    print(f"{'='*80}")
    for pde_name, res, roll, horizons in [
        ("Burgers", b_res, b_roll, b_h),
        ("KS", ks_res, ks_roll, ks_h),
    ]:
        rescor_mse = res["ResCor(D)"]["mse"]
        others = {k: v["mse"] for k, v in res.items() if k != "ResCor(D)"}
        best_other_name = min(others, key=others.get)
        best_other_mse = others[best_other_name]
        ratio = best_other_mse / (rescor_mse + 1e-12)
        print(f"  {pde_name:5s} 1-step:  ResCor(D)={rescor_mse:.6f}  "
              f"best-baseline({best_other_name})={best_other_mse:.6f}  "
              f"ratio={ratio:.2f}x")
        if horizons:
            max_h = max(horizons)
            rc_roll = roll[max_h]["ResCor(D)"]
            bo_roll = roll[max_h][best_other_name]
            ratio_r = bo_roll / (rc_roll + 1e-12)
            print(f"  {pde_name:5s} rollout-{max_h}: ResCor(D)={rc_roll:.6f}  "
                  f"best-baseline({best_other_name})={bo_roll:.6f}  "
                  f"ratio={ratio_r:.2f}x")
    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(
        description="Phase 2.5b: Harder PDEs — Burgers & KS")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Skip wandb logging")
    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
