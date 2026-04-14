"""Phase 2.5d: Scale-Up Experiment — 64x64 grids, h=100 rollouts.

Tests whether ResCor(D)'s advantage holds at larger spatial scales.

Benchmarks:
  1. Heat equation  64x64  (up from 16x16)
  2. KS equation    N=128  (up from N=64)  — 1D, H=1, W=128
  3. Game of Life   64x64  (up from 16x16)

Models:
  1. ResCor(D)  — CML base + NCA correction  (our method)
  2. PureNCA    — learned-only NCA
  3. Conv2D     — 3-layer CNN baseline

Usage (GPU pod):
    cd ~/wmca && PYTHONPATH=src python3 experiments/phase25d_scaleup.py --no-wandb
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

# Training
LR = 1e-3
EPOCHS = 50
BATCH_SIZE = 32  # smaller than default due to larger grids

# CML defaults
CML_STEPS = 15
CML_R = 3.90
CML_EPS = 0.3
CML_BETA = 0.15

# =============================================================================
# Benchmark 1: Heat Equation 64x64
# =============================================================================

HEAT_H, HEAT_W = 64, 64
HEAT_ALPHA = 0.1
HEAT_DT = 0.01
HEAT_DX = 1.0 / HEAT_H
_HEAT_COEFF = HEAT_ALPHA * HEAT_DT / (HEAT_DX * HEAT_DX)
_LAP_KERNEL_NP = np.array([[0., 1., 0.],
                            [1., -4., 1.],
                            [0., 1., 0.]], dtype=np.float32)

HEAT_N_TRAJ = 500
HEAT_TRAJ_LEN = 50
HEAT_ROLLOUT_HORIZONS = [1, 5, 10, 25, 50, 100]


def heat_step(u: np.ndarray) -> np.ndarray:
    from scipy.signal import convolve2d
    lap = convolve2d(u, _LAP_KERNEL_NP, mode="same", boundary="fill", fillvalue=0.0)
    u_new = u + _HEAT_COEFF * lap
    return np.clip(u_new, 0.0, 1.0)


def _random_gaussian_blob(h: int, w: int, rng: np.random.RandomState) -> np.ndarray:
    cy, cx = rng.uniform(0.2, 0.8) * h, rng.uniform(0.2, 0.8) * w
    sigma = rng.uniform(2.0, 8.0)  # slightly larger for 64x64
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
    """Returns (N, T+1, H, W)."""
    rng = np.random.RandomState(SEED)
    trajs = np.zeros((HEAT_N_TRAJ, HEAT_TRAJ_LEN + 1, HEAT_H, HEAT_W), dtype=np.float32)
    for i in range(HEAT_N_TRAJ):
        u = generate_heat_ic(HEAT_H, HEAT_W, rng)
        trajs[i, 0] = u
        for t in range(HEAT_TRAJ_LEN):
            u = heat_step(u)
            trajs[i, t + 1] = u
    return trajs


# =============================================================================
# Benchmark 2: Kuramoto-Sivashinsky N=128
# =============================================================================

KS_N = 128
KS_L = 22.0
KS_DX = KS_L / KS_N
KS_DT = 0.05
KS_SUB = 5
KS_N_TRAJ = 300
KS_TLEN = 100
KS_ROLLOUT_HORIZONS = [1, 5, 10, 25, 50, 100]


def ks_step_spectral(u: np.ndarray, L: float, dt: float) -> np.ndarray:
    """Semi-implicit spectral step for KS equation."""
    N = len(u)
    dt_sub = dt / KS_SUB
    k = 2.0 * np.pi * np.fft.fftfreq(N, d=L / N)
    k2 = k ** 2
    k4 = k ** 4
    L_op = -k2 - k4

    for _ in range(KS_SUB):
        u_hat = np.fft.fft(u)
        du_dx = np.real(np.fft.ifft(1j * k * u_hat))
        nl = -u * du_dx
        nl_hat = np.fft.fft(nl)
        u_hat_new = (u_hat + dt_sub * nl_hat) / (1.0 - dt_sub * L_op)
        u = np.real(np.fft.ifft(u_hat_new)).astype(np.float32)

    return u


def generate_ks_ic(N: int, L: float, rng: np.random.RandomState) -> np.ndarray:
    x = np.linspace(0, L, N, endpoint=False).astype(np.float32)
    u = np.zeros(N, dtype=np.float32)
    n_modes = rng.randint(3, 7)
    for _ in range(n_modes):
        kk = rng.randint(1, 6)
        amp = rng.uniform(0.01, 0.1)
        phase = rng.uniform(0, 2 * np.pi)
        u += amp * np.sin(2 * np.pi * kk * x / L + phase)
    return u


def generate_ks_trajectories() -> np.ndarray:
    """Returns (n_traj, traj_len+1, 1, N) — H=1, W=128."""
    rng = np.random.RandomState(SEED)
    N = KS_N
    raw_trajs = []
    for i in range(KS_N_TRAJ):
        u0 = generate_ks_ic(N, KS_L, rng)
        u = u0.copy()
        # Warm-up
        for _ in range(200):
            u = ks_step_spectral(u, KS_L, KS_DT)
        traj = [u.copy()]
        for t in range(KS_TLEN):
            u = ks_step_spectral(u, KS_L, KS_DT)
            traj.append(u.copy())
        raw_trajs.append(np.array(traj, dtype=np.float32))

    # Global normalization to [0, 1]
    all_vals = np.concatenate([t.ravel() for t in raw_trajs])
    g_min, g_max = float(all_vals.min()), float(all_vals.max())
    print(f"  KS raw range: [{g_min:.3f}, {g_max:.3f}]")

    trajs = np.zeros((KS_N_TRAJ, KS_TLEN + 1, 1, N), dtype=np.float32)
    for i, raw in enumerate(raw_trajs):
        normed = (raw - g_min) / (g_max - g_min + 1e-8)
        trajs[i, :, 0, :] = normed

    return trajs


# =============================================================================
# Benchmark 3: Game of Life 64x64
# =============================================================================

GOL_H, GOL_W = 64, 64
GOL_DENSITY = 0.3
GOL_N_TRAJ = 500
GOL_TRAJ_LEN = 20
GOL_ROLLOUT_HORIZONS = [1, 3, 5, 10, 20]


def gol_step(grid: np.ndarray) -> np.ndarray:
    from scipy.signal import convolve2d
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.float32)
    neighbors = convolve2d(grid.astype(np.float32), kernel, mode="same", boundary="wrap")
    born = (grid == 0) & (neighbors == 3)
    survive = (grid == 1) & ((neighbors == 2) | (neighbors == 3))
    return (born | survive).astype(np.float32)


def generate_gol_trajectories() -> np.ndarray:
    """Returns (N, T+1, H, W)."""
    rng = np.random.RandomState(SEED)
    trajs = np.zeros((GOL_N_TRAJ, GOL_TRAJ_LEN + 1, GOL_H, GOL_W), dtype=np.float32)
    for i in range(GOL_N_TRAJ):
        grid = (rng.rand(GOL_H, GOL_W) < GOL_DENSITY).astype(np.float32)
        trajs[i, 0] = grid
        for t in range(GOL_TRAJ_LEN):
            grid = gol_step(grid)
            trajs[i, t + 1] = grid
    return trajs


# =============================================================================
# Models
# =============================================================================

class Conv2DBaseline(nn.Module):
    """3-layer CNN: in_ch -> 16 -> 16 -> in_ch."""

    def __init__(self, in_ch: int = 1, use_sigmoid: bool = False):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, in_ch, 3, padding=1),
        ]
        if use_sigmoid:
            layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def param_count(self) -> dict[str, int]:
        return {"trained": sum(p.numel() for p in self.parameters()), "frozen": 0}


# =============================================================================
# Training & Evaluation
# =============================================================================

def train_model(model: nn.Module,
                X_train: np.ndarray, Y_train: np.ndarray,
                X_val: np.ndarray, Y_val: np.ndarray,
                loss_type: str = "mse",
                epochs: int = EPOCHS, lr: float = LR,
                batch_size: int = BATCH_SIZE,
                device: torch.device | None = None,
                model_name: str = "",
                benchmark_name: str = "") -> tuple[nn.Module, float]:
    """Train with Adam. X,Y shapes: (N, C, H, W) already.

    Returns (model, training_time_seconds).
    """
    if device is None:
        device = torch.device("cpu")

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if loss_type == "mse":
        criterion = nn.MSELoss()
    else:
        criterion = nn.BCELoss()

    X_tr = torch.from_numpy(X_train).float().to(device)
    Y_tr = torch.from_numpy(Y_train).float().to(device)
    X_v = torch.from_numpy(X_val).float().to(device)
    Y_v = torch.from_numpy(Y_val).float().to(device)

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
            vl_sum, vl_n = 0.0, 0
            for vi in range(0, len(X_v), batch_size):
                vx = X_v[vi:vi + batch_size]
                vy = Y_v[vi:vi + batch_size]
                vp = model(vx)
                vl_sum += criterion(vp, vy).item() * len(vx)
                vl_n += len(vx)
            val_loss = vl_sum / vl_n

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0:
            print(f"    [{benchmark_name}|{model_name}] Epoch {epoch+1:3d}/{epochs}  "
                  f"train={total_loss / n_batches:.6f}  val={val_loss:.6f}")

    train_time = time.time() - t0
    model.load_state_dict(best_state)
    model = model.to(device)  # keep on device for eval
    print(f"    Best val_loss: {best_val_loss:.6f}  ({train_time:.1f}s)")

    del X_tr, Y_tr, X_v, Y_v
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return model, train_time


def evaluate_1step(model: nn.Module, X_test: np.ndarray, Y_test: np.ndarray,
                   device: torch.device, batch_size: int = BATCH_SIZE):
    """Returns predictions (N, C, H, W)."""
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            xb = torch.from_numpy(X_test[i:i+batch_size]).float().to(device)
            pb = model(xb).cpu().numpy()
            preds.append(pb)
    return np.concatenate(preds, axis=0)


def make_predict_fn(model: nn.Module, H: int, W: int, device: torch.device):
    """Wrap model for single-sample rollout. x: (H, W) -> (H, W)."""
    model.eval()
    def _predict(x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            xt = torch.from_numpy(x).float().reshape(1, 1, H, W).to(device)
            out = model(xt).cpu().squeeze().numpy().reshape(H, W)
        return np.clip(out, 0, 1)
    return _predict


def multistep_rollout(predict_fn, x0: np.ndarray, n_steps: int,
                      H: int, W: int,
                      binarize: bool = False) -> np.ndarray:
    """Roll out for n_steps. Returns (n_steps, H, W)."""
    preds = np.zeros((n_steps, H, W), dtype=np.float32)
    x = x0.copy()
    for t in range(n_steps):
        raw = predict_fn(x)
        preds[t] = raw
        if binarize:
            x = (np.clip(raw, 0, 1) >= 0.5).astype(np.float32)
        else:
            x = np.clip(raw, 0, 1).astype(np.float32)
    return preds


def mse_metric(Y_true: np.ndarray, Y_pred: np.ndarray) -> float:
    return float(np.mean((Y_true - Y_pred) ** 2))


def cell_accuracy(Y_true: np.ndarray, Y_pred: np.ndarray) -> float:
    pred_binary = (Y_pred >= 0.5).astype(np.float32)
    return float(np.mean(pred_binary == Y_true))


# =============================================================================
# Per-benchmark runner
# =============================================================================

def split_trajectories(trajs: np.ndarray):
    n = len(trajs)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    return trajs[:n_train], trajs[n_train:n_train + n_val], trajs[n_train + n_val:]


def run_2d_benchmark(benchmark_name: str, trajs: np.ndarray,
                     grid_H: int, grid_W: int,
                     rollout_horizons: list[int],
                     loss_type: str,
                     binarize_rollout: bool,
                     device: torch.device):
    """Run 3 models on a 2D benchmark (Heat or GoL).

    trajs shape: (N, T+1, H, W) — no channel dim yet.
    """
    print(f"\n{'='*72}")
    print(f"  BENCHMARK: {benchmark_name}  ({grid_H}x{grid_W})")
    print(f"{'='*72}")

    train_t, val_t, test_t = split_trajectories(trajs)

    # Make pairs: (N*T, 1, H, W)
    def make_pairs(tr):
        X = tr[:, :-1].reshape(-1, grid_H, grid_W)
        Y = tr[:, 1:].reshape(-1, grid_H, grid_W)
        return X[:, np.newaxis, :, :], Y[:, np.newaxis, :, :]

    X_tr, Y_tr = make_pairs(train_t)
    X_v, Y_v = make_pairs(val_t)
    X_te, Y_te = make_pairs(test_t)
    print(f"  Train: {len(X_tr)}, Val: {len(X_v)}, Test: {len(X_te)}")

    use_sigmoid = (loss_type == "bce")

    # Build 3 models
    models_spec = {}

    # ResCor(D)
    models_spec["ResCor(D)"] = ResidualCorrectionWM(
        in_channels=1, hidden_ch=16, cml_steps=CML_STEPS,
        r=CML_R, eps=CML_EPS, beta=CML_BETA, seed=SEED)

    # PureNCA
    models_spec["PureNCA"] = PureNCA(in_channels=1, hidden_ch=16, steps=1)

    # Conv2D
    models_spec["Conv2D"] = Conv2DBaseline(in_ch=1, use_sigmoid=use_sigmoid)

    results = {}
    predict_fns = {}

    for idx, (name, model) in enumerate(models_spec.items()):
        print(f"\n  [{idx+1}/{len(models_spec)}] {name} ...")
        model, train_time = train_model(
            model, X_tr, Y_tr, X_v, Y_v,
            loss_type=loss_type, device=device,
            model_name=name, benchmark_name=benchmark_name)

        preds = evaluate_1step(model, X_te, Y_te, device)

        if loss_type == "mse":
            metric_val = mse_metric(Y_te, preds)
            metric_name = "MSE"
        else:
            metric_val = cell_accuracy(Y_te, preds)
            metric_name = "CellAcc"

        pc = model.param_count()
        print(f"    1-step {metric_name}: {metric_val:.6f}")
        print(f"    Params: trained={pc['trained']}, frozen={pc['frozen']}")

        results[name] = {
            "one_step": metric_val,
            "params_trained": pc["trained"],
            "params_frozen": pc["frozen"],
            "train_time": train_time,
        }
        predict_fns[name] = make_predict_fn(model, grid_H, grid_W, device)

        # Free GPU memory after building predict_fn
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    # Multi-step rollout
    print(f"\n  --- Multi-step rollout (horizons {rollout_horizons}) ---")
    n_rollout = min(20, len(test_t))
    max_h = max(rollout_horizons)

    rollout_results: dict[int, dict] = {h: {} for h in rollout_horizons}

    for name, pfn in predict_fns.items():
        for h in rollout_horizons:
            if h > test_t.shape[1] - 1:
                rollout_results[h][name] = float("nan")
                continue
            scores = []
            for ti in range(n_rollout):
                x0 = test_t[ti, 0]
                true_future = test_t[ti, 1:h + 1]
                if len(true_future) < h:
                    continue
                pred_future = multistep_rollout(pfn, x0, h, grid_H, grid_W,
                                                binarize=binarize_rollout)
                if loss_type == "mse":
                    scores.append(mse_metric(true_future, pred_future))
                else:
                    pred_bin = (pred_future >= 0.5).astype(np.float32)
                    scores.append(float(np.mean(pred_bin == true_future)))
            rollout_results[h][name] = float(np.mean(scores)) if scores else float("nan")

        vals = [f"h={h}: {rollout_results[h][name]:.6f}" for h in rollout_horizons]
        print(f"    {name}: {', '.join(vals)}")

    del X_tr, Y_tr, X_v, Y_v, X_te, Y_te
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    return results, rollout_results


def run_1d_benchmark(benchmark_name: str, trajs: np.ndarray,
                     grid_H: int, grid_W: int,
                     rollout_horizons: list[int],
                     device: torch.device,
                     traj_len: int):
    """Run 3 models on a 1D PDE benchmark.

    trajs shape: (N, T+1, 1, W) — already has channel dim.
    """
    print(f"\n{'='*72}")
    print(f"  BENCHMARK: {benchmark_name}  (H={grid_H}, W={grid_W})")
    print(f"{'='*72}")

    train_t, val_t, test_t = split_trajectories(trajs)

    # Make pairs: (N*T, 1, H, W)
    def make_pairs(tr):
        X = tr[:, :-1, :, :]
        Y = tr[:, 1:, :, :]
        X = X.reshape(-1, 1, grid_H, grid_W)
        Y = Y.reshape(-1, 1, grid_H, grid_W)
        return X, Y

    X_tr, Y_tr = make_pairs(train_t)
    X_v, Y_v = make_pairs(val_t)
    X_te, Y_te = make_pairs(test_t)
    print(f"  Train: {len(X_tr)}, Val: {len(X_v)}, Test: {len(X_te)}")

    models_spec = {}
    models_spec["ResCor(D)"] = ResidualCorrectionWM(
        in_channels=1, hidden_ch=16, cml_steps=CML_STEPS,
        r=CML_R, eps=CML_EPS, beta=CML_BETA, seed=SEED)
    models_spec["PureNCA"] = PureNCA(in_channels=1, hidden_ch=16, steps=1)
    models_spec["Conv2D"] = Conv2DBaseline(in_ch=1)

    results = {}
    predict_fns = {}

    for idx, (name, model) in enumerate(models_spec.items()):
        print(f"\n  [{idx+1}/{len(models_spec)}] {name} ...")
        model, train_time = train_model(
            model, X_tr, Y_tr, X_v, Y_v,
            loss_type="mse", device=device,
            model_name=name, benchmark_name=benchmark_name)

        preds = evaluate_1step(model, X_te, Y_te, device)
        metric_val = mse_metric(Y_te, preds)

        pc = model.param_count()
        print(f"    1-step MSE: {metric_val:.6f}")
        print(f"    Params: trained={pc['trained']}, frozen={pc['frozen']}")

        results[name] = {
            "one_step": metric_val,
            "params_trained": pc["trained"],
            "params_frozen": pc["frozen"],
            "train_time": train_time,
        }
        predict_fns[name] = make_predict_fn(model, grid_H, grid_W, device)

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    # Multi-step rollout
    print(f"\n  --- Multi-step rollout (horizons {rollout_horizons}) ---")
    n_rollout = min(20, len(test_t))
    max_h = min(max(rollout_horizons), traj_len)

    rollout_results: dict[int, dict] = {h: {} for h in rollout_horizons}

    for name, pfn in predict_fns.items():
        for h in rollout_horizons:
            if h > traj_len:
                rollout_results[h][name] = float("nan")
                continue
            scores = []
            for ti in range(n_rollout):
                x0 = test_t[ti, 0, :, :]  # (1, W) -> (H, W)
                true_future = test_t[ti, 1:h + 1, :, :]  # (h, 1, W)
                if len(true_future) < h:
                    continue
                pred_future = multistep_rollout(pfn, x0, h, grid_H, grid_W)
                # true_future: (h, 1, W) -> squeeze channel
                true_sq = true_future[:, 0, :]  # (h, W) = (h, H=1*W) same shape
                # pred_future is (h, H, W) = (h, 1, W) from rollout
                scores.append(mse_metric(true_sq, pred_future))
            rollout_results[h][name] = float(np.mean(scores)) if scores else float("nan")

        vals = [f"h={h}: {rollout_results[h][name]:.6f}" for h in rollout_horizons]
        print(f"    {name}: {', '.join(vals)}")

    del X_tr, Y_tr, X_v, Y_v, X_te, Y_te
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    return results, rollout_results


# =============================================================================
# Plotting
# =============================================================================

MODEL_COLORS = {
    "ResCor(D)": "tab:red",
    "PureNCA":   "tab:blue",
    "Conv2D":    "tab:green",
}

MODEL_MARKERS = {
    "ResCor(D)": "s",
    "PureNCA":   "o",
    "Conv2D":    "D",
}


def make_benchmark_plot(benchmark_name: str, rollout_results: dict,
                        metric_label: str, log_scale: bool,
                        filename_prefix: str = "phase25d"):
    plt = _get_plt()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    horizons = sorted(rollout_results.keys())
    model_names = list(rollout_results[horizons[0]].keys())

    fig, ax = plt.subplots(figsize=(9, 5))
    for name in model_names:
        vals = [rollout_results[h][name] for h in horizons]
        ax.plot(horizons, vals,
                f"{MODEL_MARKERS.get(name, 'o')}-",
                color=MODEL_COLORS.get(name, "tab:gray"),
                label=name, markersize=7, linewidth=2)
    ax.set_xlabel("Rollout horizon (steps)")
    ax.set_ylabel(metric_label)
    if log_scale:
        ax.set_yscale("log")
    ax.set_title(f"Phase 2.5d Scale-Up: {benchmark_name}")
    ax.legend(fontsize=10, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    safe_name = benchmark_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    path = PLOTS_DIR / f"{filename_prefix}_{safe_name}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved plot: {path}")


# =============================================================================
# Summary tables
# =============================================================================

def print_benchmark_table(benchmark_name: str, results: dict,
                          rollout_results: dict, metric_name: str,
                          rollout_horizons: list[int]):
    print(f"\n{'='*90}")
    print(f"  RESULTS: {benchmark_name}")
    print(f"{'='*90}")

    names = list(results.keys())
    horizons = [h for h in rollout_horizons if h in rollout_results]

    h_cols = "".join(f"  h={h:<5}" for h in horizons)
    print(f"  {'Model':<14} {'1-step':>10}  {'Trained':>8}  {'Frozen':>7}  {'Time':>7}{h_cols}")
    print(f"  {'-'*14} {'-'*10}  {'-'*8}  {'-'*7}  {'-'*7}{'  -------' * len(horizons)}")

    for name in names:
        r = results[name]
        one = f"{r['one_step']:.6f}"
        trained = f"{r['params_trained']:>8d}"
        frozen = f"{r['params_frozen']:>7d}"
        tt = f"{r['train_time']:>6.1f}s"
        rollout_vals = ""
        for h in horizons:
            v = rollout_results[h].get(name, float("nan"))
            rollout_vals += f"  {v:.5f}"
        print(f"  {name:<14} {one:>10}  {trained}  {frozen}  {tt}{rollout_vals}")


def print_advantage_table(all_results: dict):
    """Print ResCor(D) advantage ratio vs best baseline at each horizon."""
    print(f"\n{'='*90}")
    print(f"  ResCor(D) ADVANTAGE RATIO vs BEST BASELINE")
    print(f"{'='*90}")

    for bname, (results, rollout_results, horizons, metric_type) in all_results.items():
        rescor_1step = results["ResCor(D)"]["one_step"]
        baselines = {k: v["one_step"] for k, v in results.items() if k != "ResCor(D)"}
        if metric_type == "mse":
            best_bl_name = min(baselines, key=baselines.get)
        else:
            best_bl_name = max(baselines, key=baselines.get)
        best_bl_1step = baselines[best_bl_name]

        if metric_type == "mse":
            ratio_1step = best_bl_1step / (rescor_1step + 1e-12)
        else:
            ratio_1step = rescor_1step / (best_bl_1step + 1e-12)

        print(f"\n  {bname}:")
        print(f"    1-step: ResCor(D)={rescor_1step:.6f}  "
              f"best-baseline({best_bl_name})={best_bl_1step:.6f}  "
              f"ratio={ratio_1step:.2f}x")

        for h in horizons:
            rc = rollout_results[h]["ResCor(D)"]
            bl = rollout_results[h][best_bl_name]
            if metric_type == "mse":
                r = bl / (rc + 1e-12)
            else:
                r = rc / (bl + 1e-12)
            print(f"    h={h:>3d}: ResCor(D)={rc:.6f}  "
                  f"{best_bl_name}={bl:.6f}  ratio={r:.2f}x")


# =============================================================================
# Main
# =============================================================================

def run_experiment(args):
    device = pick_device()

    print("=" * 72)
    print("PHASE 2.5d: SCALE-UP EXPERIMENT")
    print("  Does ResCor(D) advantage hold at larger spatial scales?")
    print("=" * 72)
    print(f"  Device: {device}")
    print(f"  Epochs: {EPOCHS}, LR: {LR}, Batch: {BATCH_SIZE}")
    print(f"  Heat: {HEAT_H}x{HEAT_W}, {HEAT_N_TRAJ} traj x {HEAT_TRAJ_LEN} steps")
    print(f"  KS:   N={KS_N}, {KS_N_TRAJ} traj x {KS_TLEN} steps")
    print(f"  GoL:  {GOL_H}x{GOL_W}, {GOL_N_TRAJ} traj x {GOL_TRAJ_LEN} steps")

    all_results = {}

    # ---- Benchmark 1: Heat Equation 64x64 ----
    print(f"\n[1/3] Generating Heat equation trajectories ({HEAT_H}x{HEAT_W}) ...")
    t0 = time.time()
    heat_trajs = generate_heat_trajectories()
    print(f"  Generated {HEAT_N_TRAJ} trajectories x {HEAT_TRAJ_LEN} steps ({time.time()-t0:.1f}s)")

    heat_res, heat_roll = run_2d_benchmark(
        "Heat 64x64", heat_trajs, HEAT_H, HEAT_W,
        HEAT_ROLLOUT_HORIZONS, loss_type="mse",
        binarize_rollout=False, device=device)
    all_results["Heat 64x64"] = (heat_res, heat_roll, HEAT_ROLLOUT_HORIZONS, "mse")

    del heat_trajs
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    # ---- Benchmark 2: KS N=128 ----
    print(f"\n[2/3] Generating Kuramoto-Sivashinsky trajectories (N={KS_N}) ...")
    t0 = time.time()
    ks_trajs = generate_ks_trajectories()
    print(f"  Generated {KS_N_TRAJ} trajectories x {KS_TLEN} steps ({time.time()-t0:.1f}s)")

    ks_res, ks_roll = run_1d_benchmark(
        "KS N=128", ks_trajs, grid_H=1, grid_W=KS_N,
        rollout_horizons=KS_ROLLOUT_HORIZONS,
        device=device, traj_len=KS_TLEN)
    all_results["KS N=128"] = (ks_res, ks_roll, KS_ROLLOUT_HORIZONS, "mse")

    del ks_trajs
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    # ---- Benchmark 3: Game of Life 64x64 ----
    print(f"\n[3/3] Generating Game of Life trajectories ({GOL_H}x{GOL_W}) ...")
    t0 = time.time()
    gol_trajs = generate_gol_trajectories()
    print(f"  Generated {GOL_N_TRAJ} trajectories x {GOL_TRAJ_LEN} steps ({time.time()-t0:.1f}s)")

    gol_res, gol_roll = run_2d_benchmark(
        "GoL 64x64", gol_trajs, GOL_H, GOL_W,
        GOL_ROLLOUT_HORIZONS, loss_type="bce",
        binarize_rollout=True, device=device)
    all_results["GoL 64x64"] = (gol_res, gol_roll, GOL_ROLLOUT_HORIZONS, "bce")

    del gol_trajs
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    # ---- Summary Tables ----
    print_benchmark_table("Heat 64x64", heat_res, heat_roll, "MSE",
                          HEAT_ROLLOUT_HORIZONS)
    print_benchmark_table("KS N=128", ks_res, ks_roll, "MSE",
                          KS_ROLLOUT_HORIZONS)
    print_benchmark_table("GoL 64x64", gol_res, gol_roll, "CellAcc",
                          GOL_ROLLOUT_HORIZONS)

    # ---- Advantage Table ----
    print_advantage_table(all_results)

    # ---- Plots ----
    make_benchmark_plot("Heat 64x64", heat_roll,
                        "MSE (log scale)", log_scale=True)
    make_benchmark_plot("KS N=128", ks_roll,
                        "MSE (log scale)", log_scale=True)
    make_benchmark_plot("GoL 64x64", gol_roll,
                        "Cell Accuracy", log_scale=False)

    print("\n" + "=" * 72)
    print("  PHASE 2.5d COMPLETE")
    print("=" * 72)


def main():
    parser = argparse.ArgumentParser(
        description="Phase 2.5d: Scale-Up — 64x64 grids, h=100 rollouts")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Skip wandb logging")
    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
