"""Phase 2 Ablation: All 4 architecture variants + baselines on Heat & GoL.

Tests 7 models on 2 grid benchmarks (16x16) to compare hybrid CML+NCA
architectures. Quick ablation for fast iteration.

Models:
  1. PureNCA         — learned NCA, no CML
  2. CML2D(Ridge)    — fixed CML reservoir + Ridge readout (sklearn)
  3. GatedBlendWM    — Variant A: gated blend of CML + NCA
  4. CMLRegularizedNCA — Variant B: NCA with CML regularization
  5. NCAInsideCML    — Variant C: NCA replaces logistic map inside CML
  6. ResidualCorrectionWM — Variant D: CML base + NCA correction
  7. Conv2D          — neural baseline

Benchmarks:
  1. Heat equation (16x16, continuous, MSE)
  2. Game of Life  (16x16, binary, BCE / cell accuracy)

Usage:
    FORCE_CPU=1 uv run --with scikit-learn,matplotlib,scipy python experiments/phase2_ablation.py --no-wandb
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

GRID_H, GRID_W = 16, 16
GRID_SIZE = GRID_H * GRID_W

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

# Dataset sizes (smaller for 16x16 fast ablation)
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

# Ridge
RIDGE_ALPHA = 1.0

# Rollout horizons
ROLLOUT_HORIZONS = [1, 3, 5, 10]

# Variant B regularization
REG_LAMBDA = 0.1


# ===== Data Generation =====================================================

def heat_step(u: np.ndarray) -> np.ndarray:
    """Single heat equation step with finite differences + zero BCs."""
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
    """Returns (N, T+1, H, W)."""
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
    """Returns (N, T+1, H, W)."""
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


# ===== Conv2D Baseline =====================================================

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


# ===== CML2D Ridge Baseline ================================================

class CML2DRidge:
    """CML2D (frozen) + sklearn Ridge readout."""

    def __init__(self):
        self.cml = CML2D(in_channels=1, steps=CML_STEPS, r=CML_R,
                         eps=CML_EPS, beta=CML_BETA, seed=SEED)
        self.cml.eval()
        self.ridge = None

    def _features(self, X: np.ndarray) -> np.ndarray:
        """X: (N, H, W) -> (N, H*W) CML features."""
        X_t = torch.from_numpy(X).float().unsqueeze(1)
        with torch.no_grad():
            out = self.cml(X_t)
        return out.squeeze(1).reshape(len(X), -1).numpy()

    def fit(self, X_train: np.ndarray, Y_train: np.ndarray):
        Ridge = _get_ridge()
        feats = self._features(X_train)
        Y_flat = Y_train.reshape(len(Y_train), -1)
        self.ridge = Ridge(alpha=RIDGE_ALPHA)
        self.ridge.fit(feats, Y_flat)

    def predict(self, X: np.ndarray) -> np.ndarray:
        feats = self._features(X)
        pred_flat = self.ridge.predict(feats)
        return pred_flat.reshape(-1, GRID_H, GRID_W).clip(0, 1)

    def predict_one(self, x: np.ndarray) -> np.ndarray:
        return self.predict(x[np.newaxis])[0]

    def param_count(self) -> dict[str, int]:
        if self.ridge is None:
            return {"trained": 0, "frozen": 0}
        ridge_params = self.ridge.coef_.size + self.ridge.intercept_.size
        return {"trained": ridge_params, "frozen": sum(b.numel() for b in self.cml.buffers())}

    def training_time(self):
        return 0.0  # Ridge is instant relative to NN training


# ===== Training Utilities ===================================================

def train_model(model: nn.Module, X_train: np.ndarray, Y_train: np.ndarray,
                X_val: np.ndarray, Y_val: np.ndarray,
                loss_type: str = "mse",  # "mse" or "bce"
                epochs: int = EPOCHS, lr: float = LR,
                batch_size: int = BATCH_SIZE,
                model_name: str = "",
                benchmark_name: str = "",
                is_variant_b: bool = False) -> tuple[nn.Module, float]:
    """Train a spatial model (B,1,H,W)->(B,1,H,W).

    Returns (model, training_time_seconds).
    """
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

            if is_variant_b:
                # CMLRegularizedNCA returns (nca_out, cml_ref) during training
                nca_out, cml_ref = model(xb)
                pred_loss = criterion(nca_out, yb)
                reg_loss = F.mse_loss(nca_out, cml_ref.detach())
                loss = pred_loss + REG_LAMBDA * reg_loss
            else:
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
                if is_variant_b:
                    # In eval mode, CMLRegularizedNCA returns just nca_out
                    vp = model(vx)
                else:
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
    """Return predictions (N, H, W)."""
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X_test), BATCH_SIZE):
            xb = torch.from_numpy(X_test[i:i+BATCH_SIZE]).float().unsqueeze(1)
            pb = model(xb).squeeze(1).numpy()
            preds.append(pb)
    return np.concatenate(preds, axis=0)


def make_predict_fn(model: nn.Module):
    """Wrap spatial model for rollout: (H,W) -> (H,W)."""
    model.eval()
    def _predict(x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x_t = torch.from_numpy(x).float().reshape(1, 1, GRID_H, GRID_W)
            out = model(x_t).squeeze().numpy()
        return np.clip(out, 0, 1)
    return _predict


def make_predict_fn_ridge(ridge_model: CML2DRidge):
    def _predict(x: np.ndarray) -> np.ndarray:
        return ridge_model.predict_one(x)
    return _predict


def multistep_rollout(predict_fn, x0: np.ndarray, n_steps: int,
                      binarize: bool = False) -> np.ndarray:
    """Roll out predict_fn for n_steps.

    If binarize=True (GoL), feedback is thresholded at 0.5.
    Returns (n_steps, H, W) continuous predictions.
    """
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


# ===== Model Factory ========================================================

def build_models(use_sigmoid: bool = False):
    """Build all 7 models. use_sigmoid=True for GoL (BCE needs [0,1] output)."""
    models = {}

    # 1. PureNCA
    m = PureNCA(in_channels=1, hidden_ch=16, steps=1)
    models["PureNCA"] = {"model": m, "type": "nn"}

    # 2. CML2D Ridge (handled separately)
    models["CML2D(Ridge)"] = {"model": CML2DRidge(), "type": "ridge"}

    # 3. GatedBlendWM (Variant A)
    m = GatedBlendWM(in_channels=1, hidden_ch=16, cml_steps=CML_STEPS,
                     nca_steps=1, r=CML_R, eps=CML_EPS, beta=CML_BETA, seed=SEED)
    models["GatedBlend(A)"] = {"model": m, "type": "nn"}

    # 4. CMLRegularizedNCA (Variant B)
    m = CMLRegularizedNCA(in_channels=1, hidden_ch=16, nca_steps=1,
                          r=CML_R, eps=CML_EPS, beta=CML_BETA, seed=SEED)
    models["CMLReg(B)"] = {"model": m, "type": "nn_variant_b"}

    # 5. NCAInsideCML (Variant C)
    m = NCAInsideCML(in_channels=1, hidden_ch=16, steps=5,
                     eps=CML_EPS, beta=CML_BETA, seed=SEED)
    models["NCAInCML(C)"] = {"model": m, "type": "nn"}

    # 6. ResidualCorrectionWM (Variant D)
    m = ResidualCorrectionWM(in_channels=1, hidden_ch=16, cml_steps=CML_STEPS,
                             r=CML_R, eps=CML_EPS, beta=CML_BETA, seed=SEED)
    models["ResCor(D)"] = {"model": m, "type": "nn"}

    # 7. Conv2D baseline
    m = Conv2DBaseline(use_sigmoid=use_sigmoid)
    models["Conv2D"] = {"model": m, "type": "nn"}

    return models


# ===== Per-Benchmark Runner =================================================

def run_benchmark(benchmark_name: str, trajs: np.ndarray,
                  loss_type: str, binarize_rollout: bool):
    """Run all models on one benchmark.

    Returns: results dict with metrics per model.
    """
    print(f"\n{'=' * 72}")
    print(f"  BENCHMARK: {benchmark_name}")
    print(f"{'=' * 72}")

    # Split data
    train_t, val_t, test_t = split_trajectories(trajs)
    X_train, Y_train = make_pairs(train_t)
    X_val, Y_val = make_pairs(val_t)
    X_test, Y_test = make_pairs(test_t)
    print(f"  Train pairs: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    use_sigmoid = (loss_type == "bce")
    models = build_models(use_sigmoid=use_sigmoid)

    results = {}
    predict_fns = {}

    for name, spec in models.items():
        print(f"\n  --- {name} ---")

        if spec["type"] == "ridge":
            # CML2D + Ridge
            ridge_model: CML2DRidge = spec["model"]
            t0 = time.time()
            ridge_model.fit(X_train, Y_train)
            train_time = time.time() - t0
            preds = ridge_model.predict(X_test)
            pc = ridge_model.param_count()
            predict_fns[name] = make_predict_fn_ridge(ridge_model)

        elif spec["type"] == "nn_variant_b":
            # CMLRegularizedNCA — special training
            model = spec["model"]
            model, train_time = train_model(
                model, X_train, Y_train, X_val, Y_val,
                loss_type=loss_type, model_name=name,
                benchmark_name=benchmark_name, is_variant_b=True,
            )
            preds = evaluate_spatial(model, X_test, Y_test)
            pc = model.param_count()
            predict_fns[name] = make_predict_fn(model)

        else:
            # Standard NN training
            model = spec["model"]
            model, train_time = train_model(
                model, X_train, Y_train, X_val, Y_val,
                loss_type=loss_type, model_name=name,
                benchmark_name=benchmark_name,
            )
            preds = evaluate_spatial(model, X_test, Y_test)
            pc = model.param_count()
            predict_fns[name] = make_predict_fn(model)

        # 1-step metrics
        if loss_type == "mse":
            one_step = mse_metric(Y_test, preds)
            metric_name = "MSE"
        else:
            one_step = cell_accuracy(Y_test, preds)
            metric_name = "CellAcc"

        print(f"    1-step {metric_name}: {one_step:.6f}")
        print(f"    Params: trained={pc['trained']}, frozen={pc['frozen']}")
        print(f"    Train time: {train_time:.1f}s")

        results[name] = {
            "one_step": one_step,
            "params_trained": pc["trained"],
            "params_frozen": pc["frozen"],
            "train_time": train_time,
        }

    # Multi-step rollout
    print(f"\n  --- Multi-step rollout (horizons {ROLLOUT_HORIZONS}) ---")
    # Use first 20 test trajectories for rollout
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

        # Print rollout for this model
        vals = [f"h={h}: {rollout_results[h][name]:.4f}" for h in ROLLOUT_HORIZONS]
        print(f"    {name}: {', '.join(vals)}")

    return results, rollout_results, loss_type


# ===== Plotting =============================================================

MODEL_COLORS = {
    "PureNCA":      "tab:blue",
    "CML2D(Ridge)": "tab:red",
    "GatedBlend(A)": "tab:green",
    "CMLReg(B)":    "tab:orange",
    "NCAInCML(C)":  "tab:purple",
    "ResCor(D)":    "tab:pink",
    "Conv2D":       "tab:brown",
}


def make_benchmark_plot(benchmark_name: str, rollout_results: dict,
                        metric_label: str, log_scale: bool):
    plt = _get_plt()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    horizons = sorted(rollout_results.keys())
    model_names = list(rollout_results[horizons[0]].keys())
    markers = ["o", "s", "^", "D", "P", "v", "X"]

    fig, ax = plt.subplots(figsize=(9, 5))
    for mi, name in enumerate(model_names):
        vals = [rollout_results[h][name] for h in horizons]
        ax.plot(horizons, vals, f"{markers[mi % len(markers)]}-",
                color=MODEL_COLORS.get(name, "tab:gray"),
                label=name, markersize=7, linewidth=1.5)
    ax.set_xlabel("Rollout horizon (steps)")
    ax.set_ylabel(metric_label)
    if log_scale:
        ax.set_yscale("log")
    ax.set_title(f"Phase 2 Ablation: {benchmark_name}")
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    safe_name = benchmark_name.lower().replace(" ", "_")
    path = PLOTS_DIR / f"phase2_{safe_name}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved plot: {path}")


# ===== Summary Tables =======================================================

def print_benchmark_table(benchmark_name: str, results: dict,
                          rollout_results: dict, metric_name: str):
    print(f"\n{'=' * 80}")
    print(f"  RESULTS: {benchmark_name}")
    print(f"{'=' * 80}")

    names = list(results.keys())
    horizons = sorted(rollout_results.keys())

    # Header
    h_cols = "".join(f"  h={h:<4}" for h in horizons)
    print(f"  {'Model':<18} {'1-step':>8}  {'Trained':>8}  {'Frozen':>7}  {'Time':>6}{h_cols}")
    print(f"  {'-' * 18} {'-' * 8}  {'-' * 8}  {'-' * 7}  {'-' * 6}{'  ------' * len(horizons)}")

    for name in names:
        r = results[name]
        one = f"{r['one_step']:.4f}"
        trained = f"{r['params_trained']:>8d}"
        frozen = f"{r['params_frozen']:>7d}"
        tt = f"{r['train_time']:>5.1f}s"
        rollout_vals = "".join(f"  {rollout_results[h][name]:.4f}" for h in horizons)
        print(f"  {name:<18} {one:>8}  {trained}  {frozen}  {tt}{rollout_vals}")


def print_combined_ranking(all_results: dict):
    """Rank models across both benchmarks."""
    print(f"\n{'=' * 80}")
    print(f"  COMBINED RANKING ACROSS BENCHMARKS")
    print(f"{'=' * 80}")

    # Collect model names (same for both)
    model_names = None
    for bname, (results, rollout_results, loss_type) in all_results.items():
        if model_names is None:
            model_names = list(results.keys())

    # For each benchmark, rank by 1-step metric (lower MSE = better, higher acc = better)
    rankings = {name: 0.0 for name in model_names}
    for bname, (results, rollout_results, loss_type) in all_results.items():
        if loss_type == "mse":
            # Lower is better -> rank ascending
            sorted_names = sorted(model_names, key=lambda n: results[n]["one_step"])
        else:
            # Higher is better -> rank descending
            sorted_names = sorted(model_names, key=lambda n: -results[n]["one_step"])

        for rank, name in enumerate(sorted_names):
            rankings[name] += rank + 1  # 1-indexed rank

    # Also rank by average rollout performance
    rollout_rankings = {name: 0.0 for name in model_names}
    for bname, (results, rollout_results, loss_type) in all_results.items():
        horizons = sorted(rollout_results.keys())
        for name in model_names:
            avg_rollout = np.mean([rollout_results[h][name] for h in horizons])
            if loss_type == "mse":
                rollout_rankings[name] += avg_rollout  # lower is better
            else:
                rollout_rankings[name] += -avg_rollout  # higher acc is better, negate so lower = better

    # Combined score: sum of 1-step ranks + normalized rollout rank
    sorted_by_rollout = sorted(model_names, key=lambda n: rollout_rankings[n])
    for rank, name in enumerate(sorted_by_rollout):
        rankings[name] += rank + 1

    final_sorted = sorted(model_names, key=lambda n: rankings[n])

    print(f"\n  {'Rank':<6} {'Model':<18} {'Score':>8}  Details")
    print(f"  {'-' * 6} {'-' * 18} {'-' * 8}  {'-' * 40}")
    for rank, name in enumerate(final_sorted, 1):
        details = []
        for bname, (results, rollout_results, loss_type) in all_results.items():
            short = bname[:4]
            val = results[name]["one_step"]
            details.append(f"{short}={val:.4f}")
        detail_str = ", ".join(details)
        print(f"  {rank:<6} {name:<18} {rankings[name]:>8.1f}  {detail_str}")


# ===== Main =================================================================

def run_experiment(args):
    os.environ.setdefault("FORCE_CPU", "1")
    device = pick_device()

    print("=" * 72)
    print("PHASE 2 ABLATION: All Variants x Grid Benchmarks (16x16)")
    print("=" * 72)
    print(f"  Grid: {GRID_H}x{GRID_W}")
    print(f"  Epochs: {EPOCHS}, LR: {LR}, Batch: {BATCH_SIZE}")
    print(f"  Rollout horizons: {ROLLOUT_HORIZONS}")
    print(f"  Reg lambda (Variant B): {REG_LAMBDA}")
    print(f"  Device: {device}")

    all_results = {}

    # ---- Benchmark 1: Heat Equation ----
    print("\n[1/2] Generating heat equation trajectories ...")
    t0 = time.time()
    heat_trajs = generate_heat_trajectories()
    print(f"  Generated {HEAT_N_TRAJ} trajectories x {HEAT_TRAJ_LEN} steps ({time.time()-t0:.1f}s)")

    results_h, rollout_h, lt_h = run_benchmark(
        "Heat Equation", heat_trajs, loss_type="mse", binarize_rollout=False)
    all_results["Heat Equation"] = (results_h, rollout_h, lt_h)

    del heat_trajs
    gc.collect()

    # ---- Benchmark 2: Game of Life ----
    print("\n[2/2] Generating Game of Life trajectories ...")
    t0 = time.time()
    gol_trajs = generate_gol_trajectories()
    print(f"  Generated {GOL_N_TRAJ} trajectories x {GOL_TRAJ_LEN} steps ({time.time()-t0:.1f}s)")

    results_g, rollout_g, lt_g = run_benchmark(
        "Game of Life", gol_trajs, loss_type="bce", binarize_rollout=True)
    all_results["Game of Life"] = (results_g, rollout_g, lt_g)

    del gol_trajs
    gc.collect()

    # ---- Summary Tables ----
    print_benchmark_table("Heat Equation", results_h, rollout_h, "MSE")
    print_benchmark_table("Game of Life", results_g, rollout_g, "CellAcc")
    print_combined_ranking(all_results)

    # ---- Plots ----
    make_benchmark_plot("Heat Equation", rollout_h,
                        "MSE (log scale)", log_scale=True)
    make_benchmark_plot("Game of Life", rollout_g,
                        "Cell Accuracy", log_scale=False)

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()
    run_experiment(args)
