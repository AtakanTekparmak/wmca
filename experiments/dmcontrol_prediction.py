"""DMControl next-state prediction: rescor_e3c vs baselines.

Trains CML-based world models and baselines as 1-step next-state predictors
on cartpole-swingup (5D state, 1D action) and reacher-easy (6D state, 2D action).

Architecture note:
  DMControl states are 1D vectors (not spatial grids). We reshape the
  concatenated (state, action) vector to (1, 1, D+A) — a 1D "image" with
  height=1 and width=D+A. The CML's 3x3 conv2d kernel with padding on a
  (1, W) grid effectively becomes a 1x3 conv — local coupling along the
  state vector dimensions. This is analogous to how we handled KS (1D PDE).

  KNOWN CONCERN: Unlike PDE grids, state dimensions lack natural spatial
  adjacency (e.g. cos(theta) next to sin(theta) is arbitrary). The local
  coupling may not provide useful inductive bias for this task.

Models:
  1. rescor_e3c  — CML + NCA residual correction (our best)
  2. PureNCA     — learned NCA only
  3. MLP         — 2-layer MLP baseline (D+A -> 256 -> D)
  4. GRU         — GRU baseline (processes sequences with temporal memory)

Usage:
    PYTHONPATH=src python3 experiments/dmcontrol_prediction.py --no-wandb
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

# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wmca.modules.hybrid import PureNCA, ResidualCorrectionWMv9
from wmca.utils import pick_device

# ---------------------------------------------------------------------------
# Lazy imports
# ---------------------------------------------------------------------------
def _get_plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


PLOTS_DIR = PROJECT_ROOT / "experiments" / "plots"
DATA_DIR = PROJECT_ROOT / "data"

TASKS = {
    "cartpole": {
        "file": "dmcontrol_cartpole.npz",
        "state_dim": 5,
        "action_dim": 1,
    },
    "reacher": {
        "file": "dmcontrol_reacher.npz",
        "state_dim": 6,
        "action_dim": 2,
    },
}

# Training hyperparameters
EPOCHS = 100
BATCH_SIZE = 256
LR = 1e-3
ROLLOUT_HORIZONS = [1, 5, 10, 25, 50]


# ============================================================================
# Simple baseline models (not CML-based)
# ============================================================================

class SimpleMLP(nn.Module):
    """MLP baseline: Linear(D+A, 256) -> ReLU -> Linear(256, 256) -> ReLU -> Linear(256, D)."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def param_count(self) -> dict[str, int]:
        return {
            "trained": sum(p.numel() for p in self.parameters()),
            "frozen": 0,
        }


class SimpleGRU(nn.Module):
    """GRU baseline: processes sequences, has temporal memory.

    For 1-step prediction: treats input as a length-1 sequence.
    For rollout: maintains hidden state across steps.
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, h: torch.Tensor | None = None):
        # x: (B, D+A) for 1-step, or (B, T, D+A) for sequence
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, 1, D+A)
        out, h_new = self.gru(x, h)
        pred = self.fc(out[:, -1, :])  # (B, D)
        return pred, h_new

    def param_count(self) -> dict[str, int]:
        return {
            "trained": sum(p.numel() for p in self.parameters()),
            "frozen": 0,
        }


# ============================================================================
# CML wrapper: reshapes flat vectors to/from (1, 1, W) spatial format
# ============================================================================

class CMLWrapper(nn.Module):
    """Wraps a CML-based model (rescor_e3c, PureNCA) for flat vector I/O.

    Input:  (B, D+A) flat vector
    Output: (B, D)   predicted next state

    Internally reshapes to (B, 1, 1, D+A) for the CML, then extracts
    the first D columns of the (B, 1, 1, D+A) or (B, 1, 1, D) output.
    """

    def __init__(self, cml_model: nn.Module, input_dim: int, output_dim: int):
        super().__init__()
        self.cml_model = cml_model
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D+A)
        B = x.shape[0]
        x_spatial = x.view(B, 1, 1, self.input_dim)  # (B, C=1, H=1, W=D+A)
        out_spatial = self.cml_model(x_spatial)  # (B, 1, 1, out_W)
        out_flat = out_spatial.view(B, -1)  # (B, out_W)
        # Take first output_dim values
        return out_flat[:, :self.output_dim]

    def param_count(self) -> dict[str, int]:
        return self.cml_model.param_count()


# ============================================================================
# Data loading and preprocessing
# ============================================================================

def load_task_data(task_name: str):
    """Load DMControl task data, flatten trajectories, normalize.

    Returns:
        X_train, Y_train, X_val, Y_val, X_test, Y_test: flat (N, D+A) / (N, D) arrays
        norm_info: dict with state_min, state_max for denormalization
        traj_data: dict with per-trajectory test data for rollouts
    """
    task = TASKS[task_name]
    data = np.load(DATA_DIR / task["file"])

    states = data["states"]       # (500, 100, D)
    actions = data["actions"]     # (500, 100, A)
    next_states = data["next_states"]  # (500, 100, D)
    train_idx = data["train_idx"]
    val_idx = data["val_idx"]
    test_idx = data["test_idx"]

    D = task["state_dim"]
    A = task["action_dim"]

    # Compute normalization from training data only
    train_states = states[train_idx]  # (N_train, 100, D)
    train_next = next_states[train_idx]
    all_train_states = np.concatenate([
        train_states.reshape(-1, D),
        train_next.reshape(-1, D),
    ], axis=0)

    state_min = all_train_states.min(axis=0)  # (D,)
    state_max = all_train_states.max(axis=0)  # (D,)
    # Prevent division by zero
    state_range = state_max - state_min
    state_range[state_range < 1e-8] = 1.0

    norm_info = {
        "state_min": state_min,
        "state_max": state_max,
        "state_range": state_range,
    }

    def normalize_states(s):
        return (s - state_min) / state_range

    def normalize_actions(a):
        # Actions are in [-1, 1], shift to [0, 1]
        return (a + 1.0) / 2.0

    def make_flat_pairs(idx):
        """Flatten trajectories to (state, action) -> next_state pairs."""
        s = states[idx]       # (N_traj, 100, D)
        a = actions[idx]      # (N_traj, 100, A)
        ns = next_states[idx] # (N_traj, 100, D)

        s_flat = s.reshape(-1, D)
        a_flat = a.reshape(-1, A)
        ns_flat = ns.reshape(-1, D)

        s_norm = normalize_states(s_flat)
        a_norm = normalize_actions(a_flat)
        ns_norm = normalize_states(ns_flat)

        X = np.concatenate([s_norm, a_norm], axis=1).astype(np.float32)  # (N, D+A)
        Y = ns_norm.astype(np.float32)  # (N, D)
        return X, Y

    X_train, Y_train = make_flat_pairs(train_idx)
    X_val, Y_val = make_flat_pairs(val_idx)
    X_test, Y_test = make_flat_pairs(test_idx)

    # For rollout evaluation: keep per-trajectory structure
    test_states = states[test_idx]      # (N_test, 100, D)
    test_actions = actions[test_idx]    # (N_test, 100, A)
    test_next = next_states[test_idx]   # (N_test, 100, D)

    traj_data = {
        "states": normalize_states(test_states).astype(np.float32),
        "actions": normalize_actions(test_actions).astype(np.float32),
        "next_states": normalize_states(test_next).astype(np.float32),
        "n_traj": len(test_idx),
    }

    print(f"  [{task_name}] train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    print(f"  [{task_name}] state_dim={D}, action_dim={A}, input_dim={D+A}")
    print(f"  [{task_name}] state range: min={state_min}, max={state_max}")

    return X_train, Y_train, X_val, Y_val, X_test, Y_test, norm_info, traj_data


# ============================================================================
# Training loop
# ============================================================================

def train_model(model, X_train, Y_train, X_val, Y_val,
                epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR,
                device="cpu", model_name="", is_gru=False):
    """Train a model with early stopping based on val loss."""
    dev = torch.device(device) if isinstance(device, str) else device
    model = model.to(dev)

    # Handle dilation_alpha weight decay for rescor_e3c
    alpha_params = []
    other_params = []
    for pname, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "dilation_alpha" in pname:
            alpha_params.append(p)
        else:
            other_params.append(p)

    if alpha_params:
        optimizer = torch.optim.Adam([
            {"params": other_params, "weight_decay": 0.0},
            {"params": alpha_params, "weight_decay": 1.0},
        ], lr=lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    criterion = nn.MSELoss()

    X_tr = torch.from_numpy(X_train).float().to(dev)
    Y_tr = torch.from_numpy(Y_train).float().to(dev)
    X_v = torch.from_numpy(X_val).float().to(dev)
    Y_v = torch.from_numpy(Y_val).float().to(dev)

    best_val_loss = float("inf")
    best_state = None
    patience = 15
    patience_counter = 0

    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(X_tr), device=dev)
        total_loss = 0.0
        n_batches = 0

        for i in range(0, len(perm), batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = X_tr[idx], Y_tr[idx]

            if is_gru:
                pred, _ = model(xb)
            else:
                pred = model(xb)

            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        avg_train = total_loss / max(n_batches, 1)

        # Validation
        model.eval()
        with torch.no_grad():
            if is_gru:
                val_pred, _ = model(X_v)
            else:
                val_pred = model(X_v)
            val_loss = criterion(val_pred, Y_v).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"    [{model_name}] epoch {epoch+1}/{epochs}  "
                  f"train_loss={avg_train:.6f}  val_loss={val_loss:.6f}  "
                  f"best_val={best_val_loss:.6f}")

        if patience_counter >= patience:
            print(f"    [{model_name}] early stopping at epoch {epoch+1}")
            break

    train_time = time.time() - t0

    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.cpu().eval()

    print(f"    [{model_name}] training done in {train_time:.1f}s, best_val={best_val_loss:.6f}")
    return model, best_val_loss, train_time


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_1step(model, X_test, Y_test, device, is_gru=False):
    """1-step prediction MSE on test set. Returns total and per-dimension MSE."""
    dev = torch.device(device) if isinstance(device, str) else device
    model = model.to(dev).eval()

    X_t = torch.from_numpy(X_test).float().to(dev)
    Y_t = torch.from_numpy(Y_test).float().to(dev)

    with torch.no_grad():
        if is_gru:
            pred, _ = model(X_t)
        else:
            pred = model(X_t)

        # Total MSE
        mse = nn.functional.mse_loss(pred, Y_t).item()

        # Per-dimension MSE
        per_dim_mse = ((pred - Y_t) ** 2).mean(dim=0).cpu().numpy()

    model = model.cpu()
    return mse, per_dim_mse


def evaluate_rollout(model, traj_data, state_dim, action_dim, horizons,
                     device, is_gru=False, max_traj=50):
    """Multi-step autoregressive rollout evaluation.

    For each test trajectory, predict autoregressively using true actions.
    Report average MSE at each horizon.
    """
    dev = torch.device(device) if isinstance(device, str) else device
    model = model.to(dev).eval()

    states = traj_data["states"]       # (N_traj, 100, D)
    actions_data = traj_data["actions"]  # (N_traj, 100, A)
    next_states = traj_data["next_states"]  # (N_traj, 100, D)

    n_traj = min(traj_data["n_traj"], max_traj)
    max_h = max(horizons)

    # Collect MSE at each horizon
    horizon_mses = {h: [] for h in horizons}

    for t_idx in range(n_traj):
        traj_states = states[t_idx]     # (100, D)
        traj_actions = actions_data[t_idx]  # (100, A)
        traj_next = next_states[t_idx]  # (100, D)

        # Start from the first state
        current_state = traj_states[0].copy()  # (D,)
        h_state = None  # GRU hidden state

        predicted_states = []

        with torch.no_grad():
            for step in range(min(max_h, 99)):  # max 99 steps (100 states - 1)
                action = traj_actions[step]  # (A,)
                inp = np.concatenate([current_state, action])  # (D+A,)
                inp_t = torch.from_numpy(inp).float().unsqueeze(0).to(dev)

                if is_gru:
                    pred, h_state = model(inp_t, h_state)
                else:
                    pred = model(inp_t)

                pred_np = pred.squeeze(0).cpu().numpy()
                pred_np = np.clip(pred_np, 0, 1)  # keep in [0, 1] normalized range
                predicted_states.append(pred_np)
                current_state = pred_np  # autoregressive: feed prediction back

        # Compute MSE at each horizon
        for h in horizons:
            if h <= len(predicted_states):
                # Ground truth: next_states at steps 0..h-1
                gt = traj_next[:h]  # (h, D)
                pred_arr = np.stack(predicted_states[:h])  # (h, D)
                mse = np.mean((gt - pred_arr) ** 2)
                horizon_mses[h].append(mse)

    model = model.cpu()

    results = {}
    for h in horizons:
        if horizon_mses[h]:
            results[h] = float(np.mean(horizon_mses[h]))
        else:
            results[h] = float("nan")

    return results


# ============================================================================
# Model creation helpers
# ============================================================================

def create_cml_model(model_type: str, state_dim: int, action_dim: int, seed: int = 42):
    """Create a CML-based model wrapped for flat vector I/O."""
    input_dim = state_dim + action_dim

    if model_type == "rescor_e3c":
        # in_channels=1 (single "channel"), width=D+A
        # The CML operates on the first out_channels of input
        # For this 1D case: out_channels=1, and the spatial dim is the width
        inner = ResidualCorrectionWMv9(
            in_channels=1,
            out_channels=1,
            hidden_ch=32,
            cml_steps=15,
            r=3.90,
            eps=0.3,
            beta=0.15,
            seed=seed,
            use_sigmoid=False,  # We handle clipping ourselves
        )
    elif model_type == "pure_nca":
        inner = PureNCA(
            in_channels=1,
            out_channels=1,
            hidden_ch=16,
            steps=1,
            use_sigmoid=False,
        )
    else:
        raise ValueError(f"Unknown CML model type: {model_type}")

    return CMLWrapper(inner, input_dim=input_dim, output_dim=state_dim)


def create_mlp_model(state_dim: int, action_dim: int):
    """Create MLP baseline."""
    return SimpleMLP(input_dim=state_dim + action_dim, output_dim=state_dim)


def create_gru_model(state_dim: int, action_dim: int):
    """Create GRU baseline."""
    return SimpleGRU(input_dim=state_dim + action_dim, output_dim=state_dim)


# ============================================================================
# Main experiment
# ============================================================================

def run_experiment(args):
    device = pick_device()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for task_name in ["cartpole", "reacher"]:
        print(f"\n{'='*70}")
        print(f"  TASK: {task_name}")
        print(f"{'='*70}")

        task_info = TASKS[task_name]
        D = task_info["state_dim"]
        A = task_info["action_dim"]

        # Load data
        X_train, Y_train, X_val, Y_val, X_test, Y_test, norm_info, traj_data = \
            load_task_data(task_name)

        task_results = {}

        # ------------------------------------------------------------------
        # 1. rescor_e3c (seeds 0, 1, 2)
        # ------------------------------------------------------------------
        print(f"\n--- rescor_e3c ---")
        rescor_runs = []
        for seed in [0, 1, 2]:
            print(f"  seed={seed}")
            model = create_cml_model("rescor_e3c", D, A, seed=seed)
            torch.manual_seed(seed)
            np.random.seed(seed)

            model, val_loss, train_time = train_model(
                model, X_train, Y_train, X_val, Y_val,
                device=device, model_name=f"rescor_e3c_s{seed}")

            mse, per_dim = evaluate_1step(model, X_test, Y_test, device)
            rollout = evaluate_rollout(model, traj_data, D, A, ROLLOUT_HORIZONS, device)
            params = model.param_count()

            rescor_runs.append({
                "seed": seed,
                "mse": mse,
                "per_dim_mse": per_dim,
                "rollout": rollout,
                "params": params,
                "train_time": train_time,
                "val_loss": val_loss,
            })
            print(f"    1-step MSE={mse:.6f}, params={params}")
            print(f"    rollout MSE: {rollout}")

        task_results["rescor_e3c"] = rescor_runs

        # ------------------------------------------------------------------
        # 2. PureNCA (seeds 0, 1, 2)
        # ------------------------------------------------------------------
        print(f"\n--- PureNCA ---")
        nca_runs = []
        for seed in [0, 1, 2]:
            print(f"  seed={seed}")
            model = create_cml_model("pure_nca", D, A, seed=seed)
            torch.manual_seed(seed)
            np.random.seed(seed)

            model, val_loss, train_time = train_model(
                model, X_train, Y_train, X_val, Y_val,
                device=device, model_name=f"pure_nca_s{seed}")

            mse, per_dim = evaluate_1step(model, X_test, Y_test, device)
            rollout = evaluate_rollout(model, traj_data, D, A, ROLLOUT_HORIZONS, device)
            params = model.param_count()

            nca_runs.append({
                "seed": seed,
                "mse": mse,
                "per_dim_mse": per_dim,
                "rollout": rollout,
                "params": params,
                "train_time": train_time,
                "val_loss": val_loss,
            })
            print(f"    1-step MSE={mse:.6f}, params={params}")
            print(f"    rollout MSE: {rollout}")

        task_results["pure_nca"] = nca_runs

        # ------------------------------------------------------------------
        # 3. MLP (seed 42)
        # ------------------------------------------------------------------
        print(f"\n--- MLP ---")
        torch.manual_seed(42)
        np.random.seed(42)
        model = create_mlp_model(D, A)

        model, val_loss, train_time = train_model(
            model, X_train, Y_train, X_val, Y_val,
            device=device, model_name="mlp")

        mse, per_dim = evaluate_1step(model, X_test, Y_test, device)
        rollout = evaluate_rollout(model, traj_data, D, A, ROLLOUT_HORIZONS, device)
        params = model.param_count()

        task_results["mlp"] = [{
            "seed": 42,
            "mse": mse,
            "per_dim_mse": per_dim,
            "rollout": rollout,
            "params": params,
            "train_time": train_time,
            "val_loss": val_loss,
        }]
        print(f"  1-step MSE={mse:.6f}, params={params}")
        print(f"  rollout MSE: {rollout}")

        # ------------------------------------------------------------------
        # 4. GRU (seed 42)
        # ------------------------------------------------------------------
        print(f"\n--- GRU ---")
        torch.manual_seed(42)
        np.random.seed(42)
        model = create_gru_model(D, A)

        model, val_loss, train_time = train_model(
            model, X_train, Y_train, X_val, Y_val,
            device=device, model_name="gru", is_gru=True)

        mse, per_dim = evaluate_1step(model, X_test, Y_test, device, is_gru=True)
        rollout = evaluate_rollout(model, traj_data, D, A, ROLLOUT_HORIZONS, device,
                                   is_gru=True)
        params = model.param_count()

        task_results["gru"] = [{
            "seed": 42,
            "mse": mse,
            "per_dim_mse": per_dim,
            "rollout": rollout,
            "params": params,
            "train_time": train_time,
            "val_loss": val_loss,
        }]
        print(f"  1-step MSE={mse:.6f}, params={params}")
        print(f"  rollout MSE: {rollout}")

        all_results[task_name] = task_results

    # ======================================================================
    # Summary and plotting
    # ======================================================================
    print_summary(all_results)
    plot_results(all_results)

    return all_results


def print_summary(all_results):
    """Print summary tables."""
    print(f"\n\n{'='*80}")
    print("  SUMMARY TABLES")
    print(f"{'='*80}")

    for task_name, task_results in all_results.items():
        print(f"\n--- {task_name.upper()} ---")
        print(f"{'Model':<15} {'1-step MSE':>12} {'Params (T/F)':>15} "
              f"{'Time (s)':>10} {'Rollout@5':>12} {'Rollout@10':>12} "
              f"{'Rollout@25':>12} {'Rollout@50':>12}")
        print("-" * 110)

        for model_name in ["rescor_e3c", "pure_nca", "mlp", "gru"]:
            runs = task_results[model_name]

            # Average across seeds
            mses = [r["mse"] for r in runs]
            mean_mse = np.mean(mses)
            std_mse = np.std(mses) if len(mses) > 1 else 0.0

            params = runs[0]["params"]
            times = [r["train_time"] for r in runs]
            mean_time = np.mean(times)

            # Average rollout MSEs
            rollout_means = {}
            for h in ROLLOUT_HORIZONS:
                vals = [r["rollout"][h] for r in runs if h in r["rollout"]]
                rollout_means[h] = np.mean(vals) if vals else float("nan")

            mse_str = f"{mean_mse:.6f}"
            if std_mse > 0:
                mse_str += f" +/- {std_mse:.6f}"

            param_str = f"{params['trained']}/{params['frozen']}"

            print(f"{model_name:<15} {mse_str:>25} {param_str:>15} "
                  f"{mean_time:>10.1f} {rollout_means.get(5, float('nan')):>12.6f} "
                  f"{rollout_means.get(10, float('nan')):>12.6f} "
                  f"{rollout_means.get(25, float('nan')):>12.6f} "
                  f"{rollout_means.get(50, float('nan')):>12.6f}")

        # Per-dimension MSE for best seed of each model
        print(f"\n  Per-dimension 1-step MSE (best seed):")
        for model_name in ["rescor_e3c", "pure_nca", "mlp", "gru"]:
            runs = task_results[model_name]
            best_run = min(runs, key=lambda r: r["mse"])
            dims = best_run["per_dim_mse"]
            dim_str = ", ".join([f"d{i}={v:.6f}" for i, v in enumerate(dims)])
            print(f"    {model_name:<15} {dim_str}")


def plot_results(all_results):
    """Generate rollout MSE vs horizon plots."""
    plt = _get_plt()

    colors = {
        "rescor_e3c": "#e41a1c",
        "pure_nca": "#377eb8",
        "mlp": "#4daf4a",
        "gru": "#984ea3",
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax_idx, task_name in enumerate(["cartpole", "reacher"]):
        ax = axes[ax_idx]
        task_results = all_results[task_name]

        for model_name in ["rescor_e3c", "pure_nca", "mlp", "gru"]:
            runs = task_results[model_name]

            # Collect rollout curves
            all_curves = []
            for run in runs:
                curve = [run["rollout"].get(h, float("nan")) for h in ROLLOUT_HORIZONS]
                all_curves.append(curve)

            all_curves = np.array(all_curves)  # (n_seeds, n_horizons)
            mean_curve = np.mean(all_curves, axis=0)

            label = f"{model_name} ({runs[0]['params']['trained']}p)"
            ax.plot(ROLLOUT_HORIZONS, mean_curve, 'o-',
                    color=colors[model_name], label=label, linewidth=2)

            if len(runs) > 1:
                std_curve = np.std(all_curves, axis=0)
                ax.fill_between(ROLLOUT_HORIZONS,
                                mean_curve - std_curve,
                                mean_curve + std_curve,
                                alpha=0.2, color=colors[model_name])

        ax.set_xlabel("Rollout Horizon (steps)")
        ax.set_ylabel("MSE")
        ax.set_title(f"{task_name}")
        ax.legend(fontsize=8)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

    fig.suptitle("DMControl Next-State Prediction: Rollout MSE vs Horizon", fontsize=14)
    plt.tight_layout()

    out_path = PLOTS_DIR / "dmcontrol_rollout_mse.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {out_path}")
    plt.close()

    # Also make a bar chart for 1-step MSE
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax_idx, task_name in enumerate(["cartpole", "reacher"]):
        ax = axes[ax_idx]
        task_results = all_results[task_name]

        model_names = ["rescor_e3c", "pure_nca", "mlp", "gru"]
        means = []
        stds = []
        bar_colors = []

        for mn in model_names:
            runs = task_results[mn]
            mses = [r["mse"] for r in runs]
            means.append(np.mean(mses))
            stds.append(np.std(mses) if len(mses) > 1 else 0.0)
            bar_colors.append(colors[mn])

        x = np.arange(len(model_names))
        bars = ax.bar(x, means, yerr=stds, capsize=5, color=bar_colors, alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=15)
        ax.set_ylabel("1-step MSE")
        ax.set_title(f"{task_name}")
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{mean:.5f}', ha='center', va='bottom', fontsize=8)

    fig.suptitle("DMControl 1-Step Prediction MSE", fontsize=14)
    plt.tight_layout()

    out_path = PLOTS_DIR / "dmcontrol_1step_mse.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {out_path}")
    plt.close()


# ============================================================================
# Entry point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="DMControl prediction experiment")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    args = parser.parse_args()

    print("DMControl Next-State Prediction Experiment")
    print(f"  epochs={EPOCHS}, batch_size={BATCH_SIZE}, lr={LR}")
    print(f"  rollout horizons={ROLLOUT_HORIZONS}")
    print(f"  CML note: state dims lack spatial adjacency — local coupling may not help")
    print()

    run_experiment(args)


if __name__ == "__main__":
    main()
