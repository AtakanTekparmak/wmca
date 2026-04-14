"""Game of Life Prediction — Learned CML (NCA-style) vs Conv2D Baseline.

Each Learned CML model replaces the fixed logistic map f(x)=r*x*(1-x) with a
small shared neural network. Every cell looks at its 3x3 neighbourhood, feeds
it through a shared MLP (implemented as 1x1 convs after a 3x3 perception
layer), and outputs the next state — identical to a Neural Cellular Automaton.

Models compared
---------------
1. LearnedCML-1step   : 1 NCA iteration
2. LearnedCML-3step   : 3 NCA iterations (larger effective receptive field)
3. LearnedCML-5step   : 5 NCA iterations
4. LearnedCML-10step  : 10 NCA iterations
5. LearnedCML-res-3step: residual + 3 iterations (x = x + f(x))
6. Conv2D baseline     : 3-layer CNN with 3x3 kernels (from gol_prediction.py)

Usage
-----
    FORCE_CPU=1 uv run --with scikit-learn,matplotlib,scipy python experiments/gol_learned_cml.py --no-wandb
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
# Project root on path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wmca.utils import pick_device


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
DENSITY = 0.3

N_TRAJECTORIES = 1000
TRAJ_LENGTH = 20

LR = 1e-3
EPOCHS = 50
BATCH_SIZE = 64

HIDDEN_CH = 16
ROLLOUT_HORIZONS = [1, 2, 3, 5, 10]


# ===== Data Generation =====================================================
def gol_step(grid: np.ndarray) -> np.ndarray:
    """Single Game of Life step. grid: (H, W) binary {0, 1}."""
    from scipy.signal import convolve2d
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.float32)
    neighbors = convolve2d(grid.astype(np.float32), kernel,
                           mode="same", boundary="wrap")
    born = (grid == 0) & (neighbors == 3)
    survive = (grid == 1) & ((neighbors == 2) | (neighbors == 3))
    return (born | survive).astype(np.float32)


def generate_trajectories(n_traj: int = N_TRAJECTORIES,
                          traj_len: int = TRAJ_LENGTH,
                          h: int = GRID_H, w: int = GRID_W,
                          density: float = DENSITY,
                          seed: int = SEED) -> np.ndarray:
    """Returns (n_traj, traj_len+1, H, W) float32."""
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


# ===== Model Definitions ====================================================

class LearnedCML2D(nn.Module):
    """Neural Cellular Automaton — learned local update rule.

    Each cell perceives its 3x3 neighbourhood (perception layer) then maps the
    resulting feature vector through a shared MLP (implemented as 1x1 convs).
    The output is squeezed through Sigmoid to stay in [0, 1].

    Args:
        hidden_ch: Number of feature channels after perception.
        steps:     Number of NCA iterations per forward pass.
        residual:  If True, use residual connection: x_new = sigmoid(x + delta).
    """

    def __init__(self, hidden_ch: int = HIDDEN_CH,
                 steps: int = 1, residual: bool = False):
        super().__init__()
        self.steps = steps
        self.residual = residual

        # 3x3 neighbourhood perception (shared across all cells / positions)
        self.perceive = nn.Conv2d(1, hidden_ch, 3, padding=1)

        # Channel-wise update MLP (1x1 convolutions = fully connected per cell)
        self.update = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(hidden_ch, hidden_ch, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_ch, 1, 1),
        )
        if not residual:
            self.update.add_module("sigmoid", nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, H, W) in [0, 1] -> (B, 1, H, W) in [0, 1]."""
        for _ in range(self.steps):
            features = self.perceive(x)
            delta = self.update(features)
            if self.residual:
                x = torch.sigmoid(x + delta)
            else:
                x = delta
        return x

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


class Conv2DBaseline(nn.Module):
    """3-layer Conv2D baseline — same as gol_prediction.py."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ===== Training =============================================================

def train_model(model: nn.Module,
                X_train: np.ndarray, Y_train: np.ndarray,
                X_val: np.ndarray, Y_val: np.ndarray,
                epochs: int = EPOCHS, lr: float = LR,
                batch_size: int = BATCH_SIZE,
                device: torch.device | None = None) -> nn.Module:
    """Train any (B,1,H,W)->(B,1,H,W) model with BCE loss + Adam."""
    if device is None:
        device = torch.device("cpu")

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    X_tr = torch.from_numpy(X_train).float().unsqueeze(1).to(device)
    Y_tr = torch.from_numpy(Y_train).float().unsqueeze(1).to(device)
    X_v  = torch.from_numpy(X_val).float().unsqueeze(1).to(device)
    Y_v  = torch.from_numpy(Y_val).float().unsqueeze(1).to(device)

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

def cell_accuracy(Y_true: np.ndarray, Y_pred: np.ndarray,
                  threshold: float = 0.5) -> float:
    pred_binary = (Y_pred >= threshold).astype(np.float32)
    return float(np.mean(pred_binary == Y_true))


def grid_perfect_accuracy(Y_true: np.ndarray, Y_pred: np.ndarray,
                          threshold: float = 0.5) -> float:
    pred_binary = (Y_pred >= threshold).astype(np.float32)
    per_grid = np.all(
        pred_binary.reshape(len(Y_true), -1) == Y_true.reshape(len(Y_true), -1),
        axis=1,
    )
    return float(np.mean(per_grid))


def multistep_rollout(predict_fn, x0: np.ndarray,
                      n_steps: int) -> np.ndarray:
    """Roll out predict_fn for n_steps, binarising feedback.

    Returns (n_steps, H, W) continuous predictions.
    """
    preds = np.zeros((n_steps, GRID_H, GRID_W), dtype=np.float32)
    x = x0.copy()
    for t in range(n_steps):
        raw = predict_fn(x)
        preds[t] = raw
        x = (np.clip(raw, 0, 1) >= 0.5).astype(np.float32)
    return preds


def multistep_cell_accuracy(true_grids: np.ndarray,
                            pred_grids: np.ndarray) -> float:
    return float(np.mean((pred_grids >= 0.5).astype(np.float32) == true_grids))


def make_predict_fn(model: nn.Module):
    """Wrap a (B,1,H,W) model into a (H,W) -> (H,W) callable."""
    model.eval()

    def _predict(x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x_t = torch.from_numpy(x).float().reshape(1, 1, GRID_H, GRID_W)
            out = model(x_t).squeeze().numpy()
        return out

    return _predict


def evaluate_model(model: nn.Module,
                   X_test: np.ndarray, Y_test: np.ndarray):
    """Return (cell_acc, grid_acc, pred_array)."""
    model.eval()
    with torch.no_grad():
        X_t = torch.from_numpy(X_test).float().unsqueeze(1)
        pred = model(X_t).squeeze(1).numpy()
    return cell_accuracy(Y_test, pred), grid_perfect_accuracy(Y_test, pred), pred


# ===== Plotting =============================================================

MODEL_COLORS = {
    "LearnedCML-1step":    "tab:blue",
    "LearnedCML-3step":    "tab:orange",
    "LearnedCML-5step":    "tab:green",
    "LearnedCML-10step":   "tab:red",
    "LearnedCML-res-3step":"tab:purple",
    "Conv2D":              "tab:brown",
}


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

    # ---- Figure 1: Bar chart — cell accuracy for all variants ----
    fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(14, 5))
    x_pos = np.arange(len(model_names))
    cell_accs = [results[n]["cell_acc"] for n in model_names]
    grid_accs = [results[n]["grid_acc"] for n in model_names]
    colors = [MODEL_COLORS.get(n, "tab:gray") for n in model_names]

    ax1a.bar(x_pos, cell_accs, color=colors, alpha=0.85)
    ax1a.set_xticks(x_pos)
    ax1a.set_xticklabels(model_names, rotation=20, ha="right", fontsize=9)
    ax1a.set_ylabel("Cell Accuracy")
    ax1a.set_title("1-Step Cell Accuracy")
    ax1a.set_ylim(0, 1.05)
    ax1a.grid(True, alpha=0.3, axis="y")
    for i, v in enumerate(cell_accs):
        ax1a.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=8)

    ax1b.bar(x_pos, grid_accs, color=colors, alpha=0.85)
    ax1b.set_xticks(x_pos)
    ax1b.set_xticklabels(model_names, rotation=20, ha="right", fontsize=9)
    ax1b.set_ylabel("Grid-Perfect Accuracy")
    ax1b.set_title("1-Step Grid-Perfect Accuracy")
    max_ga = max(grid_accs) if max(grid_accs) > 0 else 0.05
    ax1b.set_ylim(0, max_ga * 1.35 + 0.01)
    ax1b.grid(True, alpha=0.3, axis="y")
    for i, v in enumerate(grid_accs):
        ax1b.text(i, v + max_ga * 0.02, f"{v:.4f}", ha="center", fontsize=8)

    fig1.suptitle("GoL Learned CML: 1-Step Accuracy", fontsize=13)
    fig1.tight_layout()
    path1 = PLOTS_DIR / "gol_learned_cml_accuracy_bars.png"
    fig1.savefig(path1, dpi=150)
    if wandb:
        wandb.log({"plots/accuracy_bars": wandb.Image(fig1)})
    plt.close(fig1)

    # ---- Figure 2: Line plot — rollout accuracy vs horizon ----
    fig2, ax2 = plt.subplots(figsize=(9, 5))
    markers = ["o", "s", "^", "D", "v", "P"]
    for mi, name in enumerate(model_names):
        vals = [rollout_results[h][name] for h in horizons]
        ax2.plot(horizons, vals, f"{markers[mi % len(markers)]}-",
                 color=MODEL_COLORS.get(name, "tab:gray"),
                 label=name, markersize=7)
    ax2.set_xlabel("Rollout horizon (steps)")
    ax2.set_ylabel("Cell accuracy")
    ax2.set_title("GoL Learned CML: Multi-step Rollout Accuracy")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.4, 1.02)
    fig2.tight_layout()
    path2 = PLOTS_DIR / "gol_learned_cml_rollout.png"
    fig2.savefig(path2, dpi=150)
    if wandb:
        wandb.log({"plots/rollout_accuracy": wandb.Image(fig2)})
    plt.close(fig2)

    # ---- Figure 3: Example grids — true vs best NCA model ----
    # Identify best NCA model by cell accuracy (exclude Conv2D from NCA comparison)
    nca_names = [n for n in model_names if n != "Conv2D"]
    best_nca = max(nca_names, key=lambda n: results[n]["cell_acc"])
    show_steps = [0, 2, min(4, len(true_future) - 1)]

    fig3, axes = plt.subplots(len(show_steps), 3,
                              figsize=(9, 3 * len(show_steps)))
    col_titles = ["True", best_nca, "Conv2D"]
    rollout_cols = [true_future, rollouts[best_nca], rollouts["Conv2D"]]

    for row_i, t in enumerate(show_steps):
        for col_i, (title, data) in enumerate(zip(col_titles, rollout_cols)):
            ax = axes[row_i, col_i]
            grid = data[t] if col_i == 0 else (data[t] >= 0.5).astype(np.float32)
            ax.imshow(grid, cmap="binary", vmin=0, vmax=1, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if row_i == 0:
                ax.set_title(title, fontsize=10)
            if col_i == 0:
                ax.set_ylabel(f"t+{t+1}", fontsize=10)

    fig3.suptitle(f"GoL: True vs Best NCA ({best_nca}) vs Conv2D",
                  fontsize=12, y=1.01)
    fig3.tight_layout()
    path3 = PLOTS_DIR / "gol_learned_cml_examples.png"
    fig3.savefig(path3, dpi=150, bbox_inches="tight")
    if wandb:
        wandb.log({"plots/example_grids": wandb.Image(fig3)})
    plt.close(fig3)

    print(f"  Plots saved -> {path1.name}, {path2.name}, {path3.name}")


# ===== Summary ==============================================================

def print_summary(results: dict, rollout_results: dict):
    model_names = list(results.keys())
    horizons = sorted(rollout_results.keys())

    print("\n" + "=" * 90)
    print("SUMMARY: GAME OF LIFE — LEARNED CML (NCA) EXPERIMENT")
    print("=" * 90)

    # One-step accuracy table
    col_w = 14
    header = f"{'Model':<22s}  {'Cell Acc':>{col_w}}  {'Grid-Perfect':>{col_w}}  {'Params':>{col_w}}"
    print(f"\n{header}")
    print("-" * (22 + 3 * (col_w + 2) + 4))
    for name in model_names:
        r = results[name]
        print(f"{name:<22s}  {r['cell_acc']:{col_w}.4f}  "
              f"{r['grid_acc']:{col_w}.4f}  {r['params']:{col_w}d}")

    # Multi-step rollout table
    print(f"\n--- Multi-step Rollout Cell Accuracy ---")
    hdr = f"{'Horizon':>8s}"
    for name in model_names:
        hdr += f"  {name:>22s}"
    print(hdr)
    for h in horizons:
        row = f"{h:8d}"
        for name in model_names:
            row += f"  {rollout_results[h][name]:22.4f}"
        print(row)

    print("=" * 90)


# ===== Main Experiment ======================================================

def run_experiment(args):
    os.environ.setdefault("FORCE_CPU", "1")
    device = pick_device()

    log_wandb = not args.no_wandb
    config = dict(
        grid_h=GRID_H, grid_w=GRID_W, density=DENSITY,
        n_trajectories=N_TRAJECTORIES, traj_length=TRAJ_LENGTH,
        hidden_ch=HIDDEN_CH, lr=LR, epochs=EPOCHS, batch_size=BATCH_SIZE,
    )

    if log_wandb:
        from wmca.training import init_wandb
        init_wandb("gol-learned-cml", config=config,
                   tags=["gol", "nca", "learned-cml"])

    print("=" * 72)
    print("GAME OF LIFE — LEARNED CML (NCA-style) EXPERIMENT")
    print("=" * 72)

    # ---- Data ----
    print("\n[1/9] Generating Game of Life trajectories ...")
    t0 = time.time()
    trajs = generate_trajectories()
    train_trajs, val_trajs, test_trajs = split_trajectories(trajs)
    X_train, Y_train = make_pairs(train_trajs)
    X_val,   Y_val   = make_pairs(val_trajs)
    X_test,  Y_test  = make_pairs(test_trajs)
    print(f"  {len(trajs)} trajectories x {TRAJ_LENGTH} steps on "
          f"{GRID_H}x{GRID_W}  ({time.time()-t0:.1f}s)")
    print(f"  Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

    results: dict[str, dict] = {}

    # Model specs: (label, steps, residual)
    nca_specs = [
        ("LearnedCML-1step",    1,  False),
        ("LearnedCML-3step",    3,  False),
        ("LearnedCML-res-3step", 3, True),
    ]

    trained_models: dict[str, nn.Module] = {}

    for step_i, (label, steps, residual) in enumerate(nca_specs, start=2):
        print(f"\n[{step_i}/9] {label}  (steps={steps}, residual={residual}) ...")
        t0 = time.time()
        model = LearnedCML2D(hidden_ch=HIDDEN_CH, steps=steps, residual=residual)
        model = train_model(model, X_train, Y_train, X_val, Y_val, device=device)
        cell_acc, grid_acc, _ = evaluate_model(model, X_test, Y_test)
        elapsed = time.time() - t0
        params = model.param_count()
        print(f"  Cell accuracy:    {cell_acc:.4f}")
        print(f"  Grid-perfect acc: {grid_acc:.4f}")
        print(f"  Params: {params}  ({elapsed:.1f}s)")
        results[label] = {"cell_acc": cell_acc, "grid_acc": grid_acc, "params": params}
        trained_models[label] = model

    # ---- Conv2D Baseline ----
    print(f"\n[7/9] Conv2D baseline ...")
    t0 = time.time()
    cnn = Conv2DBaseline()
    cnn = train_model(cnn, X_train, Y_train, X_val, Y_val, device=device)
    cnn_cell_acc, cnn_grid_acc, _ = evaluate_model(cnn, X_test, Y_test)
    elapsed = time.time() - t0
    cnn_params = cnn.param_count()
    print(f"  Cell accuracy:    {cnn_cell_acc:.4f}")
    print(f"  Grid-perfect acc: {cnn_grid_acc:.4f}")
    print(f"  Params: {cnn_params}  ({elapsed:.1f}s)")
    results["Conv2D"] = {"cell_acc": cnn_cell_acc, "grid_acc": cnn_grid_acc, "params": cnn_params}
    trained_models["Conv2D"] = cnn

    # ---- Multi-step rollout ----
    print(f"\n[8/9] Multi-step rollout evaluation ...")
    test_traj = test_trajs[0]
    x0 = test_traj[0]
    max_horizon = min(max(ROLLOUT_HORIZONS), TRAJ_LENGTH)
    true_future = test_traj[1:max_horizon + 1]

    rollouts: dict[str, np.ndarray] = {}
    for name, model in trained_models.items():
        rollouts[name] = multistep_rollout(make_predict_fn(model), x0, max_horizon)

    rollout_results: dict[int, dict] = {}
    model_names = list(results.keys())
    for h in ROLLOUT_HORIZONS:
        if h > max_horizon:
            break
        true_h = true_future[:h]
        row = {name: multistep_cell_accuracy(true_h, rollouts[name][:h])
               for name in model_names}
        rollout_results[h] = row
        parts = "  ".join(f"{n}={row[n]:.4f}" for n in model_names)
        print(f"  Horizon {h:>2d}:  {parts}")

    # ---- Plots ----
    print(f"\n[9/9] Generating plots ...")
    make_plots(results, rollout_results, true_future, rollouts, x0, log_wandb)

    # ---- wandb logging ----
    if log_wandb:
        import wandb
        for name, r in results.items():
            tag = name.lower().replace("-", "_")
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

    # ---- Summary ----
    print_summary(results, rollout_results)


# ===== Entry Point ==========================================================

def main():
    parser = argparse.ArgumentParser(
        description="GoL prediction: Learned CML (NCA) vs Conv2D baseline")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Skip wandb logging")
    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
