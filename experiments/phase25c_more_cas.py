"""Phase 2.5c: PureNCA generalisation to more discrete CAs (Rule 110, Wireworld).

Tests whether the Matching Principle holds: PureNCA should beat fixed CML on
discrete CAs because CML's logistic-map dynamics are a poor prior for discrete
update rules.

Models per CA:
  1. ResidualCorrection(D) -- CML base + NCA correction
  2. PureNCA              -- learned only
  3. Conv baseline        -- neural baseline (Conv2d / Conv1d-like)
  4. CML2D + Ridge        -- reservoir baseline

CA 1: Rule 110  (1D, binary, width=64, Turing complete)
CA 2: Wireworld (2D, 4-state, 16x16)

Usage:
    FORCE_CPU=1 uv run --with scikit-learn,matplotlib python experiments/phase25c_more_cas.py --no-wandb
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wmca.utils import pick_device

PLOTS_DIR = PROJECT_ROOT / "experiments" / "plots"
SEED = 42

# ── Rule 110 constants ──────────────────────────────────────────────────────
R110_WIDTH = 64
R110_N_TRAJ = 500
R110_TRAJ_LEN = 30
R110_DENSITY = 0.5

# ── Wireworld constants ─────────────────────────────────────────────────────
WW_H, WW_W = 16, 16
WW_N_TRAJ = 500
WW_TRAJ_LEN = 20
WW_CONDUCTOR_DENSITY = 0.30
WW_N_HEADS = 2  # avg electron heads per init

# ── Training ─────────────────────────────────────────────────────────────────
EPOCHS = 30
BATCH_SIZE = 64
LR = 1e-3

# ── Rollout horizons ─────────────────────────────────────────────────────────
ROLLOUT_HORIZONS = [1, 3, 5, 10]

# ── CML defaults (intentionally poor prior for discrete CAs) ────────────────
CML_STEPS = 15
CML_R = 3.90
CML_EPS = 0.3
CML_BETA = 0.15
RIDGE_ALPHA = 1.0


def _get_plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _get_ridge():
    from sklearn.linear_model import Ridge
    return Ridge


# ═══════════════════════════════════════════════════════════════════════════════
#  CA SIMULATORS
# ═══════════════════════════════════════════════════════════════════════════════

# ── Rule 110 ─────────────────────────────────────────────────────────────────
RULE_110_TABLE = {
    (1, 1, 1): 0, (1, 1, 0): 1, (1, 0, 1): 1, (1, 0, 0): 0,
    (0, 1, 1): 1, (0, 1, 0): 1, (0, 0, 1): 1, (0, 0, 0): 0,
}


def rule110_step(row: np.ndarray) -> np.ndarray:
    """One step of Rule 110 with wrap-around boundaries. row: (W,) binary."""
    W = len(row)
    out = np.zeros(W, dtype=np.float32)
    for i in range(W):
        left = int(row[(i - 1) % W])
        centre = int(row[i])
        right = int(row[(i + 1) % W])
        out[i] = RULE_110_TABLE[(left, centre, right)]
    return out


def generate_rule110(n_traj: int = R110_N_TRAJ,
                     traj_len: int = R110_TRAJ_LEN,
                     width: int = R110_WIDTH,
                     density: float = R110_DENSITY,
                     seed: int = SEED) -> np.ndarray:
    """Returns (n_traj, traj_len+1, 1, 1, width) float32.
    Shape: (N, T+1, C=1, H=1, W=64) for Conv2d compatibility."""
    rng = np.random.RandomState(seed)
    trajs = np.zeros((n_traj, traj_len + 1, 1, 1, width), dtype=np.float32)
    for i in range(n_traj):
        row = (rng.rand(width) < density).astype(np.float32)
        trajs[i, 0, 0, 0] = row
        for t in range(traj_len):
            row = rule110_step(row)
            trajs[i, t + 1, 0, 0] = row
    return trajs


# ── Wireworld ────────────────────────────────────────────────────────────────
# States: empty=0, head=1, tail=2, conductor=3

def wireworld_step(grid: np.ndarray) -> np.ndarray:
    """One Wireworld step.  grid: (H, W) int in {0,1,2,3}."""
    H, W = grid.shape
    out = np.zeros_like(grid)
    for y in range(H):
        for x in range(W):
            s = grid[y, x]
            if s == 0:
                out[y, x] = 0
            elif s == 1:
                out[y, x] = 2
            elif s == 2:
                out[y, x] = 3
            elif s == 3:
                # Count head neighbours (8-connected)
                heads = 0
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dy == 0 and dx == 0:
                            continue
                        ny, nx = (y + dy) % H, (x + dx) % W
                        if grid[ny, nx] == 1:
                            heads += 1
                out[y, x] = 1 if heads in (1, 2) else 3
    return out


def _onehot_ww(grid_int: np.ndarray) -> np.ndarray:
    """(H,W) int -> (4,H,W) float32 one-hot."""
    H, W = grid_int.shape
    oh = np.zeros((4, H, W), dtype=np.float32)
    for c in range(4):
        oh[c] = (grid_int == c).astype(np.float32)
    return oh


def generate_wireworld(n_traj: int = WW_N_TRAJ,
                       traj_len: int = WW_TRAJ_LEN,
                       H: int = WW_H, W: int = WW_W,
                       conductor_density: float = WW_CONDUCTOR_DENSITY,
                       n_heads: int = WW_N_HEADS,
                       seed: int = SEED) -> np.ndarray:
    """Returns (n_traj, traj_len+1, 4, H, W) float32 one-hot."""
    rng = np.random.RandomState(seed)
    trajs = np.zeros((n_traj, traj_len + 1, 4, H, W), dtype=np.float32)
    for i in range(n_traj):
        grid = np.zeros((H, W), dtype=np.int32)
        # place conductors
        conductor_mask = rng.rand(H, W) < conductor_density
        grid[conductor_mask] = 3
        # place electron heads on random conductor cells
        conductor_cells = list(zip(*np.where(grid == 3)))
        if len(conductor_cells) >= n_heads:
            head_idxs = rng.choice(len(conductor_cells), size=n_heads,
                                   replace=False)
            for idx in head_idxs:
                y, x = conductor_cells[idx]
                grid[y, x] = 1
        trajs[i, 0] = _onehot_ww(grid)
        for t in range(traj_len):
            grid = wireworld_step(grid)
            trajs[i, t + 1] = _onehot_ww(grid)
    return trajs


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def make_pairs(trajs: np.ndarray):
    """(N, T+1, C, H, W) -> X (N*T, C, H, W), Y (N*T, C, H, W)."""
    N, Tp1 = trajs.shape[:2]
    rest = trajs.shape[2:]
    X = trajs[:, :-1].reshape(-1, *rest)
    Y = trajs[:, 1:].reshape(-1, *rest)
    return X, Y


def split_trajs(trajs, fracs=(0.70, 0.15, 0.15)):
    n = len(trajs)
    n1 = int(fracs[0] * n)
    n2 = int(fracs[1] * n)
    return trajs[:n1], trajs[n1:n1 + n2], trajs[n1 + n2:]


# ═══════════════════════════════════════════════════════════════════════════════
#  MODELS
# ═══════════════════════════════════════════════════════════════════════════════

# ── PureNCA ──────────────────────────────────────────────────────────────────

class PureNCA_Binary(nn.Module):
    """PureNCA for binary (1-channel) CAs. Uses sigmoid output + BCE loss."""
    def __init__(self, in_ch: int = 1, hidden_ch: int = 16,
                 kernel_size: tuple = (3, 3), padding: tuple = (1, 1)):
        super().__init__()
        self.name = "PureNCA"
        self.perceive = nn.Conv2d(in_ch, hidden_ch,
                                  kernel_size, padding=padding)
        self.update = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(hidden_ch, in_ch, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.update(self.perceive(x))

    def param_count(self):
        return sum(p.numel() for p in self.parameters())


class PureNCA_Multi(nn.Module):
    """PureNCA for multi-class CAs.  Outputs raw logits (C channels)."""
    def __init__(self, n_classes: int = 4, hidden_ch: int = 32):
        super().__init__()
        self.name = "PureNCA"
        self.perceive = nn.Conv2d(n_classes, hidden_ch, 3, padding=1)
        self.update = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(hidden_ch, n_classes, 1),
        )

    def forward(self, x):
        return self.update(self.perceive(x))

    def param_count(self):
        return sum(p.numel() for p in self.parameters())


# ── ResidualCorrection(D) ────────────────────────────────────────────────────

class CML2D_Frozen(nn.Module):
    """Frozen CML2D (from hybrid.py pattern) adapted for variable channels."""
    def __init__(self, in_ch: int = 1, steps: int = CML_STEPS,
                 r: float = CML_R, eps: float = CML_EPS,
                 beta: float = CML_BETA, seed: int = 42,
                 kernel_size: tuple = (3, 3), padding: tuple = (1, 1)):
        super().__init__()
        self.in_ch = in_ch
        self.steps = steps
        self.register_buffer("r", torch.tensor(r))
        self.register_buffer("eps", torch.tensor(eps))
        self.register_buffer("beta", torch.tensor(beta))
        rng = torch.Generator().manual_seed(seed)
        K_raw = torch.rand(in_ch, 1, *kernel_size, generator=rng).abs()
        K_norm = K_raw / K_raw.sum(dim=(-1, -2), keepdim=True)
        self.register_buffer("K_local", K_norm)
        self._padding = padding

    def forward(self, drive):
        grid = drive
        r, eps, beta = self.r, self.eps, self.beta
        for _ in range(self.steps):
            mapped = r * grid * (1.0 - grid)
            local = F.conv2d(mapped, self.K_local, padding=self._padding,
                             groups=self.in_ch)
            physics = (1 - eps) * mapped + eps * local
            grid = (1 - beta) * physics + beta * drive
        return grid.clamp(1e-4, 1 - 1e-4)


class ResCorrection_Binary(nn.Module):
    """ResidualCorrection for binary CAs."""
    def __init__(self, in_ch: int = 1, hidden_ch: int = 16,
                 kernel_size: tuple = (3, 3), padding: tuple = (1, 1)):
        super().__init__()
        self.name = "ResCorrection(D)"
        self.cml = CML2D_Frozen(in_ch, kernel_size=kernel_size,
                                padding=padding)
        self.nca = nn.Sequential(
            nn.Conv2d(in_ch * 2, hidden_ch, kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv2d(hidden_ch, in_ch, 1),
        )

    def forward(self, x):
        cml_out = self.cml(x)
        correction = self.nca(torch.cat([x, cml_out], dim=1))
        return torch.clamp(cml_out + correction, 0, 1)

    def param_count(self):
        return sum(p.numel() for p in self.parameters())


class ResCorrection_Multi(nn.Module):
    """ResidualCorrection for multi-class CAs. Raw logits output."""
    def __init__(self, n_classes: int = 4, hidden_ch: int = 32):
        super().__init__()
        self.name = "ResCorrection(D)"
        self.cml = CML2D_Frozen(n_classes)
        self.nca = nn.Sequential(
            nn.Conv2d(n_classes * 2, hidden_ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_ch, n_classes, 1),
        )

    def forward(self, x):
        cml_out = self.cml(x)
        correction = self.nca(torch.cat([x, cml_out], dim=1))
        return cml_out + correction  # raw logits for CE loss

    def param_count(self):
        return sum(p.numel() for p in self.parameters())


# ── Conv baseline ────────────────────────────────────────────────────────────

class ConvBaseline_Binary(nn.Module):
    """Conv baseline for binary CAs."""
    def __init__(self, in_ch: int = 1, hidden_ch: int = 16,
                 kernel_size: tuple = (3, 3), padding: tuple = (1, 1)):
        super().__init__()
        self.name = "Conv"
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch, kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv2d(hidden_ch, hidden_ch, kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv2d(hidden_ch, in_ch, kernel_size, padding=padding),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)

    def param_count(self):
        return sum(p.numel() for p in self.parameters())


class ConvBaseline_Multi(nn.Module):
    """Conv baseline for multi-class CAs. Raw logits output."""
    def __init__(self, n_classes: int = 4, hidden_ch: int = 32):
        super().__init__()
        self.name = "Conv"
        self.net = nn.Sequential(
            nn.Conv2d(n_classes, hidden_ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_ch, hidden_ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_ch, n_classes, 3, padding=1),
        )

    def forward(self, x):
        return self.net(x)

    def param_count(self):
        return sum(p.numel() for p in self.parameters())


# ── CML2D + Ridge reservoir ─────────────────────────────────────────────────

class CML2DRidge:
    """CML2D reservoir with Ridge readout. Works for any channel count."""
    def __init__(self, in_ch: int = 1, H: int = 16, W: int = 16,
                 kernel_size: tuple = (3, 3), padding: tuple = (1, 1),
                 alpha: float = RIDGE_ALPHA, seed: int = SEED):
        self.name = "CML2D+Ridge"
        self.in_ch = in_ch
        self.H = H
        self.W = W
        self.cml = CML2D_Frozen(in_ch, kernel_size=kernel_size,
                                padding=padding, seed=seed)
        self.cml.eval()
        self.alpha = alpha
        self.ridge = None

    @torch.no_grad()
    def _features(self, X: np.ndarray) -> np.ndarray:
        """X: (N, C, H, W) -> (N, C*H*W) features."""
        X_t = torch.from_numpy(X).float()
        out = self.cml(X_t)
        return out.reshape(len(X), -1).numpy()

    def fit(self, X: np.ndarray, Y: np.ndarray):
        Ridge = _get_ridge()
        feats = self._features(X)
        Y_flat = Y.reshape(len(Y), -1)
        self.ridge = Ridge(alpha=self.alpha)
        self.ridge.fit(feats, Y_flat)

    def predict(self, X: np.ndarray) -> np.ndarray:
        feats = self._features(X)
        pred_flat = self.ridge.predict(feats)
        return pred_flat.reshape((-1,) + X.shape[1:])

    def predict_one(self, x: np.ndarray) -> np.ndarray:
        return self.predict(x[np.newaxis])[0]

    def param_count(self):
        if self.ridge is None:
            return 0
        return self.ridge.coef_.size + self.ridge.intercept_.size


# ═══════════════════════════════════════════════════════════════════════════════
#  TRAINING LOOPS
# ═══════════════════════════════════════════════════════════════════════════════

def train_binary_model(model, X_train, Y_train, X_val, Y_val,
                       epochs=EPOCHS, lr=LR, batch_size=BATCH_SIZE,
                       ca_name=""):
    """Train binary CA model with BCE loss.  X,Y: (N,C,H,W) float32."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    X_tr = torch.from_numpy(X_train).float()
    Y_tr = torch.from_numpy(Y_train).float()
    X_v = torch.from_numpy(X_val).float()
    Y_v = torch.from_numpy(Y_val).float()

    best_val = float("inf")
    best_state = None

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(X_tr))
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
            val_pred = model(X_v)
            val_loss = criterion(val_pred, Y_v).item()
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if (epoch + 1) % 10 == 0:
            print(f"    [{ca_name} {model.name}] Ep {epoch+1:3d}/{epochs}  "
                  f"train={total_loss/n_b:.6f}  val={val_loss:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def train_multiclass_model(model, X_train, Y_train_int, X_val, Y_val_int,
                           epochs=EPOCHS, lr=LR, batch_size=BATCH_SIZE,
                           ca_name=""):
    """Train multi-class CA model with CE loss.

    X: (N,C,H,W) one-hot float.  Y_int: (N,H,W) int class labels.
    Model outputs: (B,C,H,W) raw logits.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    X_tr = torch.from_numpy(X_train).float()
    Y_tr = torch.from_numpy(Y_train_int).long()
    X_v = torch.from_numpy(X_val).float()
    Y_v = torch.from_numpy(Y_val_int).long()

    best_val = float("inf")
    best_state = None

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(X_tr))
        total_loss, n_b = 0.0, 0
        for i in range(0, len(perm), batch_size):
            idx = perm[i:i + batch_size]
            logits = model(X_tr[idx])  # (B,C,H,W)
            loss = criterion(logits, Y_tr[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_b += 1

        model.eval()
        with torch.no_grad():
            val_logits = model(X_v)
            val_loss = criterion(val_logits, Y_v).item()
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if (epoch + 1) % 10 == 0:
            print(f"    [{ca_name} {model.name}] Ep {epoch+1:3d}/{epochs}  "
                  f"train={total_loss/n_b:.6f}  val={val_loss:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# ═══════════════════════════════════════════════════════════════════════════════
#  EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def binary_cell_accuracy(Y_true, Y_pred, threshold=0.5):
    pred_bin = (Y_pred >= threshold).astype(np.float32)
    return float(np.mean(pred_bin == Y_true))


def multiclass_cell_accuracy(Y_true_int, Y_pred_logits):
    """Y_true_int: (N,H,W) int.  Y_pred_logits: (N,C,H,W) float."""
    pred_cls = Y_pred_logits.argmax(axis=1)  # (N,H,W)
    return float(np.mean(pred_cls == Y_true_int))


def onehot_to_int(oh: np.ndarray) -> np.ndarray:
    """(N, C, H, W) one-hot -> (N, H, W) int."""
    return oh.argmax(axis=1)


# ── Rollout ──────────────────────────────────────────────────────────────────

def rollout_binary(predict_fn, x0, n_steps, shape_hw):
    """predict_fn: (C,H,W) -> (C,H,W).  Feed back binarised."""
    C = x0.shape[0]
    preds = np.zeros((n_steps, *x0.shape), dtype=np.float32)
    x = x0.copy()
    for t in range(n_steps):
        raw = predict_fn(x)
        preds[t] = raw
        x = (np.clip(raw, 0, 1) >= 0.5).astype(np.float32)
    return preds


def rollout_multiclass(predict_fn, x0_oh, n_steps, n_classes=4):
    """predict_fn: (C,H,W) one-hot -> (C,H,W) logits.
    Feed back argmax->one-hot."""
    preds = np.zeros((n_steps, *x0_oh.shape), dtype=np.float32)
    x = x0_oh.copy()
    for t in range(n_steps):
        logits = predict_fn(x)
        preds[t] = logits
        # argmax -> one-hot for feedback
        cls = logits.argmax(axis=0)  # (H,W)
        x = np.zeros_like(x0_oh)
        for c in range(n_classes):
            x[c] = (cls == c).astype(np.float32)
    return preds


def rollout_accuracy_binary(true_trajs, pred_seq, horizon):
    """Cell accuracy over first `horizon` steps."""
    pred_bin = (pred_seq[:horizon] >= 0.5).astype(np.float32)
    return float(np.mean(pred_bin == true_trajs[:horizon]))


def rollout_accuracy_multi(true_int_seq, pred_logits_seq, horizon):
    """Cell accuracy over first `horizon` steps.
    true_int_seq: (T,H,W), pred_logits_seq: (T,C,H,W)."""
    pred_cls = pred_logits_seq[:horizon].argmax(axis=1)  # (T,H,W)
    return float(np.mean(pred_cls == true_int_seq[:horizon]))


# ═══════════════════════════════════════════════════════════════════════════════
#  RULE 110 EXPERIMENT
# ═══════════════════════════════════════════════════════════════════════════════

def run_rule110():
    print("\n" + "=" * 72)
    print("RULE 110  (1D, binary, W=64, Turing complete)")
    print("=" * 72)

    # ── data ──
    print("\n  Generating Rule 110 trajectories ...")
    t0 = time.time()
    trajs = generate_rule110()  # (500, 31, 1, 64)
    train_t, val_t, test_t = split_trajs(trajs)
    X_train, Y_train = make_pairs(train_t)
    X_val, Y_val = make_pairs(val_t)
    X_test, Y_test = make_pairs(test_t)
    print(f"  Data: {len(trajs)} trajs x {R110_TRAJ_LEN} steps, "
          f"shape per frame = (1,1,{R110_WIDTH}), data shape={X_train.shape}")
    print(f"  Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}  "
          f"({time.time()-t0:.1f}s)")

    # Reshape for 2D convolutions: (N,1,1,64)
    # Use kernel (1,3), padding (0,1) for 1D neighbourhood
    ks = (1, 3)
    pad = (0, 1)

    results = {}

    # ── Model 1: ResidualCorrection(D) ──
    print("\n  [1/4] ResidualCorrection(D) ...")
    rc = ResCorrection_Binary(in_ch=1, hidden_ch=16,
                              kernel_size=ks, padding=pad)
    rc = train_binary_model(rc, X_train, Y_train, X_val, Y_val,
                            ca_name="R110")
    rc.eval()
    with torch.no_grad():
        rc_pred = rc(torch.from_numpy(X_test).float()).numpy()
    rc_acc = binary_cell_accuracy(Y_test, rc_pred)
    results["ResCorrection(D)"] = {"acc": rc_acc, "params": rc.param_count()}
    print(f"    Cell accuracy: {rc_acc:.4f}  params: {rc.param_count()}")

    # ── Model 2: PureNCA ──
    print("\n  [2/4] PureNCA ...")
    nca = PureNCA_Binary(in_ch=1, hidden_ch=16, kernel_size=ks, padding=pad)
    nca = train_binary_model(nca, X_train, Y_train, X_val, Y_val,
                             ca_name="R110")
    nca.eval()
    with torch.no_grad():
        nca_pred = nca(torch.from_numpy(X_test).float()).numpy()
    nca_acc = binary_cell_accuracy(Y_test, nca_pred)
    results["PureNCA"] = {"acc": nca_acc, "params": nca.param_count()}
    print(f"    Cell accuracy: {nca_acc:.4f}  params: {nca.param_count()}")

    # ── Model 3: Conv baseline ──
    print("\n  [3/4] Conv baseline ...")
    conv = ConvBaseline_Binary(in_ch=1, hidden_ch=16, kernel_size=ks,
                               padding=pad)
    conv = train_binary_model(conv, X_train, Y_train, X_val, Y_val,
                              ca_name="R110")
    conv.eval()
    with torch.no_grad():
        conv_pred = conv(torch.from_numpy(X_test).float()).numpy()
    conv_acc = binary_cell_accuracy(Y_test, conv_pred)
    results["Conv"] = {"acc": conv_acc, "params": conv.param_count()}
    print(f"    Cell accuracy: {conv_acc:.4f}  params: {conv.param_count()}")

    # ── Model 4: CML2D + Ridge ──
    print("\n  [4/4] CML2D + Ridge reservoir ...")
    ridge_m = CML2DRidge(in_ch=1, H=1, W=R110_WIDTH,
                         kernel_size=ks, padding=pad)
    ridge_m.fit(X_train, Y_train)
    ridge_pred = ridge_m.predict(X_test)
    ridge_acc = binary_cell_accuracy(Y_test, ridge_pred)
    results["CML2D+Ridge"] = {"acc": ridge_acc,
                              "params": ridge_m.param_count()}
    print(f"    Cell accuracy: {ridge_acc:.4f}  params: {ridge_m.param_count()}")

    # ── Rollout ──
    print("\n  Multi-step rollout ...")
    test_traj = test_t[0]  # (T+1, 1, W)
    x0 = test_traj[0]
    max_h = min(max(ROLLOUT_HORIZONS), R110_TRAJ_LEN)
    true_future = test_traj[1:max_h + 1]

    def _mk_pred_fn(model):
        def fn(x):
            model.eval()
            with torch.no_grad():
                return model(torch.from_numpy(x[np.newaxis]).float()
                             ).squeeze(0).numpy()
        return fn

    def _ridge_fn(x):
        return ridge_m.predict_one(x)

    rollout_res = {}
    model_rollouts = {}
    for name, pred_fn in [("ResCorrection(D)", _mk_pred_fn(rc)),
                          ("PureNCA", _mk_pred_fn(nca)),
                          ("Conv", _mk_pred_fn(conv)),
                          ("CML2D+Ridge", _ridge_fn)]:
        ro = rollout_binary(pred_fn, x0, max_h, (1, R110_WIDTH))
        model_rollouts[name] = ro

    for h in ROLLOUT_HORIZONS:
        if h > max_h:
            break
        row = {}
        for name, ro in model_rollouts.items():
            row[name] = rollout_accuracy_binary(true_future, ro, h)
        rollout_res[h] = row
        parts = "  ".join(f"{n}={v:.4f}" for n, v in row.items())
        print(f"    Horizon {h:>2d}: {parts}")

    return results, rollout_res, model_rollouts, true_future, x0


# ═══════════════════════════════════════════════════════════════════════════════
#  WIREWORLD EXPERIMENT
# ═══════════════════════════════════════════════════════════════════════════════

def run_wireworld():
    print("\n" + "=" * 72)
    print("WIREWORLD  (2D, 4-state, 16x16)")
    print("=" * 72)

    # ── data ──
    print("\n  Generating Wireworld trajectories ...")
    t0 = time.time()
    trajs = generate_wireworld()  # (500, 21, 4, 16, 16)
    train_t, val_t, test_t = split_trajs(trajs)
    X_train, Y_train = make_pairs(train_t)
    X_val, Y_val = make_pairs(val_t)
    X_test, Y_test = make_pairs(test_t)

    # int labels for CE loss
    Y_train_int = onehot_to_int(Y_train)
    Y_val_int = onehot_to_int(Y_val)
    Y_test_int = onehot_to_int(Y_test)

    print(f"  Data: {len(trajs)} trajs x {WW_TRAJ_LEN} steps, "
          f"shape per frame = (4,{WW_H},{WW_W})")
    print(f"  Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}  "
          f"({time.time()-t0:.1f}s)")

    results = {}

    # ── Model 1: ResidualCorrection(D) ──
    print("\n  [1/4] ResidualCorrection(D) ...")
    rc = ResCorrection_Multi(n_classes=4, hidden_ch=32)
    rc = train_multiclass_model(rc, X_train, Y_train_int, X_val, Y_val_int,
                                ca_name="WW")
    rc.eval()
    with torch.no_grad():
        rc_logits = rc(torch.from_numpy(X_test).float()).numpy()
    rc_acc = multiclass_cell_accuracy(Y_test_int, rc_logits)
    results["ResCorrection(D)"] = {"acc": rc_acc, "params": rc.param_count()}
    print(f"    Cell accuracy: {rc_acc:.4f}  params: {rc.param_count()}")

    # ── Model 2: PureNCA ──
    print("\n  [2/4] PureNCA ...")
    nca = PureNCA_Multi(n_classes=4, hidden_ch=32)
    nca = train_multiclass_model(nca, X_train, Y_train_int, X_val, Y_val_int,
                                 ca_name="WW")
    nca.eval()
    with torch.no_grad():
        nca_logits = nca(torch.from_numpy(X_test).float()).numpy()
    nca_acc = multiclass_cell_accuracy(Y_test_int, nca_logits)
    results["PureNCA"] = {"acc": nca_acc, "params": nca.param_count()}
    print(f"    Cell accuracy: {nca_acc:.4f}  params: {nca.param_count()}")

    # ── Model 3: Conv baseline ──
    print("\n  [3/4] Conv baseline ...")
    conv = ConvBaseline_Multi(n_classes=4, hidden_ch=32)
    conv = train_multiclass_model(conv, X_train, Y_train_int, X_val, Y_val_int,
                                  ca_name="WW")
    conv.eval()
    with torch.no_grad():
        conv_logits = conv(torch.from_numpy(X_test).float()).numpy()
    conv_acc = multiclass_cell_accuracy(Y_test_int, conv_logits)
    results["Conv"] = {"acc": conv_acc, "params": conv.param_count()}
    print(f"    Cell accuracy: {conv_acc:.4f}  params: {conv.param_count()}")

    # ── Model 4: CML2D + Ridge ──
    print("\n  [4/4] CML2D + Ridge reservoir ...")
    ridge_m = CML2DRidge(in_ch=4, H=WW_H, W=WW_W)
    ridge_m.fit(X_train, Y_train)  # Y as one-hot for ridge
    ridge_pred = ridge_m.predict(X_test)  # (N,4,H,W)
    ridge_acc = multiclass_cell_accuracy(Y_test_int, ridge_pred)
    results["CML2D+Ridge"] = {"acc": ridge_acc,
                              "params": ridge_m.param_count()}
    print(f"    Cell accuracy: {ridge_acc:.4f}  params: {ridge_m.param_count()}")

    # ── Rollout ──
    print("\n  Multi-step rollout ...")
    test_traj = test_t[0]  # (T+1, 4, H, W) one-hot
    x0_oh = test_traj[0]   # (4,H,W)
    max_h = min(max(ROLLOUT_HORIZONS), WW_TRAJ_LEN)
    true_future_oh = test_traj[1:max_h + 1]  # (T,4,H,W)
    true_future_int = onehot_to_int(true_future_oh)  # (T,H,W)

    def _mk_pred_fn(model):
        def fn(x):
            model.eval()
            with torch.no_grad():
                return model(torch.from_numpy(x[np.newaxis]).float()
                             ).squeeze(0).numpy()
        return fn

    def _ridge_fn(x):
        return ridge_m.predict_one(x)

    rollout_res = {}
    model_rollouts = {}
    for name, pred_fn in [("ResCorrection(D)", _mk_pred_fn(rc)),
                          ("PureNCA", _mk_pred_fn(nca)),
                          ("Conv", _mk_pred_fn(conv)),
                          ("CML2D+Ridge", _ridge_fn)]:
        ro = rollout_multiclass(pred_fn, x0_oh, max_h, n_classes=4)
        model_rollouts[name] = ro

    for h in ROLLOUT_HORIZONS:
        if h > max_h:
            break
        row = {}
        for name, ro in model_rollouts.items():
            row[name] = rollout_accuracy_multi(true_future_int,
                                               ro, h)
        rollout_res[h] = row
        parts = "  ".join(f"{n}={v:.4f}" for n, v in row.items())
        print(f"    Horizon {h:>2d}: {parts}")

    return results, rollout_res, model_rollouts, true_future_int, x0_oh


# ═══════════════════════════════════════════════════════════════════════════════
#  PLOTTING
# ═══════════════════════════════════════════════════════════════════════════════

MODEL_COLORS = {
    "ResCorrection(D)": "tab:red",
    "PureNCA": "tab:green",
    "Conv": "tab:blue",
    "CML2D+Ridge": "tab:orange",
}
MODEL_MARKERS = {
    "ResCorrection(D)": "s",
    "PureNCA": "o",
    "Conv": "D",
    "CML2D+Ridge": "^",
}
MODEL_ORDER = ["ResCorrection(D)", "PureNCA", "Conv", "CML2D+Ridge"]


def plot_summary(r110_results, r110_rollout, ww_results, ww_rollout):
    plt = _get_plt()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ── Row 0: Rule 110 ──
    # Bar chart
    ax = axes[0, 0]
    names = MODEL_ORDER
    accs = [r110_results[n]["acc"] for n in names]
    colors = [MODEL_COLORS[n] for n in names]
    x_pos = np.arange(len(names))
    ax.bar(x_pos, accs, color=colors, alpha=0.85)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, fontsize=8, rotation=15)
    ax.set_ylabel("Cell Accuracy")
    ax.set_title("Rule 110 -- 1-Step Cell Accuracy")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")
    for i, v in enumerate(accs):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)

    # Rollout
    ax = axes[0, 1]
    horizons = sorted(r110_rollout.keys())
    for name in MODEL_ORDER:
        vals = [r110_rollout[h][name] for h in horizons]
        ax.plot(horizons, vals, f"{MODEL_MARKERS[name]}-",
                color=MODEL_COLORS[name], label=name, markersize=7)
    ax.set_xlabel("Rollout horizon")
    ax.set_ylabel("Cell Accuracy")
    ax.set_title("Rule 110 -- Multi-step Rollout")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.3, 1.02)

    # ── Row 1: Wireworld ──
    ax = axes[1, 0]
    accs = [ww_results[n]["acc"] for n in names]
    ax.bar(x_pos, accs, color=colors, alpha=0.85)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, fontsize=8, rotation=15)
    ax.set_ylabel("Cell Accuracy")
    ax.set_title("Wireworld -- 1-Step Cell Accuracy")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")
    for i, v in enumerate(accs):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)

    ax = axes[1, 1]
    horizons = sorted(ww_rollout.keys())
    for name in MODEL_ORDER:
        vals = [ww_rollout[h][name] for h in horizons]
        ax.plot(horizons, vals, f"{MODEL_MARKERS[name]}-",
                color=MODEL_COLORS[name], label=name, markersize=7)
    ax.set_xlabel("Rollout horizon")
    ax.set_ylabel("Cell Accuracy")
    ax.set_title("Wireworld -- Multi-step Rollout")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.3, 1.02)

    fig.suptitle("Phase 2.5c: PureNCA Generalisation to Discrete CAs",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "phase25c_summary.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Summary plot saved to {PLOTS_DIR / 'phase25c_summary.png'}")


def plot_rule110_rollout(model_rollouts, true_future, x0):
    """Visualise Rule 110 rollout as space-time diagrams.
    true_future: (T, 1, 1, W), x0: (1, 1, W)."""
    plt = _get_plt()
    n_models = len(MODEL_ORDER)
    max_h = true_future.shape[0]

    # Build space-time images: each row = one timestep, shape (1, W)
    fig, axes = plt.subplots(1, n_models + 1, figsize=(3 * (n_models + 1), 4))

    # True -- squeeze to (W,) rows
    st_true = np.vstack([x0[0, 0]] + [true_future[t, 0, 0]
                                       for t in range(max_h)])
    axes[0].imshow(st_true, cmap="binary", aspect="auto",
                   interpolation="nearest")
    axes[0].set_title("True", fontsize=10)
    axes[0].set_ylabel("Time step")
    axes[0].set_xlabel("Cell")

    for i, name in enumerate(MODEL_ORDER):
        ro = model_rollouts[name]
        st = np.vstack([x0[0, 0]] + [(ro[t, 0, 0] >= 0.5).astype(float)
                                      for t in range(max_h)])
        axes[i + 1].imshow(st, cmap="binary", aspect="auto",
                           interpolation="nearest")
        axes[i + 1].set_title(name, fontsize=10)
        axes[i + 1].set_xlabel("Cell")

    fig.suptitle("Rule 110 Space-Time Diagrams", fontsize=12)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "phase25c_rule110_spacetime.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  Rule 110 space-time plot saved.")


def plot_wireworld_rollout(model_rollouts, true_future_int, x0_oh):
    """Visualise Wireworld rollout snapshots."""
    plt = _get_plt()
    # Show timesteps 0, 2, 5, 9
    show_t = [t for t in [0, 2, 5, 9] if t < true_future_int.shape[0]]
    n_cols = len(MODEL_ORDER) + 1
    n_rows = len(show_t)

    # Colour map: empty=white, head=blue, tail=red, conductor=gray
    from matplotlib.colors import ListedColormap
    ww_cmap = ListedColormap(["white", "blue", "red", "gray"])

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.5 * n_cols,
                                                       2.5 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["True"] + MODEL_ORDER

    for row_i, t in enumerate(show_t):
        # True
        ax = axes[row_i, 0]
        ax.imshow(true_future_int[t], cmap=ww_cmap, vmin=0, vmax=3,
                  interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        if row_i == 0:
            ax.set_title("True", fontsize=10)
        ax.set_ylabel(f"t+{t+1}", fontsize=10)

        for col_i, name in enumerate(MODEL_ORDER):
            ax = axes[row_i, col_i + 1]
            logits = model_rollouts[name][t]  # (4,H,W)
            cls = logits.argmax(axis=0)
            ax.imshow(cls, cmap=ww_cmap, vmin=0, vmax=3,
                      interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if row_i == 0:
                ax.set_title(name, fontsize=9)

    fig.suptitle("Wireworld Predictions", fontsize=12)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "phase25c_wireworld_snapshots.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  Wireworld snapshot plot saved.")


# ═══════════════════════════════════════════════════════════════════════════════
#  SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

def print_summary(r110_results, r110_rollout, ww_results, ww_rollout):
    print("\n" + "=" * 80)
    print("PHASE 2.5c SUMMARY: PureNCA GENERALISATION TO DISCRETE CAs")
    print("=" * 80)

    for ca_name, results, rollout in [("Rule 110", r110_results, r110_rollout),
                                       ("Wireworld", ww_results, ww_rollout)]:
        print(f"\n--- {ca_name} ---")
        print(f"  {'Model':<20s}  {'1-Step Acc':>10s}  {'Params':>10s}")
        print(f"  {'-'*45}")
        for name in MODEL_ORDER:
            r = results[name]
            print(f"  {name:<20s}  {r['acc']:10.4f}  {r['params']:10d}")

        print(f"\n  Multi-step rollout:")
        header = f"  {'Horizon':>8s}"
        for name in MODEL_ORDER:
            header += f"  {name:>16s}"
        print(header)
        for h in sorted(rollout.keys()):
            row = f"  {h:8d}"
            for name in MODEL_ORDER:
                row += f"  {rollout[h][name]:16.4f}"
            print(row)

    # Matching Principle verdict
    print("\n" + "=" * 80)
    print("MATCHING PRINCIPLE VERDICT")
    print("=" * 80)
    for ca_name, results in [("Rule 110", r110_results),
                              ("Wireworld", ww_results)]:
        nca = results["PureNCA"]["acc"]
        rc = results["ResCorrection(D)"]["acc"]
        ridge = results["CML2D+Ridge"]["acc"]
        conv_a = results["Conv"]["acc"]
        print(f"\n  {ca_name}:")
        print(f"    PureNCA={nca:.4f}  ResCorr={rc:.4f}  "
              f"Conv={conv_a:.4f}  CML+Ridge={ridge:.4f}")
        if nca >= rc and nca >= ridge:
            print(f"    -> PureNCA wins. CML prior hurts. "
                  f"Matching Principle CONFIRMED.")
        else:
            winner = max(results.items(), key=lambda kv: kv[1]["acc"])[0]
            print(f"    -> {winner} wins. Matching Principle result: "
                  f"{'CONFIRMED' if winner in ('PureNCA', 'Conv') else 'VIOLATED'}.")

    print("=" * 80)


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Phase 2.5c: PureNCA on Rule 110 + Wireworld")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("FORCE_CPU", "1")
    device = pick_device()

    print("=" * 72)
    print("PHASE 2.5c: PureNCA GENERALISATION — RULE 110 + WIREWORLD")
    print("=" * 72)

    r110_results, r110_rollout, r110_model_ro, r110_true, r110_x0 = \
        run_rule110()
    ww_results, ww_rollout, ww_model_ro, ww_true_int, ww_x0 = \
        run_wireworld()

    print("\n  Generating plots ...")
    plot_summary(r110_results, r110_rollout, ww_results, ww_rollout)
    plot_rule110_rollout(r110_model_ro, r110_true, r110_x0)
    plot_wireworld_rollout(ww_model_ro, ww_true_int, ww_x0)

    print_summary(r110_results, r110_rollout, ww_results, ww_rollout)


if __name__ == "__main__":
    main()
