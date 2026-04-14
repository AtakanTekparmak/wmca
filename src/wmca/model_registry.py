"""Model registry and training pipeline for unified ablation."""
from __future__ import annotations

import gc
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from wmca.modules.hybrid import (
    CML2D,
    CMLRegularizedNCA,
    DeepResCorGated,
    DeepResCorLite,
    GatedBlendWM,
    MatchingPrincipleGateWM,
    MoERFHomogeneousWorldModel,
    MoERFWorldModel,
    NCAInsideCML,
    PureNCA,
    ResidualCorrectionWM,
    ResidualCorrectionWMv2,
    ResidualCorrectionWMv3,
    ResidualCorrectionWMv6,
    ResidualCorrectionWMv7,
    ResidualCorrectionWMv8,
    ResidualCorrectionWMv9,
    TrajectoryAttentionWM,
)


# ===== Baseline Models ======================================================

class Conv2DBaseline(nn.Module):
    """3-layer CNN baseline."""

    def __init__(self, in_channels: int = 1, out_channels: int = 1,
                 hidden_ch: int = 16, use_sigmoid: bool = True):
        super().__init__()
        self.use_sigmoid = use_sigmoid
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_ch, hidden_ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_ch, out_channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        if self.use_sigmoid:
            out = torch.sigmoid(out)
        return out

    def param_count(self) -> dict[str, int]:
        return {
            "trained": sum(p.numel() for p in self.parameters()),
            "frozen": 0,
        }


class MLPBaseline(nn.Module):
    """MLP baseline that flattens spatial dims."""

    def __init__(self, in_channels: int = 1, out_channels: int = 1,
                 grid_h: int = 16, grid_w: int = 16,
                 hidden_dim: int = 256, use_sigmoid: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.use_sigmoid = use_sigmoid

        flat_in = in_channels * grid_h * grid_w
        flat_out = out_channels * grid_h * grid_w

        self.net = nn.Sequential(
            nn.Linear(flat_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, flat_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        out = self.net(x.reshape(B, -1))
        out = out.reshape(B, self.out_channels, self.grid_h, self.grid_w)
        if self.use_sigmoid:
            out = torch.sigmoid(out)
        return out

    def param_count(self) -> dict[str, int]:
        return {
            "trained": sum(p.numel() for p in self.parameters()),
            "frozen": 0,
        }


class CML2DRidge:
    """CML2D (frozen) + sklearn Ridge readout. Not an nn.Module."""

    def __init__(self, in_channels: int = 1, out_channels: int = 1,
                 cml_steps: int = 15, r: float = 3.90, eps: float = 0.3,
                 beta: float = 0.15, seed: int = 42):
        self.cml = CML2D(in_channels=in_channels, steps=cml_steps,
                         r=r, eps=eps, beta=beta, seed=seed)
        self.cml.eval()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ridge = None

    def _features(self, X: np.ndarray) -> np.ndarray:
        """X: (N, C, H, W) -> (N, C*H*W) CML features."""
        X_t = torch.from_numpy(X).float()
        with torch.no_grad():
            out = self.cml(X_t)
        return out.reshape(len(X), -1).numpy()

    def fit(self, X_train: np.ndarray, Y_train: np.ndarray, alpha: float = 1.0):
        from sklearn.linear_model import Ridge
        feats = self._features(X_train)
        Y_flat = Y_train.reshape(len(Y_train), -1)
        self.ridge = Ridge(alpha=alpha)
        self.ridge.fit(feats, Y_flat)

    def predict(self, X: np.ndarray) -> np.ndarray:
        feats = self._features(X)
        pred_flat = self.ridge.predict(feats)
        shape = (len(X),) + X.shape[1:]
        return pred_flat.reshape(shape).clip(0, 1).astype(np.float32)

    def predict_one(self, x: np.ndarray) -> np.ndarray:
        return self.predict(x[np.newaxis])[0]

    def param_count(self) -> dict[str, int]:
        if self.ridge is None:
            return {"trained": 0, "frozen": 0}
        ridge_params = self.ridge.coef_.size + self.ridge.intercept_.size
        return {
            "trained": ridge_params,
            "frozen": sum(b.numel() for b in self.cml.buffers()),
        }


# ===== Registry ==============================================================

MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "rescor": {
        "class": ResidualCorrectionWM,
        "description": "CML base + NCA correction",
    },
    "rescor_e2": {
        "class": ResidualCorrectionWMv2,
        "description": "ResCor + multi-stat CML readouts (E2)",
    },
    "rescor_e3": {
        "class": ResidualCorrectionWMv7,
        "description": "ResCor + E2 multi-stat + E3 dilated NCA (multi-scale RF)",
    },
    "rescor_e3b": {
        "class": ResidualCorrectionWMv8,
        "description": "ResCor + E2 multi-stat + E3b zero-init residual dilation",
    },
    "rescor_e3c": {
        "class": ResidualCorrectionWMv9,
        "description": "ResCor + E2 multi-stat + E3c (E3b + strong L2 WD on alpha)",
    },
    "rescor_traj_attn": {
        "class": TrajectoryAttentionWM,
        "description": "ResCor + hybrid hand-crafted/learned trajectory attention",
    },
    "rescor_mp_gate": {
        "class": MatchingPrincipleGateWM,
        "description": "Matching-Principle Gate: per-cell trust between CML and NCA paths",
    },
    "rescor_moe_rf": {
        "class": MoERFWorldModel,
        "description": "MoE-RF: per-cell CML-stat routing between d=1/d=2 experts",
    },
    "rescor_moe_homo": {
        "class": MoERFHomogeneousWorldModel,
        "description": "MoE-RF-Homo: ablation with K=2 same-arch (d=1) experts",
    },
    "rescor_deep_lite": {
        "class": DeepResCorLite,
        "description": "DeepResCor-Lite: 2-layer residual correction (no spatial gate)",
    },
    "rescor_deep_gated": {
        "class": DeepResCorGated,
        "description": "DeepResCor-Gated: 2-layer + CML-var spatial gate",
    },
    "rescor_e4": {
        "class": ResidualCorrectionWMv3,
        "description": "ResCor + E2 multi-stat + E4 per-channel affine drive",
    },
    "rescor_e6": {
        "class": ResidualCorrectionWMv6,
        "description": "ResCor + E1 multi-r groups + E2 multi-stat + E6 per-group correction",
    },
    "pure_nca": {
        "class": PureNCA,
        "description": "Learned NCA only",
    },
    "nca_inside_cml": {
        "class": NCAInsideCML,
        "description": "NCA replaces logistic map in CML",
    },
    "gated_blend": {
        "class": GatedBlendWM,
        "description": "Per-cell gate blends CML+NCA",
    },
    "cml_reg": {
        "class": CMLRegularizedNCA,
        "description": "NCA with CML regularization",
    },
    "conv2d": {
        "class": Conv2DBaseline,
        "description": "3-layer CNN baseline",
    },
    "mlp": {
        "class": MLPBaseline,
        "description": "MLP baseline",
    },
    "cml_ridge": {
        "class": CML2DRidge,
        "description": "Fixed CML + Ridge readout (not nn.Module)",
    },
}


def create_model(name: str, in_channels: int = 1, out_channels: int = 1,
                 grid_size: int = 16, grid_h: int | None = None,
                 grid_w: int | None = None, seed: int = 42,
                 **kwargs) -> nn.Module | CML2DRidge:
    """Factory function. Returns a model ready to train.

    Supported names: rescor, pure_nca, nca_inside_cml, gated_blend,
                     cml_reg, conv2d, mlp, cml_ridge

    ``grid_size``, ``grid_h``/``grid_w``, and ``seed`` are accepted for
    convenience but only forwarded to models that actually need them.
    """
    if grid_h is None:
        grid_h = grid_size
    if grid_w is None:
        grid_w = grid_size

    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )

    entry = MODEL_REGISTRY[name]
    cls = entry["class"]

    if name == "conv2d":
        # Conv2DBaseline uses use_sigmoid — disable for CE tasks
        use_sigmoid = out_channels == in_channels
        return cls(in_channels=in_channels, out_channels=out_channels,
                   use_sigmoid=use_sigmoid)

    if name == "mlp":
        use_sigmoid = out_channels == in_channels
        return cls(in_channels=in_channels, out_channels=out_channels,
                   grid_h=grid_h, grid_w=grid_w, use_sigmoid=use_sigmoid)

    if name == "cml_ridge":
        return cls(in_channels=in_channels, out_channels=out_channels,
                   seed=seed)

    # Hybrid models: all now accept in_channels, out_channels, seed,
    # and use_sigmoid. Cross-entropy tasks (out_ch != in_ch, or
    # identified by the caller) need raw logits, so use_sigmoid is
    # disabled whenever out_ch != in_ch.
    import inspect
    sig = inspect.signature(cls.__init__)
    kwargs_out: dict[str, Any] = {"in_channels": in_channels}
    if "out_channels" in sig.parameters:
        kwargs_out["out_channels"] = out_channels
    if "use_sigmoid" in sig.parameters:
        # For action-conditioned / classification tasks the output must
        # be logits (not squashed by sigmoid/clamp), otherwise CE loss
        # collapses to the majority class.
        kwargs_out["use_sigmoid"] = (out_channels == in_channels)
    if "seed" in sig.parameters:
        kwargs_out["seed"] = seed
    return cls(**kwargs_out)


# ===== Training ==============================================================

def _ensure_tensor(arr, dev: torch.device) -> torch.Tensor:
    """Convert numpy array or torch tensor to float tensor on *dev*."""
    if isinstance(arr, torch.Tensor):
        return arr.float().to(dev)
    return torch.from_numpy(np.asarray(arr)).float().to(dev)


def train_model(
    model: nn.Module,
    X_train,
    Y_train,
    X_val=None,
    Y_val=None,
    loss_type: str = "mse",
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str | torch.device = "cpu",
    cml_reg_lambda: float = 0.1,
    # Extra kwargs accepted (and ignored) for runner convenience
    benchmark_name: str | None = None,
    model_name: str | None = None,
) -> nn.Module:
    """Generic training loop. Handles MSE, BCE, and cross-entropy losses.

    For CMLRegularizedNCA, adds the regularization term automatically.
    Returns the trained model (best val checkpoint restored).
    """
    dev = torch.device(device) if isinstance(device, str) else device
    model = model.to(dev)

    # Split optimizer into two param groups so we can apply strong L2
    # weight decay selectively to "alpha" params (dilation gates, depth
    # gates, router weights).  Models that expose ``get_alpha_params()``
    # declare exactly which params should be penalised; for older models
    # without the method we fall back to string matching on param names.
    alpha_params: list[nn.Parameter] = []
    other_params: list[nn.Parameter] = []

    if hasattr(model, "get_alpha_params"):
        alpha_ids = {id(p) for p in model.get_alpha_params()}
        for p in model.parameters():
            if not p.requires_grad:
                continue
            (alpha_params if id(p) in alpha_ids else other_params).append(p)
    else:
        for pname, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if "dilation_alpha" in pname:
                alpha_params.append(p)
            else:
                other_params.append(p)

    if alpha_params:
        optimizer = torch.optim.Adam(
            [
                {"params": other_params, "weight_decay": 0.0},
                {"params": alpha_params, "weight_decay": 1.0},  # strong L2 on alpha
            ],
            lr=lr,
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)

    is_cml_reg = isinstance(model, CMLRegularizedNCA)

    if loss_type == "mse":
        criterion = nn.MSELoss()
    elif loss_type == "bce":
        criterion = nn.BCELoss()
    elif loss_type in ("ce", "cross_entropy"):
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unknown loss_type '{loss_type}'")

    X_tr = _ensure_tensor(X_train, dev)
    Y_tr = _ensure_tensor(Y_train, dev)
    if X_val is not None and Y_val is not None:
        X_v = _ensure_tensor(X_val, dev)
        Y_v = _ensure_tensor(Y_val, dev)
    else:
        # Fall back to using a slice of training data for validation
        n_val = max(1, len(X_tr) // 5)
        X_v = X_tr[-n_val:]
        Y_v = Y_tr[-n_val:]

    is_ce = loss_type in ("ce", "cross_entropy")

    def _ce_target(y: torch.Tensor) -> torch.Tensor:
        """Convert one-hot (N, C, H, W) -> class indices (N, H, W) for CE."""
        if y.dim() == 4 and y.shape[1] > 1:
            return y.argmax(dim=1)  # (N, H, W) long
        return y.long().squeeze(1)

    best_val_loss = float("inf")
    best_state: dict | None = None
    train_losses: list[float] = []
    val_losses: list[float] = []

    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(X_tr), device=dev)
        total_loss = 0.0
        n_batches = 0

        for i in range(0, len(perm), batch_size):
            idx = perm[i : i + batch_size]
            xb, yb = X_tr[idx], Y_tr[idx]

            if is_cml_reg:
                nca_out, cml_ref = model(xb)
                if is_ce:
                    pred_loss = criterion(nca_out, _ce_target(yb))
                    # For CE: nca_out is logits, cml_ref is in [0,1].
                    # Regularize softmax(nca_out) toward cml_ref so both
                    # are in the same [0,1] range.
                    nca_probs = torch.softmax(nca_out, dim=1)
                    reg_loss = F.mse_loss(nca_probs, cml_ref.detach())
                else:
                    pred_loss = criterion(nca_out, yb)
                    reg_loss = F.mse_loss(nca_out, cml_ref.detach())
                loss = pred_loss + cml_reg_lambda * reg_loss
            else:
                pred = model(xb)
                if is_ce:
                    loss = criterion(pred, _ce_target(yb))
                else:
                    loss = criterion(pred, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        avg_train = total_loss / max(n_batches, 1)
        train_losses.append(avg_train)

        # Validation
        model.eval()
        with torch.no_grad():
            val_sum = 0.0
            val_n = 0
            for vi in range(0, len(X_v), batch_size):
                vx = X_v[vi : vi + batch_size]
                vy = Y_v[vi : vi + batch_size]
                vp = model(vx)
                if is_ce:
                    vl = criterion(vp, _ce_target(vy)).item()
                else:
                    vl = criterion(vp, vy).item()
                val_sum += vl * len(vx)
                val_n += len(vx)
            val_loss = val_sum / max(val_n, 1)

        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    train_time = time.time() - t0

    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to(torch.device("cpu")).eval()

    del X_tr, Y_tr, X_v, Y_v
    gc.collect()

    return model


def train_ridge_model(
    cml_ridge: CML2DRidge,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    alpha: float = 1.0,
) -> tuple[CML2DRidge, dict[str, Any]]:
    """Special case for CML + Ridge readout.

    X_train, Y_train: arrays with shape (N, C, H, W).
    Returns (fitted model, stats).
    """
    t0 = time.time()
    cml_ridge.fit(X_train, Y_train, alpha=alpha)
    train_time = time.time() - t0

    stats = {
        "train_time": train_time,
        "alpha": alpha,
    }
    return cml_ridge, stats


# ===== Evaluation ============================================================

def evaluate_model(
    model: nn.Module,
    X_test,
    Y_test,
    loss_type: str = "mse",
    device: str | torch.device = "cpu",
    batch_size: int = 256,
    benchmark_name: str | None = None,
) -> dict[str, float]:
    """1-step evaluation. Returns dict of metric name -> value.

    MSE/BCE: returns the loss value.
    For binary tasks (bce), also computes cell accuracy.
    """
    dev = torch.device(device) if isinstance(device, str) else device
    model = model.to(dev).eval()

    X_t = _ensure_tensor(X_test, dev)
    Y_t = _ensure_tensor(Y_test, dev)

    if loss_type == "mse":
        criterion = nn.MSELoss()
    elif loss_type == "bce":
        criterion = nn.BCELoss()
    elif loss_type in ("ce", "cross_entropy"):
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unknown loss_type '{loss_type}'")

    is_ce = loss_type in ("ce", "cross_entropy")

    def _ce_target(y: torch.Tensor) -> torch.Tensor:
        if y.dim() == 4 and y.shape[1] > 1:
            return y.argmax(dim=1)
        return y.long().squeeze(1)

    total = 0.0
    correct = 0
    total_cells = 0
    n = 0
    with torch.no_grad():
        for i in range(0, len(X_t), batch_size):
            xb = X_t[i : i + batch_size]
            yb = Y_t[i : i + batch_size]
            pb = model(xb)
            if is_ce:
                yb_idx = _ce_target(yb)
                loss_val = criterion(pb, yb_idx).item()
                pred_cls = pb.argmax(dim=1)
                correct += (pred_cls == yb_idx).sum().item()
                total_cells += pred_cls.numel()
            else:
                loss_val = criterion(pb, yb).item()
                if loss_type == "bce":
                    pred_bin = (pb >= 0.5).float()
                    correct += (pred_bin == yb).sum().item()
                    total_cells += yb.numel()
            total += loss_val * len(xb)
            n += len(xb)

    metric_val = total / max(n, 1)

    if loss_type == "mse":
        return {"mse": metric_val}
    elif loss_type == "bce":
        acc = correct / max(total_cells, 1)
        return {"accuracy": acc, "bce": metric_val}
    else:
        acc = correct / max(total_cells, 1)
        return {"accuracy": acc, "ce": metric_val}


def evaluate_rollout(
    model: nn.Module,
    data,
    horizons: list[int] | None = None,
    benchmark_name: str | None = None,
    device: str | torch.device = "cpu",
) -> dict[int, float]:
    """Multi-step autoregressive rollout evaluation.

    Constructs a pseudo-trajectory from test data (consecutive X->Y pairs)
    and autoregressively rolls the model forward.

    Args:
        model: Trained nn.Module.
        data: BenchmarkData namedtuple (uses X_test, Y_test, meta).
        horizons: List of rollout horizons to evaluate.
        benchmark_name: Optional name for determining loss/binarize settings.
        device: Torch device string or torch.device.

    Returns:
        dict of {horizon: metric_value}.
    """
    if horizons is None:
        horizons = [1, 3, 5, 10]

    dev = torch.device(device) if isinstance(device, str) else device
    model = model.to(dev).eval()

    # Determine settings from meta
    meta = getattr(data, "meta", {}) or {}
    loss_type = meta.get("loss_type", "mse")
    binarize = loss_type in ("bce",)

    # Build a pseudo-trajectory from the first max_horizon+1 test samples
    # X_test[i] -> Y_test[i] are consecutive pairs
    max_h = max(horizons)
    X_test = data.X_test
    Y_test = data.Y_test

    # Handle action-conditioned data (input/output channels differ).
    # For benchmarks like grid_world, X = state+action_field (8ch) and
    # Y = next_state (4ch), so a naive pseudo-trajectory can't be built from
    # consecutive X->Y pairs. Skip rollout cleanly with a marker dict.
    x_ch = X_test.shape[1] if X_test.ndim >= 2 else None
    y_ch = Y_test.shape[1] if Y_test.ndim >= 2 else None
    if (meta.get("action_conditioned", False)
            or (x_ch is not None and y_ch is not None and x_ch != y_ch)):
        return {
            "action_conditioned": True,
            "skipped_reason": (
                f"Input channels {x_ch} != output channels {y_ch} "
                "(action-conditioned; rollout not applicable)"
            ),
        }

    # Convert to numpy for rollout
    if isinstance(X_test, torch.Tensor):
        X_np = X_test.cpu().numpy()
        Y_np = Y_test.cpu().numpy()
    else:
        X_np = np.asarray(X_test)
        Y_np = np.asarray(Y_test)

    # Use first sample as start, then ground truth Y for comparison
    # trajectory = [X[0], Y[0], Y[1], ..., Y[max_h-1]]
    n_available = min(len(X_np), max_h)
    trajectory = [X_np[0]]
    for i in range(n_available):
        trajectory.append(Y_np[i])
    test_trajectory = np.stack(trajectory)  # (T+1, C, H, W)

    T = len(test_trajectory) - 1
    results: dict[int, float] = {}

    for h in horizons:
        if h > T:
            results[h] = float("nan")
            continue

        x = test_trajectory[0].copy()
        preds = []

        with torch.no_grad():
            for t in range(h):
                x_t = torch.from_numpy(x).float().unsqueeze(0).to(dev)
                pred = model(x_t).squeeze(0).cpu().numpy()
                preds.append(pred)
                if binarize:
                    x = (np.clip(pred, 0, 1) >= 0.5).astype(np.float32)
                else:
                    x = np.clip(pred, 0, 1).astype(np.float32)

        pred_stack = np.stack(preds)  # (h, C, H, W)
        true_stack = test_trajectory[1 : h + 1]  # (h, C, H, W)

        if loss_type == "mse":
            results[h] = float(np.mean((true_stack - pred_stack) ** 2))
        else:
            pred_bin = (pred_stack >= 0.5).astype(np.float32)
            results[h] = float(np.mean(pred_bin == true_stack))

    return results


def evaluate_cem_planning(
    model: nn.Module,
    env: Any,
    n_episodes: int = 200,
    horizon: int = 5,
    population: int = 200,
    elite_k: int = 40,
    cem_iters: int = 3,
    max_steps: int = 50,
    device: str = "cpu",
    use_exhaustive: bool = True,
    use_soft_predictions: bool = True,
) -> dict[str, float]:
    """CEM / exhaustive planning evaluation for grid world.

    The env must support: reset() -> state (C,H,W), step(action) -> (state, reward, done),
    clone(), goal_pos, and agent_pos attributes.

    When ``use_exhaustive=True`` and the action/horizon space is small
    enough, ALL action sequences are enumerated instead of CEM sampling.

    When ``use_soft_predictions=True``, rollouts keep softmax
    probabilities instead of argmax one-hot discretization.

    Uses a fixed eval seed (12345) for paired evaluation across model seeds.

    Returns dict with success_rate, avg_steps, avg_reward, planning_method.
    """
    import itertools

    dev = torch.device(device)
    model = model.to(dev).eval()

    gs = env.grid_size
    n_cell_types = 4
    n_actions = 4

    # Fixed eval seed for paired evaluation
    rng = np.random.default_rng(12345)

    # Decide whether to use exhaustive search
    do_exhaustive = (
        use_exhaustive
        and n_actions <= 5
        and horizon <= 6
    )
    if do_exhaustive:
        seqs = list(itertools.product(range(n_actions), repeat=horizon))
        all_action_seqs = torch.tensor(seqs, dtype=torch.long, device=dev)
        total_seqs = len(all_action_seqs)
        planning_method = "Exhaustive"
    else:
        total_seqs = population
        planning_method = "CEM"

    successes = 0
    total_steps_success = 0
    total_reward = 0.0

    for ep in range(n_episodes):
        # Reset with fresh rng each episode
        env.rng = rng
        state = env.reset()
        goal_pos = env.goal_pos

        ep_reward = 0.0
        for step in range(max_steps):
            state_t = torch.from_numpy(state).float().unsqueeze(0).to(dev)
            goal_r, goal_c = goal_pos

            if do_exhaustive:
                # --- Exhaustive search ---
                action_seqs = all_action_seqs
                n_seqs = total_seqs

                cur = state_t.expand(n_seqs, -1, -1, -1).clone()
                for t in range(horizon):
                    acts = action_seqs[:, t]
                    agent_ch = cur[:, 2]
                    agent_flat = agent_ch.reshape(n_seqs, -1).argmax(dim=1)
                    ar = agent_flat // gs
                    ac = agent_flat % gs

                    af = torch.zeros(n_seqs, n_actions, gs, gs, device=dev)
                    bidx = torch.arange(n_seqs, device=dev)
                    af[bidx, acts, ar, ac] = 1.0

                    x = torch.cat([cur, af], dim=1)
                    with torch.no_grad():
                        logits = model(x)

                    if use_soft_predictions:
                        cur = F.softmax(logits, dim=1)
                    else:
                        pred_cls = logits.argmax(dim=1)
                        cur = F.one_hot(pred_cls, n_cell_types).permute(0, 3, 1, 2).float()

                # Score
                agent_ch = cur[:, 2]
                agent_flat = agent_ch.reshape(n_seqs, -1).argmax(dim=1)
                pred_ar = agent_flat // gs
                pred_ac = agent_flat % gs
                reached = (pred_ar == goal_r) & (pred_ac == goal_c)
                dist = (pred_ar - goal_r).abs() + (pred_ac - goal_c).abs()
                rewards = torch.where(reached, torch.ones_like(dist, dtype=torch.float32),
                                      -dist.float())

                best_idx = rewards.argmax()
                best_action = int(action_seqs[best_idx, 0].item())

            else:
                # --- CEM planning ---
                action_probs = torch.ones(horizon, n_actions, device=dev) / n_actions
                elite_idx = None
                action_seqs = None

                for _ in range(cem_iters):
                    action_seqs = torch.zeros(population, horizon, dtype=torch.long, device=dev)
                    for t in range(horizon):
                        action_seqs[:, t] = torch.multinomial(
                            action_probs[t].unsqueeze(0).expand(population, -1), 1
                        ).squeeze(-1)

                    # Batched rollout
                    cur = state_t.expand(population, -1, -1, -1).clone()
                    for t in range(horizon):
                        acts = action_seqs[:, t]
                        agent_ch = cur[:, 2]
                        agent_flat = agent_ch.reshape(population, -1).argmax(dim=1)
                        ar = agent_flat // gs
                        ac = agent_flat % gs

                        af = torch.zeros(population, n_actions, gs, gs, device=dev)
                        bidx = torch.arange(population, device=dev)
                        af[bidx, acts, ar, ac] = 1.0

                        x = torch.cat([cur, af], dim=1)
                        with torch.no_grad():
                            logits = model(x)

                        if use_soft_predictions:
                            cur = F.softmax(logits, dim=1)
                        else:
                            pred_cls = logits.argmax(dim=1)
                            cur = F.one_hot(pred_cls, n_cell_types).permute(0, 3, 1, 2).float()

                    # Reward: distance to goal
                    agent_ch = cur[:, 2]
                    agent_flat = agent_ch.reshape(population, -1).argmax(dim=1)
                    pred_ar = agent_flat // gs
                    pred_ac = agent_flat % gs
                    reached = (pred_ar == goal_r) & (pred_ac == goal_c)
                    dist = (pred_ar - goal_r).abs() + (pred_ac - goal_c).abs()
                    rewards = torch.where(reached, torch.ones_like(dist, dtype=torch.float32),
                                          -dist.float())

                    _, elite_idx = rewards.topk(elite_k)
                    elite_actions = action_seqs[elite_idx]
                    for t in range(horizon):
                        counts = torch.zeros(n_actions, device=dev)
                        for a in range(n_actions):
                            counts[a] = (elite_actions[:, t] == a).float().sum()
                        action_probs[t] = (counts + 0.1) / (elite_k + 0.1 * n_actions)

                best_action = int(action_seqs[elite_idx[0], 0].item())

            state, reward, done = env.step(best_action)
            ep_reward += reward

            if done:
                successes += 1
                total_steps_success += step + 1
                break

        total_reward += ep_reward

    return {
        "success_rate": successes / max(n_episodes, 1),
        "avg_steps": total_steps_success / max(successes, 1),
        "avg_reward": total_reward / max(n_episodes, 1),
        "planning_method": planning_method,
    }


# ===== Utilities =============================================================

def param_count(model: nn.Module | CML2DRidge) -> dict[str, int]:
    """Returns dict with 'trained' and 'frozen' counts.

    Works for both nn.Module (with .param_count()) and CML2DRidge.
    """
    if hasattr(model, "param_count"):
        return model.param_count()

    # Fallback for plain nn.Module without .param_count()
    return {
        "trained": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "frozen": sum(p.numel() for p in model.parameters() if not p.requires_grad),
    }
