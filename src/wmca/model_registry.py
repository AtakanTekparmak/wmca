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
    CML2DDiscreteSelect,
    CML2DHybridMrEsn,
    CML2DLearnedGateDynamic,
    CML2DLearnedGateStatic,
    CML2DMultiConfig,
    CML2DMultiR,
    CML2DRandomReservoir,
    CMLRegularizedNCA,
    ResCorMamba,
    ResCorMambaGated,
    ResCorMambaStat,
    ResCorRensDeep,
    ResCorRensStatBank,
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
from wmca.modules.discrete_rescor import DiscreteRescor, DiscreteRescorMamba


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
    "rescor_gate_static": {
        "class": ResidualCorrectionWM,
        "description": "ResCor + learned per-cell (eps, beta) from input (static gate)",
        "extra_kwargs": {"cml_gate": "static"},
    },
    "rescor_gate_dynamic": {
        "class": ResidualCorrectionWM,
        "description": "ResCor + learned per-cell (eps, beta) recomputed each CML step (dynamic gate)",
        "extra_kwargs": {"cml_gate": "dynamic"},
    },
    "rescor_discrete_global": {
        "class": ResidualCorrectionWM,
        "description": "ResCor + discrete selection over K=5 (eps, beta) configs (global softmax)",
        "extra_kwargs": {"cml_gate": "discrete_global"},
    },
    "rescor_discrete_percell": {
        "class": ResidualCorrectionWM,
        "description": "ResCor + discrete selection over K=5 (eps, beta) configs (per-cell softmax)",
        "extra_kwargs": {"cml_gate": "discrete_percell"},
    },
    "rescor_multi_config": {
        "class": ResidualCorrectionWM,
        "description": "ResCor + K=3 parallel CML passes with output blending (gradient-free CML selection)",
        "extra_kwargs": {"cml_gate": "multi_config"},
    },
    "rescor_multi_config_percell": {
        "class": ResidualCorrectionWM,
        "description": "ResCor + K=3 parallel CML passes with per-cell output blending",
        "extra_kwargs": {"cml_gate": "multi_config_percell"},
    },
    "rescor_random_reservoir_preserved": {
        "class": ResidualCorrectionWM,
        "description": "ResCor + K random-coupling frozen reservoirs (logistic preserved), softmax blend, no oracle",
        "extra_kwargs": {"cml_gate": "random_reservoir_preserved"},
    },
    "rescor_random_reservoir_full": {
        "class": ResidualCorrectionWM,
        "description": "ResCor + K random-coupling frozen reservoirs (ESN tanh, no logistic), softmax blend, no oracle",
        "extra_kwargs": {"cml_gate": "random_reservoir_full"},
    },
    "rescor_esn": {
        "class": ResidualCorrectionWM,
        "description": "ResCor + K frozen tanh reservoirs with random coupling kernels (Echo State Network template). Canonical name for A-full.",
        "extra_kwargs": {"cml_gate": "random_reservoir_full"},
    },
    "rescor_random_reservoir_full_cond": {
        "class": ResidualCorrectionWM,
        "description": "ResCor + A-full with input-conditioned gate (hypernet on [mean, var, grad-norm])",
        "extra_kwargs": {"cml_gate": "random_reservoir_full_cond"},
    },
    "rescor_esn_uniform": {
        "class": ResidualCorrectionWM,
        "description": "rescor_esn with strict 1/K averaging (zero trainable gate params)",
        "extra_kwargs": {"cml_gate": "random_reservoir_full_uniform"},
    },
    "rescor_mr": {
        "class": ResidualCorrectionWM,
        "description": "K vanilla CMLs (logistic + shared coupling) with K different r in [3.57, 3.99]. Learned softmax gate.",
        "extra_kwargs": {"cml_gate": "multi_r"},
    },
    "rescor_mr_uniform": {
        "class": ResidualCorrectionWM,
        "description": "rescor_mr with strict 1/K averaging (zero trainable gate params)",
        "extra_kwargs": {"cml_gate": "multi_r_uniform"},
    },
    "rescor_hybrid": {
        "class": ResidualCorrectionWM,
        "description": "Hybrid bank: half vanilla-r CMLs (chaos-depth axis) + half random-coupling tanh reservoirs (random-spatial axis), uniform 1/K averaged. Zero gate params.",
        "extra_kwargs": {"cml_gate": "hybrid_mr_esn"},
    },
    "rescor_rens": {
        "class": ResidualCorrectionWM,
        "description": "R-Ensemble: canonical name for rescor_mr_uniform (K vanilla CMLs spanning r in [3.57, 3.99], uniform 1/K averaging, zero gate params).",
        "extra_kwargs": {"cml_gate": "multi_r_uniform"},
    },
    "rescor_rens_deep": {
        "class": ResCorRensDeep,
        "description": "Deep stack of L rescor_rens stages (each = K=32 r-ensemble + NCA correction with residual addition). L is configurable.",
    },
    "rescor_rens_stat_full": {
        "class": ResCorRensStatBank,
        "description": "rescor_rens + stat-bank NCA with variance. NCA sees [x, cml_mean, cml_var, cml_min, cml_max] across K=32 reservoirs.",
        "extra_kwargs": {"include_var": True},
    },
    "rescor_mamba": {
        "class": ResCorMamba,
        "description": (
            "rescor_rens K=32 spatial core + per-cell Mamba SSM over K_context=4 "
            "past frames. Multi-frame input (B, 4, C, H, W). Mean-only NCA, "
            "zero-init Mamba out_proj (starts as pure rens K=32). "
            "Requires context_k=4 benchmark generation."
        ),
        "extra_kwargs": {
            "cml_K": 32, "context_k": 4, "expand": 2, "zero_init_out": True,
        },
    },
    "rescor_mamba_rand": {
        "class": ResCorMamba,
        "description": (
            "rescor_mamba with random-init Mamba out_proj (Kaiming default). "
            "Probes whether zero-init residual discipline is load-bearing."
        ),
        "extra_kwargs": {
            "cml_K": 32, "context_k": 4, "expand": 2, "zero_init_out": False,
        },
    },
    "rescor_mamba_stat": {
        "class": ResCorMambaStat,
        "description": (
            "rescor_mamba with stat-bank NCA (mean+min+max across K=32 "
            "reservoirs; no variance), zero-init Mamba out_proj."
        ),
        "extra_kwargs": {
            "cml_K": 32, "context_k": 4, "expand": 2, "zero_init_out": True,
        },
    },
    "rescor_mamba_stat_rand": {
        "class": ResCorMambaStat,
        "description": (
            "rescor_mamba_stat with random-init Mamba out_proj."
        ),
        "extra_kwargs": {
            "cml_K": 32, "context_k": 4, "expand": 2, "zero_init_out": False,
        },
    },
    "rescor_mamba_gated": {
        "class": ResCorMambaGated,
        "description": (
            "Drift-gated rescor_mamba: per-cell sigmoid attenuates the "
            "Mamba+NCA correction at high drift = sqrt(mean((x_now-cml_mean)^2)). "
            "When predictions stray from the rens K=32 manifold the gate "
            "closes, falling back to pure rens behavior. Two trainable "
            "scalars (gate_scale, gate_bias). Zero-init Mamba out_proj."
        ),
        "extra_kwargs": {
            "cml_K": 32, "context_k": 4, "expand": 2, "zero_init_out": True,
        },
    },
    "rescor_mamba_gated_rand": {
        "class": ResCorMambaGated,
        "description": (
            "rescor_mamba_gated with random-init Mamba out_proj. "
            "Architectural fix targeting mamba_rand's H=100 catastrophe "
            "on chaotic continuous benchmarks (gs/ks)."
        ),
        "extra_kwargs": {
            "cml_K": 32, "context_k": 4, "expand": 2, "zero_init_out": False,
        },
    },
    "rescor_rens_stat_no_var": {
        "class": ResCorRensStatBank,
        "description": "Variance-ablation baseline: rescor_rens + stat-bank NCA WITHOUT variance. NCA sees [x, cml_mean, cml_min, cml_max].",
        "extra_kwargs": {"include_var": False},
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
    "discrete_rescor": {
        "class": DiscreteRescor,
        "description": (
            "Discrete token rescor: CML+NCA world model for VQ-VAE token sequence prediction. "
            "Embeds discrete token indices + action index, runs CML reservoir, applies NCA "
            "correction in embedding space, outputs vocabulary logits for CE loss."
        ),
    },
    "discrete_rescor_mamba": {
        "class": DiscreteRescorMamba,
        "description": (
            "Discrete token rescor with per-cell Mamba SSM over K=4 temporal context frames. "
            "Processes K consecutive token frames through a Mamba block before feeding "
            "to the CML+NCA pipeline. Outputs vocabulary logits for CE loss."
        ),
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

    # Discrete token models — completely different constructor signature.
    # These accept vocab_size, n_actions, embed_dim instead of in_channels/out_channels.
    if name in ("discrete_rescor", "discrete_rescor_mamba"):
        import inspect
        sig = inspect.signature(cls.__init__)
        kwargs_out: dict[str, Any] = {"use_sigmoid": False, "seed": seed}
        # Forward known kwargs that match the constructor
        for k in ("vocab_size", "n_actions", "embed_dim", "hidden_ch",
                  "cml_K", "cml_steps", "r_lo", "r_hi", "mamba_context"):
            if k in sig.parameters and k in kwargs:
                kwargs_out[k] = kwargs[k]
        # Merge extra_kwargs from registry entry
        for k, v in entry.get("extra_kwargs", {}).items():
            if k in sig.parameters:
                kwargs_out[k] = v
        return cls(**kwargs_out)

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
    # Merge extra_kwargs from registry entry (e.g., cml_gate for gated variants)
    for k, v in entry.get("extra_kwargs", {}).items():
        if k in sig.parameters:
            kwargs_out[k] = v
    # Forward any extra kwargs that match the constructor signature
    for k, v in kwargs.items():
        if k in sig.parameters:
            kwargs_out[k] = v
    return cls(**kwargs_out)


# ===== Training ==============================================================

def _ensure_tensor(arr, dev: torch.device) -> torch.Tensor:
    """Convert numpy array or torch tensor to float tensor on *dev*."""
    if isinstance(arr, torch.Tensor):
        return arr.float().to(dev)
    return torch.from_numpy(np.asarray(arr)).float().to(dev)


def _extract_horizon_2_targets(Y: torch.Tensor, n_steps: int):
    """Build a t+2 supervision tensor aligned with Y (t+1 targets).

    ``_make_pairs`` flattens trajectories into contiguous (T,) blocks
    where T = n_steps. For sample i, Y[i] is the t+1 target. The t+2
    target is Y[i+1] EXCEPT when i is the last pair of a trajectory
    (i.e., (i % T) == T-1) — there is no t+2 in that case.

    Returns (Y_next, valid_mask):
      Y_next: same shape as Y, with Y_next[i] = Y[i+1] when valid (and
              a copy of Y[i] in invalid positions, never read).
      valid_mask: bool tensor of shape (N,) — True where t+2 exists.
    """
    N = Y.shape[0]
    # Y_next[i] := Y[i+1] when (i % n_steps) != n_steps-1.
    Y_next = torch.empty_like(Y)
    Y_next[:-1] = Y[1:]
    Y_next[-1] = Y[-1]  # placeholder; mask makes this unread
    idx = torch.arange(N, device=Y.device)
    valid_mask = (idx % n_steps) != (n_steps - 1)
    # Last sample of last trajectory is also invalid by construction
    valid_mask[-1] = False
    return Y_next, valid_mask


def _extract_horizon_targets(Y: torch.Tensor, n_steps: int, H: int):
    """Generalized t+1..t+H supervision tensor for multistep training.

    Sample i has Y[i] = t+1 target (offset 0). The t+(1+h) target is
    Y[i+h] for h in 0..H-1, valid only when (i % n_steps) + h <= n_steps-1.

    Returns (Y_targets, valid_mask):
      Y_targets: (N, H, *Y.shape[1:]) tensor; Y_targets[i, h] = Y[i+h]
                 with the last valid copy padded into out-of-range slots
                 (mask makes them unread).
      valid_mask: bool tensor of shape (N,) — True where ALL H offsets
                  fall inside the same trajectory (i.e. the sample has
                  a full H-step ground-truth window).

    H=2 is bit-identical to ``_extract_horizon_2_targets`` (same
    Y_targets[:,1] layout and same valid_mask).
    """
    if H < 1:
        raise ValueError(f"H must be >= 1, got {H}")
    N = Y.shape[0]
    rest = Y.shape[1:]
    Y_targets = torch.empty((N, H, *rest), dtype=Y.dtype, device=Y.device)
    Y_targets[:, 0] = Y
    for h in range(1, H):
        # Slot h: Y_targets[i, h] = Y[i + h] when in-range, else placeholder.
        Y_targets[:N - h, h] = Y[h:]
        if h > 0:
            # Padding for last h positions (these are masked out anyway).
            Y_targets[N - h:, h] = Y[N - 1]
    idx = torch.arange(N, device=Y.device)
    pos = idx % n_steps  # 0..n_steps-1 within trajectory
    # Need pos + (H-1) <= n_steps - 1  =>  pos <= n_steps - H
    valid_mask = pos <= (n_steps - H)
    # Defensive: also bound by global tensor size — last (H-1) positions
    # of the very last trajectory cannot have a t+H target.
    if H > 1:
        valid_mask[N - (H - 1):] = False
    return Y_targets, valid_mask


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
    train_noise_sigma: float = 0.0,
    pushforward: bool = False,
    pushforward_n_steps: int | None = None,
    pushforward_prob: float = 0.5,
    multistep_horizon: int = 1,
    multistep_bptt: int = 4,
    multistep_weight_schedule: str = "uniform",
    multistep_n_steps: int | None = None,
    msdc_alpha: float = 0.0,
    compile: bool = False,
    bf16: bool = False,
    # Extra kwargs accepted (and ignored) for runner convenience
    benchmark_name: str | None = None,
    model_name: str | None = None,
) -> nn.Module:
    """Generic training loop. Handles MSE, BCE, and cross-entropy losses.

    For CMLRegularizedNCA, adds the regularization term automatically.
    If ``train_noise_sigma > 0``, Gaussian noise with that std is added to
    each training batch input ``xb`` before the forward pass. Noise is
    applied only during training — ``evaluate_model`` and ``evaluate_rollout``
    see clean inputs. This is the standard dynamical-systems trick to
    regularize autoregressive rollout stability.

    If ``pushforward=True`` (Brandstetter et al. 2022, "Message Passing
    Neural PDE Solvers"), with probability ``pushforward_prob`` (default
    0.5) per training step the loss is replaced with the pushforward
    loss: a detached forward pass produces a t+1 prediction, the model is
    re-fed that prediction (advancing the K-frame buffer for rank-5
    inputs), and the resulting t+2 prediction is supervised against the
    ground-truth t+2. Requires that pairs are arranged as contiguous
    trajectory blocks of length ``pushforward_n_steps`` (the default,
    inferred from ``len(Y_train) / N_traj`` is not safe — pass it
    explicitly when known; if ``None``, falls back to assuming each
    trajectory contributes the entire training tensor, which collapses
    the mask and is supported only as a defensive default).

    Returns the trained model (best val checkpoint restored).
    """
    dev = torch.device(device) if isinstance(device, str) else device
    model = model.to(dev)

    # Optional: torch.compile for kernel-launch overhead reduction.
    # `reduce-overhead` mode uses CUDA Graphs + autotune (best for small
    # models on CUDA). Falls back silently if compile is unavailable
    # (e.g., CPU device, or model has unsupported Python-side state).
    use_compile = bool(compile) and dev.type == "cuda"
    raw_model = model  # keep eager handle for state_dict save/restore
    if use_compile:
        # `reduce-overhead` (CUDA Graphs) is unsafe in our training loop:
        # the val pass and the pushforward double-forward both reuse
        # outputs from the prior compiled call, which CUDAGraphs forbids
        # without explicit `mark_step_begin`/clone discipline.
        # `default` mode (Inductor without cudagraphs) is safe and still
        # gives most of the kernel-fusion speedup.
        compile_mode = "default"
        try:
            model = torch.compile(model, mode=compile_mode, fullgraph=False)
            print(f"[train_model] torch.compile enabled (mode={compile_mode})")
        except Exception as e:
            print(f"[train_model] torch.compile failed ({type(e).__name__}: {e}); "
                  f"falling back to eager mode")
            use_compile = False
            model = raw_model

    # Optional: bf16 mixed precision via autocast. Optimizer params stay
    # in fp32 (autocast only casts forward + loss). bf16 (not fp16) so
    # the CML logistic-map reservoir near r=3.99 doesn't underflow.
    use_bf16 = bool(bf16) and dev.type == "cuda"
    if use_bf16:
        print(f"[train_model] bf16 autocast enabled")

    # Split optimizer into two param groups so we can apply strong L2
    # weight decay selectively to "alpha" params (dilation gates, depth
    # gates, router weights).  Models that expose ``get_alpha_params()``
    # declare exactly which params should be penalised; for older models
    # without the method we fall back to string matching on param names.
    alpha_params: list[nn.Parameter] = []
    other_params: list[nn.Parameter] = []

    if hasattr(raw_model, "get_alpha_params"):
        alpha_ids = {id(p) for p in raw_model.get_alpha_params()}
        for p in raw_model.parameters():
            if not p.requires_grad:
                continue
            (alpha_params if id(p) in alpha_ids else other_params).append(p)
    else:
        for pname, p in raw_model.named_parameters():
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
        optimizer = torch.optim.Adam(raw_model.parameters(), lr=lr, weight_decay=0.0)

    is_cml_reg = isinstance(raw_model, CMLRegularizedNCA)

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

    # Build t+2 supervision tensor for pushforward, if requested.
    Y_next_tr: torch.Tensor | None = None
    pf_valid_mask: torch.Tensor | None = None
    if pushforward:
        if is_cml_reg:
            raise ValueError(
                "pushforward=True is not supported with CMLRegularizedNCA "
                "(dual-output model). Use a single-output rescor variant."
            )
        n_steps_pf = pushforward_n_steps
        if n_steps_pf is None:
            # Defensive default: treat the whole tensor as one trajectory
            # block (so only the very last sample is masked out).
            n_steps_pf = len(Y_tr)
        Y_next_tr, pf_valid_mask = _extract_horizon_2_targets(Y_tr, n_steps_pf)

    # ---- MSDC: drift-conditioned multistep loss validation ----------------
    # msdc_alpha > 0 modulates per-step multistep losses by (1 - alpha * gate),
    # so the gate gets a coherence-discrimination gradient. Enforce mutex
    # with pushforward, with H=1, and require a model exposing compute_gate.
    use_msdc = float(msdc_alpha) > 0.0
    if use_msdc:
        if pushforward:
            raise ValueError(
                "msdc_alpha > 0 is mutually exclusive with pushforward=True."
            )
        if multistep_horizon <= 1:
            raise ValueError(
                "msdc_alpha > 0 requires multistep_horizon > 1; "
                f"got multistep_horizon={multistep_horizon}."
            )
        if not hasattr(raw_model, "compute_gate"):
            raise ValueError(
                "msdc_alpha > 0 requires a gated model exposing "
                "`compute_gate`; got "
                f"{type(raw_model).__name__} which does not."
            )

    # ---- Multistep penalty NODE loss (Chakraborty et al. 2024) -------------
    # When multistep_horizon > 1, train by H-step rollout with truncated
    # BPTT through the last K_bptt steps. Mutually exclusive with pushforward.
    use_multistep = multistep_horizon > 1
    Y_ms_tr: torch.Tensor | None = None
    ms_valid_mask: torch.Tensor | None = None
    ms_step_weights: torch.Tensor | None = None
    if use_multistep:
        if pushforward:
            raise ValueError(
                "multistep_horizon > 1 is mutually exclusive with "
                "pushforward=True. Pick one training-time multi-step scheme."
            )
        if is_cml_reg:
            raise ValueError(
                "multistep_horizon > 1 is not supported with "
                "CMLRegularizedNCA (dual-output model)."
            )
        if is_ce:
            raise ValueError(
                "multistep_horizon > 1 currently supports MSE/BCE only "
                "(rollout state advance assumes continuous predictions)."
            )
        if multistep_bptt < 1 or multistep_bptt > multistep_horizon:
            raise ValueError(
                f"multistep_bptt must be in [1, multistep_horizon]; "
                f"got bptt={multistep_bptt}, H={multistep_horizon}"
            )
        n_steps_ms = multistep_n_steps
        if n_steps_ms is None:
            n_steps_ms = len(Y_tr)
        Y_ms_tr, ms_valid_mask = _extract_horizon_targets(
            Y_tr, n_steps_ms, multistep_horizon
        )
        # Per-step weights (uniform or decay).
        H_ms = multistep_horizon
        if multistep_weight_schedule == "uniform":
            w = torch.ones(H_ms, device=dev)
        elif multistep_weight_schedule == "decay":
            # Geometric decay (0.7^h), normalized to mean 1 so total scale ≈ H.
            w = torch.tensor(
                [0.7 ** h for h in range(H_ms)], dtype=torch.float32, device=dev
            )
            w = w * (H_ms / w.sum())
        else:
            raise ValueError(
                f"Unknown multistep_weight_schedule "
                f"'{multistep_weight_schedule}' (use 'uniform' or 'decay')"
            )
        ms_step_weights = w

    best_val_loss = float("inf")
    best_state: dict | None = None
    train_losses: list[float] = []
    val_losses: list[float] = []

    t0 = time.time()

    # Autocast context manager (no-op when disabled).
    from contextlib import nullcontext
    def _amp_ctx():
        if use_bf16:
            return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(X_tr), device=dev)
        total_loss = 0.0
        n_batches = 0

        for i in range(0, len(perm), batch_size):
            idx = perm[i : i + batch_size]
            xb, yb = X_tr[idx], Y_tr[idx]

            # Noise injection on input x (training only).
            # Intentionally applied to xb (the full input including any
            # action channels for action-conditioned benchmarks) — matches
            # the spec "perturb input x". Does not touch yb.
            if train_noise_sigma > 0.0:
                xb = xb + torch.randn_like(xb) * train_noise_sigma

            # ---- Pushforward branch (Brandstetter et al. 2022) -----------
            # With probability pushforward_prob and at least one valid t+2
            # target in the batch, replace the standard loss with the
            # pushforward loss: detached one-step forward, then re-feed the
            # prediction and supervise the resulting two-step output
            # against the ground-truth t+2.
            do_pushforward = (
                pushforward
                and pf_valid_mask is not None
                and Y_next_tr is not None
                and torch.rand(1, device=dev).item() < pushforward_prob
            )
            if do_pushforward:
                mask_b = pf_valid_mask[idx]
                if mask_b.any():
                    xb_v = xb[mask_b]
                    y2_v = Y_next_tr[idx][mask_b]
                    with torch.no_grad():
                        with _amp_ctx():
                            pred_t1 = model(xb_v)
                            # Clone to escape CUDAGraphs static buffer reuse
                            # if compile is enabled.
                            if use_compile:
                                pred_t1 = pred_t1.clone()
                    # Build pushforward input depending on rank.
                    # rank-4 (B, C, H, W): single-frame -> just feed pred.
                    # rank-5 (B, K, C, H, W): K-frame buffer -> shift and
                    # append the prediction at the most recent slot.
                    if xb_v.dim() == 5:
                        buf_pushed = torch.cat(
                            [xb_v[:, 1:], pred_t1.unsqueeze(1)], dim=1
                        )
                    else:
                        buf_pushed = pred_t1
                    with _amp_ctx():
                        pred_t2 = model(buf_pushed)
                        if is_ce:
                            loss = criterion(pred_t2, _ce_target(y2_v))
                        else:
                            loss = criterion(pred_t2, y2_v)
                else:
                    # No valid t+2 in this batch — fall back to standard.
                    do_pushforward = False

            # ---- Multistep penalty branch -------------------------------
            # H-step rollout, no_grad for first H-K_bptt steps, BPTT through
            # the last K_bptt steps. Skips samples without a full ground-truth
            # window of length H.
            do_multistep = bool(
                use_multistep
                and ms_valid_mask is not None
                and Y_ms_tr is not None
                and not do_pushforward
            )
            if do_multistep:
                mask_b = ms_valid_mask[idx]
                if mask_b.any():
                    xb_v = xb[mask_b]
                    y_window = Y_ms_tr[idx][mask_b]  # (B', H, *Y.shape[1:])
                    H_ms = multistep_horizon
                    K_bptt = multistep_bptt
                    n_no_grad = H_ms - K_bptt
                    state = xb_v
                    losses_h: list[torch.Tensor] = []
                    for h in range(H_ms):
                        if h < n_no_grad:
                            with torch.no_grad():
                                with _amp_ctx():
                                    pred = model(state)
                                    if use_compile:
                                        pred = pred.clone()
                        else:
                            with _amp_ctx():
                                pred = model(state)
                        gt_h = y_window[:, h]
                        # MSDC: drift-conditioned per-step weight.
                        # `weight_h` is detached (no grad through weighting)
                        # so the gate receives a coherence-aware signal but
                        # cannot trivially game the loss by closing.
                        if use_msdc:
                            with torch.no_grad():
                                gate_h = raw_model.compute_gate(state)
                                weight_h = (1.0 - float(msdc_alpha) * gate_h)
                            diff_sq = (pred.float() - gt_h.float()) ** 2
                            loss_h = (weight_h.float() * diff_sq).mean()
                        else:
                            # Loss in fp32 for numerical stability under bf16.
                            loss_h = F.mse_loss(pred.float(), gt_h.float())
                        losses_h.append(loss_h * ms_step_weights[h])
                        # Advance state.
                        if h < H_ms - 1:
                            if state.dim() == 5:
                                # K-frame buffer: shift in pred at most-recent slot.
                                pred_slot = pred.detach() if h < n_no_grad else pred
                                new_buf = torch.cat(
                                    [state[:, 1:], pred_slot.unsqueeze(1)], dim=1
                                )
                                state = new_buf
                            else:
                                state = pred.detach() if h < n_no_grad else pred
                    loss = torch.stack(losses_h).sum() / H_ms
                else:
                    # No valid sample in this batch — fall through to standard.
                    do_multistep = False

            if (not do_pushforward) and (not do_multistep):
                with _amp_ctx():
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
            # Gradient clipping: only for multistep (chaotic dynamics can
            # amplify gradients ~50x per step; Mikhaeil 2022). Skip for
            # H=1/pushforward to keep strict bit-identical backwards-compat.
            if do_multistep:
                torch.nn.utils.clip_grad_norm_(raw_model.parameters(), max_norm=1.0)
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
                with _amp_ctx():
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
            best_state = {k: v.detach().cpu().clone()
                          for k, v in raw_model.state_dict().items()}

    train_time = time.time() - t0

    if best_state is not None:
        raw_model.load_state_dict(best_state)
    raw_model = raw_model.to(torch.device("cpu")).eval()

    del X_tr, Y_tr, X_v, Y_v
    gc.collect()

    return raw_model


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


def train_discrete_rescor(
    model: nn.Module,
    tokens: np.ndarray,
    next_tokens: np.ndarray,
    actions: np.ndarray,
    tokens_val: np.ndarray | None = None,
    next_tokens_val: np.ndarray | None = None,
    actions_val: np.ndarray | None = None,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1.4e-3,
    device: str | torch.device = "cpu",
) -> nn.Module:
    """Train a discrete token world model on VQ-VAE token sequences.

    Works with both ``DiscreteRescor`` (single-frame) and
    ``DiscreteRescorMamba`` (K=4 context frames). Uses CrossEntropyLoss
    over vocabulary classes.

    Args:
        model: ``DiscreteRescor`` or ``DiscreteRescorMamba`` instance.
        tokens: (N, H, W) int64 — current token grids.
        next_tokens: (N, H, W) int64 — next-step token grids.
        actions: (N,) int64 — action indices.
        tokens_val, next_tokens_val, actions_val: optional validation set.
        epochs: number of training epochs.
        batch_size: batch size.
        lr: learning rate (sqrt-rule adjusted from 1e-3 → 1.4e-3).
        device: torch device.

    Returns:
        Trained model (best validation checkpoint restored).
    """
    dev = torch.device(device) if isinstance(device, str) else device
    model = model.to(dev)

    is_mamba = hasattr(model, "mamba_context")

    # Convert data to tensors
    tok_t = torch.from_numpy(np.asarray(tokens)).long().to(dev)
    nxt_t = torch.from_numpy(np.asarray(next_tokens)).long().to(dev)
    act_t = torch.from_numpy(np.asarray(actions)).long().to(dev)
    N = len(tok_t)

    if tokens_val is not None:
        tok_v = torch.from_numpy(np.asarray(tokens_val)).long().to(dev)
        nxt_v = torch.from_numpy(np.asarray(next_tokens_val)).long().to(dev)
        act_v = torch.from_numpy(np.asarray(actions_val)).long().to(dev)
    else:
        n_val = max(1, N // 5)
        tok_v = tok_t[-n_val:]
        nxt_v = nxt_t[-n_val:]
        act_v = act_t[-n_val:]

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    best_state = None
    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(N, device=dev)
        total_loss = 0.0
        n_batches = 0

        for i in range(0, N, batch_size):
            idx = perm[i : i + batch_size]
            t_b = tok_t[idx]   # (B, H, W)  — single-frame input
            n_b = nxt_t[idx]   # (B, H, W)  — target
            a_b = act_t[idx]   # (B,)

            if is_mamba:
                # Mamba variant needs K=4 context: build from [batch_size, 4, H, W]
                # Use shifted tokens: t_b = tokens at t, K-1 prior frames from tokens data
                ctx_k = model.mamba_context
                B = len(idx)
                ctx = torch.zeros(B, ctx_k, *t_b.shape[1:], dtype=torch.long, device=dev)
                # Fill most recent slot with current tokens
                ctx[:, -1] = t_b
                # Fill prior slots from earlier tokens (clamp to avoid negative indices)
                for k in range(ctx_k - 1):
                    offset = ctx_k - 1 - k
                    prior_idx = idx - offset
                    prior_idx = prior_idx.clamp(min=0)
                    ctx[:, k] = tok_t[prior_idx]
                logits = model(ctx, a_b)  # (B, V, H, W)
            else:
                logits = model(t_b, a_b)  # (B, V, H, W)

            # CE loss: logits (B, V, H, W), target (B, H, W) long
            loss = criterion(logits, n_b)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_train = total_loss / max(n_batches, 1)

        # Validation
        model.eval()
        with torch.no_grad():
            val_sum = 0.0
            val_n = 0
            for vi in range(0, len(tok_v), batch_size):
                vx = tok_v[vi : vi + batch_size]
                vy = nxt_v[vi : vi + batch_size]
                va = act_v[vi : vi + batch_size]
                if is_mamba:
                    ctx_k_m = model.mamba_context
                    Bv = len(vx)
                    ctx_v = torch.zeros(Bv, ctx_k_m, *vx.shape[1:], dtype=torch.long, device=dev)
                    ctx_v[:, -1] = vx
                    for k in range(ctx_k_m - 1):
                        offset = ctx_k_m - 1 - k
                        prior_idx_v = torch.arange(vi, vi + Bv, device=dev) - offset
                        prior_idx_v = prior_idx_v.clamp(min=0)
                        ctx_v[:, k] = tok_t[prior_idx_v]
                    vp = model(ctx_v, va)
                else:
                    vp = model(vx, va)
                vl = criterion(vp, vy).item()
                val_sum += vl * len(vx)
                val_n += len(vx)
            val_loss = val_sum / max(val_n, 1)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}

    train_time = time.time() - t0

    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to(torch.device("cpu")).eval()

    # Clean up GPU tensors
    del tok_t, nxt_t, act_t, tok_v, nxt_v, act_v
    gc.collect()

    return model


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
