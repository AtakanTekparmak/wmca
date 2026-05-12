from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from wmca.modules.mamba_block import MinimalMambaBlock


class CML2D(nn.Module):
    """2D Coupled Map Lattice with frozen logistic map + conv2d coupling."""

    def __init__(self, in_channels: int = 1, steps: int = 15,
                 r: float = 3.90, eps: float = 0.3, beta: float = 0.15,
                 seed: int = 42, kernel_size: int = 3):
        super().__init__()
        self.in_channels = in_channels
        self.steps = steps
        self.kernel_size = kernel_size

        self.register_buffer("r", torch.tensor(r))
        self.register_buffer("eps", torch.tensor(eps))
        self.register_buffer("beta", torch.tensor(beta))

        rng = torch.Generator().manual_seed(seed)
        K_raw = torch.rand(in_channels, 1, kernel_size, kernel_size, generator=rng).abs()
        K_norm = K_raw / K_raw.sum(dim=(-1, -2), keepdim=True)
        self.register_buffer("K_local", K_norm)

    def forward(self, drive: torch.Tensor) -> torch.Tensor:
        grid = drive
        r, eps, beta = self.r, self.eps, self.beta
        pad = self.kernel_size // 2
        for _ in range(self.steps):
            mapped = r * grid * (1.0 - grid)
            local = F.conv2d(mapped, self.K_local, padding=pad,
                             groups=self.in_channels)
            physics = (1 - eps) * mapped + eps * local
            grid = (1 - beta) * physics + beta * drive
        return grid.clamp(1e-4, 1 - 1e-4)

    def param_count(self) -> dict[str, int]:
        return {"trained": 0, "frozen": sum(b.numel() for b in self.buffers())}


class CML2DLearnedGateStatic(nn.Module):
    """CML2D with learned per-cell (eps, beta) computed once from the input.

    A tiny Conv2d(in_ch, 2, 3x3) gate maps the input state to spatially-varying
    eps and beta maps. These are computed once and held fixed across all M CML
    steps. Adds ~20 learned params. Default-initialized to match frozen defaults
    (eps=0.30, beta=0.15).
    """

    def __init__(self, in_channels: int = 1, steps: int = 15,
                 r: float = 3.90, eps_default: float = 0.3, beta_default: float = 0.15,
                 eps_max: float = 0.8, beta_max: float = 0.5,
                 seed: int = 42, kernel_size: int = 3):
        super().__init__()
        self.in_channels = in_channels
        self.steps = steps
        self.kernel_size = kernel_size
        self.eps_max = eps_max
        self.beta_max = beta_max

        self.register_buffer("r", torch.tensor(r))

        rng = torch.Generator().manual_seed(seed)
        K_raw = torch.rand(in_channels, 1, kernel_size, kernel_size, generator=rng).abs()
        K_norm = K_raw / K_raw.sum(dim=(-1, -2), keepdim=True)
        self.register_buffer("K_local", K_norm)

        # Gate: input -> (eps_map, beta_map) per cell
        self.gate = nn.Conv2d(in_channels, 2, 3, padding=1)
        # Default-init: bias so sigmoid(bias) * max = default value
        with torch.no_grad():
            self.gate.weight.zero_()
            # sigmoid(x) * max = default => x = logit(default / max)
            eps_logit = torch.log(torch.tensor(eps_default / eps_max) / (1 - eps_default / eps_max))
            beta_logit = torch.log(torch.tensor(beta_default / beta_max) / (1 - beta_default / beta_max))
            self.gate.bias[0] = eps_logit
            self.gate.bias[1] = beta_logit

    def forward(self, drive: torch.Tensor) -> torch.Tensor:
        grid = drive
        r = self.r
        pad = self.kernel_size // 2

        # Compute eps/beta maps once from input (static)
        eb = torch.sigmoid(self.gate(drive))  # (B, 2, H, W)
        eps_map = eb[:, 0:1] * self.eps_max   # (B, 1, H, W)
        beta_map = eb[:, 1:2] * self.beta_max

        for _ in range(self.steps):
            mapped = r * grid * (1.0 - grid)
            local = F.conv2d(mapped, self.K_local, padding=pad,
                             groups=self.in_channels)
            physics = (1 - eps_map) * mapped + eps_map * local
            grid = (1 - beta_map) * physics + beta_map * drive
        return grid.clamp(1e-4, 1 - 1e-4)

    def param_count(self) -> dict[str, int]:
        trained = sum(p.numel() for p in self.gate.parameters())
        frozen = sum(b.numel() for b in self.buffers())
        return {"trained": trained, "frozen": frozen}


class CML2DLearnedGateDynamic(nn.Module):
    """CML2D with learned per-cell (eps, beta) recomputed at each CML step.

    Same gate as Static variant, but the gate reads the evolving grid state
    at each step, allowing eps/beta to adapt as the CML dynamics unfold.
    Adds ~20 learned params.
    """

    def __init__(self, in_channels: int = 1, steps: int = 15,
                 r: float = 3.90, eps_default: float = 0.3, beta_default: float = 0.15,
                 eps_max: float = 0.8, beta_max: float = 0.5,
                 seed: int = 42, kernel_size: int = 3):
        super().__init__()
        self.in_channels = in_channels
        self.steps = steps
        self.kernel_size = kernel_size
        self.eps_max = eps_max
        self.beta_max = beta_max

        self.register_buffer("r", torch.tensor(r))

        rng = torch.Generator().manual_seed(seed)
        K_raw = torch.rand(in_channels, 1, kernel_size, kernel_size, generator=rng).abs()
        K_norm = K_raw / K_raw.sum(dim=(-1, -2), keepdim=True)
        self.register_buffer("K_local", K_norm)

        # Gate: current grid state -> (eps_map, beta_map) per cell
        self.gate = nn.Conv2d(in_channels, 2, 3, padding=1)
        with torch.no_grad():
            self.gate.weight.zero_()
            eps_logit = torch.log(torch.tensor(eps_default / eps_max) / (1 - eps_default / eps_max))
            beta_logit = torch.log(torch.tensor(beta_default / beta_max) / (1 - beta_default / beta_max))
            self.gate.bias[0] = eps_logit
            self.gate.bias[1] = beta_logit

    def forward(self, drive: torch.Tensor) -> torch.Tensor:
        grid = drive
        r = self.r
        pad = self.kernel_size // 2

        for _ in range(self.steps):
            # Recompute eps/beta from current grid state (dynamic)
            eb = torch.sigmoid(self.gate(grid))
            eps_map = eb[:, 0:1] * self.eps_max
            beta_map = eb[:, 1:2] * self.beta_max

            mapped = r * grid * (1.0 - grid)
            local = F.conv2d(mapped, self.K_local, padding=pad,
                             groups=self.in_channels)
            physics = (1 - eps_map) * mapped + eps_map * local
            grid = (1 - beta_map) * physics + beta_map * drive
        return grid.clamp(1e-4, 1 - 1e-4)

    def param_count(self) -> dict[str, int]:
        trained = sum(p.numel() for p in self.gate.parameters())
        frozen = sum(b.numel() for b in self.buffers())
        return {"trained": trained, "frozen": frozen}


class CML2DDiscreteSelect(nn.Module):
    """CML2D with discrete selection over K candidate (eps, beta) configs.

    Learns a softmax selection over K pre-defined (eps, beta) pairs.
    Gradient flows through the softmax weights only, completely bypassing
    the chaotic CML interior. This avoids the gradient-through-chaos problem
    that makes continuous eps/beta learning unstable (Mikhaeil et al. 2022).

    Two modes:
    - global: K logits shared across all cells (K learnable params)
    - percell: Conv2d produces K logits per cell (~K*10 learnable params)
    """

    # Candidate configs from sweep results
    CANDIDATES = [
        (0.05, 0.01),   # weak coupling + free-running (good for KS)
        (0.15, 0.05),   # weak coupling + weak drive
        (0.15, 0.15),   # weak coupling + moderate drive (good for GS)
        (0.30, 0.15),   # default (balanced)
        (0.50, 0.30),   # strong coupling + strong drive (good for discrete CAs)
    ]

    def __init__(self, in_channels: int = 1, steps: int = 15,
                 r: float = 3.90, seed: int = 42, kernel_size: int = 3,
                 mode: str = "global", tau: float = 1.0,
                 eps_default: float = 0.3, beta_default: float = 0.15):
        super().__init__()
        self.in_channels = in_channels
        self.steps = steps
        self.kernel_size = kernel_size
        self.mode = mode
        self.tau = tau
        self.K = len(self.CANDIDATES)

        self.register_buffer("r", torch.tensor(r))
        self.register_buffer("candidates_eps",
                             torch.tensor([c[0] for c in self.CANDIDATES]))
        self.register_buffer("candidates_beta",
                             torch.tensor([c[1] for c in self.CANDIDATES]))

        rng = torch.Generator().manual_seed(seed)
        K_raw = torch.rand(in_channels, 1, kernel_size, kernel_size, generator=rng).abs()
        K_norm = K_raw / K_raw.sum(dim=(-1, -2), keepdim=True)
        self.register_buffer("K_local", K_norm)

        if mode == "global":
            # K learnable logits, init so default config (index 3) is favored
            logits = torch.zeros(self.K)
            # Find closest candidate to default
            for i, (e, b) in enumerate(self.CANDIDATES):
                if abs(e - eps_default) < 0.05 and abs(b - beta_default) < 0.05:
                    logits[i] = 2.0  # favor this one at init
                    break
            self.logits = nn.Parameter(logits)
        else:  # percell
            self.gate = nn.Conv2d(in_channels, self.K, 3, padding=1)
            with torch.no_grad():
                self.gate.weight.zero_()
                self.gate.bias.zero_()
                for i, (e, b) in enumerate(self.CANDIDATES):
                    if abs(e - eps_default) < 0.05 and abs(b - beta_default) < 0.05:
                        self.gate.bias[i] = 2.0
                        break

    def forward(self, drive: torch.Tensor) -> torch.Tensor:
        B, C, H, W = drive.shape
        r = self.r
        pad = self.kernel_size // 2

        # Compute selection weights
        if self.mode == "global":
            weights = F.softmax(self.logits / self.tau, dim=0)  # (K,)
            eps = (weights * self.candidates_eps).sum()
            beta = (weights * self.candidates_beta).sum()
        else:
            logits = self.gate(drive)  # (B, K, H, W)
            weights = F.softmax(logits / self.tau, dim=1)  # (B, K, H, W)
            eps = (weights * self.candidates_eps[None, :, None, None]).sum(dim=1, keepdim=True)
            beta = (weights * self.candidates_beta[None, :, None, None]).sum(dim=1, keepdim=True)

        # Run CML with selected eps/beta (gradient only through weights, not CML)
        grid = drive
        for _ in range(self.steps):
            mapped = r * grid * (1.0 - grid)
            local = F.conv2d(mapped, self.K_local, padding=pad,
                             groups=self.in_channels)
            physics = (1 - eps) * mapped + eps * local
            grid = (1 - beta) * physics + beta * drive
        return grid.clamp(1e-4, 1 - 1e-4)

    def get_selection_info(self) -> dict:
        """Return the learned selection weights and effective eps/beta."""
        if self.mode == "global":
            weights = F.softmax(self.logits / self.tau, dim=0)
            eps = (weights * self.candidates_eps).sum().item()
            beta = (weights * self.candidates_beta).sum().item()
            return {
                "weights": {f"({e:.2f},{b:.2f})": w.item()
                           for (e, b), w in zip(self.CANDIDATES, weights)},
                "effective_eps": eps,
                "effective_beta": beta,
            }
        return {"mode": "percell"}

    def param_count(self) -> dict[str, int]:
        if self.mode == "global":
            return {"trained": self.K, "frozen": sum(b.numel() for b in self.buffers())}
        trained = sum(p.numel() for p in self.gate.parameters())
        frozen = sum(b.numel() for b in self.buffers())
        return {"trained": trained, "frozen": frozen}


class CML2DMultiConfig(nn.Module):
    """K parallel CML2D passes with distinct (eps, beta), blended by learned softmax.

    Runs K separate CML forward passes (each with its own frozen eps/beta config),
    detaches their outputs, and blends them with learned softmax weights. The gradient
    to the selection logits is d(loss)/d(blend) * cml_out_k -- no CML interior in the
    backward pass at all. This cleanly bypasses the chaotic gradient problem.

    Default K=3 candidates:
      - (0.15, 0.01): weak coupling + free-running (KS-optimal from sweep)
      - (0.30, 0.15): balanced default
      - (0.50, 0.30): strong coupling + strong drive (good for discrete CAs)
    """

    CANDIDATES = [
        (0.15, 0.01),   # KS-optimal
        (0.30, 0.15),   # default
        (0.50, 0.30),   # discrete-CA-optimal
    ]

    def __init__(self, in_channels: int = 1, steps: int = 15,
                 r: float = 3.90, seed: int = 42, kernel_size: int = 3,
                 mode: str = "global", candidates: list | None = None,
                 warm_start_idx: int | None = None, warm_start_logit: float = 2.0):
        super().__init__()
        self.in_channels = in_channels
        self.steps = steps
        self.kernel_size = kernel_size
        self.mode = mode
        if candidates is not None:
            self.CANDIDATES = list(candidates)
        self.K = len(self.CANDIDATES)

        # Build K frozen CML2D instances
        self.cmls = nn.ModuleList()
        for eps_k, beta_k in self.CANDIDATES:
            cml = CML2D(in_channels, steps, r, eps_k, beta_k, seed, kernel_size)
            for p in cml.parameters():
                p.requires_grad = False
            self.cmls.append(cml)

        # Learnable selection logits (optionally warm-started toward a candidate)
        init_logits = torch.zeros(self.K)
        if warm_start_idx is not None and 0 <= warm_start_idx < self.K:
            init_logits[warm_start_idx] = warm_start_logit
        if mode == "global":
            self.logits = nn.Parameter(init_logits.clone())
        else:  # percell
            self.gate = nn.Conv2d(in_channels, self.K, 3, padding=1)
            with torch.no_grad():
                self.gate.weight.zero_()
                self.gate.bias.copy_(init_logits)

    def forward(self, drive: torch.Tensor) -> torch.Tensor:
        B, C, H, W = drive.shape

        # Run K CML passes, detach each output (no gradient through CML)
        cml_outs = []
        for cml in self.cmls:
            with torch.no_grad():
                cml_outs.append(cml(drive))

        # Concat mode: stack all K outputs as channels, return (B, K*C, H, W)
        # No blending — the downstream NCA can pick features via conv weights.
        if self.mode == "concat":
            return torch.cat(cml_outs, dim=1)  # (B, K*C, H, W)

        # (K, B, C, H, W) -> (B, K, C, H, W)
        cml_stack = torch.stack(cml_outs, dim=1)  # detached by construction

        # Compute blending weights
        if self.mode == "global":
            weights = F.softmax(self.logits, dim=0)  # (K,)
            # Reshape for broadcast: (1, K, 1, 1, 1)
            w = weights.view(1, self.K, 1, 1, 1)
        else:
            logits = self.gate(drive)  # (B, K, H, W)
            weights = F.softmax(logits, dim=1)  # (B, K, H, W)
            w = weights.unsqueeze(2)  # (B, K, 1, H, W)

        # Weighted blend: gradient flows through w only, not through cml_stack
        blended = (w * cml_stack).sum(dim=1)  # (B, C, H, W)
        return blended

    def get_selection_info(self) -> dict:
        """Return learned weights and effective (eps, beta)."""
        if self.mode == "global":
            weights = F.softmax(self.logits, dim=0)
            eps = sum(w.item() * c[0] for w, c in zip(weights, self.CANDIDATES))
            beta = sum(w.item() * c[1] for w, c in zip(weights, self.CANDIDATES))
            return {
                "weights": {f"({e:.2f},{b:.2f})": w.item()
                           for (e, b), w in zip(self.CANDIDATES, weights)},
                "effective_eps": eps,
                "effective_beta": beta,
            }
        return {"mode": "percell"}

    def param_count(self) -> dict[str, int]:
        if self.mode == "global":
            trained = self.K
        else:
            trained = sum(p.numel() for p in self.gate.parameters())
        frozen = sum(b.numel() for m in self.cmls for b in m.buffers())
        return {"trained": trained, "frozen": frozen}


class CML2DRandomReservoir(nn.Module):
    """K parallel frozen reservoirs with random coupling kernels. No oracle (eps, beta).

    Eliminates the two oracle heuristics in CML2DMultiConfig:
      1. No hand-picked (eps, beta) per reservoir — all K share a single default.
      2. No warm-start — gate init is uniform.

    Diversity comes entirely from K distinct random coupling kernels (one per
    reservoir, distinct RNG seeds). Gradient isolation (no_grad) is preserved.

    Two modes for local dynamics:
      - ``preserved``: keeps logistic f(x) = r*x*(1-x). CA/physics inductive bias
        retained; only the coupling varies across reservoirs.
      - ``full``: drops the logistic, ESN-style tanh recurrence on [-1, 1]-centered
        grid. No physics-specific nonlinearity; all dynamical diversity is random.
    """

    def __init__(self, in_channels: int = 1, K: int = 8, steps: int = 15,
                 r: float = 3.90, eps: float = 0.3, beta: float = 0.15,
                 seed: int = 42, kernel_size: int = 3,
                 mode: str = "preserved", conditioned: bool = False,
                 cond_hidden: int = 8, gate_mode: str = "learned"):
        super().__init__()
        assert mode in ("preserved", "full"), f"Unknown mode {mode}"
        assert gate_mode in ("learned", "uniform"), f"Unknown gate_mode {gate_mode}"
        self.in_channels = in_channels
        self.K = K
        self.steps = steps
        self.kernel_size = kernel_size
        self.mode = mode
        self.conditioned = conditioned
        self.gate_mode = gate_mode

        self.register_buffer("r", torch.tensor(r))
        self.register_buffer("eps", torch.tensor(eps))
        self.register_buffer("beta", torch.tensor(beta))

        kernels = []
        for k in range(K):
            rng = torch.Generator().manual_seed(seed + 10_000 * (k + 1))
            K_raw = torch.rand(in_channels, 1, kernel_size, kernel_size,
                               generator=rng).abs()
            K_norm = K_raw / K_raw.sum(dim=(-1, -2), keepdim=True)
            kernels.append(K_norm)
        self.register_buffer("kernels", torch.stack(kernels, dim=0))

        if gate_mode == "learned":
            self.logits = nn.Parameter(torch.zeros(K))
            if conditioned:
                self.cond_gate = nn.Sequential(
                    nn.Linear(3, cond_hidden),
                    nn.ReLU(),
                    nn.Linear(cond_hidden, K),
                )
                nn.init.zeros_(self.cond_gate[-1].weight)
                nn.init.zeros_(self.cond_gate[-1].bias)

    def _run_batched(self, drive: torch.Tensor) -> torch.Tensor:
        """Run all K reservoirs in parallel via a single grouped conv2d per step.

        Replaces the K-way Python loop with one conv2d call that has K*C groups —
        PyTorch/OpenMP parallelises across groups internally, giving ~K× speedup
        on CPU and much more on MPS/CUDA.

        Output shape: (B, K, C, H, W).
        """
        B, C, H, W = drive.shape
        K = self.K
        pad = self.kernel_size // 2
        # (K, C, 1, kh, kw) -> (K*C, 1, kh, kw) for grouped conv with K*C groups
        weight = self.kernels.reshape(K * C, 1, self.kernel_size, self.kernel_size)
        drive_rep = drive.repeat(1, K, 1, 1)  # (B, K*C, H, W)
        r, eps, beta = self.r, self.eps, self.beta

        if self.mode == "preserved":
            grid = drive_rep
            for _ in range(self.steps):
                mapped = r * grid * (1.0 - grid)
                local = F.conv2d(mapped, weight, padding=pad, groups=K * C)
                physics = (1 - eps) * mapped + eps * local
                grid = (1 - beta) * physics + beta * drive_rep
            out = grid.clamp(1e-4, 1 - 1e-4)
        else:  # full
            drive_c = drive_rep * 2.0 - 1.0
            grid = drive_c
            for _ in range(self.steps):
                local = F.conv2d(grid, weight, padding=pad, groups=K * C)
                physics = (1 - eps) * grid + eps * local
                mixed = (1 - beta) * physics + beta * drive_c
                grid = torch.tanh(mixed)
            out = ((grid + 1.0) / 2.0).clamp(1e-4, 1 - 1e-4)

        return out.reshape(B, K, C, H, W)

    def _input_stats(self, drive: torch.Tensor) -> torch.Tensor:
        """Per-sample stats for the conditioned gate: (mean, var, grad-norm). Shape (B, 3)."""
        B = drive.shape[0]
        mean = drive.reshape(B, -1).mean(dim=1)
        var = drive.reshape(B, -1).var(dim=1)
        dy = drive[:, :, 1:, :] - drive[:, :, :-1, :]
        dx = drive[:, :, :, 1:] - drive[:, :, :, :-1]
        gnorm = (dy.pow(2).reshape(B, -1).mean(dim=1)
                 + dx.pow(2).reshape(B, -1).mean(dim=1)).sqrt()
        return torch.stack([mean, var, gnorm], dim=1)

    def forward(self, drive: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            stack = self._run_batched(drive)  # (B, K, C, H, W)

        if self.gate_mode == "uniform":
            # Strict 1/K averaging — zero trainable gate params.
            w = torch.full((self.K,), 1.0 / self.K,
                           device=stack.device, dtype=stack.dtype).view(1, self.K, 1, 1, 1)
            return (w * stack).sum(dim=1)

        if self.conditioned:
            stats = self._input_stats(drive)  # (B, 3)
            logits = self.logits.unsqueeze(0) + self.cond_gate(stats)  # (B, K)
            logits = logits.clamp(-20.0, 20.0)  # prevent runaway softmax
            w = F.softmax(logits, dim=1).view(-1, self.K, 1, 1, 1)
        else:
            w = F.softmax(self.logits, dim=0).view(1, self.K, 1, 1, 1)
        return (w * stack).sum(dim=1)

    def get_selection_info(self) -> dict:
        if self.gate_mode == "uniform":
            u = 1.0 / self.K
            return {
                "mode": self.mode, "K": self.K, "gate_mode": "uniform",
                "conditioned": False,
                "weights": [u] * self.K, "top_idx": 0, "top_weight": u,
                "entropy": float(torch.log(torch.tensor(float(self.K))).item()),
            }
        weights = F.softmax(self.logits, dim=0).detach()
        w_list = weights.tolist()
        return {
            "mode": self.mode,
            "K": self.K,
            "gate_mode": "learned",
            "conditioned": self.conditioned,
            "weights": w_list,
            "top_idx": int(weights.argmax().item()),
            "top_weight": float(weights.max().item()),
            "entropy": float(-(weights * (weights + 1e-12).log()).sum().item()),
        }

    def param_count(self) -> dict[str, int]:
        if self.gate_mode == "uniform":
            trained = 0
        else:
            trained = self.K
            if self.conditioned:
                trained += sum(p.numel() for p in self.cond_gate.parameters())
        return {
            "trained": trained,
            "frozen": sum(b.numel() for b in self.buffers()),
        }


class CML2DMultiR(nn.Module):
    """K parallel frozen CMLs sharing a single coupling kernel but with K different r values.

    "Vanilla rescor" scaling via chaos-depth diversity. Each of K reservoirs runs the
    SAME coupling kernel (sum-to-1 local averaging, identical to vanilla CML2D at a
    given seed), the SAME (eps, beta), but a DIFFERENT logistic r in [r_lo, r_hi].
    Diversity comes purely from chaos depth, not random spatial structure — this keeps
    the CML's physics identity intact.

    r values are log-spaced (dense near r_lo, sparse near r_hi) — linear spacing puts
    too many points in the fully-chaotic regime. The r-sweep earlier showed Heat/GS
    prefer r=3.70, KS prefers r=3.57, discrete CAs prefer r>=3.90.

    Gate modes:
      - "learned":  K-way softmax with trainable logits (uniform init, no warm-start).
      - "uniform":  Strict 1/K averaging, zero trainable gate params.
    """

    def __init__(self, in_channels: int = 1, K: int = 8, steps: int = 15,
                 r_lo: float = 3.57, r_hi: float = 3.99,
                 eps: float = 0.3, beta: float = 0.15,
                 seed: int = 42, kernel_size: int = 3,
                 gate_mode: str = "learned"):
        super().__init__()
        assert gate_mode in ("learned", "uniform"), f"Unknown gate_mode {gate_mode}"
        self.in_channels = in_channels
        self.K = K
        self.steps = steps
        self.kernel_size = kernel_size
        self.gate_mode = gate_mode

        # K different r values linearly interpolated across [r_lo, r_hi]
        if K == 1:
            r_values = torch.tensor([0.5 * (r_lo + r_hi)])
        else:
            r_values = torch.linspace(r_lo, r_hi, K)
        self.register_buffer("r_values", r_values)
        self.register_buffer("eps", torch.tensor(eps))
        self.register_buffer("beta", torch.tensor(beta))

        # One shared coupling kernel — identical to vanilla CML2D at this seed.
        rng = torch.Generator().manual_seed(seed)
        K_raw = torch.rand(in_channels, 1, kernel_size, kernel_size,
                           generator=rng).abs()
        K_norm = K_raw / K_raw.sum(dim=(-1, -2), keepdim=True)
        self.register_buffer("K_local", K_norm)

        if gate_mode == "learned":
            self.logits = nn.Parameter(torch.zeros(K))

    def _run_batched(self, drive: torch.Tensor) -> torch.Tensor:
        """Run all K CMLs in parallel. Same coupling, different r per reservoir."""
        # Logistic map r·x·(1-x) is only stable on [0, 1]; x outside explodes
        # in ~6 iterations at r≈3.99. Clamp defensively so callers that inject
        # noise or feed unbounded features don't NaN the reservoir.
        drive = drive.clamp(0.0, 1.0)
        B, C, H, W = drive.shape
        K = self.K
        pad = self.kernel_size // 2

        # Broadcast the shared kernel K times (grouped conv with K*C groups)
        weight = self.K_local.repeat(K, 1, 1, 1)  # (K*C, 1, kh, kw)
        drive_rep = drive.repeat(1, K, 1, 1)  # (B, K*C, H, W)

        # r needs to be broadcast over (B, K*C, H, W):
        # r_values (K,) -> (1, K, 1, 1, 1) -> expand C times -> reshape to (1, K*C, 1, 1)
        r_b = self.r_values.view(1, K, 1, 1, 1).expand(1, K, C, 1, 1).reshape(1, K * C, 1, 1)
        eps, beta = self.eps, self.beta

        grid = drive_rep
        for _ in range(self.steps):
            mapped = r_b * grid * (1.0 - grid)
            local = F.conv2d(mapped, weight, padding=pad, groups=K * C)
            physics = (1 - eps) * mapped + eps * local
            grid = (1 - beta) * physics + beta * drive_rep
        out = grid.clamp(1e-4, 1 - 1e-4)
        return out.reshape(B, K, C, H, W)

    def forward(self, drive: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            stack = self._run_batched(drive)  # (B, K, C, H, W)

        if self.gate_mode == "uniform":
            w = torch.full((self.K,), 1.0 / self.K,
                           device=stack.device, dtype=stack.dtype).view(1, self.K, 1, 1, 1)
        else:
            w = F.softmax(self.logits, dim=0).view(1, self.K, 1, 1, 1)
        return (w * stack).sum(dim=1)

    def get_selection_info(self) -> dict:
        if self.gate_mode == "uniform":
            u = 1.0 / self.K
            return {
                "K": self.K, "gate_mode": "uniform",
                "r_values": self.r_values.tolist(),
                "weights": [u] * self.K, "top_idx": 0, "top_weight": u,
                "entropy": float(torch.log(torch.tensor(float(self.K))).item()),
            }
        weights = F.softmax(self.logits, dim=0).detach()
        return {
            "K": self.K, "gate_mode": "learned",
            "r_values": self.r_values.tolist(),
            "weights": weights.tolist(),
            "top_idx": int(weights.argmax().item()),
            "top_weight": float(weights.max().item()),
            "top_r": float(self.r_values[weights.argmax().item()].item()),
            "entropy": float(-(weights * (weights + 1e-12).log()).sum().item()),
        }

    def param_count(self) -> dict[str, int]:
        trained = 0 if self.gate_mode == "uniform" else self.K
        return {
            "trained": trained,
            "frozen": sum(b.numel() for b in self.buffers()),
        }


class CML2DHybridMrEsn(nn.Module):
    """Hybrid MR + ESN reservoir bank with uniform 1/K averaging.

    Combines two diversity axes:
      - K_mr vanilla logistic CMLs with SHARED coupling kernel, different r values
        (chaos-depth axis — wins on heat, gol, ks, wireworld)
      - K_esn tanh reservoirs with RANDOM coupling kernels, fixed (eps, beta)
        (random-spatial axis — wins on gray_scott)

    Both banks run under no_grad, outputs concatenated along the K-axis, then
    uniformly averaged (1/K weights — zero trainable gate params).

    Total K = K_mr + K_esn. Trained params = 321 (NCA only), same as vanilla rescor.
    """

    def __init__(self, in_channels: int = 1, K_mr: int = 16, K_esn: int = 16,
                 steps: int = 15, r_lo: float = 3.57, r_hi: float = 3.99,
                 eps: float = 0.3, beta: float = 0.15,
                 seed: int = 42, kernel_size: int = 3):
        super().__init__()
        self.in_channels = in_channels
        self.K_mr = K_mr
        self.K_esn = K_esn
        self.K = K_mr + K_esn
        self.steps = steps
        self.kernel_size = kernel_size

        if K_mr == 1:
            r_values = torch.tensor([0.5 * (r_lo + r_hi)])
        else:
            r_values = torch.linspace(r_lo, r_hi, K_mr) if K_mr > 0 else torch.empty(0)
        self.register_buffer("r_values", r_values)
        self.register_buffer("eps", torch.tensor(eps))
        self.register_buffer("beta", torch.tensor(beta))

        # MR shared coupling kernel (seed-determined, matches vanilla CML2D at this seed)
        rng = torch.Generator().manual_seed(seed)
        K_raw = torch.rand(in_channels, 1, kernel_size, kernel_size,
                           generator=rng).abs()
        K_norm = K_raw / K_raw.sum(dim=(-1, -2), keepdim=True)
        self.register_buffer("mr_kernel", K_norm)

        # ESN random kernels — distinct seeds, same pattern as CML2DRandomReservoir
        esn_kernels = []
        for k in range(K_esn):
            rng_k = torch.Generator().manual_seed(seed + 10_000 * (k + 1))
            K_raw = torch.rand(in_channels, 1, kernel_size, kernel_size,
                               generator=rng_k).abs()
            K_norm = K_raw / K_raw.sum(dim=(-1, -2), keepdim=True)
            esn_kernels.append(K_norm)
        if K_esn > 0:
            self.register_buffer("esn_kernels", torch.stack(esn_kernels, dim=0))
        else:
            self.register_buffer("esn_kernels", torch.empty(0))

    def _run_mr(self, drive: torch.Tensor) -> torch.Tensor:
        """K_mr logistic CMLs with shared coupling, different r values."""
        if self.K_mr == 0:
            return drive.new_empty(drive.shape[0], 0, drive.shape[1],
                                   drive.shape[2], drive.shape[3])
        B, C, H, W = drive.shape
        K = self.K_mr
        pad = self.kernel_size // 2
        weight = self.mr_kernel.repeat(K, 1, 1, 1)  # (K*C, 1, kh, kw)
        drive_rep = drive.repeat(1, K, 1, 1)  # (B, K*C, H, W)
        r_b = self.r_values.view(1, K, 1, 1, 1).expand(1, K, C, 1, 1).reshape(1, K * C, 1, 1)
        eps, beta = self.eps, self.beta

        grid = drive_rep
        for _ in range(self.steps):
            mapped = r_b * grid * (1.0 - grid)
            local = F.conv2d(mapped, weight, padding=pad, groups=K * C)
            physics = (1 - eps) * mapped + eps * local
            grid = (1 - beta) * physics + beta * drive_rep
        out = grid.clamp(1e-4, 1 - 1e-4)
        return out.reshape(B, K, C, H, W)

    def _run_esn(self, drive: torch.Tensor) -> torch.Tensor:
        """K_esn tanh reservoirs with random coupling, fixed (eps, beta)."""
        if self.K_esn == 0:
            return drive.new_empty(drive.shape[0], 0, drive.shape[1],
                                   drive.shape[2], drive.shape[3])
        B, C, H, W = drive.shape
        K = self.K_esn
        pad = self.kernel_size // 2
        weight = self.esn_kernels.reshape(K * C, 1, self.kernel_size, self.kernel_size)
        drive_rep = drive.repeat(1, K, 1, 1)
        drive_c = drive_rep * 2.0 - 1.0
        eps, beta = self.eps, self.beta

        grid = drive_c
        for _ in range(self.steps):
            local = F.conv2d(grid, weight, padding=pad, groups=K * C)
            physics = (1 - eps) * grid + eps * local
            mixed = (1 - beta) * physics + beta * drive_c
            grid = torch.tanh(mixed)
        out = ((grid + 1.0) / 2.0).clamp(1e-4, 1 - 1e-4)
        return out.reshape(B, K, C, H, W)

    def forward(self, drive: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            mr_stack = self._run_mr(drive)
            esn_stack = self._run_esn(drive)
        stack = torch.cat([mr_stack, esn_stack], dim=1)  # (B, K_mr+K_esn, C, H, W)
        w = 1.0 / float(self.K)
        return w * stack.sum(dim=1)

    def get_selection_info(self) -> dict:
        u = 1.0 / self.K
        return {
            "K": self.K, "K_mr": self.K_mr, "K_esn": self.K_esn,
            "gate_mode": "uniform",
            "r_values": self.r_values.tolist(),
            "weights": [u] * self.K, "top_idx": 0, "top_weight": u,
            "entropy": float(torch.log(torch.tensor(float(self.K))).item()),
        }

    def param_count(self) -> dict[str, int]:
        return {"trained": 0, "frozen": sum(b.numel() for b in self.buffers())}


class ResCorRensStatBank(nn.Module):
    """rescor_rens K=32 with stat-bank NCA (C-variant from deeper_nca_plan.md).

    Same K=32 r-ensemble frozen reservoir bank as rescor_mr_uniform, but the NCA
    sees full ensemble statistics across K reservoirs (not just the mean):
      - include_var=True:  NCA input = [x, cml_mean, cml_var, cml_min, cml_max]
      - include_var=False: NCA input = [x, cml_mean, cml_min, cml_max]

    Residual is against cml_mean (same as vanilla rescor_rens).
    Variance ablation (with vs. without) isolates whether ensemble spread carries
    task-relevant information beyond the mean.
    """

    def __init__(self, in_channels: int = 1, hidden_ch: int = 16,
                 cml_steps: int = 15,
                 r_lo: float = 3.57, r_hi: float = 3.99,
                 eps: float = 0.3, beta: float = 0.15,
                 seed: int = 42, out_channels: int | None = None,
                 use_sigmoid: bool = True, kernel_size: int = 3,
                 cml_K: int = 32, include_var: bool = True):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        assert in_channels == out_channels, (
            "ResCorRensStatBank requires in_channels == out_channels"
        )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_sigmoid = use_sigmoid
        self.include_var = include_var

        self.rens = CML2DMultiR(
            in_channels=in_channels, K=cml_K, steps=cml_steps,
            r_lo=r_lo, r_hi=r_hi, eps=eps, beta=beta,
            seed=seed, kernel_size=kernel_size, gate_mode="uniform",
        )

        # Stat count: mean + (var if include_var) + min + max
        n_stats = 4 if include_var else 3
        nca_in = in_channels + n_stats * in_channels  # x + stats
        self.nca = nn.Sequential(
            nn.Conv2d(nca_in, hidden_ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_ch, out_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            stack = self.rens._run_batched(x)  # (B, K, C, H, W)
            cml_mean = stack.mean(dim=1)
            cml_min = stack.min(dim=1).values
            cml_max = stack.max(dim=1).values
            if self.include_var:
                cml_var = stack.var(dim=1)
                stats = torch.cat([cml_mean, cml_var, cml_min, cml_max], dim=1)
            else:
                stats = torch.cat([cml_mean, cml_min, cml_max], dim=1)

        correction = self.nca(torch.cat([x, stats], dim=1))
        out = cml_mean + correction
        if self.use_sigmoid:
            out = torch.clamp(out, 0, 1)
        return out

    def param_count(self) -> dict[str, int]:
        trained = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(b.numel() for b in self.rens.buffers())
        return {"trained": trained, "frozen": frozen}


class ResCorRensDeep(nn.Module):
    """Deep stack of L rescor_rens stages (each = K=32 r-ensemble + NCA correction).

    Each stage is a self-contained ResCor cell: K frozen logistic CMLs with K
    different r values, uniformly 1/K averaged, plus a tiny NCA (321 params)
    that corrects the averaged output via residual addition. Stages are chained
    — later stages receive the prior stage's corrected prediction and refine it.

    Trained params: 321 * L (NCA only, no gate params).
    Frozen params: ~43 per stage (K r-values, shared coupling kernel, eps, beta).

    Requires in_channels == out_channels (state rolls through stages).
    """

    def __init__(self, in_channels: int = 1, hidden_ch: int = 16,
                 cml_steps: int = 15, r: float = 3.90,
                 r_lo: float = 3.57, r_hi: float = 3.99,
                 eps: float = 0.3, beta: float = 0.15,
                 seed: int = 42, out_channels: int | None = None,
                 use_sigmoid: bool = True, kernel_size: int = 3,
                 cml_K: int = 32, L: int = 2):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        assert in_channels == out_channels, (
            "ResCorRensDeep requires in_channels == out_channels "
            "(state rolls through stages)"
        )
        self.L = L
        self.out_channels = out_channels
        self.use_sigmoid = use_sigmoid

        self.rens_banks = nn.ModuleList()
        self.ncas = nn.ModuleList()
        for stage_idx in range(L):
            self.rens_banks.append(
                CML2DMultiR(
                    in_channels=in_channels, K=cml_K, steps=cml_steps,
                    r_lo=r_lo, r_hi=r_hi, eps=eps, beta=beta,
                    seed=seed + stage_idx, kernel_size=kernel_size,
                    gate_mode="uniform",
                )
            )
            self.ncas.append(
                nn.Sequential(
                    nn.Conv2d(in_channels * 2, hidden_ch, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(hidden_ch, out_channels, 1),
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        state = x
        for rens, nca in zip(self.rens_banks, self.ncas):
            cml_out = rens(state)
            correction = nca(torch.cat([state, cml_out], dim=1))
            state = cml_out + correction
            if self.use_sigmoid:
                state = torch.clamp(state, 0, 1)
        return state

    def param_count(self) -> dict[str, int]:
        trained = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(b.numel() for m in self.rens_banks for b in m.buffers())
        return {"trained": trained, "frozen": frozen}


class PureNCA(nn.Module):
    """Pure learned NCA without any CML component. Baseline.

    When ``out_channels`` differs from ``in_channels`` the NCA is no
    longer iterated — a single forward pass projects the input down to
    ``out_channels`` (this is the right mode for action-conditioned
    transition tasks like grid_world where the input is ``[state|action]``
    and the output is just the next state).

    Set ``use_sigmoid=False`` to obtain raw logits (required when the
    loss is cross-entropy; sigmoid squashes logits into ``[0, 1]`` which
    collapses the cross-entropy gradient and makes the model predict the
    majority class).
    """

    def __init__(self, in_channels: int = 1, hidden_ch: int = 16,
                 steps: int = 1, out_channels: int | None = None,
                 use_sigmoid: bool = True):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_sigmoid = use_sigmoid
        self._recurrent = (out_channels == in_channels)

        self.perceive = nn.Conv2d(in_channels, hidden_ch, 3, padding=1)
        layers: list[nn.Module] = [
            nn.ReLU(),
            nn.Conv2d(hidden_ch, out_channels, 1),
        ]
        if use_sigmoid:
            layers.append(nn.Sigmoid())
        self.update = nn.Sequential(*layers)
        self.steps = steps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._recurrent:
            for _ in range(self.steps):
                x = self.update(self.perceive(x))
            return x
        # Projection mode: single pass, in_channels -> out_channels
        return self.update(self.perceive(x))

    def param_count(self) -> dict[str, int]:
        return {
            "trained": sum(p.numel() for p in self.parameters()),
            "frozen": 0,
        }


class GatedBlendWM(nn.Module):
    """Per-cell gated blend of CML (frozen) and NCA (learned).

    When ``out_channels`` != ``in_channels`` (e.g. action-conditioned
    grid_world with ``in=8``, ``out=4``), CML operates on the first
    ``out_channels`` of the input (assumed to be the state channels);
    the NCA and the gate both project down to ``out_channels``. This
    mirrors the architecture of the dedicated ``grid_world_planning``
    experiment.

    Set ``use_sigmoid=False`` to get raw logits for cross-entropy losses.
    """

    def __init__(self, in_channels: int = 1, hidden_ch: int = 16,
                 cml_steps: int = 15, nca_steps: int = 1,
                 r: float = 3.90, eps: float = 0.3, beta: float = 0.15,
                 seed: int = 42, out_channels: int | None = None,
                 use_sigmoid: bool = True, kernel_size: int = 3):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_sigmoid = use_sigmoid
        self.nca_steps = nca_steps
        self._recurrent_nca = (out_channels == in_channels)

        # CML operates on the first out_channels channels (the "state")
        self.cml_2d = CML2D(out_channels, cml_steps, r, eps, beta, seed, kernel_size=kernel_size)

        self.nca_perceive = nn.Conv2d(in_channels, hidden_ch, 3, padding=1)
        nca_tail: list[nn.Module] = [
            nn.ReLU(),
            nn.Conv2d(hidden_ch, out_channels, 1),
        ]
        if use_sigmoid:
            nca_tail.append(nn.Sigmoid())
        self.nca_update = nn.Sequential(*nca_tail)

        # Gate sees: input (in_ch) + cml_out (out_ch) + nca_out (out_ch)
        gate_in = in_channels + out_channels * 2
        self.gate = nn.Sequential(
            nn.Conv2d(gate_in, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, out_channels, 1),
            nn.Sigmoid(),  # gate itself is always a [0,1] weighting
        )

    def _nca(self, x: torch.Tensor) -> torch.Tensor:
        if self._recurrent_nca:
            for _ in range(self.nca_steps):
                x = self.nca_update(self.nca_perceive(x))
            return x
        return self.nca_update(self.nca_perceive(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        state = x[:, : self.out_channels]
        cml_out = self.cml_2d(state)
        nca_out = self._nca(x)
        g = self.gate(torch.cat([x, cml_out, nca_out], dim=1))
        return g * cml_out + (1 - g) * nca_out

    def param_count(self) -> dict[str, int]:
        trained = sum(p.numel() for p in self.parameters())
        frozen = sum(b.numel() for b in self.cml_2d.buffers())
        return {"trained": trained, "frozen": frozen}


class CMLRegularizedNCA(nn.Module):
    """NCA with CML regularization during training.

    When ``out_channels`` != ``in_channels``, CML operates on the first
    ``out_channels`` of the input (the state); the NCA is no longer
    iterated recurrently but acts as a single projection from
    ``in_channels`` to ``out_channels``. Pass ``use_sigmoid=False`` for
    cross-entropy tasks.
    """

    def __init__(self, in_channels: int = 1, hidden_ch: int = 16,
                 nca_steps: int = 1,
                 r: float = 3.90, eps: float = 0.3, beta: float = 0.15,
                 seed: int = 42, out_channels: int | None = None,
                 use_sigmoid: bool = True):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_sigmoid = use_sigmoid
        self.nca_steps = nca_steps
        self._recurrent_nca = (out_channels == in_channels)

        self.nca_perceive = nn.Conv2d(in_channels, hidden_ch, 3, padding=1)
        nca_tail: list[nn.Module] = [
            nn.ReLU(),
            nn.Conv2d(hidden_ch, out_channels, 1),
        ]
        if use_sigmoid:
            nca_tail.append(nn.Sigmoid())
        self.nca_update = nn.Sequential(*nca_tail)

        # CML regularizer operates on the first out_channels (state)
        self.cml_2d = CML2D(out_channels, 15, r, eps, beta, seed)

    def _nca(self, x: torch.Tensor) -> torch.Tensor:
        if self._recurrent_nca:
            for _ in range(self.nca_steps):
                x = self.nca_update(self.nca_perceive(x))
            return x
        return self.nca_update(self.nca_perceive(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        nca_out = self._nca(x)
        if self.training:
            state = x[:, : self.out_channels]
            cml_ref = self.cml_2d(state)
            return nca_out, cml_ref
        return nca_out

    def param_count(self) -> dict[str, int]:
        trained = sum(p.numel() for p in self.parameters())
        frozen = sum(b.numel() for b in self.cml_2d.buffers())
        return {"trained": trained, "frozen": frozen}


class NCAInsideCML(nn.Module):
    """NCA replaces the logistic map inside CML's coupling structure.

    For the same-channel case this iterates a CML-style loop with the
    logistic map replaced by a learned NCA rule. For the heterogeneous
    case (``out_channels != in_channels``, e.g. action-conditioned
    grid_world) the CML recurrence runs over the first ``out_channels``
    of the input (the state), while the NCA rule at each step is
    conditioned on the auxiliary channels (e.g. the action field) which
    are held fixed throughout the rollout.

    Set ``use_sigmoid=False`` to emit raw logits for cross-entropy
    tasks. The internal NCA rule keeps its sigmoid so the CML recurrence
    stays bounded between steps, but for logit output we add a small
    learned head that projects the final bounded state to unbounded
    logits. This is required for action-conditioned planning: without
    a proper output head, ``logit(bounded_grid)`` is compressed to a
    narrow range and cannot confidently predict state changes against
    the strong ``beta * drive`` anchor.

    The ``beta`` drive anchor is also dropped on the final iteration
    for the heterogeneous branch. During training the CE loss will
    otherwise be minimized by simply copying the input state (since
    ~99 %% of cells don't change in a single action step) and the
    multi-step rollout catastrophically collapses — the agent
    disappears because at every step the output is pulled back toward
    the input's one-hot state.
    """

    def __init__(self, in_channels: int = 1, hidden_ch: int = 16,
                 steps: int = 5, eps: float = 0.3, beta: float = 0.15,
                 seed: int = 42, out_channels: int | None = None,
                 use_sigmoid: bool = True):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_sigmoid = use_sigmoid
        self.steps = steps

        self.register_buffer("eps", torch.tensor(eps))
        self.register_buffer("beta", torch.tensor(beta))

        rng = torch.Generator().manual_seed(seed)
        # Coupling kernel operates on the state channels only
        K_raw = torch.rand(out_channels, 1, 3, 3, generator=rng).abs()
        K_norm = K_raw / K_raw.sum(dim=(-1, -2), keepdim=True)
        self.register_buffer("K_local", K_norm)

        # NCA rule takes [state | aux] concatenation, outputs new state.
        # For the vanilla same-channel case this is the familiar
        # in->hidden->in mapping; for out_ch != in_ch it additionally
        # exposes the auxiliary channels to the rule at every step.
        self.nca_rule = nn.Sequential(
            nn.Conv2d(in_channels, hidden_ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_ch, out_channels, 1),
            nn.Sigmoid(),  # keep recurrence bounded
        )

        self._same_ch = (out_channels == in_channels)

        # Learned logit head for cross-entropy tasks: projects the final
        # bounded grid + aux back to unbounded logits. Only instantiated
        # when needed (use_sigmoid=False and out_ch != in_ch); this is
        # the action-conditioned classification path. For the
        # same-channel case ``use_sigmoid=False`` still falls back to the
        # logit transform to stay compatible with older behavior.
        if (not use_sigmoid) and (not self._same_ch):
            head_in = out_channels + (in_channels - out_channels)
            self.logit_head = nn.Sequential(
                nn.Conv2d(head_in, hidden_ch, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(hidden_ch, out_channels, 1),
            )
        else:
            self.logit_head = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        eps, beta = self.eps, self.beta
        if self._same_ch:
            drive = x
            grid = x
            for _ in range(self.steps):
                mapped = self.nca_rule(grid)
                local = F.conv2d(mapped, self.K_local, padding=1,
                                 groups=self.out_channels)
                physics = (1 - eps) * mapped + eps * local
                grid = (1 - beta) * physics + beta * drive
        else:
            state = x[:, : self.out_channels]
            aux = x[:, self.out_channels:]
            drive = state
            grid = state
            n = self.steps
            for i in range(n):
                # Condition the learned rule on aux at every step
                rule_in = torch.cat([grid, aux], dim=1)
                mapped = self.nca_rule(rule_in)
                local = F.conv2d(mapped, self.K_local, padding=1,
                                 groups=self.out_channels)
                physics = (1 - eps) * mapped + eps * local
                if i < n - 1:
                    # Intermediate step: mix in the drive anchor so the
                    # recurrence is stable and bounded.
                    grid = (1 - beta) * physics + beta * drive
                else:
                    # Final step: drop the drive anchor so the output
                    # is not pulled back to the input state. Without
                    # this the action-conditioned model can never
                    # confidently predict a state change.
                    grid = physics

        if not self.use_sigmoid:
            if self.logit_head is not None:
                # Heterogeneous (action-conditioned) branch: use the
                # learned logit head so the output has unbounded logits.
                aux = x[:, self.out_channels:]
                return self.logit_head(torch.cat([grid, aux], dim=1))
            # Same-channel fallback: inverse-sigmoid of the bounded state.
            grid = grid.clamp(1e-6, 1 - 1e-6)
            return torch.log(grid / (1 - grid))
        return grid

    def param_count(self) -> dict[str, int]:
        trained = sum(p.numel() for p in self.parameters())
        frozen = sum(b.numel() for b in self.buffers())
        return {"trained": trained, "frozen": frozen}


class ResidualCorrectionWM(nn.Module):
    """CML provides base prediction, NCA learns a correction.

    For the same-channel case this is ``cml(x) + nca([x, cml(x)])``
    clamped to ``[0, 1]`` — matching the original Phase 2 design.

    For the heterogeneous case (``out_channels != in_channels``, e.g.
    action-conditioned grid_world with ``in=8``, ``out=4``), CML
    operates on the first ``out_channels`` of the input (the state),
    and the NCA correction takes ``[input | cml_out]`` -> ``out_channels``.
    This is exactly the ``ActionConditionedResCor`` architecture from
    the dedicated ``grid_world_planning`` experiment.

    Set ``use_sigmoid=False`` to emit raw logits (no ``[0, 1]`` clamp):
    required when the loss is cross-entropy, since sigmoid / clamp
    collapses the logit gradient and drives the model to the majority
    class (empty cells, ~83.6% on grid_world).
    """

    def __init__(self, in_channels: int = 1, hidden_ch: int = 16,
                 cml_steps: int = 15,
                 r: float = 3.90, eps: float = 0.3, beta: float = 0.15,
                 seed: int = 42, out_channels: int | None = None,
                 use_sigmoid: bool = True, kernel_size: int = 3,
                 cml_channels: int = 1, cml_gate: str = "none",
                 cml_candidates: list | None = None,
                 cml_warm_start_idx: int | None = None,
                 cml_warm_start_logit: float = 2.0,
                 cml_K: int = 8):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_sigmoid = use_sigmoid
        self.cml_channels = cml_channels
        # Total CML channels = out_channels * cml_channels (replicate each state channel)
        total_cml_ch = out_channels * cml_channels

        # CML variant selection
        if cml_gate == "static":
            self.cml_2d = CML2DLearnedGateStatic(
                total_cml_ch, cml_steps, r, eps, beta,
                seed=seed, kernel_size=kernel_size)
        elif cml_gate == "dynamic":
            self.cml_2d = CML2DLearnedGateDynamic(
                total_cml_ch, cml_steps, r, eps, beta,
                seed=seed, kernel_size=kernel_size)
        elif cml_gate == "discrete_global":
            self.cml_2d = CML2DDiscreteSelect(
                total_cml_ch, cml_steps, r,
                seed=seed, kernel_size=kernel_size,
                mode="global", eps_default=eps, beta_default=beta)
        elif cml_gate == "discrete_percell":
            self.cml_2d = CML2DDiscreteSelect(
                total_cml_ch, cml_steps, r,
                seed=seed, kernel_size=kernel_size,
                mode="percell", eps_default=eps, beta_default=beta)
        elif cml_gate == "multi_config":
            self.cml_2d = CML2DMultiConfig(
                total_cml_ch, cml_steps, r,
                seed=seed, kernel_size=kernel_size,
                mode="global",
                candidates=cml_candidates,
                warm_start_idx=cml_warm_start_idx,
                warm_start_logit=cml_warm_start_logit)
        elif cml_gate == "multi_config_percell":
            self.cml_2d = CML2DMultiConfig(
                total_cml_ch, cml_steps, r,
                seed=seed, kernel_size=kernel_size,
                mode="percell",
                candidates=cml_candidates,
                warm_start_idx=cml_warm_start_idx,
                warm_start_logit=cml_warm_start_logit)
        elif cml_gate == "multi_config_concat":
            self.cml_2d = CML2DMultiConfig(
                total_cml_ch, cml_steps, r,
                seed=seed, kernel_size=kernel_size,
                mode="concat",
                candidates=cml_candidates)
        elif cml_gate == "random_reservoir_preserved":
            self.cml_2d = CML2DRandomReservoir(
                total_cml_ch, K=cml_K, steps=cml_steps,
                r=r, eps=eps, beta=beta,
                seed=seed, kernel_size=kernel_size,
                mode="preserved")
        elif cml_gate == "random_reservoir_full":
            self.cml_2d = CML2DRandomReservoir(
                total_cml_ch, K=cml_K, steps=cml_steps,
                r=r, eps=eps, beta=beta,
                seed=seed, kernel_size=kernel_size,
                mode="full")
        elif cml_gate == "random_reservoir_full_cond":
            self.cml_2d = CML2DRandomReservoir(
                total_cml_ch, K=cml_K, steps=cml_steps,
                r=r, eps=eps, beta=beta,
                seed=seed, kernel_size=kernel_size,
                mode="full", conditioned=True)
        elif cml_gate == "random_reservoir_full_uniform":
            self.cml_2d = CML2DRandomReservoir(
                total_cml_ch, K=cml_K, steps=cml_steps,
                r=r, eps=eps, beta=beta,
                seed=seed, kernel_size=kernel_size,
                mode="full", gate_mode="uniform")
        elif cml_gate == "multi_r":
            self.cml_2d = CML2DMultiR(
                total_cml_ch, K=cml_K, steps=cml_steps,
                eps=eps, beta=beta,
                seed=seed, kernel_size=kernel_size,
                gate_mode="learned")
        elif cml_gate == "multi_r_uniform":
            self.cml_2d = CML2DMultiR(
                total_cml_ch, K=cml_K, steps=cml_steps,
                eps=eps, beta=beta,
                seed=seed, kernel_size=kernel_size,
                gate_mode="uniform")
        elif cml_gate == "hybrid_mr_esn":
            K_half = cml_K // 2
            self.cml_2d = CML2DHybridMrEsn(
                total_cml_ch, K_mr=K_half, K_esn=cml_K - K_half,
                steps=cml_steps, eps=eps, beta=beta,
                seed=seed, kernel_size=kernel_size)
        else:
            self.cml_2d = CML2D(total_cml_ch, cml_steps, r, eps, beta, seed, kernel_size=kernel_size)

        # NCA input channel count — concat mode passes K CML outputs instead of 1
        if cml_gate == "multi_config_concat":
            K = len(cml_candidates) if cml_candidates else 3
            cml_out_channels = K * total_cml_ch
        else:
            cml_out_channels = total_cml_ch

        # NCA correction: [x | cml_out] -> out_channels
        self.nca = nn.Sequential(
            nn.Conv2d(in_channels + cml_out_channels, hidden_ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_ch, out_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        state = x[:, : self.out_channels]
        # Expand state: repeat each channel cml_channels times
        if self.cml_channels > 1:
            cml_input = state.repeat(1, self.cml_channels, 1, 1)
            # Add small per-channel noise so each trajectory diverges
            noise = torch.randn_like(cml_input) * 0.01
            cml_input = (cml_input + noise).clamp(1e-4, 1 - 1e-4)
        else:
            cml_input = state
        cml_out = self.cml_2d(cml_input)
        correction = self.nca(torch.cat([x, cml_out], dim=1))
        # Residual from first out_channels of CML output (canonical trajectory)
        out = cml_out[:, :self.out_channels] + correction
        if self.use_sigmoid:
            out = torch.clamp(out, 0, 1)
        return out

    def param_count(self) -> dict[str, int]:
        trained = sum(p.numel() for p in self.parameters())
        frozen = sum(b.numel() for b in self.cml_2d.buffers())
        return {"trained": trained, "frozen": frozen}


class CML2DWithStats(nn.Module):
    """CML2D that returns multiple statistics collected over the trajectory.

    Same dynamics as :class:`CML2D` — frozen logistic map + conv2d coupling —
    but instead of returning only the final grid state, it collects
    several per-cell statistics over the M-step trajectory:

    * ``last``       : final grid state (identical to ``CML2D.forward``)
    * ``mean``       : arithmetic mean across all M iterations
    * ``var``        : variance across all M iterations
    * ``delta``      : ``last - first`` (total change)
    * ``last_drive`` : ``last - drive`` (residual of physics from input)

    These are all zero-extra-compute by-products of the existing M-step
    loop and are used by :class:`ResidualCorrectionWMv2` to give the
    correction NCA richer temporal features for free.
    """

    def __init__(self, in_channels: int = 1, steps: int = 15,
                 r: float = 3.90, eps: float = 0.3, beta: float = 0.15,
                 seed: int = 42, kernel_size: int = 3):
        super().__init__()
        self.in_channels = in_channels
        self.steps = steps
        self.kernel_size = kernel_size

        self.register_buffer("r", torch.tensor(r))
        self.register_buffer("eps", torch.tensor(eps))
        self.register_buffer("beta", torch.tensor(beta))

        rng = torch.Generator().manual_seed(seed)
        K_raw = torch.rand(in_channels, 1, kernel_size, kernel_size, generator=rng).abs()
        K_norm = K_raw / K_raw.sum(dim=(-1, -2), keepdim=True)
        self.register_buffer("K_local", K_norm)

    def forward(self, drive: torch.Tensor) -> dict[str, torch.Tensor]:
        grid = drive
        first = drive
        r, eps, beta = self.r, self.eps, self.beta
        pad = self.kernel_size // 2
        states: list[torch.Tensor] = []
        for _ in range(self.steps):
            mapped = r * grid * (1.0 - grid)
            local = F.conv2d(mapped, self.K_local, padding=pad,
                             groups=self.in_channels)
            physics = (1 - eps) * mapped + eps * local
            grid = (1 - beta) * physics + beta * drive
            grid = grid.clamp(1e-4, 1 - 1e-4)
            states.append(grid)

        last = grid
        stacked = torch.stack(states, dim=0)  # (M, B, C, H, W)
        mean = stacked.mean(dim=0)
        var = stacked.var(dim=0, unbiased=False)
        delta = last - first
        last_drive = last - drive

        return {
            "last": last,
            "mean": mean,
            "var": var,
            "delta": delta,
            "last_drive": last_drive,
        }

    def param_count(self) -> dict[str, int]:
        return {"trained": 0, "frozen": sum(b.numel() for b in self.buffers())}


class ResidualCorrectionWMv2(nn.Module):
    """E2: ResCor with multiple CML stat readouts as NCA correction input.

    Compared to :class:`ResidualCorrectionWM`, this model collects five
    different statistics over the frozen CML trajectory (``last``, ``mean``,
    ``var``, ``delta``, ``last_drive``) and concatenates them with the
    raw input before feeding them to the correction NCA. The residual is
    then added to ``last`` (the final CML state), matching the ResCor
    contract.

    An extra 1x1 conv is inserted in the NCA for additional mixing
    capacity since the input channel count has ~5x grown; ``hidden_ch``
    is intentionally kept at the baseline size (32 by default, matching
    the "performance over params" directive).

    For the heterogeneous / action-conditioned case (``in_ch != out_ch``),
    CML runs on the first ``out_channels`` of the input and every stat is
    added to the residual correctly. The extra auxiliary channels of the
    raw input are concatenated into the NCA input as additional context,
    but only the first ``out_channels`` of the input are sliced out for
    the per-stat concatenation (so the NCA input channel count is
    deterministic: ``in_channels + 5 * out_channels``).
    """

    def __init__(self, in_channels: int = 1, hidden_ch: int = 32,
                 cml_steps: int = 15,
                 r: float = 3.90, eps: float = 0.3, beta: float = 0.15,
                 seed: int = 42, out_channels: int | None = None,
                 use_sigmoid: bool = True):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_sigmoid = use_sigmoid

        # Multi-stat CML operates on the first out_channels of x (the state)
        self.cml_2d = CML2DWithStats(out_channels, cml_steps, r, eps, beta, seed)

        # NCA correction input = raw input (in_ch) + 5 stats (each out_ch)
        nca_in = in_channels + 5 * out_channels
        self.nca = nn.Sequential(
            nn.Conv2d(nca_in, hidden_ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_ch, hidden_ch, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_ch, out_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        state = x[:, : self.out_channels]
        stats = self.cml_2d(state)

        nca_input = torch.cat(
            [
                x,
                stats["last"],
                stats["mean"],
                stats["var"],
                stats["delta"],
                stats["last_drive"],
            ],
            dim=1,
        )
        correction = self.nca(nca_input)
        out = stats["last"] + correction
        if self.use_sigmoid:
            out = torch.clamp(out, 0, 1)
        return out

    def param_count(self) -> dict[str, int]:
        trained = sum(p.numel() for p in self.nca.parameters())
        frozen = sum(b.numel() for b in self.cml_2d.buffers())
        return {"trained": trained, "frozen": frozen}


class ResidualCorrectionWMv3(nn.Module):
    """E2 + E4: multi-stat CML readouts + per-channel learned affine drive.

    Builds on :class:`ResidualCorrectionWMv2` (E2) by applying a learned
    per-channel affine transformation ``alpha * x + beta`` to the state
    **before** it is fed into the frozen CML as drive. The affine
    parameters sit outside the CML loop, so gradients flow into them
    through the residual correction path rather than through chaos.

    Why it helps (hypothesis):
        * The logistic map's interesting dynamics live in a narrow band
          around ``x ~ 0.5``; if the input sits near 0 or 1 the chaos
          collapses to a fixed point and the CML becomes a near-identity.
        * A learned per-channel affine lets the network position each
          channel inside the map's chaotic sweet spot.
        * Gradient flow is safe because the affine is upstream of the
          ``no_grad``-style chaos (shattered gradients avoided).

    Initialisation is identity (``alpha = 1``, ``beta = 0``) so the model
    starts exactly as E2 and only moves away if the data asks for it.

    Param cost vs E2: only ``+2 * out_channels`` trainable parameters
    (alpha + beta, one per channel).
    """

    def __init__(self, in_channels: int = 1, hidden_ch: int = 32,
                 cml_steps: int = 15,
                 r: float = 3.90, eps: float = 0.3, beta: float = 0.15,
                 seed: int = 42, out_channels: int | None = None,
                 use_sigmoid: bool = True):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_sigmoid = use_sigmoid

        # E4: per-channel affine drive (learned). Initialise as identity
        # so the model starts as exactly E2 and only moves when the
        # downstream loss asks for it.
        self.drive_alpha = nn.Parameter(torch.ones(1, out_channels, 1, 1))
        self.drive_beta = nn.Parameter(torch.zeros(1, out_channels, 1, 1))

        # E2: multi-stat readout CML operates on the first out_channels
        # of the (affined) state.
        self.cml_2d = CML2DWithStats(out_channels, cml_steps, r, eps, beta, seed)

        # NCA correction input = raw input (in_ch) + 5 stats (each out_ch)
        nca_in = in_channels + 5 * out_channels
        self.nca = nn.Sequential(
            nn.Conv2d(nca_in, hidden_ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_ch, hidden_ch, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_ch, out_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract the state channels (first out_channels of x)
        state = x[:, : self.out_channels]

        # E4: apply learned affine before CML. Clamp to [0, 1] because
        # the CML expects bounded inputs.
        affined_drive = self.drive_alpha * state + self.drive_beta
        affined_drive = torch.clamp(affined_drive, 0.0, 1.0)

        # E2: multi-stat readouts from the frozen CML
        stats = self.cml_2d(affined_drive)

        nca_input = torch.cat(
            [
                x,
                stats["last"],
                stats["mean"],
                stats["var"],
                stats["delta"],
                stats["last_drive"],
            ],
            dim=1,
        )
        correction = self.nca(nca_input)
        out = stats["last"] + correction
        if self.use_sigmoid:
            out = torch.clamp(out, 0, 1)
        return out

    def param_count(self) -> dict[str, int]:
        trained = (
            sum(p.numel() for p in self.nca.parameters())
            + self.drive_alpha.numel()
            + self.drive_beta.numel()
        )
        frozen = sum(b.numel() for b in self.cml_2d.buffers())
        return {"trained": trained, "frozen": frozen}


class CML2DGrouped(nn.Module):
    """E1: Multi-r CML filterbank with G chaos personalities.

    Splits the incoming channels into ``n_groups`` equal-size groups and
    runs each group through its own :class:`CML2DWithStats` with a
    different ``(r, eps, beta)`` tuple. The output stats are concatenated
    along the channel dimension so that downstream consumers can treat the
    result as a single multi-stat readout of width ``n_groups *
    channels_per_group``.

    Group physics personalities (fixed; matches arch_plan.md E1):
      * Group 0 (anchor)    : r=3.20, eps=0.10, beta=0.6
      * Group 1 (smoother)  : r=3.50, eps=0.15, beta=0.5
      * Group 2 (transport) : r=3.69, eps=0.25, beta=0.3
      * Group 3 (edge)      : r=3.85, eps=0.20, beta=0.4

    Each group also uses a slightly different seed so the frozen coupling
    kernel ``K_local`` differs across the filterbank.
    """

    _R_VALUES = (3.20, 3.50, 3.69, 3.85)
    _EPS_VALUES = (0.10, 0.15, 0.25, 0.20)
    _BETA_VALUES = (0.6, 0.5, 0.3, 0.4)

    def __init__(self, channels_per_group: int = 1, n_groups: int = 4,
                 steps: int = 15, seed: int = 42):
        super().__init__()
        if n_groups > len(self._R_VALUES):
            raise ValueError(
                f"n_groups={n_groups} exceeds defined personalities "
                f"({len(self._R_VALUES)})"
            )
        self.n_groups = n_groups
        self.channels_per_group = channels_per_group
        self.total_channels = n_groups * channels_per_group

        self.cmls = nn.ModuleList([
            CML2DWithStats(
                in_channels=channels_per_group,
                steps=steps,
                r=self._R_VALUES[g],
                eps=self._EPS_VALUES[g],
                beta=self._BETA_VALUES[g],
                seed=seed + g,
            )
            for g in range(n_groups)
        ])

    def forward(self, drive: torch.Tensor) -> dict[str, torch.Tensor]:
        # drive: (B, n_groups * channels_per_group, H, W)
        groups = drive.chunk(self.n_groups, dim=1)

        per_group_stats = [self.cmls[g](groups[g]) for g in range(self.n_groups)]

        combined: dict[str, torch.Tensor] = {}
        for key in ("last", "mean", "var", "delta", "last_drive"):
            combined[key] = torch.cat(
                [per_group_stats[g][key] for g in range(self.n_groups)], dim=1
            )
        return combined

    def param_count(self) -> dict[str, int]:
        return {
            "trained": 0,
            "frozen": sum(b.numel() for b in self.buffers()),
        }


class ResidualCorrectionWMv6(nn.Module):
    """E1 + E2 + E6: Multi-r chaos groups + multi-stat readouts + per-group
    block-diagonal correction.

    Architecture:
      1. A learned 1x1 input projection lifts the state channels to
         ``n_groups * channels_per_group`` so that each chaos group gets
         its own dedicated slice of the CML input.
      2. :class:`CML2DGrouped` runs ``n_groups`` frozen CMLs (E1) in
         parallel, each with its own ``(r, eps, beta)`` personality and
         its own ``CML2DWithStats`` multi-stat readout (E2).
      3. Per-group NCA corrections (E6): each group has its own small
         NCA that sees the raw input ``x`` plus that group's five stats
         and produces a ``channels_per_group`` correction. The block-
         diagonal structure prevents cross-group crosstalk inside the
         NCA.
      4. A final 1x1 conv mixes the concatenated per-group outputs to
         produce the final ``out_channels`` prediction.

    For the heterogeneous / action-conditioned case (``in_ch != out_ch``),
    the CML operates on a projected version of the first ``out_channels``
    of ``x`` (the state), and each per-group NCA additionally sees the
    raw input ``x`` for auxiliary channels (e.g. one-hot action fields).
    """

    def __init__(self, in_channels: int = 1, hidden_ch: int = 32,
                 cml_steps: int = 15, n_groups: int = 4,
                 seed: int = 42, out_channels: int | None = None,
                 use_sigmoid: bool = True):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_sigmoid = use_sigmoid
        self.n_groups = n_groups

        # At least one channel per group
        self.channels_per_group = max(1, out_channels)
        self.cml_total_channels = n_groups * self.channels_per_group

        # 1x1 projection from state -> grouped CML drive
        self.input_proj = nn.Conv2d(out_channels, self.cml_total_channels, 1)

        # E1 + E2: grouped multi-r CML with multi-stat readouts
        self.cml_2d = CML2DGrouped(
            channels_per_group=self.channels_per_group,
            n_groups=n_groups,
            steps=cml_steps,
            seed=seed,
        )

        # E6: Per-group block-diagonal NCA corrections.
        # Each group sees: full raw input (in_channels) + 5 stats from
        # THAT group only (5 * channels_per_group).
        per_group_in = in_channels + 5 * self.channels_per_group
        self.group_ncas = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(per_group_in, hidden_ch, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(hidden_ch, self.channels_per_group, 1),
            )
            for _ in range(n_groups)
        ])

        # 1x1 mix across groups to produce final output
        self.output_mix = nn.Conv2d(
            n_groups * self.channels_per_group, out_channels, 1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        state = x[:, : self.out_channels]

        # Project state -> grouped CML input, squash to [0,1] so the
        # frozen CML receives a valid logistic-map drive.
        cml_drive = torch.sigmoid(self.input_proj(state))

        # Run grouped CML; each stat has shape
        # (B, n_groups * channels_per_group, H, W).
        stats = self.cml_2d(cml_drive)

        # Per-group NCA corrections.
        group_outputs: list[torch.Tensor] = []
        for g in range(self.n_groups):
            start = g * self.channels_per_group
            end = (g + 1) * self.channels_per_group
            group_input = torch.cat(
                [
                    x,
                    stats["last"][:, start:end],
                    stats["mean"][:, start:end],
                    stats["var"][:, start:end],
                    stats["delta"][:, start:end],
                    stats["last_drive"][:, start:end],
                ],
                dim=1,
            )
            group_outputs.append(self.group_ncas[g](group_input))

        all_groups = torch.cat(group_outputs, dim=1)
        out = self.output_mix(all_groups)

        if self.use_sigmoid:
            out = torch.clamp(out, 0, 1)
        return out

    def param_count(self) -> dict[str, int]:
        trained = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(b.numel() for b in self.buffers())
        return {"trained": trained, "frozen": frozen}


class ResidualCorrectionWMv7(nn.Module):
    """E2 + E3: multi-stat readouts + dilated NCA correction (multi-scale RF).

    Builds on :class:`ResidualCorrectionWMv2` (E2) by replacing the single
    3x3 perception conv with two *parallel* 3x3 convs — one with
    ``dilation=1`` (fine / local receptive field) and one with
    ``dilation=2`` (coarser 5x5 effective RF). Their outputs are
    concatenated along the channel dim, giving the correction NCA access
    to both local and mid-range spatial context without a large param
    blowup.

    Each perception branch emits ``hidden_ch // 2`` channels so that the
    concatenated hidden representation has ``hidden_ch`` channels and the
    downstream 1x1 mixing layers are identical to E2 (same params).

    Hypothesis (from arch_plan.md Extension 3):
        * Helps continuous PDEs with features that interact over 5-10
          cells — e.g. ``gray_scott`` (spot-to-spot), ``pde_wave``
          (front travelling > 1 cell/step).
        * Neutral on purely local rules (``pde_heat``, ``gol``).

    BECAUSE: CML's frozen ``K_local`` is 3x3, so after ``M`` steps its
    physical receptive field is ``(2M+1)^2`` in the accumulated state.
    With a pure 3x3 perception the correction can only compare locally;
    a dilation=2 branch lets it compare at roughly the distance travelled
    by one CML coupling step over two hops, matching the physics-side
    RF growth more closely.
    """

    def __init__(self, in_channels: int = 1, hidden_ch: int = 32,
                 cml_steps: int = 15,
                 r: float = 3.90, eps: float = 0.3, beta: float = 0.15,
                 seed: int = 42, out_channels: int | None = None,
                 use_sigmoid: bool = True):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_sigmoid = use_sigmoid

        # E2: multi-stat CML on the first out_channels of x (the state)
        self.cml_2d = CML2DWithStats(out_channels, cml_steps, r, eps, beta, seed)

        # E3: parallel dilated 3x3 perception branches.
        # Each branch takes (in_channels + 5 * out_channels) -> hidden_ch//2.
        # padding=dilation ensures same spatial dims are preserved.
        nca_in = in_channels + 5 * out_channels
        half_h = hidden_ch // 2
        self.perceive_d1 = nn.Conv2d(nca_in, half_h, 3, padding=1, dilation=1)
        self.perceive_d2 = nn.Conv2d(nca_in, half_h, 3, padding=2, dilation=2)

        # Update head (identical shape to E2): 1x1 mix -> 1x1 project.
        self.update = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(hidden_ch, hidden_ch, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_ch, out_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        state = x[:, : self.out_channels]
        stats = self.cml_2d(state)

        nca_input = torch.cat(
            [
                x,
                stats["last"],
                stats["mean"],
                stats["var"],
                stats["delta"],
                stats["last_drive"],
            ],
            dim=1,
        )

        # E3: multi-scale perception
        feat_d1 = self.perceive_d1(nca_input)
        feat_d2 = self.perceive_d2(nca_input)
        feat = torch.cat([feat_d1, feat_d2], dim=1)
        correction = self.update(feat)

        out = stats["last"] + correction
        if self.use_sigmoid:
            out = torch.clamp(out, 0, 1)
        return out

    def param_count(self) -> dict[str, int]:
        trained = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(b.numel() for b in self.cml_2d.buffers())
        return {"trained": trained, "frozen": frozen}


class ResidualCorrectionWMv9(nn.Module):
    """E2 + E3c: Multi-stat readouts + zero-init residual dilation with WD-on-alpha.

    Same as v8 (E3b) but the dilation alpha parameter is named
    ``dilation_alpha`` so training code can identify it and apply
    strong L2 weight decay. The hypothesis is that E3b drifted to
    nonzero alpha on grid_world because there was no cost to using
    the dilated branch; with strong weight decay on alpha, the model
    should only adopt dilation when the loss benefit clearly outweighs
    the L2 penalty.

    Architecture and forward are otherwise identical to
    :class:`ResidualCorrectionWMv8`.
    """

    def __init__(self, in_channels: int = 1, hidden_ch: int = 32,
                 cml_steps: int = 15,
                 r: float = 3.90, eps: float = 0.3, beta: float = 0.15,
                 seed: int = 42, out_channels: int | None = None,
                 use_sigmoid: bool = True):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_sigmoid = use_sigmoid

        # E2: multi-stat CML on the first out_channels of x (the state)
        self.cml_2d = CML2DWithStats(out_channels, cml_steps, r, eps, beta, seed)

        nca_in = in_channels + 5 * out_channels

        # Standard 3x3 dilation=1 perception — full hidden_ch capacity.
        self.perceive_d1 = nn.Conv2d(nca_in, hidden_ch, 3, padding=1, dilation=1)
        # Zero-init residual 3x3 dilation=2 branch.
        self.perceive_d2 = nn.Conv2d(nca_in, hidden_ch, 3, padding=2, dilation=2)

        # LayerScale-style per-channel alpha gate, init 0, so at init
        # the dilated branch contributes nothing. Named ``dilation_alpha``
        # (rather than ``alpha``) so the training loop can pull it out and
        # apply strong weight decay selectively.
        self.dilation_alpha = nn.Parameter(torch.zeros(1, hidden_ch, 1, 1))

        # Update head (same shape as E2): 1x1 mix -> 1x1 project.
        self.update = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(hidden_ch, hidden_ch, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_ch, out_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        state = x[:, : self.out_channels]
        stats = self.cml_2d(state)

        nca_input = torch.cat(
            [
                x,
                stats["last"],
                stats["mean"],
                stats["var"],
                stats["delta"],
                stats["last_drive"],
            ],
            dim=1,
        )

        # Standard local branch.
        h1 = self.perceive_d1(nca_input)
        # Zero-init residual dilated branch (dilation_alpha gates contribution).
        h2 = self.perceive_d2(nca_input) * self.dilation_alpha

        feat = h1 + h2  # additive residual, NOT concatenation
        correction = self.update(feat)

        out = stats["last"] + correction
        if self.use_sigmoid:
            out = torch.clamp(out, 0, 1)
        return out

    def get_alpha_params(self) -> list[nn.Parameter]:
        """Return just the dilation alpha parameters (for selective WD)."""
        return [self.dilation_alpha]

    def param_count(self) -> dict[str, int]:
        trained = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(b.numel() for b in self.cml_2d.buffers())
        return {"trained": trained, "frozen": frozen}


# =========================================================================
# Trajectory Attention: learned per-cell aggregation over CML trajectory
# =========================================================================


class CML2DWithTrajectory(nn.Module):
    """CML2D that returns hand-crafted stats (last, delta) + raw trajectory.

    Same dynamics as :class:`CML2DWithStats` — frozen logistic map + conv2d
    coupling — but returns only the two cheapest hand-crafted statistics
    (``last``, ``delta``) alongside the full stacked trajectory
    ``(B, M, C, H, W)`` for downstream learned aggregation.
    """

    def __init__(self, in_channels: int = 1, steps: int = 15,
                 r: float = 3.90, eps: float = 0.3, beta: float = 0.15,
                 seed: int = 42, kernel_size: int = 3):
        super().__init__()
        self.in_channels = in_channels
        self.steps = steps
        self.kernel_size = kernel_size

        self.register_buffer("r", torch.tensor(r))
        self.register_buffer("eps", torch.tensor(eps))
        self.register_buffer("beta", torch.tensor(beta))

        rng = torch.Generator().manual_seed(seed)
        K_raw = torch.rand(in_channels, 1, kernel_size, kernel_size, generator=rng).abs()
        K_norm = K_raw / K_raw.sum(dim=(-1, -2), keepdim=True)
        self.register_buffer("K_local", K_norm)

    def forward(self, drive: torch.Tensor) -> dict[str, torch.Tensor]:
        grid = drive
        first = drive
        r, eps, beta = self.r, self.eps, self.beta
        pad = self.kernel_size // 2
        states: list[torch.Tensor] = []
        for _ in range(self.steps):
            mapped = r * grid * (1.0 - grid)
            local = F.conv2d(mapped, self.K_local, padding=pad,
                             groups=self.in_channels)
            physics = (1 - eps) * mapped + eps * local
            grid = (1 - beta) * physics + beta * drive
            grid = grid.clamp(1e-4, 1 - 1e-4)
            states.append(grid)

        last = grid
        delta = last - first
        trajectory = torch.stack(states, dim=1)  # (B, M, C, H, W)

        return {"last": last, "delta": delta, "trajectory": trajectory}

    def param_count(self) -> dict[str, int]:
        return {"trained": 0, "frozen": sum(b.numel() for b in self.buffers())}


class TrajectoryAttention(nn.Module):
    """Learned per-cell aggregation over CML trajectory via cross-attention.

    For each spatial position independently, uses the current state as a
    query and the M trajectory states as keys/values.  Three tiny 1x1 convs
    project to ``d_k``-dim keys/queries and ``d_v``-dim values, producing
    ``d_v`` learned features per cell.
    """

    def __init__(self, in_channels: int = 1, d_k: int = 3, d_v: int = 3):
        super().__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.W_q = nn.Conv2d(in_channels, d_k, 1)
        self.W_k = nn.Conv2d(in_channels, d_k, 1)
        self.W_v = nn.Conv2d(in_channels, d_v, 1)
        # Init small so attention starts near-uniform
        for m in [self.W_q, self.W_k, self.W_v]:
            nn.init.normal_(m.weight, std=0.1)
            nn.init.zeros_(m.bias)

    def forward(self, trajectory: torch.Tensor,
                x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            trajectory: ``(B, M, C, H, W)`` — stacked CML states.
            x: ``(B, C, H, W)`` — current state (query source).

        Returns:
            ``(B, d_v, H, W)`` — learned per-cell features.
        """
        B, M, C, H, W = trajectory.shape
        N = H * W

        # Query from input
        Q = self.W_q(x)  # (B, d_k, H, W)
        Q = Q.reshape(B, self.d_k, N).permute(0, 2, 1)  # (B, N, d_k)

        # Keys and values from trajectory
        traj_flat = trajectory.reshape(B * M, C, H, W)
        K = self.W_k(traj_flat).reshape(B, M, self.d_k, N)
        K = K.permute(0, 3, 1, 2)  # (B, N, M, d_k)
        V = self.W_v(traj_flat).reshape(B, M, self.d_v, N)
        V = V.permute(0, 3, 1, 2)  # (B, N, M, d_v)

        # Attention: (B, N, 1, d_k) @ (B, N, d_k, M) -> (B, N, 1, M)
        scores = torch.matmul(
            Q.unsqueeze(2), K.transpose(-1, -2),
        ) / (self.d_k ** 0.5)
        attn = F.softmax(scores, dim=-1)  # (B, N, 1, M)

        # Aggregate: (B, N, 1, M) @ (B, N, M, d_v) -> (B, N, 1, d_v)
        out = torch.matmul(attn, V).squeeze(2)  # (B, N, d_v)
        return out.permute(0, 2, 1).reshape(B, self.d_v, H, W)


class TrajectoryAttentionWM(nn.Module):
    """E2-traj: Hybrid hand-crafted + learned trajectory aggregation.

    Keeps ``last`` and ``delta`` (hand-crafted, 0 params), replaces
    ``mean`` / ``var`` / ``last_drive`` with ``d_v`` learned features via
    per-cell cross-attention over the M=15 CML trajectory.

    Total nca_input channels = ``in_channels + 2 * out_channels + d_v``.
    For the default ``in=out=1, d_v=3`` this equals 6, identical to E3c.

    Architecture otherwise matches :class:`ResidualCorrectionWMv9`:
    dual perception (d=1, d=2) with zero-init ``dilation_alpha`` and
    strong L2 WD on alpha.
    """

    def __init__(self, in_channels: int = 1, hidden_ch: int = 32,
                 cml_steps: int = 15,
                 r: float = 3.90, eps: float = 0.3, beta: float = 0.15,
                 seed: int = 42, out_channels: int | None = None,
                 use_sigmoid: bool = True, d_k: int = 3, d_v: int = 3):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_sigmoid = use_sigmoid

        self.cml_2d = CML2DWithTrajectory(
            out_channels, cml_steps, r, eps, beta, seed,
        )
        self.traj_attn = TrajectoryAttention(out_channels, d_k, d_v)

        # nca_input = [x, last, delta, d_v learned] = in_ch + 2*out_ch + d_v
        nca_in = in_channels + 2 * out_channels + d_v

        # Standard 3x3 dilation=1 perception — full hidden_ch capacity.
        self.perceive_d1 = nn.Conv2d(nca_in, hidden_ch, 3, padding=1,
                                     dilation=1)
        # Zero-init residual 3x3 dilation=2 branch.
        self.perceive_d2 = nn.Conv2d(nca_in, hidden_ch, 3, padding=2,
                                     dilation=2)

        # LayerScale-style per-channel alpha gate, init 0, so at init
        # the dilated branch contributes nothing. Named ``dilation_alpha``
        # so the training loop applies strong weight decay selectively.
        self.dilation_alpha = nn.Parameter(torch.zeros(1, hidden_ch, 1, 1))

        # Update head (same shape as E3c): 1x1 mix -> 1x1 project.
        self.update = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(hidden_ch, hidden_ch, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_ch, out_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        state = x[:, :self.out_channels]
        cml_out = self.cml_2d(state)

        # Hand-crafted: last, delta (always useful, 0 params)
        last = cml_out["last"]
        delta = cml_out["delta"]

        # Learned: d_v features from trajectory attention
        learned = self.traj_attn(cml_out["trajectory"], state)

        nca_input = torch.cat([x, last, delta, learned], dim=1)

        # Standard local branch.
        h1 = self.perceive_d1(nca_input)
        # Zero-init residual dilated branch (dilation_alpha gates contribution).
        h2 = self.perceive_d2(nca_input) * self.dilation_alpha

        feat = h1 + h2  # additive residual, NOT concatenation
        correction = self.update(feat)

        out = last + correction
        if self.use_sigmoid:
            out = torch.clamp(out, 0, 1)
        return out

    def get_alpha_params(self) -> list[nn.Parameter]:
        """Return just the dilation alpha parameters (for selective WD)."""
        return [self.dilation_alpha]

    def param_count(self) -> dict[str, int]:
        trained = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(b.numel() for b in self.cml_2d.buffers())
        return {"trained": trained, "frozen": frozen}


# =========================================================================
# MoE-RF: Mixture-of-Experts with CML-stat routing (replaces dilation_alpha)
# =========================================================================


class MoERFWorldModel(nn.Module):
    """MoE-RF: Per-cell CML-stat routing between d=1 and d=2 perception.

    Replaces the global ``dilation_alpha`` in rescor_e3c with a tiny
    1x1 router that reads CML statistics (5 x out_channels) and produces
    per-cell softmax weights over two experts (d=1, d=2).

    At init the router is zero-init so softmax outputs uniform 0.5/0.5,
    recovering the average of both branches — a softer start than E3c's
    pure-d1 init. If the router learns constant weights it recovers E3c.

    Param budget (in=1, out=1, hidden=32):
        perceive_d1  : Conv2d(6, 32, 3x3) = 1760
        perceive_d2  : Conv2d(6, 32, 3x3) = 1760
        router       : Conv2d(5, 2, 1x1)  = 12   (5*2 + 2)
        update[1]    : Conv2d(32, 32, 1x1) = 1056
        update[3]    : Conv2d(32, 1, 1x1)  = 33
        TOTAL        : 4621 trained
    """

    def __init__(self, in_channels: int = 1, hidden_ch: int = 32,
                 cml_steps: int = 15,
                 r: float = 3.90, eps: float = 0.3, beta: float = 0.15,
                 seed: int = 42, out_channels: int | None = None,
                 use_sigmoid: bool = True):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_sigmoid = use_sigmoid

        # Frozen CML with multi-stat readouts
        self.cml_2d = CML2DWithStats(out_channels, cml_steps, r, eps, beta, seed)

        nca_in = in_channels + 5 * out_channels

        # Two structurally different experts: d=1 (local) vs d=2 (wide RF)
        self.perceive_d1 = nn.Conv2d(nca_in, hidden_ch, 3, padding=1, dilation=1)
        self.perceive_d2 = nn.Conv2d(nca_in, hidden_ch, 3, padding=2, dilation=2)

        # Router: reads 5 CML stat channels, outputs 2 expert weights per cell.
        # Zero-init so softmax starts at uniform (0.5, 0.5).
        router_in = 5 * out_channels
        self.router = nn.Conv2d(router_in, 2, 1)
        nn.init.zeros_(self.router.weight)
        nn.init.zeros_(self.router.bias)

        # Shared update head (same as rescor_e3c)
        self.update = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(hidden_ch, hidden_ch, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_ch, out_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        state = x[:, : self.out_channels]
        stats = self.cml_2d(state)

        nca_input = torch.cat(
            [x, stats["last"], stats["mean"], stats["var"],
             stats["delta"], stats["last_drive"]],
            dim=1,
        )

        # Router input: all 5 CML stat channels
        router_input = torch.cat(
            [stats["last"], stats["mean"], stats["var"],
             stats["delta"], stats["last_drive"]],
            dim=1,
        )
        # (B, 2, H, W) -> softmax over expert dim
        weights = torch.softmax(self.router(router_input), dim=1)
        w1 = weights[:, 0:1, :, :]  # (B, 1, H, W)
        w2 = weights[:, 1:2, :, :]

        h1 = self.perceive_d1(nca_input)
        h2 = self.perceive_d2(nca_input)

        feat = w1 * h1 + w2 * h2  # per-cell weighted blend
        correction = self.update(feat)

        out = stats["last"] + correction
        if self.use_sigmoid:
            out = torch.clamp(out, 0, 1)
        return out

    def get_alpha_params(self) -> list[nn.Parameter]:
        """Router params get WD to regularize toward uniform routing."""
        return list(self.router.parameters())

    def param_count(self) -> dict[str, int]:
        trained = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(b.numel() for b in self.cml_2d.buffers())
        return {"trained": trained, "frozen": frozen}


class MoERFHomogeneousWorldModel(nn.Module):
    """MoE-RF-Homo: Ablation variant with K=2 same-architecture experts.

    Both experts use dilation=1, so any performance difference vs MoE-RF
    isolates the effect of structural diversity (d=1 vs d=2) from the
    effect of per-cell routing itself.

    Same router, same update head, same param count as MoE-RF.

    Param budget (in=1, out=1, hidden=32):
        expert_a     : Conv2d(6, 32, 3x3, d=1) = 1760
        expert_b     : Conv2d(6, 32, 3x3, d=1) = 1760
        router       : Conv2d(5, 2, 1x1)       = 12
        update[1]    : Conv2d(32, 32, 1x1)      = 1056
        update[3]    : Conv2d(32, 1, 1x1)       = 33
        TOTAL        : 4621 trained
    """

    def __init__(self, in_channels: int = 1, hidden_ch: int = 32,
                 cml_steps: int = 15,
                 r: float = 3.90, eps: float = 0.3, beta: float = 0.15,
                 seed: int = 42, out_channels: int | None = None,
                 use_sigmoid: bool = True):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_sigmoid = use_sigmoid

        self.cml_2d = CML2DWithStats(out_channels, cml_steps, r, eps, beta, seed)

        nca_in = in_channels + 5 * out_channels

        # Both experts: same architecture (d=1), different random init
        self.expert_a = nn.Conv2d(nca_in, hidden_ch, 3, padding=1, dilation=1)
        self.expert_b = nn.Conv2d(nca_in, hidden_ch, 3, padding=1, dilation=1)

        # Router: identical to MoE-RF
        router_in = 5 * out_channels
        self.router = nn.Conv2d(router_in, 2, 1)
        nn.init.zeros_(self.router.weight)
        nn.init.zeros_(self.router.bias)

        self.update = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(hidden_ch, hidden_ch, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_ch, out_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        state = x[:, : self.out_channels]
        stats = self.cml_2d(state)

        nca_input = torch.cat(
            [x, stats["last"], stats["mean"], stats["var"],
             stats["delta"], stats["last_drive"]],
            dim=1,
        )

        router_input = torch.cat(
            [stats["last"], stats["mean"], stats["var"],
             stats["delta"], stats["last_drive"]],
            dim=1,
        )
        weights = torch.softmax(self.router(router_input), dim=1)
        w1 = weights[:, 0:1, :, :]
        w2 = weights[:, 1:2, :, :]

        ha = self.expert_a(nca_input)
        hb = self.expert_b(nca_input)

        feat = w1 * ha + w2 * hb
        correction = self.update(feat)

        out = stats["last"] + correction
        if self.use_sigmoid:
            out = torch.clamp(out, 0, 1)
        return out

    def get_alpha_params(self) -> list[nn.Parameter]:
        """Router params get WD to regularize toward uniform routing."""
        return list(self.router.parameters())

    def param_count(self) -> dict[str, int]:
        trained = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(b.numel() for b in self.cml_2d.buffers())
        return {"trained": trained, "frozen": frozen}


# =========================================================================
# DeepResCor: Two-layer residual correction
# =========================================================================


class DeepResCorLite(nn.Module):
    """DeepResCor-Lite: Two-layer residual correction (no spatial gate).

    Layer 1 is a full rescor_e3c (dual perception + update head).
    Layer 2 is a tiny NCA that reads [x_state, h1] and produces a
    small additive correction scaled by ``depth_alpha`` (zero-init,
    WD=1.0).

    The second layer has NO CML — running CML on the already-corrected
    h1 produces near-identity dynamics (physically unjustified) and
    wastes compute.

    Param budget (in=1, out=1, hidden=32):
        --- Layer 1 (full rescor_e3c) ---
        perceive_d1   : Conv2d(6, 32, 3x3)  = 1760
        perceive_d2   : Conv2d(6, 32, 3x3)  = 1760
        dilation_alpha: (1, 32, 1, 1)        = 32
        update[1]     : Conv2d(32, 32, 1x1)  = 1056
        update[3]     : Conv2d(32, 1, 1x1)   = 33
        subtotal L1   : 4641

        --- Layer 2 (tiny NCA on [state, h1]) ---
        l2_perceive   : Conv2d(2, 8, 3x3)   = 152  (2*8*9 + 8)
        l2_update[1]  : Conv2d(8, 1, 1x1)   = 9    (8*1 + 1)
        depth_alpha   : scalar               = 1
        subtotal L2   : 162

        TOTAL         : 4803 trained (+3.5%)
    """

    def __init__(self, in_channels: int = 1, hidden_ch: int = 32,
                 l2_hidden: int = 8, cml_steps: int = 15,
                 r: float = 3.90, eps: float = 0.3, beta: float = 0.15,
                 seed: int = 42, out_channels: int | None = None,
                 use_sigmoid: bool = True):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_sigmoid = use_sigmoid

        # === Layer 1: full rescor_e3c ===
        self.cml_2d = CML2DWithStats(out_channels, cml_steps, r, eps, beta, seed)

        nca_in = in_channels + 5 * out_channels

        self.perceive_d1 = nn.Conv2d(nca_in, hidden_ch, 3, padding=1, dilation=1)
        self.perceive_d2 = nn.Conv2d(nca_in, hidden_ch, 3, padding=2, dilation=2)
        self.dilation_alpha = nn.Parameter(torch.zeros(1, hidden_ch, 1, 1))

        self.update = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(hidden_ch, hidden_ch, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_ch, out_channels, 1),
        )

        # === Layer 2: tiny NCA on [state, h1] ===
        l2_in = out_channels + out_channels  # [original state, L1 output]
        self.l2_perceive = nn.Conv2d(l2_in, l2_hidden, 3, padding=1)
        self.l2_update = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(l2_hidden, out_channels, 1),
        )

        # Scalar depth gate, zero-init, WD=1.0 → L2 starts as identity
        self.depth_alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        state = x[:, : self.out_channels]
        stats = self.cml_2d(state)

        nca_input = torch.cat(
            [x, stats["last"], stats["mean"], stats["var"],
             stats["delta"], stats["last_drive"]],
            dim=1,
        )

        # Layer 1: rescor_e3c
        h1_feat = self.perceive_d1(nca_input)
        h2_feat = self.perceive_d2(nca_input) * self.dilation_alpha
        feat = h1_feat + h2_feat
        correction1 = self.update(feat)
        h1 = stats["last"] + correction1  # L1 output

        # Layer 2: tiny NCA on [state, h1]
        l2_input = torch.cat([state, h1], dim=1)
        correction2 = self.l2_update(self.l2_perceive(l2_input))
        out = h1 + self.depth_alpha * correction2

        if self.use_sigmoid:
            out = torch.clamp(out, 0, 1)
        return out

    def get_alpha_params(self) -> list[nn.Parameter]:
        """Both dilation_alpha and depth_alpha get WD=1.0."""
        return [self.dilation_alpha, self.depth_alpha]

    def param_count(self) -> dict[str, int]:
        trained = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(b.numel() for b in self.cml_2d.buffers())
        return {"trained": trained, "frozen": frozen}


class DeepResCorGated(nn.Module):
    """DeepResCor-Gated: Two-layer residual correction with spatial gate.

    Same as DeepResCor-Lite but Layer 2's correction is spatially gated
    by CML uncertainty signals: ``sigmoid(Conv2d([var, last_drive]))``.

    This tests the hypothesis "CML variance = free uncertainty estimate":
    the model should learn to apply L2 corrections primarily where CML
    is uncertain (high variance) or where the physics residual is large.

    Param budget (in=1, out=1, hidden=32):
        --- Layer 1 (full rescor_e3c) ---
        perceive_d1   : Conv2d(6, 32, 3x3)  = 1760
        perceive_d2   : Conv2d(6, 32, 3x3)  = 1760
        dilation_alpha: (1, 32, 1, 1)        = 32
        update[1]     : Conv2d(32, 32, 1x1)  = 1056
        update[3]     : Conv2d(32, 1, 1x1)   = 33
        subtotal L1   : 4641

        --- Layer 2 (tiny NCA + spatial gate) ---
        l2_perceive   : Conv2d(2, 8, 3x3)   = 152
        l2_update[1]  : Conv2d(8, 1, 1x1)   = 9
        spatial_gate  : Conv2d(2, 1, 1x1)   = 3   (2*1 + 1)
        depth_alpha   : scalar               = 1
        subtotal L2   : 165

        TOTAL         : 4806 trained (+3.6%)
    """

    def __init__(self, in_channels: int = 1, hidden_ch: int = 32,
                 l2_hidden: int = 8, cml_steps: int = 15,
                 r: float = 3.90, eps: float = 0.3, beta: float = 0.15,
                 seed: int = 42, out_channels: int | None = None,
                 use_sigmoid: bool = True):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_sigmoid = use_sigmoid

        # === Layer 1: full rescor_e3c ===
        self.cml_2d = CML2DWithStats(out_channels, cml_steps, r, eps, beta, seed)

        nca_in = in_channels + 5 * out_channels

        self.perceive_d1 = nn.Conv2d(nca_in, hidden_ch, 3, padding=1, dilation=1)
        self.perceive_d2 = nn.Conv2d(nca_in, hidden_ch, 3, padding=2, dilation=2)
        self.dilation_alpha = nn.Parameter(torch.zeros(1, hidden_ch, 1, 1))

        self.update = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(hidden_ch, hidden_ch, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_ch, out_channels, 1),
        )

        # === Layer 2: tiny NCA + spatial gate ===
        l2_in = out_channels + out_channels
        self.l2_perceive = nn.Conv2d(l2_in, l2_hidden, 3, padding=1)
        self.l2_update = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(l2_hidden, out_channels, 1),
        )

        # Spatial gate from CML variance + last_drive (uncertainty signals).
        # Input: [var, last_drive] each out_channels -> 1 channel sigmoid gate.
        # Zero-init bias so gate starts at 0.5 (neutral).
        gate_in = 2 * out_channels
        self.spatial_gate = nn.Conv2d(gate_in, 1, 1)
        nn.init.zeros_(self.spatial_gate.weight)
        nn.init.zeros_(self.spatial_gate.bias)

        self.depth_alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        state = x[:, : self.out_channels]
        stats = self.cml_2d(state)

        nca_input = torch.cat(
            [x, stats["last"], stats["mean"], stats["var"],
             stats["delta"], stats["last_drive"]],
            dim=1,
        )

        # Layer 1: rescor_e3c
        h1_feat = self.perceive_d1(nca_input)
        h2_feat = self.perceive_d2(nca_input) * self.dilation_alpha
        feat = h1_feat + h2_feat
        correction1 = self.update(feat)
        h1 = stats["last"] + correction1

        # Layer 2: gated tiny NCA
        l2_input = torch.cat([state, h1], dim=1)
        correction2 = self.l2_update(self.l2_perceive(l2_input))

        # Spatial gate: sigmoid(f(var, last_drive)) -> (B, 1, H, W)
        gate_input = torch.cat([stats["var"], stats["last_drive"]], dim=1)
        gate = torch.sigmoid(self.spatial_gate(gate_input))

        out = h1 + self.depth_alpha * gate * correction2

        if self.use_sigmoid:
            out = torch.clamp(out, 0, 1)
        return out

    def get_alpha_params(self) -> list[nn.Parameter]:
        """dilation_alpha, depth_alpha, and spatial_gate params get WD=1.0."""
        return [self.dilation_alpha, self.depth_alpha,
                *list(self.spatial_gate.parameters())]

    def param_count(self) -> dict[str, int]:
        trained = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(b.numel() for b in self.cml_2d.buffers())
        return {"trained": trained, "frozen": frozen}


class ResidualCorrectionWMv8(nn.Module):
    """E2 + E3b: Multi-stat readouts + zero-init residual dilated branch.

    Builds on :class:`ResidualCorrectionWMv2` (E2) by adding a parallel
    3x3 ``dilation=2`` perception branch whose contribution is gated by a
    LayerScale-style per-channel ``alpha`` parameter initialised to zero.
    At initialisation the dilated branch contributes exactly nothing, so
    the model is equivalent to E2 — any departure from E2 has to be
    earned by training.

    Design points:
      1. Both branches use the FULL ``hidden_ch`` (not ``hidden_ch // 2``
         like E3/v7). This gives the d1 branch full capacity even when
         ``alpha`` stays near zero; the d2 branch is a pure additive
         residual on top.
      2. ``alpha`` is a per-channel scalar (LayerScale style): one
         learnable value per output channel of the perception hidden
         representation. Each channel can independently decide how much
         dilated context to use.
      3. Additive fusion (``h1 + h2 * alpha``), NOT concatenation. This
         preserves the "E2 fallback at init" property — no training step
         is required to recover E2 behaviour.
      4. Hypothesis: non-regressive on action-conditioned tasks where
         wide receptive field hurts (``grid_world``), and strictly
         better than E2 on PDEs where multi-scale features help
         (``heat``, ``gray_scott``, ``pde_wave``).

    Param cost vs E2: one extra Conv2d (3x3, ``nca_in -> hidden_ch``) +
    ``hidden_ch`` scalar alphas. Roughly ~1.5x E2 trained params.
    """

    def __init__(self, in_channels: int = 1, hidden_ch: int = 32,
                 cml_steps: int = 15,
                 r: float = 3.90, eps: float = 0.3, beta: float = 0.15,
                 seed: int = 42, out_channels: int | None = None,
                 use_sigmoid: bool = True):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_sigmoid = use_sigmoid

        # E2: multi-stat CML on the first out_channels of x (the state)
        self.cml_2d = CML2DWithStats(out_channels, cml_steps, r, eps, beta, seed)

        nca_in = in_channels + 5 * out_channels

        # Standard 3x3 dilation=1 perception — full hidden_ch capacity.
        self.perceive_d1 = nn.Conv2d(nca_in, hidden_ch, 3, padding=1, dilation=1)
        # Zero-init residual 3x3 dilation=2 branch.
        self.perceive_d2 = nn.Conv2d(nca_in, hidden_ch, 3, padding=2, dilation=2)

        # LayerScale-style per-channel alpha gate, init 0, so at init
        # the dilated branch contributes nothing.
        self.alpha = nn.Parameter(torch.zeros(1, hidden_ch, 1, 1))

        # Update head (same shape as E2): 1x1 mix -> 1x1 project.
        self.update = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(hidden_ch, hidden_ch, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_ch, out_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        state = x[:, : self.out_channels]
        stats = self.cml_2d(state)

        nca_input = torch.cat(
            [
                x,
                stats["last"],
                stats["mean"],
                stats["var"],
                stats["delta"],
                stats["last_drive"],
            ],
            dim=1,
        )

        # Standard local branch.
        h1 = self.perceive_d1(nca_input)
        # Zero-init residual dilated branch (alpha gates contribution).
        h2 = self.perceive_d2(nca_input) * self.alpha

        feat = h1 + h2  # additive residual, NOT concatenation
        correction = self.update(feat)

        out = stats["last"] + correction
        if self.use_sigmoid:
            out = torch.clamp(out, 0, 1)
        return out

    def param_count(self) -> dict[str, int]:
        trained = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(b.numel() for b in self.cml_2d.buffers())
        return {"trained": trained, "frozen": frozen}


# =========================================================================
# Matching-Principle Gate: per-cell trust between CML-based and NCA paths
# =========================================================================


class MatchingPrincipleGateWM(nn.Module):
    """Per-cell trust gate between CML-based and NCA-based correction.

    Tests whether the Matching Principle can be LEARNED rather than imposed.
    The trust gate uses CML trajectory statistics to decide per-cell whether
    to trust the CML-based correction (Path A) or the pure NCA correction
    (Path B).

    When trust is high (CML dynamics match target): output ~ CML path (A)
    When trust is low (CML dynamics don't match):   output ~ NCA path (B)

    Path A is a full :class:`ResidualCorrectionWMv9` (E3c) architecture:
    dual perception (d=1, d=2) with zero-init dilation alpha plus the
    2-layer 1x1 update head.  Path B is a minimal pure NCA (hidden_ch=8).
    The trust gate is a 2->4->1 MLP on CML ``var`` and ``last_drive``
    stats, zero-init so it starts at sigmoid(0)=0.5 (equal blend).

    Param budget (in=out=1, hidden=32):
        Path A (E3c):  4641 trained
        Path B (NCA):    89 trained
        Trust gate:      17 trained
        TOTAL:         4747 trained, 12 frozen
    """

    def __init__(self, in_channels: int = 1, hidden_ch: int = 32,
                 cml_steps: int = 15,
                 r: float = 3.90, eps: float = 0.3, beta: float = 0.15,
                 seed: int = 42, out_channels: int | None = None,
                 use_sigmoid: bool = True):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_sigmoid = use_sigmoid

        # Shared CML (frozen) — both paths read from the same stats
        self.cml_2d = CML2DWithStats(out_channels, cml_steps, r, eps, beta, seed)

        # ---- Path A: full E3c architecture -----------------------------------
        nca_in_a = in_channels + 5 * out_channels  # [x, 5 stats]

        # Standard 3x3 dilation=1 perception — full hidden_ch capacity.
        self.perceive_d1 = nn.Conv2d(nca_in_a, hidden_ch, 3, padding=1,
                                     dilation=1)
        # Zero-init residual 3x3 dilation=2 branch.
        self.perceive_d2 = nn.Conv2d(nca_in_a, hidden_ch, 3, padding=2,
                                     dilation=2)

        # LayerScale-style per-channel alpha gate, init 0, so at init
        # the dilated branch contributes nothing.
        self.dilation_alpha = nn.Parameter(torch.zeros(1, hidden_ch, 1, 1))

        # Update head (same shape as E3c): 1x1 mix -> 1x1 project.
        self.update_a = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(hidden_ch, hidden_ch, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_ch, out_channels, 1),
        )

        # ---- Path B: minimal pure NCA (no CML involvement) -------------------
        hc_b = 8
        self.perceive_b = nn.Conv2d(in_channels, hc_b, 3, padding=1)
        self.update_b = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(hc_b, out_channels, 1),
        )

        # ---- Trust gate: MLP on CML stats -> per-cell scalar -----------------
        # Input: 2 most discriminative stats (var + last_drive)
        self.trust_gate = nn.Sequential(
            nn.Conv2d(2 * out_channels, 4, 1),
            nn.ReLU(),
            nn.Conv2d(4, 1, 1),
            # No sigmoid here — applied in forward
        )
        # Zero-init the last layer so gate starts at sigmoid(0)=0.5
        nn.init.zeros_(self.trust_gate[-1].weight)
        nn.init.zeros_(self.trust_gate[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        state = x[:, :self.out_channels]
        stats = self.cml_2d(state)

        # ---- Path A: CML-based correction (E3c) -----------------------------
        nca_input_a = torch.cat(
            [
                x,
                stats["last"],
                stats["mean"],
                stats["var"],
                stats["delta"],
                stats["last_drive"],
            ],
            dim=1,
        )

        # Standard local branch.
        h1 = self.perceive_d1(nca_input_a)
        # Zero-init residual dilated branch (dilation_alpha gates contribution).
        h2 = self.perceive_d2(nca_input_a) * self.dilation_alpha

        feat = h1 + h2  # additive residual, NOT concatenation
        correction_a = self.update_a(feat)
        out_a = stats["last"] + correction_a

        # ---- Path B: pure NCA correction (no CML) ---------------------------
        correction_b = self.update_b(self.perceive_b(x))
        out_b = state + correction_b

        # ---- Trust gate from CML stats ---------------------------------------
        gate_input = torch.cat([stats["var"], stats["last_drive"]], dim=1)
        trust = torch.sigmoid(self.trust_gate(gate_input))  # (B, 1, H, W)

        # Blend: high trust -> use CML path, low trust -> use NCA path
        out = trust * out_a + (1 - trust) * out_b

        if self.use_sigmoid:
            out = torch.clamp(out, 0, 1)
        return out

    def get_alpha_params(self) -> list[nn.Parameter]:
        """Return dilation alpha only. Trust gate is NOT penalised — it must
        be free to move away from 0.5 so the model can learn to discriminate
        between CML-appropriate and NCA-appropriate dynamics."""
        return [self.dilation_alpha]

    def param_count(self) -> dict[str, int]:
        trained = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(b.numel() for b in self.cml_2d.buffers())
        return {"trained": trained, "frozen": frozen}


# ============================================================================
# rescor_mamba: rescor_rens K=32 spatial + per-cell Mamba SSM temporal context
# ============================================================================


class ResCorMamba(nn.Module):
    """rescor_rens K=32 spatial core + per-cell Mamba SSM temporal context.

    Input:  x_seq (B, context_k, C, H, W) — most recent frame last.
    Output: (B, C, H, W)                   — predicted next frame.

    Spatial: rens K=32 applied to x_seq[:, -1]. No temporal rens passes
    (the past is carried by Mamba).

    Temporal: at each spatial position, run a K-step selective-SSM over
    the context_k past frames. d_model=16, d_state=8, d_conv=4,
    expand=2 (d_inner=32).

    NCA correction: sees [x_now, cml_mean, mamba_feat] in mean-only mode.
    Residual against cml_mean.
    """

    def __init__(
        self,
        in_channels: int = 1,
        hidden_ch: int = 16,
        cml_steps: int = 15,
        r_lo: float = 3.57,
        r_hi: float = 3.99,
        eps: float = 0.3,
        beta: float = 0.15,
        seed: int = 42,
        out_channels: int | None = None,
        use_sigmoid: bool = True,
        kernel_size: int = 3,
        cml_K: int = 32,
        context_k: int = 4,
        d_model: int = 16,
        d_state: int = 8,
        d_conv: int = 4,
        expand: int = 2,
        zero_init_out: bool = True,
    ):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        assert in_channels == out_channels, (
            "ResCorMamba requires in_channels == out_channels"
        )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_sigmoid = use_sigmoid
        self.context_k = context_k
        self.d_model = d_model

        # Spatial core: rens K=32 (uniform 1/K averaging)
        self.rens = CML2DMultiR(
            in_channels=in_channels, K=cml_K, steps=cml_steps,
            r_lo=r_lo, r_hi=r_hi, eps=eps, beta=beta,
            seed=seed, kernel_size=kernel_size, gate_mode="uniform",
        )

        # Temporal: per-cell linear project to d_model, then Mamba block.
        self.in_proj = nn.Linear(in_channels, d_model, bias=True)
        self.mamba = MinimalMambaBlock(
            d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand,
            zero_init_out=zero_init_out,
        )

        # NCA correction: [x_now (C), cml_mean (C), mamba_feat (d_model)]
        nca_in = in_channels + in_channels + d_model
        self.nca = nn.Sequential(
            nn.Conv2d(nca_in, hidden_ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_ch, out_channels, 1),
        )

    def _mamba_feat(self, x_seq: torch.Tensor) -> torch.Tensor:
        """x_seq: (B, K, C, H, W) -> (B, d_model, H, W)."""
        B, K, C, H, W = x_seq.shape
        # (B, K, C, H, W) -> (B, H, W, K, C) -> (B*H*W, K, C)
        seq = x_seq.permute(0, 3, 4, 1, 2).reshape(B * H * W, K, C)
        seq = self.in_proj(seq)                      # (B*H*W, K, d_model)
        h_seq = self.mamba(seq)                      # (B*H*W, K, d_model)
        h_last = h_seq[:, -1, :]                     # (B*H*W, d_model)
        feat = h_last.reshape(B, H, W, self.d_model).permute(0, 3, 1, 2)
        return feat.contiguous()

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        # Accept (B, C, H, W) as a back-compat convenience: treat as K=1.
        if x_seq.dim() == 4:
            x_seq = x_seq.unsqueeze(1)
        assert x_seq.dim() == 5, f"Expected rank-5 (B,K,C,H,W); got {x_seq.shape}"
        x_now = x_seq[:, -1]                         # (B, C, H, W)

        with torch.no_grad():
            stack = self.rens._run_batched(x_now)    # (B, K_rens, C, H, W)
            cml_mean = stack.mean(dim=1)             # (B, C, H, W)

        temporal_feat = self._mamba_feat(x_seq)      # (B, d_model, H, W)

        nca_in = torch.cat([x_now, cml_mean, temporal_feat], dim=1)
        correction = self.nca(nca_in)
        out = cml_mean + correction
        if self.use_sigmoid:
            out = torch.clamp(out, 0, 1)
        return out

    def param_count(self) -> dict[str, int]:
        trained = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(b.numel() for b in self.rens.buffers())
        return {"trained": trained, "frozen": frozen}


class ResCorMambaStat(nn.Module):
    """ResCorMamba variant with stat-bank NCA.

    NCA input: [x_now, cml_mean, cml_min, cml_max, mamba_feat]
    (min/max over the K=32 reservoir bank, matching ResCorRensStatBank
    include_var=False layout).

    Residual: cml_mean (unchanged).
    """

    def __init__(
        self,
        in_channels: int = 1,
        hidden_ch: int = 16,
        cml_steps: int = 15,
        r_lo: float = 3.57,
        r_hi: float = 3.99,
        eps: float = 0.3,
        beta: float = 0.15,
        seed: int = 42,
        out_channels: int | None = None,
        use_sigmoid: bool = True,
        kernel_size: int = 3,
        cml_K: int = 32,
        context_k: int = 4,
        d_model: int = 16,
        d_state: int = 8,
        d_conv: int = 4,
        expand: int = 2,
        zero_init_out: bool = True,
    ):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        assert in_channels == out_channels, (
            "ResCorMambaStat requires in_channels == out_channels"
        )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_sigmoid = use_sigmoid
        self.context_k = context_k
        self.d_model = d_model

        self.rens = CML2DMultiR(
            in_channels=in_channels, K=cml_K, steps=cml_steps,
            r_lo=r_lo, r_hi=r_hi, eps=eps, beta=beta,
            seed=seed, kernel_size=kernel_size, gate_mode="uniform",
        )

        self.in_proj = nn.Linear(in_channels, d_model, bias=True)
        self.mamba = MinimalMambaBlock(
            d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand,
            zero_init_out=zero_init_out,
        )

        # NCA input: [x_now, cml_mean, cml_min, cml_max, mamba_feat]
        # 4·C + d_model channels.
        nca_in = in_channels * 4 + d_model
        self.nca = nn.Sequential(
            nn.Conv2d(nca_in, hidden_ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_ch, out_channels, 1),
        )

    def _mamba_feat(self, x_seq: torch.Tensor) -> torch.Tensor:
        B, K, C, H, W = x_seq.shape
        seq = x_seq.permute(0, 3, 4, 1, 2).reshape(B * H * W, K, C)
        seq = self.in_proj(seq)
        h_seq = self.mamba(seq)
        h_last = h_seq[:, -1, :]
        feat = h_last.reshape(B, H, W, self.d_model).permute(0, 3, 1, 2)
        return feat.contiguous()

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        if x_seq.dim() == 4:
            x_seq = x_seq.unsqueeze(1)
        assert x_seq.dim() == 5, f"Expected rank-5 (B,K,C,H,W); got {x_seq.shape}"
        x_now = x_seq[:, -1]

        with torch.no_grad():
            stack = self.rens._run_batched(x_now)   # (B, K_rens, C, H, W)
            cml_mean = stack.mean(dim=1)
            cml_min = stack.min(dim=1).values
            cml_max = stack.max(dim=1).values

        temporal_feat = self._mamba_feat(x_seq)
        nca_in = torch.cat(
            [x_now, cml_mean, cml_min, cml_max, temporal_feat], dim=1
        )
        correction = self.nca(nca_in)
        out = cml_mean + correction
        if self.use_sigmoid:
            out = torch.clamp(out, 0, 1)
        return out

    def param_count(self) -> dict[str, int]:
        trained = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(b.numel() for b in self.rens.buffers())
        return {"trained": trained, "frozen": frozen}


class ResCorMambaGated(nn.Module):
    """Drift-gated ResCorMamba: auto-attenuate Mamba contribution at high drift.

    Same backbone as :class:`ResCorMamba` (rens K=32 spatial core +
    per-cell Mamba SSM over context_k=4 past frames + NCA correction).
    The single architectural change is a per-cell sigmoid gate on the
    NCA correction:

        drift = sqrt( mean_C( (x_now - cml_mean)**2 ) )       # (B, 1, H, W)
        gate  = sigmoid( gate_scale * (-drift + gate_bias) )  # (B, 1, H, W)
        correction = NCA([x_now, cml_mean, mamba_feat]) * gate
        out = cml_mean + correction

    When predictions drift far from the rens "neutral" prediction
    (cml_mean), the gate closes (gate -> 0) and the model falls back
    to ``cml_mean`` (pure rens K=32 behavior — mediocre but never
    catastrophic). When near-manifold (drift -> 0), gate -> sigmoid(
    gate_bias) and the full Mamba contribution is preserved.

    Two new trainable scalars: ``gate_scale`` (steepness) and
    ``gate_bias`` (where the gate turns off). Initialized so the gate
    is non-degenerate at the start of training.

    Important: ``cml_mean`` is detached when computing ``drift`` — the
    rens reservoir has no trainable parameters but we still don't want
    gate-side gradients to flow through that subgraph.
    """

    def __init__(
        self,
        in_channels: int = 1,
        hidden_ch: int = 16,
        cml_steps: int = 15,
        r_lo: float = 3.57,
        r_hi: float = 3.99,
        eps: float = 0.3,
        beta: float = 0.15,
        seed: int = 42,
        out_channels: int | None = None,
        use_sigmoid: bool = True,
        kernel_size: int = 3,
        cml_K: int = 32,
        context_k: int = 4,
        d_model: int = 16,
        d_state: int = 8,
        d_conv: int = 4,
        expand: int = 2,
        zero_init_out: bool = True,
        gate_scale_init: float = 1.0,
        gate_bias_init: float = 0.5,
    ):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        assert in_channels == out_channels, (
            "ResCorMambaGated requires in_channels == out_channels"
        )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_sigmoid = use_sigmoid
        self.context_k = context_k
        self.d_model = d_model

        self.rens = CML2DMultiR(
            in_channels=in_channels, K=cml_K, steps=cml_steps,
            r_lo=r_lo, r_hi=r_hi, eps=eps, beta=beta,
            seed=seed, kernel_size=kernel_size, gate_mode="uniform",
        )

        self.in_proj = nn.Linear(in_channels, d_model, bias=True)
        self.mamba = MinimalMambaBlock(
            d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand,
            zero_init_out=zero_init_out,
        )

        nca_in = in_channels + in_channels + d_model
        self.nca = nn.Sequential(
            nn.Conv2d(nca_in, hidden_ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_ch, out_channels, 1),
        )

        # Two trainable scalars governing the drift-gated attenuation.
        # gate_scale starts small (so the gate is roughly flat in drift
        # space at init); gate_bias starts at 0.5 so initial gate value
        # at drift=0 is sigmoid(gate_scale*gate_bias) = sigmoid(0.5) ~ 0.62
        # — non-degenerate, allowing both training signal directions.
        self.gate_scale = nn.Parameter(torch.tensor(float(gate_scale_init)))
        self.gate_bias = nn.Parameter(torch.tensor(float(gate_bias_init)))

    def _mamba_feat(self, x_seq: torch.Tensor) -> torch.Tensor:
        """x_seq: (B, K, C, H, W) -> (B, d_model, H, W)."""
        B, K, C, H, W = x_seq.shape
        seq = x_seq.permute(0, 3, 4, 1, 2).reshape(B * H * W, K, C)
        seq = self.in_proj(seq)
        h_seq = self.mamba(seq)
        h_last = h_seq[:, -1, :]
        feat = h_last.reshape(B, H, W, self.d_model).permute(0, 3, 1, 2)
        return feat.contiguous()

    def compute_gate(self, x_now: torch.Tensor,
                     cml_mean: torch.Tensor | None = None) -> torch.Tensor:
        """Drift-gated sigmoid. Returns (B, 1, H, W) gate tensor.

        If ``x_now`` is rank-5 ``(B, K, C, H, W)`` (a context buffer), the
        most-recent frame is used (so callers in the multistep loop can
        pass the full state directly). If ``cml_mean`` is None, it is
        computed from the rens reservoir on the fly — useful for the
        MSDC drift-conditioned multistep loss path which doesn't have
        cml_mean cached.

        Detaches cml_mean so gate gradients don't propagate through the
        rens reservoir's subgraph (rens has no trainable params, but
        keeping the graph clean is cheap and removes a class of
        confusion).
        """
        if x_now.dim() == 5:
            x_now = x_now[:, -1]
        if cml_mean is None:
            with torch.no_grad():
                stack = self.rens._run_batched(x_now)
                cml_mean = stack.mean(dim=1)
        diff = x_now - cml_mean.detach()
        drift = (diff ** 2).mean(dim=1, keepdim=True).clamp_min(0.0).sqrt()
        gate = torch.sigmoid(self.gate_scale * (-drift + self.gate_bias))
        return gate

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        if x_seq.dim() == 4:
            x_seq = x_seq.unsqueeze(1)
        assert x_seq.dim() == 5, f"Expected rank-5 (B,K,C,H,W); got {x_seq.shape}"
        x_now = x_seq[:, -1]                         # (B, C, H, W)

        with torch.no_grad():
            stack = self.rens._run_batched(x_now)    # (B, K_rens, C, H, W)
            cml_mean = stack.mean(dim=1)             # (B, C, H, W)

        temporal_feat = self._mamba_feat(x_seq)      # (B, d_model, H, W)

        nca_in = torch.cat([x_now, cml_mean, temporal_feat], dim=1)
        correction = self.nca(nca_in)                # (B, C, H, W)

        gate = self.compute_gate(x_now, cml_mean)    # (B, 1, H, W)
        correction = correction * gate

        out = cml_mean + correction
        if self.use_sigmoid:
            out = torch.clamp(out, 0, 1)
        return out

    def param_count(self) -> dict[str, int]:
        trained = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(b.numel() for b in self.rens.buffers())
        return {"trained": trained, "frozen": frozen}
