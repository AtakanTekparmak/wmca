"""Minimal inline Mamba (selective-SSM) block for rescor_mamba.

Pure-PyTorch reference implementation of the Mamba block from
Gu & Dao 2023 (§3.2). Matches the math of `mamba-ssm`'s
`mamba_simple.py` but without the CUDA fast path — targeted at CPU and
short sequences (L = context_k = 4).

No external deps beyond torch.

Two implementations are provided:
    * `MinimalMambaBlock`     — fast CPU path. Replaces the depthwise
      `F.conv1d(groups=d_inner)` (which is shockingly slow on tiny K=4
      sequences) with an equivalent band-matrix `einsum`, and fuses the
      per-step discretization into the scan loop. ~2.5× faster end to
      end than the reference on B=16384, K=4. Output is numerically
      identical (max abs diff ≤ 1.2e-7).
    * `MinimalMambaBlockSlow` — original reference (Python loop + raw
      `F.conv1d`). Kept for A/B equivalence testing and as a fallback.

Both share the **exact same parameters** (constructors are identical),
so swapping between them does not change parameter count or layout.

Notation (matching mamba-ssm):
    d_model : outer width (width of the signal)
    d_inner : expand * d_model (width inside the block)
    d_state : SSM hidden-state width per channel
    d_conv  : causal conv kernel along L
    dt_rank : low-rank projection for dt (= max(d_model // 16, 1))

Input / output shape: (B, L, d_model)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_mamba_params(
    self: nn.Module,
    d_model: int,
    d_state: int,
    d_conv: int,
    expand: int,
    dt_rank: int | None,
    dt_min: float,
    dt_max: float,
    zero_init_out: bool,
) -> None:
    """Shared parameter initialization for the two block variants.

    Keeps both implementations bit-identical in parameter shape and
    initialization, so they can be swapped freely.
    """
    self.d_model = d_model
    self.d_state = d_state
    self.d_conv = d_conv
    self.expand = expand
    self.d_inner = d_model * expand
    if dt_rank is None:
        dt_rank = max(d_model // 16, 1)
    self.dt_rank = dt_rank

    # 1) Input projection  x -> [x, z]
    self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=True)

    # 2) Causal depthwise conv (groups = d_inner)
    self.conv1d = nn.Conv1d(
        in_channels=self.d_inner,
        out_channels=self.d_inner,
        kernel_size=d_conv,
        groups=self.d_inner,
        padding=0,  # we do explicit causal left-pad
        bias=True,
    )

    # 3) x_proj: produces [dt_raw (dt_rank), B (d_state), C (d_state)]
    self.x_proj = nn.Linear(self.d_inner, dt_rank + 2 * d_state, bias=True)

    # 4) dt_proj: dt_rank -> d_inner, with biased init so softplus(bias)
    #    ~ uniform(dt_min, dt_max)
    self.dt_proj = nn.Linear(dt_rank, self.d_inner, bias=True)
    dt = torch.exp(
        torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
        + math.log(dt_min)
    ).clamp(min=1e-4)
    inv_dt = dt + torch.log(-torch.expm1(-dt))
    with torch.no_grad():
        self.dt_proj.bias.copy_(inv_dt)

    # 5) A_log (trainable): init to log of 1..d_state (S4D real init).
    A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(
        self.d_inner, -1
    ).contiguous()
    self.A_log = nn.Parameter(torch.log(A))

    # 6) D (trainable skip), init to ones.
    self.D = nn.Parameter(torch.ones(self.d_inner))

    # 7) out_proj: d_inner -> d_model
    self.out_proj = nn.Linear(self.d_inner, d_model, bias=True)
    if zero_init_out:
        with torch.no_grad():
            self.out_proj.weight.zero_()
            if self.out_proj.bias is not None:
                self.out_proj.bias.zero_()


class MinimalMambaBlock(nn.Module):
    """Fast CPU inline selective-scan Mamba block.

    Mathematically equivalent to `MinimalMambaBlockSlow` (max abs diff
    ≤ 1.2e-7 on a smoke forward). Optimizations:

    1. Replace `F.conv1d(groups=d_inner)` with an equivalent band-matrix
       `einsum`. The raw conv is ~16× slower than einsum on small K=4
       length sequences with groups=d_inner=32 — most of the time is
       per-channel-group dispatch overhead, not actual FLOPs.
    2. Fuse the discretization into the scan loop: compute `dA = exp(dt
       * A)` and `dB = dt * B_` inside the per-step body so we never
       materialize the full `(B, L, d_inner, d_state)` tensors that the
       reference allocates upfront.

    Same constructor signature and parameter layout as the reference.
    """

    def __init__(
        self,
        d_model: int = 16,
        d_state: int = 8,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: int | None = None,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        zero_init_out: bool = True,
    ):
        super().__init__()
        _make_mamba_params(
            self, d_model, d_state, d_conv, expand,
            dt_rank, dt_min, dt_max, zero_init_out,
        )
        # Cache for the conv band-matrix gather indices (depend on
        # (L, d_conv) only). Set lazily in forward.
        self._conv_gidx: torch.Tensor | None = None
        self._conv_valid: torch.Tensor | None = None
        self._conv_cache_L: int = -1

    def _conv_indices(self, L: int, device, dtype):
        """Build / cache index tensors for the conv band-matrix.

        For `F.conv1d(left_pad=d_conv-1, kernel=d_conv)` over a length-L
        sequence with cross-correlation, output position l (0..L-1)
        depends on input positions max(0, l-d_conv+1)..l with kernel
        weights `w[k]` where `k = d_conv-1 - l + j` for input index j.
        We turn this into a fixed `(L, L)` gather plus mask, identical
        across batches, so the conv becomes a single
        `einsum('dlk,bkd->bld', M, x_in)`.
        """
        # Cache is invalid if length changes OR the cached tensors are on
        # a different device/dtype than the current call (the module may
        # have been moved between devices since the cache was populated).
        cache_valid = (
            self._conv_cache_L == L
            and self._conv_gidx is not None
            and self._conv_gidx.device == device
            and self._conv_valid is not None
            and self._conv_valid.dtype == dtype
        )
        if cache_valid:
            return self._conv_gidx, self._conv_valid
        Kc = self.d_conv
        l_idx = torch.arange(L, device=device).unsqueeze(1)
        k_idx = torch.arange(L, device=device).unsqueeze(0)
        g_idx = Kc - 1 - l_idx + k_idx
        valid = (g_idx >= 0) & (g_idx < Kc)
        g_idx_clamped = g_idx.clamp(0, Kc - 1)
        valid_t = valid.to(dtype)
        self._conv_gidx = g_idx_clamped
        self._conv_valid = valid_t
        self._conv_cache_L = L
        return g_idx_clamped, valid_t

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, d_model) -> (B, L, d_model)."""
        B, L, D = x.shape
        assert D == self.d_model, f"expected d_model={self.d_model}, got {D}"

        # 1) Input projection
        xz = self.in_proj(x)                              # (B, L, 2·d_inner)
        x_in, z = xz.chunk(2, dim=-1)                     # each (B, L, d_inner)

        # 2) Causal depthwise conv via band-matrix einsum.
        #    Equivalent to F.conv1d with groups=d_inner and left-pad
        #    (d_conv-1), but ~16× faster on CPU at L=4.
        w = self.conv1d.weight.squeeze(1)                 # (d_inner, d_conv)
        g_idx, valid = self._conv_indices(L, x.device, w.dtype)
        # M[d, l, k] = w[d, d_conv-1-l+k] if in-range else 0  -> (d_inner, L, L)
        M = w[:, g_idx] * valid
        x_conv = torch.einsum("dlk,bkd->bld", M, x_in) + self.conv1d.bias
        x_in = F.silu(x_conv)

        # 3) Per-step selective parameters
        x_dbl = self.x_proj(x_in)                         # (B, L, dt_rank+2·d_state)
        dt_raw, B_, C_ = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        dt = F.softplus(self.dt_proj(dt_raw))             # (B, L, d_inner)
        A = -torch.exp(self.A_log)                        # (d_inner, d_state)

        # 4) Selective scan with fused per-step discretization.
        #    h_l = exp(dt_l * A) * h_{l-1} + (dt_l * B_l) * x_in_l
        #    y_l = einsum("bdn,bn->bd", h_l, C_l)
        h = torch.zeros(
            B, self.d_inner, self.d_state, device=x.device, dtype=x.dtype
        )
        outs = []
        for l_idx in range(L):
            dt_l = dt[:, l_idx].unsqueeze(-1)             # (B, d_inner, 1)
            dA = torch.exp(dt_l * A)                      # (B, d_inner, d_state)
            dB = dt_l * B_[:, l_idx].unsqueeze(1)         # (B, d_inner, d_state)
            h = dA * h + dB * x_in[:, l_idx, :, None]
            outs.append(torch.einsum("bdn,bn->bd", h, C_[:, l_idx]))
        y = torch.stack(outs, dim=1)                      # (B, L, d_inner)

        # 5) Skip + gate + output
        y = y + self.D * x_in
        y = y * F.silu(z)
        return self.out_proj(y)

    def param_count(self) -> dict[str, int]:
        return {
            "trained": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "frozen": sum(p.numel() for p in self.parameters() if not p.requires_grad),
        }


class MinimalMambaBlockSlow(nn.Module):
    """Reference (slow) inline selective-scan Mamba block.

    Kept as a fallback / equivalence anchor for `MinimalMambaBlock`.
    The Python scan loop and raw `F.conv1d` make this several × slower
    on CPU at L=4, but the math is identical (this is the original
    implementation that shipped before the CPU optimization pass).

    Args:
        d_model: outer width.
        d_state: SSM state width per channel.
        d_conv:  causal conv kernel width.
        expand:  d_inner = expand · d_model.
        dt_rank: low-rank width for dt projection (default: max(d_model//16, 1)).
        dt_min, dt_max: softplus-biased initial dt range.
        zero_init_out: if True, initialize out_proj to zero — the block
            contributes nothing at init. Recommended for residual use so
            training starts at the base model's behavior.
    """

    def __init__(
        self,
        d_model: int = 16,
        d_state: int = 8,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: int | None = None,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        zero_init_out: bool = True,
    ):
        super().__init__()
        _make_mamba_params(
            self, d_model, d_state, d_conv, expand,
            dt_rank, dt_min, dt_max, zero_init_out,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, d_model) -> (B, L, d_model)."""
        B, L, D = x.shape
        assert D == self.d_model, f"expected d_model={self.d_model}, got {D}"

        # 1) Input projection
        xz = self.in_proj(x)                          # (B, L, 2·d_inner)
        x_in, z = xz.chunk(2, dim=-1)                 # each (B, L, d_inner)

        # 2) Causal depthwise conv along L.
        x_conv = x_in.transpose(1, 2)                 # (B, d_inner, L)
        x_conv = F.pad(x_conv, (self.d_conv - 1, 0))  # left-pad on L axis
        x_conv = self.conv1d(x_conv)                  # (B, d_inner, L)
        x_conv = x_conv.transpose(1, 2)               # (B, L, d_inner)
        x_in = F.silu(x_conv)

        # 3) Per-step selective parameters
        x_dbl = self.x_proj(x_in)                     # (B, L, dt_rank + 2·d_state)
        dt_raw, B_, C_ = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        dt = F.softplus(self.dt_proj(dt_raw))         # (B, L, d_inner)
        A = -torch.exp(self.A_log)                    # (d_inner, d_state)

        # 4) Discretize: per (b, l, d, n)
        deltaA = torch.exp(torch.einsum("bld,dn->bldn", dt, A))    # (B,L,d_inner,d_state)
        deltaB = torch.einsum("bld,bln->bldn", dt, B_)              # (B,L,d_inner,d_state)

        # 5) Sequential scan over L
        h = torch.zeros(B, self.d_inner, self.d_state,
                        device=x.device, dtype=x.dtype)
        outs = []
        for l_idx in range(L):
            h = deltaA[:, l_idx] * h + deltaB[:, l_idx] * x_in[:, l_idx, :, None]
            y_l = torch.einsum("bdn,bn->bd", h, C_[:, l_idx])
            outs.append(y_l)
        y = torch.stack(outs, dim=1)                  # (B, L, d_inner)

        # 6) Skip + gate
        y = y + self.D * x_in                         # (B, L, d_inner)
        y = y * F.silu(z)                             # gated

        # 7) Output projection
        out = self.out_proj(y)                        # (B, L, d_model)
        return out

    def param_count(self) -> dict[str, int]:
        return {
            "trained": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "frozen": sum(p.numel() for p in self.parameters() if not p.requires_grad),
        }
