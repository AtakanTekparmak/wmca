"""ResCorformer — Sliding-Window Attention + 2D RoPE with rescor as FFN.

Scale-up architecture #4 from `arch_plan.md` ("rescorformer — SWA + RoPE").
Target: 64x64 prediction. A single transformer block (L=1) where
  - Attention = Sliding Window Attention (SWA, window=8 patches)
  - Positional encoding = 2D Rotary Position Embedding (row/col factorized),
    written inline below with no external dependencies
  - FFN is REPLACED by a rescor residual on the unpatchified 64x64 grid

This scaffold is a PURE DRAFT. It does not import from `wmca.modules.*`;
instead the CML bank + NCA head are injected via the constructor and marked
with `# TODO` where the real `CML2DMultiR` drop-in will happen.

Design choices (matches arch_plan.md spec):
  - patch_size = 4x4, input 64x64 -> 16x16 = 256 tokens
  - d_model = 32, n_heads = 4 (d_head = 8)
  - window = 8 patches (captures ~2-patch neighborhood in each direction)
  - 2D RoPE: apply rotary indep to ROW index and COL index on disjoint halves
    of the head dimension (head_dim = 8 -> 4 dims row-rot, 4 dims col-rot, so
    each axis gets 2 rotary pairs). Zero extra params; resolution-agnostic.
  - Rescor FFN: LN -> unpatch proj to (B, 1, 64, 64) -> sigmoid -> CML bank
    -> NCA correction -> residual add back in token space. The "stats.last +
    correction" pattern from arch_plan.md is captured as (cml_out + nca_out).

Param budget (spec target: ~7.4K trained, 12 frozen):
  - token_proj (Linear 16 -> 32):                  ~544
  - ln_attn (LayerNorm 32):                         64
  - q/k/v/o (4 x Linear 32->32, no bias):         4096
  - ln_ffn (LayerNorm 32):                          64
  - unpatch_proj (Linear 32 -> 16):                ~528
  - nca_correction (~6ch in, 32 hidden, 1 out):   ~1900
  - repatch_proj (Linear 16 -> 32):                ~544
  - output_proj (Linear 32 -> 16):                 ~528
  ------------------------------------------------
  approx:                                         ~8268 trained
  frozen CML buffers (K=32 multi-r bank):           ~12

(Actual number will shift depending on whether biases are enabled on the
projections — the 7.4K target in arch_plan.md assumed bias=False everywhere.)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# TODO: enable the real CML bank at integration time.
# from wmca.modules.hybrid import CML2DMultiR  # enable when integrating


# ---------------------------------------------------------------------------
# 2D Rotary Position Embedding (inline, no external dep)
# ---------------------------------------------------------------------------


def _rope_freqs_1d(dim: int, seq_len: int, base: float = 10000.0,
                   device=None, dtype=torch.float32) -> torch.Tensor:
    """Standard 1D RoPE frequency matrix of shape (seq_len, dim/2)."""
    assert dim % 2 == 0, "RoPE dim must be even"
    half = dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=device, dtype=dtype) / half))
    t = torch.arange(seq_len, device=device, dtype=dtype)
    return torch.einsum("i,j->ij", t, inv_freq)          # (seq_len, dim/2)


def _apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Apply rotary to last dim of x. freqs: (seq, dim/2). x: (..., seq, dim)."""
    cos = freqs.cos()[..., None, :, :]                    # broadcastable over heads/batch
    sin = freqs.sin()[..., None, :, :]
    x1, x2 = x.chunk(2, dim=-1)                           # split head dim in half
    # Rotate pairs: (x1, x2) -> (x1*cos - x2*sin, x1*sin + x2*cos)
    return torch.cat([x1 * cos[..., 0, :, :] - x2 * sin[..., 0, :, :],
                      x1 * sin[..., 0, :, :] + x2 * cos[..., 0, :, :]], dim=-1)


def apply_2d_rope(x: torch.Tensor, n_rows: int, n_cols: int) -> torch.Tensor:
    """Apply 2D RoPE to (B, n_heads, seq=n_rows*n_cols, head_dim).

    Splits head_dim in half: first half rotates by row index, second half by
    column index. Zero trainable params. Token order assumed row-major:
    token_idx = r * n_cols + c.
    """
    B, H, S, D = x.shape
    assert S == n_rows * n_cols, "sequence length mismatch with grid shape"
    assert D % 2 == 0, "head dim must be even for 2D RoPE"
    half = D // 2

    row_ids = torch.arange(n_rows, device=x.device).repeat_interleave(n_cols)  # (S,)
    col_ids = torch.arange(n_cols, device=x.device).repeat(n_rows)             # (S,)

    row_freqs = _rope_freqs_1d(half, n_rows, device=x.device, dtype=x.dtype)   # (n_rows, half/2)
    col_freqs = _rope_freqs_1d(half, n_cols, device=x.device, dtype=x.dtype)   # (n_cols, half/2)

    row_f = row_freqs[row_ids]           # (S, half/2)
    col_f = col_freqs[col_ids]           # (S, half/2)

    x_row, x_col = x[..., :half], x[..., half:]

    def _rot(xh: torch.Tensor, freqs_s: torch.Tensor) -> torch.Tensor:
        # xh: (B, H, S, half). freqs_s: (S, half/2).
        cos = freqs_s.cos()[None, None, :, :]                         # (1, 1, S, half/2)
        sin = freqs_s.sin()[None, None, :, :]
        a, b = xh.chunk(2, dim=-1)                                    # each (B, H, S, half/2)
        return torch.cat([a * cos - b * sin, a * sin + b * cos], dim=-1)

    x_row = _rot(x_row, row_f)
    x_col = _rot(x_col, col_f)
    return torch.cat([x_row, x_col], dim=-1)


# ---------------------------------------------------------------------------
# Sliding Window Attention (masked softmax attention with a local window)
# ---------------------------------------------------------------------------


def _swa_mask(n_rows: int, n_cols: int, window: int, device) -> torch.Tensor:
    """Boolean SWA mask of shape (S, S); True = attend allowed.

    Attends to all tokens whose (row, col) L-inf distance <= window/2. For
    window=8 with 16x16 patches this is a ~8-neighborhood in each direction.
    """
    half = window // 2
    rows = torch.arange(n_rows, device=device).repeat_interleave(n_cols)
    cols = torch.arange(n_cols, device=device).repeat(n_rows)
    dr = rows[:, None] - rows[None, :]
    dc = cols[:, None] - cols[None, :]
    return (dr.abs() <= half) & (dc.abs() <= half)


# ---------------------------------------------------------------------------
# Rescorformer block (L=1)
# ---------------------------------------------------------------------------


class ResCorformer(nn.Module):
    """Single rescorformer block (L=1) over 64x64 input with 4x4 patches.

    Shape contract:
        in:  (B, 1, 64, 64) values in [0, 1]
        out: (B, 1, 64, 64) values in [0, 1]

    The CML bank and NCA correction are injected. At integration both become
    real `wmca.modules` instances; at scaffold time they default to
    `nn.Identity`-style stubs so shape tests pass.
    """

    def __init__(
        self,
        img_size: int = 64,
        patch_size: int = 4,
        d_model: int = 32,
        n_heads: int = 4,
        window: int = 8,
        cml: nn.Module | None = None,
        nca: nn.Module | None = None,
    ):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.img_size = img_size
        self.patch_size = patch_size
        self.n_rows = img_size // patch_size
        self.n_cols = img_size // patch_size
        self.n_tokens = self.n_rows * self.n_cols
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.window = window
        self.patch_dim = patch_size * patch_size                    # = 16 for 4x4 patches

        # ---- Patch / unpatch projections ----
        self.token_proj = nn.Linear(self.patch_dim, d_model, bias=False)
        self.output_proj = nn.Linear(d_model, self.patch_dim, bias=False)

        # ---- Attention ----
        self.ln_attn = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        # ---- Rescor FFN replacement ----
        self.ln_ffn = nn.LayerNorm(d_model)
        self.unpatch_proj = nn.Linear(d_model, self.patch_dim, bias=False)
        self.repatch_proj = nn.Linear(self.patch_dim, d_model, bias=False)

        # CML bank on the unpatchified 64x64 grid.
        # TODO: at integration replace stub with
        #   CML2DMultiR(in_channels=1, K=32, steps=10, gate_mode="uniform")
        self.cml = cml if cml is not None else nn.Identity()

        # NCA correction head: in_ch = 1 (sigmoid spatial) + 1 (cml out) = 2
        # for the stub. Real integration will use 6ch (+ stats).
        self.nca = nca if nca is not None else _DefaultNCA(in_ch=2)

    # ---------------- shape helpers ----------------

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 1, H, W) -> (B, n_tokens, patch_dim)."""
        B, C, H, W = x.shape
        assert C == 1, f"expected single channel, got {C}"
        p = self.patch_size
        x = x.unfold(2, p, p).unfold(3, p, p)        # (B, 1, n_r, n_c, p, p)
        x = x.contiguous().view(B, self.n_tokens, p * p)
        return x

    def _unpatchify(self, tokens: torch.Tensor) -> torch.Tensor:
        """(B, n_tokens, patch_dim) -> (B, 1, H, W)."""
        B = tokens.shape[0]
        p = self.patch_size
        x = tokens.view(B, self.n_rows, self.n_cols, p, p)
        x = x.permute(0, 1, 3, 2, 4).contiguous()     # (B, n_r, p, n_c, p)
        return x.view(B, 1, self.img_size, self.img_size)

    # ---------------- forward ----------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]

        # 1. Patchify -> tokens
        patches = self._patchify(x)                                   # (B, S, 16)
        tokens = self.token_proj(patches)                             # (B, S, d_model)

        # 2. Attention sub-block with 2D RoPE + SWA
        residual = tokens
        x_ln = self.ln_attn(tokens)
        q = self.q_proj(x_ln).view(B, self.n_tokens, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_ln).view(B, self.n_tokens, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_ln).view(B, self.n_tokens, self.n_heads, self.head_dim).transpose(1, 2)

        q = apply_2d_rope(q, self.n_rows, self.n_cols)
        k = apply_2d_rope(k, self.n_rows, self.n_cols)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = _swa_mask(self.n_rows, self.n_cols, self.window, x.device)   # (S, S) bool
        attn_scores = attn_scores.masked_fill(~mask[None, None], float("-inf"))
        attn = F.softmax(attn_scores, dim=-1)
        attn_out = torch.matmul(attn, v)                                   # (B, H, S, d_h)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, self.n_tokens, self.d_model)
        tokens = residual + self.o_proj(attn_out)

        # 3. Rescor FFN replacement
        residual = tokens
        x_ln = self.ln_ffn(tokens)
        unpatched = self.unpatch_proj(x_ln)                               # (B, S, 16)
        spatial = self._unpatchify(unpatched)                             # (B, 1, 64, 64)
        spatial = torch.sigmoid(spatial)                                  # into [0, 1] for CML

        cml_out = self.cml(spatial)                                       # (B, 1, 64, 64)
        nca_in = torch.cat([spatial, cml_out], dim=1)                     # (B, 2, 64, 64) stub
        correction = self.nca(nca_in)                                     # (B, 1, 64, 64)
        spatial_out = cml_out + correction                                # "stats.last + correction"

        re_patches = self._patchify(spatial_out)                          # (B, S, 16)
        tokens = residual + self.repatch_proj(re_patches)

        # 4. Output projection back to pixel patches
        out_patches = self.output_proj(tokens)                            # (B, S, 16)
        out = self._unpatchify(out_patches)                               # (B, 1, 64, 64)
        out = torch.clamp(out, 0.0, 1.0)
        return out

    def param_count(self) -> dict[str, int]:
        trained = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(b.numel() for b in self.buffers())
        return {"trained": trained, "frozen": frozen}


class _DefaultNCA(nn.Module):
    """Stub NCA correction head used only for shape-testing. See rescor_ms."""

    def __init__(self, in_ch: int, hidden: int = 32):
        super().__init__()
        self.perceive = nn.Conv2d(in_ch, hidden, kernel_size=3, padding=1)
        self.update = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(hidden, hidden, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )
        nn.init.zeros_(self.update[-1].weight)
        nn.init.zeros_(self.update[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.update(self.perceive(x))
