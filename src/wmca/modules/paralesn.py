from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


# ---------------------------------------------------------------------------
# Parallel scan (FFT / sequential / Triton)
# ---------------------------------------------------------------------------

def fft_causal_conv(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    B, T, N_h = b.shape
    powers = torch.arange(T, device=b.device, dtype=b.real.dtype).unsqueeze(1)
    kernel = a.unsqueeze(0).pow(powers)
    kernel_padded = F.pad(kernel, (0, 0, 0, T))
    b_padded = F.pad(b, (0, 0, 0, T))
    K_f = torch.fft.fft(kernel_padded, dim=0)
    B_f = torch.fft.fft(b_padded, dim=1)
    H_f = K_f.unsqueeze(0) * B_f
    h = torch.fft.ifft(H_f, dim=1)[:, :T]
    return h


def sequential_scan(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    B, T, N_h = b.shape
    h = torch.zeros(B, T, N_h, dtype=b.dtype, device=b.device)
    h_prev = torch.zeros(B, N_h, dtype=b.dtype, device=b.device)
    for t in range(T):
        h_prev = a.unsqueeze(0) * h_prev + b[:, t]
        h[:, t] = h_prev
    return h


# ---------------------------------------------------------------------------
# Triton fused scan kernels
# ---------------------------------------------------------------------------

if HAS_TRITON:

    @triton.jit
    def _scan_fwd_kernel(
        a_re_ptr, a_im_ptr,
        b_re_ptr, b_im_ptr,
        h_re_ptr, h_im_ptr,
        B, T, N_h,
        stride_b_batch, stride_b_time,
        BLOCK_N: tl.constexpr,
    ):
        pid_batch = tl.program_id(0)
        pid_chunk = tl.program_id(1)
        offs_n = pid_chunk * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N_h
        ar = tl.load(a_re_ptr + offs_n, mask=mask_n, other=0.0)
        ai = tl.load(a_im_ptr + offs_n, mask=mask_n, other=0.0)
        hr = tl.zeros([BLOCK_N], dtype=tl.float32)
        hi = tl.zeros([BLOCK_N], dtype=tl.float32)
        base = pid_batch * stride_b_batch
        for t in range(T):
            offset = base + t * stride_b_time + offs_n
            br = tl.load(b_re_ptr + offset, mask=mask_n, other=0.0)
            bi = tl.load(b_im_ptr + offset, mask=mask_n, other=0.0)
            new_hr = ar * hr - ai * hi + br
            new_hi = ar * hi + ai * hr + bi
            hr = new_hr
            hi = new_hi
            tl.store(h_re_ptr + offset, hr, mask=mask_n)
            tl.store(h_im_ptr + offset, hi, mask=mask_n)

    @triton.jit
    def _scan_bwd_kernel(
        a_re_ptr, a_im_ptr,
        grad_h_re_ptr, grad_h_im_ptr,
        grad_b_re_ptr, grad_b_im_ptr,
        B, T, N_h,
        stride_b_batch, stride_b_time,
        BLOCK_N: tl.constexpr,
    ):
        pid_batch = tl.program_id(0)
        pid_chunk = tl.program_id(1)
        offs_n = pid_chunk * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N_h
        ar = tl.load(a_re_ptr + offs_n, mask=mask_n, other=0.0)
        ai_neg = -tl.load(a_im_ptr + offs_n, mask=mask_n, other=0.0)
        gr = tl.zeros([BLOCK_N], dtype=tl.float32)
        gi = tl.zeros([BLOCK_N], dtype=tl.float32)
        base = pid_batch * stride_b_batch
        for t_rev in range(T):
            t = T - 1 - t_rev
            offset = base + t * stride_b_time + offs_n
            dhr = tl.load(grad_h_re_ptr + offset, mask=mask_n, other=0.0)
            dhi = tl.load(grad_h_im_ptr + offset, mask=mask_n, other=0.0)
            new_gr = dhr + ar * gr - ai_neg * gi
            new_gi = dhi + ar * gi + ai_neg * gr
            gr = new_gr
            gi = new_gi
            tl.store(grad_b_re_ptr + offset, gr, mask=mask_n)
            tl.store(grad_b_im_ptr + offset, gi, mask=mask_n)


class TritonScanFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        B, T, N_h = b.shape
        a_re = a.real.contiguous()
        a_im = a.imag.contiguous()
        b_re = b.real.contiguous()
        b_im = b.imag.contiguous()
        h_re = torch.empty_like(b_re)
        h_im = torch.empty_like(b_im)
        BLOCK_N = triton.next_power_of_2(min(N_h, 256))
        grid = (B, triton.cdiv(N_h, BLOCK_N))
        _scan_fwd_kernel[grid](
            a_re, a_im, b_re, b_im, h_re, h_im,
            B, T, N_h, T * N_h, N_h, BLOCK_N=BLOCK_N,
        )
        h = torch.complex(h_re, h_im)
        ctx.save_for_backward(a_re, a_im)
        ctx.B = B
        ctx.T = T
        ctx.N_h = N_h
        return h

    @staticmethod
    def backward(ctx, grad_h):
        a_re, a_im = ctx.saved_tensors
        B, T, N_h = ctx.B, ctx.T, ctx.N_h
        grad_h_re = grad_h.real.contiguous()
        grad_h_im = grad_h.imag.contiguous()
        grad_b_re = torch.empty_like(grad_h_re)
        grad_b_im = torch.empty_like(grad_h_im)
        BLOCK_N = triton.next_power_of_2(min(N_h, 256))
        grid = (B, triton.cdiv(N_h, BLOCK_N))
        _scan_bwd_kernel[grid](
            a_re, a_im, grad_h_re, grad_h_im, grad_b_re, grad_b_im,
            B, T, N_h, T * N_h, N_h, BLOCK_N=BLOCK_N,
        )
        return None, torch.complex(grad_b_re, grad_b_im)


def triton_scan(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return TritonScanFn.apply(a, b)


# ---------------------------------------------------------------------------
# ParalESN Layer (linear time-mixing)
# ---------------------------------------------------------------------------

class ParalESNLayer(nn.Module):
    def __init__(self, cfg, layer_idx: int, input_size: int, rng: torch.Generator):
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        N_h = cfg.hidden_size

        rho = torch.empty(N_h).uniform_(cfg.rho_min, cfg.rho_max, generator=rng)
        theta = torch.empty(N_h).uniform_(cfg.theta_min, cfg.theta_max, generator=rng)
        Lambda_h = rho * torch.exp(1j * theta.to(torch.cfloat))
        Lambda_bar_h = (1 - cfg.tau) + cfg.tau * Lambda_h
        self.register_buffer("Lambda_bar_h", Lambda_bar_h)

        abs_lambda = Lambda_bar_h.abs()
        assert (abs_lambda < 1.0).all(), (
            f"Layer {layer_idx}: ESP violated! max|Λ̄_h|={abs_lambda.max().item():.6f}"
        )

        scale = torch.sqrt(1.0 - abs_lambda ** 2) * cfg.omega_in

        if layer_idx == 0:
            W_in_complex = torch.complex(
                torch.randn(N_h, input_size, generator=rng),
                torch.randn(N_h, input_size, generator=rng),
            )
            row_norms = W_in_complex.abs().pow(2).sum(dim=1).sqrt().clamp(min=1e-8)
            W_in_complex = W_in_complex / row_norms.unsqueeze(1) * scale.unsqueeze(1).to(torch.cfloat)
            self.register_buffer("W_in", W_in_complex)
            self.register_buffer("ring_scale", torch.empty(0))
            self.register_buffer("ring_shift", torch.empty(0))
        else:
            ring_scale = torch.complex(
                torch.randn(N_h, generator=rng),
                torch.randn(N_h, generator=rng),
            )
            ring_scale = ring_scale / ring_scale.abs().clamp(min=1e-8) * scale.to(torch.cfloat)
            self.register_buffer("ring_scale", ring_scale)
            ring_shift = torch.complex(
                torch.randn(N_h, generator=rng) * cfg.omega_b,
                torch.randn(N_h, generator=rng) * cfg.omega_b,
            )
            self.register_buffer("ring_shift", ring_shift)
            self.register_buffer("W_in", torch.empty(0))

        self.register_buffer("b", torch.complex(
            torch.randn(N_h, generator=rng) * cfg.omega_b,
            torch.randn(N_h, generator=rng) * cfg.omega_b,
        ))

        k = cfg.mix_kernel_size
        self.register_buffer("W_mix", torch.randn(1, 1, k, generator=rng) / math.sqrt(k))
        self.register_buffer("b_mix", torch.randn(N_h, generator=rng) * 0.01)

        # Learned output projection: aligns random reservoir basis → semantic residual stream
        # Zero-init so ParalESN contributes nothing initially (standard residual trick)
        self.out_proj = nn.Linear(N_h, N_h, bias=False)
        nn.init.zeros_(self.out_proj.weight)

    def _input_projection(self, x: torch.Tensor) -> torch.Tensor:
        if self.layer_idx == 0:
            return torch.matmul(x.to(self.W_in.dtype), self.W_in.T)
        else:
            x_complex = x.to(self.ring_scale.dtype)
            x_shifted = torch.roll(x_complex, shifts=1, dims=-1)
            return self.ring_scale * x_shifted + self.ring_shift

    def _mix(self, h: torch.Tensor) -> torch.Tensor:
        B, T, N_h = h.shape
        h_real = h.real
        h_flat = h_real.reshape(B * T, 1, N_h)
        pad = self.cfg.mix_kernel_size // 2
        h_conv = F.conv1d(h_flat, self.W_mix, padding=pad)[:, 0, :N_h]
        h_conv = h_conv.reshape(B, T, N_h)
        return torch.tanh(h_conv + self.b_mix)

    def _mix_single(self, h: torch.Tensor) -> torch.Tensor:
        h_real = h.real
        h_flat = h_real.unsqueeze(1)
        pad = self.cfg.mix_kernel_size // 2
        h_conv = F.conv1d(h_flat, self.W_mix, padding=pad)[:, 0, :h_real.shape[1]]
        return torch.tanh(h_conv + self.b_mix)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        projected = self._input_projection(x)
        drive = self.cfg.tau * (projected + self.b)

        if HAS_TRITON and x.device.type == "cuda":
            h = triton_scan(self.Lambda_bar_h, drive)
        elif self.cfg.use_fft:
            h = fft_causal_conv(self.Lambda_bar_h, drive)
        else:
            h = sequential_scan(self.Lambda_bar_h, drive)

        z = self._mix(h)
        z = self.out_proj(z)
        return h, z

    def forward_token(self, x: torch.Tensor, h_prev: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.layer_idx == 0:
            projected = torch.matmul(x.to(self.W_in.dtype), self.W_in.T)
        else:
            x_complex = x.to(self.ring_scale.dtype)
            projected = self.ring_scale * torch.roll(x_complex, shifts=1, dims=-1) + self.ring_shift
        drive = self.cfg.tau * (projected + self.b)
        h = self.Lambda_bar_h * h_prev + drive
        z = self._mix_single(h)
        z = self.out_proj(z)
        return h, z
