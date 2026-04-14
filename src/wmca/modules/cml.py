from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CML(nn.Module):
    """Driven Coupled Map Lattice reservoir.

    Column-stochastic W_cc + positive K_local + convex injection
    guarantees grid stays in [0, 1]. All parameters are fixed buffers.
    """

    def __init__(self, C: int, steps: int, kernel_size: int,
                 r: float, eps: float, beta: float, rng: torch.Generator):
        super().__init__()
        self.steps = steps
        self.kernel_size = kernel_size

        self.register_buffer("r", torch.full((C,), r))
        self.register_buffer("eps", torch.full((C,), eps))
        self.register_buffer("beta", torch.full((C,), beta))

        # Positive stochastic kernel
        K_raw = torch.rand(1, 1, kernel_size, generator=rng)
        self.register_buffer("K_local", K_raw / K_raw.sum())

        # Column-stochastic sparse coupling
        W_cc = torch.rand(C, C, generator=rng)
        mask = (torch.rand(C, C, generator=rng) < 0.2).float()
        W_cc = W_cc * mask
        W_cc = W_cc / W_cc.sum(dim=0, keepdim=True).clamp(min=1e-8)
        self.register_buffer("W_cc", W_cc)

    def forward(self, drive: torch.Tensor, readout: str = "final") -> torch.Tensor:
        """Run M steps of driven CML.

        Args:
            drive: [N, C] in [0, 1] (sigmoid of input features)
            readout: "final" (last state), "traj_mean" (average all steps),
                     "even_mean" (average even-indexed steps only)
        Returns: grid [N, C] in [0, 1]
        """
        r = self.r.unsqueeze(0)
        eps = self.eps.unsqueeze(0)
        beta = self.beta.unsqueeze(0)
        one_minus_eps = 1.0 - eps
        one_minus_beta = 1.0 - beta

        k = self.kernel_size
        pad = k // 2

        accumulate = readout != "final"
        grid = drive
        if accumulate:
            traj_sum = torch.zeros_like(drive)
            even_sum = torch.zeros_like(drive)
            n_even = 0

        for step_i in range(self.steps):
            mapped = r * grid * (1.0 - grid)

            m3 = mapped.unsqueeze(1)
            m_pad = torch.cat([m3[:, :, -pad:], m3, m3[:, :, :pad]], dim=2)
            local = F.conv1d(m_pad, self.K_local).squeeze(1)

            global_cc = mapped @ self.W_cc
            coupled = 0.5 * (local + global_cc)
            physics = one_minus_eps * mapped + eps * coupled
            grid = one_minus_beta * physics + beta * drive

            if accumulate:
                traj_sum = traj_sum + grid
                if step_i % 2 == 0:
                    even_sum = even_sum + grid
                    n_even += 1

        if readout == "traj_mean":
            out = traj_sum / self.steps
        elif readout == "even_mean":
            out = even_sum / max(n_even, 1)
        else:
            out = grid
        return out.clamp(1e-4, 1.0 - 1e-4)


class CausalSequenceCML(nn.Module):
    """
    1D Causal Sequence Lattice.
    Tokens are spatially coupled via causal diffusion across time.
    Channels are independent universes with distinct physics parameters.
    """
    def __init__(self, C: int, steps: int, kernel_size: int, rng: torch.Generator):
        super().__init__()
        self.steps = steps
        self.k = kernel_size
        self.C = C

        # 1(c) "Physics Personalities": Every channel gets its own r, eps, beta
        self.register_buffer("r", torch.empty(C).uniform_(3.6, 3.99, generator=rng))
        self.register_buffer("eps", torch.empty(C).uniform_(0.1, 0.4, generator=rng))
        self.register_buffer("beta", torch.empty(C).uniform_(0.05, 0.2, generator=rng))
        # Causal temporal coupling kernel (e.g., k=3 -> looks at t-2, t-1, t)
        # Shape: [C, 1, k] for depthwise conv1d
        K_raw = torch.rand(C, 1, kernel_size, generator=rng)
        self.register_buffer("K_causal", K_raw / K_raw.sum(dim=-1, keepdim=True))

    def forward(self, drive: torch.Tensor) -> torch.Tensor:
        # drive: [B, T, C] -> reshape to [B, C, T] for temporal sequence mixing
        B, T, C = drive.shape
        grid = drive.transpose(1, 2)
        drive_time = grid.clone()
        # Expand for broadcast
        r = self.r.view(1, C, 1)
        eps = self.eps.view(1, C, 1)
        beta = self.beta.view(1, C, 1)
        # Strict causal padding (pad past, do not pad future)
        pad_left = self.k - 1
        for _ in range(self.steps):
            # 1. Non-linear Chaos
            mapped = r * grid * (1.0 - grid)

            # 2. Causal Sequence Diffusion (Time mixes forward!)
            mapped_padded = F.pad(mapped, (pad_left, 0))

            # Depthwise conv1d: each channel diffuses using its own K_causal
            local = F.conv1d(mapped_padded, self.K_causal, groups=C)

            # 3. Thermodynamic Update
            physics = (1.0 - eps) * mapped + eps * local
            grid = (1.0 - beta) * physics + beta * drive_time
        # Back to [B, T, C]
        return grid.clamp(1e-4, 1.0 - 1e-4).transpose(1, 2)
