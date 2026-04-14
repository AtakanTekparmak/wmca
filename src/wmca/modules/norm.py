from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Enforces S^{d-1} geometry and expands variance for the sigmoid drive."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight
