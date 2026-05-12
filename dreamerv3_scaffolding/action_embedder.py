"""Action embedder for the rescor RSSM.

Maps a discrete action id (e.g. one of 17 Crafter actions) to a
(B, C_mr, 16, 16) grid that is added to the rescor input BEFORE the
frozen CML bank runs. This is Option (B) "drive modulation" from the
DreamerV3 fork plan: the chaotic reservoir amplifies small input
perturbations, so injecting the action here lets the frozen dynamics
genuinely be steered by action selection.

Param budget
------------
With action_dim=17, hidden=512, K_mr_channels=2, grid=16 the module is::

    Linear(17, 512)    => 17*512 + 512 =  9_216
    Linear(512, 512)   => 512*2*16*16 + 512 = 262_656  (too large)

To keep the total sequence-block parameter count near the plan's
~13K target, the default embedder uses a small bottleneck::

    Linear(17, 24)     => 17*24 + 24     =    432
    Linear(24, 2*16*16)=> 24*512 + 512   = 12_800

Total: ~13_232 params. Adjust `hidden` if a different budget is desired.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ActionEmbedder(nn.Module):
    """discrete action -> (B, C, 16, 16) drive perturbation grid.

    Parameters
    ----------
    action_dim : int
        Number of discrete actions. Crafter uses 17.
    hidden : int
        Bottleneck hidden dimension. Default 24 to hit the ~13K budget
        documented in `dreamerv3_fork_plan.md`.
    n_channels : int
        Channels of the rescor grid (C in the (B, C, H, W) reshape of
        Dreamer's deter state). With deter=512, n_channels=2.
    grid_h, grid_w : int
        Spatial dimensions of the rescor grid. 16x16 to match the
        existing CML2DMultiR hero configuration.
    """

    def __init__(
        self,
        action_dim: int = 17,
        hidden: int = 24,
        n_channels: int = 2,
        grid_h: int = 16,
        grid_w: int = 16,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.n_channels = n_channels
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.out_numel = n_channels * grid_h * grid_w

        self.fc1 = nn.Linear(action_dim, hidden)
        self.fc2 = nn.Linear(hidden, self.out_numel)

        # Learnable gain on the output — if it collapses to zero the
        # action is not steering the reservoir, which is the falsification
        # criterion for Option B (see risk #1 in the fork plan).
        self.gain = nn.Parameter(torch.tensor(1.0))

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        """a : (B,) long tensor of discrete action ids -> (B, C, H, W)."""
        if a.dim() == 2 and a.shape[-1] == self.action_dim:
            # Already one-hot / continuous action distribution — accept it.
            onehot = a.float()
        else:
            onehot = F.one_hot(a.long(), num_classes=self.action_dim).float()
        h = F.relu(self.fc1(onehot))
        g = self.fc2(h)
        g = g.view(-1, self.n_channels, self.grid_h, self.grid_w)
        return self.gain * g

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
