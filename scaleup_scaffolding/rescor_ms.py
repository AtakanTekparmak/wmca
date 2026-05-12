"""RescorMSPyramid — Multi-Scale U-Net wrapper for rescor.

Scale-up architecture #3 from `arch_plan.md` ("rescor_ms — Multi-Scale Rescor").
Target: 64x64 prediction with a U-Net-like pyramid (64 -> 32 -> 16 -> 32 -> 64)
where each scale runs its own frozen CML reservoir + learned NCA head.

This scaffold is a PURE DRAFT. It never touches `wmca.modules.*`; instead it
takes its CML bank and NCA heads as constructor arguments so the integration
step can drop in the real `CML2DMultiR` bank (K=32, uniform gate) without any
edits to the main project. The integration site is marked with `# TODO` below.

Design choices (matches `arch_plan.md` spec, hero-calibrated):
  - Each level uses a K=32 `CML2DMultiR` bank with `gate_mode="uniform"`
    (1/K averaging, zero gate params). This matches the single-seed hero
    config before its 2026-04-22 multi-seed demotion; even under the demoted
    verdict it's still the simplest scale-parameterized bank to test.
  - Encoder uses sigmoid + stride-2 conv to keep CML inputs in [0, 1].
  - Decoder uses bilinear upsample + additive skip connections (channel count
    stays at 1 throughout, so no concat channel-dim issues).
  - CML step budget per spec: M=5 (64x64) -> M=10 (32x32) -> M=15 (16x16).
    Stronger chaos is used at the bottleneck where the receptive field matters.

Param budget (spec target: ~4K trained, 36 frozen):
  - encoder_down_0 (Conv2d 1->1, 3x3, stride 2):        ~10
  - encoder_down_1 (Conv2d 1->1, 3x3, stride 2):        ~10
  - nca_skip_0 (tiny NCA at 64x64, 6->16->1, 1x1 head): ~400
  - nca_skip_1 (tiny NCA at 32x32, 6->16->1, 1x1 head): ~400
  - nca_bottleneck (rescor_e3c head at 16x16):         ~2600
  - nca_up_1 (post-upsample NCA at 32x32, 2->16->1):   ~300
  - nca_up_0 (post-upsample NCA at 64x64, 2->16->1):   ~300
  - output_proj (Conv2d 1->1, 1x1):                     ~2
  ------------------------------------------------------
  approx:                                              ~4022 trained
  frozen CML buffers (K=32 banks at 3 scales, ~12 each, shared kernel): ~36

The NCA module signature below is intentionally minimal — it is a pluggable
`nn.Module` that consumes `(B, C_in, H, W)` and returns `(B, 1, H, W)`. At
integration time the user can swap in the real rescor_e3c dual-perception
head without changing this file.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# TODO: enable the real CML bank at integration time.
# from wmca.modules.hybrid import CML2DMultiR  # enable when integrating


class _TinyNCA(nn.Module):
    """Placeholder NCA head used by the shape-test. 1x1 convs only, no chaos.

    The REAL integration will replace this with a proper NCA (dual perception
    + update head) from `wmca.modules` — see arch_plan.md section 3 for the
    bottleneck recipe (rescor_e3c at 16x16).
    """

    def __init__(self, in_ch: int, hidden: int = 16, out_ch: int = 1):
        super().__init__()
        self.perceive = nn.Conv2d(in_ch, hidden, kernel_size=3, padding=1)
        self.update = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(hidden, hidden, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden, out_ch, kernel_size=1),
        )
        # Zero-init final layer so the skip-path behaves as identity at init.
        nn.init.zeros_(self.update[-1].weight)
        nn.init.zeros_(self.update[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.update(self.perceive(x))


class RescorMSPyramid(nn.Module):
    """U-Net-style pyramid wrapping rescor at three resolutions.

    Shape contract:
        in:  (B, 1, 64, 64) with values in [0, 1]
        out: (B, 1, 64, 64) with values in [0, 1]

    Constructor takes three CML banks (one per scale) and five NCA heads. At
    scaffold time these can be `None` or placeholder `nn.Identity`-ish modules
    for shape-testing; at integration the CMLs become `CML2DMultiR(K=32,
    gate_mode="uniform", steps=M_scale)` and the NCAs become rescor_e3c-style
    heads.

    Args:
        cml_level0 / cml_level1 / cml_level2: callables (B, 1, H, W) -> (B, 1, H, W)
            (real type at integration: `CML2DMultiR`).
        nca_skip_0 / nca_skip_1 / nca_bottleneck / nca_up_1 / nca_up_0:
            callables (B, C_in, H, W) -> (B, 1, H, W).
    """

    def __init__(
        self,
        cml_level0: nn.Module | None = None,
        cml_level1: nn.Module | None = None,
        cml_level2: nn.Module | None = None,
        nca_skip_0: nn.Module | None = None,
        nca_skip_1: nn.Module | None = None,
        nca_bottleneck: nn.Module | None = None,
        nca_up_1: nn.Module | None = None,
        nca_up_0: nn.Module | None = None,
    ):
        super().__init__()

        # -------- CML banks (one per scale) --------
        # TODO: at integration replace the Identity stubs with
        #   CML2DMultiR(in_channels=1, K=32, steps=5,  gate_mode="uniform")  # 64x64
        #   CML2DMultiR(in_channels=1, K=32, steps=10, gate_mode="uniform")  # 32x32
        #   CML2DMultiR(in_channels=1, K=32, steps=15, gate_mode="uniform")  # 16x16
        self.cml_level0 = cml_level0 if cml_level0 is not None else nn.Identity()
        self.cml_level1 = cml_level1 if cml_level1 is not None else nn.Identity()
        self.cml_level2 = cml_level2 if cml_level2 is not None else nn.Identity()

        # -------- Encoder downsamples (stride-2 convs) --------
        self.down_0 = nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1)
        self.down_1 = nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1)

        # -------- NCA heads --------
        # Encoder skip NCAs take [input, cml_out] -> 2 channels (in the stub).
        # At integration, full stats (last, mean, var, delta, last_drive) make
        # it 6 channels; that is the real in_ch for the integrated NCA.
        self.nca_skip_0 = nca_skip_0 if nca_skip_0 is not None else _TinyNCA(in_ch=2)
        self.nca_skip_1 = nca_skip_1 if nca_skip_1 is not None else _TinyNCA(in_ch=2)

        # Bottleneck uses the full rescor_e3c head at 16x16.
        # TODO: integration target is `ResCorE3C` (or equivalent) at 16x16.
        self.nca_bottleneck = (
            nca_bottleneck if nca_bottleneck is not None else _TinyNCA(in_ch=2, hidden=32)
        )

        # Decoder NCAs process the summed upsample+skip (1 channel in stub).
        self.nca_up_1 = nca_up_1 if nca_up_1 is not None else _TinyNCA(in_ch=1)
        self.nca_up_0 = nca_up_0 if nca_up_0 is not None else _TinyNCA(in_ch=1)

        # Output projection keeps things in [0, 1] after a final 1x1 mix.
        self.output_proj = nn.Conv2d(1, 1, kernel_size=1)

    # ---------- shape helpers ----------

    @staticmethod
    def _concat_stats(x: torch.Tensor, cml_out: torch.Tensor) -> torch.Tensor:
        """Stub stat concat: just [x, cml_out] for shape-testing.

        At integration, this becomes cat([x, stats.last, stats.mean, stats.var,
        stats.delta, stats.last_drive]) -> 6 channels. We keep it at 2 channels
        here so the placeholder NCAs match.
        """
        return torch.cat([x, cml_out], dim=1)

    # ---------- forward ----------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 1, 64, 64) -> (B, 1, 64, 64), values in [0, 1]."""
        assert x.dim() == 4 and x.shape[1] == 1, f"expected (B,1,H,W), got {x.shape}"

        # ----- Encoder -----
        # Level 0 (64x64)
        z0 = torch.sigmoid(x)                       # ensure [0, 1] for CML drive
        cml0 = self.cml_level0(z0)                  # (B, 1, 64, 64)
        skip0 = self.nca_skip_0(self._concat_stats(z0, cml0))  # (B, 1, 64, 64)

        # Level 1 (32x32)
        z1 = torch.sigmoid(self.down_0(z0))         # (B, 1, 32, 32)
        cml1 = self.cml_level1(z1)
        skip1 = self.nca_skip_1(self._concat_stats(z1, cml1))  # (B, 1, 32, 32)

        # Level 2 — bottleneck (16x16)
        z2 = torch.sigmoid(self.down_1(z1))         # (B, 1, 16, 16)
        cml2 = self.cml_level2(z2)
        h2 = self.nca_bottleneck(self._concat_stats(z2, cml2))  # (B, 1, 16, 16)

        # ----- Decoder (bilinear upsample + additive skip) -----
        u1 = F.interpolate(h2, size=(32, 32), mode="bilinear", align_corners=False) + skip1
        h1 = u1 + self.nca_up_1(u1)                 # residual refinement

        u0 = F.interpolate(h1, size=(64, 64), mode="bilinear", align_corners=False) + skip0
        h0 = u0 + self.nca_up_0(u0)

        out = torch.clamp(self.output_proj(h0), 0.0, 1.0)
        return out

    def param_count(self) -> dict[str, int]:
        trained = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(b.numel() for b in self.buffers())
        return {"trained": trained, "frozen": frozen}
