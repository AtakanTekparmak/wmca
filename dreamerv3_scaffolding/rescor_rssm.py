"""RescorRSSM — drop-in replacement for DreamerV3's GRU-based RSSM.

Interfaces mirror NM512/dreamerv3-torch's `networks.RSSM`:

    .initial(batch_size)       -> dict{"deter", "stoch", "logit", ...}
    .obs_step(prev_state, prev_action, embed, is_first) -> (post, prior)
    .img_step(prev_state, prev_action) -> prior

Core idea
---------
Dreamer keeps `h_t in R^deter` (deter=512). We reshape that to a
(B, 2, 16, 16) grid, run one CML2DMultiR K=32 step + NCA correction
(i.e. the existing `rescor_rens` hero), then reshape back.

Action conditioning is Option B (drive modulation): an
`ActionEmbedder` produces a (B, 2, 16, 16) grid that is ADDED to the
reshaped deter grid BEFORE the frozen CML bank runs. The frozen chaotic
dynamics amplify this perturbation, so the action genuinely steers the
trajectory without touching the frozen CML weights.

Parameter budget (sequence block only)
--------------------------------------
    CML2DMultiR K=32, gate_mode="uniform"    : 0 trainable (32 buffers)
    NCA correction (inside ResidualCorrectionWM) : 321
    ActionEmbedder (hidden=24)                : ~13_232
    -----------------------------------------------------
    Total trainable in sequence block         : ~13_553  (~14K)

Compare with Dreamer's GRU at deter=512: ~1.5M params.

Status
------
This file is a STUB. Actual Dreamer integration happens after:
  1. Forking NM512/dreamerv3-torch.
  2. `pip install -e ../wmca` so `wmca.modules.hybrid.CML2DMultiR`
     and `wmca.modules.hybrid.ResidualCorrectionWM` are importable.
  3. Copying this file into `dreamerv3/networks_rescor.py` in the fork.
  4. Wiring it into the RSSM factory in `dreamerv3/models.py`.

Every `# TODO` marks a place where the original Dreamer code path
must be matched exactly (shape/naming/dtype) — do NOT guess, read
NM512's `networks.py` and mirror the contract.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

# NOTE: Import paths below resolve only AFTER `pip install -e ../wmca`
# inside the Dreamer fork. Guarded so this scaffold file itself can be
# loaded for inspection without the wmca package available.
try:
    from wmca.modules.hybrid import CML2DMultiR  # type: ignore
    from wmca.model_registry import create_model  # type: ignore
except ImportError:  # scaffolding-time fallback
    CML2DMultiR = None  # type: ignore
    create_model = None  # type: ignore

from action_embedder import ActionEmbedder


class RescorRSSM(nn.Module):
    """Rescor-based recurrent state space model for DreamerV3.

    Parameters
    ----------
    deter : int
        Dreamer's deterministic hidden size. Must equal C*H*W. With
        deter=512 and H=W=16, C=2.
    stoch : int
        Dreamer's stochastic latent width. Unchanged from default
        (keeps the categorical posterior if the M2 rollout probe says
        we need it — see fork plan section 4).
    discrete : int
        Categorical classes per latent dim (default 32 in Dreamer).
    action_dim : int
        Number of discrete actions (17 for Crafter).
    K : int
        Number of CMLs in the frozen r-ensemble. Hero is K=32.
    grid_h, grid_w : int
        Spatial reshape. 16x16 matches the rescor_rens hero config.
    n_channels : int
        Channels after reshape. Set so that `n_channels*grid_h*grid_w == deter`.
    seed : int
        Propagated to CML2DMultiR so the frozen coupling is reproducible.
    """

    def __init__(
        self,
        deter: int = 512,
        stoch: int = 32,
        discrete: int = 32,
        action_dim: int = 17,
        K: int = 32,
        grid_h: int = 16,
        grid_w: int = 16,
        n_channels: int = 2,
        seed: int = 42,
    ):
        super().__init__()
        assert n_channels * grid_h * grid_w == deter, (
            f"reshape mismatch: {n_channels}*{grid_h}*{grid_w} != deter={deter}"
        )
        self.deter = deter
        self.stoch = stoch
        self.discrete = discrete
        self.action_dim = action_dim
        self.K = K
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.n_channels = n_channels

        # --- Frozen chaotic reservoir bank (K=32 CMLs, uniform averaging).
        # gate_mode="uniform" -> zero trainable gate params. All K coupling
        # kernels and r values are non-trainable buffers.
        # TODO: when wiring into Dreamer, prefer
        #     rescor = create_model("rescor_rens", in_channels=n_channels,
        #                            grid_h=grid_h, grid_w=grid_w, seed=seed)
        # so the full ResidualCorrectionWM (CML + NCA correction) is used.
        # For the stub we build the bare CML for clarity.
        if CML2DMultiR is not None:
            self.cml_bank = CML2DMultiR(
                in_channels=n_channels,
                K=K,
                steps=15,
                seed=seed,
                gate_mode="uniform",
            )
        else:
            self.cml_bank = None  # set at install-time

        # --- 321-param NCA correction head.
        # In the real integration we use `ResidualCorrectionWM` from
        # model_registry (it bundles cml + nca with the canonical
        # residual add). Kept separate here so each piece is visible.
        # TODO: replace this placeholder with
        #     self.rescor = create_model("rescor_rens", ...)
        # and route grids through `self.rescor(x)` in _rescor_step.
        self.nca_correction = nn.Conv2d(
            n_channels, n_channels, kernel_size=3, padding=1, bias=True
        )  # placeholder — real NCA is the 321-param module inside ResCor

        # --- Action drive (Option B).
        self.action_embedder = ActionEmbedder(
            action_dim=action_dim,
            hidden=24,
            n_channels=n_channels,
            grid_h=grid_h,
            grid_w=grid_w,
        )

        # --- Prior / posterior heads over the stochastic latent.
        # These are the EXACT same MLPs Dreamer uses — we must mirror the
        # shapes from NM512's `networks.RSSM` so downstream KL balancing
        # and symlog transforms remain untouched.
        # TODO: copy hidden sizes and activation from NM512 config.
        stoch_out = stoch * discrete
        self.prior_mlp = nn.Sequential(
            nn.Linear(deter, 512), nn.SiLU(), nn.Linear(512, stoch_out)
        )
        self.post_mlp = nn.Sequential(
            # posterior sees deter + embed; embed dim is set by Dreamer
            # encoder (typically 1024 on Crafter).
            # TODO: parameterize embed_dim from config.
            nn.Linear(deter + 1024, 512), nn.SiLU(), nn.Linear(512, stoch_out)
        )

    # ------------------------------------------------------------------
    # Dreamer RSSM interface
    # ------------------------------------------------------------------
    def initial(self, batch_size: int) -> dict[str, torch.Tensor]:
        """Return a zeroed initial state dict.

        Dreamer expects keys: `deter`, `stoch`, `logit` (and optionally
        `mean`/`std` for continuous variants — NM512 uses discrete).
        """
        device = next(self.parameters()).device
        return {
            "deter": torch.zeros(batch_size, self.deter, device=device),
            "logit": torch.zeros(
                batch_size, self.stoch, self.discrete, device=device
            ),
            "stoch": torch.zeros(
                batch_size, self.stoch, self.discrete, device=device
            ),
        }

    def _rescor_step(
        self, h_prev: torch.Tensor, action_emb: torch.Tensor
    ) -> torch.Tensor:
        """Flat deter vector -> one rescor K=32 step -> flat deter vector."""
        B = h_prev.shape[0]
        grid = h_prev.view(B, self.n_channels, self.grid_h, self.grid_w)
        # Option B: add action drive BEFORE the frozen CML bank.
        grid = grid + action_emb
        # TODO: once `self.rescor` is the real ResidualCorrectionWM,
        # replace the next two lines with `grid = self.rescor(grid)`.
        if self.cml_bank is not None:
            cml_out = self.cml_bank(grid)
        else:
            cml_out = grid  # scaffolding-time no-op
        delta = self.nca_correction(cml_out)  # placeholder NCA
        h_next_grid = cml_out + delta
        return h_next_grid.reshape(B, self.deter)

    def obs_step(
        self,
        prev_state: dict[str, torch.Tensor],
        prev_action: torch.Tensor,
        embed: torch.Tensor,
        is_first: torch.Tensor | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """One step conditioned on the observation (training/inference).

        Returns (posterior_state, prior_state). See NM512 `networks.py`.

        TODO: honor `is_first` by resetting the affected rows of
        `prev_state` to `self.initial(...)` — Dreamer does this to
        zero-out state at episode boundaries in batched rollouts.
        """
        # 1. Advance deter via rescor.
        a_emb = self.action_embedder(prev_action)
        # Concat previous stoch into grid? Fork plan section 2 calls for
        # `[h_grid, z_grid, a_emb_grid]` concat — done here by ADDITION
        # (Option B); adding z_prev is optional. For now we fold z into
        # the MLP branches only, matching Dreamer's default GRU closely.
        # TODO: decide whether prev_stoch should also be added to grid.
        deter = self._rescor_step(prev_state["deter"], a_emb)

        # 2. Prior from deter.
        prior_logit = self.prior_mlp(deter).view(-1, self.stoch, self.discrete)
        # 3. Posterior from [deter, embed].
        post_logit = self.post_mlp(torch.cat([deter, embed], dim=-1))
        post_logit = post_logit.view(-1, self.stoch, self.discrete)

        # TODO: sample categoricals with straight-through estimator —
        # copy `_sample_stoch` helper from NM512 `networks.py`.
        post_stoch = F.one_hot(post_logit.argmax(-1), self.discrete).float()
        prior_stoch = F.one_hot(prior_logit.argmax(-1), self.discrete).float()

        post = {"deter": deter, "logit": post_logit, "stoch": post_stoch}
        prior = {"deter": deter, "logit": prior_logit, "stoch": prior_stoch}
        return post, prior

    def img_step(
        self,
        prev_state: dict[str, torch.Tensor],
        prev_action: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """One imagination step (no observation). Returns prior only."""
        a_emb = self.action_embedder(prev_action)
        deter = self._rescor_step(prev_state["deter"], a_emb)
        prior_logit = self.prior_mlp(deter).view(-1, self.stoch, self.discrete)
        # TODO: sample with straight-through from logits.
        prior_stoch = F.one_hot(prior_logit.argmax(-1), self.discrete).float()
        return {"deter": deter, "logit": prior_logit, "stoch": prior_stoch}

    # ------------------------------------------------------------------
    # Bookkeeping
    # ------------------------------------------------------------------
    def param_count(self) -> dict[str, int]:
        trained = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        frozen += sum(b.numel() for b in self.buffers())
        return {"trained": trained, "frozen": frozen}
