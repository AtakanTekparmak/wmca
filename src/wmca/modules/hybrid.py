from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CML2D(nn.Module):
    """2D Coupled Map Lattice with frozen logistic map + conv2d coupling."""

    def __init__(self, in_channels: int = 1, steps: int = 15,
                 r: float = 3.90, eps: float = 0.3, beta: float = 0.15,
                 seed: int = 42):
        super().__init__()
        self.in_channels = in_channels
        self.steps = steps

        self.register_buffer("r", torch.tensor(r))
        self.register_buffer("eps", torch.tensor(eps))
        self.register_buffer("beta", torch.tensor(beta))

        rng = torch.Generator().manual_seed(seed)
        K_raw = torch.rand(in_channels, 1, 3, 3, generator=rng).abs()
        K_norm = K_raw / K_raw.sum(dim=(-1, -2), keepdim=True)
        self.register_buffer("K_local", K_norm)

    def forward(self, drive: torch.Tensor) -> torch.Tensor:
        grid = drive
        r, eps, beta = self.r, self.eps, self.beta
        for _ in range(self.steps):
            mapped = r * grid * (1.0 - grid)
            local = F.conv2d(mapped, self.K_local, padding=1,
                             groups=self.in_channels)
            physics = (1 - eps) * mapped + eps * local
            grid = (1 - beta) * physics + beta * drive
        return grid.clamp(1e-4, 1 - 1e-4)

    def param_count(self) -> dict[str, int]:
        return {"trained": 0, "frozen": sum(b.numel() for b in self.buffers())}


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
                 use_sigmoid: bool = True):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_sigmoid = use_sigmoid
        self.nca_steps = nca_steps
        self._recurrent_nca = (out_channels == in_channels)

        # CML operates on the first out_channels channels (the "state")
        self.cml_2d = CML2D(out_channels, cml_steps, r, eps, beta, seed)

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
                 use_sigmoid: bool = True):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_sigmoid = use_sigmoid

        # CML operates on the first out_channels of x (the state)
        self.cml_2d = CML2D(out_channels, cml_steps, r, eps, beta, seed)

        # NCA correction: [x | cml_out] -> out_channels
        self.nca = nn.Sequential(
            nn.Conv2d(in_channels + out_channels, hidden_ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_ch, out_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        state = x[:, : self.out_channels]
        cml_out = self.cml_2d(state)
        correction = self.nca(torch.cat([x, cml_out], dim=1))
        out = cml_out + correction
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
                 seed: int = 42):
        super().__init__()
        self.in_channels = in_channels
        self.steps = steps

        self.register_buffer("r", torch.tensor(r))
        self.register_buffer("eps", torch.tensor(eps))
        self.register_buffer("beta", torch.tensor(beta))

        rng = torch.Generator().manual_seed(seed)
        K_raw = torch.rand(in_channels, 1, 3, 3, generator=rng).abs()
        K_norm = K_raw / K_raw.sum(dim=(-1, -2), keepdim=True)
        self.register_buffer("K_local", K_norm)

    def forward(self, drive: torch.Tensor) -> dict[str, torch.Tensor]:
        grid = drive
        first = drive
        r, eps, beta = self.r, self.eps, self.beta
        states: list[torch.Tensor] = []
        for _ in range(self.steps):
            mapped = r * grid * (1.0 - grid)
            local = F.conv2d(mapped, self.K_local, padding=1,
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
                 seed: int = 42):
        super().__init__()
        self.in_channels = in_channels
        self.steps = steps

        self.register_buffer("r", torch.tensor(r))
        self.register_buffer("eps", torch.tensor(eps))
        self.register_buffer("beta", torch.tensor(beta))

        rng = torch.Generator().manual_seed(seed)
        K_raw = torch.rand(in_channels, 1, 3, 3, generator=rng).abs()
        K_norm = K_raw / K_raw.sum(dim=(-1, -2), keepdim=True)
        self.register_buffer("K_local", K_norm)

    def forward(self, drive: torch.Tensor) -> dict[str, torch.Tensor]:
        grid = drive
        first = drive
        r, eps, beta = self.r, self.eps, self.beta
        states: list[torch.Tensor] = []
        for _ in range(self.steps):
            mapped = r * grid * (1.0 - grid)
            local = F.conv2d(mapped, self.K_local, padding=1,
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
