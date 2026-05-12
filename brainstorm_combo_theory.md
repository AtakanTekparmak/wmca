# Combo theory + risk: drift-gated hybrid + multistep penalty

Status: theory + risk analysis, 2026-04-29 (Sprint Day 4 prep, pre-experiment).
Goal: predict mechanistically whether `ResCorMambaGated` trained with
`multistep_horizon=8` is worth GPU before we burn a run on it. Companion
to §brainstorm_combo_synergy (which proposes interaction designs); this
doc is the prior — does the *vanilla* combo even work?

Self-contained inputs:

| Variant | gs step1 MSE | gs H=15 abs (ratio) | gs H=100 abs (cos_div) |
|---|---|---|---|
| rens K=32 | 2.05e-5 | 9.66e-4 (47×) | 3.50e-2 (0.014) |
| mamba_rand H=1 | ~1.4e-6 | 6.47e-5 (36×) | 2.63e-1 (0.30 CATASTROPHE) |
| pushforward_mamba | — | 3.24e-3 (107×) | 3.27e-1 (0.47 CATASTROPHE) |
| **multistep_mamba H=8** | **~1.4e-4** (100× regression) | **1.84e-2 (1.66× PASS)** | **3.51e-2 (0.04 PASS)** |
| Drift-gated `ResCorMambaGated` H=1 | not yet trained at full scale | hypothesis: ≈ mamba_rand | hypothesis: < mamba_rand catastrophe |
| **Combo (this doc)** | ? | ? | ? |

Mikhaeil 2022 ceiling: stable rollout horizon ≈ log(1/ε)/λ̂. For r=3.99
logistic λ̂≈0.69, ε=1e-6 → ~20 steps. Empirically we see 15–20 step
stable horizons. Multistep H=8 trains right under this ceiling — that's
why it works.

---

## 1. Failure-mode taxonomy — orthogonal or redundant?

The two mechanisms operate at *different levels of the rollout
pathology*. Let me factor the failure into three mechanisms:

| Mechanism | What goes wrong | Multistep H=8 fix? | Drift gate fix? |
|---|---|---|---|
| **(A) Exposure bias** | Train on clean inputs → at test, inputs are own predictions, distribution shifted | YES — train with own predictions in the loop (Bengio 2015, Brandstetter 2022, Chakraborty 2024) | NO — gate doesn't change train distribution |
| **(B) K-frame buffer pollution** | One bad prediction pollutes the next K=4 inputs to mamba; mamba over-trusts pollution | partial (model learns to be robust to it) | YES — gate attenuates mamba contribution when drift signal is high, falling back to rens-like averaging |
| **(C) Chaos floor** (Mikhaeil) | Information loss is fundamental: ~log(1/ε)/λ steps and you have 0 mutual information w/ truth | trains *up to* the ceiling but does not push past it | NO — gate adds zero new information per step |

**Key insight**: (A) and (B) are different. (A) is about training
distribution; (B) is about per-step damage propagation. Multistep H=8
addresses (A) and *partially* addresses (B) (because the model trains
through a polluted buffer and learns to denoise it). Drift gate
addresses (B) directly and explicitly, with *zero* effect on (A).

So mechanistically the combination is **not redundant** — it should
super-add on systems where both (A) and (B) bite. The catch is that
multistep already takes a partial bite at (B), so the gate's marginal
contribution is reduced. Predicted regime: combo ≈ multistep H=8 plus a
small fraction of the drift-gated H=1 marginal benefit, *minus*
training-interaction loss (see §3).

Reference for orthogonality framing: Brandstetter 2022 (pushforward,
ICLR) explicitly notes their trick mitigates (A) but does *not* address
the underlying chaotic divergence; Mikhaeil 2022 NeurIPS frames (C) as
the hard ceiling no training trick can cross; Chakraborty 2024
(MP-NODE, 2407.00568) frames multistep penalty as a way to **train
through** (A) without bumping into (C) by chunking into windows shorter
than the Lyapunov horizon — which is exactly why H=8 works.

---

## 2. Mechanistic prediction of the combo

Predictions vs. our four key dials:

### 2a. step-1 MSE (vs multistep_mamba H=8 baseline)
- Multistep alone: ~1.4e-4 (100× regression vs H=1)
- Combo predicted: **same order, possibly 1.5–3× worse**.
- Reasoning: gate adds 2 scalar params with their own gradient
  contention; if gate stays near-open at h=0 (low drift on clean
  inputs), it should not block the step-1 loss. But if §3 risks
  manifest (gate closes everywhere), step-1 MSE drifts toward the
  rens-like ~2e-5 (which is still good!) plus poisoned multistep
  contribution. Net: 1.4e-4 to ~5e-4 plausible.

### 2b. gs H=15 abs MSE (vs mamba_rand H=1 baseline 6.47e-5)
- mamba_rand alone: 6.47e-5 (36× win over rens)
- multistep alone: 1.84e-2 (very poor short-horizon!)
- Combo predicted: **between 5e-4 and 5e-3**.
- Reasoning: at H=15 we're close to the Lyapunov ceiling; the gate
  should be partially closed in the late-rollout steps but mostly open
  in early steps. Best-case: gate-open early steps recover most of the
  mamba_rand short-horizon advantage. The MSE is dominated by the late
  steps though — H=15 abs MSE is *cumulative* over the rollout. So the
  late-step fallback is where most of the value lives.
- The combo H=15 figure is the most uncertain prediction in this doc.

### 2c. gs H=100 abs MSE (vs multistep H=8 baseline 3.51e-2)
- multistep alone: 3.51e-2 (cos_div 0.04, stable)
- Combo predicted: **3e-2 to 5e-2**, cos_div ≤ 0.05.
- Reasoning: at H=100 we're far past the Lyapunov ceiling, deep in
  invariant-measure territory. Multistep handles this by training the
  long-horizon distribution explicitly (a la Jiang 2023 / 2306.01187
  invariant-measure preservation). Gate fallback to rens-like behavior
  at high drift further dampens any late-rollout blow-ups. Both
  mechanisms push in the same direction here. Modest improvement
  expected, NOT a 2× win.

### 2d. Gate dynamics during training
- Initial state: gate ≈ 0.62 (sigmoid(0.5)); fully ungated baseline.
- Single-step training: gate sees `||x_t - cml_mean.detach()||` on
  *ground-truth* x_t — small drifts, gate stays near-open everywhere.
- Multistep H=8 training: from h=2 onward, x_t is the model's own
  prediction. Drift is *systematically larger* for h≥2 than h=0 (this
  is provable: multistep rollouts wander). Gate gradient: pressure to
  close at h≥2.
- **Predicted equilibrium**: gate becomes a "step-2-or-later detector"
  rather than a "drift detector". This is the key risk (R1, §3).
  Weak-form synergy: even a step-index proxy gate is better than
  nothing because it does *attenuate mamba contribution late in
  rollout, where it's most dangerous*. Strong-form synergy (drift
  discrimination across coherent vs. incoherent late-rollout
  trajectories) is unlikely to emerge from this training regime alone.

---

## 3. Three risk scenarios

### R1. Gate collapse to step-index detector
- **Mechanism**: as in §2d. Multistep training systematically presents
  high-drift inputs at h≥2 and low-drift inputs at h=0 (clean ground
  truth). Gate learns this trivially, becomes a step-index proxy with
  no sensitivity to whether a particular trajectory is well-behaved or
  drifting.
- **Detection**: log gate output `g_h` averaged per rollout step. If
  `g_0 ≫ g_1 ≫ ... ≫ g_7` and there's no within-step variance, gate
  collapsed to step-index.
- **Mitigation**: (a) feed the gate ground-truth-conditioned drift on
  occasional batches (mix `x_t` with real next-frame to break the
  step-index correlation); (b) detach the gate's gradient on the
  multistep penalty contribution and only train it on H=1 reconstruction
  (decouples gate training from the noisy multistep gradient — closely
  related to the proposal in §brainstorm_combo_synergy 2.4); (c) use
  curriculum: train mamba+gate on H=1 first to lock in a drift-aware
  gate, *then* unfreeze and add multistep penalty.
- Reference class: MoE routing collapse (Chi 2022 EMNLP, 2204.09179)
  shows two-scalar gates under noisy gradients tend to collapse to
  trivial classifiers; same phenomenon, smaller scale.

### R2. Drift signal poisoning
- **Mechanism**: the drift signal `||x_t - cml_mean.detach()||` is a
  reasonable indicator under in-distribution rollouts (small=coherent,
  large=drifting). But under multistep training the predictions are
  systematically biased (mean-shifted from a slightly-off correction at
  h=1). The drift signal becomes a constant offset rather than a
  discriminator. Gate then either saturates one direction or gets
  re-anchored on the wrong distribution.
- **Detection**: histogram drift values over a rollout; if drift values
  are bimodal cleanly (h=0 vs h≥1), gate can learn from them. If they
  are unimodal-shifted, gate has nothing to learn.
- **Mitigation**: switch the drift signal to a step-relative form: `||x_t -
  predicted_x_t_from_only_cml||` (residual from rens-like prediction).
  This removes the systematic bias because the rens prediction tracks the
  trajectory's mean shift. Also: add EMA on `cml_mean` instead of detach,
  so it adapts during multistep training.

### R3. Gradient interaction / two-scalar starvation
- **Mechanism**: gate has 2 scalar params; mamba+NCA has ~5300. Under
  multistep training, gradients amplify ~50× per step (per Mikhaeil
  2022 logistic constraint). With clip-at-1.0, the global gradient
  budget gets distributed disproportionately, AND the noise floor on
  the two scalars is higher than on the mamba parameters (smaller
  effective batch per param). Empirically two-scalar sigmoid gates in
  noisy regimes either (a) saturate within ~50 steps, or (b)
  high-frequency oscillate.
- **Detection**: track gate_bias and gate_scale per step. Saturation:
  monotonic toward extreme. Oscillation: high variance with no trend.
- **Mitigation**: (a) lower LR specifically on gate params (factor
  10–100); (b) add weight decay on gate params; (c) gradient-clip gate
  params *separately* with a much tighter clip than the rest of the
  model; (d) initialize `gate_bias` larger (e.g. 1.0 rather than 0.5)
  so initial sigmoid is closer to 1.0 — preserves H=1 baseline behavior
  while gate is learning.

All three are real but **not show-stoppers**. R1 is the most likely; R3
is the hardest to detect.

---

## 4. Reference class from literature

I searched ~10 queries for prior work combining architectural gating
with multistep / scheduled-sampling / pushforward training in
chaotic/dynamical contexts. Findings:

- **Bengio 2015** (1506.03099, scheduled sampling) — original exposure
  bias / scheduled sampling for RNNs. No architectural gating, just
  curriculum. Reference baseline for our (A).
- **Brandstetter 2022 ICLR** (2202.03376, pushforward + neural PDE) —
  noise injection as exposure-bias mitigation. No architectural gating.
- **Chakraborty 2024** (2407.00568 MP-NODE, 2410.05572 MP-FNO) —
  multistep penalty for chaos. No gating; pure training-loss
  intervention. Notes that loss landscape non-convexity is the main
  problem and that windowing under the Lyapunov horizon helps. Closest
  reference for our (A) treatment.
- **Chen 2024 NeurIPS** (2407.01392, Diffusion Forcing) — per-token
  noise levels acting as a soft "gate" on how much each timestep is
  trusted. This is conceptually closest to drift-conditioned multistep:
  noise level plays the role our drift gate plays, and training is over
  partial-rollout sequences. A direct theoretical analogue.
- **Huang 2025 NeurIPS** (2506.08009, Self Forcing) — autoregressive
  rollout *during training* with KV cache, holistic video-level loss.
  Combines architectural causal-attention with multistep training; no
  per-step gating but the holistic loss plays a similar role.
- **Pathak / Wikner 2018** (hybrid forecast: knowledge-based model +
  reservoir) — analogous structural hybrid (rens=reservoir,
  mamba=neural correction), but no multistep penalty and no drift
  gating. Our setup is closest to this lineage in *architecture*, with
  Chakraborty 2024 in *training*.
- **Jiang 2023** (2306.01187, invariant-measure-preserving neural
  operators) — Sinkhorn divergence on summary statistics, replaces
  RMSE-on-trajectory with statistic-matching. Different mechanism for
  attacking long-horizon chaos. Could combine *additionally* with our
  combo (future work).
- **Asabuki & Clopath 2025** (Nature Comms, predictive alignment) —
  biologically-motivated chaos-taming via local prediction-feedback
  alignment. Mechanism orthogonal to ours (not a fork-or-train trick),
  but reinforces that "tame chaos by aligning with internal
  predictions" is a productive frame.
- **Engelken et al. 2023** (Phys Rev Research, Lyapunov spectra of
  RNNs) and **Mikhaeil 2022** — set the chaos ceiling.

**Verdict on reference class**: the *closest* published analogue is
**Diffusion Forcing (Chen 2024)** — per-token noise level acts like
our per-step drift gate, training is over rollouts. Self Forcing (Huang
2025) is the next closest. Neither is identical: both work in a
diffusion / token-prediction frame and don't have our reservoir-residual
structure. Notably I found *no* paper that directly combines (i)
architectural gating between two prediction pathways and (ii) multistep
penalty / scheduled-sampling training, *for chaotic continuous
dynamics*. This is a small but nontrivial gap. The combination is
defensible as novel; not so novel that it's reckless to predict it
should work in some regime.

---

## 5. Predicted outcome ranges

Concrete predictions for the *vanilla* combo (`ResCorMambaGated` +
`multistep_horizon=8`, no §brainstorm_combo_synergy modifications):

### Best case (~15% probability)
- step-1 MSE: 5e-5 to 1e-4 (mostly the multistep cost)
- gs H=15 abs: ≤ 5e-4 (within 8× of mamba_rand H=1, well below rens)
- gs H=100 abs: ≤ 2.5e-2, cos_div ≤ 0.03
- Mechanism: gate learns a real drift discriminator, attenuates
  mamba_rand at high-drift late-rollout steps, preserves it on
  near-manifold steps. Multistep penalty fixes invariant-measure
  preservation.
- Conditions: probably requires Mitigation R1(c) (curriculum) and
  R2 (drift-signal redesign). Without those, best case unlikely.

### Boring middle (~55% probability)
- step-1 MSE: 1e-4 to 5e-4
- gs H=15 abs: 1e-3 to 1e-2 (similar order to multistep alone, possibly
  marginally better due to gate fallback)
- gs H=100 abs: 3e-2 to 5e-2, cos_div ≤ 0.06
- Mechanism: gate collapses to step-index detector (R1 manifests),
  effectively becomes a "use less mamba late in rollout" heuristic.
  This is *additive* not synergistic — slight improvement on H=15 vs
  multistep alone, similar H=100.
- This is what we should expect. Multistep H=8 is a strong baseline;
  gate is a small additional regularizer.

### Worst case (~30% probability)
- step-1 MSE: ≥ 1e-3 (gate destabilizes early-step training)
- gs H=15 abs: ≥ 1e-2 (no improvement on multistep alone)
- gs H=100 abs: 4e-2 to 8e-2, cos_div potentially worse than multistep
  alone
- Mechanism: gate either saturates (R3) or oscillates; gradient noise
  from multistep penalty floods the two-scalar gate; gate provides no
  useful signal *and* costs us some mamba expressivity.

Expected value: similar-to-marginally-better than multistep H=8 alone
on H=15 and H=100, with non-trivial probability of regression.

---

## 6. Pre-experiment go/no-go

**Recommendation: SOFT NO on the vanilla combo.** Prefer cheaper
diagnostics first.

Reasons:

1. **The reference class predicts incremental, not synergistic, gains.**
   No paper combining (A)-mitigation training with (B)-mitigation
   gating shows multiplicative gains in chaos forecasting. Diffusion
   Forcing (the closest analogue) gets its gains from the diffusion
   denoising structure, not from the per-token noise level acting as a
   gate.
2. **R1 is highly likely**, and once it manifests we've spent the GPU
   without learning whether the *non-trivial* version (drift gate
   actually discriminating coherence) would have worked.
3. **The §brainstorm_combo_synergy proposals are explicitly designed
   around the failure modes I identified above.** Running vanilla combo
   first essentially throws away the failure-mode analysis.

### Cheaper diagnostics first (in priority order):

**D1. Gate-dynamics probe (≤2 GPU-hr)**: train `ResCorMambaGated` H=1
to convergence; then *evaluate* gate output on rollouts of length 50
(no training). Plot `g_h` vs h for both clean trajectories and
intentionally-poisoned ones (initial perturbation 1e-3). This tells us
whether the drift signal (under H=1 training, before any multistep
poisoning) actually discriminates coherent from drifting predictions.
If the gate doesn't discriminate even under clean training, the combo
has no chance — the drift signal is broken and we need R2's
redesigned signal first.

**D2. Drift-signal histogram probe (≤30 GPU-min)**: same as D1, but
just measure the drift histogram on multistep H=8-trained mamba_rand
(no gate). If the histogram is unimodal-shifted (R2 manifests), redesign
the drift signal before adding the gate.

**D3. Two-stage curriculum probe (≤6 GPU-hr)**: train H=1 with gate to
convergence; freeze gate; then enable multistep H=8 only on mamba+NCA.
If this gives a nontrivial improvement over multistep alone, that's
strong evidence the gate has value. If not, the drift gate is not a
useful inductive bias for this task and we should pivot.

**D4. Vanilla combo (full run, ≤24 GPU-hr)**: only after D1–D3 are
green. Running this first risks ambiguous results.

**Optional D5 (if D1 succeeds but D3 fails)**: implement one
§brainstorm_combo_synergy proposal (likely 2.4, "decouple gate gradient
from multistep penalty") and re-run.

### Bottom line

The combo is theoretically defensible but mechanistically fragile.
Vanilla stacking is more likely to land in the "boring middle" than to
synergize. The *expected* outcome is a marginal H=15 improvement over
multistep alone, with H=100 essentially unchanged — useful as
confirmation but not as a sprint deliverable. Run D1 + D2 first
(combined ≤3 GPU-hr); decide on full combo based on those.

---

## References (queried, year + first author)

- Bengio 2015 — Scheduled Sampling (1506.03099)
- Brandstetter 2022 ICLR — Message Passing Neural PDE Solvers / pushforward (2202.03376)
- Chakraborty 2024 — MP-NODE (2407.00568); MP-FNO/UNet (2410.05572)
- Chen 2024 NeurIPS — Diffusion Forcing (2407.01392)
- Chi 2022 — Sparse MoE representation collapse (2204.09179)
- Engelken 2023 — Lyapunov spectra of chaotic RNNs (PRR)
- Huang 2025 NeurIPS — Self Forcing (2506.08009)
- Jiang 2023 — Invariant-measure-preserving neural operators (2306.01187)
- Mikhaeil 2022 NeurIPS — On the difficulty of learning chaotic dynamics with RNNs (2110.07238)
- Pathak / Wikner 2018 — Hybrid forecasting (Chaos)
- Asabuki & Clopath 2025 — Predictive alignment (Nat Comms)
