# Drift-gated hybrid + multistep penalty: synergy brainstorm

Status: research-mode brainstorm, 2026-04-29 (Sprint Day 4 prep).
Goal: find an interaction between drift-gated hybrid (`ResCorMambaGated`)
and multistep penalty training that gives more than additive gains on the
all-horizon-stability sprint targets (gs H=15 absolute MSE near
mamba_rand baseline; gs H=100 absolute MSE near multistep H=8 baseline).
Sprint is **deterministic** (no stochastic latents); ideas must fit within
≤1 week of dev work.

Key data points carried in from §63 / §64 / §65 / §66 (Day 0-3):

| Variant                       | gs step1 MSE      | gs H=15 abs       | gs H=100 abs / cos_div |
|-------------------------------|-------------------|-------------------|------------------------|
| rens K=32 (reference)         | 2.05e-5           | 9.66e-4           | 3.50e-2  / —           |
| mamba_rand (Day 0 H=1)        | ~1.4e-6 (15× win) | ~6e-5 (15× win)   | ~1.8e-3 / **0.30 catastrophe** |
| mamba_rand multistep H=8 (Day 2-3) | ~1.4e-4 (~100× regression) | ~9e-4 (ratio 1.66) | ~2.6e-4 / **0.04 stable** |
| Synergy target                | ≤2× of Day 0      | ≤2× of Day 0      | ≤2× of Day 2-3         |

The two trades are opposite in time: H=1 great early, multistep great
late. Drift gate is the natural mediator. But trivial stacking (just
train `ResCorMambaGated` with `multistep_horizon=8`) is **not** obviously
synergistic — see §1.

---

## 1. Failure-mode reframing — why naive stacking might NOT synergize

Three concrete interaction risks:

**1.1. Gate freezes at near-uniform during multistep training.** During
multistep rollout the model's own predictions become the input from h≥2
onward. By construction, those predictions diverge from `cml_mean` more
than ground-truth frames do. So during training the gate sees a
*systematically higher drift distribution* than at single-step
inference. With `gate_scale_init=1.0, gate_bias_init=0.5` (initial gate
~0.62), gradient pressure will push `gate_bias` *down* and/or `gate_scale`
*up* to attenuate at this elevated drift floor. By end of training the
gate is essentially "off" on rollout inputs and "on" on h=0 inputs —
i.e. a glorified step-aware switch that hasn't learned to discriminate
*coherent vs. drifting* predictions, only *first-step vs. rollout-step*.
That's the wrong axis. It mirrors the §61 / Task #32 noise-injection
trap where a regularizer learned to compensate for a training artifact
rather than the rollout failure mode.

**1.2. Drift signal is poisoned by the training distribution.** The drift
metric `||x_now - cml_mean.detach()||` is computed on the model's own
predictions during multistep rollout. If the predictions are biased
(e.g. mean-shifted from a slightly-off correction at h=1), the drift
signal becomes a constant offset rather than a discriminator. The
detach() on `cml_mean` was correct under single-step training (clean
graph) but exposes us to this issue when the input `x_now` is itself a
multi-step prediction.

**1.3. Multistep penalty's gradient noise floods the gate's two
scalars.** The gate has only two trainable scalars vs. mamba+NCA's
~5300 params. Multistep training amplifies gradients ~50× per step
(Mikhaeil 2022 — that's why we clip at 1.0). Two scalar params get a
disproportionate share of the clipped budget *and* a disproportionate
share of the noise. Empirically, two-scalar gates under noisy gradients
tend to either (a) collapse to one of the sigmoid extremes within ~50
steps, or (b) oscillate. Neither is the calibrated drift detector we
want.

**Summary**: trivial stacking (Variant A in §3) likely produces a gate
that learned step-index, not drift; or a gate that has collapsed to one
extreme; or a gate that drifts during training. Synergy requires
*explicitly* shaping the interaction so the gate sees a learning signal
about *coherence*, not about *step index*.

---

## 2. Six synergy proposals

### 2.1. Drift-conditioned step weight (the user's hypothesis #1)

- **One-line**: in the multistep loss, weight step h's loss by `(1 -
  gate(drift_h))` so we only penalize H=h prediction quality when the
  prediction is *claimed coherent*.
- **Math sketch** (modifying the loop in `model_registry.py:843-878`):
  ```
  for h in range(H_ms):
      pred = model(state)          # forward
      gate_h = model.compute_gate(state[:,-1] if 5d else state, cml_mean_of(state))
      w_h    = (1.0 - gate_h.detach().mean())   # high when gate thinks "drifted"
      loss_h = mse(pred, gt_h) * (alpha + (1-alpha) * w_h)
      losses_h.append(loss_h * ms_step_weights[h])
  ```
  with `alpha=0.5` so we never zero a step out entirely. Detach the gate
  from the weight (otherwise the gate gets a trivial gradient: "claim
  high drift to lower your own loss weight").
- **Effort**: 1-3 days (small loop change + new training-config kwarg
  `drift_conditioned_loss=True` + 2-seed sweep at H_ms=8 on gs).
- **Why synergy not additive**: ties the gate's behavior *during
  training* to the rollout objective. The gate now has a meaningful
  gradient: closing it on a drifted step *reduces that step's
  accountability* but `(1-gate)` weighting redirects loss budget toward
  the truly drifted steps. Net effect: gate learns to be *honest* about
  coherence (because being dishonest means under-weighting the steps
  where the prediction quality is actually graded). And the multistep
  loss focuses budget on the failure modes we care about.
- **Risk**: with alpha tuned wrong the model can game `(1-gate)` into a
  free pass on hard steps. Detach is critical. Also: requires
  re-running the cml_mean computation per rollout step, ~+15% training
  wall-clock.

### 2.2. Gate-temperature curriculum anneal (user #2)

- **One-line**: linearly decay `gate_scale` from a hot init (e.g. 5.0,
  near-binary) to its trainable equilibrium (~1.0) over the first 30%
  of training, then unfreeze.
- **Math sketch**: parameterize `gate_scale_eff = clamp(gate_scale,
  min=schedule(epoch))`. Schedule: linear from 5.0 at epoch 0 to 0.1 at
  epoch 0.3*E, then no-op (trainable scalar takes over).
- **Effort**: ≤1 day. Add an epoch-aware hook in `train_model`'s loop
  pre-forward.
- **Why synergy**: forces the model to live with a hard gate during the
  exposure-bias-heavy multistep training (so it learns to *not produce*
  drifted predictions in the first place rather than relying on the
  gate to clean them up). Then anneal opens the gate to allow finer
  near-manifold detail. This is a "curriculum from architectural
  brittleness toward calibrated softness" rather than "stack two
  stabilizers." Specifically targets failure mode 1.1 (gate freezing as
  step-index discriminator) by making the gate hard before the rollout
  distribution shift can poison it.
- **Risk**: schedule is one more hyperparameter; if the anneal is too
  fast we lose the curriculum effect, too slow we lose half the
  trainable capacity. 5.0 → 0.1 is a guess; should sweep on one seed
  before multi-seed.

### 2.3. Gate as auxiliary calibration regularizer (user #3, sharpened)

- **One-line**: add an auxiliary loss `λ * gate * (pred -
  cml_mean.detach())^2` so the gate is *forced* to close when the
  correction is large (regardless of whether the correction is
  helpful).
- **Math sketch**: `loss_aux = lambda_cal * (gate.detach() * correction
  ** 2).mean()` — note: detach `gate`, not `correction`. The correction
  gets a soft L2 weighted by gate magnitude; the gate gets the
  symmetric penalty `(1 - gate.detach()) * correction**2.detach()` via
  the gate's own grad path. Cleaner formulation: minimize the
  *covariance* `E[gate * |correction|]` so they're forced to be
  anti-correlated.
- **Effort**: 1-3 days.
- **Why synergy**: gives the gate a *direct* signal independent of the
  multistep loss — it has to discriminate large-correction from
  small-correction states. Multistep loss alone gives only an
  end-to-end signal that, as failure mode 1.3 notes, gets crushed under
  gradient clipping. The aux loss is a "prior" that says: a confident
  gate (high) means a small correction. Forces calibration so by the
  time the multistep gradient signal arrives, the gate is already
  pointing the right direction.
- **Risk**: the aux loss can drag the gate to always-zero (trivial
  solution: gate=0 → aux loss = 0). Need a dual-anchored formulation
  (penalty on both gate=0-with-large-correction AND gate=1-with-large-
  correction) or rely on the multistep loss as the counterforce. Pick
  λ small (1e-3 ish) and verify gate doesn't collapse.

### 2.4. Two-tower decoupled training (user #4)

- **One-line**: train two specialist models (mamba_rand single-step for
  H=15 fidelity, mamba_rand multistep H=8 for H=100 stability), then at
  inference gate-mix their outputs by observed drift.
- **Math sketch**: `pred_t = gate(drift) * pred_specialist_short(x_t) +
  (1 - gate(drift)) * pred_specialist_long(x_t)` where the gate is
  fitted post-hoc on a held-out rollout trace by least-squares (or just
  hard-thresholded at drift_p50).
- **Effort**: 1-3 days. Two existing training runs (already have one),
  plus a small inference-time mixer.
- **Why synergy**: completely decouples the two failure modes
  architecturally. No gate-vs-multistep gradient interference because
  the gate is *not trained jointly*. We get the H=1 fidelity of Day 0
  AND the H=100 stability of Day 2-3 with no compromise — provided the
  drift signal at inference reliably discriminates which specialist is
  better. The price is 2× param count and 2× inference compute.
- **Risk**: doubles inference cost (relevant for the Dreamer fork's
  imagination rollouts where this is the inner loop). Also: at the
  cross-over drift threshold you get a discontinuity that itself can
  destabilize a long rollout. Fix with a soft mixer + ensemble
  variance, but that's a third hyperparameter.

### 2.5. Gradient-routed dual loss (user #5, modified)

- **One-line**: backprop multistep loss only through Mamba (NCA frozen
  for multistep loss); backprop a single-step loss only through gate +
  NCA. Cross-block gradient firewall.
- **Math sketch**: maintain two loss heads in the same forward pass —
  `loss_ms` (multistep H=8) and `loss_h1` (single-step from h=0 of the
  rollout). Apply per-parameter backward hooks: gate params get only
  `loss_h1.backward()`; mamba params get only `loss_ms.backward()`; NCA
  params get a weighted blend `α * loss_h1 + (1-α) * loss_ms`. Use
  `optimizer.zero_grad()` then two `.backward(retain_graph=True)` calls
  with parameter-group filtering.
- **Effort**: 1-3 days (PyTorch parameter-group plumbing is a known
  dance, but the gradient routing logic is fiddly).
- **Why synergy**: directly addresses failure modes 1.1, 1.2, 1.3 *all
  at once*. Gate doesn't see rollout-distribution drift inputs (no
  failure 1.1). Drift signal is computed on h=0 only, where it's clean
  (no failure 1.2). Gate's two scalars only see clean single-step
  gradients (no clipped/amplified noise — failure 1.3). Mamba gets the
  rollout signal it needs for stability. NCA gets both. Decoupling at
  the parameter group level = surgical synergy.
- **Risk**: most complex of the six. Easy to silently mis-route a
  gradient. PyTorch's parameter-group hooks have sharp edges with
  optimizer state. Highest cognitive cost to debug.

### 2.6. Drift-adaptive K_bptt (user #6)

- **One-line**: in multistep training, dynamically truncate K_bptt per
  sample based on observed drift — high drift → fewer BPTT steps.
- **Math sketch**: at the start of each rollout, after the no-grad
  warmup `n_no_grad`, check `drift_t = compute_drift(state[:,-1],
  cml_mean)`. If `drift_t > drift_thresh`, set `K_bptt_eff = max(1,
  K_bptt - 2)` for this sample (or set `n_no_grad` higher). This sample
  gets less rollout-grad signal because its rollout has already
  diverged.
- **Effort**: ≤1 day.
- **Why synergy**: the multistep penalty's worst behavior is when
  gradients are amplified through chains of off-manifold predictions —
  exactly the regime where drift is high. Adapting K_bptt to drift
  means we *trust the model's gradient signal less* in regimes where
  the prediction has already gone bad, and *more* when it's still on
  manifold. The gate informs which samples deserve deep BPTT. This
  uses the gate as a *training-time gradient sentinel*, not just an
  inference-time mediator.
- **Risk**: per-sample K_bptt is awkward in batched code (current
  implementation processes all samples in a batch with the same K_bptt
  loop). Either group-by-K_bptt within the batch (2-3 micro-batches
  per training step, 2-3× wall-clock cost) or accept that K_bptt is
  per-batch (mode of the per-sample drift), losing some of the signal.

---

## 3. Top-2 picks

### Pick A: Proposal 2.1 (drift-conditioned step weight)

**Reasoned recommendation**: highest synergy potential per dev-day. It
*directly* couples the gate's output to the multistep loss's per-step
budget, so the two changes are no longer separable. Failure modes 1.1
and 1.2 are mitigated because the gate now has a clear gradient
signal about coherence (its claim about coherence determines which
steps' losses count). It's also the cleanest experimental signal: if
it works, you'll see lower H=1 cost than vanilla multistep (because the
loss-budget is reallocated away from grossly-drifted late steps) AND
preserved H=100 stability (because the gate is properly calibrated).

**First-experiment config**:
- Variant: new `rescor_mamba_gated_msdc` (mamba gated + multistep
  drift-conditioned), 5348 params + the 2 gate scalars.
- `multistep_horizon=8`, `multistep_bptt=4` (match Day 2-3).
- `drift_conditioned_loss=True`, `alpha=0.5` (the floor weight, so
  gate=1 → step weight=0.5; gate=0 → step weight=1.0).
- Optimizer: AdamW lr=3e-4, grad-clip 1.0 (carry over from Day 2-3).
- 200 trajs × 100 ep × 3 seeds, gs only for the first cut, then ks
  + heat if gs passes the synergy gate (§4).
- Compute budget: ~2-3 hours on the H100 pod (carry over Day 2-3
  per-seed runtime).

### Pick B: Proposal 2.5 (gradient-routed dual loss)

**Reasoned recommendation**: highest synergy *ceiling* — it explicitly
fixes all three failure modes — but more dev cost. Worth running in
parallel with Pick A because they attack the same problem from
different angles (training-loss reshaping vs. parameter-group
isolation), and the signal from one informs the other. If A works, we
learn that the gate's training-time signal needs to be coupled to the
loss but not isolated parameter-wise. If B works, we learn the
opposite: parameter isolation is sufficient. If both work, we have a
choice between a simpler-to-implement (A) and a more-modular (B)
solution.

**First-experiment config**:
- Variant: `rescor_mamba_gated_dual` — same `ResCorMambaGated` backbone.
- Two loss heads: `loss_h1 = mse(model(x_t), x_{t+1})` (single-step,
  from the h=0 of the rollout); `loss_ms = sum_h mse(model(state_h),
  x_{t+1+h})` for h in 0..7.
- Param routing:
  - gate_scale, gate_bias: gradients only from `loss_h1`.
  - mamba.* : gradients only from `loss_ms`.
  - nca.* : gradients from `0.5*loss_h1 + 0.5*loss_ms`.
- `multistep_bptt=4`, `H=8`, AdamW lr=3e-4, grad-clip 1.0.
- 200 trajs × 100 ep × 2 seeds (one fewer than Pick A because the dev
  cost is higher and we want a fast read first).
- Compute budget: ~2-3 hours per seed.

---

## 4. Decision rule — what counts as "synergy"

A run from Pick A or Pick B counts as a **synergy result** if and only
if BOTH of the following hold on gs (3-seed median, bf16):

1. **Step-1 + H=15 fidelity preserved**: gs H=15 absolute MSE within
   2× of mamba_rand single-step (Day 0) baseline, i.e. ≤ ~1.2e-4.
2. **H=100 stability preserved**: gs H=100 absolute MSE within 2× of
   mamba_rand multistep H=8 (Day 2-3) baseline, i.e. ≤ ~5.2e-4.

Equivalently: gs H=15 is no worse than 2× of "good early" baseline AND
gs H=100 is no worse than 2× of "good late" baseline. Hitting both is
synergy; hitting one is just additive (we already know we can hit
either alone).

Soft tiebreakers (only consulted if 1 and 2 are both met):
- gs H=100 cos_div ≤ 0.10 (no near-orthogonal drift, like the §65
  pushforward catastrophe at 0.47).
- ks H=100 absolute MSE not regressed >2× from Day 0.
- heat is *not* in the gate — diffusion's near-zero attractor breaks
  the metric (§64-§66 caveat carried forward).

A **partial synergy** (one of 1 or 2 met, the other within 3-5×) is a
weak positive: log it, run a second seed, but don't ship it as the
sprint outcome.

---

## 5. Open questions

1. **Drift metric scale**: is `mean_C((x - cml_mean)^2)^0.5` the right
   coherence signal, or should it be something normalized (e.g. by
   `cml_var`)? An unnormalized drift metric tracks both "wrong
   direction" and "high-variance region" — only the first is what the
   gate should respond to. Fixing this might be a free uplift before
   any of the six proposals.

2. **Gate distribution at end of multistep training**: we have no
   instrumentation for where `gate_scale, gate_bias` end up on a
   multistep run. Day 2-3 didn't include the gated variant. Adding a
   tiny logging hook (`epoch_end: log gate stats`) to the next run
   would directly tell us whether failure mode 1.1 is real before we
   commit to one of the proposals.

3. **Pick A's `alpha` floor**: I picked 0.5 because it leaves a
   non-trivial baseline weight even when gate=1, but the right value
   probably depends on the drift distribution. Could be the difference
   between synergy and a regression. Sweep 0.25/0.5/0.75 on one seed
   first.

4. **Which substrate to validate first**: gs is the headline (per §63
   /§65/§66 framing). But ks is where Day 0 had less variance and
   pushforward had a real win. Does running on ks first risk
   overfitting the synergy-detection rule to the easier substrate? My
   read: gs first, ks second, on the principle that the failure mode
   we're attacking is gs-shaped.

5. **Two-tower (Pick 2.4) as a fallback**: if both A and B fail to
   meet the §4 decision rule, is the two-tower architecture (decouple
   at inference) acceptable as a "we didn't synergize but we
   architecturally avoided the trade" outcome? Worth pre-registering a
   third tier so we don't wander into it post-hoc.

6. **Gate-init for multistep**: current init is `gate_scale=1.0,
   gate_bias=0.5` → gate≈0.62 at drift=0. For multistep training we
   probably want a higher initial gate (so the model uses the
   correction at h=0 and learns to attenuate as drift accumulates). Try
   `gate_bias=1.0` (gate≈0.73 at drift=0) for the first multistep run?
