# Brainstorm: Training-Loss / Curriculum / Regularization Levers for WMCA Rollout Stability

Status: research-mode brainstorm only. No code outside this doc.
Scope: training-time changes that target free-run rollout divergence at H=15 / H=50 / H=100+ on the existing rescor_rens K=32 (321 params) and rescor_mamba_rand (~5.3K params) cores. Stochastic-output / posterior-matching objectives are explicitly excluded; these are levers for the deterministic core.

---

## 1. Failure-mode analysis

### Why single-step MSE produces models that diverge autoregressively

Training minimizes E_{(x_t, x_{t+1}) ~ D_train}[ ||f(x_t) - x_{t+1}||^2 ] under the implicit assumption that f's input at every prediction step is drawn from the same distribution D_train as during training. That assumption is exactly what autoregressive rollout violates. At step k of free-run, the input is f^{k-1}(x_0), and even an arbitrarily small per-step error (epsilon) accumulates: x_t^pred = x_t^true + Sum_{i<t} J(x_i) * epsilon_i, where J is the Jacobian along the trajectory. On chaotic substrates, J's spectral radius is > 1 along an unstable manifold (Mikhaeil et al. 2022 measured ~15,000x amplification at r=3.99, M=15 for the logistic map alone — exactly the regime our K=32 reservoir spans). Single-step MSE has no signal that distinguishes a model whose prediction lies on the data manifold from one that lies just-off it; both incur the same loss. So gradient descent picks whichever generalizes step-1 best, with no preference for staying near the manifold under iteration. This is the classic "exposure bias" of teacher-forced training (Bengio et al. 2015 / Ranzato et al. 2016), specialized to a chaotic dynamical system.

### Concrete diagnosis for our cells

`findings.md` §59-§60 quantify the regime: rens K=32 reaches step-1 MSE ~1e-3 to 1e-6 (excellent fit) but H=15 ratio of 17-126x on gs/ks/Crafter latents, climbing to 6056x at H=100 on gs. cos_div crosses ~0.03 at H=50 — predictions have rotated meaningfully off the target manifold. This is not under-fit step-1; it is per-step error projected onto unstable Jacobian directions. §62 shows mamba_rand's K=4 temporal context cuts the H=15 ratio ~3x on gs at one-quarter the data, but its absolute step-1 MSE is ~2.3x worse — the temporal context softens the *direction* of the error without reducing its *magnitude*. Both cores are pinned by the same root cause: the training objective rewards step-1 fit on D_train without rewarding rollout-manifold-staying. Section 2 enumerates levers that re-introduce the rollout-manifold signal into the gradient.

---

## 2. Training-loss / curriculum / regularization ideas

### 2.1 H-step rollout MSE with truncated BPTT (TBPTT-H)

**1-line summary**: Roll H>1 steps in the forward, sum step-wise MSE, backprop through a truncated window of K_bptt steps.

**Math sketch**:

  L = Sum_{k=1..H} w_k * || f^k_theta(x_0) - x_k ||^2
  with stop_gradient applied to f^j_theta(x_0) for j < (k - K_bptt).
  Default H = 4, 8, 16; K_bptt = 4.
  Weights w_k uniform initially, optionally w_k = exp(-alpha * k) to dampen long-horizon explosion.

**Why it should help**:
- **Short (H=15)**: directly trains on the regime we care about most. The k=1..15 terms are exactly what we evaluate.
- **Mid (H=50)**: H=8 or 16 with stop_gradient at K_bptt=4 still pushes the model to keep predictions on-manifold for several steps, because the *forward* error from step k > K_bptt still contributes to the step k=K_bptt+1 backward signal via the no-grad input.
- **Long (H=100+)**: indirect; we don't unroll that far, but reducing the per-step Jacobian-amplification factor by training on H=8 transfers most of the way (cf. Mikhaeil 2022: training horizon and rollout horizon decouple once Lyapunov signature is suppressed).

**Implementation**: 1 day. Touches `train_model` in `model_registry.py`: add `rollout_h: int = 1` and `bptt_window: int = None` kwargs, change the inner loop to step the model rollout_h times and accumulate loss. Need consecutive frame triples / quadruples in the batch, which means the training data sampler has to be updated (cheap — the existing Y_train order is consecutive in the synthetic benchmarks).

**Compute overhead**: ~H × (1 + K_bptt/H × bwd-fwd-ratio). At H=8, K_bptt=4: ~5x training time vs single-step. At H=4, K_bptt=4 (full BPTT): ~4x.

**Risk**:
- Gradient explosion on the chaotic cells (gs, ks). Need gradient clipping (max_norm=1.0) — almost certainly load-bearing. Without clipping the training will NaN by epoch 5 on ks.
- For mamba_rand the K=4 temporal context interacts: a rollout step also rolls the SSM state, so K_bptt=4 unrolls 4 SSM steps × 4 frames = 16 ops, not 4. Memory overhead is ~16x, not 4x.
- Loss-shape conflict: w_k uniform makes step-15 and step-1 same weight, but step-15 is ~1000x larger naturally on chaotic cells, so the gradient is dominated by long-horizon noise. Likely need w_k = 1 / (running_mean(L_k) + eps) — adaptive normalization.

### 2.2 Scheduled sampling (K-curriculum)

**1-line summary**: Mix teacher-forced and free-run inputs during training with a per-batch Bernoulli schedule, annealed over epochs.

**Math sketch**:

  At step k of the H-step rollout, with prob p_epoch use teacher-forced x_k_true,
  with prob 1 - p_epoch use the model's own f(x_{k-1}_pred).
  p_epoch decays from 1.0 (teacher) to 0.0 (free-run) over training.
  Schedule: p_epoch = max(0, 1 - epoch / epochs * 1.5), so the model spends the
  last third of training in pure free-run.

**Why it should help**:
- **Short (H=15)**: bridges the train-test distribution gap exactly; the free-run phase tail of training measures and corrects on-manifold drift directly.
- **Mid (H=50)**: same mechanism; free-run K mixed with H=8 unroll exposes the model to step-{4..8} drift inputs at training time.
- **Long (H=100+)**: indirect via the same Lyapunov-suppression mechanism as 2.1.

**Implementation**: 1.5 days. Same hooks as 2.1 plus a Bernoulli mask per rollout step. Add `scheduled_sampling_schedule: Callable[[int], float]` to `train_model`.

**Compute overhead**: same as 2.1 (~4-5x at H=8).

**Risk**:
- The original Bengio et al. 2015 scheduled-sampling result is a discrete-output method; for continuous outputs you don't get the "argmax decoder" mismatch that motivated it, but the manifold-drift problem still applies. Less theoretically grounded but empirically standard.
- If decay is too aggressive, free-run early in training can train the model to predict its own bad outputs (collapse to a fixed point). Monitor: spectral radius of f's Jacobian over training. Decay slower if Jacobian shrinks below 1.0 (would indicate the model is learning a contractive map at the cost of step-1 accuracy).

### 2.3 Hindsight multi-horizon (multi-target)

**1-line summary**: Train one model that predicts frame_{t+1}, frame_{t+2}, ..., frame_{t+H} simultaneously from x_t, with weights schedule.

**Math sketch**:

  Let g_theta(x_t) -> [y_1, y_2, ..., y_H] be H separate output heads (or a single
  head conditioned on h in {1..H}).
  L = Sum_{h=1..H} w_h * MSE(y_h, x_{t+h})
  with w_h linear / log-spaced.

  At inference: still autoregressive (use y_1 to advance state). The other heads
  are auxiliary — they enforce that the *internal representation* of x_t carries
  signal about x_{t+H}, which biases f toward maps that don't lose long-horizon
  info on the first step.

**Why it should help**:
- **Short (H=15)**: the model's first-step prediction is regularized to be informative about future steps, so it picks first-step solutions that lie on the long-horizon-predictable submanifold.
- **Mid (H=50)**: directly tested at training time if H=50.
- **Long (H=100+)**: weaker; you can't realistically train H=100 heads. But a hierarchical version (H ∈ {1, 2, 5, 15, 50}) still pushes representation toward long-horizon information density.

**Implementation**: 2-3 days. New module `MultiHorizonHead` wrapping `ResidualCorrectionWM`. Or simpler: a head conditioned on horizon h via FiLM / embedding. For mamba_rand, the SSM state already carries multi-horizon info — adding a head reads it out.

**Compute overhead**: 1.2x to 1.5x (head forward is cheap; main cost is the H-step ground-truth gather, similar memory cost).

**Risk**:
- Doesn't guarantee rollout stability. The model could ace hindsight-50 prediction without ever rolling — a good representation does not imply a good iterated map. This is a regularizer on the *representation*, not the dynamics. May want to combine with 2.1.
- Output channel pressure: rens has 1ch output; H=8 hindsight gives 8ch output — 8x the readout params. Not catastrophic at 321 params -> ~2K, but worth tracking.

### 2.4 Contrastive teacher-forcing-vs-free-run regularization

**1-line summary**: Penalize the divergence between teacher-forced trajectory and free-run trajectory at every step.

**Math sketch**:

  Let traj_TF[k] = x_{t+k} (teacher forced — ground truth).
  Let traj_FR[k] = f^k(x_t) (free run).
  L = MSE(traj_TF[1], traj_FR[1])  // standard step-1 loss
    + lambda * Sum_{k=2..H} || traj_TF[k] - traj_FR[k] ||^2 / k
  with stop_gradient through teacher-forced trajectory.

  The 1/k decay weights early-step staying-on-manifold higher than late-step
  (which is dominated by Lyapunov amplification anyway).

**Why it should help**:
- This is essentially 2.1 written differently — but with the explicit "anchor to teacher-forced" framing, you can apply an L1 or Huber loss in the auxiliary term (more robust to chaotic spikes) while keeping L2 on step-1.
- The 1/k weighting is the missing piece in 2.1's risk discussion — it makes the auxiliary loss converge naturally.

**Why short / mid / long**: same as 2.1, but with better-behaved gradients in late H.

**Implementation**: 1.5 days. Sub-variant of 2.1.

**Compute overhead**: ~4-5x at H=8.

**Risk**: Choice of L1 / Huber for the auxiliary term is load-bearing — L2 on H=8 chaotic dynamics will explode. Recommend Huber with delta = sqrt(MSE_step1) * 3.

### 2.5 Energy regularization via frozen autoencoder

**1-line summary**: Train (or take) a small autoencoder on the same training data; penalize the reconstruction error of model predictions to keep them on the data manifold.

**Math sketch**:

  Pre-train AE_phi (encoder e + decoder d) such that d(e(x_t)) ~= x_t for all
  x_t in D_train, with the bottleneck small enough that off-manifold inputs are
  reconstructed worse than on-manifold ones.

  L = MSE(f(x_t), x_{t+1}) + mu * || d(e(f(x_t))) - f(x_t) ||^2

  The second term is the "energy" — large when f(x_t) lies off the AE manifold.
  AE is frozen during world-model training.

**Why it should help**:
- **Short (H=15)**: at training time we only see step-1 outputs, but the AE energy term prevents step-1 from drifting off-manifold even when it's still close to ground truth in MSE — captures *where* the prediction lies, not just how close to ground truth.
- **Mid (H=50)** / **Long (H=100+)**: indirect. If the model maps D_train -> D_train it must also map D_train -> D_train iteratively, so iterated rollout stays on-manifold. This is the cleanest mechanism for long-horizon stability that doesn't require unrolling at training time.

**Implementation**: 2 days for the AE pre-training (small CNN AE on the same training trajectories), ~0.5 day to wire the energy term into `train_model`. Total ~2.5 days. Requires a new `wmca/manifold_ae.py` module.

**Compute overhead**: 1.2x training (AE forward is small). Pre-training adds ~30 min of one-time cost.

**Risk**:
- AE bottleneck size is the central hyperparameter. Too narrow: AE's manifold is tighter than the true data manifold, model is over-regularized into a contractive map. Too wide: AE reconstructs off-manifold inputs well too, the regularizer is toothless. Need to sweep bottleneck dim ∈ {16, 32, 64, 128} per benchmark.
- Pre-training a separate AE is borderline "auxiliary task"; the user excluded those without specific connection. The connection here is direct: AE energy is a manifold-distance proxy for rollout stability. Keep the AE small (~5K params) and train on the same trajectories so it doesn't introduce out-of-distribution leakage.
- For Crafter latents: the AE substrate would have to be the latent itself (16x16 chaotic 4ch) — riskier than pre-training on smooth heat trajectories. Empirically untested.

### 2.6 Lyapunov / spectral-norm penalty on the Jacobian

**1-line summary**: Directly penalize the operator norm of the per-step Jacobian Jf to be < 1 on training inputs.

**Math sketch**:

  At each training step, sample a random unit vector v and compute
  Jf(x_t) v via vector-Jacobian product (autograd.grad).
  L = MSE(f(x_t), x_{t+1}) + nu * (max(0, ||Jf(x_t) v||_2 - tau))^2
  with tau ∈ {0.95, 1.0, 1.05} — slightly subcontractive.

  Power iteration with 1-2 steps gives a tight estimate of the spectral radius
  for ~1.5x forward cost.

**Why it should help**:
- This is the cleanest theoretical fix. Mikhaeil et al. 2022's 15,000x amplification is exactly the spectral-radius cube of training without this penalty. Penalizing the Jacobian operator norm directly attacks the root cause.
- **Short / mid / long**: all benefit. Spectral radius < 1 implies bounded rollout error growth (in expectation) — that's the textbook stability theorem.

**Implementation**: 2-3 days. Need vector-Jacobian product (functorch / torch.autograd.grad), 1-2 power-iteration steps for tighter estimate. Touches `train_model`. Subtle — the model is a CNN/NCA, vJp implementation needs care for spatial dims.

**Compute overhead**: 2x to 3x (one extra fwd-bwd per power-iter step).

**Risk**:
- Forcing spectral radius < 1 turns the model into a contractive map, which means every trajectory eventually decays to a fixed point. On heat (which decays naturally), this is benign; on gs/ks (which preserve a chaotic attractor), forcing contractivity destroys the dynamics — the model can't represent the attractor. **This is a real risk.**
- Mitigation: penalize only when ||Jf v|| > tau with tau = 1.05 (allow mild expansion), and use a trajectory-averaged penalty (so the *Lyapunov* exponent is bounded, not the per-step Jacobian — cf. Pathak et al. 2018's reservoir computing approach).
- Cleaner mitigation: only apply this penalty to the *correction* head (NCA), not the frozen reservoir. The reservoir's Lyapunov is fixed by construction; the NCA's is what we're worried about. This gives a contractive correction on top of an expansive reservoir — exactly the regime that worked for echo-state networks (cf. §52 in findings.md).

### 2.7 Consistency / diffusion-forcing-style training

**1-line summary**: Train f to satisfy f(noise_eta(x_t)) ~ x_{t+1} for varying eta, enforcing robustness to small perturbations of the input.

**Math sketch** (deterministic version — no stochastic *output*):

  For each training pair (x_t, x_{t+1}), sample a perturbation magnitude
  eta ~ Uniform([0, eta_max]) and a unit direction d ~ Uniform(S^{n-1}).
  L = MSE(f(x_t + eta * d), x_{t+1}) + zeta * MSE(f(x_t + eta * d), f(x_t))

  The first term is the standard prediction loss with input noise (reduces to
  §61's noise-injection at zeta=0).
  The second term is the consistency loss — predictions should be insensitive
  to small input perturbations.

**Why it should help**:
- **Short (H=15)**: the consistency loss is the missing piece §61 didn't include. §61 found noise injection alone fails (absolute MSE worse). The consistency term *targets* the failure mode: §61's H=15 ratio improvement was a noise-floor artifact because step-1 MSE got worse. The consistency loss says "f's prediction at x + eta should match f's prediction at x, not just x_{t+1}" — directly preventing the step-1 floor from rising.
- **Mid / long**: same mechanism — robustness to perturbations is robustness to drift.

**Implementation**: 1 day. Touches `train_model`. We already have `train_noise_sigma` (§61); add `consistency_lambda`.

**Compute overhead**: 2x (need an extra forward on the clean input to compute f(x_t)).

**Risk**:
- This is essentially a Tikhonov-like regularizer (penalize ||Jf||^2 implicitly via finite-difference). Closely related to 2.6 but cheaper. Risk: sets up a tradeoff with step-1 accuracy that may be unfavorable on KS where chaos is severe.
- Different from §61: §61 only had the first term. The second term is the actual mechanism. Worth re-running §61's protocol with this added.

### 2.8 Perturbation-robustness via train-time MC-style spatial dropout

**1-line summary**: At training time only, apply spatial dropout (random per-pixel zeroing) to the input x_t — but evaluate without dropout at test time.

**Math sketch**:

  At train: x_t' = mask * x_t, where mask ~ Bernoulli(1 - p_drop) per pixel.
  L = MSE(f(x_t'), x_{t+1})
  At test: dropout disabled (standard).
  p_drop ∈ {0.05, 0.1, 0.2}.

  This is *deterministic at test time* — the user's "no stochastic output"
  constraint is satisfied. Dropout is purely a training-time augmentation.

**Why it should help**:
- **Short (H=15)**: similar to noise injection (§61) but discrete — masking a pixel is a more *local* perturbation than Gaussian noise everywhere. May avoid the spectral-radius confusion that §61 hit. More natural for the CML reservoir, which couples nearest neighbors — masking a pixel forces the model to interpolate from neighbors, which is exactly the rollout-resilience skill.
- **Mid / long**: indirect via same mechanism.

**Implementation**: 0.5 day. Add `spatial_dropout_p: float = 0.0` to `train_model`. Apply in the forward path before model call.

**Compute overhead**: 1.0x (negligible).

**Risk**:
- §61 already showed Gaussian input noise hurts on heat / ks. Spatial dropout might do the same — needs a small sanity sweep. The reason to think dropout could differ: dropout preserves the values of unmasked pixels exactly, so the CML reservoir's internal logistic-map regime is undisturbed (cf. §61's drive.clamp fix that was needed because Gaussian noise pushed CML drive out of [0,1]).
- Cheap experiment to run alongside others — a 1-day sanity sweep with low risk.

### 2.9 Trajectory-attention regularizer (anchor to past)

**1-line summary**: Penalize predictions that drift far from the recent observed trajectory.

**Math sketch**:

  Given a window x_{t-K..t} of past frames (already available for mamba_rand),
  define an anchor a = mean(x_{t-K..t}) and penalize:
  L = MSE(f(x_t), x_{t+1}) + xi * max(0, ||f(x_t) - a|| - rho)

  rho = train-time-estimated trajectory diameter.
  Hinge loss: only penalize when prediction lies further from anchor than typical.

**Why it should help**:
- **Short (H=15)**: doesn't help much on its own — single-step error from anchor is small.
- **Mid / long**: Free-run drift typically pulls the trajectory off the data submanifold. Anchoring to recent past pulls it back. This is essentially a Bayesian prior of "trajectory is locally smooth in state space" — true for heat, partially true for gs, somewhat true for KS.

**Implementation**: 1-1.5 days. Only natural for mamba_rand (which already takes K=4 past frames). For rens (single-frame input), would require expanding input to a window — bigger architectural change.

**Compute overhead**: 1.05x (one extra MSE term).

**Risk**:
- KS dynamics genuinely move quickly through state space; anchoring could degrade step-1. Need rho calibrated per benchmark.
- Couples poorly with the rens path. Mamba-only.

### 2.10 Curriculum: easy-then-hard benchmarks (cross-substrate generalization)

**1-line summary**: Pre-train on diffusive (heat-like) dynamics, fine-tune on chaotic (gs/ks) dynamics — within the same architecture.

**Math sketch**:

  Stage 1 (epochs 0..N1): train on heat-style smooth trajectories.
  Stage 2 (epochs N1..N): train on gs/ks/Crafter trajectories.
  All else equal.

**Why it should help**:
- **Short / mid / long**: smooth dynamics force the model to learn a non-trivial dynamics map (not just a copy). Chaotic dynamics then *adjust* that dynamics map without having to discover the existence of dynamics from scratch. Plausibly reduces the basin of pathological solutions (e.g., the `out_proj` zero-init artifact in §62 where the temporal block never unfroze).

**Implementation**: 1 day. New benchmark sequencing in the training script; no model changes.

**Compute overhead**: 1.0x (same total epochs, just split).

**Risk**:
- The kind of failure §62 documented (zero-init suppressing the temporal feature) might also happen here — the heat-pretrained model may settle on a near-identity map that's hard to perturb on gs. Mitigation: random-init a small "delta" head that's added on top of the heat-pretrained model in stage 2.
- Doesn't address the core rollout-stability issue directly; complements rather than replaces 2.1-2.7.

### 2.11 Imagination consistency (self-consistency over rollout)

**1-line summary**: Roll the model H steps, then re-input one of the intermediate predictions as a "starting point" and check the rollout matches.

**Math sketch**:

  pred_traj = [f(x_0), f^2(x_0), ..., f^H(x_0)]
  reroll = [f(pred_traj[k]), f^2(pred_traj[k]), ..., f^{H-k}(pred_traj[k])]

  L_consistency = Sum_{k=1..H/2} MSE(reroll, pred_traj[k+1:])

  This is independent of ground truth — purely a self-consistency constraint on
  the dynamics map.

**Why it should help**:
- **Short / mid / long**: enforces that f is a true Markov dynamics map (the prediction at step k from x_0 should equal the prediction at step k from f^j(x_0) for any j), which is automatically satisfied for any deterministic f but is *not* required by the per-step MSE loss. The constraint disambiguates among many models that all fit step-1 MSE equally well, picking ones that compose cleanly under iteration.

**Implementation**: 2 days. Touches `train_model`. Need extra rollout step plus consistency MSE.

**Compute overhead**: 3x to 4x.

**Risk**:
- Self-consistency is automatically satisfied if f is run deterministically without stochasticity. So this only buys something if combined with 2.7 (consistency loss with input perturbation) — then it becomes "consistent under perturbation," which is meaningful. Standalone it's redundant.

### 2.12 Adaptive loss-shape: rollout-loss-aware sample weighting

**1-line summary**: Detect which training samples produce models with high H=15 rollout MSE, upweight those samples.

**Math sketch**:

  Periodically (every few epochs), do a quick H=15 rollout from each training
  sample x_0_i, compute per-sample rollout MSE r_i.
  Sample weight w_i = softmax(beta * r_i) — upweight hard-to-stabilize samples.
  Standard MSE loss continues, just weighted.

**Why it should help**:
- **Short (H=15)**: directly. The samples that destabilize rollouts are precisely the ones where the dynamics tangent escapes the training manifold; upweighting them puts gradient signal on them.
- **Mid / long**: if H=15 is fixed during reweighting, only mid horizons benefit secondarily.

**Implementation**: 1.5 days. Adds a periodic rollout pass (cheap — eval-only) and sample weighting in the loss.

**Compute overhead**: 1.3x (extra rollout eval every few epochs).

**Risk**:
- If the rollout MSE is dominated by chaotic Lyapunov amplification (which it is on KS), reweighting can't fix that — the model legitimately can't predict KS at H=15 in absolute terms. Reweighting will then push gradient on all samples uniformly. So expect to need this combined with 2.1 / 2.7 — it's a meta-strategy, not a standalone fix.

---

## 3. Top-3 picks for the 1-week sprint

Pre-registration: each pick states the architecture, the experimental scope, and the success criterion. Use §59 / §60 / §61 protocols for the rollout probe.

### Pick 1 (CHEAP, ~1.5 days): Consistency loss + scheduled-sampling H=4 (combination of 2.2 + 2.7)

- **Architecture**: rescor_rens K=32 (primary) + rescor_mamba_rand (companion at full data, Task #33 baseline).
- **Why**: §61's noise-injection failure was specifically attributed to "absolute step-1 MSE got worse, so the H=15 ratio improvement was a noise-floor artifact." The consistency loss (2.7) directly fixes this — penalize *prediction inconsistency* under perturbation, not just teacher-forced fit. Scheduled sampling at H=4 (very mild — only 4 unrolls) adds free-run distribution coverage in late training without huge compute cost.
- **Effort**: 1.5 days. Add `consistency_lambda` and `scheduled_sampling` kwargs to `train_model`. Reuses the §61 noise-injection scaffolding.
- **Compute**: ~3x training. With 30-45h base for the 90-cell sweep → ~120h. CPU-feasible if we cut to 3 seeds × 3 benchmarks × 2 variants = 18 cells.
- **Success criterion**: H=15 absolute MSE on gs ≤ σ=0.0 baseline on at least 2 of 3 seeds, AND H=15 ratio < 10x on gs (vs §59's 17.17x).
- **Risk**: Lowest of the three. If it fails, we learn something specific about the noise-floor mechanism rather than ruling out a class of approaches.

### Pick 2 (MID, ~3 days): TBPTT-H rollout MSE with H=8, K_bptt=4 (idea 2.1)

- **Architecture**: rescor_mamba_rand (primary). Skip rens for this pick — rens has no temporal context, so TBPTT is rolling a stateless function and the only memory dependence is via the predicted output sequence. Mamba's K=4 SSM context is the natural match.
- **Why**: Most directly attacks the failure mode. Full free-run training over 8 steps with truncated backprop trains the temporal component on its own predictions. §62's mamba_rand result (17.79x H=15 ratio at sanity scope) suggests there's headroom — the Mamba block is doing *something* useful at K=4 already; TBPTT-H=8 should compound that.
- **Effort**: 2-3 days. Touches `train_model`, `ResCorMamba.forward` may need a "stateful" mode for SSM state passthrough. Adaptive loss weighting w_k = 1 / (running_mean(L_k)) is load-bearing — implement carefully.
- **Compute**: 5-8x training. The full Task #33 protocol (200 trajs × 100 ep) at 8x → 800h sequential. Need GPU. Scope: 3 seeds × 2 benchmarks (gs, ks) × 1 variant = 6 cells, ~3.5 days wall on a single GPU.
- **Success criterion**: H=15 ratio on gs < 8x (cuts the §62 mamba_rand 17.79x by another 2x), AND step-1 MSE on gs within 1.5x of rens K=32 (closes the §62 step-1 gap).
- **Risk**: Gradient explosion. Need clipping at max_norm=1.0 and possibly Huber loss on the long-horizon component. If clipping is too aggressive, training is slow; if too loose, NaN by epoch 5 on KS.

### Pick 3 (AMBITIOUS, ~1 week): Lyapunov-spectral penalty on the NCA correction only (idea 2.6, restricted form)

- **Architecture**: rescor_rens K=32 (primary). Mamba_rand has more components → more places for the Jacobian penalty to interact unexpectedly; the rens NCA is the simplest target.
- **Why**: Most theoretically grounded. Only attempt if Picks 1 and 2 don't get H=15 ratio < 5x — that's the regime where the cheaper levers haven't worked and we need the principled fix. Restricting the spectral penalty to the NCA correction (not the frozen reservoir) avoids the "destroy the chaotic attractor" failure mode. If this works it's the publishable headline: "spectral-radius-bounded correction on top of an expansive reservoir achieves H=15 stability on Crafter latents at 321 trained params."
- **Effort**: 4-6 days. New file: `wmca/spectral_reg.py` with vJp + power-iteration. Touches `train_model` (new kwarg + integration). Subtle: the NCA is a small CNN with spatial dims — vJp implementation has to handle the convolution structure correctly. Expect 1 day debugging.
- **Compute**: 2.5x training. Power-iteration adds 2 fwd-bwds per training step. Single-GPU feasible at 3 seeds × 3 benchmarks × 1 variant.
- **Success criterion**: H=15 ratio on gs < 3x AND on Crafter latents < 4x AND step-1 MSE not more than 1.5x worse than rens baseline.
- **Risk**: vJp implementation bugs (silent wrong-direction penalty). Spectral radius < 1 forced too tightly = contractive collapse (model decays to a constant). Mitigation: tau = 1.05 (allow mild expansion), monitor Jacobian estimate during training, abort if power-iter estimate < 0.7 for >5 epochs (collapse signature).

---

## 4. Open questions / unknowns

1. **Substrate priority**: Should the sprint focus on Crafter latents (the substrate that actually matters for the kept-posterior fork, per §60) or synthetic gs/ks (faster iteration, simpler sanity)? Current default: synthetic for the first ablation, Crafter for the winning approach. Confirm.

2. **GPU availability**: TBPTT-H=8 (Pick 2) realistically needs a GPU for the 200-traj × 100-ep × 8-rollout protocol. CPU-only scope would be 50 trajs × 100 ep, which is the §62 sanity scope — borderline reproducible at that small scale. User input: GPU available for the sprint?

3. **Connection to kept-posterior fork**: §60 retired the M2 stability gate. Is the "training-stability" framing still useful given the posterior corrects each step at imagination time? The argument for *yes*: Dreamer's imagination loss is the thing the actor optimizes against, so a more stable deterministic core means cleaner imagination targets and faster actor convergence. But this is an indirect chain — the user may want a direct measurement of imagination quality (Crafter score) rather than the rollout-stability metric. Confirm scope.

4. **Action conditioning**: All M2 probes since §60 are action-conditioned (actions teacher-forced from test traj). Should the rollout-loss training in 2.1 / 2.2 also feed teacher-forced actions during the H-step unroll, or generate actions stochastically? Current default: teacher-forced actions, free-run states (consistent with §60's evaluation protocol).

5. **Loss scaling on the chaotic cells**: KS has step-1 MSE ~1e-7; gs ~1e-6; Crafter latents ~1e-3. A unified rollout-loss objective at H=8 gets a 4-5 orders-of-magnitude scale spread across substrates, which the optimizer Adam handles via per-param adaptive scaling but the per-batch loss landscape doesn't. Should we normalize per-substrate before summing in multi-benchmark training? Or keep substrate-isolated training? Default: substrate-isolated (matches §59-§61 protocol).

6. **Combination strategies**: Picks 1+2+3 are not mutually exclusive. A logical end state would be "rens K=32 with consistency loss + spectral penalty + scheduled sampling" — does the sprint scope include trying combinations, or only the three picks individually first? Default: individuals first, save combinations for a follow-up sprint after the per-idea ablation results are in.
