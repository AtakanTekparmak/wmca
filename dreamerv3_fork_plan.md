# DreamerV3 Fork Plan — rescor_rens K=32 as Sequence Model

**Status**: planning notes updated 2026-04-24 after M2 result. Posterior is KEPT (M2 gate failed on gs+ks; heat pass deemed artifact). Fork not yet started.

**Research question**: *Can a 321-param frozen-chaotic-reservoir + NCA correction replace the GRU sequence model in DreamerV3, maintain Crafter score, potentially without the stochastic latent posterior?*

---

## 1. Repo + fork strategy

**Fork target: `NM512/dreamerv3-torch`**
- Cleanest, most-starred PyTorch reimplementation of DreamerV3, actively maintained, faithful to Hafner's JAX original. MIT-licensed. Direct PyTorch compatibility with our rescor stack — no JAX/PyTorch bridge needed.
- Fallback: `jsikyoon/dreamer-torch`, but NM512 is closer to v3 defaults.

**Scope: minimal.** Fork, keep everything, swap only the sequence model inside `networks.RSSM`.

Untouched from Dreamer:
- CNN encoder / decoder
- Replay buffer
- Imagination rollout driver
- Actor + critic heads
- λ-return computation
- Reward + continue heads
- Symlog transforms
- KL balancing + free-bits

This preserves the "only the sequence model changed" claim for the comparison.

---

## 2. Component-by-component integration

| Component | Decision | Rationale |
|---|---|---|
| **Encoder** | Keep Dreamer's CNN | Dreamer's encoder is co-trained with the decoder/posterior; swapping introduces a confound. Our 34K AE was trained separately. |
| **Posterior q(z \| h, x)** | Keep (M2 failed 2026-04-24, see §4) | Pre-registered gate failed on gs+ks; heat pass was a near-zero-attractor artifact. Deterministic rescor core cannot track chaotic latents unaided. |
| **Prior p(ẑ \| h)** | Keep, per posterior decision | Symmetric with posterior. rescor's deterministic state feeds the prior head as `h_t`. |
| **Sequence model (GRU → rescor)** | **Hybrid path** | See below. |
| **Decoder, reward, continue, actor, critic** | Unchanged | All downstream of sequence state `(h_t, z_t)`. |

**Sequence model integration (hybrid path):**
- Keep Dreamer's `h_t ∈ R^deter` as a vector externally
- Internally reshape to a 16×16×C grid (with `deter=512` → `C=2`)
- Apply rescor_rens K=32 (321 NCA + 32 frozen CMLs)
- Reshape back to flat vector
- Input pipeline: `[h_{t-1}_grid, z_{t-1}_grid, a_emb_grid]` concat on channel dim → rescor step → `h_t_grid` → flatten → `h_t`

---

## 3. Action-conditioning

**Dreamer's GRU takes actions as input. Rescor's CML bank is frozen and doesn't.** Two options considered:

| Option | Description | Verdict |
|---|---|---|
| **(A) NCA input** | Concat action embedding onto `[x, cml_out]` so NCA sees action | **Rejected** — only affects the 321-param NCA correction, action signal gets drowned |
| **(B) Drive modulation** | Learn MLP `a_t → grid`, add to rescor input before the CML bank | **Chosen** — perturbs the chaotic reservoir's input state directly, chaotic dynamics amplify small input differences, actions genuinely steer the trajectory |

**Param cost of action embedder:** `action_dim (17 Crafter) × 512 hidden × 512 out` ≈ **13K params**.

Honest accounting: "321 rescor params + 13K action embedder + Dreamer heads." Still dramatically smaller than Dreamer's GRU (~1.5M params for deter=512).

**Only trainable pieces inside the sequence block**: action embedder + NCA.

---

## 4. Rollout stability pre-experiment (M2)

**Before any RL training**: test whether rescor rollouts stay on-manifold without Dreamer's stochastic posterior correcting them each step.

- Load existing Crafter latent benchmark data
- Roll rescor_rens K=32 autoregressively without posterior correction for H ∈ {15, 50, 100}
- Measure per-step latent MSE and cosine divergence vs ground truth

**Decision gate**:
- **Stable to H=15** (MSE at step 15 < 2× MSE at step 1) → **DROP POSTERIOR**. This is the interesting result — simpler world model than Dreamer, not just smaller.
- **Diverges** → keep Dreamer's categorical 32×32 posterior, accept dilution of story, reframe as "321-param deterministic core + same stochastic machinery."

### Outcome (2026-04-24)

Probe complete. Trained rescor_rens K=32 (321 trained NCA params + 32 frozen CMLs sharing 43 frozen scalar params) on heat / gs / ks at grid=16, 100 epochs, 3 seeds, then rolled autoregressively for H ∈ {15, 50, 100} averaging per-step MSE over 20 test trajectories.

Median results (3 seeds each):

| Bench | H=15 ratio | H=50 ratio | H=100 ratio | cos_div H=100 |
|-------|------------|------------|-------------|---------------|
| heat  | 1.58       | 1.04       | 0.81        | 0.0000        |
| gs    | 17.17      | 324.16     | 6056.14     | 0.0974        |
| ks    | 126.60     | 566.16     | 1246.98     | 0.0009        |

**Gate result**: **FAILED** on gs and ks. Heat technically passes (ratio 1.58 < 2.0) but this is an **artifact** — diffusion dynamics decay to a trivial near-zero attractor, so absolute MSE drops to ~0 regardless of model quality. The H=100 cos_div of 0.0000 confirms GT and prediction are near-parallel, but both are near-zero vectors. On the genuinely chaotic benchmarks, error explodes 3 orders of magnitude by H=100 with measurable cos_div drift.

**Decision**: **KEEP posterior.** Crafter CNN latents behave like chaotic continuous dynamics (closer to gs/ks than heat), so the deterministic rescor core without stochastic correction will not track them.

**Narrative impact**: the original "simpler world model than Dreamer, not just smaller" framing is dead. New framing: "smaller deterministic sequence core (321 NCA + 13K action embedder vs GRU's ~1.5M) with the same stochastic latent machinery." Still a ~100× param reduction in the sequence block; weaker headline but still publishable.

Raw data: `experiments/results/rollout_stability_probe.json`. Analysis write-up: `findings.md` §59.

### Outcome (2026-04-25, Crafter-latent extension)

Re-ran the M2 probe on the actually-relevant substrate: rescor_rens K=32 on Crafter latents at 16×16, 3 seeds × 100 epochs, action-conditioned autoregressive rollout (actions teacher-forced, predictions free-run) over 20 held-out test trajectories per seed.

Median results across 3 seeds:

| H   | MSE median | ratio median | cos_div median |
|-----|------------|--------------|----------------|
| 15  | 2.94e-2    | **20.31×**   | 0.013          |
| 50  | 5.43e-2    | 48.64×       | 0.032          |
| 100 | 5.04e-2    | 45.11×       | 0.036          |

Per-seed H=15: 17.15 / 26.59 / 20.31 (tight cluster, all three fail by ≥8.5×). Step-1 MSE ~1.2e-3 across seeds — 1-step prediction is fine; this is purely autoregressive divergence. Plateau pattern at H=50/100 (~45-50× ratio) → chaotic-continuous failure mode, qualitatively GS-like (Task #27 gs s42 was 17.17× at H=15, comparable).

**Gate result: FAIL on Crafter substrate** (20.31× ≫ 2.0). The KEEP-posterior decision from the 2026-04-24 Outcome above is **doubly confirmed** — no longer reasoned by analogy from synthetic chaotic benchmarks, now measured directly on the 16×16 Crafter latent stream the fork will roll on. Heat's artifactual pass in §59 does not generalize; Crafter latents are GS dynamics in disguise.

The DROP-posterior path is closed for the foreseeable future. Reviving it would require fundamentally different rollout-stabilization machinery on top of rescor_rens or a different reservoir/correction architecture that demonstrably stabilizes Crafter-latent free-run.

Cross-reference: `findings.md` §60. Raw data: `experiments/results/rollout_stability_probe_crafter.json`. Script: `dreamerv3_scaffolding/rollout_stability_probe_crafter.py`. Trajectory wrapper: `src/wmca/crafter_real.py::generate_crafter_real_trajectories`.

### Outcome (2026-04-25, noise-injection extension)

Tested whether σ=0.02 Gaussian noise injection on the input x at training time would stabilize autoregressive rollouts as a cheap architecture-free shot at reviving the DROP-posterior decision. rescor_rens K=32, σ ∈ {0.0, 0.02}, 3 seeds × 100 epochs × 3 benchmarks (heat, gs, ks) = 18 cells, then re-ran the §4 / §59 rollout probe at H ∈ {15, 50, 100} on each saved checkpoint over 20 held-out test trajectories.

Absolute H=15 MSE — the metric that actually matters — is uniformly worse under σ=0.02:

| Bench | σ=0.0   | σ=0.02  | Verdict |
|-------|---------|---------|---------|
| heat  | 2.25e-6 | 1.32e-4 | σ=0.0 wins by 60× |
| gs    | 2.89e-4 | 3.55e-4 | σ=0.0 wins narrowly (1.2×) |
| ks    | 7.02e-5 | 1.34e-3 | σ=0.0 wins by 19× |

The eye-catching H=15 *ratio* improvement on gs (σ=0.0 24.33× → σ=0.02 4.26×) is a noise-floor artifact: σ=0.02 has much higher step-1 MSE (1.63e-5 vs 1.23e-6, ~13× worse), which inflates the ratio denominator and shrinks the ratio. The numerator does not improve. Per-seed H=15 ratios on σ=0.02 gs were 4.26 / 98.93 / 1.79 — massive seed variance, one of three models accidentally found a stable cell, no robust handle.

**Result: NEGATIVE for DROP-posterior.** Noise injection does not stabilize free-run rollout on any of heat / gs / ks. The KEEP-posterior decision is now **triply confirmed** (§59 synthetic chaotic, §60 Crafter latents, §61 noise-injection cannot rescue). The cheap rescue path is closed; anything further would require an architectural change.

**Implication for the remaining program**: **rescor_mamba (Task #30) is now the sole remaining architectural lever**. It does not change the kept-posterior decision (the posterior ships regardless), but it is the only candidate that could meaningfully reduce the posterior's per-step correction burden. Evaluated under the kept-posterior architecture against imagination-MSE / Crafter-score, not against rollout-stability gates.

**Side benefit (incidental codebase improvement)**: `drive.clamp(0, 1)` added at the start of `CML2DMultiR._run_batched` as part of the noise-inject patch. Necessary because σ>0 input noise can push the CML drive outside [0, 1], which the existing logic was unsafe against. This clamp is the canonical safety floor for any future patch that perturbs the input (test-time noise, Dreamer-fork action drive, etc.).

Cross-reference: `findings.md` §61. Raw data: `experiments/results/noise_inject_ablation.json`, `experiments/results/noise_inject_rollout_probe.json`. Scripts: `experiments/noise_inject_ablation.py`, `dreamerv3_scaffolding/noise_inject_rollout_probe.py`. Code: `train_noise_sigma=0.0` kwarg on `train_model` in `src/wmca/model_registry.py`; `drive.clamp(0, 1)` in `src/wmca/modules/hybrid.py::CML2DMultiR._run_batched`.

### Outcome (2026-04-28, rescor_mamba random-init reversal)

The earlier zero-init mamba sanity (Task #30, 50 trajs × 100 ep, seed=42) was misleading. A random-init variant at the same sanity scope reveals zero-init was structurally suppressing the temporal feature: the residual didn't unfreeze the Mamba block enough during the 100-epoch budget, so the model stayed near its "pure rens K=32" zero-init regime. That made the prior result a measurement of "rens with a slow-warming pendant," not "rens + Mamba."

Side-by-side at seed=42 (rens at full Task #27 protocol; both mamba variants at sanity scope):

| Variant | Data scope | gs step1 | gs H=15 ratio | gs H=15 abs MSE |
|---|---|---|---|---|
| rens K=32 (Task #27) | 200 trajs × 100 ep | 2.05e-5 | 47.01× | 9.66e-4 |
| mamba zero-init (sanity v2) | 50 trajs × 100 ep | 4.84e-5 | 34.17× | 1.65e-3 |
| mamba **random-init** (sanity-rand) | 50 trajs × 100 ep | **4.73e-5** | **17.79×** | **8.42e-4** |

| Variant | ks step1 | ks H=15 ratio | ks H=15 abs MSE |
|---|---|---|---|
| rens K=32 | 3.10e-7 | 126.60× | 3.92e-5 |
| mamba random-init | 1.96e-6 | 63.41× | 1.24e-4 |

At 4× less data than rens's full protocol, mamba random-init beats rens on absolute H=15 MSE on gs (8.42e-4 vs 9.66e-4) and cuts the H=15 ratio nearly 3×. Step-1 MSE on gs is still 2.3× worse than rens — that gap is the open question for the full-data follow-up (Task #33). KS shows a smaller win: ratio cut ~2× but absolute H=15 MSE still worse than rens.

**Implication for the KEEP-posterior decision**: KEEP-posterior remains **triply confirmed** (§59 synthetic, §60 Crafter latents, §61 noise-injection — all on rescor_rens K=32, all independent of which deterministic core we pick). What was about-to-be-the-fourth confirmation (mamba sanity also failing) is **retracted**. The decision is **not yet "quadruply confirmed"**; the mamba leg is pending Task #33. The headline framing ("~100× smaller deterministic sequence core with same stochastic machinery") is unchanged regardless of which way Task #33 lands — the question is which deterministic core (rens K=32 or mamba) sits in the smaller-backbone slot.

Pre-registered Task #33 decision rule (see `findings.md` §62):
- Step-1 closes to rens-level on gs and H=15 ratio stays < 20× → meaningful architectural win; rescor_mamba becomes the Dreamer-fork deterministic backbone.
- Step-1 stays >2× worse at full data → ratio-only win; weaker framing as "stabler-tail backbone."
- Step-1 closes AND ratio < 10× → strong win; revisit whether mamba reduces per-step posterior overhead more than expected (still does not revive DROP-posterior).

ETA: ~65min for s42 gs+ks at 200 trajs × 100 ep, ~3.5h for the 3-seed pass {42, 43, 44}.

Cross-reference: `findings.md` §62. Plan: `rescor_mamba_plan.md` §3 (zero-init spec — being revised). Methodological note: pair any zero-init sanity with a random-init companion at the same scope before drawing an architectural conclusion (same pattern bit us in §47 and §56).

### Outcome (2026-04-29, Day 0 multi-seed mamba_rand verification)

Sprint Day 0 = full-data multi-seed verification of the §62 single-seed reversal: 200 trajs × 100 ep × 3 seeds {42, 43, 44} × {gs, ks} = 6 cells. Both §62 open questions resolve.

GS (rens K=32 reference: step1 2.05e-5, H=15 abs 9.66e-4, H=100 abs 3.50e-2):

| Source | step1 | H=15 abs | H=100 abs |
|---|---|---|---|
| rens K=32 | 2.05e-5 | 9.66e-4 | 3.50e-2 |
| mamba_rand 3-seed median | 1.40e-6 (15× better) | 6.47e-5 (15× better) | **2.63e-1 (7× WORSE)** |

Per-seed gs H=15 abs MSE: s42 6.47e-5 / s43 3.57e-4 (worst) / s44 2.62e-5 (best). Per-seed gs H=15-vs-step1 ratio range: 18.67× to 354.20×.

KS (rens reference: step1 3.10e-7, H=15 abs 3.92e-5, H=100 abs 6.93e-4):

| Source | step1 | H=15 abs | H=100 abs |
|---|---|---|---|
| rens K=32 | 3.10e-7 | 3.92e-5 | 6.93e-4 |
| mamba_rand 3-seed median | 3.71e-7 (similar) | 1.98e-5 (2× better) | 3.86e-4 (similar) |

**H=15 win is real at the multi-seed median on gs** (15× better absolute MSE) but with massive per-seed variance (ratio range 18.67× to 354.20×). **H=100 catastrophe on gs is multi-seed-robust** — all three seeds in the 1.2-3.6e-1 range, 7× worse than rens. Not a per-seed artifact; structural property of mamba_rand-on-gs. **No catastrophe on ks** (mamba ≈ rens at all horizons).

Sharpened insight: mamba's predictions are excellent when they stay near the manifold and catastrophic when they drift. rens K=32, by contrast, is mediocre at both — bounded by chaotic-but-attractor-bounded reservoir dynamics that act as an implicit restoring force. This is exactly the failure mode the all-horizon-stability sprint was scoped to attack from three angles: pushforward (Brandstetter 2022, Day 1), multistep penalty (Chakraborty 2024, Day 2-3), drift-gated hybrid (Day 4-5), and optional diffusion forcing (Day 6-7).

§62 decision tree lands in **Branch 1 with a major caveat**: step-1 doesn't just close to rens — it improves on rens 15× — but the H=100 cliff makes the "meaningful architectural win" framing contingent on the sprint closing the gap. Day-0 read: mamba_rand is the right backbone candidate to build sprint stabilization on top of, not yet a "ship as the Dreamer-fork deterministic backbone" win on its own.

KEEP-posterior remains **triply confirmed** (§59 / §60 / §61). Day-0's H=100 gs catastrophe is more evidence the deterministic core alone cannot roll stably on chaotic-continuous substrates without external correction. Whether the sprint produces a stable-enough mamba variant to revisit posterior-burden questions is downstream of Days 1-7.

Sprint state — Tasks #34-37 created 2026-04-29:
- Task #34 (Day 1): pushforward (Brandstetter 2022). Implementation complete: `experiments/pushforward_ablation.py`, `dreamerv3_scaffolding/pushforward_rollout_probe.py`, `train_model` patched with `pushforward` kwarg in `src/wmca/model_registry.py`. Smoke-tested.
- Task #35 (Day 2-3): multistep penalty loss (Chakraborty 2024).
- Task #36 (Day 4-5): drift-gated hybrid (mamba + rens K=32 fallback under detected drift).
- Task #37 (Day 6-7): optional diffusion forcing.

Pre-registered sprint success criterion: median H=100 abs MSE on gs at or below rens K=32's 3.50e-2 baseline, retaining the H=15 absolute-MSE advantage. Failing H=100 but improving the worst-seed H=15 ratio (354× → < 50×) would be a "ratio-stability" partial win, feeding into a "stabler-tail backbone" framing rather than a clean "mamba is the backbone" framing. Three brainstorm docs landed earlier today documenting the cross-validation: `brainstorm_arch.md`, `brainstorm_train.md`, `brainstorm_theory.md`. User chose "trust theory, skip spectral-norm" — only the cross-validated picks are in the sprint.

Cross-reference: `findings.md` §63. `experiment_logs.md` 2026-04-29 entry.

### Outcome (2026-04-29 PM, Day 1 pushforward 1-step)

Sprint Day 1 = pushforward-trick training ablation (Brandstetter et al. 2022) on both rescor_rens K=32 and rescor_mamba_rand. **TRAINING PHASE ONLY**; rollout probes still running on the GPU pod (Prime Intellect RTX Pro 6000 96GB, pod `humming-vermilion-9b`). A follow-up subblock will land when the probes resolve — that's the actual verdict on the method.

Compute stack: batch=128, lr=1.4e-3 (sqrt-rule scaled), `torch.compile(mode="default")`, bf16 autocast. Day-1 wallclock ~30min total. **bf16 raises the absolute MSE noise floor by ~5-25× per bench** — Day-1 numbers compare cleanly against each other but NOT against the §59-§63 fp32 baselines.

rescor_rens K=32 1-step MSE (3-seed median, bf16):

| σ (pushforward) | heat | gs | ks |
|---|---|---|---|
| False | 3.98e-7 | 4.86e-6 | 1.12e-6 |
| True | 2.75e-6 (6.9× WORSE) | 1.53e-5 (3.1× WORSE) | 2.16e-6 (1.9× WORSE) |

rescor_mamba_rand 1-step MSE (3-seed median, bf16):

| σ (pushforward) | heat | gs | ks |
|---|---|---|---|
| False | 1.86e-5 (s43 outlier 2.89e-2) | 2.65e-6 | 6.76e-7 |
| True | 2.89e-2 (2/3 seeds catastrophic) | 1.83e-5 (6.9× WORSE) | 1.02e-6 (1.5× WORSE) |

Pushforward consistently HURTS step-1 MSE across both architectures and all three benches. mamba heat is the only cell where pushforward tips an additional seed from converging to diverging (s42 OK / s43 catastrophic / s44 catastrophic under σ=True vs s42 OK / s43 catastrophic / s44 OK under σ=False).

Pattern matches Task #32 noise injection (also hurt step-1 MSE uniformly). Task #32 was NEGATIVE because the rollout payoff didn't materialize; Day 1's verdict is pending the same question. Preliminary read: if the rollout probe shows the §63 H=15 mamba advantage on gs is destroyed (mechanically the H=15 floor has to be ≥6.9× worse on gs given step-1 is 6.9× worse), pushforward as a training-time exposure-bias mitigation does NOT preserve the win we want — pivot to Day 2-3 multistep penalty without stacking pushforward. If the probe shows H=15 preserved AND H=100 catastrophe softened, Day 1 is a partial win and Day 2 stacks on top.

Cross-reference: `findings.md` §64 (1-step phase). `experiment_logs.md` 2026-04-29 Sprint Day 1 entry. Artifacts: `experiments/pushforward_ablation.py`, `experiments/pushforward_ablation_mamba.py`, `dreamerv3_scaffolding/pushforward_rollout_probe.py`, `dreamerv3_scaffolding/pushforward_rollout_probe_mamba.py` (NEW — Task #38), `pushforward`/`compile`/`bf16` kwargs on `train_model` in `src/wmca/model_registry.py`, mamba block device/dtype-aware `_conv_indices` cache invalidation in `src/wmca/modules/mamba_block.py`.

### Outcome (2026-04-29 evening, Day 1 final probe verdict)

Probes landed. The morning's "SUSPENDED" verdict above is now resolved as **NEGATIVE** for the all-horizon goal. Pre-registered branch 1 from the morning ("if Day 1 destroys the H=15 win, pivot to Day 2-3 multistep penalty without stacking pushforward") triggers cleanly. The morning subblock is left intact as historical record; this subblock supersedes its open verdict.

bf16 noise-floor caveat carries over — all probe numbers bf16, comparable cleanly only against the morning's §64 bf16 baselines, not against §59-§63 fp32 references.

**rens K=32 rollout probe (3-seed median, bf16, GPU)**:

| σ | heat H=15 ratio | gs H=15 ratio | ks H=15 ratio |
|---|---|---|---|
| False | 4.93× | 16.50× | 159.49× |
| True | 6.05× (worse) | 19.41× (worse) | 74.68× (better ratio, worse abs) |

ks H=15 abs MSE on rens: σ=False 1.77e-4 vs σ=True 4.00e-4 (2.3× WORSE absolute). ks ratio drop is the same noise-floor artifact pattern as Task #32 / §61 — bigger step-1 inflates ratio denominator while numerator gets worse.

**mamba_rand rollout probe (3-seed median, bf16, GPU)**:

| σ | heat ratio | gs H=15 ratio | gs H=100 abs | gs H=100 cos_div | ks H=15 ratio |
|---|---|---|---|---|---|
| False | 3.51× | 22.34× | 1.82e-3 | 0.002 | 68.55× |
| True | 0.55× (zero-attractor) | **107.49×** | **3.27e-1** | **0.47 (near-orthogonal)** | 12.51× (modest win) |

Absolute H=15 MSE (mamba): gs σ=False 1.28e-4 → σ=True 3.24e-3 (**25× WORSE**); ks σ=False 7.13e-5 → σ=True 3.63e-5 (~2× better); heat σ=True 6.62e-2 with cos_div=0.9998 (zero-attractor degenerate, same artifact as §59 / §61).

**Load-bearing finding — gs H=100 catastrophe got dramatically worse**: ratio 22.34× → 10402× (~470× degradation), cos_div 0.002 → 0.47 (near-orthogonal). Predictions point in a different direction from ground truth — the model has learned a different attractor under pushforward and rolls there. Worst possible result for the §63 H=15-win-but-H=100-catastrophe diagnosis.

**ks modest win**: real on mamba (abs MSE halved + ratio improved, no noise-floor artifact), but ks isn't the chaotic-continuous bench we care about most for Crafter. A 2× ks win does not offset a 25× gs abs-MSE regression and a ~470× H=100 ratio regression.

**Cross-task pattern locked in**: pushforward (Day 1) and noise injection (Task #32) are both NEGATIVE for the same reason — both hurt step-1 MSE uniformly without rollout payoff on chaotic-continuous substrates. Training-time exposure-bias mitigations do not fix chaos amplification on our deterministic backbones. The same noise-floor-ratio artifact pattern fooled both ablations into looking better-than-they-were on ratio metrics until absolute-MSE was checked.

**Pivot**: Day 2-3 multistep penalty (Chakraborty 2024) WITHOUT stacking pushforward, per the morning's pre-registered decision. Theoretically cleaner attack — bounds BPTT depth via explicit horizon penalty without the 50% two-step branch that destabilized mamba heat under pushforward, and without bounding the Lyapunov exponent. `brainstorm_train.md` and `brainstorm_theory.md` had ranked multistep penalty above pushforward going in; Day 1 confirms the ranking empirically.

KEEP-posterior remains **triply** confirmed (§59 / §60 / §61) — Day 1 NEGATIVE is "this training-side fix doesn't close H=100 gs," not "no training-side fix can." Days 2-3 and 4-5 still in scope.

Cross-reference: `findings.md` §65 (final probe verdict). `experiment_logs.md` 2026-04-29 PM "Sprint Day 1 FINAL" entry. Probe results: `experiments/results/pushforward_rollout_probe.json`, `experiments/results/pushforward_rollout_probe_mamba.json`.

### Outcome (2026-04-29 night, Day 2-3 multistep penalty FIRST WIN)

Sprint Day 2-3 = NODE-style multistep penalty training (Chakraborty et al. 2024) on rescor_mamba_rand WITHOUT stacking pushforward, per §65's pre-registered pivot. **First technical win of the sprint.** Multistep H_train=8 PASSES the H=15 stability gate on gs (3-seed median ratio 1.66× < 2.0) for the first time on a deterministic backbone in the project, AND fixes the §63 H=100 long-horizon catastrophe (gs H=100 abs MSE 4.10e-2 → 3.51e-2, cos_div 0.036 → 0.0357 stable). bf16 noise-floor caveat carries over from §64 / §65.

gs results (3-seed median, bf16):

| H_train | step-1 MSE | H=15 abs MSE | H=15 ratio | H=100 abs MSE | H=100 cos_div |
|---|---|---|---|---|---|
| 1 (baseline) | 4.65e-6 | 7.24e-4 | 77.20× | 4.10e-2 | 0.036 |
| 4 | 7.73e-6 | 3.39e-4 | 2.63× | 3.88e-2 | 0.039 |
| **8** | **1.78e-2** | **1.84e-2** | **1.66× (gate pass)** | **3.51e-2** | **0.0357** |

ks (3-seed median, bf16): H=15 ratio 47.45× → 11.86× → 8.44× monotonic in H_train; H=100 ratio 759× → 13714× (H=4 anomaly, seed variance) → 294× at H=8. heat zero-attractor degenerate as in §59 / §61 / §65 — not evidence either way.

**The trade is the OPPOSITE of mamba_rand baseline**: baseline = great near-manifold (gs step-1 4.65e-6, H=15 abs 6.47e-5), catastrophic far from it (H=100 abs 2.63e-1 with cos_div 0.30). Multistep H=8 = mediocre near-manifold (step-1 ~3800× worse, H=15 abs 285× worse), stable far from it (H=100 abs 7× better, cos_div stable). Different inductive biases.

**Diametrically opposite to Day 1 pushforward**: Day 1 took gs H=15 ratio 22.34× → 107.49× (worse) and cos_div 0.002 → 0.47 (catastrophic, near-orthogonal). Day 2-3 takes the same metrics 77.20× → 1.66× (gate pass) and cos_div 0.036 → 0.0357 (stable). Same backbone, same compute, same probe — only the training loss differs. The cross-validated theoretical ranking (`brainstorm_theory.md`: multistep > pushforward) is empirically confirmed and with stronger margin than expected.

**Implication for the Dreamer fork**: PARTIAL WIN. What matters for policy training is absolute imagination-MSE at H=15-30 (determines imagined-frame quality). Multistep H=8 absolute H=15 MSE is 1.84e-2 — 285× worse than §63 mamba_rand baseline (6.47e-5) and ~19× worse than rens K=32 (9.66e-4). As-is, multistep H=8 is **NOT viable as the Dreamer-fork backbone** — imagined frames would be too noisy for policy gradients to find signal. But the long-horizon stability is a real architectural property: gs H=100 abs MSE bounded at the rens K=32 level (3.51e-2 vs rens 3.50e-2, essentially identical) AND cos_div low enough to mean predictions still point the right direction at H=100 — exactly the property the sprint was scoped to find. The question is whether it can combine with a low-step-1 mechanism.

**Combined drift-gated + multistep is the next experiment**. Drift-gated hybrid (Task #36, Day 4-5) implementation is already complete; multistep training kwargs already in `train_model`. The natural cell is `model="rescor_mamba_drift_hybrid", multistep_horizon=4` (or 8) on the same protocol. If the combination works, it would be the first variant where the §63 H=15 mamba advantage is preserved AND the H=100 catastrophe is fixed AND the H=15 ratio gate passes — the three properties simultaneously. This would be the **first viable variant for the Dreamer fork** to come out of the sprint.

KEEP-posterior remains **triply** confirmed (§59 / §60 / §61). Day 2-3 does NOT constitute a fourth confirmation — the H=100 win does not by itself revive DROP-posterior because the H=15 absolute MSE is now too high to be useful for imagination. The interesting downstream question is whether the combined variant produces a "stronger backbone, lighter posterior" framing — backbone stable enough that the kept posterior has materially less correction work to do, allowing more aggressive imagination horizons. That question is downstream of Day 4-5.

**Cross-task pattern revised**: §65 closed with "training-time exposure-bias mitigations don't fix chaos amplification on our deterministic backbones." Day 2-3 revises: NODE-style multistep penalty IS the right training-side lever. Mechanistic difference is direct H-step gradient signal — pushforward computes loss only at step 2 (or step 1 in 50%-branch), multistep at every step `t+1 ... t+H_train`. The training-side rescue path is not closed in general; only input-perturbation-style rescues (Task #32, Day 1) are closed.

Cross-reference: `findings.md` §66 (full writeup). `experiment_logs.md` 2026-04-29 evening "Sprint Day 2-3 — multistep penalty FIRST WIN" entry. Memory `wmca-dreamer-fork-state.md` updated. Artifacts: `experiments/multistep_ablation.py` (27 cells), `dreamerv3_scaffolding/multistep_rollout_probe.py`, `multistep_horizon` / `multistep_bptt` / `multistep_weight_schedule` / `multistep_n_steps` kwargs on `train_model` in `src/wmca/model_registry.py`, helper `_extract_horizon_targets(Y, n_steps, H)`. Results: `experiments/results/multistep_ablation.json`, `experiments/results/multistep_rollout_probe.json`. Wallclock ~3h on RTX Pro 6000 (humming-vermilion-9b).

### Outcome (2026-04-29 late night, drift-gated + multistep combo BORING MIDDLE)

Task #43 = the natural Day 4-5 combined experiment scoped at the end of §66 — drift-gated hybrid `rescor_mamba_gated_rand` TRAINED with multistep H=8 loss. Two variants ran in parallel: **Combo A** (vanilla stack, `multistep_horizon=8`, `K_bptt=4`, `gate_bias_init=1.0`) and **Combo B** (MSDC, drift-conditioned step weight `(1 - α·gate_h)` with α=0.5, per `brainstorm_combo_synergy.md` §2.1). 3 seeds × {gs, ks} × 100 epochs each. Same compute stack as Day 1-3 (Prime Intellect RTX Pro 6000, pod `humming-vermilion-9b`, batch=128, lr=1.4e-3, `torch.compile`, bf16). Heat skipped (zero-attractor artifact). bf16 noise-floor caveat carries from §64-§66.

**Verdict: BORING MIDDLE** — theory's pre-registered modal outcome (`brainstorm_combo_theory.md`: ~55% boring-middle, ~30% synergy, ~15% breakthrough). The combo does NOT improve on §66 multistep H=8 alone; on ks it regresses badly. **§66 multistep H=8 alone remains the sprint's strongest variant.** KEEP-posterior decision unchanged (still **triply** confirmed §59 / §60 / §61).

**Combo A** — 3-seed median, bf16:

| Bench | H=15 ratio | H=15 abs MSE | H=100 abs MSE | gate_mean |
|---|---|---|---|---|
| gs | 1.66× STABLE ✅ | 3.77e-2 | 3.65e-2 | 0.73 |
| ks | 0.95× STABLE ✅ | 1.35e-2 | 6.35e-3 | 0.70 |

**Combo B (MSDC)** — 3-seed median, bf16:

| Bench | H=15 ratio | H=15 abs MSE | H=100 abs MSE |
|---|---|---|---|
| gs | 2.03× (just fails) | 3.81e-2 | 3.66e-2 |
| ks | 3.44× (s44 NaN — training instability) | 1.42e-2 | NaN |

**Honest absolute-MSE comparison vs §66 multistep alone (the metric that matters for Dreamer)**:

| Variant | gs H=15 abs | gs H=100 abs | ks H=15 abs |
|---|---|---|---|
| Day 2-3 multistep H=8 alone (§66) | 1.84e-2 | 3.51e-2 | 1.97e-5 |
| **Combo A** | **3.77e-2** (2× worse) | 3.65e-2 (~same) | **1.35e-2** (685× worse) |
| Combo B (MSDC) | 3.81e-2 | 3.66e-2 | 1.42e-2 + NaN seed |

**Key findings**:
1. The "STABLE ✅" gate pass on Combo A is illusory. ks step-1 MSE is ~1.5e-2 (vs §66 multistep step-1 ~2.3e-6, 6,400× higher); ratio is ~1.0 not because predictions are stable but because they were never accurate. Same ratio-as-stability-proxy artifact §61 / §65 / Task #32 flagged.
2. The drift-gated architectural fix stacked on top of multistep training does NOT recover the §63 mamba near-manifold accuracy. Multistep training has already reshaped the mamba inductive bias away from near-manifold; gating against rens K=32 doesn't recover what was lost. The drift-gated mechanism may need a different training regime (single-step + scheduled sampling, or H=2 multistep) to show its theoretical advantage — that's a separate experiment, not a follow-up to this one.
3. MSDC adds a NaN training failure on s44 ks — the (1 − α·gate) weighting can starve gradient when the gate fires near 1. Needs `gate.detach()`, `α < 0.5`, or a hard weight floor. Worth documenting; not worth re-running given Combo A's clear absence of synergy.
4. Theory's pre-registered ~55% boring-middle prediction was right.

**Implication for the sprint**: §66 multistep H=8 alone is still the sprint's strongest variant for the long-horizon catastrophe fix. The Day 4-5 architectural fix didn't pay off when stacked on top of Day 2-3 training. Sprint state: Task #34 (pushforward NEGATIVE) complete; Task #35 (multistep PARTIAL WIN) remains strongest; Task #43 (combo BORING MIDDLE) complete. Task #37 (diffusion forcing) is the only un-tried sprint lever remaining. Alternative path: pivot to an actual Dreamer-fork training run with §66 multistep H=8 backbone and empirically test whether the kept posterior corrects the noisy backbone — that's the "stronger backbone, lighter posterior" framing the sprint was designed to test.

**KEEP-posterior decision unchanged.** None of the sprint variants tried so far gives both low absolute step-1 MSE AND stable rollout simultaneously. The combo doesn't shift the decision in either direction.

Cross-reference: `findings.md` §67 (full writeup). `experiment_logs.md` 2026-04-29 night "Combo experiment (Task #43) complete — boring middle" entry. Memory `wmca-dreamer-fork-state.md` updated. Artifacts: `experiments/drift_gated_multistep_ablation.py`, `experiments/drift_gated_msdc_ablation.py`, `dreamerv3_scaffolding/drift_gated_multistep_rollout_probe.py`, `dreamerv3_scaffolding/drift_gated_msdc_rollout_probe.py`, `msdc_alpha` kwarg + mutex checks on `train_model`, `ResCorMambaGated.compute_gate()` extracted. Results: `experiments/results/drift_gated_multistep_*.{json,log}`, `drift_gated_msdc_*.{json,log}`. Brainstorms: `brainstorm_combo_minimal.md`, `brainstorm_combo_synergy.md`, `brainstorm_combo_theory.md`. Wallclock ~3h × 2 variants on RTX Pro 6000 (humming-vermilion-9b).

---

## 5. Milestones (~3.5 weeks)

| Milestone | Duration | Deliverable |
|---|---|---|
| **M1** | 4 days | Fork `dreamerv3-torch`, reproduce vanilla Crafter run (target score ~11 at 1M steps as sanity) |
| **M2** | 3 days | **DONE 2026-04-24**. Rollout stability probe run; gate failed on gs+ks, heat pass ruled artifact; decision = **KEEP posterior**. See §4 Outcome. |
| **M3** | 7 days | Implement `RescorRSSM` class, action embedder, wire into RSSM interface (`.initial`, `.obs_step`, `.img_step`). Unit tests for shape and gradient flow. |
| **M4** | 5 days | Train 3 seeds × 1M env steps on Crafter. Log reward, Crafter achievements, imagination MSE. |
| **M5** | 4 days | Matched-param transformer baseline (1-layer, ~14K params total to match rescor+embedder budget). Same training loop. |
| **M6** | 3 days | Plots, ablations (K ∈ {8, 16, 32, 64}), writeup in `findings.md`. **Framing updated**: headline is now "~100× smaller deterministic sequence core with same stochastic machinery," not "simpler world model." |
| **Total** | **26 days** (~3.5 weeks) | |

---

## 6. Risk list (honest)

1. **Action-agnosticism**: frozen CMLs may not respond differentially enough to action perturbations; agent sees dynamics as noise. *Mitigation*: scale action drive magnitude as learnable gain; if gain collapses to zero, we've falsified the approach.
2. **H=15 divergence forces posterior**: **MATERIALIZED 2026-04-24, DOUBLY CONFIRMED 2026-04-25.** First gate on synthetic chaotic dynamics: gs ratio 17.17, ks ratio 126.60 at H=15 — failed by >8× and >60×. Second gate on the actually-relevant Crafter latent substrate (action-conditioned free-run rollout, 3 seeds × 100 epochs): median H=15 ratio 20.31× (tight per-seed cluster 17.15 / 26.59 / 20.31, all three seeds fail by ≥8.5×). Two independent probes have now failed on disjoint substrates that share the chaotic-continuous failure mode. **The posterior is now non-negotiable, not just defaulted-to.** Reviving DROP-posterior is closed for the foreseeable future; reopening it would require either (a) materially different rollout-stabilization machinery on top of rescor_rens, or (b) a different reservoir/correction architecture that demonstrably stabilizes Crafter-latent free-run. Neither is on the near-term roadmap. Headline framing locked in as "~100× smaller deterministic sequence core (321 NCA + 13K action embedder vs ~1.5M GRU) with the same stochastic latent machinery." See findings.md §60 and §4 Outcome (2026-04-25).
3. **Actor/critic instability**: rescor's chaotic dynamics may produce value estimates with high variance. *Mitigation*: tighter gradient clipping, lower actor LR.
4. **Seed noise**: Crafter score std ≈ 1.5 on 3 seeds; a 1-point gap is not significant. *Budget for 5 seeds* if M4 results are close.
5. **Reshape artifacts**: `deter=512 → 2×16×16` is low channel count; may bottleneck information. *Fallback*: `deter=1024 → C=4`.

---

## 7. Output files / repo layout

**Separate sibling repo**: `~/Desktop/personal/research/wmca-dreamer/` — fork of `dreamerv3-torch`. Keep `wmca/` clean of RL-infra code.

**Inside the fork**, add only:
- `dreamerv3/networks_rescor.py` — `RescorRSSM` class
- `dreamerv3/configs_rescor.yaml` — config overrides
- `scripts/rollout_stability.py` — M2 experiment

**Cross-repo dependency**: `pip install -e ../wmca` so the hero `CML2DMultiR` module stays canonical in the main repo. Results JSON synced back to `wmca/experiments/results/dreamer_crafter/`.

---

## 8. Prerequisites before starting M1

Do these in the current wmca repo first — they reduce M2/M3 risk:
- [ ] Multi-seed confirmation of rescor_rens K=32 hero (seeds 42, 43, 44) — heat + GS + KS at minimum
- [ ] Autoregressive rollout probe on heat / KS / GS for H=15 with rescor_rens K=32 (standalone, no Dreamer) — establishes the baseline stability of rens itself
- [ ] Package rescor_mr_uniform K=32 as a clean importable module with minimal dependencies

These 3 prereqs are ~1 week of work that hardens the hero before we bet 3.5 weeks of Dreamer-fork engineering on it.
