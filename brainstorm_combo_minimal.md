# Drift-Gated × Multistep — Simplest Integration Brainstorm

Date: 2026-04-29 (Sprint Day 2-3)
Scope: combine the Day 4-5 architectural fix (`ResCorMambaGated`) with the Day 2-3 training-time fix (multistep penalty) using the **smallest possible code path**. Research-mode brainstorm; no implementation here.

Both stabilizers are orthogonal:
- `rescor_mamba_gated_rand` — architectural; trainable scalars `gate_scale`, `gate_bias` + per-cell sigmoid drift gate on the NCA correction. Already wired into the registry.
- `multistep_horizon=H, multistep_bptt=K` — loss/training; `train_model` already supports it. Mutually exclusive with pushforward.

The trivial combination is one ablation script that passes both kwargs. No new architecture, no new loss code.

---

## 1. Trivial combination spec

**One script**, `experiments/drift_gated_multistep_ablation.py`, copy-paste of `drift_gated_ablation.py` with the multistep kwargs from `multistep_ablation.py` spliced into the `train_model` call.

Ablation cell matrix (deliberately minimal):

| Axis | Values | Count |
|---|---|---|
| Model | `rescor_mamba_gated_rand` | 1 |
| `multistep_horizon` | `8` (with `multistep_bptt=4`) | 1 |
| Seeds | `42, 43, 44` | 3 |
| Benchmarks | `heat`, `gs`, `ks` | 3 |
| Epochs | `100` | — |
| Grid / N_steps / N_traj | `16 / 105 / 200` (sprint defaults) | — |

**Total: 9 training runs.** Compute budget identical to the gated-only ablation (`drift_gated_ablation.py`) — multistep BPTT through 4 of 8 rolled steps roughly doubles per-batch FLOPs, so estimate ~2× wallclock, ~30-60 min total at the RTX Pro 6000 operating point.

Reported metrics (per cell):
- 1-step MSE on test split (cheap, the registry's default eval).
- `gate_scale_init/final`, `gate_bias_init/final` — diagnostic for failure-mode 1.
- Optional: H=15 absolute MSE rollout via the existing rollout probe (post-hoc on saved ckpts).

Ablation script outputs:
- `experiments/results/drift_gated_multistep_ablation.json`
- `experiments/results/drift_gated_multistep_ckpts/{bench}_seed{seed}.pt`

No new `model_registry` entries, no new modules — only kwarg plumbing.

---

## 2. Failure-mode analysis

Five hypotheses for why the trivial combination might evaporate the gains:

1. **Gate collapse to closed (gate ≈ 0).** Multistep training intentionally trades 1-step accuracy for rollout shape, so step-1 predictions sit far from `cml_mean`. The drift signal `||x_t − cml_mean.detach()||` is now huge on every training step, which pushes `gate_bias` down (or `gate_scale` up the wrong way) until `gate ≈ 0`. The model collapses to pure rens K=32 and the multistep gains are lost.

2. **Drift-signal poisoning / non-stationarity.** Even if the gate doesn't fully collapse, the *meaning* of the drift signal changes during multistep training. Early multistep steps (no_grad rollout) push the input distribution off-manifold; later BPTT steps differ qualitatively. The gate sees a moving target and ends up gating on noise instead of true off-manifold signal.

3. **Gradient interaction at the gate parameters.** The gate scalars are scalar-valued, sit on the H_train=8-step rollout, and receive gradient contributions from every BPTT step. Small scalars + long credit-assignment chain + chaotic dynamics (gs, ks) → very high-variance updates. We've already seen the multistep BPTT path produce 100×-poisoned 1-step MSE; the gate scalars may oscillate hard.

4. **Compounding NaN / bf16 instability.** The bf16 multistep stack is already on the edge (the multistep ablation has a `--no-bf16` fallback flag). Adding the gate's `(diff**2).mean().clamp_min(0).sqrt()` op during a bf16-autocast multistep BPTT could produce NaN gradients on the chaotic benchmarks (gs especially).

5. **Eval-mismatch artifact.** 1-step MSE under multistep training is already a poisoned metric. We can't tell from 1-step MSE alone whether the gate helped — we need H=15 / H=100 rollouts. If we ship the script without a rollout sweep, we may declare "no win" (high 1-step MSE) when in fact the rollout shape is fine.

---

## 3. Drop-in mitigations

For each failure mode, the smallest fix that doesn't require a new module:

1. **Gate collapse → bias init bump + optional gate-scale freeze.** Pass `gate_bias_init=1.0` (instead of default `0.5`) when constructing `rescor_mamba_gated_rand` for the multistep run. This sets initial gate at drift=0 to `sigmoid(1.0) ≈ 0.73` — more open by default. If gate_bias drifts down to ~0 mid-training, additionally freeze `gate_scale` (`m.gate_scale.requires_grad_(False)`) to stop the gate from sharpening. **Cost: 2 lines.**

2. **Drift-signal poisoning → re-anchored drift.** Replace the drift signal with `||x_t − pure_rens_predict(x_t)||` where `pure_rens_predict` runs the rens path with its own NCA correction (i.e. the fully-trained rescor_rens output). This requires a small change to `compute_gate` — accept a drift override tensor — but the simpler workaround is to **just use the unchanged drift formula and accept the bias**, since `cml_mean` already serves as a stable reference (its gradient is detached). Skip this mitigation in the first run unless mitigation 1 fails.

3. **Gradient interaction → detach gate from multistep loss.** Wrap the gate parameters under a `multistep_only_no_grad` scope: train the Mamba/NCA weights with H=8 multistep, but compute the gate scalars only on the H=1 step's loss path. Cleanest expression: keep `gate_scale.requires_grad=True` only inside an explicit `_train_step_h1()` pass; freeze them during the H>1 BPTT pass. **Cost: ~10 lines in `train_model`, only if mitigation 1 fails.**

4. **bf16 instability → fall back to fp32 with `--no-bf16`.** Already supported by `multistep_ablation.py`. Ship the combo script with the same flag and default to `bf16=True`; if NaNs surface, rerun with `--no-bf16`. **Cost: 0 lines (reuse existing flag).**

5. **Eval mismatch → save ckpts; run H=15 rollout probe post-hoc.** The drift_gated and multistep ablations both already save per-cell checkpoints. Reuse the existing rollout probe (used for noise_inject + multistep follow-ups) on the combo's ckpts. Decision uses H=15 abs MSE, not 1-step MSE. **Cost: 0 lines (reuse rollout probe).**

---

## 4. Recommended variant for first run

**Run this single config first**:

```
model            = rescor_mamba_gated_rand
gate_bias_init   = 1.0          # mitigation #1
gate_scale_init  = 1.0          # default (don't freeze yet)
multistep_horizon = 8
multistep_bptt    = 4
multistep_weight_schedule = "uniform"
seeds  = [42, 43, 44]
benchs = ["heat", "gs", "ks"]
epochs = 100
batch_size = 128, lr = 1.4e-3, bf16 = True, compile = True
```

**Justification.** This is the trivial combination plus only mitigation #1 (a 1-line change: pass `gate_bias_init=1.0` to `create_model`). No code changes outside the new ablation script. Total ~9 cells, ~30-60 min on the GPU pod. Yields:

- Diagnostic on whether the gate collapses under multistep training (read off `gate_bias_final`).
- 1-step MSE on all three benches (sanity).
- Saved ckpts for the post-hoc H=15 rollout probe — the actual decision metric.

If gate doesn't collapse and H=15 gs is competitive: **strong combo signal**, advance to a wider sweep (H ∈ {4, 8} × gate_bias_init ∈ {0.5, 1.0}).

If gate collapses (`gate_bias_final < 0.1` on gs at all 3 seeds): apply mitigation #3 (detach gate from multistep loss) and rerun.

If 1-step MSE explodes / NaNs: rerun with `--no-bf16`.

---

## 5. Decision rule

Define the combo a **win** iff *all three* hold on the H=15 rollout probe (median over 3 seeds):

| Bench | Threshold | Justification |
|---|---|---|
| `heat` | ≤ 1e-4 | trivially solvable; if combo loses here, something is broken |
| `gs`   | ≤ 5e-4 | ~50× better than multistep-alone gs (1-step ~2.5e-2 poisoned floor); ~5× worse than mamba_rand H=15 win baseline; this is the "stable rollouts beat step-1 chasing" sweet spot |
| `ks`   | ≤ 5e-4 | ks was already not a catastrophe zone; combo must not regress |

Define the combo a **partial win** iff `gs H=15 ≤ 1e-3` AND `gs H=100 ≤ 5e-3` (i.e. the H=100 catastrophe — 7× worse than rens at H=100 — is *eliminated*, even if H=15 doesn't beat the 5e-4 bar). This unblocks the all-horizon-stability sprint thesis even without the absolute win.

Define the combo a **loss** iff `gs H=15 > 5e-3` (worse than multistep-alone) OR `gate_bias_final < 0.1` on all 3 gs seeds (gate collapse).

Threshold rationale anchor: the project's existing mamba_rand multi-seed median at H=15 on gs is ~3.3e-5 (15× better than rens K=32). Asking the combo to be within 15× of that baseline (5e-4) while also fixing H=100 is the tightest "both worlds" claim that is plausibly achievable and clearly meaningful.

---

## 6. Open questions for user

- **Q1**: Is the post-hoc H=15 rollout probe acceptable as the decision metric, or do we want H=15 inside the ablation loop (adds ~5 min per cell)? Recommendation: post-hoc, since ckpts are saved.
- **Q2**: Should the smoke test be CPU (slow, no compile/bf16, like `drift_gated_ablation.py --smoke`) or GPU (fast, exercises the full bf16+compile path)? GPU smoke catches stack issues earlier; CPU smoke is what's currently wired into both parent scripts.
