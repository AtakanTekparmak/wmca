# Noise-Injection Training Ablation for rescor_rens K=32 — Design Doc

**Status:** Plan only. No code changes until user approval.
**Owner:** atakantekerparmak@gmail.com
**Date:** 2026-04-24
**Linked tasks:** #27 (rollout-stability probe), DROP-posterior revival gate.

---

## 0. Motivation (recap)

Task #27's rollout-stability probe found that rescor_rens K=32 (our hero) diverges under autoregressive rollout on chaotic synthetic dynamics:
- **gs:** H=15 MSE ratio ≈ **17×** vs step-1.
- **ks:** H=15 MSE ratio ≈ **127×** vs step-1.

The decision gate (stable to H=15 iff median ratio < 2) fails on gs and ks → current stance is KEEP the Dreamer posterior.

**Hypothesis:** rescor_rens is trained on *clean* (X, Y) one-step pairs only. Small errors compound along an autoregressive rollout because the model never saw noisy inputs at train time. Gaussian noise injection on the input is a standard regularizer for dynamical-systems neural networks (Doya '92; used throughout ESN, NeuralODE, and RNN literature) — it trains the model to be locally Lipschitz around the data manifold, which suppresses compounding errors during rollout.

**Claim to test:** adding σ=0.02 Gaussian noise to model input x at *training time only* (clean at eval) meaningfully reduces the H=15 divergence ratio on gs and ks without changing the architecture.

This is the cheapest possible attempt at reviving the DROP-posterior decision.

---

## 1. Patch location — **Option A chosen**

### The two options

**(A) Plumb `train_noise_sigma` kwarg into `train_model`** in `src/wmca/model_registry.py`, apply `xb = xb + torch.randn_like(xb) * sigma` immediately before the model call inside the training loop.

**(B) Add `training_noise_sigma` attribute to `ResidualCorrectionWM.__init__` in `src/wmca/modules/hybrid.py`, apply inside `forward()` gated on `self.training`.

### Decision: **(A)** — training-loop kwarg on `train_model`.

### Justification

1. **Architecture-agnostic.** The probe follow-up may want to compare `rescor_rens` (K=32) vs other variants (`rescor_rens_stat_full`, `rescor_mamba` later). Putting noise on the *trainer* rather than the *model* means every variant gets the exact same treatment with one flag — no per-class plumbing.
2. **Model stays pure.** `ResidualCorrectionWM.forward()` already contains a `cml_channels > 1` branch that adds σ=0.01 noise to the *replicated CML input* (see `hybrid.py` lines 1363–1367). Stacking another noise source inside `forward()` would conflate two different noise semantics and muddy reproducibility of all prior rescor results. Keep `forward()` deterministic w.r.t. its input.
3. **Clean eval semantics.** Option A trivially matches the user's spec "clean at inference": the noise lives inside `train_model`'s per-batch loop, is never touched by `evaluate_model` / `evaluate_rollout` / the probe.
4. **Minimal surface area.** One kwarg, four lines of code, zero new tests required to trust the existing training path.
5. **Backwards compatible.** Default `train_noise_sigma=0.0` → identical to current behavior, bit-for-bit.

### Exact patch target

- **File:** `/Users/atakantekparmak/Desktop/personal/research/wmca/src/wmca/model_registry.py`
- **Function:** `train_model` (currently lines 411–577).
- **Insertion points:**
  - **Signature:** add `train_noise_sigma: float = 0.0,` right after `cml_reg_lambda: float = 0.1,` (current line 422).
  - **Loop body:** inside the `for i in range(0, len(perm), batch_size):` loop (current line 512), immediately after `xb, yb = X_tr[idx], Y_tr[idx]` (line 514) and *before* the `if is_cml_reg:` branch (line 516), add the noise injection.

---

## 2. Concrete code diff (text only — not applied)

```python
# src/wmca/model_registry.py

 def train_model(
     model: nn.Module,
     X_train,
     Y_train,
     X_val=None,
     Y_val=None,
     loss_type: str = "mse",
     epochs: int = 30,
     batch_size: int = 64,
     lr: float = 1e-3,
     device: str | torch.device = "cpu",
     cml_reg_lambda: float = 0.1,
+    train_noise_sigma: float = 0.0,
     # Extra kwargs accepted (and ignored) for runner convenience
     benchmark_name: str | None = None,
     model_name: str | None = None,
 ) -> nn.Module:
     """Generic training loop. Handles MSE, BCE, and cross-entropy losses.

     For CMLRegularizedNCA, adds the regularization term automatically.
+    If ``train_noise_sigma > 0``, Gaussian noise with that std is added to
+    each training batch input ``xb`` before the forward pass. Noise is
+    applied only during training — ``evaluate_model`` and ``evaluate_rollout``
+    see clean inputs. This is the standard dynamical-systems trick to
+    regularize autoregressive rollout stability.
     Returns the trained model (best val checkpoint restored).
     """
```

And inside the training loop:

```python
         for i in range(0, len(perm), batch_size):
             idx = perm[i : i + batch_size]
             xb, yb = X_tr[idx], Y_tr[idx]

+            # Noise injection on input x (training only).
+            # Intentionally applied to xb (the full input including any
+            # action channels for action-conditioned benchmarks) — matches
+            # the spec "perturb input x". Does not touch yb.
+            if train_noise_sigma > 0.0:
+                xb = xb + torch.randn_like(xb) * train_noise_sigma
+
             if is_cml_reg:
                 nca_out, cml_ref = model(xb)
```

That's the entire source change. **~8 lines total.** Nothing else in the codebase needs modification.

### Notes on the patch

- We do **not** clamp `xb` back into [0, 1]. σ=0.02 is small enough that this rarely matters; clamping would add a nonlinearity the spec doesn't authorize. If the rescor internal `cml_channels > 1` branch later clamps the CML input to `[1e-4, 1 - 1e-4]`, that's untouched.
- We apply noise **after** the random permutation batch slice, so the noise pattern differs every epoch and every batch — good for the regularization effect.
- `torch.randn_like(xb)` allocates on the same device as `xb`, which at this point is already on `dev` (line 480: `X_tr = _ensure_tensor(X_train, dev)`). No device-mismatch risk.
- The existing `CMLRegularizedNCA` regularization path (line 516 branch) sees the *noisy* `xb` — it both goes through `model(xb)` and indirectly through the internal CML reference. This is consistent: regularization toward the model's own CML readout on noisy input is still a valid target. Not our concern for this experiment (rescor_rens is not `CMLRegularizedNCA`), but worth flagging.

---

## 3. Experiment script design

### File
- **New:** `/Users/atakantekparmak/Desktop/personal/research/wmca/experiments/noise_inject_ablation.py`
- **Modeled on:** `experiments/phase1_honest_baseline.py` (same resume-from-JSON pattern, same `create_model` + `train_model` pipeline).

### Protocol

| Dimension | Values |
|---|---|
| model | `rescor_rens` (K=32) — fixed |
| noise σ | `{0.0, 0.02}` — 2 levels |
| seeds | `{42, 43, 44}` — 3 seeds |
| benchmarks | `{heat, gs, ks}` — same subset as `rollout_stability_probe.py` |
| epochs | 100 |
| grid_size | 16 |
| n_steps | 105 (matches probe) |
| n_trajectories | 200 (matches probe) |
| batch_size | 64 |
| lr | 1e-3 |

Total cells: **2 × 3 × 3 = 18.** Each cell trains one model and saves its checkpoint plus the 1-step test MSE. σ=0.0 cell is the *parity baseline*: it must reproduce the probe's results (modulo float noise) or we have a harness bug.

### Key structural differences from `phase1_honest_baseline.py`

1. **Hardcoded VARIANTS = ["rescor_rens"].** Outer loop is over `SIGMAS = [0.0, 0.02]`, not over `VARIANTS`.
2. **Dataset generation matches the probe**, not phase1:
   ```python
   data = gen_fn(grid_size=16, seed=seed, n_steps=105, n_trajectories=200)
   ```
   so that the probe follow-up can reuse the same test trajectories.
3. **Must save model checkpoints** (the probe follow-up needs them). Write state_dict alongside the JSON:
   ```
   experiments/results/noise_inject_ckpts/rens_K32_sigma<σ>_<bench>_seed<seed>.pt
   ```
4. **Pass `train_noise_sigma=sigma` to `train_model`.**

### Skeleton (pseudocode — full file written at implementation time)

```python
# experiments/noise_inject_ablation.py
SIGMAS = [0.0, 0.02]
SEEDS = [42, 43, 44]
BENCHMARKS = {"heat": generate_heat, "gs": generate_gray_scott, "ks": generate_ks}
K = 32
EPOCHS = 100
N_STEPS = 105
N_TRAJECTORIES = 200
GRID = 16

def run_one(sigma, bench_name, gen_fn, seed):
    data = gen_fn(grid_size=GRID, seed=seed, n_steps=N_STEPS,
                  n_trajectories=N_TRAJECTORIES)
    meta = data.meta
    m = create_model("rescor_rens",
                     in_channels=meta["in_channels"],
                     out_channels=meta["out_channels"],
                     grid_size=GRID, seed=seed, cml_K=K)
    m = train_model(m, data.X_train, data.Y_train,
                    X_val=data.X_val, Y_val=data.Y_val,
                    loss_type=meta["loss_type"],
                    epochs=EPOCHS, batch_size=64, lr=1e-3,
                    train_noise_sigma=sigma)   # <-- new kwarg
    mse_1step = evaluate_model(m, data.X_test, data.Y_test, meta)
    ckpt = Path(f"experiments/results/noise_inject_ckpts/"
                f"rens_K32_sigma{sigma}_{bench_name}_seed{seed}.pt")
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(m.state_dict(), ckpt)
    return {"score": mse_1step, "sigma": sigma, "bench": bench_name,
            "seed": seed, "ckpt": str(ckpt), "params": m.param_count()}

def main():
    out_path = Path("experiments/results/noise_inject_ablation.json")
    # resume-from-JSON pattern identical to phase1_honest_baseline.py
    # nested dict: results[sigma_str][bench_name][seed_str] = cell
    # incremental save after each cell
    # summary block at the end with per-(sigma, bench) 1-step medians
```

### Output JSON schema

```
experiments/results/noise_inject_ablation.json
{
  "protocol": {...},
  "per_cell": {
    "sigma=0.0":  {"heat": {"42": {...}, "43": {...}, "44": {...}}, "gs": {...}, "ks": {...}},
    "sigma=0.02": {...}
  },
  "summary_1step": {
    "sigma=0.0":  {"heat": {"median":..., "mean":..., "std":...}, ...},
    "sigma=0.02": {...}
  }
}
```

Checkpoints saved separately in `experiments/results/noise_inject_ckpts/`.

---

## 4. Probe follow-up

After training completes, we need to run the exact same autoregressive rollout protocol as `dreamerv3_scaffolding/rollout_stability_probe.py` on the noise-trained checkpoints and compare the divergence ratios.

### Options considered
- **(a)** Extend `rollout_stability_probe.py` with a `--variant-tag` / `--load-ckpt-pattern` CLI arg so it loads pre-trained weights instead of re-training.
- **(b)** Clone the probe into `dreamerv3_scaffolding/noise_inject_rollout_probe.py` with noise-specific paths hardcoded.

### Decision: **(b)** — clone the probe.

### Justification
- The original probe is a decision artifact for Task #27 and is referenced in `findings.md` / `dreamerv3_fork_plan.md`. Modifying its CLI risks invalidating the reproducibility of the H=15/17×/127× numbers it produced.
- The clone will be ~30 lines different (skip the `train_model` call, load checkpoints instead; iterate over σ ∈ {0.0, 0.02}; write to a different JSON). Not worth a refactor.
- Future-me can delete the clone once the DROP-posterior decision is settled.

### `noise_inject_rollout_probe.py` spec

- **Inputs:** expects `experiments/results/noise_inject_ckpts/rens_K32_sigma<σ>_<bench>_seed<seed>.pt` to exist for every cell from §3.
- **For each (σ, bench, seed):**
  1. Regenerate the same dataset (seed-matched `gen_fn(grid_size=16, seed=seed, n_steps=105, n_trajectories=200)`) to get identical `X_test, Y_test` trajectory layout.
  2. Rebuild the model skeleton: `create_model("rescor_rens", ..., cml_K=32, seed=seed)`.
  3. Load state dict from the checkpoint.
  4. Run the probe's rollout loop over `N_ROLLOUT_TRAJS=20` test trajectories; collect `mse_per_step[0..99]` and `cosine_div_per_step[0..99]`.
  5. Extract per-step MSE and ratio-vs-step-1 at H ∈ {15, 50, 100}.
- **Output:** `experiments/results/noise_inject_rollout_probe.json`, mirrors the probe's JSON schema but nested under `sigma` as the outer key:
  ```
  {
    "per_cell": {"sigma=0.0": {"heat": {...}, ...}, "sigma=0.02": {...}},
    "summary_median": {"sigma=0.0": {"heat": {"H=15": {"ratio_median":...},...}}, ...},
    "decision_gate": {"sigma=0.02": {"heat": {"stable_to_H15":...},...}},
    "protocol": {...}
  }
  ```
- **Report:** per (σ, bench) → median ratio at H=15, H=50, H=100 plus step-1 MSE parity check (σ=0.0 cells should match the original probe within 5%; if they don't, harness bug, fail loudly).

---

## 5. Success criterion

### Primary gate (decision-shaped)

Let `R_σ(bench, H)` = median across seeds of `mse(H) / mse(1)` for noise level σ on benchmark `bench`.

- **WIN** → if **`R_0.02(gs, 15) < 2`** AND **`R_0.02(ks, 15) < 2`**. This is the same gate as the original probe. If both chaotic benchmarks are stable under noise injection, we **revive DROP-posterior**. Heat is a secondary parity check (already stable at R ≈ 1).
- **PARTIAL WIN** → `R_0.02(gs, 15)` drops to <5× or `R_0.02(ks, 15)` drops to <10×, but not below 2. Noise helps but isn't enough. Note in `findings.md` and move on to `rescor_mamba` (original dreamer-fork plan). The partial-win numbers are still informative for the paper.
- **NO EFFECT** → `R_0.02` within ±20% of `R_0.0` on both gs and ks. Noise injection does not help rescor specifically. Document as a negative result; this is still a useful data point because it tells us rescor's instability is *not* a data-distribution issue but an architecture issue.

### Secondary sanity checks

1. **1-step MSE parity.** σ=0.02 should not degrade 1-step MSE by more than ~2× vs σ=0.0. If it does, we've hurt the model more than we've helped it.
2. **σ=0.0 parity with the original probe.** Our sigma=0.0 rollout ratios on gs/ks must reproduce the probe's ~17× / ~127× H=15 numbers within ±20%. If they don't, harness drift.
3. **Heat stability is preserved.** `R_0.02(heat, 15)` should remain < 2.

### Report artifacts

- `findings.md` entry with the before/after ratio table.
- A 3-benchmark × 2-σ plot of mse_per_step curves (log-y, H on x-axis) — drop into `experiments/results/plots/noise_inject.png` (nice-to-have, not a gate).

---

## 6. Compute estimate

### Training phase (§3)
- 2 σ × 3 benchmarks × 3 seeds = **18 training runs**.
- Phase 1 per-run time on this hardware (heat/gs/ks, 100 epochs, K=32): ~8–12 min each (from the rollout-stability probe log: ~600–700s per seed per bench).
- **Training wallclock:** 18 × 10 min ≈ **3 h** (serial), up to 5 h if gs/ks run slower than heat.

### Probe phase (§4)
- 18 checkpoints × (100-step rollout × 20 trajectories) = `18 × 2000 = 36k` forward passes. rescor_rens at grid 16, K=32 is ~30ms per forward on CPU, so ~18 min total. On GPU negligible.
- **Probe wallclock:** ~20 min.

### Total: **~3–5 h wallclock**, CPU-serial or modest GPU.

### Parallelism notes
- Each cell is fully independent, so if we want to shave time we can spawn the 18 training cells as separate processes (e.g., one per benchmark in parallel → 3× speedup, ~1 h). Not required; resume-from-JSON handles graceful restarts anyway.
- Safe to run this in parallel with other work since the probe follow-up only reads its own checkpoint directory.

---

## 7. Open questions (flag for user)

1. **Noise inside `use_sigmoid=True` boundary conditions?** σ=0.02 on an input in [0, 1] can push values slightly outside [0, 1] (≈5% of pixels will be). The model's internal logic (`cml_input.clamp(1e-4, 1-1e-4)`, `output.clamp(0, 1)`) is untouched but effectively sees a slightly-off-manifold input. This is *the point* of noise injection, but worth confirming: should we explicitly clamp `xb` back into [0, 1] after adding noise? **My recommendation: no** — the spec says "minimal perturbation" and clamping back would just create a weird truncated-Gaussian regime. But if user disagrees, one-line change.

2. **BCE / CE benchmarks in scope?** The probe tests heat/gs/ks (all MSE). Noise on binary targets like `gol` would be a different story (σ=0.02 on a {0,1} input is essentially a label-noise regime). We are following the probe exactly, so we stay on MSE benchmarks. Flag in case user wants to extend later.

3. **Do we want both σ=0.0 runs, or can we reuse the existing probe's 3 seeds directly?** The existing `rollout_stability_probe.json` already has σ=0.0 × 3 seeds × 3 benchmarks. Option: skip the 9 σ=0.0 training runs, reuse the existing probe's `mse_per_step`. **Tradeoff:** reusing saves ~1.5 h compute but means we have to trust the probe and our new code give identical results on σ=0.0, which we otherwise would have checked via the parity sanity check (§5 item 2). **My recommendation: run σ=0.0 fresh anyway** — the parity check is the only way to catch harness drift, and 1.5 h is cheap.

4. **Checkpoint storage cost?** rescor_rens K=32 has 321 trained + ~O(few k) frozen buffers. 18 checkpoints × ~50 KB = ~1 MB. Negligible, but let me know if you want checkpoints deleted after the probe consumes them.

5. **`train_noise_sigma=0.0` behavior guarantee.** I want to verify the patch is truly a no-op when σ=0. The current plan gates the noise on `if train_noise_sigma > 0.0:` so there is no `torch.randn_like` call in the σ=0 path — bit-for-bit identical to today's training. Good.

6. **Future variants.** If this experiment wins, we'll probably want to sweep σ ∈ {0.005, 0.01, 0.02, 0.05, 0.1} to find the optimum. Out of scope for this design doc; mention for follow-up planning.

---

## 8. Appendix — files touched / created

### Modified
- `/Users/atakantekparmak/Desktop/personal/research/wmca/src/wmca/model_registry.py` (+8 lines, signature + loop body of `train_model`).

### Created
- `/Users/atakantekparmak/Desktop/personal/research/wmca/experiments/noise_inject_ablation.py` (~150 lines, modeled on `phase1_honest_baseline.py`).
- `/Users/atakantekparmak/Desktop/personal/research/wmca/dreamerv3_scaffolding/noise_inject_rollout_probe.py` (~180 lines, clone of `rollout_stability_probe.py`).
- `/Users/atakantekparmak/Desktop/personal/research/wmca/experiments/results/noise_inject_ablation.json` (output).
- `/Users/atakantekparmak/Desktop/personal/research/wmca/experiments/results/noise_inject_rollout_probe.json` (output).
- `/Users/atakantekparmak/Desktop/personal/research/wmca/experiments/results/noise_inject_ckpts/*.pt` (18 checkpoints).

### Untouched (by design)
- `src/wmca/modules/hybrid.py` — model architectures stay clean.
- `dreamerv3_scaffolding/rollout_stability_probe.py` — original probe preserved for reproducibility.
- All other experiment scripts.
