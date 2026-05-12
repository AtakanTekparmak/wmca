# Deeper-NCA Plan (on top of rescor_rens K=32)

**Status**: planning notes, 2026-04-21. No experiments run yet.

**Context**: `rescor_rens K=32` (canonical name for `rescor_mr_uniform K=32` — 321 trained, no gate, beats oracle k5 on 4/6) is the current hero. `rescor_rens_deep` (L-stages of rens+NCA+residual) failed catastrophically at L=2 (heat 2000× worse, gol −18.67pp) — chaos re-injection destabilizes clean predictions. This plan is the follow-up: **keep the CML side at one rens_K=32 stage, deepen the learned NCA side**.

---

## Starting candidate set


| Label                 | Forward sketch                                                             | Params                        | Hypothesis                                                             |
| --------------------- | -------------------------------------------------------------------------- | ----------------------------- | ---------------------------------------------------------------------- |
| **A**                 | Conv(2,16,3,d=1) → Conv(16,16,3,d=1) → Conv(16,1,1)                        | ~2,641                        | 2-layer NCA, RF=5×5. Local motif parsing.                              |
| **B**                 | Conv(2,16,3,d=1) → Conv(16,16,3,d=2) → Conv(16,1,1)                        | ~2,641                        | Dilated NCA, RF=7×7 sparse. Multi-scale RF for Turing patterns.        |
| **C**                 | [drive, cml_mean, cml_var, cml_min, cml_max] → Conv(5,16,3) → Conv(16,1,1) | ~900–3,100 (depends on depth) | Ensemble-spread stat-bank — variance flags chaos-disagreement regions. |
| **D**                 | NCA applied M=3 times weight-tied on cml_out (no CML re-injection)         | 321                           | Iterative refinement, free param cost.                                 |
| **E**                 | Wider hidden (h=32 or 64, single layer)                                    | 641–1,281                     | Width without depth/RF.                                                |
| **F** (new, proposed) | Dual rens banks (short 3×3 + long 5×5 dilated kernels) + stat-bank NCA     | ~3,100                        | Two reservoir scales for GS's two-diffusion-rate structure.            |


---

## Proposer opus — key claims

1. **C is the biggest information-theoretic gain**: NCA currently sees only `cml_mean` across K=32 reservoirs; variance across r values is a free signal (≈ derivative w.r.t. chaos parameter ≈ local Lyapunov). Highest EV, cheapest.
2. **B (dilated)** targets GS's wavelength of ~6–8 px specifically but may regress gol (dilation skips the immediately-adjacent cells gol's 8-neighbor rule needs).
3. **D is safe** if and only if the iterated NCA is a contraction (‖∂NCA/∂y‖ < 1 at fixed point). rens_deep failed because each stage re-ran the chaotic logistic; D iterates only the correction MLP, no r∈[3.57,3.99] multiplier.
4. **GS gap is a physics mismatch, not a capacity gap.** rens is single-channel logistic; GS is two coupled channels with Du≈2Dv. A/B/C/D alone probably won't close it. Proposer's fix: **F (dual-reservoir)**.
5. Final rank: **C > F > A > D > B**. Drop E.

## Critic opus — key pushback

1. **A/B framing is unfalsifiable**: "longer-range correlations" without specifying *which* correlation or *where* it matters. On GS the characteristic length is ~5–10 cells so RF=5×5 is a direct test — but nothing ties the RF argument to specific benchmark physics for A/B.
2. **C is a relabeling of E3c** (prior `CML2DWithStats` experiments with multi-stat readouts). Claim of novelty is thin — the reservoir is different but the stat-bank trick itself has already been tried. If E3c underperformed on CML2D, the prior on C winning is weak.
3. **D is cope**: weight-tied iteration is strictly fewer degrees of freedom than untied deeper. Parameter efficiency isn't the bottleneck here — we have 321 params on a 321-floor, not 321 out of 10M.
4. **Overfitting floor**: 2641 params × 6 benchmarks × 30 epochs is trivially memorizable. Without **≥5-seed variance bars**, a "win" from A or B is indistinguishable from seed noise. Running A or B without multi-seed is ablation-budget waste.
5. **GS gap is fundamentally a reservoir-design problem** — single-channel logistic dynamics can't reconstruct a second channel that was never computed. Adding NCA depth on top of a single-channel reservoir doesn't change the reservoir's structure. Only fix: two-channel reservoir (C marginally, F properly).
6. **Run ONLY C, with variance-ablation** (train with and without `cml_var` in the input stat bank). Isolates the falsifiable signal: does ensemble spread carry task-relevant information beyond the mean?

---

## Synthesis

### Both agree

- **C is the safest, most mechanistically sound bet** (proposer #1, critic #6).
- **GS gap is likely a reservoir-design problem**, not an NCA-capacity problem (proposer #4, critic #5).
- **Kill D and E** outright (both agree). D offers nothing over A; E was already rejected.

### Disagreements

- Proposer wants A and B included in the sweep; critic says A/B are unfalsifiable without multi-seed and dominate the ablation budget.
- Proposer proposes F as a GS fix; critic agrees F-style reservoir redesign is the right direction but treats it as a separate initiative, not a deeper-NCA variant.

### Critic correctly flagged

- **C's E3c redundancy** — before running C, we must check `findings.md §§ ResidualCorrectionWMv3/E3c` for what the stat-bank trick looked like on CML2D and why it didn't win. If the construction is identical, we need a sharper mechanistic argument than "we're doing it on a different reservoir."

### Proposer correctly flagged

- **D's safety condition is mechanically distinct from rens_deep's failure mode** (no chaos multiplier in the loop). Even though D is unlikely to outperform A, the argument that "iterating the correction is safe where iterating the reservoir isn't" is a clean conceptual test. Low-cost, informative.

---

## Refined final plan (what we actually run)

### Phase 1 — single experiment, information-theoretic bet

**C (stat-bank NCA)**, with a **clean variance ablation**: two runs.

1. `C_full`: NCA sees `[drive, cml_mean, cml_var, cml_min, cml_max]`.
2. `C_no_var`: NCA sees `[drive, cml_mean, cml_min, cml_max]` (drop variance).

Both use the same single-layer architecture: `Conv(in_ch, 16, 3) → ReLU → Conv(16, out_ch, 1)`.

- If `C_full > C_no_var`: variance matters → ensemble spread carries task-relevant information beyond the mean → green light for more ensemble-statistics work.
- If `C_full ≈ C_no_var`: variance is noise → the K=32 averaging is already lossless at our scale → confirms the rens mean is the right projection.
- If both lose to vanilla rens: stat-bank construction is a dead end (C ≈ E3c all over again).

Param cost: ~900–1,200 trained per run. 2 runs × 6 benchmarks ≈ 1h.

### Phase 2 — only if Phase 1 moves numbers

**A (2-layer dense)** and/or **B (dilated)**, run **with 3 seeds** each for variance bars. Targets: gol (A), GS + pattern-benchmarks (B).
Param cost: ~2,641 × 3 seeds × 6 benchmarks × 2 variants = expensive. **Only run if C gives a clear signal** that capacity on the correction side matters.

### Phase 3 — separate track, GS-specific

**F (dual-reservoir stat-bank)** as its own initiative, NOT under the "deeper-NCA" umbrella. Fixing the GS gap requires a two-reservoir design with different coupling scales. This is a reservoir change, not an NCA change.

### Dropped

- **D (iterative weight-tied NCA)**: Critic correct — strictly dominated by A. The "safety analysis" argument is academically neat but adds no expected value.
- **E (wider hidden)**: Both agree. Width without depth/RF wastes capacity.

---

## Runnable checklist

- Verify C is not a bit-identical re-run of E3c (read findings §§ on ResidualCorrectionWMv2/v3/E3c)
- Build `CML2DMultiR` variant that exposes `[mean, var, min, max]` across K
- Write `experiments/stat_bank_rens_ablation.py` with `C_full` and `C_no_var` configurations
- Run Phase 1 on all 6 benchmarks (~1h)
- Evaluate: does variance matter? → branch to Phase 2 or Phase 3
- If Phase 1 wins, run multi-seed (3 seeds) on A and B — no single-seed "wins" accepted for these wide variants

---

## Hard constraints going forward

1. **No single-seed wins for variants with >1500 params** on this benchmark set. 3 seeds minimum. Critic's overfitting-floor argument stands.
2. **Always compare against `rescor_rens K=32` at 321 params** as the reference. Any variant that doesn't beat the 321-param baseline by a margin larger than 3-seed std is a negative result.
3. **GS gap closures via NCA changes should be greeted with skepticism**; a clean win on GS from A/B/C probably reflects data fit, not genuine inductive-bias improvement. Confirm with held-out seeds AND a physics-motivated sanity check (does the correction pattern look like the missing second-channel dynamics?).

