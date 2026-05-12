# Crafter-Latent Rollout Stability Probe — Design Doc

Status: proposal, no code changes yet. Extends Task #27 (synthetic heat/gs/ks
rollout probe) to Crafter-latent dynamics, the realistic environment whose
stability actually governs the "drop Dreamer posterior" decision.

Author: atakantekerparmak@gmail.com
Date: 2026-04-24

---

## 0. Motivation (one paragraph)

Task #27 shipped: rescor_rens K=32 is unstable on synthetic chaotic dynamics
(gs 17x MSE blowup at H=15, ks 127x; heat passes but via diffusive collapse).
Before turning to rescor_mamba, we need to know whether the *actual* target
distribution — encoded Crafter frames — is chaotic enough to break rescor.
If Crafter latents are milder than gs/ks, rescor may be stable at H=15 here,
which would revive the "drop Dreamer posterior" path without needing a new
architecture.

---

## 1. Data inspection (as of 2026-04-24)

Inspected `/Users/atakantekparmak/Desktop/personal/research/wmca/experiments/crafter_data/`:

```
actions.npy       (100000,)              int64
frames.npy        (100000, 3, 64, 64)    float32   (~4.6 GB)
next_frames.npy   (100000, 3, 64, 64)    float32   (~4.6 GB)
frame_encoder.pt  ~274 KB                (frozen AE checkpoint)
```

Data **exists** on disk. **No separate data collection is needed.** Size is
double what `generate_crafter_real` asks for by default (100K vs 50K).

### How are episodes delimited?

There is **no explicit flag** (no `done` array, no `-1` sentinel in actions;
actions are all in `[0, 16]`). The pairs stream
`(frames[i], actions[i], next_frames[i])` is produced by `_collect_frames` in
`/Users/atakantekparmak/Desktop/personal/research/wmca/src/wmca/crafter_real.py`,
which resets on `done` mid-collection and drops the terminal observation —
so the only signal is a discontinuity in the pair stream:
`frames[i+1] != next_frames[i]` iff a reset happened between step `i` and
step `i+1`.

I scanned all 100 000 pairs for this equality and found **602 episodes** with
the following statistics:

| stat          | value |
| ------------- | ----- |
| min length    | 34    |
| median length | 164   |
| mean length   | 166.1 |
| max length    | 318   |
| p10 / p25     | 126 / 147 |
| p75 / p90 / p95 / p99 | 188 / 215 / 240 / 276 |
| episodes >= 105 | 553 / 602 |
| episodes >= 50  | 587 / 602 |
| episodes >= 15  | 602 / 602 |

Restricted to the **test split** (last 15 % of rows, indices >= 85 000),
**91 episodes** are fully contained with:

| stat | value |
| ---- | ----- |
| min / median / max | 43 / 159 / 314 |
| >= 105 | 80 / 91 |
| >= 50  | 86 / 91 |
| >= 15  | 91 / 91 |

**Bottom line:** 80 trajectories of length >= 105 exist in the test split —
far above the `N_ROLLOUT_TRAJS = 20` the synthetic probe uses. H=100 is
comfortably reachable; no extra data collection needed.

---

## 2. Trajectory wrapper design

Add a new function **next to** (not replacing) `generate_crafter_real` in
`/Users/atakantekparmak/Desktop/personal/research/wmca/src/wmca/crafter_real.py`:

```
def generate_crafter_real_trajectories(
    grid_size: int = 16,
    n_frames: int = 100_000,
    seed: int = 42,
    device: str | torch.device = "cpu",
    data_dir: str = "experiments/crafter_data",
    encoder_path: str | None = None,
    min_traj_len: int = 105,
    max_test_trajectories: int | None = None,   # None = all that qualify
) -> BenchmarkData
```

### What it does

1. Load `frames.npy`, `actions.npy`, `next_frames.npy` (same load path as
   `generate_crafter_real`). **Do not re-collect** — raise a clear error if
   they are missing.
2. Detect episode boundaries with the `frames[i+1] != next_frames[i]` scan
   described above (chunked for memory; measured at ~a few seconds on this
   machine). Cache the boundary list in
   `experiments/crafter_data/episode_boundaries.npy` so repeat runs skip
   the scan.
3. Encode all `frames` and `next_frames` via `FrozenFrameEncoder` with the
   existing `_encode_batched` helper.
4. Build `X, Y` exactly as `generate_crafter_real` does (2-channel X with
   action field, 1-channel Y). Keep the existing 70/15/15 pairwise split for
   the train / val / test *pairs* — this preserves drop-in compatibility
   with `train_model`, so nothing changes for training-time losses.
5. Additionally, collect **test trajectories**. For each episode fully
   contained in the test region (start index >= `n_train + n_val`) with
   `len >= min_traj_len`, build a `TrajectorySpec`:

```
@dataclass
class TrajectorySpec:
    encoded_frames: torch.Tensor   # (T+1, 1, grid_size, grid_size) float32
    actions:        torch.Tensor   # (T,) int64 in [0, 16]
    episode_index:  int            # for debugging
```

   where `encoded_frames[0]` = `enc_frames[ep_start]`, and
   `encoded_frames[t+1]` = `enc_next_frames[ep_start + t]`.
   (Using `enc_next_frames` for the successors is cheaper and exactly correct
   — `next_frames[i]` is the environment's observation after action `i`, and
   it matches `frames[i+1]` within an episode.)

   Cap the count to `max_test_trajectories` (default: all that qualify) to
   control probe runtime.

6. Return a `BenchmarkData` whose `meta` dict carries an extra key
   `"test_trajectories": List[TrajectorySpec]`. This keeps the existing
   namedtuple shape (`X_train, Y_train, X_val, Y_val, X_test, Y_test, meta`)
   so `train_model` is unaffected; the probe reads trajectories from
   `data.meta["test_trajectories"]`.

### Why put the list in `meta` instead of extending the namedtuple

`BenchmarkData` is a `namedtuple` used in many places; widening it is a
breaking change with no benefit here. `meta` is already a free-form dict
passed through intact.

### Action encoding

The probe rollout must feed the **recorded** action at step `t`, not a
random one. `TrajectorySpec.actions` stores the raw int action, and the
probe converts it to the same action-field form `generate_crafter_real`
uses at training time — a 1-channel plane filled with `(a+1)/17`. This
guarantees the probe input distribution matches training.

---

## 3. Probe extension

**Decision: create a sibling file**
`/Users/atakantekparmak/Desktop/personal/research/wmca/dreamerv3_scaffolding/rollout_stability_probe_crafter.py`
rather than overload `rollout_stability_probe.py`.

**Justification:**
* Different rollout loop shape: synthetic probes pick a single seed frame
  per trajectory and roll forward via `model(x)`; the Crafter probe must
  re-construct the 2-channel input at every step by stacking the predicted
  latent with the *recorded* action-field for step `t`. That's not a
  drop-in replacement for the synthetic loop.
* Different data API (`data.meta["test_trajectories"]` vs the
  contiguous-pairs trick).
* Keeps Task #27 artifact (`rollout_stability_probe.py` and its JSON)
  untouched for provenance.
* Two scripts is cheaper than one parameterised one: a shared helper would
  have to abstract over (a) seed-frame selection, (b) input-reassembly,
  (c) channel counts, (d) BenchmarkData shape — diminishing returns.

### Sketch of the new probe's inner loop

Per test trajectory `traj` of length `T`:

```
x_frame = traj.encoded_frames[0]                 # (1, 1, 16, 16)
for s in range(max_h):
    if s >= T: break
    a = traj.actions[s].item()
    action_field = build_action_field(a, grid=16) # (1, 1, 16, 16)
    x = torch.cat([x_frame, action_field], dim=1) # (1, 2, 16, 16)
    pred = model(x.unsqueeze(0)).squeeze(0)       # (1, 1, 16, 16)
    gt   = traj.encoded_frames[s + 1]             # (1, 1, 16, 16)
    mse  = ((pred - gt) ** 2).mean().item()
    per_step_mse[s] += mse / n_use
    per_step_cos[s] += cosine_div(pred, gt) / n_use
    x_frame = pred       # NO clamp — encoded latents are not in [0, 1]
```

Important deltas vs the synthetic probe:

* **No `pred.clamp(0, 1)`.** Encoder latents are unbounded; clamping
  biases the rollout. (Synthetic probe clamps because heat/gs/ks are in
  `[0, 1]`.)
* **Action channel refreshed every step** from the recorded sequence.
* **Shorter rollouts for short episodes.** If a test trajectory has only
  `T < max_h` steps, contribute per-step MSE only for `s < T` and carry a
  per-step denominator (count of trajectories reaching that step) rather
  than dividing by a fixed `n_use`. This avoids mixing "trajectory ended"
  with "model diverged".

### Output JSON

`experiments/results/rollout_stability_probe_crafter.json`
Same shape as the synthetic probe's JSON: `per_cell[seed]`,
`summary_median`, `decision_gate`, `protocol`. Add a new protocol field
`n_trajectories_per_step: List[int]` recording how many trajs survived
long enough for each `s`.

---

## 4. Training setup

Identical to Task #27's synthetic probe protocol, except the benchmark:

| knob                 | value                      |
| -------------------- | -------------------------- |
| model                | `rescor_rens`              |
| K (CML count)        | 32                         |
| seeds                | `[42, 43, 44]` (3 seeds)   |
| epochs               | 100                        |
| batch_size           | 64                         |
| lr                   | 1e-3                       |
| grid_size            | 16                         |
| loss_type            | `mse` (from meta)          |
| in_channels          | 2 (from meta)              |
| out_channels         | 1 (from meta)              |
| horizons             | `[15, 50, 100]`            |
| n_rollout_trajectories | 20 (cap; 80+ available)  |

The benchmark's `meta` already provides `loss_type='mse'`, `in_channels=2`,
`out_channels=1`. No re-tuning, no hyperparameter sweep — this is a
direct re-run of the canonical rescor_rens K=32 recipe against a different
target distribution.

### Training-pair budget

100 000 pairs * 0.70 = 70 000 training pairs, already comparable to the
synthetic probe's `200 trajs * 105 steps = 21 000` pairs (in fact larger).
No cap needed; if wall-clock is a concern, truncate via `n_frames`.

### Compute estimate

Synthetic probe took ~X seconds per (bench, seed) cell (see
`experiments/results/rollout_stability_probe.log`). Crafter has ~3.3x
more training pairs, so budget ~3.3x per cell, times 3 seeds:
roughly an overnight run on the development machine, same order of
magnitude as Task #27.

---

## 5. Baselines / decision gate

Same primary gate as the synthetic probe:

> rescor_rens K=32 is **stable on Crafter-latent** iff
> `median_seed(MSE_15 / MSE_1) < 2.0`.

Report at H = 15, 50, 100:

* `mse_final`
* `mse_ratio_vs_step1` (primary)
* `cosine_div_final`

### Note on cos_div interpretation

Crafter encoded frames are action-conditioned, so even a *perfect* model
produces genuine per-step change in the latent — `cos_div` grows with `H`
even under the true dynamics, unlike heat/gs/ks where small `H`
ground-truth frames are nearly identical to `t=0`. Report cos_div for
continuity with Task #27 but **do not gate on it**; gate on the
`MSE_H / MSE_1` ratio.

### Decision matrix

| Outcome (median H=15 ratio) | Interpretation | Action |
| --------------------------- | -------------- | ------ |
| < 2.0                        | stable on the real target | DROP Dreamer posterior; resume rescor-only DreamerV3 fork |
| 2.0 .. 5.0                   | marginal; probably fine with a posterior as a safety net | KEEP posterior; de-prioritise rescor_mamba |
| > 5.0                        | unstable even on Crafter | KEEP posterior **and** promote rescor_mamba ablation |

(The Task #27 synthetic results were 17x / 127x, which would map to the
third row if they held on Crafter.)

---

## 6. Risk: episode-length distribution

**Measured, not feared:** 80 of 91 test-split episodes have length >= 105,
so H = 100 is fine. No contingency needed for 100K frames.

Nevertheless, the probe should be defensive:

1. `min_traj_len` defaults to 105 (to cover H=100). If fewer than
   `N_ROLLOUT_TRAJS` (20) trajectories qualify, the probe **drops back to
   H=50** automatically and logs a warning.
2. If fewer than 20 qualify at length 50 either, the probe drops to H=15
   and logs a stronger warning.
3. **Never concatenate across episodes.** Concatenation splices in an
   environment reset, which would make the model look unstable for the
   wrong reason.
4. If the user wants a longer horizon than the data supports, point them
   at `_collect_frames` with a larger `n_frames` (~30 min for another
   100K). We currently have no reason to do this.

---

## 7. Open questions

1. **Encoder reconstruction baseline.** We're measuring model MSE on
   encoded latents. The encoder itself has reconstruction error — should
   the probe also report `MSE(encode(x_{t+1}) - encode(decode(encode(x_{t+1}))))`
   as a floor, to give the ratio a reference scale? Non-blocking; can be
   a follow-up.
2. **Per-trajectory vs averaged curves.** The synthetic probe averages
   per-step MSE across trajectories. Crafter episodes vary wildly in
   content (early exploration vs. late survival). Should we also report
   p90 of MSE_15 across trajectories, to catch the case where the average
   passes but a minority of trajectories blow up? Recommended addition.
3. **Seed frame choice.** We seed from `traj.encoded_frames[0]`, i.e. the
   episode start. A stricter test would seed mid-episode (where the
   latent distribution is richer than spawn frames). Propose: seed at
   `t = 0` for the first pass, add a mid-episode ablation if stability
   looks borderline.
4. **Action distribution shift.** Training data is random-policy; a
   future Dreamer policy would bias actions. Out of scope for this probe
   (rescor is a next-step predictor, not an actor), but worth flagging
   in the follow-up design.
5. **Shared `_collect_frames` seed.** `generate_crafter_real` uses `seed`
   both for data collection *and* model training. Once the `.npy` cache
   exists, the `seed` arg only affects training. We should document this
   in the new wrapper's docstring to avoid confusion.
6. **Grid size 16 assumption.** `grid_size` is fixed at 16 because that
   is what the frozen encoder outputs. We should either assert this in
   the new wrapper or document the coupling.

---

## Files that will change (for reviewer)

| File | Change |
| ---- | ------ |
| `src/wmca/crafter_real.py` | **Add** `generate_crafter_real_trajectories` and `TrajectorySpec` dataclass. Do not touch the existing function. |
| `dreamerv3_scaffolding/rollout_stability_probe_crafter.py` | **New** file, ~250 lines, modeled on `rollout_stability_probe.py`. |
| `experiments/results/rollout_stability_probe_crafter.json` | **New** artifact. |
| `experiments/crafter_data/episode_boundaries.npy` | **New** cache (~5 KB), written on first call. |

No changes to `benchmarks.py`, the model registry, or any synthetic probe
code. No training-path regression risk.
