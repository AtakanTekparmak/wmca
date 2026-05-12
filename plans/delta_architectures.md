# Path C — Delta-Token / Delta-Embedding World Modeling: Two Architectures

**Status:** design (2026-05-08). Pre-empirical; numbers below are param-count
estimates and time budgets, not measured results.

## 0. Motivation & constraints

From the delta probe (`experiments/_cmdr_delta_probe.py` on Crafter):

- **27%** of pixels change per frame (P50; mean ≈ same order).
- **Bimodal** per-frame MSE: NOOP frames vs action frames cluster on opposite
  sides of a clear gap. A *single global* delta embedding cannot capture this
  support — the change is sparse, spatially localized, and the location matters.
- **16×16 patch-MSE** has structure: most patches are near-zero, a handful are
  large. ⇒ keep the 16×16 spatial grid; predict per-patch deltas.

This rules out "frame-level scalar delta" or "single 64-D delta vector" as the
prediction target. It is consistent with the existing CML+NCA stack, which
*requires* a 2D grid input (`CML2DMultiR` runs grouped 2D conv on
`(B, 1, H, W)`).

Cross-task evidence that motivates Path C:

- Atari latent (Plan 0 Path A) showed continuous Rescor predicts at **2.6×
  stable** at H=100 on Breakout — the continuous-MSE-on-latents recipe works
  on a deterministic-looking environment but only with strong continuity in
  the latent.
- Crafter is sparser and more discrete-looking than Breakout. So we need
  *both* a continuous and a discrete delta path; the discrete one is the
  hedge against MSE collapsing onto blurred per-pixel means.

## 1. Shared components

### 1.1 `DeltaEncoder` (shared by A and B)

Goal: encode `(frame_t, frame_{t+1})` → per-patch delta embedding
`(B, D, 16, 16)` with `D=64`. The patch grid is the same one CML+NCA already
uses, so the embedding lines up with the 16×16 reservoir geometry.

Design choice — **shared siamese branches + 1×1 fusion**:
- Inductive bias of "delta is symmetric in (a, b)" is partially encoded by
  shared weights (each frame is encoded by the same CNN; concat then
  fuse).
- Cheaper than a full concatenated 6-channel encoder.

```
frame_t ────────┐
(B,3,64,64)     │
                ├── shared FrameBranch ──► (B,32,16,16) ─┐
frame_{t+1} ────┘                                        │
(B,3,64,64)                                              ├─ concat ─► (B,64,16,16)
                                                         │             │
                                                         │             ▼
                                                         │     1×1 fusion conv
                                                         │     (64 → D=64)
                                                         │             │
                                                         │             ▼
                                                         │     (B,D=64,16,16)
                                                         │     delta_emb_grid
                                                         │
                                shared FrameBranch ──────┘
                                (Conv 3→16 s2 → 16→32 s2 → 32→32 p1)
```

Param count (rough, `D=64`):
- FrameBranch (shared, applied twice): 3·16·16 + 16·32·16 + 32·32·9 ≈ 19K
- Fusion 1×1: 64·64 + 64 ≈ 4.2K
- **Total: ~23K trainable** (one shared branch, used twice).

This module is **identical between A and B** — only the head downstream of
`delta_emb_grid` differs.

### 1.2 `DeltaDecoder` (used for visualisation + pixel-space loss option)

Goal: apply a predicted delta embedding to `frame_{t+1}` to reconstruct
`frame_{t+2}`.

Strategy — concatenate the delta embedding grid with a downsampled version
of `frame_{t+1}` and decode jointly:

```
frame_{t+1}  ──► AvgPool 4×  ──► (B,3,16,16) ──┐
                                                ├─ concat ─► (B,3+D,16,16)
delta_emb_grid (B,D=64,16,16) ─────────────────┘                  │
                                                                  ▼
                                                          (Conv (3+D)→64 p1
                                                           → ConvT 64→32 s2
                                                           → ConvT 32→3 s2 → sigmoid)
                                                                  │
                                                                  ▼
                                                          frame_{t+2}_hat
                                                          (B,3,64,64)
```

Param count: ~38K trainable. Used at *training time* only when we add
pixel-space loss; rollouts can stay in embedding space and decode once at
the end (cheaper, more stable).

## 2. Architecture A — Continuous Delta Embeddings

### 2.1 Forward pass

```
Step 1: encode delta from observed frame pair (training only)
  delta_gt = DeltaEncoder(frame_t, frame_{t+1})         # (B,D,16,16)

Step 2: predict next delta given current delta + action
  RescorDeltaContinuous(delta_emb, action) -> delta_pred:

      x      = delta_emb                                  # (B,D,16,16)
      a_emb  = action_embed(action).expand(.,.,16,16)    # (B,D,16,16)
      mix    = concat([x, a_emb])                         # (B,2D,16,16)
      drive  = sigmoid(input_proj_1x1(mix))               # (B,1,16,16) ∈ [0,1]
      cml    = CML2DMultiR(K=32, steps=15)(drive)         # (B,1,16,16)  no_grad
      nca_in = concat([drive, cml])                       # (B,2,16,16)
      corr   = NCA(nca_in)                                # (B,D,16,16)
      cml_e  = cml_proj_1x1(cml)                          # (B,D,16,16)
      delta_pred = corr + cml_e                           # (B,D,16,16)

Step 3 (training): MSE loss
  L = MSE(delta_pred, delta_gt)
      [optional pixel-space term:
       L += λ * MSE(DeltaDecoder(frame_{t+1}, delta_pred), frame_{t+2})]

Step 4 (rollout):
  delta_t = DeltaEncoder(f_{t-1}, f_t)              # warmup, observed
  for h in 1..H:
      delta_t = RescorDeltaContinuous(delta_t, a_t)
      # (optionally) f_{t+1} = DeltaDecoder(f_t, delta_t); f_t = f_{t+1}
  return delta_pred at H, or decoded frame at H
```

### 2.2 Module diagram

```
                            ┌──────── DeltaEncoder (shared) ────────┐
  (f_t, f_{t+1}) ─────────► │ siamese conv + 1×1 fusion             │ ──► δ_t (B,D,16,16)
                            └────────────────────────────────────────┘
                                              │
                                              ▼
            ┌─────────────────── RescorDeltaContinuous ────────────────────┐
            │                                                              │
            │  δ_t ⊕ act_emb ──► 1×1 conv ──► sigmoid ──► drive (1ch)      │
            │                                                │             │
            │                                                ▼             │
            │                                  CML2DMultiR (K=32, frozen) │
            │                                                │             │
            │                                                ▼             │
            │  drive ──┐    cml_out ──┐    [drive ⊕ cml] ──► NCA ──► corr │
            │          │              │                                   │
            │          │              ▼                                   │
            │          │      cml_proj 1×1 ──► cml_e (D)                  │
            │          │                                │                 │
            │          └────────► (consistency loss?)   ▼                 │
            │                                  corr + cml_e ──► δ_{t+1}   │
            └──────────────────────────────────────────────────────────────┘
                                              │
                                              ▼
                            ┌──── DeltaDecoder (training/eval only) ─────┐
            (f_t, δ_{t+1}) ►│ concat → conv stack → upsample to 64×64    │ ──► f_{t+1}_hat
                            └──────────────────────────────────────────────┘
```

### 2.3 Param count (Architecture A)

| Module                         | Trained params | Notes                              |
|--------------------------------|----------------|-------------------------------------|
| DeltaEncoder (shared siamese)  | ~23K           | identical to §1.1                   |
| Action embedding (17 × 64)     | ~1.1K          | Crafter has 17 actions              |
| input_proj (2D=128 → 1, 1×1)   | ~0.13K         |                                     |
| CML2DMultiR (K=32, frozen)     | 0              | r-buffer, kernel-buffer, no learn   |
| NCA (Conv2→16 p1 → Conv16→64)  | ~1.4K          | mirrors existing DiscreteRescor NCA |
| cml_proj (1 → 64, 1×1)         | ~0.13K         |                                     |
| (output head, optional Conv D→D)| ~4.2K         | identity-init OK; or skip           |
| DeltaDecoder (eval only)       | ~38K           | not in core predictor budget        |
| **Predictor only (no decoder)**| **~30K**       | encoder + rescor                    |
| **Full A with decoder**        | **~70K**       | encoder + rescor + decoder          |

This matches the ~70K target in the brief.

## 3. Architecture B — Discrete Delta Tokens

### 3.1 Forward pass

```
Step 1: encode delta from frame pair, then quantise (training)
  delta_emb     = DeltaEncoder(frame_t, frame_{t+1})    # (B,D,16,16)
  delta_q, idx, vq_loss = VectorQuantizer(delta_emb)    # idx (B,16,16) long
                                                        # vq_loss = commitment

Step 2: predict next delta tokens given current delta tokens + action
  DiscreteRescorDelta(idx, action) -> logits (B, V=512, 16, 16):

      tok_emb  = token_embed(idx).permute(0,3,1,2)           # (B,D,16,16)
      a_emb    = action_embed(action).expand(.,.,16,16)      # (B,D,16,16)
      mix      = concat([tok_emb, a_emb])                    # (B,2D,16,16)
      drive    = sigmoid(input_proj_1x1(mix))                # (B,1,16,16)
      cml      = CML2DMultiR(K=32, steps=15)(drive)          # (B,1,16,16) no_grad
      nca_in   = concat([drive, cml])                        # (B,2,16,16)
      corr     = NCA(nca_in)                                 # (B,D,16,16)
      cml_e    = cml_proj_1x1(cml)                           # (B,D,16,16)
      embed    = corr + cml_e                                # (B,D,16,16)
      logits   = output_head_1x1(embed)                      # (B,V,16,16)

Step 3 (training): CE on the next delta tokens + VQ commitment
  L = CE(logits, idx_{t+1})  +  vq_loss

Step 4 (rollout): autoregressive token prediction
  idx_t = VQ.encode(DeltaEncoder(f_{t-1}, f_t))               # warmup
  for h in 1..H:
      logits_{t+1} = DiscreteRescorDelta(idx_t, a_t)
      idx_{t+1}    = argmax(logits_{t+1}, dim=1)              # or sample
      idx_t        = idx_{t+1}
  # decode at terminal:
  delta_emb_H = VQ.decode_indices(idx_H)
  f_H         = DeltaDecoder(f_0, delta_emb_H)                # see note
```

Note on rollout decoding: with discrete deltas, applying the delta requires
either a *delta-conditional decoder* (DeltaDecoder above), or composing
deltas in embedding space and decoding once at H. We will use the latter:
`f_H = DeltaDecoder(f_0, sum_h VQ.decode(idx_h))` is wrong because deltas
are not additive in pixel space; instead, decode every step and pass the
reconstructed frame forward. Cost: ~16× more compute than embedding-only
rollout, but evaluator-only (training is teacher-forced).

### 3.2 Module diagram

```
   (f_t, f_{t+1}) ─► DeltaEncoder ─► δ_t ─► VectorQuantizer ─► idx_t (B,16,16) long
                                                      │              │ (commit loss)
                                                      ▼              ▼
                                              codebook (V=512 × D=64, EMA)
                                                                     │
   ┌──────────── DiscreteRescorDelta (≈ existing DiscreteRescor) ────┘────┐
   │                                                                       │
   │  idx_t  ──► token_embed ──► (B,D,H,W) ──┐                             │
   │                                          ├─ concat ─► 1×1 ─► sigmoid │
   │  action ──► action_embed ──► tile  ─────┘                             │
   │                                                  │ drive (1ch)        │
   │                                                  ▼                    │
   │                                       CML2DMultiR (K=32, frozen)      │
   │                                                  │                    │
   │                                                  ▼                    │
   │           drive ┐         cml ┐                                       │
   │                 │             │                                       │
   │                 └─► concat ──► NCA ──► corr (D)                       │
   │                                            +                          │
   │                              cml_proj ──► cml_e (D)                   │
   │                                            =                          │
   │                                          embed ──► output_head ──► logits │
   └────────────────────────────────────────────────────────────────────── ┘
                                                          │
                                                          ▼
                                              CE vs idx_{t+1} (B,16,16)
```

### 3.3 Param count (Architecture B)

| Module                         | Trained params | Notes                                 |
|--------------------------------|----------------|----------------------------------------|
| DeltaEncoder (shared siamese)  | ~23K           | same as A                              |
| VectorQuantizer codebook       | 0 trained      | (V·D = 32K buffer, EMA-updated)        |
| token_embed (V=512 × D=64)     | ~33K           | trained                                |
| action_embed (17 × D=64)       | ~1.1K          |                                        |
| input_proj (2D → 1, 1×1)       | ~0.13K         |                                        |
| CML2DMultiR (K=32)             | 0              | frozen                                 |
| NCA                            | ~1.4K          |                                        |
| cml_proj (1 → 64, 1×1)         | ~0.13K         |                                        |
| output_head (D → V=512, 1×1)   | ~33K           | the second large block                 |
| DeltaDecoder (eval only)       | ~38K           | shared with A                          |
| **Predictor only**             | **~92K**       | encoder + VQ-trainable + rescor        |
| **Full B with decoder**        | **~130K**      |                                        |
| **Full B w/o decoder, w/o VQ EMA buf** | **~92K** | |

Higher than A by ~30K: the cost is roughly `V·D` twice (token_embed +
output_head), which is unavoidable for a vocabulary head.

## 4. Comparison

| Aspect                       | A — Continuous                            | B — Discrete                                      |
|------------------------------|-------------------------------------------|---------------------------------------------------|
| Loss                         | MSE on delta embeddings                   | CE on delta token indices (+ commit)              |
| Trained params (predictor)   | ~30K                                      | ~92K                                              |
| Multi-hypothesis             | No (deterministic MSE)                    | Yes (sample from softmax)                         |
| Autoregressive stability     | Same regime as Breakout 2.6× @ H=100      | Unknown; CE often sharper, but H=100 untested     |
| MSE-blur failure mode        | Mode collapse to mean delta on action steps| Cannot blur — discrete tokens                    |
| Training complexity          | Plain MSE                                 | VQ + straight-through + commitment                |
| CML compatibility            | Native (continuous → 1ch sigmoid → CML)   | Indirect (token embed → 1ch sigmoid → CML)        |
| Rollout decode cost          | One decoder call at H (or per-step)       | Per-step decode required (deltas not additive)    |
| Reuse from Plan 0 / Path C   | Reuses CML+NCA pattern, new encoder       | Reuses DiscreteRescor verbatim, swap input dist   |
| Risk                         | Embedding collapses; MSE blurs sparse delta| Codebook dies (>50% dead); CE sharpens to wrong mode |
| First-result-cost (1 day)    | ~3 GPU-hr                                 | ~4 GPU-hr                                         |

Recommendation if both produce a number in week 1: **A is the cleaner falsifier**
(smaller, simpler loss, directly comparable to Breakout 2.6×). **B is the
hedge** for the case where Crafter delta distribution is too multimodal for
MSE — which the bimodal NOOP-vs-action probe result already suggests is
likely.

## 5. File list

New files (all under `src/wmca/modules/` and `experiments/`):

```
src/wmca/modules/
  delta_encoder.py            # DeltaEncoder + DeltaDecoder (shared by A and B)
  rescor_delta_continuous.py  # RescorDeltaContinuous (A)
  # discrete delta uses existing src/wmca/modules/discrete_rescor.py
  # VQ uses existing src/wmca/modules/vqvae.py (VectorQuantizer)

experiments/
  train_delta_encoder.py      # train DeltaEncoder by reconstruction:
                              #   MSE(DeltaDecoder(f_t, DeltaEncoder(f_t,f_{t+1})), f_{t+1})
  train_delta_vq.py           # train VQ on delta_emb_grid (uses encoder from above)
  train_rescor_delta_cont.py  # Architecture A end-to-end
  train_discrete_rescor_delta.py  # Architecture B end-to-end
  rollout_delta_continuous.py # H=1..100 sweep on A, log delta-MSE + pixel-MSE
  rollout_delta_discrete.py   # H=1..100 sweep on B, log token accuracy + pixel-MSE
```

Modified files (minimal):

```
src/wmca/model_registry.py    # register "rescor_delta_cont", "rescor_delta_disc"
src/wmca/modules/__init__.py  # export DeltaEncoder, DeltaDecoder, RescorDeltaContinuous
```

## 6. Experiment plan (1 day per architecture, ~7 days total with rollouts)

### Day 1 — Shared: `DeltaEncoder` reconstruction sanity check

- Train `DeltaEncoder + DeltaDecoder` on Crafter pairs to minimise
  `MSE(DeltaDecoder(f_t, DeltaEncoder(f_t, f_{t+1})), f_{t+1})`.
- 30 min on a single 4090. Gate: per-pixel MSE within 2× of FrameAutoencoder
  baseline (~5e-3). If worse, the encoder is too small — revisit §1.1 widths.
- Snapshot encoder; freeze it for downstream A and B (so they share the same
  embedding space and are comparable).

### Day 2 — Architecture A: continuous delta Rescor

- Plug frozen `DeltaEncoder` into `RescorDeltaContinuous`.
- Train with teacher forcing, MSE on delta embeddings, 50 epochs.
- 1 GPU-hr.
- Eval: H=1..100 rollout, report:
  - delta-embedding MSE @ H (vs ground-truth delta embedding)
  - pixel MSE @ H via DeltaDecoder
  - **Pre-registered gate:** H=100 ratio (rel to H=1) ≤ 5×. Breakout did 2.6×;
    Crafter is sparser so we soften the bound.

### Day 3 — Shared: train VQ on delta embeddings

- Take frozen DeltaEncoder; train VectorQuantizer on
  `DeltaEncoder(f_t, f_{t+1})` over the dataset.
- V=512, D=64, EMA codebook. 30 min.
- Gate: codebook utilization > 30% AND per-token reconstruction MSE within
  2× of continuous embedding.

### Day 4 — Architecture B: discrete delta Rescor

- Reuse `DiscreteRescor` with vocab=512; feed delta-token grid as input,
  delta-token grid at t+1 as target.
- CE loss + 0.25 · commitment loss, 50 epochs. 1 GPU-hr.
- Eval: H=1..100 token accuracy, plus DeltaDecoder pixel MSE.
- **Pre-registered gate:** H=100 token accuracy ≥ 80% (Crafter has many
  unchanged patches per step, so even a "predict no-change" baseline will
  hit ≥73% — anything ≤ that is degenerate).

### Day 5 — A vs B head-to-head

- Same pixel-MSE @ H=1, 5, 15, 100, on the same eval set.
- Decision rule:
  - Both ≤ 2× FrameAutoencoder pixel MSE → write up; pick smaller.
  - Only A → continuous wins; abandon B.
  - Only B → multimodal hypothesis confirmed; B is the path.
  - Neither → escalate; the encoder is the bottleneck, not the predictor.
    Re-investigate §1.1 and consider explicit motion-mask channel.

## 7. GPU budget

| Stage                                  | Hardware    | Wall time   |
|----------------------------------------|-------------|-------------|
| DeltaEncoder reconstruction            | 1× 4090     | ~30 min     |
| Train VQ on delta embeddings           | 1× 4090     | ~30 min     |
| Architecture A training (50 epochs)    | 1× 4090     | ~1 hr       |
| Architecture B training (50 epochs)    | 1× 4090     | ~1 hr       |
| Rollout sweep A (H=1..100, 5 seeds)    | 1× 4090     | ~30 min     |
| Rollout sweep B (H=1..100, 5 seeds)    | 1× 4090     | ~30 min     |
| **Total per-architecture**             |             | **~3-4 hr** |
| **Total both architectures**           |             | **~7-8 hr** |

Sits comfortably within one local 4090-day. No cluster needed for the MVP.
If we extend to multi-seed (5 seeds × 5 r-grids) for the final Rescor table,
multiply by ~10×, putting it at ~3 days on one card or one day on a Prime
Intellect 4×4090 pod.

## 8. Open questions / risks

1. **Delta embedding collapse.** If `DeltaEncoder` learns to ignore the
   second frame and output something close to a fixed embedding (encoder
   exploits the loss), both A and B inherit the failure. Mitigation: enforce
   `DeltaEncoder(f, f) ≈ 0` via auxiliary loss on a held-out NOOP slice.
2. **VQ codebook collapse on deltas.** Most patches are zero-delta; the VQ
   may allocate ≥ 50% of its codebook to "no change" tokens. Mitigation:
   monitor codebook utilization every epoch; if < 20% on Day 3, halve V to
   256 and re-train.
3. **CE sharpens to wrong mode (B).** With argmax at rollout, B may lock onto
   the most-frequent token (likely the no-change one) and never re-engage.
   Mitigation: report both argmax and top-1-sampled rollout; if argmax
   degenerates and sampled is fine, the argmax behaviour is a known CE-MAP
   pathology, not a model failure.
4. **Pixel-space loss vs embedding-space loss for A.** Embedding-space loss
   is what worked on Breakout (Path A). Pixel-space adds a decoder gradient
   path; we will train embedding-space first and only add pixel loss if the
   H=100 gate fails.
5. **Action conditioning depth.** Both A and B tile the action embedding
   over the 16×16 grid before fusion. If action signal washes out (sparse
   actions, small batch), upgrade to FiLM-style modulation on the NCA
   correction.

## 9. Dependencies on existing code

- `src/wmca/modules/hybrid.py::CML2DMultiR` — used unchanged.
- `src/wmca/modules/vqvae.py::VectorQuantizer` — used unchanged for B.
- `src/wmca/modules/discrete_rescor.py::DiscreteRescor` — used as-is for B's
  predictor (only the input distribution changes — delta tokens, not full
  frame tokens).
- Existing `experiments/crafter_data/{frames.npy, next_frames.npy,
  actions.npy}` — used for training data.

No changes to `CML2DMultiR`, `VectorQuantizer`, or `DiscreteRescor` are
required. The two new modules are `DeltaEncoder/DeltaDecoder` and
`RescorDeltaContinuous`.
