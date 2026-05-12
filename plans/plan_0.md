# Plan 0 — Atari Latent Probe + Iris-Style Discrete Token WM

**Date**: 2026-05-04
**Status**: *planning*
**Goal**: Build two new evaluation pipelines — (A) Atari latent rollout prediction to test rescor on pixel-level game dynamics beyond Crafter, and (C) an Iris-style VQ-VAE + discrete token world model to explore whether the CML+NCA architecture transfers to discrete sequence prediction.

---

## Context

The existing Atari benchmarks (Pong, Breakout) are **ceilinged** — every model including the 197K MLP hits >99% accuracy. They're 1-step classification on tiny deterministic grids and tell us nothing about rollout stability or pixel-level modeling.

There is **zero** discrete token infrastructure in the codebase. No VQ-VAE, no codebook, no Gumbel-softmax, no autoregressive transformer over tokens. The project operates entirely on continuous [0,1]-bounded latents.

Both paths build infrastructure that the project needs:
- **Path A** fills the Atari-shaped hole — if rescor is a world model, it needs Atari evidence.
- **Path C** explores whether the CML+NCA architecture extends to the Iris/Delta-Iris paradigm, which is the current SOTA for model-based RL with discrete latents.

---

## Path A: Atari Latent Rollout Probe (~3 days, ~$15 GPU)

### Goal
Train rescor on Atari autoencoder latents with action conditioning, then measure autoregressive rollout stability at H={15, 50, 100}. Answer: does rescor's spatial inductive bias help on Atari dynamics beyond the trivial 1-step classification ceiling?

### A.1 — Train Atari Frame Encoder (~4h GPU + 2h dev)

**What**: A continuous autoencoder for Atari frames. Two options evaluated — pick one:
- **(a) Render to 64×64 RGB**: Assign colors per one-hot channel (ball=white, paddles=green, walls=gray), upscale to 64×64. Then use the existing FrameEncoder architecture (64×64×3 → 16×16×1 → 64×64×3).
- **(b) Grid-native AE**: Build autoencoder directly on 4ch one-hot grids: (4, H, W) → (1, H, W) → (4, H, W). No spatial compression needed since Atari grids are already small (Pong 16×32, Breakout 20×16).

Option (b) is simpler and avoids the rendering step. Option (a) is preferred if you want the "latent dynamics" framing to mirror Crafter exactly (pixels → AE → latents → dynamics).

**New files**:
- `experiments/train_atari_encoder.py` — collects Atari frames via random play, trains encoder

**Implementation**:
1. Use `src/wmca/envs/atari_pong.py` — both Pong and Breakout envs. Run random policy.
2. If option (a): render one-hot grids to 64×64 RGB, save as `(N, 3, 64, 64)` uint8. If option (b): save raw one-hot grids as `(N, 4, H, W)` float32.
3. Build autoencoder:
   - Option (a): Identical to Crafter FrameEncoder: Encoder(3→32, 4×4 s=2) → ReLU → Encoder(32→64, 4×4 s=2) → ReLU → Encoder(64→1, 1×1) → Sigmoid → (B, 1, 16, 16). Decoder symmetric.
   - Option (b): Grid-native: Encoder(4→16, 3×3 p=1) → ReLU → Encoder(16→1, 3×3 p=1) → Sigmoid → (B, 1, H, W). Decoder(1→16, 3×3 p=1) → ReLU → Decoder(16→4, 3×3 p=1) → Sigmoid. No strided convs — Atari grids are already the right resolution.
4. Train with MSE loss, batch=128, Adam lr=1e-3, 50 epochs. Save checkpoint to `experiments/atari_data/frame_encoder.pt`.
5. Collect latent data: encode all frames → `experiments/atari_data/pong_frames.npy` (shape: (N, 1, H, W) where H×W = env grid size), `pong_actions.npy`, `pong_next_frames.npy`.
5. Collect Breakout data identically → `experiments/atari_data/breakout_frames.npy`, etc.

**Validation**: Reconstruction PSNR > 25 dB (option a) or > 35 dB (option b — easier, no compression). Visual sanity check.

### A.2 — Action-Conditioned Latent Training (~6h GPU + 3h dev)

**What**: Train rescor_rens K=32 and rescor_mamba_rand on Atari latent state prediction, action-conditioned. Input: [encoded_frame (1ch) + action_field (1ch)] → (2, H, W). Output: next_encoded_frame (1, H, W). Continuous MSE loss.

**New files**:
- `src/wmca/atari_real.py` — `AtariLatentBenchmark` class, sibling of `crafter_real.py`
- `dreamerv3_scaffolding/rollout_stability_probe_atari.py` — rollout probe for Atari latents

**Implementation**:
1. `AtariLatentBenchmark` class:
   - Loads `pong_frames.npy`, `pong_actions.npy`, `pong_next_frames.npy`
   - Action field: single float channel, value = (action+1)/N_ACTIONS (Pong: {0.33, 0.67, 1.0} for {stay, up, down})
   - X = concat([frame(1ch), action_field(1ch)]) → (2, H, W). Y = next_frame → (1, H, W)
   - Grid size matches env: Pong 16×32, Breakout 20×16 (no spatial compression — AE latent = env grid size)
   - Train/val/test split: 70/15/15
   - `meta["loss_type"] = "mse"`, `meta["action_conditioned"] = True`
2. Train `rescor_rens` (in_ch=2, out_ch=1, grid_size env-dependent) and `rescor_mamba_rand` (K=4 context).
   - **Note**: `create_model` derives `use_sigmoid` from channel equality (out_ch == in_ch) and doesn't accept it as a kwarg. For Atari latent MSE (in_ch=2, out_ch=1), either (a) add a `use_sigmoid` override kwarg to `create_model`, or (b) instantiate `ResidualCorrectionWM` directly with `use_sigmoid=True` to ensure output is clamped to [0,1] matching AE latents.
   - 3 seeds × 2 models × 100 epochs each
   - Training stack: bf16 + torch.compile + batch=128 + lr=1.4e-3 sqrt-rule (same as sprint)
3. 1-step evaluation: per-cell MSE on test set

### A.3 — Action-Conditioned Autoregressive Rollout (~2h GPU + 4h dev)

**What**: The current rollout evaluation (`evaluate_rollout` in `model_registry.py`) **bails out** when `in_ch != out_ch` because it can't autoregressively feed actions. We need a new rollout probe that handles action-conditioned autoregression.

**New files**:
- `dreamerv3_scaffolding/rollout_stability_probe_atari.py`

**Implementation**:
1. For each test trajectory: load T consecutive (frame, action, next_frame) triples
2. Rollout loop:
   - t=0: start with ground-truth frame_0
   - For each step h:
     - Build input: [pred_frame_h, action_field(a_h)]
     - Model forward → pred_frame_{h+1}
     - Record MSE(pred_frame_{h+1}, gt_frame_{h+1})
     - pred_frame becomes input for next step
3. Run for H={15, 50, 100}. Report per-step MSE, MSE/step1 ratio, cosine divergence.
4. Action sequence is ground-truth (open-loop action, closed-loop state) — we're testing dynamics prediction, not policy.
5. 20 test trajectories per evaluation.

**Decision criteria**:
- **PASS**: H=15 MSE < 2× step-1 MSE AND H=100 cos_div < 0.10. Rescor tracks Atari dynamics stably.
- **MARGINAL**: H=15 MSE 2-5× step-1. Rollout degrades but doesn't catastrophize.
- **FAIL**: H=100 cos_div > 0.30. Atari dynamics are harder than they look.

### A.4 — Files Created (Path A)

| File | Purpose |
|------|---------|
| `experiments/train_atari_encoder.py` | Train autoencoder on Atari frames |
| `experiments/atari_data/pong_frames.npy` | Encoded Pong latents |
| `experiments/atari_data/pong_actions.npy` | Action sequences |
| `experiments/atari_data/pong_next_frames.npy` | Next-frame latents |
| `experiments/atari_data/breakout_frames.npy` | Encoded Breakout latents |
| `experiments/atari_data/breakout_actions.npy` | Action sequences |
| `experiments/atari_data/breakout_next_frames.npy` | Next-frame latents |
| `experiments/atari_data/frame_encoder.pt` | Trained encoder checkpoint |
| `src/wmca/atari_real.py` | AtariLatentBenchmark data loader |
| `dreamerv3_scaffolding/rollout_stability_probe_atari.py` | Action-conditioned rollout probe |

**GPU budget**: ~12h total on RTX Pro 6000. ~$17.

---

## Path C: Iris-Style Discrete Token World Model (~2 weeks, ~$50 GPU)

### Goal
Build a VQ-VAE for Crafter frames, then adapt rescor to predict discrete token sequences autoregressively. Compare discrete-token rescor to continuous-latent rescor. Answer: does the CML+NCA architecture work for discrete sequence modeling, or is it fundamentally a continuous-dynamics architecture?

### C.1 — VQ-VAE for Crafter Frames (~8h GPU + 4h dev)

**What**: Vector-quantized autoencoder for Crafter 64×64×3 frames. Encodes to a grid of discrete token indices (e.g., 16×16 tokens from a codebook of size 256-512). The decoder reconstructs from token embeddings.

**New files**:
- `src/wmca/modules/vqvae.py` — VQ-VAE module
- `experiments/train_vqvae_crafter.py` — training script

**Implementation**:
1. Architecture:
   - Encoder: Conv2d(3→64, 4×4 s=2) → ReLU → Conv2d(64→128, 4×4 s=2) → ReLU → Conv2d(128→64, 3×3 p=1) → (B, 64, 16, 16) continuous latent
   - VectorQuantizer: codebook of size V ∈ {256, 512}, embedding dim=64. Straight-through gradient. Commitment loss β=0.25.
   - Decoder: ConvTranspose2d(64→128, 4×4 s=2) → ReLU → ConvTranspose2d(128→64, 4×4 s=2) → ReLU → Conv2d(64→3, 1×1) → Sigmoid
   - Total loss: L_recon + β * L_commit + 0.1 * L_codebook (EMA or learned)
   - ~2-5M params (most in decoder convs). Comparable to Iris's encoder budget.
2. Train on existing Crafter frame data (`experiments/crafter_data/` — needs raw frames, not encoded). If raw frames aren't available, collect them via `crafter` env random play.
3. After training: encode all frames → token index grid (B, 16, 16) of dtype long in [0, V-1].
4. Save: `experiments/crafter_data/tokens.npy`, `next_tokens.npy`, `actions.npy`, `vqvae_checkpoint.pt`, `codebook.pt`.

**Validation**: Reconstruction PSNR > 24 dB. Codebook utilization > 50% (no dead codes). Visual sanity check: decode random tokens, verify recognizable Crafter frames.

### C.2 — Discrete Token Prediction Model (~4h dev)

**What**: Adapt rescor for discrete token prediction. Input: token embedding grid + action embedding. Output: per-position logits over vocabulary V. CE loss over token indices.

**New files**:
- `src/wmca/modules/discrete_rescor.py` — DiscreteRescor class
- Added to `src/wmca/model_registry.py` MODEL_REGISTRY

**Implementation**:
1. `DiscreteRescor` class (takes discrete token indices as input, embeds internally):
   ```python
   class DiscreteRescor(nn.Module):
       def __init__(self, vocab_size, n_actions, embed_dim=64, hidden_ch=16, cml_K=32, ...):
           self.token_embed = nn.Embedding(vocab_size, embed_dim)   # V → 64
           self.action_embed = nn.Embedding(n_actions, embed_dim)   # n_actions → 64
           # Combine token + action embeddings → 1ch CML input (squashed to [0,1])
           self.input_proj = nn.Conv2d(embed_dim * 2, 1, 1)        # 128 → 1
           # Standard rescor_rens K=32 CML bank + NCA
           self.rens = CML2DMultiR(in_channels=1, K=cml_K, ...)
           self.nca = nn.Sequential(
               nn.Conv2d(1 + 1, hidden_ch, 3, padding=1),           # [input, cml_mean]
               nn.ReLU(),
               nn.Conv2d(hidden_ch, embed_dim, 1),                 # → embedding dim
           )
           self.output_head = nn.Conv2d(embed_dim, vocab_size, 1)  # → logits

       def forward(self, tokens, action):
           # tokens: (B, H, W) long — discrete token indices
           # action: (B,) long — discrete action index
           B, H, W = tokens.shape
           tok_emb = self.token_embed(tokens).permute(0, 3, 1, 2)     # (B, 64, H, W)
           act_emb = self.action_embed(action)                        # (B, 64)
           act_emb = act_emb.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
           combined = torch.cat([tok_emb, act_emb], dim=1)            # (B, 128, H, W)
           cml_input = torch.sigmoid(self.input_proj(combined))       # (B, 1, H, W) — squash
           cml_out = self.rens(cml_input)
           correction = self.nca(torch.cat([cml_input, cml_out], dim=1))
           logits = self.output_head(correction + cml_out)            # (B, V, H, W)
           return logits  # raw logits for CE loss
   ```
2. Key design decisions:
   - Model accepts discrete token indices (long tensors), NOT pre-embedded features. Embeddings happen internally.
   - Action is also an integer index, embedded to the same dimension as tokens.
   - Token + action embeddings are concatenated (2*embed_dim=128 channels), projected to 1ch via 1×1 conv, squashed to [0,1] for CML.
   - CML operates on 1-channel continuous embedding (not discrete tokens). The embedding provides the continuous substrate the CML needs.
   - The NCA correction produces token embeddings, which are projected to vocabulary logits.
   - Residual: `cml_out + nca_correction` in embedding space, then logit projection.
   - No sigmoid on output — raw logits for CrossEntropyLoss.
3. Register as `"discrete_rescor"` in MODEL_REGISTRY.

**Why this design**: The CML reservoir requires continuous [0,1] bounded inputs (logistic map domain). We can't feed discrete token IDs directly. The embedding projects tokens into a continuous space where the CML's chaotic dynamics can operate. The NCA corrects in embedding space, then a linear head maps to vocabulary logits. This mirrors how Iris uses a transformer over discrete token embeddings — we swap the transformer for rescor.

### C.3 — Discrete Token Training Loop (~4h dev)

**What**: Training loop for discrete token prediction. Uses CrossEntropyLoss over vocabulary. Handles token reshaping.

**Modifications to existing files**:
- `src/wmca/model_registry.py` — add special handling in `train_model` for discrete token prediction, OR write a standalone training function

**Implementation**:
1. Training data format: `tokens = (N, H, W)` long tensor of token indices. `actions = (N,)` long tensor of action indices. `next_tokens = (N, H, W)` long tensor.
2. For mamba variants: K=4 context of token sequences → embed K consecutive tokens, stack as (B, K*embed_dim, H, W) or adapt internally.
3. Loss: `F.cross_entropy(logits, Y)` where logits is (B, V, H, W) and Y is (B, H, W) of class indices. PyTorch CE handles the shape automatically (channels=classes).
4. Metric: per-token accuracy = fraction of (H*W) positions where argmax(logits) == Y.
5. Multistep H=8 training: autoregressive over token predictions, accumulate per-step CE loss, truncated BPTT through last K_bptt=4 steps.

**Standalone training function** (cleaner than modifying train_model):
```python
def train_discrete_rescor(model, tokens, actions, next_tokens, epochs=100, ...):
    # tokens: (N, H, W) long, actions: (N,), next_tokens: (N, H, W) long
    # Build input: embed tokens + action field
    # Train with CE loss
    # Return best checkpoint
```

### C.4 — Discrete Token Autoregressive Generation (~4h dev)

**What**: Autoregressive generation of token sequences. Start from ground-truth token grid, roll forward H steps.

**New files**:
- `dreamerv3_scaffolding/discrete_token_rollout_probe.py`

**Implementation**:
1. Start with ground-truth token grid t=0 (from test set).
2. For h in 0..H-1:
   - Embed current token grid: `(H, W) → (embed_dim, H, W)` via token_embed lookup
   - Concat action embedding: `(embed_dim+action_embed_ch, H, W)`
   - Model forward → logits `(V, H, W)`
   - Sample or argmax → next token grid `(H, W)`
   - Feed back as input for next step
3. Metrics:
   - Per-step token accuracy: fraction of positions where predicted token == GT token
   - Sequence-level metrics: % of grids with 0 errors, mean edit distance
   - Decode final grid through VQ-VAE decoder → visual quality check
4. Also run continuous latent baseline for comparison:
   - Train standard rescor_rens on the VQ-VAE's continuous latents (the pre-quantization 64-dim feature maps)
   - Roll same H steps with MSE loss
   - Decode predictions through VQ-VAE decoder
   - Compare PSNR/SSIM of decoded frames

### C.5 — Comparison Matrix

Run both discrete and continuous models, compare:

| Model | Latent | Loss | H=15 acc | H=50 acc | H=100 acc | Decoded PSNR |
|-------|--------|------|----------|----------|-----------|--------------|
| rescor_rens (continuous) | VQ-VAE pre-quant (64ch) | MSE | ? | ? | ? | ? |
| discrete_rescor | VQ-VAE tokens (V=512) | CE | ? | ? | ? | ? |
| discrete_rescor_mamba | VQ-VAE tokens + K=4 context | CE | ? | ? | ? | ? |

### C.6 — Files Created (Path C)

| File | Purpose |
|------|---------|
| `src/wmca/modules/vqvae.py` | VQ-VAE module (Encoder, VectorQuantizer, Decoder) |
| `experiments/train_vqvae_crafter.py` | VQ-VAE training script |
| `experiments/crafter_data/vqvae_checkpoint.pt` | Trained VQ-VAE weights |
| `experiments/crafter_data/tokens.npy` | Encoded discrete token grids |
| `experiments/crafter_data/next_tokens.npy` | Next-frame token grids |
| `src/wmca/modules/discrete_rescor.py` | DiscreteRescor model class |
| `dreamerv3_scaffolding/discrete_token_rollout_probe.py` | Autoregressive token generation + evaluation |

**Modified files**:
| File | Change |
|------|--------|
| `src/wmca/model_registry.py` | Add `discrete_rescor` to MODEL_REGISTRY, add `train_discrete_rescor` function |
| `src/wmca/modules/__init__.py` | Export new modules |

**GPU budget**: ~35h total on RTX Pro 6000. ~$50.
- VQ-VAE training: ~8h
- Token encoding: ~1h
- 2 models × 3 seeds × 100 epochs: ~18h
- Rollout probes: ~2h
- Continuous baseline: ~6h

---

## Independence & Parallelization

Path A and Path C are **completely independent** — they share no files, no data, and no models. They can be executed in parallel on separate GPU pods or sequentially on one.

Within Path C, work streams that can be parallelized:
- C.1 (VQ-VAE training) — runs first, unblocks everything else
- C.2 (model class) + C.3 (training loop) — can be developed on CPU while VQ-VAE trains

Within Path A, work streams that can be parallelized:
- A.1 (encoder training) — runs first, unblocks A.2/A.3
- A.3 (rollout probe code) — can be written while A.2 trains

---

## Testing Strategy

### Path A
- Autoencoder reconstruction PSNR > 25 dB on Atari frames
- Latent data sanity: `np.allclose(encode(decode(encode(x))), encode(x), atol=1e-3)`
- 1-step MSE decreases over training epochs (no NaN)
- Rollout: H=15 MSE < 5× step-1 MSE for at least one model variant
- Visual: decode rollout predictions back to pixels — should look like Atari frames, not noise

### Path C
- VQ-VAE reconstruction PSNR > 24 dB on Crafter frames
- Codebook utilization > 50% (fewer than half the codes are "dead")
- Visual: decoded reconstructions are recognizable Crafter screens
- Discrete model: training accuracy increases (CE loss decreases)
- Token accuracy at H=1 > 80% (simple next-token prediction is easy)
- Rollout: autoregressive tokens decode to visually coherent Crafter frames at H=15
- Comparison: continuous latent baseline provides a floor for decoded PSNR

---

## GPU Budget Summary

| Item | GPU Hours | Cost (@$1.35/hr) |
|------|-----------|-------------------|
| A.1 — Atari encoder training | 4h | $5.40 |
| A.2 — Atari latent training (6 runs × ~1h) | 6h | $8.10 |
| A.3 — Atari rollout probe | 2h | $2.70 |
| **Path A subtotal** | **12h** | **$16.20** |
| C.1 — VQ-VAE training | 8h | $10.80 |
| C.1 — Token encoding | 1h | $1.35 |
| C.2/C.3 — Discrete rescor training (6 runs × ~3h) | 18h | $24.30 |
| C.4 — Rollout probes | 2h | $2.70 |
| C.4 — Continuous baseline training | 6h | $8.10 |
| **Path C subtotal** | **35h** | **$47.25** |
| **Total** | **47h** | **$63.45** |

All figures assume RTX Pro 6000 on Prime Intellect dc_gnu. On-demand, never spot.

---

## Immediate Actions (Today)

1. **[ ] Provision GPU pod** — RTX Pro 6000 via `/provision-prime-gpu` (~5 min)
2. **[x] Path A.1** — Collect Atari frames, train frame encoder (~4h GPU) — done 2026-05-07; Pong 33.1 dB, Breakout 38.6 dB after `torch.manual_seed` fix
3. **[ ] Path C.1** — Start VQ-VAE training on Crafter frames (~8h GPU, runs overnight)
4. **[ ] Path C.2** — Implement `DiscreteRescor` class on CPU (parallel with GPU training)
5. **[x] Path A.3** — Write `rollout_stability_probe_atari.py` on CPU (parallel) — done 2026-05-07; rens Breakout H=100 2.6× STABLE, Pong 36.8× chaotic; mamba Pong 28.6× / Breakout 7.4×; mamba Pong seed spread 14×–165×
6. **[x] Path A.2** — Once encoder is trained, launch Atari latent training — done 2026-05-07; 3 seeds × 2 models × 100 epochs × 2 benches; mamba ~1.7× lower 1-step MSE than rens
7. **[ ] Path C.3/C.4** — Once VQ-VAE is trained, launch discrete token training + probes
