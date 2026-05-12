# Architecture Plan — Scaling Rescor

Five new architectures for scaling rescor beyond 16x16 and single-step prediction. All maintain the core principle: frozen chaotic reservoir + learned correction.

---

## 1. `rescor_mr` — Multi-R Ensemble

**Priority: 1 (implement first)**

Run 4 CMLs with different r values on the same input in parallel. Concatenate stats for 4x richer features at zero extra learned params in the reservoir.

**CML configs:**

- r=3.70, eps=0.20, beta=0.30 (period-doubling regime)
- r=3.85, eps=0.25, beta=0.25 (3-cycle window)
- r=3.90, eps=0.30, beta=0.15 (full chaos, current default)
- r=3.95, eps=0.35, beta=0.10 (near-boundary chaos)

**Forward pass:**

```
input x (B, 1, H, W)
  -> Run 4 CMLs on same input, each with different (r, eps, beta)
  -> Concatenate stats: last(4ch), mean(4ch), var(4ch), delta(4ch), last_drive(4ch) = 20ch
  -> NCA input: [x(1ch), all_stats(20ch)] = 21 channels
  -> Dual perception (d=1 + d=2, hidden=32) with zero-init alpha
  -> Update head: ReLU + Conv2d(32,32,1) + ReLU + Conv2d(32,1,1)
  -> Residual: output = cml_r390_last + correction
```

**Params:** ~13.3K trained, 48 frozen | **Target:** 16x16

**Rationale:** Purest test of physics diversity. Different r values produce qualitatively different dynamics (period-doubling vs full chaos vs near-boundary). The NCA can pick the most informative features per cell. Zero extra learned params in the reservoir.

---

## 2. `rescor_deep` — Deep Rescor (3 layers)

**Priority: 2**

Stack 3 rescor layers sequentially with different r values per layer. Strong chaos first (coarse), weaker chaos later (fine). Zero-init depth alphas.

**Layer config:**

- Layer 1: r=3.90, M=15 steps (full chaos, coarse correction)
- Layer 2: r=3.70, M=10 steps (period-doubling, medium correction)
- Layer 3: r=3.50, M=5 steps (weak chaos, fine detail)

**Forward pass:**

```
h = x
for each layer:
    cml_input = clamp(h, 0, 1)
    stats = cml[layer](cml_input)
    nca_in = cat([x, h, stats.last, stats.mean, stats.var, stats.delta, stats.last_drive])  # 7ch
    feat = perceive[layer](nca_in)  # Conv2d(7, 32, 3x3)
    correction = update[layer](feat)
    h = h + depth_alpha[layer] * correction  # zero-init alpha
return clamp(h, 0, 1)
```

**Params:** ~6.3K trained, 36 frozen | **Target:** 16x16

**Rationale:** Pure vertical scaling. Tests iterative refinement with cascading physics regimes. At init, collapses to identity (all alphas = 0), must earn each layer. Layer 3 uses hidden_ch=16 (smaller, fine correction only).

---

## 3. `rescor_ms` — Multi-Scale Rescor

**Priority: 3**

U-Net-like pyramid with CML+NCA at each resolution level. Scales rescor from 16x16 to 64x64.

**Levels:**

- Level 0 (64x64): CML M=5 steps + NCA
- Level 1 (32x32): CML M=10 steps + NCA
- Level 2 (16x16, bottleneck): CML M=15 steps + full rescor_e3c (dual perception + 2-layer head)

**Forward pass:**

```
Encoder (downsample):
  z0 = sigmoid(proj(x))              # (B, 1, 64, 64)
  skip0 = nca_0([z0, cml_64(z0)])
  z1 = sigmoid(stride2_conv(z0))     # (B, 1, 32, 32)
  skip1 = nca_1([z1, cml_32(z1)])
  z2 = sigmoid(stride2_conv(z1))     # (B, 1, 16, 16)
  h2 = rescor_bottleneck(z2)         # full rescor_e3c at 16x16

Decoder (upsample):
  u1 = bilinear_up(h2, 32) + skip1
  h1 = nca_up_1(u1)
  u0 = bilinear_up(h1, 64) + skip0
  h0 = nca_up_0(u0)
  return clamp(output_proj(h0), 0, 1)
```

**Params:** ~4K trained, 36 frozen | **Target:** 64x64

**Rationale:** Natural resolution scaling. Bottleneck is literally existing rescor at 16x16 (proven). CML at each level operates at resolutions where its dynamics have been validated. Additive skip connections keep channels fixed.

---

## 4. `rescorformer` — Rescorformer (SWA + RoPE)

**Priority: 4**

Transformer block where FFN is replaced by rescor. Sliding Window Attention (SWA) for efficient local-global mixing over spatial patches. 2D RoPE for position encoding (0 extra params, resolution-agnostic).

**Config:**

- Patch size: 4x4 (64x64 input -> 256 patches/tokens)
- d_model=32, n_heads=4, d_k=d_v=8
- SWA window size: 8 patches (captures ~2-patch neighborhood in each direction)
- 2D RoPE: rotary embeddings applied independently to row/column patch indices
- L=1 block (single transformer layer)

**Forward pass:**

```
patches = patchify(x, 4)                     # (B, 256, 16)
tokens = linear_proj(patches) + 2D_RoPE      # (B, 256, 32), RoPE = 0 params

# SWA block
residual = tokens
tokens = layer_norm(tokens)
q, k, v = W_q(tokens), W_k(tokens), W_v(tokens)
# Apply RoPE to q, k
attn_out = sliding_window_attention(q, k, v, window=8)
tokens = residual + W_o(attn_out)

# Rescor FFN replacement
residual = tokens
tokens = layer_norm(tokens)
spatial = sigmoid(unpatch(unpatch_proj(tokens)))   # (B, 1, 64, 64)
stats = cml(spatial)
correction = nca([spatial, stats])
spatial_out = stats.last + correction
tokens = residual + repatch_proj(patchify(spatial_out))

out = unpatch(output_proj(tokens))            # (B, 1, 64, 64)
```

**Params:** ~7.4K trained, 12 frozen | **Target:** 64x64

**Rationale:** Clean separation of concerns: SWA handles inter-patch dependencies (global context), rescor handles intra-patch physics (local dynamics). 2D RoPE gives resolution-agnostic position encoding. SWA is O(N*W) not O(N^2). 250x param reduction per block vs standard ViT FFN.

---

## 5. `rescor_mamba` — Rescor + Mamba

**Priority: 5**

Mamba SSM for temporal dynamics (across K=5 frames), CML+NCA for spatial dynamics (within frame). Per-cell Mamba: each spatial position gets an independent 1D selective scan over its temporal history.

**Config:**

- Temporal: d_model=16, d_state=8, d_conv=4 (small SSM)
- Spatial: CML M=15 + NCA with hidden_ch=32
- Context: K=5 frames (current + 4 history)

**Forward pass:**

```
x_seq: (B, K=5, 1, H, W)

# Per-cell temporal modeling
x_flat = reshape(x_seq, (B*H*W, K, 1))
x_proj = linear(x_flat)                      # (B*H*W, K, 16)
mamba_out = mamba_block(x_proj)               # (B*H*W, K, 16)
temporal_feat = mamba_out[:, -1, :]           # (B*H*W, 16) last step

# Reshape back to spatial
spatial_feat = reshape(temporal_feat, (B, 16, H, W))

# CML+NCA on spatial features
cml_input = sigmoid(conv1x1(spatial_feat))    # (B, 1, H, W)
stats = cml(cml_input)
nca_in = cat([cml_input, spatial_feat, stats])  # 22ch
correction = nca(nca_in)
out = stats.last + correction
```

**Params:** ~9.2K trained, 140 frozen (CML 12 + Mamba A_log 128) | **Target:** 16x16, multi-frame

**Rationale:** First rescor variant with explicit temporal modeling. All current variants are single-step (frame t -> t+1). Mamba gives access to velocity, acceleration implicitly. Per-pixel independence means compute scales as O(H*W*K*d), same pattern as CML.

**Note:** Requires multi-frame training data pipeline (current benchmarks generate single-step pairs, not K-frame sequences).

---

## Summary


| Architecture              | Trained | Frozen | Target Res | CML Steps    | New Capability              |
| ------------------------- | ------- | ------ | ---------- | ------------ | --------------------------- |
| rescor_e3c (current best) | 4,641   | 12     | 16x16      | 15           | baseline                    |
| **rescor_mr**             | 13,281  | 48     | 16x16      | 60 (4x15)    | multi-physics               |
| **rescor_deep**           | 6,318   | 36     | 16x16      | 30 (15+10+5) | iterative refinement        |
| **rescor_ms**             | ~4,000  | 36     | 64x64      | 30 (5+10+15) | multi-resolution            |
| **rescorformer**          | ~7,400  | 12     | 64x64      | 10           | global attention (SWA+RoPE) |
| **rescor_mamba**          | ~9,200  | 140    | 16x16      | 15           | temporal dynamics           |


## 6. CML Scaling Ablation

**Priority: 0 (run alongside rescor_mr)**

The CML reservoir is entirely frozen, so scaling it costs **zero extra trained params**. The only cost is forward-pass FLOPS. This ablation systematically varies each CML dimension while keeping the NCA correction fixed (vanilla rescor, 321 trained params).

### Ablation Axes

**A. CML Steps (M) — iteration depth**
Controls how many logistic map iterations the reservoir runs. More steps = richer dynamics, larger effective receptive field (each step propagates info by kernel_size/2 pixels).


| M   | Effective RF     | Frozen Params | Notes                               |
| --- | ---------------- | ------------- | ----------------------------------- |
| 5   | ~11x11           | 12            | fast, shi kiallow dynamics          |
| 15  | ~31x31 (default) | 12            | current default, covers 16x16 grid  |
| 30  | ~61x61           | 12            | exceeds grid, dynamics may saturate |
| 50  | ~101x101         | 12            | deep chaos, test if more steps help |
| 100 | ~201x201         | 12            | extreme, diminishing returns?       |


**B. Coupling Kernel Size — spatial reach per step**
The depthwise conv that couples neighboring cells. Wider kernel = faster information propagation per step.


| Kernel        | Frozen Params              | Notes                              |
| ------------- | -------------------------- | ---------------------------------- |
| 3x3 (default) | 12 (r,eps,beta + 9 kernel) | local coupling, proven             |
| 5x5           | 28                         | medium range                       |
| 7x7           | 52                         | wide coupling, fewer steps needed? |


**C. Multi-Channel CML — parallel dynamical trajectories**
Run the CML on C>1 channels simultaneously. Extra channels are initialized with random perturbations of the input. Each channel evolves independently through the logistic map with shared coupling, producing C parallel trajectories from slightly different initial conditions.


| Channels    | Frozen Params | NCA Input Channels | Notes                     |
| ----------- | ------------- | ------------------ | ------------------------- |
| 1 (default) | 12            | 6 (with stats)     | current                   |
| 4           | 12            | 24 (4x stats)      | 4 parallel trajectories   |
| 8           | 12            | 48                 | richer but NCA head grows |
| 16          | 12            | 96                 | diminishing returns?      |


Note: CML frozen params don't change because the kernel is depthwise (shared across channels). But the NCA perception conv grows: Conv2d(6*C, hidden, 3x3).

**D. Coupling Strength (eps) — diffusion rate**
Controls how much neighboring cells influence each other. Higher eps = stronger spatial mixing per step.


| eps            | Regime             | Notes                    |
| -------------- | ------------------ | ------------------------ |
| 0.05           | very weak coupling | cells nearly independent |
| 0.15           | weak               | slow diffusion           |
| 0.30 (default) | moderate           | balanced                 |
| 0.50           | strong             | fast diffusion           |
| 0.70           | very strong        | rapid homogenization     |
| 0.90           | near-full coupling | cells converge quickly   |


**E. Drive Strength (beta) — input anchoring**
Controls how much the original input is re-injected at each CML step. Higher beta = dynamics stay closer to input, lower beta = dynamics evolve more freely.


| beta           | Regime       | Notes                                       |
| -------------- | ------------ | ------------------------------------------- |
| 0.01           | free-running | CML nearly autonomous, forgets input        |
| 0.05           | weak drive   | slow anchoring                              |
| 0.15 (default) | moderate     | balanced                                    |
| 0.30           | strong drive | dynamics track input closely                |
| 0.50           | very strong  | CML output ≈ input (reservoir less useful?) |


**F. Logistic Map Parameter (r) — chaos depth**
Already explored in findings.md (Section 1). Key values:


| r              | Regime          | Effective Rank | Notes                            |
| -------------- | --------------- | -------------- | -------------------------------- |
| 3.57           | chaos onset     | 11             | barely useful                    |
| 3.70           | period-doubling | 51             | moderate                         |
| 3.85           | 3-cycle window  | ~80            | rich but structured              |
| 3.90 (default) | full chaos      | ~110           | current default                  |
| 3.95           | near-boundary   | ~120           | richest                          |
| 3.99           | deep chaos      | 130            | max richness, possibly too noisy |


### Completed Sweep Results

**F. r-sweep (complete, 2026-04-16)**


| r        | Heat (MSE↓) | GoL (Acc↑) | Gray-Scott (MSE↓) | KS (MSE↓)   | Rule110 (Acc↑) | Wireworld (Acc↑) |
| -------- | ----------- | ---------- | ----------------- | ----------- | -------------- | ---------------- |
| 3.57     | 3.26e-7     | 96.0%      | 7.39e-6           | **5.95e-7** | 96.9%          | 98.3%            |
| **3.70** | **2.47e-8** | 94.3%      | **4.49e-6**       | 3.92e-6     | 96.9%          | 98.3%            |
| 3.85     | 1.78e-7     | 89.0%      | 4.75e-6           | 7.12e-7     | 96.9%          | 98.2%            |
| 3.90     | 6.40e-8     | **96.0%**  | 5.52e-6           | 2.14e-6     | 96.9%          | 99.1%            |
| 3.95     | 4.93e-6     | 96.0%      | 6.15e-6           | 1.21e-6     | 96.9%          | **99.1%**        |
| 3.99     | 3.08e-7     | 95.2%      | 6.53e-6           | 2.76e-6     | 96.9%          | 99.0%            |


Findings: Heat/Gray-Scott prefer r=3.70 (period-doubling matches diffusion). KS prefers r=3.57 (chaos onset). GoL/Wireworld prefer r=3.90-3.95 (full chaos). Rule110 invariant. r=3.95 catastrophic for heat (200x worse than r=3.70).

**B. Kernel-sweep (complete, 2026-04-16)**


| Kernel | Heat (MSE↓) | GoL (Acc↑) | Gray-Scott (MSE↓) | KS (MSE↓)   | Rule110 (Acc↑) | Wireworld (Acc↑) |
| ------ | ----------- | ---------- | ----------------- | ----------- | -------------- | ---------------- |
| 3x3    | 1.71e-7     | **96.0%**  | **6.99e-6**       | 1.80e-6     | 96.9%          | **98.3%**        |
| 5x5    | **5.99e-8** | 95.7%      | 8.11e-6           | 3.18e-6     | 96.9%          | 98.3%            |
| 7x7    | 8.99e-7     | 95.7%      | 8.66e-6           | **5.13e-7** | 96.9%          | 98.3%            |


Findings: Heat prefers 5x5 (3x better, wider coupling matches diffusion). KS prefers 7x7 (3.5x better, longer-range correlations). Gray-Scott prefers 3x3 (sharp local gradients). GoL/Wireworld/Rule110 invariant. No single kernel dominates.

**C. Channel-sweep (complete, 2026-04-16)**


| Channels | Trained | Heat (MSE↓) | GoL (Acc↑) | Gray-Scott (MSE↓) | KS (MSE↓)   | Rule110 (Acc↑) | Wireworld (Acc↑) |
| -------- | ------- | ----------- | ---------- | ----------------- | ----------- | -------------- | ---------------- |
| **1**    | 321     | **7.77e-8** | 95.8%      | **4.15e-6**       | **1.04e-6** | **96.9%**      | **98.3%**        |
| 4        | 753     | 2.85e-6     | **96.0%**  | 1.73e-5           | 2.03e-6     | 96.9%          | 98.3%            |
| 8        | 1,329   | 1.66e-5     | 96.0%      | 1.03e-4           | 3.15e-6     | 96.5%          | 70.0%            |
| 16       | 2,481   | 1.40e-4     | 95.4%      | 1.71e-2           | 4.67e-6     | 96.9%          | 98.2%            |


Findings: More channels hurts. Single-channel CML wins 5/6 benchmarks. Noise injection poisons logistic map dynamics. Multi-r ensemble (rescor_mr) is the right diversity approach. Multi-channel CML is NOT recommended.

**D. Eps-sweep (complete, 2026-04-17)**


| eps  | Heat (MSE↓) | GoL (Acc↑) | Gray-Scott (MSE↓) | KS (MSE↓)   | Rule110 (Acc↑) | Wireworld (Acc↑) |
| ---- | ----------- | ---------- | ----------------- | ----------- | -------------- | ---------------- |
| 0.05 | **9.62e-8** | 95.6%      | 4.91e-6           | 4.29e-7     | 96.9%          | 98.3%            |
| 0.15 | 5.36e-7     | 94.5%      | **3.44e-6**       | **3.33e-9** | 96.9%          | 98.3%            |
| 0.30 | 9.09e-8     | 95.6%      | 7.50e-6           | 5.17e-6     | 96.9%          | **99.1%**        |
| 0.50 | 1.52e-6     | 83.2%      | 8.95e-6           | 1.52e-6     | 96.9%          | **99.1%**        |
| 0.70 | 1.66e-6     | **95.8%**  | 1.36e-5           | 2.63e-6     | 96.9%          | 98.3%            |


Findings: KS at eps=0.15 = 3.33e-9 (1500x better than default). Weak coupling best for chaotic PDEs. GoL catastrophic at eps=0.50. Wireworld prefers eps=0.30-0.50.

**E. Beta-sweep (complete, 2026-04-17)**


| beta | Heat (MSE↓) | GoL (Acc↑) | Gray-Scott (MSE↓) | KS (MSE↓)   | Rule110 (Acc↑) | Wireworld (Acc↑) |
| ---- | ----------- | ---------- | ----------------- | ----------- | -------------- | ---------------- |
| 0.01 | 2.69e-7     | 94.9%      | 5.19e-6           | **1.50e-8** | 96.9%          | 70.0%            |
| 0.05 | 1.85e-7     | 94.8%      | 4.37e-6           | 1.15e-7     | 96.9%          | 98.3%            |
| 0.15 | **4.23e-8** | 95.7%      | **3.13e-6**       | 1.99e-6     | 96.9%          | **99.0%**        |
| 0.30 | 2.49e-6     | 95.3%      | 2.08e-5           | 1.61e-6     | 96.9%          | 99.0%            |
| 0.50 | 2.40e-6     | **95.8%**  | 2.02e-5           | 1.27e-6     | 96.9%          | 98.3%            |


Findings: KS beta=0.01 best (133x better, free-running chaos). Heat/GS prefer default beta=0.15. Wireworld catastrophic at beta=0.01 (70%). Combined optimal KS: (r=3.57, eps=0.15, beta=0.01).

**Learned Gate Ablation (complete, 2026-04-17)**


| Model           | Params | Heat (MSE↓) | GoL (Acc↑) | GS (MSE↓)   | KS (MSE↓)   | Rule110 | Wireworld |
| --------------- | ------ | ----------- | ---------- | ----------- | ----------- | ------- | --------- |
| rescor (frozen) | 321    | **8.56e-8** | 95.9%      | 7.03e-6     | 3.91e-6     | 96.9%   | **99.1%** |
| gate_static     | 341    | 7.27e-7     | 95.9%      | **6.72e-6** | **1.90e-7** | 96.9%   | 98.3%     |
| gate_dynamic    | 341    | 3.31e-6     | 95.4%      | 1.00e-5     | 7.70e-7     | 96.9%   | 98.3%     |


Findings: Static gate helps KS 20x but hurts heat/wireworld. Dynamic gate worse than static everywhere. 20-param gate not expressive enough to find sweep-optimal values. Root cause: gradient through 15 chaotic CML steps (Lyapunov 0.642 * 15 ~ 15,000x amplification) drowns the gate signal — Mikhaeil et al. NeurIPS 2022. See findings.md Section 44.

**Gate Init Diagnostic (complete, 2026-04-17)** — findings.md Section 45

Initialized CML2DLearnedGateStatic bias at sweep-optimal per benchmark. KS: optimal init wins 42x and gate stays near init (stable basin). Heat / Gray-Scott: gate DRIFTS away even when started at optimum — gradient through chaos actively corrupts the gate. Continuous learning unsalvageable by better init for most benchmarks.

**Discrete Selection Gate — BROKEN (complete, 2026-04-17)** — findings.md Section 46

`CML2DDiscreteSelect`: K=5 candidate (eps, beta) pairs, softmax weights, then `eps = sum(w_k * eps_k)` and ONE CML pass. The softmax was a reparameterization, not a gradient firewall — `d(cml_out)/d(eps)` still explodes through chaos. Result: gate stayed at default with 65-77% weight on ALL benchmarks. KS 2.7x (vs sweep's 1500x).

**Multi-Config CML — FIXED (complete, 2026-04-18)** — findings.md Section 47

`CML2DMultiConfig`: K=3 separate CML forward passes with fixed (eps, beta): (0.15, 0.01), (0.30, 0.15), (0.50, 0.30). Each CML under `torch.no_grad()`, outputs detached, blended with learned softmax weights. Gradient to logits bypasses CML interior entirely.


| Model                    | Params | Heat        | GoL   | GS          | KS          | Rule110 | Wireworld |
| ------------------------ | ------ | ----------- | ----- | ----------- | ----------- | ------- | --------- |
| rescor                   | 321    | **4.02e-8** | 95.7% | 6.34e-6     | 4.33e-6     | 96.9%   | **99.1%** |
| discrete_global (broken) | 326    | 1.35e-7     | 96.0% | 8.28e-6     | 1.64e-6     | 96.9%   | 98.3%     |
| multi_config (fixed)     | 324    | 2.63e-6     | 96.0% | **3.05e-6** | **5.94e-7** | 96.9%   | 98.2%     |


Learned selections: heat (0.50, 0.30) 80% — WRONG candidate, 65x regression. gol default 83%. gs blend (0.50, 0.30)/(0.30, 0.15) — 2x better. ks (0.30, 0.15) 62% — 7x better. wireworld even split, slight regression.

**Key insight:** clean gradient flow (detached CML outputs) is **necessary but not sufficient** — discrete selection with broken gradient (sum of weighted scalars) doesn't help; multi_config with detached CML outputs DOES help on KS / Gray-Scott but picks the wrong candidate on heat because the K=3 set misses heat-optimal (0.05, 0.15) and Gray-Scott-optimal (0.15, 0.15). Candidate set coverage matters. Continuous per-cell (eps, beta) learning is ruled out; future work must either (a) expand the candidate pool to span sweep optima per benchmark, or (b) move to a rescor_mr-style ensemble that also varies r.

**Multi-Config CML K=5 + Warm-Start — CURRENT BEST (complete, 2026-04-18)** — findings.md Section 48

Script: `experiments/multi_config_k5_ablation.py`. K=5 candidate pool spans sweep optima: (0.05, 0.15), (0.15, 0.01), (0.15, 0.15), (0.30, 0.15), (0.50, 0.30). Global scalar softmax with warm-start (+2.0 logit on idx 3 default — init 65% / 9% / 9% / 9% / 9%). 326 trained params, detached CML outputs.


| Model               | Params  | Heat               | GoL   | GS                 | KS                  | Rule110 | Wireworld |
| ------------------- | ------- | ------------------ | ----- | ------------------ | ------------------- | ------- | --------- |
| rescor              | 321     | 5.35e-7            | 95.3% | 7.11e-6            | 6.02e-6             | 96.9%   | 98.3%     |
| **multi_config_k5** | **326** | **8.84e-8 (6.1x)** | 94.8% | **2.77e-6 (2.6x)** | **2.83e-7 (21.3x)** | 96.9%   | **99.0%** |


Wins 4/6 benchmarks, no major regressions. Argmax stays at default (0.30, 0.15) with 53-82% weight on ALL benchmarks — the wins come from the SOFT BLEND of the 9% tails, not from correct argmax. Warm-start is sticky; the gate never commits to non-default candidates. K=5 fixes the heat regression from K=3 (2.63e-6 -> 8.84e-8) because (a) heat-optimal is on the menu and (b) warm-start prevents collapse onto (0.50, 0.30).

**Concat-K=5 Diversity Ablation — WORSE THAN BLEND (complete, 2026-04-20)** — findings.md Section 49

Script: `experiments/concat_k5_ablation.py`. Same K=5 candidate pool, but CONCATENATE all 5 CML outputs as 5 channels into NCA instead of softmax blending. NCA input = 1 + 5 = 6 channels. ~947 trained params (3x multi_config_k5). Tests the diversity hypothesis.


| Benchmark  | rescor  | concat_k5 | multi_config_k5 | Winner                           |
| ---------- | ------- | --------- | --------------- | -------------------------------- |
| heat       | 2.06e-7 | 5.57e-6   | 8.84e-8         | multi_config >> rescor > concat  |
| gol        | 95.98%  | 95.98%    | 94.84%          | tie (rescor/concat)              |
| gray_scott | 2.88e-6 | 9.94e-6   | 2.77e-6         | multi_config ~= rescor > concat  |
| ks         | 1.07e-5 | 1.02e-6   | 2.83e-7         | multi_config >> concat >> rescor |
| rule110    | 96.93%  | 96.99%    | 96.93%          | tie                              |
| wireworld  | 99.14%  | 99.02%    | 99.02%          | rescor (marginal)                |


Concat is WORSE than softmax blend on 4/6 benchmarks despite 3x the params. Softmax blend beats concat on heat by 63x and KS by 3.6x. Diversity hypothesis partially wrong — giving the NCA all 5 CML outputs as input channels hurts on heat/GS/wireworld. The scalar softmax blend produces a single "compromise" CML output that's easier for the NCA to correct than 5 separate inconsistent trajectories. The softmax blend is the load-bearing mechanism, not diversity.

**Final learnable-CML verdict:** multi_config_k5 + warm-start is the best variant found (326 params, wins 4/6 vs rescor). Softmax blend is load-bearing; diversity-only (concat) is worse; continuous / per-cell learning is ruled out.

> **HERO STATUS DEMOTED 2026-04-22.** The subsections below previously designated `rescor_mr_uniform K=32` / `rescor_rens K=32` as the project hero based on a single-seed 5W/1T/0L vs rescor result. A 3-seed replication at seeds 42/43/44 (experiments/rens_k32_multiseed.py, results/rens_k32_multiseed.json) failed to reproduce those numbers. Honest 3-seed verdict: **2W (gol, ks) / 2T (rule110, wireworld≈) / 2L (heat, gs) vs rescor**. Four of the five claimed wins were favorable RNG draws. The prior single-seed results below are preserved as project history but must be read as "single-seed; not replicated at seeds 43, 44." See findings.md Section 57.

> **UPDATE 2026-04-24 — hero PARTIALLY restored under honest 100-epoch protocol.** Phase 1 (experiments/phase1_honest_baseline.py, results/phase1_honest_baseline.json) ran rescor_rens K=32, rescor_rens_stat_full, rescor_rens_stat_no_var at 3 seeds × 100 epochs × 6 benchmarks = 54 runs. Under the new 3-seed × 100-epoch protocol (adopted from the §57 "next actions" list), rens K=32 recovers to **3W/1T/2L vs rescor AND 3W/1T/2L vs k5 oracle by medians** — wins heat (5.75e-8, 1.5× k5), gol (95.95%), and gs (2.20e-6, 1.3× k5 — the GS gap closes). The 30-epoch demotion was largely a compute-budget artifact. Three complementary specialists emerge (rens → smooth PDEs; stat_full → complex-spatial CAs, best on wireworld; stat_no_var → chaotic targets requiring tight variance, best on ks AND tightest distribution, 4W/1T/1L vs k5). Caveats preserved: rens still loses wireworld (97.67% vs stat_full 98.73%); rens still loses ks (2.11e-7 vs stat_no_var 1.07e-7); the §53 "18× better than k5 on heat" legacy claim is permanently dead. The 2026-04-22 demotion callout above is **kept as project history** — the §57 reading was correct for 30-epoch data; the §58 partial restoration does not retroactively resurrect §53's single-seed claims. See findings.md Section 58.

**Random-Coupling Reservoir — ORACLE-FREE VARIANT (complete, 2026-04-20)** — findings.md Section 50

Script: `experiments/random_reservoir_ablation.py`. Follow-up to multi_config_k5 (current best) designed to strip its two oracle heuristics: (i) K=5 hand-picked (eps, beta) candidates placed at per-benchmark sweep optima, and (ii) warm-start of the softmax gate on idx 3 (default).

`CML2DRandomReservoir`: K=8 frozen reservoirs, each with a DIFFERENT random coupling conv kernel (distinct RNG seeds), SHARED default (eps=0.30, beta=0.15) across all K, softmax gate with NO warm-start (uniform init). Detached outputs, same gradient-isolation pattern. 329 trained params (1ch), 75 frozen.

Naming note: **rescor_esn** (Echo State Network) is the canonical name for A-full. Both names refer to K frozen tanh reservoirs with random coupling + softmax gate, no oracle, no warm-start. A-full is an alias; use rescor_esn going forward.

Two sub-modes:

- **A-preserved**: keeps logistic map f(x)=r*x*(1-x); only the coupling kernel is randomized.
- **A-full** (= **rescor_esn**): drops logistic, uses tanh-based ESN-style recurrence on [-1, 1]-centered grid. No physics-specific nonlinearity.


| Model             | Params  | Oracle?  | Heat        | GoL    | GS          | KS          | Rule110 | Wireworld  |
| ----------------- | ------- | -------- | ----------- | ------ | ----------- | ----------- | ------- | ---------- |
| rescor            | 321     | no       | 5.35e-7     | 95.32% | 7.11e-6     | 6.02e-6     | 96.93%  | 98.26%     |
| multi_config_k5   | 326     | yes (x2) | **8.85e-8** | 94.84% | **2.77e-6** | **2.83e-7** | 96.93%  | 99.02%     |
| A-preserved       | 329     | no       | 8.15e-6     | 91.60% | 1.63e-5     | 1.72e-6     | 96.93%  | 98.26%     |
| **A-full (tanh)** | **329** | **no**   | 1.29e-6     | 94.88% | 4.52e-6     | 1.14e-6     | 96.93%  | **99.13%** |


Key: A-full strictly beats A-preserved on 5/6, ties rule110 — zero losses. The logistic map is NOT load-bearing under random coupling; generic tanh reservoir wins. A-full beats rescor 3W/2T/1L with ZERO oracle knowledge (gray_scott 1.57x, KS 5.28x, wireworld 99.13% > 98.26%; loses only on heat 2.4x). A-full beats k5 on wireworld. k5's oracle advantage still buys something on chaotic PDEs (heat 14.5x, GS 1.6x, KS 4.0x over A-full), but on discrete-CA benchmarks A-full matches or beats k5. Gate entropy stays near ln(8) — same soft-blend pattern.

**Random-reservoir sub-sequence (2026-04-20):** A-preserved (logistic + random kernel, oracle-free) → A-full / rescor_esn K=8 (tanh + random kernel, oracle-free) → **B = "random-k5"** (logistic + fixed 3x3 kernel + random (eps, beta) candidates drawn from [0,0.8]x[0,0.5], uniform gate init — isolates the oracle variable in multi_config_k5; 2W/1T/3L vs rescor, sits between A-preserved and A-full) → **D = A-full + input-conditioned gate** (tiny hypernet maps per-sample [mean, var, grad-norm] → K perturbation logits added to a learnable global bias; 104 extra hypernet params; NEGATIVE RESULT — strictly worse than A-full on every benchmark that completes, crashes on KS (NaN) and rule110 (BCE input outside [0,1]) even with zero-init hypernet and logit clamp) → **rescor_esn K-scaling {K=8, K=16, K=32}** (scale-pilled thesis test; see below).

**rescor_esn K-Scaling Ablation (complete, 2026-04-20)** — findings.md Section 52

Script: `experiments/cml_scaling_ablation.py`. Results: `experiments/results/cml_scaling_ablation.json`. Ablates K ∈ {8, 16, 32} for rescor_esn, vs rescor and multi_config_k5 baselines, 30 epochs, grid=16, seed=42.

Required a forward-pass vectorization: Python K-loop in `CML2DRandomReservoir._run_batched` replaced with a single grouped `F.conv2d(..., groups=K*C)` per step. Dropped K=8 forward pass from ~1.3s/batch to ~19ms, made K=32 feasible. First-8-kernel invariance verified (K=16 shares identical kernels 0-7 with K=8, bit-identical outputs on those indices).


| Benchmark  | rescor  | k5 (oracle) | esn K=8    | esn K=16 | esn K=32   | Best K              |
| ---------- | ------- | ----------- | ---------- | -------- | ---------- | ------------------- |
| heat       | 5.35e-7 | **8.85e-8** | 1.29e-6    | 1.08e-6  | 2.56e-6    | K=16                |
| gol        | 95.32%  | 94.84%      | 94.88%     | 94.97%   | **95.37%** | K=32 (beats rescor) |
| gray_scott | 7.11e-6 | **2.77e-6** | 4.52e-6    | 5.67e-6  | 4.32e-6    | K=32                |
| ks         | 6.02e-6 | **2.83e-7** | 1.14e-6    | 1.56e-6  | 1.29e-6    | K=8                 |
| rule110    | 96.93%  | 96.93%      | **96.93%** | 78.83%   | **96.93%** | K=8 / K=32          |
| wireworld  | 98.26%  | **99.02%**  | **99.13%** | 98.26%   | 98.27%     | K=8                 |


Gate entropies vs max uniform ln(K) ∈ {2.08, 2.77, 3.47}:

- K=8: 1.85–2.08 (near uniform).
- K=16: 1.67–2.77 (mixed; KS committed at 0.62 top weight).
- K=32: 2.35–3.47 (near max uniform except KS 2.35 / top 0.47).

Runtimes (post-vectorization): K=16 ~32 min, K=32 ~55 min.

**Verdict: K-scaling curve is non-monotonic; optimal K is benchmark-specific.**

- **Scale-pilled thesis FALSE in strict form.** K=16 is the worst of the three on 4/6 benchmarks. More reservoirs is NOT monotonically better within 30 epochs.
- **K=16 rule110 catastrophe** (96.93% → 78.83%, −18pp) while K=32 recovers cleanly. Likely unlucky kernel draws at indices 8-15 that K=32 samples past — but single-seed, multi-seed replication needed before treating as structural.
- **K=32 claims gol outright** (95.37% > rescor 95.32% > k5 94.84%) — first oracle-free win on gol vs rescor.
- **Interior optimum on heat**: K=16 > K=8 > K=32 (U-curve).
- **Learning-horizon hypothesis**: gate entropy stays at or near ln(K) max uniform at K=16 / K=32. With each reservoir contributing ~1/K at init, the per-reservoir gradient is O(1/K) and the gate can't commit in a 30-epoch budget. Likely what caps the scale.
- **k5's oracle advantage on chaotic PDEs is not closed by scaling K** (13x heat, 1.56x gs, 4.0x ks over best esn K). Discrete CAs (gol, rule110, wireworld): rescor_esn matches or beats k5.
- **Matching Principle at the architectural scale**: the right reservoir-pool size depends on target dynamics (heat K=16, gol K=32, gs K=32, ks K=8, rule110 K=8/32, wireworld K=8). No single K dominates.

Next natural moves: multi-seed replication of K=16 rule110 (single highest-value follow-up); longer training budget to test the learning-horizon hypothesis; hybrid candidate pool (oracle-placed + random-coupling reservoirs); test-time candidate search.

**Uniform Gate + Multi-r Chaos-Depth Ablation (complete, 2026-04-20)** — findings.md Section 53

Script: `experiments/uniform_and_mr_ablation.py`. Results: `experiments/results/uniform_and_mr_ablation.json`. 54-cell ablation testing two orthogonal axes on top of rescor_esn: (i) uniform gate (frozen 1/K averaging, zero trainable gate params), (ii) rescor_mr (K vanilla CMLs with K different r values linearly spaced over [3.57, 3.99] — chaos-depth diversity instead of random-coupling diversity). Registered variants:

- **rescor_esn_uniform**: rescor_esn with `gate_mode="uniform"`. 321 trained params (IDENTICAL to vanilla rescor).
- **rescor_mr**: K CMLs with logistic map, shared 3x3 coupling + shared (eps, beta), different r per reservoir. Learned softmax gate. 329 trained params.
- **rescor_mr_uniform**: rescor_mr with `gate_mode="uniform"`. 321 trained params.

Code hooks: `CML2DRandomReservoir` gained `gate_mode={"learned","uniform"}`; new `CML2DMultiR` class in `src/wmca/modules/hybrid.py`; three new entries (`rescor_esn_uniform`, `rescor_mr`, `rescor_mr_uniform`) in `model_registry.py`.

Headline results (vs k5 oracle and rescor, grid=16, 30 epochs, seed=42):


| Benchmark  | rescor  | k5 (oracle) | Best oracle-free                                                                                                                                | Winner                                                           |
| ---------- | ------- | ----------- | ----------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| heat       | 5.35e-7 | 8.85e-8     | **mr_uniform K=32 → 4.87e-9 (18x k5) [single-seed; not replicated at seeds 43, 44 — multi-seed mean 8.86e-7 ± 1.03e-6, see 2026-04-22 update]** | oracle-free (single-seed only; flipped to LOSS under multi-seed) |
| gol        | 95.32%  | 94.84%      | mr_learned K=16 → 96.01%                                                                                                                        | oracle-free (+1.17pp)                                            |
| gray_scott | 7.11e-6 | **2.77e-6** | esn_uniform K=32 → 3.87e-6                                                                                                                      | **oracle (the one holdout)**                                     |
| ks         | 6.02e-6 | 2.83e-7     | mr_learned K=16 → 2.39e-7                                                                                                                       | oracle-free (1.18x)                                              |
| rule110    | 96.93%  | 96.93%      | tie                                                                                                                                             | tie                                                              |
| wireworld  | 98.26%  | 99.02%      | mr_learned K=32 / mr_uniform K=8 → 99.89%                                                                                                       | oracle-free (+0.87pp)                                            |


**Verdict (single-seed; not replicated at seeds 43, 44): rescor_mr_uniform K=32 was designated current hero.** 321 trained params (identical footprint to vanilla rescor), zero gate, zero oracle, zero warm-start — just K=32 frozen chaos-depth-diverse reservoirs averaged 1/K. Single-seed vs rescor: 5W/1T/0L. Single-seed vs k5: 3W/1T/2L. Single-seed heat = 4.87e-9 (18x better than the prior oracle champion).

**UPDATE 2026-04-22 — hero designation withdrawn.** Multi-seed replication (seeds 42/43/44, same code, 30 epochs, grid=16) gives honest verdict **2W / 2T / 2L vs rescor**: heat 8.86e-7 ± 1.03e-6 (LOSS, 1.7× worse than rescor, 10× worse than k5), gol 95.76% ± 0.23pp (WIN), gs 1.60e-5 ± 1.30e-5 (LOSS, 2.3× worse than rescor, 5.8× worse than k5), ks 1.81e-6 ± 1.62e-6 (WIN), rule110 96.94% (tie ceiling), wireworld 98.20% ± 0.52pp (marginal loss/tie). Per-seed heat spans ~23× across seeds (2.31e-6 / 2.46e-7 / 1.01e-7); the stored 4.87e-9 is 20× below the best of the three replication seeds. Four of the five S53 claimed wins were favorable RNG draws. See findings.md Section 57.

Key axis findings:

- **Removing the gate wins on ESN at matched K.** esn_uniform beats esn_learned on 5/6 benchmarks. The S52 K=16 rule110 catastrophe (78.83%) is explained: gate-commitment problem, not fundamental scaling — uniform 1/K averaging recovers to 96.93%. Strongly supports the S52 learning-horizon hypothesis (O(1/K) per-reservoir gradient starves the gate in 30 epochs).
- **MR's learned gate is benchmark-dependent** (unlike ESN). Helps on ks (commits toward low-r, matching S38 optimum) and on gol/wireworld; hurts on heat/gs.
- **Diversity-axis Matching Principle.** Chaos-depth (MR) wins heat/ks/wireworld. Random coupling (ESN) wins gs. GoL split. Rule110 invariant. No single diversity axis dominates.
- **Oracle-free closes the chaotic-PDE gap on heat.** The MR pool [3.57, 3.99] linearly spaced includes r values near the heat-optimum 3.70 (S38); averaging does a soft sweep without needing the answer up front. First time "the sweep itself is the reservoir" produces strict improvement over oracle placement.
- **Parameter cost.** Uniform variants have 321 trained params regardless of K. Frozen compute grows with K, optimizer surface doesn't. Cleanest "free capacity" result we have.

**Sequence of learnable/gate variants tried:** rescor (baseline) → gate_static/dynamic (continuous, broken) → optimal-init gate (broken except KS) → discrete_select (broken gradient) → multi_config_k3 (clean gradient, candidate gap) → multi_config_k5 + warm-start (2 oracles, beat on 4/6 by new champion) → concat_k5 (worse than blend) → random_reservoir A-preserved (oracle-free, strictly worse than rescor on 3/6) → random_reservoir A-full / rescor_esn K=8 (oracle-free, beats rescor 3W/2T/1L) → B = random-k5 (oracle variable isolated; 2W/1T/3L) → D = A-full + input-conditioned gate (NEGATIVE, crashes on KS/rule110) → rescor_esn K-scaling {8,16,32} (non-monotonic; K=16 rule110 catastrophe; scale-pilled thesis false in strict form) → **rescor_esn_uniform / rescor_mr / rescor_mr_uniform K∈{8,16,32} — single-seed claim: oracle-free beats oracle on 4/6; rescor_mr_uniform K=32 (= rescor_rens K=32) was designated CURRENT HERO on single-seed evidence. DEMOTED 2026-04-22 under multi-seed replication (seeds 42/43/44, 30 epochs): honest verdict 2W/2T/2L vs rescor, rens LOSES to k5 on heat (10×) and ks (6.4×). Not replicated at 30 epochs.** → **rescor_hybrid K∈{16,32} (MR+ESN uniform average, NEGATIVE — dilution hypothesis confirmed, never the single best, GS actively worse than either pure axis)** → **rescor_rens_deep L=2 (depth via stacked chaotic stages, ABORTED NEGATIVE — heat 1.01e-5 ~2000x worse than L=1, gol 77.31% −18.67pp; residuals compound at depth; L=3 killed early)** → **rescor_rens_stat_full / rescor_rens_stat_no_var (stat-bank NCA with mean+var+min+max over the K=32 rens bank, 753 / 609 trained params, NEGATIVE at 30 epochs — strictly worse than rens K=32 on heat/gs by ~3 orders at 30 epochs, 1W/1T/4L for C_full and 2W/1T/3L for C_no_var; heat epoch diagnostic at {30,60,100,150} shows it's optimization-limited not structural; Phase 2 A/B deeper-NCA DEFERRED until methodology fix)** → **Phase 1 Honest Baseline 2026-04-24 (rens K=32, stat_full, stat_no_var at 3 seeds × 100 epochs × 6 benchmarks = 54 runs, experiments/phase1_honest_baseline.py, results/phase1_honest_baseline.json): HERO PARTIALLY RESTORED. rens K=32 recovers to 3W/1T/2L vs rescor AND 3W/1T/2L vs k5 oracle by medians — wins heat (5.75e-8, 1.5× k5), gol (95.95%), and gs (2.20e-6, 1.3× k5 — GS gap closes). The 30-epoch demotion was largely a compute-budget artifact. Three complementary specialists emerge: rens (smooth PDEs), stat_full (complex-spatial CAs, best on wireworld 98.73%), stat_no_var (chaotic targets, best on ks 1.07e-7 AND tightest distribution, 4W/1T/1L vs k5). Caveats preserved: rens still loses wireworld at 100 epochs (overfitting asymmetry); rens still loses ks to stat_no_var; the 18× heat legacy claim is permanently dead (honest number is 1.5×). Not a full restoration — a partial reversal with an asterisk.** Next steps (updated 2026-04-24 after Phase 1 100-epoch re-validation):

1. **Three-specialists investigation.** Why does rens-mean-only beat mean+var+min+max on heat/gol/gs but lose wireworld to stat_full? Hypothesis: smooth-PDE targets are well-predicted from the ensemble mean alone; complex-spatial CAs benefit from variance as a local-disagreement signal. Probe: per-channel gradient contribution in the first NCA conv, averaged over training, across benchmarks.
2. **ks-variance tradeoff.** Why does adding var hurt ks stability (stat_full bimodal with seed=43 outlier vs stat_no_var tight 8.13e-8 – 1.97e-7)? Probe: train stat_full on ks with the var channel frozen to its initial value vs learned; identify whether the learning dynamics are the source of bimodality.
3. **Learned variance gate.** One scalar gate on the var input channel (not CML interior, clean gradient). Tests whether the NCA can learn per-benchmark var utility — unifies the three specialists under a single variant.
4. **Per-benchmark epoch budget for rens-on-wireworld.** Validate the 30-epoch asymmetry: run rens K=32 on wireworld at {30, 50, 70, 100} and see whether regression is monotonic or cliff-shaped; if cliff-shaped, consider early-stopping on discrete-CA targets.
5. **Multi-seed replication (seeds 42/43/44, 100 epochs) of other prior "hero" candidates** — k5 oracle, multi_config_k5, rescor_esn K∈{8,16,32}, rescor_hybrid — under the new Phase 1 protocol, to check whether any of THEIR §57-demoted claims also partially recover. Lower priority than items 1–4 (less likely to produce new architectural wins given Phase 1 has already re-established rens as hero), but required before any of those prior single-seed claims can be cited.
6. (Still open, de-prioritized) Git-history check on the rens K=32 path between the original single-seed run and the 2026-04-22 replication — low value now that Phase 1 has re-established rens at 100 epochs; any residual code drift would show up in the Phase 1 numbers. GS-specific structured coupling priors — now a separate lever for higher-res benchmarks, not urgent (rens K=32 beats k5 on gs at 100 epochs). "Deepen the NCA, not the CML" (single rens K=32 stage + deeper learned NCA) — still a reasonable next architectural move, now targetable at 100 epochs under the Phase 1 protocol.

**DONE (2026-04-24):** 100-epoch multi-seed replication of rens K=32 — completed under Phase 1. This item is dropped from the open-work list.

**Methodology caveat (from S56 heat epoch diagnostic, 2026-04-22):** any future ablation that widens the NCA input or the NCA learned surface (extra stats, extra candidates, extra auxiliary channels, deeper NCA) must use **≥60 training epochs or zero-initialize the extra-channel weights of the first NCA conv**. At 30 epochs, a 5-channel-input NCA fails to drive mixed-information extra channels toward their optimal (near-zero for unhelpful channels) — we're measuring optimization efficiency, not representational capacity. Concretely: rescor_rens_stat_full at 30 epochs = 6.04e-6 on heat (1240× worse than rens K=32's single-seed value) but at 150 epochs = 1.18e-7 (k5-oracle-class). **rens K=32 was designated hero at matched budget on single-seed evidence; see the 2026-04-22 multi-seed demotion above — the 4.87e-9 used as the "rens K=32" baseline in this caveat did NOT replicate (multi-seed mean 8.86e-7). The 30-epoch wider-input epoch-tax argument still stands, but "rens K=32 at 30 epochs" is not a stable reference point and the stat_full-vs-rens ratios cited here are computed against an outlier seed-42 number.**

**Hybrid MR+ESN Ablation (complete, 2026-04-21)** — findings.md Section 54

Script: `experiments/hybrid_ablation.py`. Results: `experiments/results/hybrid_ablation.json`. Tests the S53 "next move" of mixing chaos-depth (MR) and random-coupling (ESN) reservoirs in one frozen bank under uniform 1/K averaging, specifically to close the gray_scott gap vs k5 (the one benchmark oracle still won in S53).

- **rescor_hybrid**: new class `CML2DHybridMrEsn` in `src/wmca/modules/hybrid.py`. K_mr logistic CMLs (shared 3x3 coupling, K_mr r values linearly spaced over [3.57, 3.99]) + K_esn tanh ESN reservoirs (random per-reservoir kernels, shared eps=0.30/beta=0.15). Both banks under `torch.no_grad()`, outputs concatenated along K-axis, uniform 1/K averaging with K = K_mr + K_esn. Zero trainable gate. 321 trained params (IDENTICAL to vanilla rescor). Registered as `rescor_hybrid` with `cml_gate="hybrid_mr_esn"` in `model_registry.py`.
- Tested two configs: K=16 (8 MR + 8 ESN) and K=32 (16 MR + 16 ESN). 30 epochs, grid=16, seed=42, all 6 benchmarks, ~2h wall clock.

Results (vs the key baselines):


| Benchmark  | rescor  | k5          | mr_u K=32   | esn_u K=32 | hybrid K=16 | hybrid K=32 |
| ---------- | ------- | ----------- | ----------- | ---------- | ----------- | ----------- |
| heat       | 5.35e-7 | 8.85e-8     | **4.87e-9** | 2.55e-6    | 1.63e-7     | 3.99e-7     |
| gol        | 95.32%  | 94.84%      | 95.98%      | 95.79%     | 95.86%      | 95.96%      |
| gray_scott | 7.11e-6 | **2.77e-6** | 4.34e-6     | 3.87e-6    | 9.43e-6     | 9.67e-6     |
| ks         | 6.02e-6 | 2.83e-7     | 2.42e-6     | 1.70e-6    | 5.62e-6     | **2.68e-7** |
| rule110    | 96.93%  | 96.93%      | 96.93%      | 96.93%     | 96.93%      | 96.93%      |
| wireworld  | 98.26%  | 99.02%      | 99.11%      | 98.25%     | 98.25%      | 99.13%      |


**Verdict: NEGATIVE RESULT. Dilution hypothesis confirmed. (At single-seed, mr_uniform K=32 remained hero — see 2026-04-22 update above; multi-seed replication has since demoted that claim. Hybrid is still strictly worse than both pure axes regardless.)**

- Hybrid K=32 is NEVER the single best cell: 82x worse than mr_uniform K=32 on heat; GS worse than BOTH pure axes and even worse than rescor baseline (hypothesis target FAILED); KS beats k5 but loses to mr_learned K=16; gol/rule110/wireworld tie.
- **Averaging across different diversity axes dilutes both signals rather than capturing best-of-both.** MR's r-axis specialists are diluted by ESN reservoirs with no chaos-depth affinity; ESN's random-coupling diversity is diluted by MR's shared 3x3 kernel. Each pure axis is strong on its own turf; 1/K mixing across axes averages into mediocrity.
- Silver lining: hybrid K=32 does beat k5 on KS (2.68e-7 vs 2.83e-7, 1.06x) — the combined bank has useful diversity, but not enough to dominate any benchmark.

GS gap vs k5 (2.77e-6 oracle vs 4.34e-6 mr_uniform K=32) remains open; hybrid MR+ESN under uniform averaging is not the right lever. Next GS attempts should target structured coupling priors (diffusion-like kernels biasing the ESN bank) or a single learned scalar α weighting mean(MR) vs mean(ESN), not the uniform mix.

### Ablation Protocol

Run vanilla rescor (321 trained params) on all 16x16 benchmarks, varying one CML axis at a time while holding all others at defaults. Report MSE/accuracy per benchmark.

**Default CML config:** r=3.90, eps=0.30, beta=0.15, M=15, kernel=3x3, channels=1

**Sweeps (each on all benchmarks):**

1. M sweep: [5, 15, 30, 50, 100] — 5 runs
2. Kernel sweep: [3, 5, 7] — 3 runs
3. Channel sweep: [1, 4, 8, 16] — 4 runs (NCA head adjusted)
4. eps sweep: [0.05, 0.15, 0.30, 0.50, 0.70] — 5 runs
5. beta sweep: [0.01, 0.05, 0.15, 0.30, 0.50] — 5 runs
6. r sweep: [3.57, 3.70, 3.85, 3.90, 3.95, 3.99] — 6 runs

Total: 28 configs x N benchmarks. All configs have 321 trained params (except channel sweep, where NCA input changes).

---

## 7. SD VAE Comparison (GameNGen Apples-to-Apples)

**Priority: after CML ablation**

Use Stable Diffusion 1.4's VAE (the same encoder/decoder GameNGen uses) so the only variable is the world model itself.

**Setup:**

- Resize DOOM frames to 128x128
- SD VAE encoder (8x downsample): 128x128x3 -> **16x16x4 latent** (exactly our grid size)
- Train rescor on 16x16x4 latent prediction (in_ch=4+action, out_ch=4)
- Decode with SD VAE decoder: 16x16x4 -> 128x128x3
- Compute PSNR/SSIM/LPIPS vs ground truth

**Comparison:**

- GameNGen: U-Net diffusion (~860M params) on SD VAE latents
- Ours: rescor (~1-5K params) on the same SD VAE latents
- Same encoder, same decoder, same game. Pure world model comparison.

**Dependencies:**

```python
from diffusers import AutoencoderKL
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
# ~83M params, frozen, not counted in either model's budget
```

**Expected params:**

- rescor with in_ch=5 (4 latent + 1 action), out_ch=4: ~2-3K trained
- rescor_mp_gate: ~8-10K trained
- GameNGen U-Net: ~860M trained

---

## Implementation Order

1. **CML scaling ablation** — run alongside rescor_mr, zero new code needed (just config changes)
2. rescor_mr — works on all existing 16x16 benchmarks immediately
3. rescor_deep — same, extends proven vertical pattern
4. rescor_ms — needed for 64x64 push (Crafter real, APEBench higher res)
5. rescorformer — most novel, needs 64x64 benchmarks
6. rescor_mamba — needs multi-frame data pipeline changes

