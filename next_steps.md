# Next Steps — WMCA

**Status**: post-Atari-latent experiments (WMCA Plan 0 Path A complete), 2026-05-07.

### Atari Path A Outcome

ResCorRens K=32 vs ResCorMamba K=4 on Pong + Breakout latent dynamics. Full pipeline: 25K frames → GridAE → latents → rescor → rollout H=15/50/100.

- **ResCorRens K=32** is the reliable choice: Breakout solved (MSE 0.010 at all horizons), Pong drifts moderately (0.096 at H=100). 68 min training on MPS.
- **ResCorMamba K=4** has higher ceiling (3-4× better on Pong, best seed) but catastrophic seed variance (worst seed 165× ratio). 10h training on MPS.
- **Encoder PSNR regression** was a shuffle bug, not architecture — Breakout now 38.6 dB, Pong 33.1 dB.
- **VQ-VAE on Crafter** complete: 50K steps, 92.8% usage, 100K tokens encoded (93% usage).
- **DiscreteRescor smoke** passed on synthetic tokens. Module validated.

### Atari rollout results — final 3-seed median ratios

Headline ratio numbers from Path A.3 (3-seed median, 20 test trajectories, H=100 ratio = MSE(H=100) / MSE(step-1)):

| Variant | Bench | H=100 ratio | Verdict |
|---|---|---|---|
| rescor_rens K=32 | Breakout | **2.6×** | STABLE — passes `plans/plan_0.md` §A.3 PASS rule |
| rescor_rens K=32 | Pong | **36.8×** | chaotic — fails rule |
| rescor_mamba_rand K=4 | Pong | **28.6×** | 1.3× better than rens, still fails rule |
| rescor_mamba_rand K=4 | Breakout | **7.4×** | 2.9× worse than rens, MARGINAL |

**Bench-specific winner inversion**: rens wins Breakout long-horizon (wider K=32 reservoir matches the static block layout); mamba wins Pong long-horizon (K=4 SSM tracks the high-frequency ball trajectory). Neither variant is universally better. Echoes the gs/ks sprint pattern.

**Mamba per-seed variance on Pong H=100: 14×–165× spread.** Compared to rens K=32's <1.4× spread on the same data, this is the s43-style pathology recurring (TODO-F still pending; 165× outlier was seed 43). Median of 28.6× understates tail risk. Whenever mamba is summarized in any future writeup, include the per-seed spread, not just the median.

### Next decision point — what to run after Path A

The natural sequel: **run multistep H=8 (§66, the sprint's strongest synthetic-substrate variant) on Atari latents**. With Path A infrastructure now built, this is a ~half-day GPU follow-up. Two outcomes:

- **WIN: multistep H=8 fixes Pong H=100 on Atari** → first all-horizon stable variant on a non-synthetic substrate. Promote to Dreamer fork backbone candidate.
- **LOSS: multistep H=8 doesn't transfer** → multistep is gs/ks-specific. Same shape as Day 1 pushforward (gs LOSS, ks partial) but flipped. Pivot to Dreamer fork with rens K=32 as the Breakout-validated pragmatic backbone.

Either way, the cost is bounded (~half day GPU, ~$5). This is now the cheapest-informative move after Path A and before either Path C completion or Dreamer fork. Open follow-up: TODO-F-style diagnostic on the Pong H=100 mamba 165× outlier (seed 43) — re-run with a different optimizer-state seed to test whether the bad-rollout regime is initialization-basin-dependent. Half-day max, $0–$2 GPU; cheap and would explain a project-wide recurring pattern.

### Remaining: Path C (DiscreteRescor on real Crafter tokens)

- **Blocked on**: training script for DiscreteRescor on real token data
- **Data ready**: tokens.npy (99,999 × 16×16), next_tokens.npy, actions.npy
- **Module validated**: smoke test PASS on synthetic tokens
- **Next**: spawn agent to write + run `_cmdr_train_disc.py` → 3 seeds × 100ep → rollout probe

---

## Prioritized work items

### 0. **DiscreteRescor on real Crafter tokens** — *immediate next step*

- Train DiscreteRescor (vocab=512, n_actions=18, embed_dim=64) on 99,999 Crafter token pairs.
- 3 seeds × 100 epochs, batch=32, lr=1.4e-3 on MPS. Estimated ~2-4h.
- Smoke-validated module (loss ↓, no NaN) — low risk.
- Follow with rollout probe H=15/50/100 on discrete token sequences.
- **Why**: completes WMCA Plan 0 Path C. Only remaining blocker before full pipeline results.

### 1. **Final mamba vs rens comparison table** — *paper-ready summary*

- Build head-to-head table: ResCorRens K=32 vs ResCorMamba K=4 on Pong + Breakout.
- Metrics: step-1 MSE, H=15/50/100 MSE, ratio, cos_div, training time, param count.
- Include seed variance statistics (median + min/max range).
- Already have all raw numbers — just formatting.

### 2. **DreamerV3 fork decision** — *based on Path A results*

- Atari latent results inform the fork backbone choice: ResCorRens K=32 (reliable) vs ResCorMamba K=4 (high variance, better ceiling).
- See `dreamerv3_fork_plan.md` §5. Decision after Path C also complete.

### 3. **PMHN / Mamba-2 block upgrade** — *low priority optimization*

- Replace `MinimalMambaBlock` with Mamba-2 / Structured State Space Duality (SSD).
- 2-8× speedup over Mamba-1 scan. Makes CPU runs viable.
- ~2-3 days dev. Only if Mamba sees more use.

### 4. **Pixel-space scale-ups** (Tasks #28, #29)

- rescor_ms (U-Net 64×64) and rescorformer (SWA + RoPE 64×64).
- Targets Crafter Real pixel-space bottleneck.
- ~1 week each. Independent of latent work.

---

## Recommended sequence

```
0 (DiscreteRescor on Crafter tokens, ~2-4h MPS)
├── If done → 1 (comparison table) → update docs → 2 (fork decision)
└── Either way → 3/4 (scale-ups, post-sprint)
```

**My read**: finish Path C (step 0) to close out WMCA Plan 0. Then document everything and decide on Dreamer fork. Scale-ups are parallelizable with a GPU pod.

---

## Completed (WMCA Plan 0, Path A)

| Component | Status | Artifacts |
|-----------|--------|-----------|
| Atari data | ✅ | 25K frames each (Pong + Breakout) |
| Atari encoder | ✅ | Breakout 38.6 dB, Pong 33.1 dB (after shuffle fix) |
| Atari latents | ✅ | 25K latents each, diversity verified |
| VQ-VAE | ✅ | 50K steps, 92.8% usage |
| VQ-VAE tokens | ✅ | 99,999 tokens (16×16, 93% usage) |
| DiscreteRescor smoke | ✅ | Loss ↓, no NaN, 4.7× random baseline |
| ResCorRens K=32 | ✅ | 68 min MPS, Breakout solved, Pong drifts |
| ResCor Mamba K=4 | ✅ | 10h MPS, higher ceiling, catastrophic variance |
| Atari rollout probe | ✅ | Rens reliable, Mamba high-variance |
| Mamba rollout probe | ✅ | Best seed 3-4× better, worst seed 165× ratio |
| DiscreteRescor training | ⏳ | Script needed |
| Discrete rollout probe | ⏳ | After training |

---

**Pod state**: No GPU pod active. M4 Mac Mini MPS for all Plan 0 work. Re-provision GPU (RTX Pro 6000, $1.35/hr) only if needed for scale-ups or Dreamer fork.

**Commander note**: API-spawned Opus/Sonnet agents are the reliable orchestration pattern. Monolithic `run_wmca_mps.py` crashed. Individual `_cmdr_*.py` scripts with agent supervision work well.
