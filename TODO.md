# WMCA Sprint TODO — Detailed

**Last updated**: 2026-05-07 (Commander session — Atari Path A complete, Path C pending)

## WMCA Plan 0 Status

### Path A — Atari Latent World Modeling ✅ COMPLETE

| Stage | Status | Details |
|-------|--------|---------|
| A.1 — Atari data generation | ✅ | 25K frames each (Pong 4×16×32, Breakout 4×20×16) |
| A.2 — Atari encoder training | ✅ | GridAE: Breakout 38.58 dB, Pong 33.11 dB (after shuffle fix) |
| A.3 — Atari latent encoding | ✅ | 25K latents per game, diversity restored |
| A.4 — Rescor_rens training | ✅ | 3 seeds × 100ep, Pong median 2.65e-3, Breakout 1.75e-3 |
| A.5 — Rescor rollout probe | ✅ | Breakout solved (flat MSE 0.010), Pong drifts (0.096 at H=100) |
| A.6 — Rescor_mamba training | ✅ | 3 seeds × 100ep, ~10h MPS, higher variance |
| A.7 — Mamba rollout probe | ✅ | Best seed beats rens on Pong (3-4×), catastrophic seed variance |

### Path C — VQ-VAE + DiscreteRescor ⏳ PARTIAL

| Stage | Status | Details |
|-------|--------|---------|
| C.1 — VQ-VAE training | ✅ | 50K steps, 92.8% codebook usage, PSNR ~34 dB |
| C.2 — Token encoding | ✅ | 99,999 tokens (16×16), 93% codebook usage |
| C.3 — DiscreteRescor smoke | ✅ | Synthetic tokens PASS — loss decreases, no NaN |
| C.4 — DiscreteRescor training | ⏳ | Train on real Crafter tokens — 3 seeds × 100ep |
| C.5 — Discrete rollout probe | ⏳ | Autoregressive H=15/50/100 on token sequences |

### Blockers

- **DiscreteRescor training on real tokens**: script needed (`_cmdr_train_disc.py` or similar). Module validated (smoke pass), data ready (tokens.npy + next_tokens.npy + actions.npy).
- **Pipeline orchestration**: `run_wmca_mps.py` crashed silently. Individual `_cmdr_*.py` scripts + agent orchestration is the reliable pattern.

---

## Sprint window (original, now stale)

**Sprint window**: 2026-04-29 → 2026-05-05 (1 week)
**Goal**: stabilize all-horizon (H=15 → H=100+) deterministic rollout for `rescor_mamba_rand` on chaotic continuous benchmarks (gs/ks); preserve the H=15 absolute-MSE win (15× over rens K=32) while killing the H=100 catastrophe (3-seed median 2.63e-1 abs, 7× worse than rens).

**Pod**: `humming-vermilion-9b` (id `6dc63052f55b49d8897778e941264af1`) — RTX Pro 6000 96GB on dc_gnu, $1.35/hr. Provisioned 2026-04-29 ~10:22 UTC. Tear down at sprint end (TODO-8).

**Cron job watching pod**: NONE active. Earlier crons (`f0885c43`, `1b46699f`, `b6efbba4`) all canceled — pod is idle between sprint days. Re-add when next ablation launches.

**Sprint-wide decision rule**: a method WINS if it improves H=15 absolute MSE on gs by ≥2× over the rescor_mamba_rand baseline (3-seed median 6.47e-5) WITHOUT making H=100 worse. Soft-WIN: keeps H=15 win and reduces H=100 abs MSE by ≥2×.

---

## First-order items (active sprint execution)

### TODO-1 — Day 1: pushforward ablation on rens K=32

- **What**: train rescor_rens K=32 with `pushforward ∈ {False, True}` × 3 seeds × 3 benches × 100 ep. 18 cells, ~30-60 min on RTX Pro 6000.
- **Why**: cheapest lever, near-universal published win in neural-PDE since Brandstetter 2022. Validates that exposure-bias mitigation transfers to our setup before applying to mamba.
- **How**:
  - 1.1 ✅ Script written: `experiments/pushforward_ablation.py` with `--smoke` flag
  - 1.2 ✅ `train_model` patched with `pushforward`, `pushforward_n_steps`, `pushforward_prob` kwargs (in `src/wmca/model_registry.py`)
  - 1.3 ✅ Smoke-test on local CPU passed (loss decreases, no NaN, bit-identical with `pushforward=False` to legacy)
  - 1.4 ✅ Pod patched to inject `device="cuda"` in benchmark + train_model calls (CUDA wasn't being used initially)
  - 1.5 ✅ Launched in tmux session `sprint` on `humming-vermilion-9b`
  - 1.6 ✅ Ablation finished. 18 cells in `pushforward_ablation.json`. Wallclock ~6 min on GPU after optimization stack.
  - 1.7 ✅ σ=False reproduces phase1 medians within bf16 noise floor (~5-25× higher floor due to bf16, expected).
- **Done when**: `pushforward_ablation.json` has 18 cells (or all valid retries), checkpoints saved, σ=0 reproduces baseline within seed variance
- **Depends on**: TODO-3 (probe) before results can be evaluated, but training itself doesn't depend on the probe
- **Risk**: NaN from pushforward step (if model emits out-of-range values). CML clamp inside `_run_batched` guards against the obvious case. If still NaN, reduce `pushforward_prob` to 0.25.
- **Maps to**: Task #34
- **ETA**: ~30-60 min GPU after launch

### TODO-2 — Day 1: pushforward ablation on rescor_mamba_rand

- **What**: same matrix, but mamba_rand with `context_k=4`. 18 cells.
- **Why**: directly tests whether exposure-bias mitigation fixes the H=100 catastrophe on the variant we actually care about.
- **How**:
  - 2.1 ✅ Script written: `experiments/pushforward_ablation_mamba.py` (sibling of TODO-1's, mamba-specific)
  - 2.2 ✅ Same patch path for `device="cuda"` applied
  - 2.3 ✅ Launched sequentially after TODO-1 in same tmux session
  - 2.4 ✅ Ablation finished. 18 cells. Wallclock ~25 min on GPU.
  - 2.5 ✅ σ=0 cells match Day 0 baseline (modulo bf16 noise floor + s43 heat known instability)
- **Done when**: 18 mamba cells saved, σ=0 reproduces Day 0 baseline
- **Depends on**: TODO-1 finishes (single GPU = sequential)
- **Risk**: NaN as in TODO-1. Plus rank-5 input handling in pushforward — already smoke-tested on local CPU.
- **Maps to**: Task #34
- **ETA**: ~1.5-4.5h GPU after TODO-1 done

### TODO-3 — Day 1: write `pushforward_rollout_probe_mamba.py`

- **What**: sibling of `pushforward_rollout_probe.py` that loads mamba_rand checkpoints, uses rank-5 input + K=4 rolling buffer for autoregressive rollout.
- **Why**: existing probe is rens-only (rank-4). Mamba checkpoints from TODO-2 need a different rollout loop.
- **How**:
  - 3.1 Read existing `pushforward_rollout_probe.py` and `rollout_stability_probe_mamba.py`
  - 3.2 Adapt: replace `MODEL = "rescor_rens"` with mamba_rand, replace single-frame `x` state with K=4 buffer (seed from ground-truth frames 0..K-1, advance with predictions)
  - 3.3 Update checkpoint dir to `experiments/results/pushforward_ckpts_mamba/`
  - 3.4 Save to `experiments/results/pushforward_rollout_probe_mamba.json`
  - 3.5 ✅ Done. Probe run on pod alongside rens probe; landed clean (3-seed median ratios computed for σ ∈ {False, True} × {heat, gs, ks}).
- **Done when**: ✅ Smoke + full run done.
- **Maps to**: Task #38 (completed)
- **ETA**: actual ~30 min code, ~3 min probe inference on GPU

### TODO-4 — Day 1: run rollout probes + analyze

- **What**: load checkpoints from TODO-1/2 and roll H={15,50,100} on each. Compare per-step MSE / cos_div / ratio across `pushforward ∈ {False, True}`.
- **Why**: the actual decision-making step. 1-step MSE alone doesn't tell us if pushforward fixed the drift problem.
- **How**:
  - 4.1 Run `dreamerv3_scaffolding/pushforward_rollout_probe.py` (rens variant) — already exists
  - 4.2 Run `dreamerv3_scaffolding/pushforward_rollout_probe_mamba.py` (TODO-3) — once written
  - 4.3 Compute per-bench median across 3 seeds: step1 MSE, H=15 abs MSE + ratio, H=50 abs + ratio, H=100 abs + ratio, cos_div at each horizon
  - 4.4 ✅ Comparison table built. **VERDICT: NEGATIVE** on gs (mamba_rand σ=True gs H=100 ratio 10402×, cos_div 0.47 — near-orthogonal divergence; absolute H=15 MSE 25× WORSE). Modest ks win on mamba (H=15 abs MSE halved, ratio 12.51× vs 68.55×). Heat result is zero-attractor artifact (cos_div ~1).
- **Verdict**: LOSS on gs (the bench that matters most), partial win on ks. Pivot to TODO-5 (multistep penalty).
- **Maps to**: Task #34 (completed 2026-04-29 evening)

### TODO-5 — Day 2-3: multistep penalty NODE loss (Chakraborty 2024)

- **What**: H-step rollout MSE training with truncated BPTT. Roll model H=8 in training, accumulate per-step MSE, backprop only through last K_bptt=4 steps.
- **Why**: theoretically cleanest attack on Mikhaeil 2022's chaos amplification — bounds BPTT depth without bounding Lyapunov exponent. Cross-validated by both training-loss and theory brainstorms.
- **How**:
  - 5.1 Spawn opus to design + implement: read `train_model`, add `multistep_horizon: int = 1, multistep_bptt: int = 4, multistep_weight_schedule: str = "uniform"` kwargs
  - 5.2 Algorithm sketch:
    ```
    state = x_t  (with detached history)
    losses = []
    for h in range(H):
        if h < (H - K_bptt):
            with torch.no_grad():
                pred = model(state)
        else:
            pred = model(state)
        gt   = y_{t+h}
        losses.append(MSE(pred, gt) * weight[h])
        state = update_buffer(state, pred)
    loss = sum(losses)
    ```
  - 5.3 Build `experiments/multistep_ablation.py` — `multistep_horizon ∈ {1, 4, 8}` × 3 seeds × 3 benches × 100 ep, on mamba_rand only (its K=4 SSM is the natural match; rens is stateless)
  - 5.4 Smoke test on CPU
  - 5.5 Sync + launch on pod
  - 5.6 Probe trained checkpoints (need `multistep_rollout_probe_mamba.py` — sibling of TODO-3's)
  - 5.7 Compare H ∈ {1, 4, 8} variants on H=15/50/100 abs MSE
- **Done when**: 27 cells (3 H values × 3 seeds × 3 benches) trained, probed, compared
- **Depends on**: TODO-1/2/4 results inform whether to start (if pushforward already wins big, multistep might be redundant)
- **Risk**: gradient explosion despite truncated BPTT. Mitigations: gradient clipping at 1.0, lower LR (3e-4 instead of 1e-3), weight schedule that down-weights later steps.
- **Maps to**: Task #35 (in_progress 2026-04-29 evening)
- **ETA**: ~3-5h dev (opus afe26cc522ae3c636 running now) + ~1-2h GPU on optimized stack = ~half day wallclock if smoke passes
- **Status**: opus implementing now per TODO-5 spec. Smoke required before full launch.

### TODO-6 — Day 4-5: drift-gated hybrid (`ResCorMambaGated`)

- **What**: new architectural variant — per-cell gate using `||x_t − cml_mean||` as drift signal, auto-attenuates Mamba contribution at high drift. When predictions stay near manifold, full mamba power; when drift is large, fall back toward pure-rens behavior.
- **Why**: novel architectural lever from arch brainstorm; CPU-cheap; orthogonal to training-loss methods (TODO-1/5).
- **How**:
  - 6.1 Spawn opus to design `ResCorMambaGated` class:
    ```python
    drift = ((x_t - cml_mean) ** 2).mean(dim=1, keepdim=True).sqrt()  # (B, 1, H, W)
    gate  = torch.sigmoid(self.gate_scale * (-drift + self.gate_bias))  # (B, 1, H, W)
    correction = self.nca(...) * gate
    out = cml_mean + correction
    ```
  - 6.2 Add registry entries: `rescor_mamba_gated`, `rescor_mamba_gated_rand`
  - 6.3 Smoke test forward pass + training stability
  - 6.4 Train + probe at standard protocol (3 seeds × 3 benches × 100 ep)
  - 6.5 Optional follow-up: combine drift gating with pushforward training (TODO-1) — synergy test
- **Done when**: variant trained + probed; comparison table vs vanilla mamba_rand
- **Depends on**: nothing structurally; benefits from TODO-1/5 results to know if other levers already work
- **Risk**: gate could collapse to identity (always fully open or always fully closed). Mitigation: initialize gate_scale small, monitor gate distribution during training.
- **Maps to**: Task #36
- **ETA**: ~1-2 days dev + ~3-5h GPU = ~2 days wallclock

### TODO-7 — Day 6-7 (CONDITIONAL): diffusion forcing / Self Forcing

- **What**: train mamba_rand with diffusion-forcing-style loss (Chen et al. NeurIPS 2024) or Self Forcing variant (Huang 2025). Subsumes pushforward + multistep as special cases.
- **Why**: empirically strongest published method for past-training-horizon rollout in continuous video. Natural finale if Tier 1+2 are partial wins.
- **How**:
  - 7.1 Decision gate: ONLY proceed if (TODO-4 OR TODO-5 result) is SOFT-WIN, not full WIN, not LOSS. (Full-WIN = no need; LOSS = pushforward family doesn't transfer, diffusion forcing won't help either.)
  - 7.2 Spawn opus to study Chen 2024 + Huang 2025 papers (read brainstorm_theory.md)
  - 7.3 Implement adapted diffusion-forcing training loop: noise schedule per-timestep, denoising during rollout, `pyramid_noise=True`
  - 7.4 Train + probe
- **Done when**: variant trained + probed; comparison vs all prior variants
- **Depends on**: TODO-4 + TODO-5 results
- **Risk**: HIGH — most engineering, most novel, GPU-heavy. May not finish in sprint window. Acceptable to time-box at 3 days max and abandon.
- **Maps to**: Task #37
- **ETA**: ~3-5d dev + ~10-20h GPU. Skip if Tier 1+2 give full wins.

### TODO-8 — Pod tear-down (humming-vermilion-9b) ✅ DONE 2026-05-01

- Killed by user 2026-05-01. Pod `humming-vermilion-9b` (id `6dc63052f55b49d8897778e941264af1`) terminated. Billing stopped.
- Re-provision via `/provision-prime-gpu` (~5 min) when launching Task #41 Crafter probe or Dreamer fork training.
- **Maps to**: Task #39 ✅

---

## Second-order items (post-sprint, follow-on)

### TODO-A — Per-day documentation update

- **What**: background opus appends new section to findings.md, dated entry to experiment_logs.md, outcome subblock to dreamerv3_fork_plan.md §4, memory update.
- **Trigger**: after each TODO-1/2/4/5/6/7 produces results
- **Effort**: ~30 min opus per day
- **Pattern**: established (used for §59-§63)

### TODO-B — Multi-seed Crafter-latent probe of winning variant

- **What**: re-run the Crafter probe on whichever variant wins TODO-4 or TODO-5 — 3 seeds × 100 ep on Crafter latents.
- **Why**: a win on synthetic gs/ks doesn't necessarily transfer. Crafter latents are the actually-relevant substrate for the Dreamer fork. Without this, the fork narrative is shaky.
- **How**:
  - B.1 Pick winning variant from TODO-4/5/6 results
  - B.2 Build sibling of `rollout_stability_probe_crafter.py` for the new variant
  - B.3 Run 3 seeds × 100 ep, ~6h GPU
  - B.4 Compare H=15/50/100 abs MSE to plain rescor_rens K=32 baseline (Task #31 results)
- **Done when**: Crafter-latent table for winning variant exists; verdict re-confirmed (or revised) on actually-relevant substrate
- **Trigger**: TODO-4/5 produces a clear winner
- **Effort**: ~1 day wallclock + ~6h GPU
- **Maps to**: new task to be created on win

### TODO-C — Begin DreamerV3 fork (M3-M6)

- **What**: implement `RescorRSSM` class inside forked `NM512/dreamerv3-torch`, wire into Dreamer's RSSM interface, train on Crafter, compare to vanilla GRU baseline.
- **Why**: ultimate test — does the smaller deterministic core actually maintain Crafter score?
- **How**: see `dreamerv3_fork_plan.md` §5 (M3-M6 milestones already detailed there)
- **Trigger**: sprint either identifies a stable variant (best case) OR sprint exhausts and we proceed with whatever we have
- **Effort**: ~3-4 weeks per fork plan
- **Maps to**: not yet a task — create when triggered

### TODO-D — Pixel-space scale-up

- **What**: build `rescor_ms` (U-Net 64×64 pyramid) and `rescorformer` (SWA + RoPE at 64×64) — already-pending Tasks #28, #29
- **Why**: orthogonal to all-horizon stability work. Targets 64×64 Crafter Real pipeline (the actual paper bottleneck for pixel-space modeling).
- **Trigger**: post-sprint, if architecture work winds down or stability is "good enough"
- **Effort**: ~1 week each per task description
- **Maps to**: Tasks #28, #29

### TODO-E — Sprint write-up

- **What**: draft technical note covering: Day 0 multi-seed mamba_rand result, Day 1-7 outcomes (positive + negative), comparison to rens K=32 baseline, implications for Dreamer fork narrative.
- **Format**: workshop paper or technical note (10-15 pages). Sections: motivation (chaos amplification), architecture (rescor_mamba diagram from `mamba_diagram.md`), methods (pushforward/multistep/drift-gated/diffusion-forcing), experiments (results tables), discussion.
- **Trigger**: sprint end (TODO-8 fired)
- **Effort**: ~3-5 days writing + figure work
- **Maps to**: not yet a task

### TODO-F — Investigate s43-style "bad rollout regime" on gs

- **What**: diagnostic — why does mamba_rand seed 43 specifically have H=15 ratio 354× when seeds 42/44 are 36× / 18×?
- **Hypotheses**:
  - Initialization-dependent attractor basin (model lands in suboptimal local minimum that has bad rollout properties even though step-1 MSE is fine)
  - Optimization noise (stochastic batch ordering happens to hit a pathological gradient)
  - Data-init dependence (s43-generated trajectories happen to have a feature that's harder to model)
- **How**: train s43 with 3 different optimizer seed initializations (Adam state different), see if all blow up or just some. Check loss curve shapes for s43 vs s44.
- **Trigger**: optional, low priority — only if we have spare time and sprint has a clear winner
- **Effort**: half day diagnostic
- **Maps to**: not yet a task

### TODO-G — `mamba-2` block (Dao & Gu ICML 2024)

- **What**: replace `MinimalMambaBlock` with the Mamba-2 / SSD formulation. Mamba-2 has a 2-8× speedup over Mamba-1 due to matrix-multiplication form (Structured State Space Duality).
- **Why**: Mamba-1's selective scan is bandwidth-bound on small K (we measured ~30ms per forward at B·H·W=16K). Mamba-2 might cut this by 5-10× and make CPU runs feasible again.
- **Trigger**: post-sprint optimization, OR if we want to ablate larger context_k (>=8).
- **Effort**: ~2-3 days dev + verification
- **Maps to**: not yet a task

### TODO-H — Optimize CML reservoir on GPU

- **What**: batch 15 logistic-map iterations × 32 reservoirs × grouped conv2d as a single kernel-fused op (CUDA Graph or torch.compile).
- **Why**: reservoir is the spatial-core hot path. Currently 15 sequential conv2d calls per forward — ~30ms launch overhead-dominated on small grids. 16×16 grids especially.
- **Trigger**: only if sprint extends to bigger grids (64×64) where reservoir cost dominates
- **Effort**: ~2 days dev
- **Maps to**: not yet a task

---

## Third-order items (long-term, post-fork, post-paper)

### TODO-α — Compare to Delta-Iris (Crafter)

- The original strategic motivation. Once Dreamer fork (TODO-C) gives a Crafter score for rescor-RSSM, compare head-to-head with Delta-Iris under matched compute.

### TODO-β — Open-source rescor as a standalone module

- Current `wmca` repo is research-mode. Extract the canonical rescor variants (rescor_rens K=32, rescor_mamba_rand, the winner from this sprint) into a small `pip install rescor`-able package.

### TODO-γ — Token-based world models

- All current rescor work is on continuous latents. Could the same architecture work on discrete token sequences (à la Iris / Delta-Iris)? The CML reservoir would need adaptation; NCA correction stays similar.

### TODO-δ — Alternate base reservoirs

- Logistic map is one chaotic dynamical system. Others: tent map, Hénon map, Lorenz attractor (continuous-time). Are any meaningfully better for autoregressive prediction?

### TODO-ε — Scaling beyond 96×96 grids

- Mamba per-cell scan compute scales as O(H·W·K·d²). At 96×96 grids with K=4, d_model=16, that's 36k effective batch — still GPU-friendly. But Mamba grouped scan won't fit at 256×256.

### TODO-ζ — Theoretical analysis

- Connect rescor's empirical behavior to existing theory on chaotic prediction (Lyapunov exponents, conditional Lyapunov ratios, mutual information rates). Prove (or disprove) that rescor_mamba_rand's H=15 absolute-MSE win has a theoretical floor.

### TODO-η — Production deployment

- Far-out: if rescor-RSSM beats GRU at scale, productionize. World-model serving infrastructure, INT8 quantization (rescor's small param count is friendly here), distributed training.

---

## Failure-mode TODOs (what to do if things break)

### TODO-X1 — Pushforward NaNs during training
- Symptoms: training loss → NaN within first 10 epochs
- First-line fix: reduce `pushforward_prob` to 0.25
- Second-line: tighter gradient clipping (1.0 → 0.5)
- Last resort: clamp predictions in pushforward step (`pred.clamp(0,1)` before re-feeding)

### TODO-X2 — Pod terminated unexpectedly (preemption / mistake)
- Recovery: re-provision identical pod via TODO-8 reverse procedure; rsync project; resume from JSON. All ablation scripts have resume-from-JSON, no progress lost beyond the in-flight cell.
- Prevention: do NOT use spot pods (lesson learned). On-demand only.

### TODO-X3 — Day 1 LOSS (pushforward worse than baseline)
- Skip TODO-5 multistep (likely won't help if exposure-bias mitigation already failed)
- Jump to TODO-6 drift-gated (architectural rather than training-time)
- Re-evaluate sprint after Day 4-5

### TODO-X4 — All Tier 1/2 lose
- Sprint outcome is "deterministic core can't beat chaos at H=100 with these methods"
- Pivot: write up negative results (TODO-E), start Dreamer fork (TODO-C) with current best variant (mamba_rand at H=15 only), accept KEEP-posterior is the final answer
- Fall back to scale-up tasks (TODO-D) for the paper

### TODO-X5 — GPU compute exceeds budget
- Soft cap: $50 (≈37h on dc_gnu)
- Hard cap: $100 (≈74h on dc_gnu)
- If hit: tear down pod, reassess from current state, possibly extend sprint to second week

---

## Pod operations log

| Time (UTC) | Event |
|---|---|
| 2026-04-29 10:11 | First pod `humming-vermilion-9` (id `09d95a19...`, datacrunch FI Spot) created. |
| 2026-04-29 10:11 | Terminated — user said no spot. |
| 2026-04-29 10:12 | Second pod `humming-vermilion-9` (id `519aa5f8...`, dc_gnu US, $1.35/hr) created. |
| 2026-04-29 10:14 | SSH ok, GPU visible. CUDA stack works (after upgrade torch nightly cu128 for Blackwell sm_120). Smoke tests pass. |
| 2026-04-29 10:16 | Day 1 ablations launched. GPU at 0% — turns out scripts don't auto-move to cuda. |
| 2026-04-29 10:18 | Patched scripts on pod with `device="cuda"`. Relaunched. GPU at 49%. |
| 2026-04-29 ~10:20 | Sibling claude session terminated this pod by mistake. |
| 2026-04-29 10:22 | Third pod `humming-vermilion-9b` (id `6dc63052...`) created (replacement). Same dc_gnu config. |
| 2026-04-29 10:24 | Setup re-done, ablations relaunched. GPU at 42%, training rens cell s42 σ=False. |
| 2026-04-29 ~11:00 | Day 1 ablation showing only 75% GPU util at 1.4% VRAM — kernel-launch-bound. Killed run, opus applied optimizations (batch=128 + sqrt-rule LR, torch.compile mode="default", bf16 autocast). 2.6× / 7-10× speedup measured. |
| 2026-04-29 11:22 | Optimized Day 1 ablations relaunched. |
| 2026-04-29 ~11:55 | Day 1 ablations done. Wallclock ~30 min total (rens 6 min, mamba 25 min). |
| 2026-04-29 ~13:55 | Day 1 rollout probes done on pod (rens + mamba sequential, ~3 min total). Verdict NEGATIVE on gs (most important bench). |
| 2026-04-29 evening | Cron `b6efbba4` canceled — pod idle between sprint days. Day 2-3 multistep penalty implementation underway via opus. |

---

## Cron / monitoring jobs

- All canceled (pod idle between sprint days). Re-add when next ablation/probe launches.
- Cron history: `f0885c43` (Day 1 watch), `1b46699f` (Day 1 telegram-aware watch — replaced when telegram dropped), `b6efbba4` (Day 1 final watch — canceled when probes finished).

---

## Task references (post-sprint state, 2026-04-30)

| TODO | Task ID | Status |
|---|---|---|
| TODO-1 + TODO-2 + TODO-4 | #34 | ✅ completed (NEGATIVE) |
| TODO-3 | #38 | ✅ completed |
| TODO-5 (multistep) | #35 | ✅ completed (**FIRST WIN** — multistep H=8 gs gate pass) |
| TODO-6 (drift-gated alone) | #36 | ✅ completed (rolled into combo) |
| Combo (drift-gated + multistep) | #43 | ✅ completed (BORING MIDDLE) |
| TODO-7 (diffusion forcing) | #37 | pending — conditional on TODO-B outcome |
| TODO-8 (pod tear-down) | #39 | ✅ done 2026-05-01 (user killed) |
| TODO-A (per-day docs) | #40 | recurring |
| TODO-B (Crafter probe of multistep H=8) | #41 | **pending — recommended next move** |
| TODO-C (Dreamer fork M3+) | dreamerv3_fork_plan.md M3+ | not yet a task |
| TODO-D (pixel-space scale-ups) | #28, #29 | pending — post-sprint |
| TODO-E (sprint write-up) | #42 | pending — start in parallel with TODO-B |
| TODO-F, G, H | (new at sprint end) | not yet |
| TODO-α through η | (post-fork) | far future |

**Sprint verdict**: multistep H=8 is the only meaningful win. Combo with drift-gated didn't synergize. Theory's 55% boring-middle prediction held. KEEP-posterior decision unchanged for Dreamer fork.

**Recommended next action**: TODO-B (Crafter probe of multistep H=8) — see `next_steps.md` for sequencing.

---

## Post-sprint: Atari Latent Dynamics (Path A from `plans/plan_0.md`)

### Path A.1 — Train Atari frame encoder ✅ COMPLETE 2026-05-07
- Grid-native AE (option b, no spatial compression). Pong **33.1 dB** PSNR, Breakout **38.6 dB** PSNR on 500 traj × 50 steps per bench.
- Bug fixed: `torch.manual_seed` was not reset between dataset construction and AE training, causing latent collapse on sparse Breakout frames (initial 25.1 dB → fixed 38.6 dB). Pong latents also had ~zero std pre-fix; std 0.039 post-fix.
- Artifacts: `experiments/train_atari_encoder.py` (or `_cmdr_atari_ae_v2.py`), `experiments/atari_data/frame_encoder.pt`, `experiments/atari_data/{pong,breakout}_*.npy`.

### Path A.2 — Action-conditioned 1-step training ✅ COMPLETE 2026-05-07
- 3 seeds × 2 models × 100 epochs × 2 benches = 12 runs. `rescor_rens` K=32 and `rescor_mamba_rand` K=4. Standard sprint stack (bf16 + `torch.compile` + batch=128 + lr=1.4e-3) on GPU; orchestrated via API-spawned agents on M4 MPS for the day's actual run.
- Patched `create_model` → `ResidualCorrectionWM` instantiation to allow `use_sigmoid=True` override for Atari case (in_ch=2, out_ch=1).
- 1-step MSE: mamba ~1.7× lower than rens on both benches (consistent with §63 synthetic).
- Artifacts: `src/wmca/atari_real.py` (`AtariLatentBenchmark`), checkpoints in `experiments/atari_data/{rescor,mamba}_{game}_seed{42,43,44}.pt`.

### Path A.3 — Action-conditioned autoregressive rollout ✅ COMPLETE 2026-05-07
- Wrote `dreamerv3_scaffolding/rollout_stability_probe_atari.py` (or `_cmdr_atari_rollout.py` / `_cmdr_mamba_rollout.py` per agent run). Open-loop action, closed-loop state. 20 test traj × H ∈ {15, 50, 100} × 3 seeds.
- 3-seed median H=100 ratios:
  - rescor_rens K=32: Breakout **2.6×** STABLE / Pong **36.8×** chaotic
  - rescor_mamba_rand K=4: Pong **28.6×** (1.3× better than rens) / Breakout **7.4×** (2.9× worse than rens)
- Bench-specific winner inversion confirmed.

### Note on mamba seed variance
**Mamba's per-seed Pong H=100 ratios spread 14×–165×.** Compared to rens K=32's <1.4× spread on the same data, this is a 12× best-vs-worst gap on the same bench/model — same s43-style pathology pattern flagged in §63 / TODO-F. The 165× outlier was seed 43 (recurring; this is at least the second project-wide instance of seed 43 producing a mamba bad-rollout regime). Reporting mamba results at the median (28.6×) understates the tail risk; reporting at the worst seed (165×) overstates the typical case. **Both numbers should be present whenever this variant is summarized.**

**Implication**: when mamba is summarized in any future writeup or downstream comparison, include the per-seed spread, not just the median. Mamba on Atari has the same fat-tail problem mamba had on synthetic gs/ks: low median, fat tail. Mitigations are open (TODO-F-style optimizer-state-seed diagnostic, gate_init reseeding, etc.).

**Mapping**: not yet a task. Plan reference: `plans/plan_0.md` §A.1, §A.2, §A.3 — all three Path A checkboxes (lines 335, 338, 339) now satisfied. Path C (Iris-style discrete tokens) untouched as of 2026-05-07 evening.

**Next decision** (mirrored in `next_steps.md`): run multistep H=8 (§66) on Atari latents (~half day GPU; tests whether multistep transfers to non-synthetic substrate) OR pivot to Dreamer fork training with rens K=32 as the Breakout-validated backbone.
