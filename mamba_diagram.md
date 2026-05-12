# `rescor_mamba_rand` Architecture Diagram

**Status**: as of 2026-04-29 sprint Day 0.

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║                          rescor_mamba_rand forward                             ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                                ║
║  INPUT                                                                          ║
║  x_seq : (B, K=4, C, H, W)        ← K=4 frame temporal context                  ║
║                                                                                ║
║  ┌─ split last frame ────────────────────────────────────────────────────────┐ ║
║  │                                                                            │ ║
║  │  x_t  := x_seq[:, -1]    (B, C, H, W)  ← current frame (for spatial core)  │ ║
║  │                                                                            │ ║
║  └────────────────────────────────────────────────────────────────────────────┘ ║
║                                                                                ║
║  ╔══════════════════════════════════════╗   ╔═══════════════════════════════╗ ║
║  ║   SPATIAL CORE (rescor_rens K=32)    ║   ║   TEMPORAL CORE (per-cell)    ║ ║
║  ║                                      ║   ║                               ║ ║
║  ║   x_t : (B, C, H, W)                 ║   ║   x_seq : (B, K, C, H, W)     ║ ║
║  ║   ─→ clamp to [0,1]                  ║   ║   ─→ permute to (B,H,W,K,C)   ║ ║
║  ║   ─→ broadcast to 32 reservoirs      ║   ║   ─→ flatten BHW              ║ ║
║  ║      r_values ∈ [3.57, 3.99]         ║   ║      shape: (B·H·W, K, C)     ║ ║
║  ║   ─→ FROZEN logistic map             ║   ║   ─→ Linear C → d_model=16    ║ ║
║  ║      f(x)= r·x·(1-x), 15 iters       ║   ║      shape: (B·H·W, K, 16)    ║ ║
║  ║      with shared 3×3 coupling        ║   ║                               ║ ║
║  ║   ─→ stack: (B, 32, C, H, W)         ║   ║   ┌────────────────────────┐  ║ ║
║  ║   ─→ uniform mean over K=32:         ║   ║   │  MinimalMambaBlock      │  ║ ║
║  ║                                      ║   ║   │  d_model=16  expand=2   │  ║ ║
║  ║   cml_mean : (B, C, H, W)            ║   ║   │  d_state=8   d_conv=4   │  ║ ║
║  ║                                      ║   ║   │  selective scan over K  │  ║ ║
║  ║   (43 frozen scalar params total:    ║   ║   │  ─→ (B·H·W, K, 16)      │  ║ ║
║  ║    32 r-values + coupling + ε,β)     ║   ║   │  ─→ take last step      │  ║ ║
║  ║                                      ║   ║   │  ─→ (B·H·W, 16)         │  ║ ║
║  ║                                      ║   ║   │                         │  ║ ║
║  ║                                      ║   ║   │  out_proj is **random   │  ║ ║
║  ║                                      ║   ║   │  init** in this variant │  ║ ║
║  ║                                      ║   ║   │  (vs zero-init in the   │  ║ ║
║  ║                                      ║   ║   │  base rescor_mamba —    │  ║ ║
║  ║                                      ║   ║   │  zero-init suppressed   │  ║ ║
║  ║                                      ║   ║   │  the temporal feature)  │  ║ ║
║  ║                                      ║   ║   └────────────────────────┘  ║ ║
║  ║                                      ║   ║                               ║ ║
║  ║                                      ║   ║   ─→ unflatten BHW            ║ ║
║  ║                                      ║   ║   ─→ permute to (B,16,H,W)    ║ ║
║  ║                                      ║   ║                               ║ ║
║  ║                                      ║   ║   mamba_temp : (B, 16, H, W)  ║ ║
║  ╚══════════════════════════════════════╝   ╚═══════════════════════════════╝ ║
║                  │                                       │                     ║
║                  │  cml_mean                             │  mamba_temp         ║
║                  ▼                                       ▼                     ║
║       ┌─────────────────────────────────────────────────────────────────┐      ║
║       │                       NCA CORRECTION HEAD                      │      ║
║       │                                                                 │      ║
║       │  concat([x_t,  cml_mean,  mamba_temp])    on channel dim         │      ║
║       │  shape: (B, C + C + 16, H, W)            = (B, C+C+16, H, W)    │      ║
║       │                                                                 │      ║
║       │  ─→ Conv2d 3×3,  hidden=32                                       │      ║
║       │  ─→ GroupNorm + GELU                                             │      ║
║       │  ─→ Conv2d 3×3,  → (B, C, H, W)                                  │      ║
║       │                                                                 │      ║
║       │  correction : (B, C, H, W)                                       │      ║
║       │                                                                 │      ║
║       │   (rescor_mamba:      ~5,346 trained params, mean-only NCA)     │      ║
║       │   (rescor_mamba_stat: ~5,634 trained params, stat-bank NCA)     │      ║
║       └─────────────────────────────────────────────────────────────────┘      ║
║                                       │                                        ║
║                                       ▼                                        ║
║                                                                                ║
║              OUTPUT  =  cml_mean  +  correction                                 ║
║              shape:  (B, C, H, W)                                               ║
║                                                                                ║
╚═══════════════════════════════════════════════════════════════════════════════╝
```

## Variants

| Name | NCA input | Mamba `out_proj` init | Trained params |
|---|---|---|---|
| `rescor_mamba` | mean-only `[x, cml_mean, mamba_temp]` | zero | 5,346 |
| `rescor_mamba_rand` | mean-only | random (normal) | 5,346 |
| `rescor_mamba_stat` | stat-bank `[x, cml_mean, cml_min, cml_max, mamba_temp]` | zero | 5,634 |
| `rescor_mamba_stat_rand` | stat-bank | random | 5,634 |

## Hyperparams (current default)

| Param | Value |
|---|---|
| `cml_K` (number of CMLs) | 32 |
| `cml_steps` (logistic iterations per forward) | 15 |
| `r_values` | linspace(3.57, 3.99, 32) |
| `kernel_size` (CML coupling) | 3 |
| `context_k` (Mamba history window) | 4 |
| `d_model` | 16 |
| `expand` (Mamba inner = expand·d_model) | 2 |
| `d_state` | 8 |
| `d_conv` (depthwise causal conv before scan) | 4 |
| `nca_hidden` | 32 |

## Key empirical findings

| Property | Result |
|---|---|
| step-1 MSE on gs (3-seed median, 200 trajs × 100 ep) | 1.40e-6 (15× better than rens K=32's 2.05e-5) |
| H=15 absolute MSE on gs (median) | 6.47e-5 (15× better than rens 9.66e-4) |
| H=100 absolute MSE on gs (median) | 2.63e-1 (**7× WORSE** than rens 3.50e-2 → catastrophic divergence) |
| Seed variance on gs H=15 | massive: 18.67× to 354.20× ratio across {s42, s43, s44} |

**Failure mode**: when the model's predictions stay near the data manifold, mamba's temporal context lets it crush rens. When predictions drift even slightly off-manifold, the K=4 frame buffer becomes a closed-loop error amplifier (each prediction polluting the next 4 inputs) and the trajectory diverges fast.

This is exactly the failure pattern the sprint targets:
- **pushforward (Day 1)** — exposes model to its own predictions during training
- **multistep penalty (Day 2-3)** — rolls H=8 steps in training with truncated BPTT
- **drift-gated hybrid (Day 4-5)** — auto-attenuates Mamba contribution when predictions drift

## File pointers

- Implementation: `src/wmca/modules/hybrid.py` (`ResCorMamba`, `ResCorMambaStat` classes)
- Mamba block: `src/wmca/modules/mamba_block.py` (`MinimalMambaBlock`, `MinimalMambaBlockSlow`)
- Registry: `src/wmca/model_registry.py` (4 entries)
- Probe: `dreamerv3_scaffolding/rollout_stability_probe_mamba.py`
- Plan: `rescor_mamba_plan.md`
- Day 0 results: `experiments/results/rollout_stability_probe_mamba.json`
