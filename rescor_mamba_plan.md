# rescor_mamba — Implementation Plan

**Status:** design doc, awaiting user approval before any code changes.
**Goal:** close the H=15 autoregressive-rollout gap on chaotic targets (gs 17×, ks 127× MSE blowup) identified in Task #27. Hypothesis: giving the per-cell model access to K past frames reduces step-1 MSE (velocity/acceleration information is now explicit) which compounds favorably through the rollout.

## Decisions already locked in

1. Spatial core = **rescor_rens K=32** (`CML2DMultiR` with `gate_mode="uniform"`, K=32 r-values linearly spaced over [3.57, 3.99]). NOT the vanilla-CML K=1 of `arch_plan.md` §5.
2. Temporal context **K=4 frames** (current + 3 past). Parameterize `context_k` so varying K is a constructor kwarg, nothing else.
3. Spatial param budget: the rens bank contributes 0 trained params; NCA accounts for the trained budget (~321 for mean-only variant, ~up to ~0.7 k for the stat-bank variant at in_ch=1). Mamba block adds a few k more.

---

## 1. Architecture

### 1.1 Forward pass (exact shapes)

Inputs:
- `x_seq ∈ R^(B, K, C, H, W)` — last K frames, most recent last. `K=4`, `C ∈ {1, 2, 4}` depending on benchmark.
- Throughout, `H=W=16` for 2-D benchmarks; `H=1`, `W=grid_width` for ks / rule110.

Block composition:

```
x_seq : (B, K, C, H, W)
  │
  ├── Spatial-now:   x_now = x_seq[:, -1]                            # (B, C, H, W)
  │
  ├── Spatial-rens bank (applied only to x_now, under torch.no_grad):
  │       rens_stack = self.rens._run_batched(x_now)                 # (B, K_rens=32, C, H, W)
  │       cml_mean   = rens_stack.mean(dim=1)                        # (B, C, H, W)
  │       [optional stats if use_stat_bank=True:
  │         cml_var, cml_min, cml_max over dim=1]
  │
  ├── Per-cell Mamba over temporal axis:
  │       # reshape to flatten spatial into a giant batch of independent sequences
  │       seq = x_seq.permute(0, 3, 4, 1, 2)                         # (B, H, W, K, C)
  │       seq = seq.reshape(B * H * W, K, C)                         # (B*H*W, K, C)
  │       seq_proj = self.in_proj(seq)                               # (B*H*W, K, d_model)
  │       h_seq = self.mamba_block(seq_proj)                         # (B*H*W, K, d_model)
  │       h_last = h_seq[:, -1, :]                                   # (B*H*W, d_model) — last-step readout
  │       temporal_feat = h_last.reshape(B, H, W, d_model).permute(0, 3, 1, 2)
  │                                                                  # (B, d_model, H, W)
  │
  ├── NCA correction:
  │       nca_in = cat([x_now, cml_mean, temporal_feat (+stats)], dim=1)
  │                                                                  # (B, C + C + d_model [+ 3C], H, W)
  │       correction = self.nca(nca_in)                              # (B, C, H, W)
  │
  └── Residual:
          out = cml_mean + correction
          if use_sigmoid: out = out.clamp(0, 1)
          return out                                                 # (B, C, H, W)
```

### 1.2 Design choices inside the block

* **Where the rens bank sits.** Applied **only to the current frame** `x_seq[:, -1]`. The K=32 reservoir is the proven spatial prior; running it K_context times would multiply compute by 4× and gain nothing — temporal information is already carried by the Mamba branch.
* **Where the Mamba scan sits.** Per spatial position `(h, w)`, over the temporal axis `K_context=4`, flat-batched across `B*H*W` to amortize the scan on CPU. d_model=16, d_state=8, d_conv=4 matches §5 of `arch_plan.md`.
* **What the NCA sees.** Two variants, both supported; wire behind one `use_stat_bank: bool` flag (defaulting False for matched-budget parity with vanilla rescor_rens, but exposing the stat-bank for Phase-1-style variance experiments):
  * `use_stat_bank=False` → `nca_in = [x_now, cml_mean, temporal_feat]` → `C + C + d_model` channels.
  * `use_stat_bank=True`  → `nca_in = [x_now, cml_mean, cml_var, cml_min, cml_max, temporal_feat]` → `5C + d_model` channels.
* **Residual target.** `cml_mean` (unchanged — same as `rescor_rens`). The Mamba feature is routed THROUGH the NCA, never added directly to the prediction, so at init (zero-init last NCA conv bias & kaiming weights as today) the model reduces to pure rens K=32 and can only earn the Mamba signal if it helps.

### 1.3 Why a single Mamba block (not stacked)

Depth inside the temporal branch does not help the K=4 context (the rescor_rens_deep L=2 negative result at `findings.md §55` is the relevant warning for depth-via-stacking). One block with d_state=8 has enough capacity to encode velocity + acceleration over 4 frames; more depth would just widen the param budget without buying the stability we actually need.

---

## 2. Param count

For the **default C=1 config (heat, ks, gol, rule110, wireworld-per-channel)**:

### Trained params

| Block | Param breakdown | Count |
|---|---|---|
| `in_proj`: Linear(C=1 → d_model=16) | 1·16 + 16 | 32 |
| Mamba block — selective scan | see §7.2 | 1,856 |
| NCA conv1: Conv2d(C + C + d_model = 1+1+16 = 18 → 16, 3×3) | 18·16·9 + 16 | 2,608 |
| NCA conv2: Conv2d(16 → 1, 1×1) | 16·1·1 + 1 | 17 |
| **Total trained (default, `use_stat_bank=False`)** | | **≈ 4,513** |

With `use_stat_bank=True`:
- NCA conv1 input = 1 + 4·1 + 16 = 21 → 16, 3×3 → 21·16·9 + 16 = 3,040.
- Total trained ≈ **4,945**.

For C=2 (gray_scott), add:
- `in_proj`: 2·16 + 16 = 48 (+16).
- NCA conv1 input = 2 + 2 + 16 = 20 → 16, 3×3 → 2,896 (+288).
- NCA conv2: 16·2 + 2 = 34 (+17).
- Total trained ≈ **4,834 (stat_bank=False)** or **5,410 (stat_bank=True)**.

**Target envelope: ~4.5 k–5.5 k trained**, within the user-specified 5–10 k band.

### Mamba-block internal param breakdown (d_model=16, d_state=8, d_conv=4, expand=2, inner=d_model·expand=32)

Following the canonical selective-SSM parameterization (`mamba-ssm` equivalent, inline impl — see §7):

| Sub-block | Param breakdown | Count |
|---|---|---|
| `in_proj` (d_model → 2·d_inner)        | 16·64 + 64 | 1,088 |
| `conv1d` (d_inner, kernel=d_conv=4, groups=d_inner) | 32·4 + 32 | 160 |
| `x_proj` (d_inner → dt_rank + 2·d_state) | 32·(1 + 16) + 17 = 32·17 + 17 | 561  |
| `dt_proj` (dt_rank=1 → d_inner)        | 1·32 + 32 | 64   |
| `A_log` (frozen-ish but learnable, d_inner·d_state) | 32·8 | 256  |
| `D` (d_inner) | 32 | 32   |
| `out_proj` (d_inner → d_model) | 32·16 + 16 | 528  |
| **Mamba block subtotal (trained)** | | **2,689** |

Note: earlier §2 table rolled up a rough "1,856" estimate — more careful accounting is ~2.7 k. This pushes total trained to **~5.3 k (stat_bank=False) / ~5.7 k (stat_bank=True)** at C=1. Still inside the user's 5–10 k envelope, but the honest number is higher than §5 of `arch_plan.md` claimed. **Flag for user** (§8, open question 3): we can shrink by setting `d_inner = d_model` (drop `expand` to 1), which roughly halves the block to ~1.4 k trained and drops the total to ~4.0 k.

### Frozen params

| Source | Count |
|---|---|
| `CML2DMultiR` (K=32): `r_values (32) + eps (1) + beta (1) + K_local (kernel_size² = 9)` | 43 |
| **Total frozen** | **43** |

No CUDA kernels, no SSM buffers — Mamba's `A_log`, `D`, conv1d weights are all trained.

---

## 3. Data pipeline changes

### 3.1 New helper in `src/wmca/benchmarks.py`

```python
def _make_k_frame_pairs(trajs: np.ndarray, K: int = 4):
    """(N_traj, T+1, ...) -> X (N*(T+1-K), K, ...), Y (N*(T+1-K), ...).

    Builds K-frame context windows. For trajectory i with frames f_0..f_T:
      sample j (0 <= j <= T-K): X[j] = [f_j, f_{j+1}, ..., f_{j+K-1}],
                                 Y[j] = f_{j+K}

    K=1 is equivalent to _make_pairs (single-frame input).
    """
    N, Tp1 = trajs.shape[:2]
    rest = trajs.shape[2:]
    T = Tp1 - 1
    n_windows = T - K + 1
    if n_windows <= 0:
        raise ValueError(f"Trajectory too short (T+1={Tp1}) for K={K}")
    # (N, n_windows, K, *rest)
    windows = np.stack([trajs[:, j:j+K] for j in range(n_windows)], axis=1)
    X = windows.reshape(-1, K, *rest)
    Y = trajs[:, K:].reshape(-1, *rest)
    return X, Y
```

Shape contract: `X: (N*, K, C, H, W)` with K in axis=1. `Y: (N*, C, H, W)` — single-frame target (next step only, no sequence-to-sequence yet).

### 3.2 Wiring into generate_*

Add a `context_k: int = 1` kwarg to each of `generate_heat`, `generate_ks`, `generate_gray_scott`, `generate_gol`, `generate_rule110`, `generate_wireworld`. When `context_k > 1`, route through `_make_k_frame_pairs(trajs, K=context_k)` instead of `_make_pairs`. Default `context_k=1` is a **pure pass-through**: existing call sites / experiments / the 16-benchmark harness must not change behavior.

The `meta` dict gains `"context_k": context_k` so the trainer / evaluator / rollout probe know the expected input rank.

For 1-D benchmarks (ks, rule110): the existing `_pairs_1d` / equivalent pattern extends the same way (rest = `(1, 1, W)`), no special casing needed since `_make_k_frame_pairs` uses `trajs.shape[2:]` as `rest`.

### 3.3 Trajectory length requirement

For `n_steps=50` (heat default) and K=4, `n_windows = 50 - 4 + 1 = 47` per trajectory — plenty. For rollout-stability, the existing probe uses `n_steps=105`, which gives 102 windows per trajectory — also fine.

---

## 4. Training-loop compatibility

`train_model` in `model_registry.py` does:

```python
for i in range(0, len(perm), batch_size):
    idx = perm[i : i + batch_size]
    xb, yb = X_tr[idx], Y_tr[idx]
    ...
    pred = model(xb)
```

It never inspects `xb.shape` beyond the leading batch dim. An `xb` of rank 5 `(B, K, C, H, W)` will flow through without code changes. The MSE / BCE / CE criteria all operate on `pred` vs `yb` with matching rank-4 shapes — the K axis is entirely inside the model.

**Chosen approach: (a) keep input rank-5; do NOT flatten K into channels at the harness boundary.** Reasons:
- Zero changes to `train_model` / `evaluate_model`.
- The Mamba block wants a clean `(B*H*W, K, C)` sequence anyway — flattening K into channels would force us to split them back out inside the model, which is uglier.
- Keeps the tensor semantics visible to anyone reading the model: "5-D = sequence of grids".

The **one** thing we must verify: `X_val`/`X_test` also get stacked as rank-5 when `context_k > 1`. This follows automatically from the benchmarks change in §3.

Rollout probe compatibility: see §6.

---

## 5. Registration

### 5.1 New module file `src/wmca/modules/mamba_block.py`

Hosts the `MinimalMambaBlock` (see §7). Keeps `hybrid.py` from growing past 3 k LOC.

### 5.2 New class `ResCorMamba` in `src/wmca/modules/hybrid.py`

```python
class ResCorMamba(nn.Module):
    """rescor_rens K=32 (spatial) + per-cell Mamba SSM (temporal, K=4 frames).

    Input:  (B, context_k, C, H, W) — most recent frame last.
    Output: (B, C, H, W)            — predicted next frame.
    """
    def __init__(self, in_channels=1, hidden_ch=16, cml_steps=15,
                 r_lo=3.57, r_hi=3.99, eps=0.3, beta=0.15,
                 seed=42, out_channels=None, use_sigmoid=True, kernel_size=3,
                 cml_K=32, context_k=4,
                 d_model=16, d_state=8, d_conv=4, expand=2,
                 use_stat_bank=False): ...
    def forward(self, x_seq): ...
    def param_count(self): ...
```

Uses `CML2DMultiR(K=32, gate_mode="uniform")` internally, identical instantiation to `ResCorRensStatBank`. `use_stat_bank=True` rebuilds the NCA conv1 input with mean+var+min+max (same stat-bank as `ResCorRensStatBank`).

### 5.3 Registry entries in `src/wmca/model_registry.py`

Import added:

```python
from wmca.modules.hybrid import (
    ...
    ResCorMamba,
    ...
)
```

Entries:

```python
"rescor_mamba": {
    "class": ResCorMamba,
    "description": (
        "rescor_rens K=32 spatial core + per-cell Mamba SSM over K_context=4 "
        "past frames. Multi-frame input (B, 4, C, H, W). Requires context_k=4 "
        "benchmark generation."
    ),
    "extra_kwargs": {"cml_K": 32, "context_k": 4, "use_stat_bank": False},
},
"rescor_mamba_stat": {
    "class": ResCorMamba,
    "description": (
        "rescor_mamba with stat-bank NCA (mean+var+min+max across K=32 rens bank)."
    ),
    "extra_kwargs": {"cml_K": 32, "context_k": 4, "use_stat_bank": True},
},
```

`create_model` already forwards matching-name kwargs via `inspect.signature`, so `context_k`, `d_model`, `d_state`, `d_conv` surface naturally to `create_model("rescor_mamba", ..., context_k=5)` if an experiment wants to sweep K.

---

## 6. Rollout-probe compatibility

The existing `dreamerv3_scaffolding/rollout_stability_probe.py` maintains single-frame state `x` and does:

```python
x = X_test[ti * T].unsqueeze(0)     # (1, C, H, W)
for s in range(max_h):
    pred = model(x)                  # rank-4 -> rank-4
    x = pred.clamp(0.0, 1.0)         # new single frame
```

For `rescor_mamba`, model input is rank-5. The probe needs a **rolling K-frame buffer**. Design:

### 6.1 Modified rollout loop (new function `rollout_rescor_mamba` in the probe script, switched on when `meta["context_k"] > 1`)

```python
K = meta["context_k"]                    # 4

# Seed the buffer with K ground-truth frames from the test trajectory.
# Trajectory i starts at X_test[i*T]; the K frames are [X[i*T], Y[i*T], ..., Y[i*T + K-2]].
buffer = [X_test[ti * T]]
for k in range(K - 1):
    buffer.append(Y_test[ti * T + k])
buf = torch.stack(buffer, dim=0).unsqueeze(0)      # (1, K, C, H, W)

# Ground-truth alignment: after consuming K-1 Y's to build the buffer,
# the first predicted frame corresponds to frame index K in the trajectory,
# i.e. Y_test[ti*T + K - 1]. Horizon h=1 evaluates the first prediction.
for s in range(max_h):
    pred = model(buf)                               # (1, C, H, W)
    gt   = Y_test[ti * T + K - 1 + s].unsqueeze(0)  # (1, C, H, W)
    ...accumulate per_step_mse[s]...
    # Advance buffer: drop oldest, append prediction.
    buf = torch.cat([buf[:, 1:], pred.clamp(0.0, 1.0).unsqueeze(1)], dim=1)
```

### 6.2 Bookkeeping

* `N_STEPS = 105` is already large enough: we need `T >= (K-1) + max_h` = `3 + 100 = 103`. Fine.
* Probe's `n_test_traj = X_test.shape[0] // T` must use the *single-frame* T from the benchmark, not (T-K+1). **This is a subtle trap**: `_make_k_frame_pairs` produces `T - K + 1` pairs per trajectory, so `X_test.shape[0] // T` would be wrong. Fix: the probe's trajectory reconstruction **re-generates** the benchmark fresh (or stores `meta["context_k"]` + `meta["n_steps"]` and computes `n_pairs_per_traj = meta["n_steps"] - K + 1`). Cleanest path: generate the benchmark with `context_k=1` for the probe's data buffer, and with `context_k=4` only for training. That way the probe keeps its single-frame ground-truth indexing and constructs the K-frame rolling buffer itself. Downside: we generate data twice. Upside: no index-math bugs. **Choose this path.**

### 6.3 Seeding policy

Initial K frames come from the **ground-truth test trajectory** (frames 0..K-1), consistent with how every other rollout-eval in this project seeds the first step. This is not "giving the model K steps of free ground truth"; it is the initialization every SSM uses. Subsequent frames are pure model predictions.

---

## 7. Mamba dep — inline vs pip

### 7.1 Decision: implement a **minimal selective-scan inline** in `src/wmca/modules/mamba_block.py`. Do NOT add `mamba-ssm` as a dep.

Evidence:
* `pyproject.toml` lists only `torch, numpy, wandb, python-dotenv`. `uv pip list | grep -i mamba` returns nothing — not currently installed.
* `mamba-ssm` ships the selective-scan CUDA kernel as its hot path (`selective_scan_cuda`). Without CUDA it falls back to a reference PyTorch impl which is itself ~80 LOC and requires the same math we'd write anyway.
* The project runs on CPU (per user brief and per `experiments/` scripts — `train_model` defaults to `device="cpu"`).
* Adding an optional CUDA dep for a block we can write in ~100 LOC of pure PyTorch adds setup friction for a negative result we might not want to keep.
* The inline version can be validated against the official reference impl on a single GPU box later if needed.

### 7.2 Inline selective-SSM math

Following Gu & Dao 2023 §3.2 (notation as in the `mamba-ssm` reference code):

```
Given x ∈ R^(B, L, d_model), per time step l:

  # Input projection splits into two halves of width d_inner = expand · d_model.
  xz = in_proj(x)                              # (B, L, 2·d_inner)
  x_in, z = xz.chunk(2, dim=-1)                # each (B, L, d_inner)

  # Short causal depthwise conv along L (kernel d_conv), then SiLU.
  x_in = silu(depthwise_causal_conv1d(x_in))   # (B, L, d_inner)

  # Per-step selective parameters.
  x_dbl = x_proj(x_in)                         # (B, L, dt_rank + 2·d_state)
  dt_raw, B_, C_ = split(x_dbl, [dt_rank, d_state, d_state])
  dt = softplus(dt_proj(dt_raw))               # (B, L, d_inner) — positive step sizes
  A  = -exp(A_log)                             # (d_inner, d_state), strictly negative
  #   A_log is a learnable tensor initialised from the official "S4D real" init.

  # Discretize: per (b, l, d, n):
  #   ΔA = exp(dt * A),  ΔB = dt * B_
  deltaA = exp(einsum("bld,dn->bldn", dt, A))              # (B, L, d_inner, d_state)
  deltaB = einsum("bld,bln->bldn", dt, B_)                  # (B, L, d_inner, d_state)

  # Sequential scan over L with hidden state h ∈ R^(B, d_inner, d_state):
  h = zeros(B, d_inner, d_state)
  outs = []
  for l in range(L):
      h = deltaA[:, l] * h + deltaB[:, l] * x_in[:, l, :, None]  # selective recurrence
      y_l = einsum("bdn,bn->bd", h, C_[:, l])                    # read-out
      outs.append(y_l)
  y = stack(outs, dim=1)                                         # (B, L, d_inner)
  y = y + D * x_in                                               # skip (per-channel)
  y = y * silu(z)                                                # gating

  out = out_proj(y)                                              # (B, L, d_model)
```

For our case, **L = context_k = 4**. A Python for-loop with 4 iterations is fine on CPU — no need for the `associative_scan` optimization. Tensors stay small: `(B·H·W=B·256, 4, d_inner=32)` ≈ `256·B·4·32 = 32 k B` floats per activation tensor, well within memory.

Initializations (matching `mamba-ssm/mamba_ssm/modules/mamba_simple.py`):
* `A_log = log(repeat(arange(1, d_state+1), "n -> d n", d=d_inner))` — diagonal, negative, structured.
* `D = ones(d_inner)`.
* `dt_proj.bias` initialized to `inv_softplus(uniform(dt_min=0.001, dt_max=0.1))`.
* All other projections: default Kaiming.

### 7.3 Testing the inline block

Before declaring the arch ready, write one `test_mamba_block.py` (kept out of the main src tree, in `experiments/` or similar) that:
1. Asserts output shape `(B, L, d_model)` for random inputs.
2. Asserts gradient flows to every trainable param.
3. Optionally diffs against `mamba-ssm` reference impl on an installed GPU box (not required for CPU-only merge).

---

## 8. Open questions for the user

1. **d_inner = expand·d_model: expand=2 or expand=1?** expand=2 (~5.3 k total trained) matches the original Mamba paper; expand=1 (~4.0 k total trained) matches the spirit of our 321-param NCA discipline. Recommend **expand=1** for the first run — if MSE-step-1 doesn't improve materially vs rens K=32, expand=2 won't save it either.
2. **Zero-init path: should the Mamba branch be zero-init'd so the model starts as pure rens K=32?** Concretely: `out_proj.weight.zero_()` so at epoch 0, `temporal_feat` contributes nothing. This is the same zero-init-residual discipline that worked for `rescor_e3c` (`findings.md §55` caveat). Recommend **yes** (add as default); gives the cleanest A/B vs rens.
3. **Stat-bank or no stat-bank as the hero mamba variant?** Phase 1 showed `stat_no_var` is the tightest-distribution variant on ks; stat-bank + Mamba is the richest model. Recommend **start with `use_stat_bank=False` (mean-only)** for the step-1 / H=15 A/B; if it wins, sweep `stat_bank ∈ {off, on, no_var}` as follow-ups.
4. **Rollout-probe data duplication.** §6.2's choice to generate the benchmark once with `context_k=1` (for probe ground-truth buffer) and once with `context_k=4` (for training) duplicates data-gen cost by 2×. On heat/gs/ks at n_trajectories=200, grid=16, this is seconds — acceptable. Confirm OK.
5. **ks spatial shape.** ks is `(N, 1, 1, W)` — spatial dim collapses to width. The Mamba per-cell view becomes `(B*1*W, K=4, C=1)`. Still principled (each x-position gets its own scan), but there's no 2-D locality to exploit. Not a blocker, just noting the asymmetry.
6. **rescor_mamba_deep (out of scope)?** Given `rescor_rens_deep L=2` was a negative result, we should NOT stack rescor_mamba cells. Confirm we skip any depth-via-stacking variant for this arch.
7. **Training-epoch budget.** Phase 1 established 100 epochs as the honest baseline. Given the wider NCA input (extra `d_model=16` channels going into conv1), the `findings.md §56` caveat applies — at 30 epochs the Mamba channels won't drive. Recommend **100 epochs as the minimum** for any rescor_mamba report. Confirm.

---

## Acceptance criteria (for the post-implementation A/B)

* `rescor_mamba` trains to completion on heat / gs / ks / gol / rule110 / wireworld at 100 epochs, 3 seeds.
* Step-1 MSE **matches or beats** `rescor_rens K=32` on at least heat/gs/ks (the chaotic-PDE targets — the theoretical prediction).
* H=15 ratio (median(MSE_15 / MSE_1)) **drops** vs the Task #27 numbers (17× gs, 127× ks). Target: H=15 ratio < 5× on gs and < 10× on ks. If both fail, the temporal-context hypothesis is falsified and we revisit.
* Trained param count within **±20% of the 5.3 k estimate in §2** after real instantiation.
* Existing `rescor_rens` numbers (heat 5.75e-8, gol 95.95%, gs 2.20e-6, etc.) do not regress — achieved by keeping `context_k=1` as default in all `generate_*` functions.
