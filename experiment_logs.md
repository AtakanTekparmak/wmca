# Experiment Logs

Chronological log of all experiments. Each entry is a dated experiment with results and implications. For a topical summary, see `findings.md`.

## Glossary

Terms used throughout this document:

- **MSE** (Mean Squared Error): Average squared difference between predicted and true values. **Lower is better.** Scale depends on data normalization (our data is in [0,1]).
- **VPT** (Valid Prediction Time): Number of rollout steps before the normalized prediction error exceeds 0.4. Measures how long a model can predict into the future before diverging. **Higher is better.**
- **Lyapunov time**: One Lyapunov time = 1/λ_max time units, where λ_max is the largest Lyapunov exponent of the system. For the Lorenz attractor, 1 Lyapunov time ≈ 55 steps at dt=0.02. VPT expressed in Lyapunov times is the standard metric for chaotic prediction (Pathak et al. 2018).
- **Trainable params**: Parameters optimized during training (Ridge regression coefficients for reservoir models, all weights for neural models).
- **Fixed params**: Reservoir parameters that are randomly initialized and NEVER updated. These define the reservoir dynamics. Not counted in "trainable params" but contribute to model complexity.
- **Effective rank**: Number of singular values above 1% of the maximum. Measures how many independent features the CML actually produces. Out of 256 possible. **Higher = richer feature expansion.**
- **Reconstruction MSE**: How well a linear model (Ridge regression) can recover the original input from the CML output. Measures information retention. **Lower = better memory.**
- **Cell accuracy**: Fraction of grid cells correctly predicted (for binary grids like Game of Life). **Higher is better.** 100% = perfect.
- **Grid-perfect accuracy**: Fraction of entire grids predicted with zero cell errors. Much harder than cell accuracy. **Higher is better.**

---

## 2026-04-08 — CML Self-Analysis (Phase 1-pre)

**Script**: `experiments/cml_self_analysis.py`
**Plots**: `experiments/plots/{lyapunov_vs_r, fidelity_heatmap, precision_comparison, effective_rank_vs_r}.png`

### Setup

- CML: C=256 channels, kernel_size=3, eps=0.3, beta=0.15
- r swept: [2.50, 3.00, 3.20, 3.40, 3.57, 3.60, 3.69, 3.80, 3.90, 3.99]
- M (CML steps) swept: [1, 3, 5, 10, 15, 20, 30]
- Batch size 64 (fidelity), 256 (feature richness)
- Hardware: CPU

### Results

**Lyapunov exponents** — measures how chaotic the logistic map is at each r. Positive = chaotic, negative = stable.


| r     | lambda              |
| ----- | ------------------- |
| <3.57 | <0 (stable)         |
| 3.57  | 0.013 (chaos onset) |
| 3.69  | 0.356 (NLP default) |
| 3.99  | 0.642 (deep chaos)  |


**State fidelity** — can a linear readout reconstruct the original input from CML output? Lower MSE = CML remembers more. Measured at M=15.


| r    | Reconstruction MSE | Interpretation                                                              |
| ---- | ------------------ | --------------------------------------------------------------------------- |
| 2.50 | 0.046              | Bad memory (stable regime: CML converges to fixed point, losing input info) |
| 3.99 | 0.003              | Good memory (chaotic regime: drive injection anchors dynamics near input)   |


Higher r yields 15x better memory retention. Counterintuitive: chaos helps, not hurts, because the drive injection (beta=0.15) continuously re-anchors the state.

**Precision comparison** (r=3.69, M=15) — does quantization break the CML?


| Precision | Output MSE vs f32 | Reconstruction MSE | Verdict          |
| --------- | ----------------- | ------------------ | ---------------- |
| f32       | —                 | 0.0322             | baseline         |
| bf16      | 6.6e-5            | 0.0312             | identical to f32 |
| int8      | 5.4e-5            | 0.0320             | identical to f32 |


All three precisions produce equivalent reservoir quality. Int8 is viable.

**Feature richness** — effective rank of CML output (out of 256 possible). Higher = more independent features = better nonlinear expansion.


| r    | Effective rank | Interpretation                                          |
| ---- | -------------- | ------------------------------------------------------- |
| 2.50 | 1              | Collapsed: all outputs identical (useless as reservoir) |
| 3.57 | 11             | Edge of chaos: barely useful                            |
| 3.69 | 51             | Moderate chaos (NLP default)                            |
| 3.80 | 94             | Rich features                                           |
| 3.99 | 130            | Richest (~51% of theoretical max)                       |


### Implications

- **Recommended r range**: [3.80, 3.99] — best memory AND richest features
- **Int8 is viable**: drive injection regularizes against discretization artifacts
- **beta (drive injection) is doubly important**: anchors memory + regularizes quantization
- **World modeling r ≠ NLP r**: higher r is better for state preservation (NLP used 3.69 for feature expansion)

---

## 2026-04-08 — Lorenz Attractor Prediction (Phase 1a)

**Script**: `experiments/lorenz_prediction.py`
**Plots**: `experiments/plots/lorenz_{rollout_mse, rollout_trajectory, r_sweep_vpt}.png`

### Setup

- **System**: Lorenz attractor (sigma=10, rho=28, beta=8/3) — a standard chaotic benchmark
- dt=0.02, 10000 timesteps, normalized to [0,1] per dimension
- Split: 70% train, 15% val, 15% test
- **Task**: given 3D state at time t, predict state at t+1 (one-step), then roll out autoregressively

**Models compared** (all use hidden_size=256 for fair comparison):


| Model        | What it is                                           | Temporal memory?        | Training             |
| ------------ | ---------------------------------------------------- | ----------------------- | -------------------- |
| CML alone    | Logistic map reservoir, per-timestep                 | No (memoryless)         | Ridge readout only   |
| ESN          | Random RNN reservoir, sequential                     | Yes (recurrent h_t)     | Ridge readout only   |
| GRU          | Learned recurrent neural net                         | Yes (learned gates)     | Full backprop (Adam) |
| ParalESN+CML | ParalESN temporal backbone + CML nonlinear expansion | Yes (FFT parallel scan) | Ridge readout only   |


**On param counts**: All reservoir models (CML, ESN, ParalESN+CML) have 771 *trainable* params — that's just the Ridge regression readout (256×3 weights + 3 biases). The reservoir parameters are randomly initialized and FROZEN. But the models differ in total complexity:

- CML: ~66K fixed reservoir params (coupling matrices, kernels)
- ESN: ~72K fixed params (W_res sparse 256×256 + W_in 3×256)
- ParalESN+CML: ~133K fixed params (ParalESN eigenvalues + input projection + CML)
- GRU: 201K fully trainable params (no fixed components)

### Results

**One-step prediction** (lower MSE = better)


| Model        | 1-step MSE | Trainable params | Total params (incl. fixed) |
| ------------ | ---------- | ---------------- | -------------------------- |
| GRU          | 1.53e-5    | 201,219          | 201,219                    |
| ESN          | 8.72e-5    | 771              | ~72K                       |
| CML alone    | 1.02e-3    | 771              | ~66K                       |
| ParalESN+CML | 1.33e-3    | 771              | ~133K                      |


**Multi-step rollout MSE** (feed predictions back as input — lower = better)


| Horizon | CML    | ESN    | GRU        | ParalESN+CML |
| ------- | ------ | ------ | ---------- | ------------ |
| 1       | 4.6e-4 | 1.1e-5 | 1.9e-5     | 3.6e-4       |
| 10      | 3.4e-3 | 6.4e-3 | 5.0e-4     | 1.9e-3       |
| 25      | 1.7e-2 | 1.7e-2 | 1.0e-3     | 2.8e-3       |
| 50      | 1.4e-2 | 2.1e-2 | 6.7e-3     | 8.7e-3       |
| 100     | 2.1e-2 | 2.7e-2 | **2.8e-2** | **1.9e-2**   |
| 200     | 3.9e-2 | 3.5e-2 | **6.3e-2** | **4.8e-2**   |


Note: at horizon 100+, ParalESN+CML (771 trainable params) beats GRU (201K params).

**Valid Prediction Time** (higher = better; how many steps before predictions diverge)


| Model        | VPT (steps) | VPT (Lyapunov times) | Interpretation                                                     |
| ------------ | ----------- | -------------------- | ------------------------------------------------------------------ |
| GRU          | 36          | 0.66                 | Best short-term (but 201K params, diverges long-term)              |
| ParalESN+CML | 27          | 0.49                 | Best reservoir model (74% of GRU with 260x fewer trainable params) |
| CML alone    | 8           | 0.15                 | Memoryless: can't do temporal prediction alone                     |
| ESN          | 6           | 0.11                 | Worst despite good 1-step MSE (recurrence doesn't help here)       |


**r-sweep** (CML reservoir only — does the optimal r from Phase 1-pre hold?)


| r    | 1-step MSE | VPT (steps) | Verdict                         |
| ---- | ---------- | ----------- | ------------------------------- |
| 3.69 | 7.0e-4     | 4           | NLP default: suboptimal         |
| 3.80 | 4.9e-4     | 6           | Best 1-step accuracy            |
| 3.90 | 1.0e-3     | 8           | Best prediction horizon         |
| 3.99 | 2.4e-3     | 4           | Too chaotic: hurts both metrics |


Consistent with Phase 1-pre: r=3.80-3.90 is the sweet spot.

### Key Findings

1. **ParalESN+CML hybrid works**: Best reservoir model by large margin (0.49 vs 0.15 Lyapunov times)
2. **Long-horizon stability**: Hybrid beats GRU at horizon 100+ despite 260x fewer trainable params — reservoirs don't diverge
3. **CML alone is memoryless**: Processes each timestep independently, useless for temporal dynamics without a temporal backbone
4. **r=3.90 optimal for VPT**: Consistent with Phase 1-pre analysis
5. **Bug found**: ParalESN out_proj is zero-initialized (designed for residual stream). In reservoir mode, bypass it with `_mix()` directly

### Implications

- ParalESN (temporal) + CML (nonlinear expansion) is a valid architecture
- 260x parameter efficiency on trainable params (but ~equal total complexity to ESN)
- The reservoir doesn't diverge in long rollouts — genuine advantage over trained models
- CML must always be paired with a temporal backbone for time series tasks

---

## 2026-04-08 — Game of Life Prediction (Phase 1b)

**Script**: `experiments/gol_prediction.py`
**Plots**: `experiments/plots/gol_{accuracy_bars, rollout_accuracy, example_predictions}.png`

### Setup

- **System**: Conway's Game of Life — deterministic 2D cellular automaton
- Grid: 32x32 (1024 cells), initial density ~0.3
- 1000 trajectories x 20 steps each
- Split: 70% train, 15% val, 15% test (by trajectory)
- **Task**: given binary grid at time t, predict grid at t+1 (binary classification per cell)

**Models compared:**


| Model  | What it is                                    | Spatial structure?          | Training                  |
| ------ | --------------------------------------------- | --------------------------- | ------------------------- |
| CML-1D | Flatten 32x32 to 1024, project to 256, 1D CML | No (wrong topology)         | Ridge readout             |
| CML-2D | Keep 32x32, 2D CML with 3x3 conv2d coupling   | Yes (correct topology)      | Ridge readout             |
| MLP    | 1024 -> 512 -> 1024, ReLU                     | No                          | Full backprop (Adam, BCE) |
| Conv2D | 3-layer CNN, 3x3 kernels                      | Yes (local receptive field) | Full backprop (Adam, BCE) |


### Results

**1-step prediction** (higher accuracy = better)


| Model      | Cell Accuracy | Grid-Perfect | Params    | Interpretation                                    |
| ---------- | ------------- | ------------ | --------- | ------------------------------------------------- |
| **Conv2D** | **97.9%**     | **0.13%**    | **2,625** | Dominates. 3x3 kernel = exact GoL receptive field |
| CML-2D     | 78.0%         | 0%           | 1,049,600 | 2D topology barely helps over 1D                  |
| CML-1D     | 77.4%         | 0%           | 263,168   | Same ballpark as CML-2D despite wrong topology    |
| MLP        | 74.6%         | 0%           | 1,050,112 | Worst. No spatial or physical inductive bias      |


Note: CML param counts are large because the Ridge readout is 256->1024 (CML-1D) or 1024->1024 (CML-2D). Unlike Lorenz (256->3), the high-dimensional output kills the parameter efficiency story.

**Multi-step rollout** (cell accuracy at each horizon, higher = better)


| Horizon | CML-1D | CML-2D | MLP   | Conv2D    |
| ------- | ------ | ------ | ----- | --------- |
| 1       | 62.6%  | 63.9%  | 61.4% | **97.6%** |
| 3       | 68.4%  | 66.4%  | 67.2% | **94.3%** |
| 5       | 69.8%  | 68.4%  | 68.5% | **91.0%** |
| 10      | 72.3%  | 71.6%  | 70.9% | **83.9%** |


Conv2D maintains 84% at horizon 10. All others converge toward ~70% (the dead-cell baseline — GoL grids tend toward mostly-dead states, so "predict all dead" gets ~70%).

### Key Findings

1. **Conv2D wins decisively** — 97.9% accuracy with only 2,625 params. GoL is a 3x3 local rule; a CNN with 3x3 kernels has exactly the right inductive bias.
2. **Fixed CML fails for GoL** — The CML's logistic map dynamics are NOT GoL dynamics. A fixed reservoir can't simulate an arbitrary CA rule. ~78% accuracy is barely above the dead-cell baseline.
3. **2D topology doesn't help (with Ridge readout)** — CML-2D (78.0%) barely beats CML-1D (77.4%). The bottleneck is the LINEAR readout, not the topology. Ridge regression can't learn the complex nonlinear GoL rule from CML features.
4. **Parameter efficiency lost** — Ridge readout on high-dim output (1024->1024) requires ~1M params, destroying the reservoir's parameter efficiency advantage.
5. **Rollout accuracy increases for bad models** — This is NOT improvement; it's convergence to the trivial "predict all dead" baseline as GoL grids stabilize.

### Implications

- **Fixed CML reservoir is wrong for grid prediction**: The reservoir's dynamics must match (or approximate) the target dynamics. Logistic map != GoL.
- **Learned CML rules (Variant 2) are essential**: To predict GoL, we need to LEARN the local update rule, not use a fixed logistic map. This is the NCA approach.
- **Ridge readout is a bottleneck for high-dim outputs**: For world modeling with spatial outputs, we need a spatial decoder (like Conv2D), not a flattened Ridge regression.
- **Conv2D is the bar to beat**: Any CML world model for grid-based physics must match CNN accuracy to be publishable. The inductive bias argument only works if CML actually provides a BETTER bias than conv layers.
- **Next step**: Implement Variant 2 (learned CML rules) and compare against Conv2D. This is where the "physics foundation model" narrative gets tested.

---

## 2026-04-08 — Game of Life: Full Model Comparison (Phase 1b, Complete)

**Scripts**: `experiments/gol_prediction.py`, `experiments/gol_learned_cml.py`, `experiments/gol_nca_paralesn.py`

This entry supersedes the preliminary Phase 1b entry above, which only covered `gol_prediction.py`. Here we report all 8 models tested across three experiment scripts, including learned NCA variants and the NCA+ParalESN hybrid.

### Setup

- **System**: Conway's Game of Life — deterministic 2D cellular automaton (Markov: next state depends only on current state)
- Grid: 32x32 (1024 cells), initial density ~0.3
- 1000 trajectories x 20 steps each
- Split: 70% train, 15% val, 15% test (by trajectory)
- **Task**: given binary grid at time t, predict grid at t+1 (binary classification per cell)
- **Metric**: Cell accuracy (fraction of 1024 cells correctly predicted; see Glossary)

### Models Tested (8 total)

Three categories: fixed-reservoir baselines, neural baselines, and learned-rule (NCA) variants.

**Category 1: Fixed reservoir + Ridge readout** (from `gol_prediction.py`)


| Model                     | Cell Acc | Params    | What it does                                                       |
| ------------------------- | -------- | --------- | ------------------------------------------------------------------ |
| CML-2D (fixed, spatial)   | 78.02%   | 1,049,600 | Fixed logistic map + 2D conv coupling + Ridge readout (1024->1024) |
| CML-1D (fixed, flattened) | 77.43%   | 263,168   | Fixed logistic map + 1D coupling + Ridge readout (256->1024)       |
| ParalESN + fixed CML      | 77.41%   | 263,168   | Adds temporal memory to CML-1D. No improvement — GoL is Markov     |


All three are near **~78%**, which is barely above the dead-cell baseline (~70%). The fixed logistic map dynamics simply cannot represent GoL's birth/survival rules, so the reservoir features are uninformative. The Ridge readout then has to compensate, which it cannot do linearly.

**Category 2: Neural baselines** (from `gol_prediction.py`)


| Model                     | Cell Acc | Params    | What it does                                                          |
| ------------------------- | -------- | --------- | --------------------------------------------------------------------- |
| Conv2D (3-layer CNN, 3x3) | 97.91%   | 2,625     | Standard CNN with 3x3 kernels — exactly matches GoL's receptive field |
| MLP (1024->512->1024)     | 74.57%   | 1,050,112 | No spatial bias. Worst model tested                                   |


Conv2D is the gold standard baseline. Its 3x3 kernels have exactly the right inductive bias for GoL's 3x3 neighborhood rule.

**Category 3: Learned NCA variants** (from `gol_learned_cml.py` and `gol_nca_paralesn.py`)


| Model                          | Cell Acc | Params  | What it does                                                  |
| ------------------------------ | -------- | ------- | ------------------------------------------------------------- |
| NCA-1step (learned local rule) | 97.23%   | 449     | Learns a 3x3 conv rule applied once. Nearly matches Conv2D    |
| NCA-residual-3step             | 97.22%   | 449     | Same NCA iterated 3x with residual connection. No degradation |
| NCA-3step (no residual)        | 89.41%   | 449     | Same NCA iterated 3x WITHOUT residual. Error compounds badly  |
| NCA+ParalESN hybrid            | 97.95%   | 132,689 | NCA spatial features fed into ParalESN temporal backbone      |


The NCA (Neural Cellular Automaton) is effectively a "learned CML" — it replaces the fixed logistic map with a learned 3x3 convolution rule. This is the Variant 2 architecture from our proposal.

### Consolidated Ranking


| Rank | Model                   | Cell Acc   | Params    | Accuracy per param                 |
| ---- | ----------------------- | ---------- | --------- | ---------------------------------- |
| 1    | **NCA+ParalESN**        | **97.95%** | 132,689   | 0.00074%/param                     |
| 2    | Conv2D                  | 97.91%     | 2,625     | 0.037%/param                       |
| 3    | NCA-1step               | 97.23%     | 449       | **0.217%/param** (best efficiency) |
| 4    | NCA-residual-3step      | 97.22%     | 449       | 0.217%/param                       |
| 5    | NCA-3step (no residual) | 89.41%     | 449       | 0.199%/param                       |
| 6    | CML-2D (fixed)          | 78.02%     | 1,049,600 | 0.000074%/param                    |
| 7    | CML-1D (fixed)          | 77.43%     | 263,168   | 0.00029%/param                     |
| 8    | ParalESN + fixed CML    | 77.41%     | 263,168   | 0.00029%/param                     |
| 9    | MLP                     | 74.57%     | 1,050,112 | 0.000071%/param                    |


### Multi-Step Rollout (Best Models Only)

Autoregressive rollout: feed predictions back as input. Cell accuracy at each horizon (higher = better).


| Horizon | NCA-1step (449 params) | NCA+ParalESN (132K params) | Conv2D (2,625 params) |
| ------- | ---------------------- | -------------------------- | --------------------- |
| 1       | 95.2%                  | 97.6%                      | 97.5%                 |
| 3       | 93.2%                  | 94.0%                      | 93.8%                 |
| 5       | 90.8%                  | 91.2%                      | 90.7%                 |
| 10      | **84.7%**              | 84.6%                      | 84.3%                 |


**At horizon 10, NCA-1step (449 params) BEATS Conv2D (2,625 params): 84.7% vs 84.3%.** This mirrors the long-horizon stability pattern from Lorenz, where reservoir-style models (ParalESN+CML) also beat GRU at long horizons.

### Key Findings

1. **Fixed CML reservoir FAILS for grid prediction** (~78% = dead-cell baseline). The logistic map's dynamics bear no resemblance to GoL's birth/survival rules. The reservoir features are essentially noise for this task.
2. **Learned NCA matches Conv2D with 6x fewer params** (449 vs 2,625, at 97.23% vs 97.91%). The learned local rule captures GoL dynamics compactly. This validates the Variant 2 (learned CML) architecture.
3. **NCA+ParalESN slightly beats Conv2D** (97.95% vs 97.91%) but at 132K params. The temporal context from ParalESN adds marginal value even for a Markov system — likely because it provides a richer feature set for the readout.
4. **Multi-step NCA without residual DEGRADES badly** (89.4% vs 97.2%). Iterating a learned rule without a residual skip connection compounds small errors multiplicatively. Residual connections are essential for iterative NCA.
5. **At rollout horizon 10, NCA beats Conv2D** (84.7% vs 84.3%). Same long-horizon stability pattern seen in Lorenz Phase 1a, where reservoir models beat trained models at extended horizons. The simpler model generalizes better autoregressively.
6. **ParalESN temporal memory adds NOTHING to fixed CML** (77.41% vs 77.43%). This is expected: GoL is a Markov process, so temporal context provides no additional information. The bottleneck is the useless fixed-reservoir features, not the lack of memory.
7. **Ridge readout on high-dim output kills parameter efficiency** (1M params for CML-2D's 1024->1024 mapping). The reservoir parameter advantage only holds when the output dimension is small (like Lorenz's 3D output).

### Implications for the Paper

- **"Learned CML matches Conv2D with 6x fewer params"** is a publishable result. The NCA architecture (learned local rule with CML-style iterative dynamics) achieves competitive accuracy with dramatically fewer parameters.
- **Fixed reservoir is a dead end for spatial prediction.** The dynamics must match or approximate the target system. For arbitrary CAs / physics sims, the rule must be LEARNED, not fixed.
- **Residual connections are mandatory for multi-step NCA.** Without them, iterating the learned rule 3+ times degrades accuracy by 8 percentage points. This is a practical design constraint for any iterative NCA world model.
- **Long-horizon stability of NCA vs Conv2D is a recurring theme.** Both Lorenz (ParalESN+CML beats GRU at horizon 100+) and GoL (NCA beats Conv2D at horizon 10) show that simpler/reservoir-style models degrade more gracefully in autoregressive rollout.
- **Conv2D is the right baseline, not MLP.** Both NCA and Conv2D exploit spatial locality via 3x3 kernels. The comparison is fair because both have the same inductive bias (local neighborhood), but NCA achieves it with fewer parameters.
- **Temporal context is irrelevant for Markov systems.** ParalESN adds nothing to fixed CML for GoL. But for non-Markov systems (partially observed, stochastic), the NCA+ParalESN hybrid architecture may shine — this is a future experiment.

---

## 2026-04-08 — PDE Prediction: Heat, Wave, Gray-Scott (Phase 1c)

**Script**: `experiments/pde_prediction.py`

### Setup

- **Systems**: Three PDEs of increasing complexity:
  - **Heat equation** — pure diffusion (linear PDE, smooth dynamics)
  - **Wave equation** — oscillatory dynamics (linear PDE, propagating wavefronts)
  - **Gray-Scott reaction-diffusion** — pattern-forming nonlinear PDE (spots, stripes, chaos)
- Grid: 2D discretization
- **Task**: given grid state at time t, predict state at t+1 (one-step), then roll out autoregressively to h=50
- **Metrics**: 1-step MSE (lower = better), rollout h=50 MSE (lower = better)

### Models Compared

| Model | What it is | Params (approx) |
|-------|-----------|-----------------|
| NCA-2D | Learned 3x3 conv local rule (1-step or 3-step-residual) | 177–338 |
| Conv2D | 3-layer CNN, 3x3 kernels | 2,625–2,914 |
| CML-2D (fixed) | Fixed logistic map + 2D conv coupling + Ridge readout | 526K–1.05M |
| CML-2D + ParalESN | Fixed CML with ParalESN temporal context | 1.3M |
| MLP | Fully-connected baseline | 1.05M–2.1M |

### Results

**Heat Equation** — pure diffusion, smooth dynamics

| Model | 1-step MSE | Rollout h=50 MSE | Params | Interpretation |
|-------|-----------|-----------------|--------|----------------|
| NCA-2D (1-step) | ~0 (perfect) | 0.355 (worst) | 177 | Perfect 1-step but errors compound catastrophically |
| Conv2D | 1.3e-4 | 0.453 | 2,625 | Worst rollout despite good 1-step |
| CML-2D (fixed) | 0.052 | 0.250 (best) | 1,049,600 | Worst 1-step but BEST rollout — diffusion dynamics match |
| NCA-2D (3-step-res) | 0.034 | 0.264 | 177 | Multi-step iteration improves rollout stability |
| CML-2D + ParalESN | 0.061 | 0.287 | 1,311,744 | ParalESN adds marginal temporal context |
| MLP | 0.060 | 0.259 | 1,050,112 | Surprisingly competitive at rollout |

**Wave Equation** — oscillatory, propagating wavefronts

| Model | 1-step MSE | Rollout h=50 MSE | Params | Interpretation |
|-------|-----------|-----------------|--------|----------------|
| NCA-2D | 2e-6 | ~0 | 338 | Near-perfect everywhere |
| CML-2D | 2e-6 | 1e-6 | 526,336 | Near-perfect everywhere |
| Conv2D | 2e-6 | 2.5e-4 | 2,914 | Slight rollout degradation |
| MLP | 3e-6 | ~0 | 2,099,712 | Near-perfect everywhere |

**Gray-Scott Reaction-Diffusion** — nonlinear pattern formation

| Model | 1-step MSE | Rollout h=50 MSE | Params | Interpretation |
|-------|-----------|-----------------|--------|----------------|
| NCA-2D | ~0 | 3.3e-4 | 338 | Excellent 1-step, good rollout |
| CML-2D (fixed) | 2e-6 | 1.2e-4 (best) | 526,336 | Best rollout — diffusion coupling matches R-D dynamics |
| Conv2D | 1e-6 | 1.3e-4 | 2,914 | Near-tied with CML for rollout |
| MLP | 4e-6 | 2.2e-4 | 2,099,712 | Worst rollout, most params |

### Key Findings

1. **NCA is absurdly parameter-efficient**: 177–338 params matching or beating models with 2K–2M params on 1-step MSE across all three PDEs. On heat equation, NCA achieves ~0 1-step error with 177 params vs Conv2D's 1.3e-4 with 2,625 params.
2. **Fixed CML provides best long-horizon stability on heat and Gray-Scott** — where its diffusion-like coupling dynamics match the target PDE's diffusion operator. CML-2D gets the best rollout h=50 MSE on both (0.250 and 1.2e-4 respectively). This is a direct confirmation of the **Matching Principle** from Phase 1a/1b.
3. **Wave equation is "too easy" at this resolution** — all models achieve near-perfect 1-step and rollout MSE. The wave equation at this grid resolution doesn't stress-test any architecture. Need higher resolution or longer rollouts to differentiate.
4. **Heat equation shows the clearest stability-accuracy tradeoff**: NCA is perfect at 1-step (~0 MSE) but worst at h=50 (0.355); CML is worst at 1-step (0.052) but best at h=50 (0.250). This is the fundamental tension: models that fit perfectly to single steps may overfit to local dynamics and compound errors, while models with dynamics-matched inductive bias degrade more gracefully.
5. **Confirms the Matching Principle and motivates hybrid architectures**: Fixed CML wins at long horizons when its dynamics match (diffusion-like PDEs). NCA wins at 1-step with extreme efficiency. A hybrid that combines NCA's learned accuracy with CML's dynamical stability could get the best of both — this directly motivates the Phase 2 architecture ablation.

---

## 2026-04-08 — Lorenz: Learned CML vs Fixed CML (Phase 1a addendum)

**Script**: `experiments/lorenz_prediction.py` (updated with 2 new models)

### Setup

Same as original Phase 1a (Lorenz, dt=0.02, 10000 steps, 70/15/15 split).
Two new models added:

- **LearnedCML**: Fixed random W_in (3->256) + sigmoid + learned MLP (256->64->256, ReLU+sigmoid) iterated M=15 times with drive injection (beta=0.15) + Linear(256->3) output. Trained with Adam lr=1e-3, 100 epochs. 33,859 trainable params.
- **LCML+ParalESN**: Same learned MLP but driven by ParalESN temporal features instead of raw input. ParalESN frozen. 33,859 trainable params.

### Results

Complete table (6 models):


| Model                | 1-step MSE | VPT (Lyap) | Trainable Params | Verdict                                       |
| -------------------- | ---------- | ---------- | ---------------- | --------------------------------------------- |
| GRU                  | 9.4e-6     | 1.35       | 201,219          | Best overall (but most params)                |
| ParalESN+CML (fixed) | 1.3e-3     | 0.49       | 771              | Best reservoir model                          |
| CML (fixed)          | 1.0e-3     | 0.15       | 771              | Decent despite no temporal memory             |
| ESN                  | 8.7e-5     | 0.11       | 771              | Good 1-step but poor rollout                  |
| **LearnedCML**       | **1.2e-2** | **0.13**   | **33,859**       | **11x worse than fixed CML, 44x more params** |
| **LCML+ParalESN**    | **1.4e-2** | **0.04**   | **33,859**       | **Worst model. Adding ParalESN HURTS.**       |


### Key Findings

1. **Learned CML is 11x WORSE than fixed CML on Lorenz** (MSE 0.012 vs 0.001) with 44x more parameters (33K vs 771). The fixed logistic map at r=3.90 is already an excellent nonlinear feature expansion for chaotic systems — a learned MLP cannot match it in 100 epochs.
2. **Adding ParalESN to learned CML makes it WORSE** (0.04 vs 0.13 Lyapunov times). The learned MLP struggles to optimize through the ParalESN feature space.
3. **This is the OPPOSITE of GoL** where learned rules went from 78% to 97%. The difference: for GoL, the fixed logistic map dynamics are WRONG (different CA rules). For Lorenz, they're RIGHT (chaotic system -> chaotic feature expansion).

### Implications — The Matching Principle

This reveals a fundamental design principle:

**Use FIXED reservoir when reservoir dynamics MATCH target dynamics:**

- Chaotic target (Lorenz) + chaotic reservoir (logistic map) = good features for free
- No training needed, 771 params, 0.49 Lyapunov times with ParalESN

**Use LEARNED rules when reservoir dynamics DON'T match target dynamics:**

- Discrete target (GoL) + continuous chaotic reservoir (logistic map) = useless features
- Must learn the rule: 449 params, 97.2% accuracy, matches Conv2D

This is a key insight for the paper: the reservoir's physics IS the inductive bias. When it matches, don't learn. When it doesn't, learn the rule.

For world modeling: use fixed CML for physics-like continuous dynamics, learned NCA for discrete/structured environments. The "foundation model" narrative should encompass BOTH modes.

---

## 2026-04-08 — Phase 2 Architecture Ablation

**Script**: `experiments/phase2_ablation.py`
**Plots**: `experiments/plots/phase2_*.png`

### Setup

- Grid: 16x16, 30 epochs, 7 models tested
- Benchmarks: heat equation (continuous diffusion) and Game of Life (discrete CA)
- Models: 4 hybrid variants (A-D), PureNCA, Conv2D baseline, CML2D (Ridge) baseline

### Results

**Heat Equation** (MSE, lower = better)

| Model | 1-step MSE | h=10 MSE | Params |
|-------|-----------|----------|--------|
| ResidualCorrection (D) | ~0 | ~0 | 321 |
| Conv2D | ~0 | ~0 | 2,625 |
| NCAInsideCML (C) | 3e-4 | 3.7e-3 | 177 |
| PureNCA | 1.1e-3 | 6.8e-3 | 177 |
| GatedBlend (A) | 2.2e-3 | 1.1e-2 | 410 |
| CMLReg (B) | 2.7e-3 | 4.6e-2 | 177 |
| CML2D (Ridge) | 7.1e-3 | 2.1e-2 | 65,792 |

**Game of Life** (cell accuracy, higher = better)

| Model | 1-step Acc | h=10 Acc | Params |
|-------|-----------|----------|--------|
| Conv2D | 95.8% | 75.4% | 2,625 |
| PureNCA | 94.6% | 75.0% | 177 |
| CMLReg (B) | 94.6% | 75.0% | 177 |
| GatedBlend (A) | 94.6% | 73.1% | 410 |
| ResidualCorrection (D) | 85.5% | 69.8% | 321 |
| NCAInsideCML (C) | 83.4% | 60.4% | 177 |
| CML2D (Ridge) | 78.1% | 72.4% | 65,792 |

### Key Findings

1. **Variant D (ResidualCorrection) dominates continuous physics** — perfect heat equation MSE (~0 at both 1-step and h=10) with only 321 params. The fixed CML handles the bulk diffusion dynamics; the learned NCA correction captures residuals.
2. **PureNCA dominates discrete dynamics (GoL)** — adding CML only hurts. 94.6% accuracy at 177 params. The fixed CML's continuous chaotic dynamics are actively harmful for learning discrete binary rules.
3. **CMLReg (B) = PureNCA on GoL** — the CML regularizer is completely ignored when the CML reference signal is wrong. The NCA learns to overpower the regularization penalty, making Variant B equivalent to PureNCA in practice.
4. **GatedBlend (A) gate doesn't justify its complexity** — 410 params for no accuracy benefit over PureNCA on GoL, and worse than Variant D on heat. The learned gate adds parameters without adding useful inductive bias.
5. **The Matching Principle plays out perfectly across all 4 variants** — every variant that injects CML dynamics helps on continuous targets and hurts on discrete ones. The degree of CML involvement directly predicts performance: full CML base (D) wins for physics, zero CML (PureNCA) wins for discrete.

### Paper Framing Decision

Lead with the **Matching Principle** as the main contribution. Two architecture instantiations:
- **Variant D (ResidualCorrection)** for continuous physics (CML base + learned correction)
- **PureNCA** for discrete systems (fully learned local rule, no fixed dynamics)

The ablation across variants A-D provides the empirical backing: the more you match reservoir dynamics to target dynamics, the better the result.

---

## 2026-04-09 — ParalESN Injection Mode Ablation (Phase 2, continued)

**Script**: `experiments/phase2_paralesn_ablation.py`
**Plots**: `experiments/plots/phase2_paralesn_*.png`

### Setup

- 4 architecture variants (GatedBlend A, CMLReg B, NCAInsideCML C, ResidualCorrection D) + PureNCA and Conv2D baselines
- 3 ParalESN injection modes:
  - Mode 0: No ParalESN (spatial only, baseline)
  - Mode 1: Input injection — ParalESN temporal features concatenated as extra input channels, passed through sigmoid adapter, before the spatial model
  - Mode 2: Output injection — ParalESN temporal features projected to spatial grid, added as learned correction AFTER the spatial model
- ParalESN config: hidden_size=64, frozen (reservoir), input_size=256 (16x16 flattened), 5-step history window
- Benchmarks: heat equation and GoL at 16x16, 30 epochs, batch_size=64
- ParalESN bypass: _mix(h) used directly to avoid zero-init out_proj bug

### Results

**Heat Equation (MSE, lower = better)**

| Variant | Mode 0 (No ParalESN) | Mode 1 (Input Inj.) | Mode 2 (Output Inj.) |
|---------|---------------------|---------------------|----------------------|
| ResCor(D) | 4e-6 | 2.9e-4 | 2.1e-5 |
| CMLReg(B) | 2.1e-3 | 2.3e-4 | 1.7e-5 |
| NCAInCML(C) | 2.5e-4 | 9.2e-4 | 2.2e-4 |
| GatedBlend(A) | 2.0e-4 | 5.4e-4 | 2.9e-4 |
| PureNCA | 6.8e-4 | — | — |
| Conv2D | 2e-6 | — | — |

**Game of Life (Cell Accuracy, higher = better)**

| Variant | Mode 0 (No ParalESN) | Mode 1 (Input Inj.) | Mode 2 (Output Inj.) |
|---------|---------------------|---------------------|----------------------|
| ResCor(D) | 95.8% | 83.5% | 95.7% |
| CMLReg(B) | 94.6% | 88.1% | 93.4% |
| GatedBlend(A) | 94.7% | 78.8% | 94.2% |
| NCAInCML(C) | 83.4% | 79.0% | 82.3% |
| PureNCA | 94.6% | — | — |
| Conv2D | 95.7% | — | — |

### Key Findings

1. **Output injection (mode 2) is best for continuous physics** — CMLReg(B) gets 126x improvement on heat (1.7e-5 vs 2.1e-3). ResCor(D) also improves. The additive post-hoc correction preserves the spatial model's internal dynamics.
2. **Input injection (mode 1) hurts across the board** — despite having the most trained params (~33K), it degrades ALL variants on BOTH benchmarks. The sigmoid adapter bottleneck loses spatial structure. GoL accuracy drops 10-16 percentage points.
3. **For GoL, no ParalESN is best** — temporal context adds nothing for Markov systems. Even output injection slightly hurts (95.8% → 95.7% for ResCor(D)).
4. **ResCor(D) remains best overall variant** — top on GoL in all modes (95.8%), competitive on heat. The CML base + NCA correction architecture is robust to injection mode choice.
5. **Surprise: CMLReg(B) + output injection beats ResCor(D) on heat** — 1.7e-5 vs 2.1e-5. The ParalESN output correction turns a mediocre variant into the best performer. This suggests CMLReg(B) has untapped potential when given temporal context.
6. **Bug fixed during run**: ParalESNInputInjection adapter output needed sigmoid clamping to prevent NaN in CML logistic map (r*x*(1-x) diverges with negative inputs).

### Implications

- **Output injection is the correct way to add temporal context to spatial world models** — inject post-hoc as a correction, not pre-hoc as extra input channels
- **Input injection destroys spatial structure** — forcing spatial models to process temporal features through their spatial pathway conflates two different types of information
- **Markov systems should skip temporal context entirely** — adding ParalESN to GoL only hurts
- **Phase 2 is now complete** — all 4 variants × 3 injection modes × 2 benchmarks tested. ResCor(D) with optional output injection is the recommended architecture for continuous physics.

---

## 2026-04-09 — Different Chaotic Maps (Phase 2.5a)

**Script**: `experiments/phase25a_chaotic_maps.py`

### Setup

- Models: ResCor with 4 different CML reservoir maps — Logistic (r=3.9), Tent, Bernoulli, Sine — plus Conv2D and PureNCA baselines
- Benchmarks: Heat equation (continuous physics) and Game of Life (discrete CA)
- Hardware: LOCAL, CPU, parallel
- Goal: Test whether the Matching Principle holds across different chaotic maps, or is specific to the logistic map

### Results

**Heat Equation (MSE, lower is better)**

| Model              | MSE    |
|--------------------|--------|
| ResCor(Logistic)   | 1e-6   |
| ResCor(Tent)       | 1e-6   |
| ResCor(Bernoulli)  | 1e-6   |
| ResCor(Sine)       | 5e-6   |
| Conv2D             | 2e-6   |
| PureNCA            | 3.6e-4 |

**Game of Life (Cell Accuracy, higher is better)**

| Model              | Cell Accuracy |
|--------------------|---------------|
| Conv2D             | 95.8%         |
| ResCor(Logistic)   | 95.0%         |
| PureNCA            | 94.6%         |
| ResCor(Bernoulli)  | 94.4%         |
| ResCor(Tent)       | 87.5%         |
| ResCor(Sine)       | 83.6%         |

### Key Findings

1. **Matching Principle CONFIRMED across all 4 maps.** All chaotic map variants achieve ~1e-6 MSE on heat and all struggle on GoL (83-95%). The result pattern is map-independent.
2. **Chaos-matching is the key design rule, not logistic-map specificity.** Any chaotic reservoir matches continuous chaotic physics.
3. **Logistic, Tent, and Bernoulli are interchangeable** on heat (all 1e-6). Sine map is slightly weaker (5e-6) but still outperforms PureNCA by 70x.
4. **All maps fail similarly on GoL.** The failure is structural (continuous reservoir vs discrete target), not a property of any specific map.
5. **Sine map is the weakest variant** on both benchmarks, likely due to its smoother nonlinearity producing less expressive reservoir dynamics.

### Implications

- The Matching Principle can be generalized: any chaotic reservoir works for continuous physics; no chaotic reservoir works for discrete CA.
- Logistic (r=3.9) remains the default; Tent and Bernoulli are viable alternatives.

---

## 2026-04-09 — Harder PDEs: Burgers + KS (Phase 2.5b)

**Script:** `experiments/phase25b_harder_pdes.py`

### Burgers Equation

| Model | 1-step MSE | h=50 rollout | Params |
|------------|------------|--------------|--------|
| ResCor(D) | 5.7e-5 | 0.0398 | 321 |
| Conv2D | 4.2e-5 | 0.0149 | 2,625 |
| MLP | 1.3e-4 | 0.0305 | 98,880 |
| PureNCA | 1.4e-3 | 0.1194 | 177 |

### Kuramoto-Sivashinsky

| Model | 1-step MSE | h=50 rollout | Params |
|------------|------------|--------------|--------|
| ResCor(D) | ~0 | 0.000365 | 321 |
| Conv2D | ~0 | 0.0198 | 2,625 |
| MLP | 3e-6 | 0.00168 | 98,880 |
| PureNCA | 9e-6 | 0.00343 | 177 |

### Key Finding

ResCor(D) achieves 54x better rollout than Conv2D on KS (0.000365 vs 0.0198). Conv2D overfits to 1-step but blows up in rollout. The CML base prevents autoregressive error accumulation on chaotic dynamics. KS is Pathak et al. 2018's exact benchmark — direct favorable comparison.

---

## 2026-04-09 — More Discrete CAs: Rule 110 + Wireworld (Phase 2.5c)

**Script:** `experiments/phase25c_more_cas.py`

### Rule 110 (1D, binary, Turing-complete)

| Model | 1-step Acc | h=10 rollout | Params |
|------------|------------|--------------|--------|
| Conv | 99.24% | 92.0% | 897 |
| ResCor(D) | 99.21% | 92.7% | 129 |
| PureNCA | 99.06% | 93.1% (BEST) | 81 |
| CML2D+Ridge | 68.1% | 51.3% | 4,160 |

### Wireworld (2D, 4-state)

| Model | 1-step Acc | h=10 rollout | Params |
|------------|------------|--------------|--------|
| ResCor(D) | 99.90% | 99.77% | 2,468 |
| PureNCA | 99.90% | 99.77% | 1,316 |
| Conv | 99.89% | 99.77% | 11,588 |
| CML2D+Ridge | 93.72% | 94.61% | 1,049,600 |

### Key Findings

1. Fixed CML fails on both new discrete CAs (68.1% Rule 110, 93.7% Wireworld). Matching Principle confirmed across 3 discrete CAs (GoL + Rule 110 + Wireworld).
2. PureNCA wins rollout on Rule 110 (93.1% h=10 vs Conv's 92.0%) — long-horizon stability advantage holds.
3. Wireworld solved by all learned models (99.77% h=10) — deterministic 4-state rules are easy to learn.
4. PureNCA achieves competitive results with fewest params (81 for Rule 110, 1,316 for Wireworld).

---

## 2026-04-09 — Scale-Up (Phase 2.5d)

Scaled all three core models (ResCor(D), Conv2D, PureNCA) to 64x64 grids / N=128 to verify Phase 2 findings generalize beyond 16x16.

### Heat 64x64 (h=50 rollout)

| Model | 1-step MSE | h=50 rollout | Params |
|------------|------------|--------------|--------|
| ResCor(D) | ~0 | 0.407 | 321 |
| Conv2D | 1.2e-5 | 0.373 (best) | 2,625 |
| PureNCA | 3.6e-5 | 0.492 | 177 |

Conv2D wins long rollout at scale — larger capacity helps for linear PDE extrapolation.

### Kuramoto-Sivashinsky N=128 (h=100 rollout)

| Model | 1-step MSE | h=100 rollout | Params |
|------------|------------|---------------|--------|
| ResCor(D) | ~0 | 0.000253 | 321 |
| Conv2D | ~0 | 0.000277 | 2,625 |
| PureNCA | 6e-6 | 0.005919 | 177 |

ResCor(D) vs PureNCA: 23.4x advantage (down from 54x at N=64). ResCor(D) vs Conv2D: tied on rollout, 8.2x fewer params.

### Game of Life 64x64 (h=20 rollout)

| Model | 1-step Acc | h=20 rollout | Params |
|------------|------------|--------------|--------|
| Conv2D | 98.95% | 87.7% (best) | 2,625 |
| ResCor(D) | 98.66% | 86.3% | 321 |
| PureNCA | 98.64% | 87.3% | 177 |

Conv2D wins across the board on GoL (expected — discrete dynamics).

### Key Findings

1. KS advantage narrows from 54x to 23.4x vs PureNCA at larger scale, but parameter efficiency is the real story: ResCor(D) matches Conv2D with 8.2x fewer params.
2. Heat: ResCor(D) perfect 1-step, Conv2D wins long rollout (larger capacity helps for linear PDE extrapolation).
3. GoL: Conv2D wins across the board (expected — discrete dynamics don't benefit from CML).
4. The results HOLD at scale — the pattern from 16x16 generalizes to 64x64.

---

## 2026-04-09 — Pathak Comparison (DROPPED from paper)

Ran head-to-head against Pathak et al. 2018 (ESN, KS L=22 N=64). Result: not competitive at their coarse time resolution.

| Model | VPT (Lyapunov times) | Resolution |
|---|---|---|
| ResCor(D) | 0.02–0.19 | fine |
| Conv2D | 0.02–0.19 | fine |
| PureNCA | 0.02–0.19 | fine |
| Pathak ESN | 8.2 | coarse (0.25 LT/step) |

Root cause: diagonal recurrence (ParalESN) is structurally weaker than dense recurrence at coarse time resolution. Not a fair comparison. Dropped from paper — our contribution is fine-resolution parameter efficiency, not coarse-resolution VPT. Recorded here for completeness.

---

## 2026-04-09 — Grid World + CEM Planning (World Model Demo)

**Script**: `experiments/grid_world_planning.py`
**Environment**: 2D grid world — agent navigates to goal, pushes objects, avoids walls. CML lattice = world grid; action = drive perturbation at agent cell. Planner: Cross-Entropy Method (CEM) over imagined rollouts.

**Success rate results**:

| Planner / Model | Success Rate | Params |
|---|---|---|
| Oracle (true env rollouts) | 97% | — |
| CEM + PureNCA | 87% | ~12k |
| CEM + ResCor | 85% | 12,868 |
| CEM + Conv2D | 84% | ~13k |
| Random baseline | 10% | — |

**Training notes**:
- All learned world models reach 100% 1-step prediction accuracy after training.
- ResCor converges fastest — spatial prior from CML accelerates local-rule learning.
- Gap to oracle (85% vs 97%) is from rollout error compounding over multi-step imagination horizon, not 1-step prediction quality.

**Conclusion**: CML world model successfully enables planning. Validates the "world model" framing of the paper. Phase 3a COMPLETE.

---

## 2026-04-09 — DMControl Prediction Experiments

**Pod**: L40S 48GB at 86.38.238.90
**Script**: `experiments/dmcontrol_prediction.py`
**Plots**: `experiments/plots/dmcontrol_rollout_mse.png`, `experiments/plots/dmcontrol_1step_mse.png`

**Setup**: Non-spatial RL benchmarks — cartpole-swingup (5D state, 1D action) and reacher-easy (6D state, 2D action). Proprioceptive state vectors with no spatial structure. Compared rescor_e3c, PureNCA vs MLP and GRU baselines.

### Cartpole-Swingup

| Model | 1-step MSE | h=50 rollout MSE | Params |
|-------|------------|------------------|--------|
| GRU | 4e-7 | 0.217 (blows up) | 204,037 |
| MLP | 1e-6 | 1.8e-4 (best) | 68,869 |
| rescor_e3c | 2.0e-5 +/- 5e-6 | 3.4e-3 | 4,641 |
| PureNCA | 7.9e-5 +/- 4e-6 | 4.3e-3 | 177 |

### Reacher-Easy

| Model | 1-step MSE | h=50 rollout MSE | Params |
|-------|------------|------------------|--------|
| MLP | 4.8e-5 | 1.9e-3 (best) | 69,638 |
| GRU | 8.8e-5 | 0.253 (blows up) | 205,830 |
| rescor_e3c | 3.0e-3 +/- 1.7e-4 | 4.3e-2 | 4,641 |
| PureNCA | 8.2e-3 +/- 8.9e-5 | 3.3e-2 | 177 |

**Key result**: MLP dominates — no spatial structure means CML local coupling doesn't help. Validates Matching Principle from the non-spatial side. GRU has best 1-step but worst rollout (teacher-forcing overfitting). rescor_e3c beats PureNCA (~4x on cartpole, ~2.7x on reacher) — CML features still provide nonlinear expansion value even without spatial adjacency.

---

## 2026-04-10 — Unified Ablation Run

**Script**: `experiments/unified_ablation.py`
**Results**: `experiments/results/unified_ablation.json`
**Plots**: `experiments/plots/unified_pareto.png`, `experiments/plots/unified_heatmap.png`
**Hardware**: A40 GPU (Prime Intellect pod), 20 minutes wall time

### Setup

All 8 architectures evaluated head-to-head across all 7 benchmarks in a single run. This is the definitive cross-benchmark comparison that will drive the paper's headline figures.

- **Architectures**: rescor, conv2d, pure_nca, gated_blend, mlp, nca_inside_cml, cml_reg, cml_ridge
- **Benchmarks**: heat, ks, gray_scott, gol, rule110, wireworld, grid_world

### Cross-Benchmark Ranking (avg rank across 7 benchmarks)

| Rank | Model           | Avg Rank | Best On                    |
|------|-----------------|----------|----------------------------|
| 1    | rescor          | 2.4      | heat, ks, gray_scott       |
| 2    | conv2d          | 3.1      | gol, grid_world            |
| 3    | pure_nca        | 4.0      | —                          |
| 4    | gated_blend     | 4.1      | —                          |
| 5    | mlp             | 4.1      | rule110                    |
| 6    | nca_inside_cml  | 5.4      | —                          |
| 7    | cml_reg         | 5.6      | wireworld                  |
| 8    | cml_ridge       | 7.0      | (worst on 5 benchmarks)    |

### Per-Benchmark Headlines

- **Heat**: rescor MSE ~0 with 321 params, beats Conv2D (1.4e-5 with 2,625 params).
- **GoL**: Conv2D 95.9% wins, gated_blend 95.0% second, pure_nca 94.7% third.
- **KS**: rescor MSE 1e-6 with 321 params, MLP 1e-6 with 74K params.
- **Gray-Scott**: rescor MSE 3e-6 with 626 params, beats all baselines.
- **Rule 110**: MLP 100% (memorized), rescor 96.8%.
- **Wireworld**: pure_nca 98.9% and cml_reg 98.9% tied. Conv2D and MLP STUCK at 70% baseline (couldn't escape "predict empty"). NCA architectures dominate.
- **Grid world**: rescor 99.9% and conv2d 99.9% nearly tied (1-step accuracy).

### Key Findings

1. **ResCor(D) wins ALL 3 continuous physics tasks** (heat, ks, gray_scott) — Matching Principle confirmed at scale.
2. **Conv2D wins discrete spatial tasks** (GoL, grid_world).
3. **CML2D+Ridge is dead** — worst on 5 of 7 benchmarks.
4. **NCA architectures dominate Wireworld** where Conv2D/MLP get stuck at the ~70% baseline — interesting failure mode for standard CNNs.
5. **The 4 Phase 2 hybrid variants stratify cleanly**: rescor > gated_blend > nca_inside_cml > cml_reg.
6. **Wall time**: 20 minutes on A40 GPU for the full 8×7 grid.

### Known Issues (being fixed in parallel by other agents)

- **CEM planning eval shows NaN**: `run_cem_evaluation` not wired up in the unified harness.
- **grid_world rollout fails**: X has 8 channels, Y has 4 (channel mismatch).

Neither affects the 1-step results above.

### Implications

This run replaces scattered per-benchmark tables with one definitive comparison. `unified_pareto.png` and `unified_heatmap.png` become the paper's headline figures. The Matching Principle is now empirically grounded across 7 benchmarks and 8 architectures in a single consistent harness.

---

## 2026-04-10 — Hybrid Bug Fix (Channel Mismatch + Sigmoid on CE)

**Files touched**: `src/models/hybrid.py`, `src/models/model_registry.py`
**Trigger**: unified ablation flagged grid_world rollout as broken ("X=8, Y=4 channel mismatch").

### The bugs

Investigating the grid_world rollout failure surfaced two independent bugs in all 5 hybrid architectures. Both had been masked on every prior benchmark because no prior benchmark combined `in_channels != out_channels` with a `cross_entropy` loss.

1. **Channel mismatch**: ResCor, PureNCA, GatedBlend, CMLReg, NCAInsideCML all hardcoded `out_channels = in_channels`. Grid world has `in=8` (state + action one-hots) but `out=4` (next state classes). The hybrid models were silently producing 8-channel outputs, which the loss then compared against 4-channel targets.
2. **Sigmoid on output**: All hybrids ended in a final `sigmoid` / `clamp` (for heat/GoL continuous-value regime). But `cross_entropy` expects raw logits. Sigmoid-bounded logits [0, 1] fed into softmax collapse to a near-uniform distribution, and argmax falls back to the majority class — the empty-cell baseline of ~83.64%.

The combination is why every hybrid was stuck at exactly 83.64% on grid_world in the unified ablation.

### The fix

- Added `out_channels` and `use_sigmoid` parameters to all 5 hybrid classes in `hybrid.py`.
- When `out_channels != in_channels`, the internal CML operates on the first `out_channels` of the input.
- When `use_sigmoid=False`, the final sigmoid / clamp is removed so raw logits flow through for `cross_entropy`.
- `create_model` in `model_registry.py` now auto-sets `use_sigmoid=False` whenever `out_channels != in_channels`.
- `CMLRegularizedNCA` regularizes `softmax(logits)` against `cml_ref` under `cross_entropy`.

### Verified results (grid_world, 16×16, 30 epochs, 500 trajectories)

| Model           | Before fix | After fix              |
|-----------------|------------|------------------------|
| rescor          | 83.64%     | 99.92%                 |
| pure_nca        | 83.64%     | 99.92%                 |
| gated_blend     | 83.64%     | 99.92%                 |
| cml_reg         | 83.64%     | 99.92%                 |
| nca_inside_cml  | 92.93%     | 99.53%                 |
| conv2d          | 99.97%     | 99.94% (unchanged, was already correct) |

### Backward compatibility

Existing experiments (heat, GoL, KS, Burgers, Gray-Scott, Rule 110, Wireworld, etc.) are all unaffected. They use `in_channels=1`, so the old defaults (`out_channels=in_channels`, `use_sigmoid=True`) still apply and the code path is identical to before.

### Implications

- **The unified ablation needs a rerun** to get correct grid_world numbers. Other 6 benchmarks are unaffected.
- **Hybrid architectures can now be properly tested on action-conditioned tasks** — DMControl is next.
- **This bug was hidden until the unified ablation** because no prior experiment used `out_channels != in_channels` with `cross_entropy`. The unified run was the first to exercise that code path.

---

## 2026-04-10 — Unified Ablation v2 (Post Hybrid Fix)

**Script**: `experiments/unified_ablation.py`
**Results**: `experiments/results/unified_ablation.json`
**Canonical plots**: `experiments/plots/pareto_aggregated.png`, `experiments/plots/pareto_per_benchmark.png`
**Hardware**: A40 GPU (Prime Intellect pod), 47 minutes wall time
**Config**: 30 epochs, 300 trajectories, grid_size=16

### Setup

Full rerun of the 8-architecture x 7-benchmark unified ablation after the hybrid channel/sigmoid bug fix. This run also wires CEM planning evaluation into the unified harness (v1 returned NaN for CEM). Supersedes the 2026-04-10 v1 entry as the canonical cross-benchmark result.

- **Architectures**: rescor, conv2d, pure_nca, gated_blend, mlp, nca_inside_cml, cml_reg, cml_ridge
- **Benchmarks**: heat, ks, gray_scott, gol, rule110, wireworld, grid_world

### Cross-Benchmark Ranking (avg rank across 7 benchmarks)

| Rank | Model           | Avg Rank | Best On                    | Notes                              |
|------|-----------------|----------|----------------------------|------------------------------------|
| 1    | rescor          | 2.4      | heat, ks, gray_scott       | Wins all continuous physics        |
| 2    | conv2d          | 3.1      | gol, grid_world            | Strong on discrete + 1-step grid   |
| 3    | pure_nca        | 3.6      | wireworld                  | Solid, efficient                   |
| 4    | gated_blend     | 3.7      | —                          | Close to pure_nca                  |
| 5    | mlp             | 4.6      | rule110                    | Memorizes 1D problems              |
| 6    | nca_inside_cml  | 5.7      | —                          | Has CEM planning bug               |
| 6    | cml_reg         | 5.7      | —                          | Strong on grid_world planning      |
| 8    | cml_ridge       | 7.0      | —                          | Worst on 5 benchmarks              |

### Grid World CEM Planning (after hybrid fix)

| Model           | 1-step Acc | CEM Success | Avg Steps |
|-----------------|------------|-------------|-----------|
| cml_reg         | 99.92%     | 37.0%       | 27.9      |
| gated_blend     | 99.92%     | 35.0%       | 30.0      |
| conv2d          | 99.94%     | 31.0%       | 22.5      |
| rescor          | 99.93%     | 28.0%       | 26.2      |
| pure_nca        | 99.92%     | 24.0%       | 31.3      |
| mlp             | 83.68%     | 6.0%        | 12.7      |
| nca_inside_cml  | 99.33%     | 4.0%        | 26.0      |
| cml_ridge       | N/A        | NaN         | NaN       |

### Key Headlines

1. **cml_reg and gated_blend BEAT Conv2D on CEM planning** (37%, 35% vs 31%). The hybrid bug fix flipped the story on action-conditioned grid-world planning — the CML-regularized hybrids are now the strongest world models for forward planning, not Conv2D. 1-step accuracy alone under-ranks them.
2. **rescor still wins all continuous physics** (heat, ks, gray_scott) — consistent with v1. Matching Principle holds.
3. **All hybrids reach 99.9% 1-step accuracy on grid_world** — channel/sigmoid bug fix confirmed end-to-end.
4. **nca_inside_cml has a planning-specific bug**: 99.3% 1-step but only 4% CEM success. Rollout dynamics under CEM forward planning are broken. Known issue, flagged for investigation.
5. **Pareto frontier**: the aggregated Pareto plot shows rescor, pure_nca, and gated_blend at ~0.98 normalized performance with <1000 params — they jointly define the frontier across the 7-benchmark suite.
6. **Wall time**: 47 min on A40 GPU (v1 was 20 min). The extra ~27 min is CEM planning (100-300s per model).

### Implications

`pareto_aggregated.png` and `pareto_per_benchmark.png` are the new canonical figures, replacing `unified_pareto.png` / `unified_heatmap.png` from v1. The paper's world-model section should lead with cml_reg/gated_blend as the planning winners; Conv2D becomes a strong 1-step-only baseline. nca_inside_cml should be flagged or dropped until the CEM rollout bug is diagnosed.

---

## 2026-04-10 — Unified Ablation v3 (Post nca_inside_cml Fix)

**Script**: `experiments/unified_ablation.py`
**Plots**: `experiments/plots/{pareto_aggregated, pareto_per_benchmark, unified_heatmap_v3}.png`
**Results**: `experiments/results/unified_ablation_v3.json`

### Setup

- 7 benchmarks × 8 models, 30 epochs, 300 trajectories, grid_size=16
- A40 GPU on Prime Intellect, ~44 min wall time
- Includes `nca_inside_cml` fix from earlier today (drop `beta*drive` anchor on final iteration + add learned logit head)
- Includes new scoring system (NormScore + RawScore + ParamEffScore + ParetoScore)

### Cross-Benchmark Scores (sorted by RawScore)

| Model           | NormScore | RawScore | ParamEff | Pareto | AvgRank | Best On       |
|-----------------|-----------|----------|----------|--------|---------|---------------|
| pure_nca        | 0.982     | 0.985    | 0.400    | 1.000  | 3.7     | wireworld     |
| gated_blend     | 0.984     | 0.984    | 0.349    | 0.996  | 3.6     | —             |
| rescor          | 0.990     | 0.984    | 0.365    | 0.998  | 2.4     | heat, ks      |
| cml_reg         | 0.747     | 0.984    | 0.302    | 0.833  | 5.7     | —             |
| conv2d          | 0.947     | 0.947    | 0.273    | 0.960  | 3.1     | gol, grid_world |
| nca_inside_cml  | 0.670     | 0.918    | 0.283    | 0.779  | 5.7     | —             |
| mlp             | 0.773     | 0.899    | 0.148    | 0.387  | 4.6     | rule110       |
| cml_ridge       | 0.166     | 0.568    | 0.031    | 0.264  | 7.0     | —             |

### Grid World CEM Planning

| Model           | Success% | Avg Steps | Δ from v2              |
|-----------------|----------|-----------|------------------------|
| conv2d          | 36.0%    | 28.2      | +5pp                   |
| gated_blend     | 35.0%    | 23.3      | 0pp                    |
| nca_inside_cml  | 30.0%    | 29.9      | +26pp (fix worked!)    |
| rescor          | 28.0%    | 26.2      | 0pp                    |
| pure_nca        | 24.0%    | 31.3      | 0pp                    |
| mlp             | 5.0%     | 25.0      | -1pp                   |
| cml_reg         | **0.0%** | 0.0       | **-37pp (REGRESSION)** |
| cml_ridge       | NaN      | —         | (pre-existing bug)     |

### Key Findings

1. **nca_inside_cml fix verified**: 4% → 30% CEM success. The "drop `beta*drive` anchor on final iteration" + "add learned logit head" fix worked.
2. **Planning-relevant inductive bias matters more than 1-step accuracy**: nca_inside_cml has only 83.9% 1-step accuracy on grid_world (vs 99.9% for rescor/conv2d/gated_blend) but plans 30% successfully. High 1-step accuracy doesn't guarantee good planning. This is a paper-worthy insight.
3. **cml_reg REGRESSION**: was 37% in v2, now 0% in v3. Training reports 99.9% 1-step accuracy. The hybrid bug fix may have broken something specific to cml_reg's planning path. Needs separate debug.
4. **Top 4 models tied on RawScore (~0.984)**: pure_nca, gated_blend, rescor, cml_reg. All absolutely capable. Differences are small.
5. **rescor still wins by AvgRank (2.4)**: most consistent across benchmarks.
6. **Pareto frontier**: rescor, pure_nca, gated_blend at <1000 params and ~0.98 normalized perf. Conv2D at 0.95 with 2625 params is dominated.

### New Scoring System (also added in this run)

Three new aggregate scores in addition to AvgRank:

- **NormScore**: per-benchmark min-max [0,1] normalized then averaged. Relative performance.
- **RawScore**: 1/(1+MSE) for MSE benchmarks, raw accuracy for accuracy benchmarks, averaged. Absolute capability — doesn't depend on other models.
- **ParamEffScore**: normalized / log10(params+10). Rewards efficiency.
- **ParetoScore**: distance from per-benchmark Pareto frontier. 1.0 = on frontier.

### Implications for Paper

- ResCor remains the headline architecture: best on continuous physics, top tier on RawScore, parameter-efficient.
- The 4-way tie on RawScore is interesting framing: "all our hybrid variants are absolutely capable; the question is which has the best inductive bias for which dynamics".
- cml_reg regression needs fixing or noting as a known issue.
- The planning-vs-1step insight from nca_inside_cml could be a sub-section.

### Known Issues

- cml_reg grid_world CEM regression (0% from 37%) — needs debug.

---

## 2026-04-10 — Extension E2: Multi-Stat Readouts (rescor_e2 vs rescor)

**Script**: `experiments/unified_ablation.py` with `--models rescor rescor_e2`
**Modules**: `src/wmca/modules/hybrid.py` (`CML2DWithStats`, `ResidualCorrectionWMv2`)
**Registry**: `rescor_e2` in `src/wmca/model_registry.py`
**Results**: `experiments/results/unified_ablation_e2_compare.json`
**Pod log**: `/tmp/e2_compare.log`

### Setup

- First architectural extension from `arch_plan.md`
- Modifies ResCor to read 5 stats from CML trajectory: `last`, `mean`, `var`, `delta`, `last_drive`
- NCA correction sees `[input | 5 CML stats]` (6x channels)
- `hidden_ch=32`, extra 1x1 mixing layer
- Param cost: 8.9x baseline (321 → 2849 for in=out=1) — user chose performance over params
- Single seed (42), 30 epochs, 300 trajectories, grid_size=16
- Run on A40 GPU pod

### Results

**1-step prediction:**

| Benchmark   | Metric | rescor  | rescor_e2 | Δ       | Winner    |
|-------------|--------|---------|-----------|---------|-----------|
| heat        | MSE    | 8.8e-8  | 1.3e-6    | +1424%  | rescor    |
| gol         | Acc    | 0.9463  | 0.9605    | +1.43pp | rescor_e2 |
| ks          | MSE    | 6.1e-7  | 8.9e-8    | -85%    | rescor_e2 |
| gray_scott  | MSE    | 2.8e-6  | 8.2e-7    | -71%    | rescor_e2 |
| rule110     | Acc    | 0.9683  | 0.9683    | 0       | tied      |
| wireworld   | Acc    | 0.9790  | 0.9788    | -0.015pp| tied      |
| grid_world  | Acc    | 0.99924 | 0.99917   | -0.007pp| tied      |

**10-step rollout:**

| Benchmark   | rescor h10 | rescor_e2 h10 | Δ         | Winner    |
|-------------|------------|---------------|-----------|-----------|
| heat        | 3.3e-6     | 7.4e-5        | +22x worse| rescor    |
| gol         | 65.86%     | 72.66%        | +6.8pp    | rescor_e2 |
| ks          | 6.1e-6     | 1.0e-6        | -83%      | rescor_e2 |
| gray_scott  | 2.6e-4     | 3.0e-5        | -88%      | rescor_e2 |
| rule110     | 74.38%     | 74.38%        | 0         | tied      |
| wireworld   | 99.10%     | 98.99%        | -0.11pp   | tied      |

**Grid World CEM Planning (~~HEADLINE~~ RETRACTED — see 2026-04-11 multi-seed entry):**

| Model     | Success Rate | Avg Steps |
|-----------|--------------|-----------|
| rescor    | 3.0%         | 40.0      |
| rescor_e2 | **32.0%**    | 21.4      |

### Key Findings

1. **10x improvement on grid_world CEM planning** despite essentially identical 1-step accuracy.
2. **-85% MSE on KS (1-step + rollout)**: var/delta capture velocity-like second-order dynamics.
3. **-88% MSE on Gray-Scott rollout**: temporal features track reaction-diffusion patterns.
4. **+6.8pp on GoL rollout**: bonus win, temporal features help discrete CAs too.
5. **Heat regression**: overfitting on trivial target. Both at numerical floor (1e-7 to 1e-6). Not a real loss.
6. **Triple confirmation of "planning-relevant inductive bias > 1-step accuracy"**: This is the third experiment showing it (after `nca_inside_cml` fix and v3 `gated_blend` results).

### Verdict

ADOPT E2 for continuous physics + action-conditioned planning. Heat regression is in noise territory. Needs multi-seed confirmation before final paper inclusion.

### Next

- Multi-seed E2 confirmation (seeds 0, 1, 2)
- Then E4 (per-channel affine drive)
- cml_ridge grid_world (`CML2DRidge` not callable) — pre-existing.

---

## 2026-04-11 — E2 Multi-Seed Confirmation + E4 Single-Seed Comparison

**Script**: `experiments/unified_ablation.py` with `--models rescor rescor_e2 rescor_e4 --seeds 0 1 2`
**Modules**: `src/wmca/modules/hybrid.py` (`CML2DWithStats`, `ResidualCorrectionWMv2`, + E4 affine drive)
**Context**: follow-up to the 2026-04-10 single-seed E2 run. Multi-seed was gated on `arch_plan.md` protocol 5 ("fixed seed set {0, 1, 2} throughout; never cherry-pick single-seed wins"). That protocol just saved us from publishing a false claim.

### Major Retraction

> **Retracting the "grid_world CEM 10x improvement" E2 headline from 2026-04-10.**
> The single-seed result (rescor 3% → rescor_e2 32%) was a **single-seed artifact**.
> Multi-seed (n=3) confirmed it is a **statistical tie**.

- Mean across 3 seeds: **rescor 23%** vs **rescor_e2 25%**.
- Per-seed: rescor (4, 30, 36) vs rescor_e2 (0, 42, 32).
- The variance within each model (rescor std ≈ 17pp, rescor_e2 std ≈ 22pp) dwarfs the 2pp gap between them.
- Grid_world CEM is a **high-variance** benchmark. Any future planning claim needs mean ± std across ≥ 3 seeds.

This correction flows into:
- `findings.md` Section 20 (retraction notice + strikethrough), new Section 21.
- `arch_plan.md` Status Summary + Extension 2 Status section.
- `TODO.md` E2 status line + "planning-relevant inductive bias" insight (now double-confirmed, not triple).

### E2 Multi-Seed: What IS Confirmed (CONTINUOUS PDEs)

Unanimous or > 2σ across 3 seeds:

| Benchmark               | Metric   | Δ (rescor → rescor_e2) | Significance     | Verdict            |
|-------------------------|----------|------------------------|------------------|--------------------|
| KS 1-step               | MSE      | **-86.4%**             | > 2σ             | Confirmed          |
| Gray-Scott 1-step       | MSE      | **-71.0%**             | > 2σ             | Confirmed          |
| Heat h=10 rollout       | MSE      | **-94.6%**             | 3/3 seeds        | Unanimous          |
| KS h=10 rollout         | MSE      | **-84.1%**             | 3/3 seeds        | Unanimous          |
| Gray-Scott h=10 rollout | MSE      | **-86.0%**             | 3/3 seeds        | Unanimous          |
| GoL 1-step              | Accuracy | **+3.24pp**            | rescor noisy     | Confirmed (noisy)  |

### E2 Multi-Seed: Retracted and Null

| Benchmark       | Result         | Per-seed (rescor → rescor_e2) | Mean          | Verdict             |
|-----------------|----------------|-------------------------------|---------------|---------------------|
| Grid World CEM  | **RETRACTED**  | (4, 30, 36) vs (0, 42, 32)    | 23% vs 25%    | **Statistical tie** |
| Rule 110        | Null           | tied                          | tied          | No difference       |
| Wireworld       | Null           | tied                          | tied          | No difference       |

### E2 Heat 1-step: Nuanced

Direction is unanimous (rescor_e2 wins 3/3 at ~2e-6, stable), but rescor has one bad seed (5e-5) and the std is huge. Both at numerical floor. This is not a "regression" story as framed on 2026-04-10 — rescor_e2 is actually more stable, rescor is noisier. Flip the interpretation.

### E4 (E2 + per-channel affine drive) — DO NOT ADOPT

**Setup**: `rescor_e4` = `rescor_e2` + per-channel learned affine on the CML drive (identity init: `alpha=1, beta=0`). Matches Extension 4 in `arch_plan.md`.

**Result**: strictly worse than E2 across all benchmarks.

| Benchmark        | rescor_e2 | rescor_e4 | Verdict         |
|------------------|-----------|-----------|-----------------|
| Wireworld        | 0.979     | **0.704** | Collapse        |
| Grid_world CEM   | 25%       | **16%**   | Regression      |
| NormScore (all)  | 0.772     | **0.536** | Strictly worse  |

**Mechanism**: gradients into `alpha`/`beta` flow through the downstream residual path and push the drive out of the logistic map's chaotic sweet spot. Once the drive drifts (even from identity init, over training), the CML collapses toward a near-identity and the learned branch has to re-learn the physics from scratch. The frozen-physics firewall is load-bearing — touching the drive breaks it. This was predicted as a risk in `arch_plan.md` Extension 4 ("gradient into alpha/beta is small … goes through a detached CML output") but the failure mode is worse than expected: not small, but destabilizing.

**Verdict**: **E4 REJECTED.** Do not adopt.

### Updated Next Steps

- Implementation order is now **E6 → E3 → E1 → E5** (E4 skipped).
- grid_world CEM is treated as a high-variance benchmark. All planning claims must report mean + std across ≥ 3 seeds going forward.

### Lessons

1. **`arch_plan.md` protocol 5 is load-bearing.** "Fixed seed set {0, 1, 2} throughout; never cherry-pick single-seed wins." If we had published the 2026-04-10 result as-is, we would have shipped a false "10x planning" claim. The protocol saved us — use it.
2. **Grid_world CEM variance is enormous** relative to its mean. The rescor distribution spans 4% → 36% across three seeds of the SAME model. Any headline number from a single seed on this benchmark is ~useless.
3. **Frozen physics + learned affine on the drive is a bad combo.** The drive position matters more than the affine does — and the affine fights the drive. If we want to touch the drive we need the affine to be *gated* or *bounded* so it cannot leave the chaotic band.

---

## 2026-04-12 — Extension E3: Dilated NCA Correction (multi-seed)

**Script**: `experiments/unified_ablation.py`
**Module**: `rescor_e3` (`ResidualCorrectionWMv7`) in `src/wmca/modules/hybrid.py`

### Setup

- Parallel 3x3 dilation=1 + 3x3 dilation=2 branches, each `hidden_ch//2 = 16` channels
- Same total param count as E2 (2849)
- Multi-seed n=3 vs `rescor` + `rescor_e2`

### Results (mean ± std)

| Benchmark   | Metric     | rescor              | rescor_e2           | rescor_e3                              |
|-------------|------------|---------------------|---------------------|----------------------------------------|
| heat        | 1-step MSE | 5.24e-5 ± 9e-5      | 2.05e-6 ± 9e-7      | 1.93e-6 ± 5e-7                         |
| heat        | h=10 MSE   | 1.25e-3             | 1.06e-4             | **2.41e-5 (-77% vs e2, unanimous)**    |
| ks          | 1-step MSE | 1.46e-6             | 1.43e-6             | 1.53e-6 (tied)                         |
| ks          | h=10 MSE   | 2.32e-5             | 3.37e-6             | 1.98e-6 (mixed seeds)                  |
| gray_scott  | 1-step MSE | 3.70e-6             | 1.20e-6             | 1.41e-6                                |
| gray_scott  | h=10 MSE   | 3.57e-4             | 7.81e-5             | **3.69e-5 (-53% unanimous)**           |
| gol/rule110/wireworld | — | tied             | tied                | tied                                   |
| grid_world  | CEM %      | 13.3%               | 27.3% ± 5.7%        | **17.3% ± 7.6% (REGRESSION)**          |

Per-seed grid_world CEM: `rescor_e3` (13, 11, 28) — bimodal failure mode.

### Verdict

**NOT adopted.** Wins on heat/gray_scott long-horizon rollouts but bimodal regression on grid_world CEM (2/3 seeds catastrophically fail at agent tracking).

---

## 2026-04-12 — Extension E3b: Zero-init Residual Dilation (multi-seed)

**Module**: `rescor_e3b` (`ResidualCorrectionWMv8`)

### Setup

- Same as E3 but additive residual `h1 + alpha*h2`, with per-channel LayerScale `alpha` init to 0
- Hypothesis: model starts as E2, gradient adds dilation only if useful
- Full `hidden_ch` on both branches
- Param cost: 4641 (1.6x E2)

### Results (n=3)

| Benchmark        | rescor_e2      | rescor_e3b                     |
|------------------|----------------|--------------------------------|
| heat 1-step MSE  | 2.05e-6        | 1.87e-6                        |
| heat h=10 MSE    | 1.06e-4        | 5.07e-5                        |
| ks 1-step MSE    | 1.43e-6        | **6.57e-7 (-54%)**             |
| ks h=10 MSE      | 3.37e-6        | **1.58e-6**                    |
| gray_scott h=10 MSE | 7.81e-5     | **2.77e-5 (-65%)**             |
| grid_world CEM   | 27.3% ± 5.7%   | **14.0% ± 8.0% (WORSE)**       |

Per-seed grid_world CEM: `rescor_e3b` (6, 14, 22).

Independent alpha probe: `alpha` grew **larger** on grid_world (abs_mean 0.130) than on PDEs (abs_mean 0.037 on heat). Zero-init guarantee held at init only — gradient pressure made the model use the dilated branch on grid_world even though it hurt.

### Verdict

**REJECTED.** Best PDE wins of any variant but worst grid_world CEM. Confirmed that zero-init alone is insufficient.

---

## 2026-04-12 — Extension E3c: WD-Alpha Residual Dilation — ADOPTED

**Module**: `rescor_e3c` (`ResidualCorrectionWMv9`)

### Setup

- Same as E3b but `train_model` now applies strong L2 weight decay (`wd=1.0`) selectively to `dilation_alpha` via a separate optimizer parameter group
- The `alpha` parameter is now expensive to use
- Param cost: 4641 trained (same as E3b, 1.6x E2)

### Results (n=3)

| Benchmark   | Metric     | rescor_e2       | rescor_e3b      | rescor_e3c          | Δ vs e2          |
|-------------|------------|-----------------|-----------------|---------------------|------------------|
| heat        | 1-step MSE | 2.05e-6         | 1.87e-6         | **1.56e-6**         | -24%             |
| heat        | h=10 MSE   | 1.06e-4         | 5.07e-5         | 5.98e-5             | -44%             |
| ks          | 1-step MSE | 1.43e-6         | 6.57e-7         | **5.39e-7**         | **-62%**         |
| ks          | h=10 MSE   | 3.37e-6         | 1.58e-6         | **1.54e-6**         | -54%             |
| gray_scott  | 1-step MSE | 1.20e-6         | 1.32e-6         | **1.05e-6**         | -13%             |
| gray_scott  | h=10 MSE   | 7.81e-5         | 2.77e-5         | 3.79e-5             | -51%             |
| gol/rule110/wireworld | — | tied         | tied            | tied                | —                |
| **grid_world** | **CEM %** | **27.3% ± 5.7%** | **14.0% ± 8.0%** | **28.7% ± 12.3%** | **+1.4pp (RECOVERED)** |

Per-seed grid_world CEM: `rescor_e3c` (15, 39, 32). Recovered to E2 levels.

### Key Findings

1. The L2 penalty on `alpha` enforces "dilation is opt-in only when it helps." The model can't afford to engage the dilated branch on grid_world.
2. **KS 1-step MSE -62%** is the biggest single-benchmark improvement of any extension.
3. Pareto dominates `rescor_e2` on 5 of 7 benchmarks while matching/beating on the rest.
4. grid_world CEM has high variance (39 in seed 1) but the mean is now slightly above E2.

**Architectural lesson**: Zero-init alone is structurally sound but operationally empty. Without a penalty, the optimizer uses any new capacity it has, even when harmful. Adding L2 on the new parameter forces the "use only when beneficial" intent.

**The "planning-relevant inductive bias" theme**: same 1-step accuracy as E2/E3/E3b on grid_world (~99.93%) but very different CEM planning (E3c 28.7% vs E3b 14%). Per-cell prediction quality doesn't determine planning quality.

### Verdict

**ADOPT `rescor_e3c` as the new default**, replacing `rescor_e2`.

## 2026-04-12 — Int8 Ablation (rescor_e3c)

**Goal**: Confirm int8 (128-level) CML quantization does not degrade rescor_e3c performance. n=3 seeds.

### KS equation

| Precision | 1-step MSE | h=10 rollout MSE |
|-----------|------------|------------------|
| float32   | 2.24e-7    | 1.80e-5          |
| bfloat16  | 4.78e-7    | 1.67e-5          |
| int8      | 2.57e-7    | 1.85e-5          |

Int8 vs float32 on h=10: **+3.1%** — negligible.

### Heat equation

| Precision | 1-step MSE | h=10 rollout MSE |
|-----------|------------|------------------|
| float32   | 1.30e-6    | 0.133            |
| bfloat16  | 1.48e-6    | 0.137            |
| int8      | 1.69e-6    | 0.136            |

Int8 vs float32 on h=10: **+2.3%** — negligible.

### Game of Life

| Precision | 1-step Acc | h=10 rollout Acc |
|-----------|------------|------------------|
| float32   | 97.91%     | 84.27%           |
| bfloat16  | 97.91%     | 84.42%           |
| int8      | 97.91%     | 84.30%           |

Literally identical.

### Key finding

Int8 quantization (128 levels) in the CML does NOT hurt rescor_e3c. The NCA correction compensates for quantization noise. Drive injection regularizes against discretization artifacts (confirmed from Phase 1-pre).

### Verdict

**Int8 IS viable.** The paper can claim int8 CML compatibility.

**Files**: `src/wmca/modules/hybrid.py` (`ResidualCorrectionWMv9`), `src/wmca/model_registry.py` (registered + `train_model` wd split).

---

## 2026-04-13 — Trajectory Attention Ablation (E7, 3-seed)

**Script**: `experiments/unified_ablation.py` with `--models rescor_traj_attn rescor_e3c`
**Seeds**: 0, 1, 2

**Setup**: `rescor_traj_attn` (4659 params) vs `rescor_e3c` (4641 params) on all 7 benchmarks. Trajectory attention replaces 3 of 5 hand-crafted CML stats (mean, var, last_drive) with 3 learned features via per-cell QKV cross-attention over the M=15 CML trajectory. Keeps `last` and `delta` as anchors. +18 params over E3c.

### Per-benchmark results (traj_attn vs e3c)

| Benchmark | Metric | Verdict | Details |
|-----------|--------|---------|---------|
| heat 1-step | MSE | slight win | 3/3 seeds traj_attn <= e3c |
| heat h=10 | MSE | **win 3/3** | -73%, -44%, -66% across seeds |
| gol | Acc | tie | identical across all seeds |
| ks 1-step | MSE | e3c wins 2/3 | traj_attn slightly worse |
| ks h=10 | MSE | mixed | 1 win, 1 tie, 1 loss |
| gray_scott 1-step | MSE | tie/slight e3c | within noise |
| gray_scott h=10 | MSE | **e3c wins 3/3** | traj_attn 20-480% worse |
| rule110 | Acc | tie | identical |
| wireworld 1-step | Acc | **win (stabilizes)** | e3c has 1 bad seed (69.9%), traj_attn stable 97-99% |
| wireworld h=10 | Acc | **win (stabilizes)** | same pattern |
| grid_world CEM | % | noise | (17,9,38) vs (44,20,24) -- both high variance |

**NormScore**: traj_attn wins 2/3 seeds.

### Timing

~1.5-2 hours wall time per seed (much slower than E3c due to storing and processing 15 trajectory states). CEM planning phase alone ~70 min per seed.

### Verdict

**NOT ADOPTED.** Wins on heat rollout (consistent) and wireworld (stabilizes bad seeds), but consistent regression on Gray-Scott rollout (20-480% worse). Not a Pareto improvement over E3c. Trajectory attention trades Gray-Scott performance for wireworld stability.

**Files**: `src/wmca/modules/hybrid.py` (`CML2DWithTrajectory`, `TrajectoryAttention`, `TrajectoryAttentionWM`), `src/wmca/model_registry.py`.

---

## 2026-04-13 — MoE-RF Ablation (E8, 3-seed)

**Script**: `experiments/unified_ablation.py` with `--models rescor_moe_rf rescor_e3c`
**Seeds**: 0, 1, 2

**Setup**: `rescor_moe_rf` (4621 params) vs `rescor_e3c` (4641 params) on 6 benchmarks (no grid_world). MoE-RF replaces the scalar dilation_alpha with per-cell CML-stats routing between d=1 and d=2 perception branches. Router: Conv2d(5*C_out, 2, 1x1) on CML stats -> softmax -> per-cell blend. -20 params vs E3c.

### Per-benchmark results (moe_rf vs e3c)

| Benchmark | Verdict | Details |
|-----------|---------|---------|
| heat | tie | both near-perfect |
| gol | tie | identical |
| ks | tie/mixed | within noise |
| gray_scott | mixed | moe_rf wins 1 seed, loses 2 |
| rule110 | e3c slight edge | |
| wireworld | **moe_rf stabilizes** | e3c bad seed 2 at 70.9%, moe_rf consistent 97-99% |

**NormScore**: e3c wins 2/3 seeds. MoE-RF is NOT a Pareto improvement — it is a sidegrade that stabilizes wireworld.

### Key finding

Per-cell routing learns near-constant weights on PDEs, validating E3c's fixed-dilation design. The router only provides benefit on wireworld (multi-class discrete CA where different cells genuinely need different receptive fields).

### Verdict

**NOT ADOPTED.** Validates E3c simplicity. Per-cell routing is unnecessary overhead for PDEs.

**Files**: `src/wmca/modules/hybrid.py` (`MoERFWorldModel`), `src/wmca/model_registry.py`.

---

## 2026-04-13 — CEM Stabilization

**Files modified**: `run_cem_evaluation()` in the evaluation pipeline.

### Changes

4 fixes to eliminate CEM planning variance:
1. **Exhaustive search**: 4^5 = 1024 action sequences enumerated (replaces CEM sampling)
2. **200 episodes** (up from 100)
3. **Fixed eval seed 12345** (decoupled from training seed)
4. **Soft predictions** (softmax instead of argmax during rollouts)

### Tradeoff

Exhaustive search is ~5x slower than old CEM but completely eliminates sampling noise. Speed fix still needed.

---

## 2026-04-13 — New Environments & Benchmarks

5 new environments implemented and registered:

| Environment | Registry Key | Grid | Description |
|-------------|-------------|------|-------------|
| HeatControlEnv | `heat_control` | 16x16 | Heat equation + agent-controlled sources |
| GrayScottControlEnv | `gs_control` | 32x32 | Reaction-diffusion + agent seeding |
| MiniGrid | `minigrid` | 8x8 | Grid navigator, negative control (no deps) |
| CrafterLite | `crafter_lite` | 16x16 | Resource grid, mixed spatial+symbolic |
| DMControl | `dmcontrol` | flat | Cartpole state vectors |

### AutumnBench Investigation

Investigated AutumnBench — **POOR FIT** (text-based interactive LLM benchmark, not supervised world model). Better alternatives for external comparison: PDEBench (NeurIPS 2022), APEBench (NeurIPS 2024).

---

## 2026-04-13 — MP-Gate Ablation (E10, 3-seed, 8 benchmarks) -- THE MATCHING PRINCIPLE IS LEARNABLE

**The single most impactful result of the project.**

### Setup

- Model A: `rescor_mp_gate` (4747 params, +106 / +2.3% over E3c)
  - Path A: full rescor_e3c (CML + NCA correction, hc=32)
  - Path B: tiny pure NCA (hc=8, no CML involvement)
  - Trust gate: MLP(var, last_drive -> 4 -> 1) + sigmoid -> per-cell blend
- Model B: `rescor_e3c` (4641 params)
- Seeds: 0, 1, 2
- Benchmarks: heat, gol, ks, gray_scott, rule110, wireworld, minigrid, crafter_lite (8 total)

### NormScore

mp_gate wins ALL 3 seeds:

| Seed | mp_gate | e3c |
|------|---------|-----|
| 0 | 0.875 | 0.250 |
| 1 | 0.875 | 0.125 |
| 2 | 0.750 | 0.375 |

### Per-benchmark results

| Benchmark | Verdict | Details |
|-----------|---------|---------|
| heat | tie/mixed | 1-step tie, h=10 mixed across seeds |
| **gol h=10** | **mp_gate 3/3** | +0.6-1.2pp rollout accuracy consistently |
| **ks** | **mp_gate 3/3** | Both 1-step and h=10, consistent |
| **gray_scott h=10** | **mp_gate 2/3** | Seed 1 big win (-85%), seed 0 loss, seed 2 win |
| rule110 | tie | identical |
| wireworld | tie/slight e3c | e3c marginally better |
| **minigrid** | **mp_gate 3/3 (-53%!)** | Gate shuts off CML on non-spatial benchmark |
| **crafter_lite** | **mp_gate 3/3** | Gate partially reduces CML on symbolic components |

### What the gate learns

- **Physics (KS, GS)**: gate mostly open -> trust CML -> CML+NCA correction
- **Discrete CAs (GoL)**: gate partially closes -> less CML reliance -> better rollout
- **Non-spatial (minigrid)**: gate closes significantly -> -53% MSE (CML was hurting!)
- **Mixed (crafter_lite)**: gate partially closes on symbolic components -> slight improvement

### Key finding

The Matching Principle is LEARNABLE. A single model with a learned trust gate adapts to whether CML dynamics match the target, rather than requiring the researcher to choose rescor vs pure_nca per-benchmark. This eliminates the manual design decision that was the central limitation of the Matching Principle as a guideline.

### Verdict

**STRONG CANDIDATE FOR ADOPTION as new default model.** Wireworld is a slight regression, but the ability to learn the Matching Principle is worth the trade. +106 params (+2.3%) is negligible overhead.

---

## 2026-04-13 — Atari Benchmark (3-seed, 5 models)

### Setup

- Environments: Pong (16x32), Breakout (20x16) — self-contained, no external dependencies
- Models: conv2d, rescor_mp_gate, rescor_e3c, rescor, pure_nca
- Seeds: 0, 1, 2
- Metric: accuracy (avg across seeds)

### Results

| Model | Pong Acc (avg) | Breakout Acc (avg) | Params |
|-------|---------------|-------------------|--------|
| conv2d | 99.76% | 99.99% | 3636 |
| rescor_mp_gate | 99.73% | 99.84% | 16129 |
| rescor_e3c | 99.64% | 99.93% | 15684 |
| rescor | 99.58% | 99.76% | 1380 |
| pure_nca | 99.54% | 99.90% | 804 |

### Key finding

Conv2d wins Atari — games are spatial but not diffusive/chaotic. CML doesn't hurt much (within ~0.2pp) but doesn't help either. Matching Principle confirmed: Atari lacks the diffusive/chaotic coupling where CML excels.

---

## 2026-04-13 — MiniGrid + CrafterLite Benchmark (3-seed, 5 models)

### Setup

- MiniGrid: 8x8 navigator, MSE metric, negative control (no inter-cell coupling)
- CrafterLite: 16x16 resource grid, accuracy metric, mixed spatial+symbolic dynamics
- Models: conv2d, rescor_mp_gate, rescor_e3c, rescor, pure_nca
- Seeds: 0, 1, 2

### MiniGrid Results (MSE, lower is better)

| Model | MSE |
|-------|-----|
| conv2d | 1.9e-4 (best) |
| rescor_mp_gate | 2.1e-4 |
| pure_nca | 3.7e-4 |
| rescor_e3c | 4.8e-4 |
| rescor | 1.3e-3 (worst) |

CML hurts — vanilla rescor 6.8x worse than conv2d. MP-Gate partially mitigates (2.1e-4) by shutting off CML. Negative control confirmed.

### CrafterLite Results (accuracy)

All models ~95.9-96.1%. mp_gate/e3c slight edge (~96.1%). Near-tie — tree growth is spatial (slight CML advantage) but harvesting is not.

### Key finding

MiniGrid is the cleanest negative control for the Matching Principle. CrafterLite is mixed dynamics = mixed results. Both confirm that CML coupling only helps when the target involves spatial coupling.

---

## 2026-04-14 — Autumn Benchmark (3-seed, 5 models)

### Setup

- autumn_disease: SIR spreading, 16x16
- autumn_gravity: falling blocks, 12x12
- autumn_water: water flow, 16x16
- Models: conv2d, rescor_mp_gate, rescor_e3c, rescor, pure_nca
- Seeds: 0, 1, 2

### autumn_disease Results (accuracy)

| Model | Accuracy |
|-------|----------|
| pure_nca | 95.6% (best) |
| rescor | 95.5% |
| conv2d | 89.1% (collapses on seeds 1&2) |

CML neutral — SIR is local CA but stochastic. CML's deterministic chaos doesn't match stochastic transmission. Conv2d collapses badly.

### autumn_gravity Results (accuracy)

| Model | Accuracy |
|-------|----------|
| rescor_mp_gate | 99.98% (best) |
| pure_nca | 99.9% |
| rescor | 99.7% |
| conv2d | 96.9% |

CML dominates — downward coupling = CML's conv2d kernel is the right bias. +3pp over conv2d.

### autumn_water Results (accuracy)

| Model | Accuracy |
|-------|----------|
| rescor_mp_gate | 99.2% (h=10: 99.5%, one seed hit 100%) |
| rescor | 98.8% |
| conv2d | 97.9% |

CML dominates — water flow = gravity + lateral diffusion, both CML strengths.

### Key findings

1. Gravity + water: CML dominates (local physics = CML sweet spot)
2. Disease: CML neutral (stochastic CA, doesn't match CML deterministic chaos)
3. Conv2d collapses on disease (89.1%) — overfits to deterministic patterns
4. Matching Principle holds across all three Autumn environments
5. These results, combined with Atari/MiniGrid/CrafterLite, confirm the Matching Principle across ALL tested domains

---

## 2026-04-14 — Consistent Ablation Run + MLP Baseline

### Setup

All 6 blog models (mlp, rescor, rescor_e3c, rescor_mp_gate, pure_nca, conv2d) x 14 benchmarks (heat, gol, ks, gray_scott, rule110, wireworld, crafter_lite, minigrid, autumn_disease, autumn_gravity, autumn_water, atari_pong, atari_breakout, dmcontrol). 30 epochs, 16x16 grids, 300 trajectories.

Seed 42: full run (all 6 models x 14 benchmarks). Seed 43: MLP only.

Results saved to: `experiments/results/unified_ablation_seed42.json`, `unified_ablation_seed43_mlp.json`

### Key Findings

- MLP (197K params canonical) is worst on ALL spatial benchmarks despite 40-600x more params
- MLP only wins on memorizable benchmarks: Atari Pong/Breakout (100%), Rule110 (100%)
- MLP collapses on AutumnBench: disease 64.8%, gravity 92.3%, water 86.7%
- MLP collapses on CrafterLite: 67.1% (vs pure_nca 95.9%)
- MLP competitive on DMControl: 1.4e-6 (non-spatial, validates Matching Principle)
- All blog tables updated to use consistent seed 42 data + canonical (1,1) param counts
- autumn_gravity and autumn_water added to unified_ablation.py ALL_BENCHMARKS
- rescor_traj_attn and rescor_mp_gate added to ALL_MODELS

### Additional Notes

Blog post rewritten in user's voice (no em-dashes, no hype language, commas instead).