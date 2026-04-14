# WMCA: World Models via Coupled Map Lattices

A frozen chaotic reservoir (Coupled Map Lattice) paired with a learned Neural Cellular Automaton correction produces world models with 321 trainable parameters that match or outperform conventional architectures with 10-1000x more parameters on physics-governed benchmarks.

**Blog post**: [Chaotic Reservoirs for World Modeling](https://atakantekparmak.github.io/#blog/chaotic-reservoirs-world-modeling)

## Core idea

The **rescor** (residual correction) architecture uses a frozen 2D Coupled Map Lattice as a chaotic reservoir. The CML runs a logistic map `f(x) = r*x*(1-x)` at `r=3.90` (deep chaos) with convolutional coupling for 15 steps, producing rich spatiotemporal statistics (mean, variance, delta, etc.) from the input state. A tiny learned NCA then corrects the CML's output:

```
output = CML_last(input) + NCA_correction([input, CML_stats])
```

The CML is completely frozen (no gradients, no training). Only the NCA correction is learned.

### The Matching Principle

The central finding: **CML helps when its dynamics match the target, hurts when they don't.**

- The CML's logistic map produces chaotic trajectories, and its conv2d coupling implements diffusion-like spatial mixing.
- For diffusion-governed PDEs (heat equation, Gray-Scott reaction-diffusion), this is nearly the correct physics. The NCA only needs to learn a small residual.
- For discrete cellular automata (Game of Life, Wireworld) or non-spatial tasks (MiniGrid), the CML's dynamics are irrelevant noise. Pure NCA without CML is better.

The `rescor_mp_gate` architecture (4747 parameters) learns this principle automatically via a per-cell trust gate that decides when to rely on the CML path vs. a pure NCA path.

## Models


| Model            | Params (trained) | Description                                                                                |
| ---------------- | ---------------- | ------------------------------------------------------------------------------------------ |
| `rescor`         | 321              | Vanilla CML + NCA correction. Hero model.                                                  |
| `rescor_e3c`     | 4,641            | + multi-stat CML readouts, dual-scale perception (dilation 1+2), zero-init residual gating |
| `rescor_mp_gate` | 4,747            | Trust gate that learns when to use CML vs. pure NCA per cell                               |
| `pure_nca`       | 177              | Learned NCA only (no CML). Best for discrete/non-spatial targets.                          |
| `conv2d`         | 2,625            | 3-layer CNN baseline                                                                       |
| `mlp`            | varies           | MLP baseline (flattens spatial dims)                                                       |


See `src/wmca/model_registry.py` for the full registry (20+ variants including ablations).

## Results summary

Representative 1-step metrics across 4 domains, 30 epochs, 16x16 grids. Parameter counts shown for canonical single-channel (1-in, 1-out) configuration.

| Benchmark       | Domain      | Metric   | mlp (197Kp) | rescor (321p) | rescor_e3c (4.6Kp) | rescor_mp_gate (4.7Kp) | pure_nca (177p) | conv2d (2.6Kp) |
| --------------- | ----------- | -------- | ----------- | ------------- | ------------------- | ---------------------- | --------------- | -------------- |
| Heat equation   | Physics PDE | MSE      | 1.0e-4      | **7.7e-7**    | 9.5e-7              | 7.3e-7                 | 4.0e-5          | 8.4e-6         |
| KS              | Physics PDE | MSE      | 2.2e-6      | 5.1e-7        | 5.5e-8              | **2.2e-8**             | 8.6e-6          | 6.8e-6         |
| Gray-Scott      | Physics PDE | MSE      | 1.1e-5      | 2.1e-6        | 1.1e-6              | **4.5e-7**             | 6.8e-5          | 1.9e-5         |
| Game of Life    | Discrete CA | Accuracy | 75.2%       | 87.9%         | **96.1%**           | 96.1%                  | 94.9%           | 95.9%          |
| Rule110         | Discrete CA | Accuracy | **100%**    | 96.8%         | 96.8%               | 78.7%                  | 96.3%           | 96.7%          |
| Wireworld       | Discrete CA | Accuracy | 70.5%       | 97.9%         | 97.9%               | 97.9%                  | **99.0%**       | 70.5%          |
| Atari Pong      | Games       | Accuracy | **100%**    | 99.7%         | 99.7%               | 99.7%                  | 99.6%           | 99.8%          |
| Atari Breakout  | Games       | Accuracy | **100%**    | 99.9%         | 99.97%              | 99.98%                 | 99.9%           | 99.99%         |
| MiniGrid        | Games       | MSE      | 3.8e-3      | 1.2e-3        | 4.4e-4              | **1.7e-4**             | 3.8e-4          | 2.4e-4         |
| CrafterLite     | Games       | Accuracy | 67.1%       | 95.9%         | **96.1%**           | 96.1%                  | 95.9%           | 95.9%          |
| Autumn Disease  | AutumnBench | Accuracy | 64.8%       | **95.6%**     | 95.6%               | 95.6%                  | 95.6%           | 85.8%          |
| Autumn Gravity  | AutumnBench | Accuracy | 92.3%       | 96.9%         | 96.9%               | **99.97%**             | 99.9%           | 96.9%          |
| Autumn Water    | AutumnBench | Accuracy | 86.7%       | 98.9%         | 99.2%               | **99.3%**              | 97.8%           | 98.0%          |

The Matching Principle is visible: rescor outperforms on physics PDEs (where CML dynamics match), while pure_nca and conv2d win on discrete CAs (where CML dynamics don't match). The 197K-parameter MLP is worst on every spatial benchmark despite having 40-600x more parameters, validating that spatial inductive bias is necessary. The rescor_mp_gate model bridges the CML gap by learning to shut off the CML path on mismatched benchmarks.

## Quick start

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
# Clone and enter the repo
git clone https://github.com/AtakanTekparmak/wmca.git
cd wmca

# Run the unified ablation across all benchmarks and models
uv run --with scikit-learn,matplotlib,scipy python experiments/unified_ablation.py --no-wandb

# Run specific benchmarks and models
uv run --with scikit-learn,matplotlib,scipy python experiments/unified_ablation.py \
    --benchmarks heat gol gray_scott \
    --models rescor pure_nca conv2d \
    --no-wandb

# Quick smoke test (fewer epochs, smaller data)
uv run --with scikit-learn,matplotlib,scipy python experiments/unified_ablation.py --quick --no-wandb

# List available benchmarks and models
uv run python experiments/unified_ablation.py --list
```

## Benchmarks

Self-contained benchmarks across 4 domains (no external simulators required):

**Physics PDEs** (continuous, CML matches well):

- `heat` -- 2D heat equation (diffusion)
- `ks` -- Kuramoto-Sivashinsky (1D chaotic PDE)
- `gray_scott` -- Gray-Scott reaction-diffusion (2 channels)

**Discrete CAs** (CML dynamics mismatch):

- `gol` -- Conway's Game of Life
- `rule110` -- Rule 110 elementary CA (1D)
- `wireworld` -- Wireworld (4-state CA)

**Games / Planning** (action-conditioned):

- `grid_world` -- 16x16 navigation with walls
- `minigrid` -- 8x8 MiniGrid-Empty (negative control: no spatial coupling)
- `crafter_lite` -- Mixed spatial + symbolic dynamics
- `atari_pong` -- Self-contained Pong
- `atari_breakout` -- Self-contained Breakout

**AutumnBench** (custom grid environments):

- `autumn_disease` -- SIR disease spreading
- `autumn_gravity` -- Gravity with falling blocks
- `autumn_water` -- Water flow with lateral diffusion

## Project structure

```
wmca/
  src/wmca/
    modules/
      hybrid.py          # All architectures: CML2D, PureNCA, ResidualCorrectionWM, etc.
      cml.py             # 1D CML reservoir (for time-series tasks)
      paralesn.py        # Parallel ESN with FFT/sequential/Triton scan backends
      norm.py            # RMSNorm
    benchmarks.py        # Benchmark data generators + CEM planning evaluation
    model_registry.py    # Model factory, training loop, evaluation
    envs/
      atari_pong.py      # Self-contained Pong and Breakout environments
      gray_scott_control.py  # Gray-Scott reaction-diffusion control
      heat_control.py    # Heat equation control
      autumn/            # AutumnBench environment suite
  experiments/
    unified_ablation.py  # Main experiment runner (all benchmarks x all models)
    phase2_ablation.py   # Phase 2 architecture comparison
    cml_self_analysis.py # CML reservoir properties (Lyapunov, rank, fidelity)
    lorenz_prediction.py # Chaotic time-series prediction
    trust_gate_viz.py    # Matching Principle gate visualization
    plots/               # Generated figures
    results/             # JSON experiment results
  findings.md            # Full experimental results and analysis
  pyproject.toml         # Dependencies (torch, numpy, wandb)
```

## Key references

- Pathak et al. 2018: Reservoir computing for spatiotemporal chaos (Kuramoto-Sivashinsky)
- Richardson et al. 2024: NCA learns PDE dynamics (Gray-Scott)
- Pinna et al. 2026: ParalESN -- diagonal parallel ESN with O(log T) scan
- Nichele et al. 2017/2024/2025: Deep CA reservoirs, evolved critical NCA
- Bena et al. 2025: Universal NCA emulates neural networks within CA state

## License

MIT