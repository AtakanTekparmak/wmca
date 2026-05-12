# Resources

## Papers:

### arxiv papers

#### Reservoir Computing

- https://arxiv.org/pdf/1703.02806
- https://arxiv.org/pdf/2407.09501
- https://arxiv.org/pdf/2508.02218
- https://arxiv.org/pdf/2601.22296
- https://arxiv.org/pdf/2410.02536
- https://arxiv.org/pdf/2308.02902

#### EGROLL

- https://arxiv.org/pdf/2511.16652

## Blogs

- https://andthattoo.dev/blog/es2n_paralesn
- https://allenai.org/blog/molmospaces (high resource blog with paper, data, code)

## Code: 

- https://github.com/d0rc/egg.c (egroll in c)
- https://github.com/ESHyperscale/nano-egg (egroll training in int)

## Benchmarks

- PDEBench (NeurIPS 2022): https://arxiv.org/pdf/2210.07182, code: github.com/pdebench/PDEBench
- APEBench (NeurIPS 2024): https://arxiv.org/pdf/2411.00180, code: github.com/tum-pbs/apebench
- The Well (NeurIPS 2024): https://arxiv.org/pdf/2412.00568, code: github.com/PolymathicAI/the_well
- SmallWorlds (Nov 2025): https://arxiv.org/pdf/2511.23465
- AutumnBench / Benchmarking World-Model Learning: https://arxiv.org/pdf/2510.19788

## World Models

- DELTA-IRIS (ICML 2024): https://arxiv.org/pdf/2406.19320, code: github.com/vmicheli/delta-iris, weights: huggingface.co/vmicheli/delta-iris
- DIAMOND (NeurIPS 2024): https://arxiv.org/pdf/2405.12399, code: github.com/eloialonso/diamond
- GameNGen (ICLR 2025): https://arxiv.org/pdf/2408.14837
- IRIS (ICLR 2023): https://arxiv.org/pdf/2209.00588, code: github.com/eloialonso/iris
- DreamerV3 (Nature 2025): https://arxiv.org/pdf/2301.04104, code: github.com/danijar/dreamerv3
- STORM (NeurIPS 2023): https://arxiv.org/pdf/2310.09615, code: github.com/weipu-zhang/STORM
- TWM (ICLR 2023): https://arxiv.org/pdf/2303.07109, code: github.com/jrobine/twm
- Genie (ICML 2024): https://arxiv.org/pdf/2402.15391
- TD-MPC2 (ICLR 2024): https://arxiv.org/pdf/2310.16828, code: github.com/nicklashansen/tdmpc2

## Neural Operators / PDE Solvers

- FNO (ICLR 2021): https://arxiv.org/pdf/2010.08895, code: github.com/neuraloperator/neuraloperator
- PDE-Refiner (NeurIPS 2023): https://arxiv.org/pdf/2308.05732
- PhyDNet (CVPR 2020) — "physics model + learned correction" for video prediction, validates rescor pattern

## NCA / Cellular Automata

- "Learning spatio-temporal patterns with NCA" (Richardson et al., PLOS Comp Bio 2024) — NCA on Gray-Scott, closest prior work to rescor
- "It's Hard for Neural Networks to Learn GoL" (IJCNN 2021): https://arxiv.org/pdf/2009.01398
- LifeGPT (2024): https://arxiv.org/pdf/2409.12182, code: github.com/lamm-mit/LifeGPT
- AutomataGPT (Advanced Science 2025)
- muNCA (2021): https://arxiv.org/pdf/2111.13545 — 68-param NCA texture generation

## MBRL

- MBPO (NeurIPS 2019): https://arxiv.org/pdf/1906.08253, code: github.com/jannerm/mbpo
- PETS (NeurIPS 2018): https://arxiv.org/pdf/1805.12114
- MBRL-Lib (Facebook): https://arxiv.org/pdf/2104.10159, code: github.com/facebookresearch/mbrl-lib

## Architecture Ideas

- PhyDNet: physics model + learned correction (same pattern as rescor but learned physics branch)
- Rescorformer: attention + rescor (FFN replaced by CML+NCA)
- Multi-scale NCA: U-Net-like pyramid with NCA at each level
- S4ND: state space models for multidimensional signals