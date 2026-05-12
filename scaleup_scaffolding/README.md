# Scale-up Scaffolding

Pre-build drafts for two scale-up architectures from `../arch_plan.md`:
`rescor_ms` (Multi-Scale U-Net pyramid 64 → 32 → 16 → 32 → 64) and
`rescorformer` (SWA + 2D RoPE with rescor as the FFN).

These files are **scaffolds only**. They do not import from `wmca.modules.*`
and do not touch anything in `src/wmca/` or `experiments/`. They are safe to
drop in while long-running experiments are active.

## Files

| file | purpose |
| --- | --- |
| `rescor_ms.py` | `RescorMSPyramid` — U-Net pyramid wrapper, CML+NCA per scale |
| `rescorformer.py` | `ResCorformer` — L=1 transformer block, SWA, 2D RoPE, rescor FFN |
| `test_shapes.py` | `python test_shapes.py` → forward-pass shape + range check |

Both architectures default to stubbed CML / NCA modules (`nn.Identity` and a
tiny placeholder NCA) so `test_shapes.py` runs standalone with only `torch`.

## Shape contract

Both modules use the same contract:

```
in:  (B, 1, 64, 64)  values in [0, 1]
out: (B, 1, 64, 64)  values in [0, 1]
```

* `RescorMSPyramid` has three CML banks (one per scale at 64, 32, 16) and
  five NCA heads (two skip, one bottleneck, two up-sample refine). Skip
  connections are additive to keep channel count at 1.
* `ResCorformer` has a single CML bank applied to the unpatchified 64×64
  grid inside the FFN position of the transformer block.

## Param budgets (from arch_plan.md)

* `rescor_ms` — target ~4 K trained, ~36 frozen.
* `rescorformer` — target ~7.4 K trained, ~12 frozen. (Achieved count in this
  scaffold is slightly higher because `_DefaultNCA` is a placeholder; real
  integration will drop to spec.)

## 3-step integration plan

1. **Drop in `CML2DMultiR`.**
   In each scaffold, replace the `nn.Identity` stub with
   `CML2DMultiR(in_channels=1, K=32, gate_mode="uniform", steps=M_scale)`
   at the three `# TODO: ... CML2DMultiR ...` comments. Use the commented-out
   `from wmca.modules.hybrid import CML2DMultiR` line at the top of each file.
   Replace `_TinyNCA` / `_DefaultNCA` with the real rescor_e3c NCA head from
   `wmca.modules` at the bottleneck (keep tiny NCAs at the other scales).

2. **Register in `model_registry.py`.**
   Add two new entries:
   * `rescor_ms` → `{"cml_gate": "multi_r_uniform", "arch": "ms_pyramid", ...}`
   * `rescorformer` → `{"cml_gate": "multi_r_uniform", "arch": "swa_rope_rescor", ...}`
   with the same param-budget sanity assertions used for existing heroes.

3. **Write an ablation script.**
   `experiments/scaleup_pyramid_ablation.py` — runs both architectures on the
   64×64 benchmarks (crafter_real, doom_pixel, higher-res APEBench). Use
   seeds 42/43/44 from day 1 per the S57 multi-seed methodology rule (no
   single-seed hero claims). Compare against the existing 16×16 hero
   (`rescor_mr_uniform K=32`) upsampled tile-wise as a sanity floor.

## Caveats forward-linked from findings.md

* **Methodology caveat (S56 heat epoch diagnostic).** Any NCA with a widened
  input must run ≥60 epochs or zero-init the extra-channel weights of the
  first perceive conv. Both scaffolds already zero-init the final conv in
  their stub NCAs; preserve that at integration.
* **Hero status caveat (S57, 2026-04-22).** `rescor_mr_uniform K=32` was
  demoted from hero under multi-seed replication. The `gate_mode="uniform"`
  bank is still the simplest correct drop-in for scale-up ablations, but all
  new claims must be multi-seed from day 1.
