# dreamerv3_scaffolding/

Pre-implementation scaffolding for the DreamerV3 fork experiment
described in `../dreamerv3_fork_plan.md`. **Nothing here has been
executed or imported by the live experiment code.** The directory is a
staging area for code that will later be copied into a separate fork
of `NM512/dreamerv3-torch`.

## Contents

| File | Purpose |
|---|---|
| `rescor_rssm.py` | Stub `RescorRSSM` class implementing Dreamer's `.initial / .obs_step / .img_step` interface on top of `wmca.modules.hybrid.CML2DMultiR` (K=32) + a 321-param NCA correction. Action conditioning via Option B (drive modulation). Every integration point is marked `# TODO`. |
| `action_embedder.py` | `ActionEmbedder` nn.Module: discrete action id -> (B, 2, 16, 16) grid added to the rescor input. ~13K params with `hidden=24`. |
| `rollout_stability_probe.py` | Standalone M2 prerequisite experiment. Loads or trains rescor_rens K=32 and rolls it autoregressively on heat for H in {15, 50, 100}, reporting per-step MSE / cosine divergence. Writes a JSON and a go/no-go verdict. **Not yet launched.** |

## What is NOT done

- NM512/dreamerv3-torch has **not** been forked.
- `pip install -e ../wmca` has **not** been run inside any Dreamer env.
- `rescor_rssm.py` has **not** been wired into Dreamer's config/factory.
- `rollout_stability_probe.py` has **not** been executed.
- No checkpoint of rescor_rens K=32 has been saved for the probe.

## Next manual steps (in order)

1. Confirm prerequisites listed in `dreamerv3_fork_plan.md` section 8
   (multi-seed rescor_rens K=32 hero, autoregressive probe on
   heat/KS/GS, package `rescor_mr_uniform`).
2. Run `rollout_stability_probe.py` on a frozen rescor_rens K=32
   checkpoint. Decide posterior-keep vs posterior-drop from the output
   JSON's `_decision.recommendation`.
3. Fork `NM512/dreamerv3-torch` into a sibling directory:
   `~/Desktop/personal/research/wmca-dreamer/`.
4. Inside that fork, `pip install -e ../wmca` so
   `wmca.modules.hybrid.CML2DMultiR` and `wmca.model_registry` are
   importable.
5. Copy `rescor_rssm.py` -> `dreamerv3/networks_rescor.py` and
   `action_embedder.py` -> `dreamerv3/action_embedder.py` in the fork.
6. Resolve every `# TODO` in `rescor_rssm.py` against
   NM512's `networks.py` (sample helpers, embed dim, is_first reset).
7. Add a `configs_rescor.yaml` override that points the sequence model
   factory at `RescorRSSM` instead of the default GRU RSSM.
8. Run M1 sanity (vanilla Crafter score ~11 at 1M steps) before
   launching M4 rescor training.

## Constraints honored while writing this scaffold

- **No existing files in `src/wmca/**` or `experiments/**` were
  modified.** A long-running experiment is using them.
- No packages installed, no networks cloned.
- Scripts reference `wmca.*` modules via path injection; they run only
  after the main wmca repo is installable or on `PYTHONPATH`.
