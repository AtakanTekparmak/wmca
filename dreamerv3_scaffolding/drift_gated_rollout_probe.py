"""Drift-gated rollout-stability probe for rescor_mamba_gated_rand.

Sibling of ``pushforward_rollout_probe_mamba.py``. Loads the gated
checkpoints produced by ``experiments/drift_gated_ablation.py`` and
reports per-step MSE + cosine-divergence at H={15, 50, 100} for
benches × seeds.

Key extra: also logs the per-step **mean gate value** (averaged over
spatial cells and the rollout-trajectory dimension). This tells us if
the gate is doing what we want — closing as predictions drift from the
manifold during rollout.

Expects ``experiments/results/drift_gated_ckpts/{model}_{bench}_seed{seed}.pt``.

Saves:
  - experiments/results/drift_gated_rollout_probe.json (or _smoke.json)

Usage:
    PYTHONPATH=src PYTHONUNBUFFERED=1 uv run --with scikit-learn,scipy \\
        python dreamerv3_scaffolding/drift_gated_rollout_probe.py
    PYTHONPATH=src uv run python dreamerv3_scaffolding/drift_gated_rollout_probe.py --smoke
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from statistics import median

import numpy as np
import torch

from wmca.benchmarks import generate_gray_scott, generate_heat, generate_ks
from wmca.model_registry import create_model


BENCHMARKS = {
    "heat": generate_heat,
    "gs": generate_gray_scott,
    "ks": generate_ks,
}

MODEL = "rescor_mamba_gated_rand"
SEEDS = [42, 43, 44]
N_STEPS = 105
N_TRAJECTORIES = 200
GRID = 16
CONTEXT_K = 4
N_ROLLOUT_TRAJS = 20
HORIZONS = [15, 50, 100]

CKPT_DIR = Path("experiments/results/drift_gated_ckpts")


def cosine_div(pred: np.ndarray, true: np.ndarray) -> float:
    p = pred.reshape(-1)
    t = true.reshape(-1)
    num = float((p * t).sum())
    den = float(np.linalg.norm(p) * np.linalg.norm(t)) + 1e-8
    return 1.0 - num / den


def ckpt_path_for(bench_name, seed):
    return CKPT_DIR / f"{MODEL}_{bench_name}_seed{seed}.pt"


def probe_one(bench_name, gen_fn, seed, max_h: int | None = None,
              device: str = "cpu"):
    t0 = time.time()
    # Re-generate with context_k=1 for single-frame ground-truth trajectories.
    data = gen_fn(grid_size=GRID, seed=seed, n_steps=N_STEPS,
                  n_trajectories=N_TRAJECTORIES, context_k=1)
    meta = data.meta

    model = create_model(
        MODEL,
        in_channels=meta["in_channels"],
        out_channels=meta["out_channels"],
        grid_size=GRID,
        seed=seed,
    )
    pc = model.param_count() if hasattr(model, "param_count") else {
        "trained": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "frozen": sum(p.numel() for p in model.parameters() if not p.requires_grad),
    }

    ckpt = ckpt_path_for(bench_name, seed)
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model = model.to(device)

    # Capture gate scalars from loaded checkpoint for diagnostics.
    gate_scale_loaded = float(model.gate_scale.detach().cpu().item())
    gate_bias_loaded = float(model.gate_bias.detach().cpu().item())

    T = N_STEPS
    X_test = data.X_test
    Y_test = data.Y_test
    n_test_traj = X_test.shape[0] // T
    n_use = min(N_ROLLOUT_TRAJS, n_test_traj)

    probe_max_h = max_h if max_h is not None else max(HORIZONS)
    per_step_mse = np.zeros(probe_max_h, dtype=np.float64)
    per_step_cos = np.zeros(probe_max_h, dtype=np.float64)
    # Per-step mean gate (averaged over spatial cells AND rollout trajs).
    per_step_gate = np.zeros(probe_max_h, dtype=np.float64)
    # Track per-step counts in case some trajectories terminate early.
    per_step_count = np.zeros(probe_max_h, dtype=np.float64)

    model.eval()
    K = CONTEXT_K
    with torch.no_grad():
        for ti in range(n_use):
            seed_frames = [X_test[ti * T]]
            for k in range(K - 1):
                seed_frames.append(Y_test[ti * T + k])
            buf = torch.stack(seed_frames, dim=0).unsqueeze(0).to(device)  # (1,K,C,H,W)

            for s in range(probe_max_h):
                # Compute the gate for the CURRENT input buffer (mean over
                # spatial cells). This mirrors what forward() will use.
                x_now = buf[:, -1]
                stack = model.rens._run_batched(x_now)
                cml_mean = stack.mean(dim=1)
                gate_t = model.compute_gate(x_now, cml_mean)        # (1, 1, H, W)
                gate_mean = float(gate_t.mean().item())

                pred = model(buf)                                    # (1,C,H,W)
                gt_idx = ti * T + K - 1 + s
                if gt_idx >= Y_test.shape[0] or (gt_idx // T) != ti:
                    break
                gt = Y_test[gt_idx].unsqueeze(0).to(device)
                per_step_mse[s] += float(((pred - gt) ** 2).mean().item())
                p_np = pred.cpu().numpy()
                g_np = gt.cpu().numpy()
                per_step_cos[s] += cosine_div(p_np, g_np)
                per_step_gate[s] += gate_mean
                per_step_count[s] += 1.0
                pred_clamped = pred.clamp(0.0, 1.0).unsqueeze(1)      # (1,1,C,H,W)
                buf = torch.cat([buf[:, 1:], pred_clamped], dim=1)

    # Average per-step: divide by counts (avoids bias when some trajs
    # terminate early due to running off the end of T=105).
    valid = per_step_count > 0
    per_step_mse[valid] = per_step_mse[valid] / per_step_count[valid]
    per_step_cos[valid] = per_step_cos[valid] / per_step_count[valid]
    per_step_gate[valid] = per_step_gate[valid] / per_step_count[valid]

    horizons_out = {}
    mse_step1 = float(per_step_mse[0])
    for h in HORIZONS:
        if h > probe_max_h:
            continue
        mse_h = float(per_step_mse[h - 1])
        horizons_out[f"H={h}"] = {
            "mse_final": mse_h,
            "cosine_div_final": float(per_step_cos[h - 1]),
            "mse_ratio_vs_step1": (mse_h / mse_step1) if mse_step1 > 0 else float("inf"),
            "gate_mean_at_h": float(per_step_gate[h - 1]),
        }

    return {
        "model": MODEL,
        "bench": bench_name,
        "seed": seed,
        "params": pc,
        "probe_time_s": time.time() - t0,
        "mse_per_step": per_step_mse.tolist(),
        "cosine_div_per_step": per_step_cos.tolist(),
        "gate_per_step": per_step_gate.tolist(),
        "horizons": horizons_out,
        "n_rollout_trajectories": n_use,
        "ckpt": str(ckpt),
        "max_h": probe_max_h,
        "gate_scale_loaded": gate_scale_loaded,
        "gate_bias_loaded": gate_bias_loaded,
    }


def main(smoke: bool = False, smoke_max_h: int = 15,
         smoke_seeds: list[int] | None = None,
         smoke_benches: list[str] | None = None):
    out_path = Path("experiments/results/drift_gated_rollout_probe.json")
    if smoke:
        out_path = Path("experiments/results/drift_gated_rollout_probe_smoke.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = {bn: {} for bn in BENCHMARKS}

    seeds_to_run = SEEDS
    benches_to_run = dict(BENCHMARKS)
    if smoke:
        seeds_to_run = smoke_seeds if smoke_seeds is not None else [42]
        if smoke_benches is not None:
            benches_to_run = {b: BENCHMARKS[b] for b in smoke_benches}
        else:
            benches_to_run = {"heat": generate_heat}

    if out_path.exists():
        try:
            with open(out_path, "r") as f:
                prior = json.load(f)
            src = prior.get("per_cell", prior)
            n_loaded = 0
            for bn in BENCHMARKS:
                if bn not in src:
                    continue
                for seed_str, cell in src[bn].items():
                    if cell and cell.get("mse_per_step"):
                        results[bn][seed_str] = cell
                        n_loaded += 1
            print(f"[resume] loaded {n_loaded} completed cells from {out_path}")
        except Exception as e:
            print(f"[resume] failed ({e}); starting fresh")

    for bn, gf in benches_to_run.items():
        for seed in seeds_to_run:
            seed_str = str(seed)
            if (results[bn].get(seed_str)
                    and results[bn][seed_str].get("mse_per_step")):
                r = results[bn][seed_str]
                h15 = r.get("horizons", {}).get("H=15", {})
                print(f"[{bn:4s} seed={seed}] resumed  "
                      f"H15_ratio={h15.get('mse_ratio_vs_step1', float('nan')):.2f}")
                continue
            ckpt = ckpt_path_for(bn, seed)
            if not ckpt.exists():
                print(f"[{bn:4s} seed={seed}] MISSING checkpoint {ckpt}; skipping")
                results[bn][seed_str] = {
                    "error": "checkpoint missing", "ckpt": str(ckpt),
                    "model": MODEL, "bench": bn, "seed": seed,
                }
                continue
            print("=" * 78)
            print(f"probe  {MODEL}  {bn}  seed={seed}  context_k={CONTEXT_K}")
            print("=" * 78)
            try:
                r = probe_one(bn, gf, seed,
                              max_h=smoke_max_h if smoke else None)
                results[bn][seed_str] = r
                print(f"  probe time: {r['probe_time_s']:.1f}s  params={r['params']}")
                print(f"  gate scalars from ckpt: scale={r['gate_scale_loaded']:.4f}  "
                      f"bias={r['gate_bias_loaded']:.4f}")
                print(f"  step 1 MSE = {r['mse_per_step'][0]:.4e}   "
                      f"step 1 gate_mean = {r['gate_per_step'][0]:.4f}")
                for hh in HORIZONS:
                    if hh > r["max_h"]:
                        continue
                    c = r["horizons"][f"H={hh}"]
                    print(f"  H={hh:3d}  MSE={c['mse_final']:.4e}  "
                          f"ratio_vs_step1={c['mse_ratio_vs_step1']:.2f}  "
                          f"cos_div={c['cosine_div_final']:.4f}  "
                          f"gate_mean={c['gate_mean_at_h']:.4f}")
            except Exception as e:
                print(f"  FAILED: {type(e).__name__}: {e}")
                import traceback; traceback.print_exc()
                results[bn][seed_str] = {
                    "error": str(e), "model": MODEL, "bench": bn, "seed": seed,
                }
            with open(out_path, "w") as f:
                json.dump({
                    "per_cell": results,
                    "protocol": {
                        "model": MODEL, "context_k": CONTEXT_K, "grid": GRID,
                        "n_steps": N_STEPS,
                        "n_trajectories": N_TRAJECTORIES,
                        "seeds": SEEDS,
                        "horizons": HORIZONS,
                        "n_rollout_trajectories": N_ROLLOUT_TRAJS,
                        "smoke": smoke,
                    },
                }, f, indent=2)
            gc.collect()

    print()
    print("=" * 90)
    print("SUMMARY — drift-gated mamba_rand rollout stability "
          "(median across seeds)")
    print("=" * 90)
    hdr = (f"{'bench':6s}  {'H':>3s}  {'MSE median':>12s}  "
           f"{'ratio median':>14s}  {'cos_div med':>12s}  {'gate med':>10s}")
    print(hdr)
    print("-" * len(hdr))
    summary = {bn: {} for bn in BENCHMARKS}
    for bn in BENCHMARKS:
        for hh in HORIZONS:
            ratios, mses, cosds, gates = [], [], [], []
            for seed_str, cell in results[bn].items():
                if "horizons" not in cell:
                    continue
                c = cell["horizons"].get(f"H={hh}")
                if not c:
                    continue
                ratios.append(c["mse_ratio_vs_step1"])
                mses.append(c["mse_final"])
                cosds.append(c["cosine_div_final"])
                gates.append(c["gate_mean_at_h"])
            if ratios:
                s = {
                    "mse_median": median(mses),
                    "ratio_median": median(ratios),
                    "cos_div_median": median(cosds),
                    "gate_mean_median": median(gates),
                    "n_seeds": len(ratios),
                }
                summary[bn][f"H={hh}"] = s
                print(f"{bn:6s}  {hh:>3d}  {s['mse_median']:>12.4e}  "
                      f"{s['ratio_median']:>14.2f}  "
                      f"{s['cos_div_median']:>12.4f}  "
                      f"{s['gate_mean_median']:>10.4f}")

    print()
    print("=" * 90)
    print("DECISION GATE — stable to H=15 iff median(MSE_15 / MSE_1) < 2.0")
    print("=" * 90)
    gate = {}
    for bn in BENCHMARKS:
        s15 = summary[bn].get("H=15", {})
        r = s15.get("ratio_median")
        stable = (r is not None) and (r < 2.0)
        gate[bn] = {"stable_to_H15": bool(stable), "ratio_median": r}
        r_str = f"{r:.2f}" if r is not None else "n/a"
        print(f"{bn:6s}  median_ratio_H15={r_str:>6s}   "
              f"{'STABLE' if stable else 'unstable'}")

    with open(out_path, "w") as f:
        json.dump({
            "per_cell": results,
            "summary_median": summary,
            "decision_gate": gate,
            "protocol": {
                "model": MODEL, "context_k": CONTEXT_K, "grid": GRID,
                "n_steps": N_STEPS,
                "n_trajectories": N_TRAJECTORIES,
                "seeds": SEEDS,
                "horizons": HORIZONS,
                "n_rollout_trajectories": N_ROLLOUT_TRAJS,
                "smoke": smoke,
            },
        }, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--smoke-max-h", type=int, default=15)
    args = p.parse_args()
    main(smoke=args.smoke, smoke_max_h=args.smoke_max_h)
