"""Pushforward-trick rollout-stability probe for rescor_mamba_rand.

Sibling of ``pushforward_rollout_probe.py`` (rens K=32). Loads the
mamba_rand checkpoints produced by
``experiments/pushforward_ablation_mamba.py`` and reports per-step
MSE + ratio at H={15, 50, 100} for pushforward in {False, True} ×
benches × seeds.

Mamba_rand takes rank-5 input (B, K=4, C, H, W). The probe maintains
a rolling K-frame buffer seeded from K ground-truth test frames and
advances by appending model predictions. Per the rollout-stability
plan §6.2, we re-generate the benchmark with ``context_k=1`` for
single-frame ground-truth indexing, and build the K-frame buffer
manually.

Expects ``experiments/results/pushforward_ckpts_mamba/mamba_rand_pf{0|1}_{bench}_seed{seed}.pt``.

Saves:
  - experiments/results/pushforward_rollout_probe_mamba.json

Usage:
    PYTHONPATH=src PYTHONUNBUFFERED=1 uv run --with scikit-learn,scipy \\
        python dreamerv3_scaffolding/pushforward_rollout_probe_mamba.py
"""
from __future__ import annotations

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

PUSHFORWARDS = [False, True]
SEEDS = [42, 43, 44]
MODEL = "rescor_mamba_rand"
N_STEPS = 105
N_TRAJECTORIES = 200
GRID = 16
CONTEXT_K = 4
N_ROLLOUT_TRAJS = 20
HORIZONS = [15, 50, 100]

CKPT_DIR = Path("experiments/results/pushforward_ckpts_mamba")


def cosine_div(pred: np.ndarray, true: np.ndarray) -> float:
    p = pred.reshape(-1)
    t = true.reshape(-1)
    num = float((p * t).sum())
    den = float(np.linalg.norm(p) * np.linalg.norm(t)) + 1e-8
    return 1.0 - num / den


def ckpt_path_for(pf, bench_name, seed):
    pf_tag = 1 if pf else 0
    return CKPT_DIR / f"mamba_rand_pf{pf_tag}_{bench_name}_seed{seed}.pt"


def probe_one(pf, bench_name, gen_fn, seed, max_h: int | None = None):
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

    ckpt = ckpt_path_for(pf, bench_name, seed)
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(state)

    T = N_STEPS
    X_test = data.X_test                                 # (N_traj*T, C, H, W)
    Y_test = data.Y_test
    n_test_traj = X_test.shape[0] // T
    n_use = min(N_ROLLOUT_TRAJS, n_test_traj)

    probe_max_h = max_h if max_h is not None else max(HORIZONS)
    per_step_mse = np.zeros(probe_max_h, dtype=np.float64)
    per_step_cos = np.zeros(probe_max_h, dtype=np.float64)

    model.eval()
    device = next(model.parameters()).device
    K = CONTEXT_K
    with torch.no_grad():
        for ti in range(n_use):
            # Seed K-frame buffer with ground-truth: [X[ti*T], Y[ti*T..K-2]]
            seed_frames = [X_test[ti * T]]
            for k in range(K - 1):
                seed_frames.append(Y_test[ti * T + k])
            buf = torch.stack(seed_frames, dim=0).unsqueeze(0).to(device)  # (1,K,C,H,W)

            for s in range(probe_max_h):
                pred = model(buf)                                          # (1,C,H,W)
                gt_idx = ti * T + K - 1 + s
                if gt_idx >= Y_test.shape[0] or (gt_idx // T) != ti:
                    break
                gt = Y_test[gt_idx].unsqueeze(0).to(device)
                per_step_mse[s] += float(((pred - gt) ** 2).mean().item()) / n_use
                p_np = pred.cpu().numpy()
                g_np = gt.cpu().numpy()
                per_step_cos[s] += cosine_div(p_np, g_np) / n_use
                pred_clamped = pred.clamp(0.0, 1.0).unsqueeze(1)            # (1,1,C,H,W)
                buf = torch.cat([buf[:, 1:], pred_clamped], dim=1)

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
        }

    return {
        "pushforward": pf,
        "bench": bench_name,
        "seed": seed,
        "params": pc,
        "probe_time_s": time.time() - t0,
        "mse_per_step": per_step_mse.tolist(),
        "cosine_div_per_step": per_step_cos.tolist(),
        "horizons": horizons_out,
        "n_rollout_trajectories": n_use,
        "ckpt": str(ckpt),
        "max_h": probe_max_h,
    }


def pf_key(pf: bool) -> str:
    return f"pushforward={pf}"


def main(smoke: bool = False, smoke_max_h: int = 15):
    out_path = Path("experiments/results/pushforward_rollout_probe_mamba.json")
    if smoke:
        out_path = Path("experiments/results/pushforward_rollout_probe_mamba_smoke.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = {pf_key(p): {bn: {} for bn in BENCHMARKS} for p in PUSHFORWARDS}

    pfs_to_run = PUSHFORWARDS
    seeds_to_run = SEEDS
    benches_to_run = dict(BENCHMARKS)
    if smoke:
        pfs_to_run = [True]
        seeds_to_run = [42]
        benches_to_run = {"heat": generate_heat}

    if out_path.exists():
        try:
            with open(out_path, "r") as f:
                prior = json.load(f)
            src = prior.get("per_cell", prior)
            n_loaded = 0
            for pk in results:
                if pk not in src:
                    continue
                for bn in BENCHMARKS:
                    if bn not in src[pk]:
                        continue
                    for seed_str, cell in src[pk][bn].items():
                        if cell and cell.get("mse_per_step"):
                            results[pk][bn][seed_str] = cell
                            n_loaded += 1
            print(f"[resume] loaded {n_loaded} completed cells from {out_path}")
        except Exception as e:
            print(f"[resume] failed ({e}); starting fresh")

    for pf in pfs_to_run:
        pk = pf_key(pf)
        for bn, gf in benches_to_run.items():
            for seed in seeds_to_run:
                seed_str = str(seed)
                if (results[pk][bn].get(seed_str)
                        and results[pk][bn][seed_str].get("mse_per_step")):
                    r = results[pk][bn][seed_str]
                    h15 = r.get("horizons", {}).get("H=15", {})
                    print(f"[pf={pf} {bn:4s} seed={seed}] resumed  "
                          f"H15_ratio={h15.get('mse_ratio_vs_step1', float('nan')):.2f}")
                    continue
                ckpt = ckpt_path_for(pf, bn, seed)
                if not ckpt.exists():
                    print(f"[pf={pf} {bn:4s} seed={seed}] MISSING checkpoint {ckpt}; skipping")
                    results[pk][bn][seed_str] = {
                        "error": "checkpoint missing", "ckpt": str(ckpt),
                        "pushforward": pf, "bench": bn, "seed": seed,
                    }
                    continue
                print("=" * 78)
                print(f"probe  pushforward={pf}  {bn}  seed={seed}  context_k={CONTEXT_K}")
                print("=" * 78)
                try:
                    r = probe_one(pf, bn, gf, seed,
                                  max_h=smoke_max_h if smoke else None)
                    results[pk][bn][seed_str] = r
                    print(f"  probe time: {r['probe_time_s']:.1f}s  params={r['params']}")
                    print(f"  step 1 MSE = {r['mse_per_step'][0]:.4e}")
                    for hh in HORIZONS:
                        if hh > r["max_h"]:
                            continue
                        c = r["horizons"][f"H={hh}"]
                        print(f"  H={hh:3d}  MSE={c['mse_final']:.4e}  "
                              f"ratio_vs_step1={c['mse_ratio_vs_step1']:.2f}  "
                              f"cos_div={c['cosine_div_final']:.4f}")
                except Exception as e:
                    print(f"  FAILED: {type(e).__name__}: {e}")
                    import traceback; traceback.print_exc()
                    results[pk][bn][seed_str] = {
                        "error": str(e), "pushforward": pf, "bench": bn, "seed": seed,
                    }
                with open(out_path, "w") as f:
                    json.dump({
                        "per_cell": results,
                        "protocol": {
                            "model": MODEL, "context_k": CONTEXT_K, "grid": GRID,
                            "n_steps": N_STEPS,
                            "n_trajectories": N_TRAJECTORIES,
                            "pushforwards": PUSHFORWARDS, "seeds": SEEDS,
                            "horizons": HORIZONS,
                            "n_rollout_trajectories": N_ROLLOUT_TRAJS,
                            "smoke": smoke,
                        },
                    }, f, indent=2)
                gc.collect()

    print()
    print("=" * 90)
    print("SUMMARY — pushforward mamba_rand rollout stability (median across seeds)")
    print("=" * 90)
    hdr = (f"{'pushfwd':>8s}  {'bench':6s}  {'H':>3s}  {'MSE median':>12s}  "
           f"{'ratio median':>14s}  {'cos_div med':>12s}")
    print(hdr)
    print("-" * len(hdr))
    summary = {pf_key(p): {bn: {} for bn in BENCHMARKS} for p in PUSHFORWARDS}
    for pf in PUSHFORWARDS:
        pk = pf_key(pf)
        for bn in BENCHMARKS:
            for hh in HORIZONS:
                ratios, mses, cosds = [], [], []
                for seed_str, cell in results[pk][bn].items():
                    if "horizons" not in cell:
                        continue
                    c = cell["horizons"].get(f"H={hh}")
                    if not c:
                        continue
                    ratios.append(c["mse_ratio_vs_step1"])
                    mses.append(c["mse_final"])
                    cosds.append(c["cosine_div_final"])
                if ratios:
                    s = {
                        "mse_median": median(mses),
                        "ratio_median": median(ratios),
                        "cos_div_median": median(cosds),
                        "n_seeds": len(ratios),
                    }
                    summary[pk][bn][f"H={hh}"] = s
                    print(f"{str(pf):>8s}  {bn:6s}  {hh:>3d}  "
                          f"{s['mse_median']:>12.4e}  "
                          f"{s['ratio_median']:>14.2f}  "
                          f"{s['cos_div_median']:>12.4f}")

    print()
    print("=" * 90)
    print("DECISION GATE — stable to H=15 iff median(MSE_15 / MSE_1) < 2.0")
    print("=" * 90)
    gate = {pf_key(p): {} for p in PUSHFORWARDS}
    for pf in PUSHFORWARDS:
        pk = pf_key(pf)
        for bn in BENCHMARKS:
            s15 = summary[pk][bn].get("H=15", {})
            r = s15.get("ratio_median")
            stable = (r is not None) and (r < 2.0)
            gate[pk][bn] = {"stable_to_H15": bool(stable), "ratio_median": r}
            r_str = f"{r:.2f}" if r is not None else "n/a"
            print(f"pf={str(pf):<5} {bn:6s}  median_ratio_H15={r_str:>6s}   "
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
                "pushforwards": PUSHFORWARDS, "seeds": SEEDS,
                "horizons": HORIZONS,
                "n_rollout_trajectories": N_ROLLOUT_TRAJS,
                "smoke": smoke,
            },
        }, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--smoke-max-h", type=int, default=15)
    args = p.parse_args()
    main(smoke=args.smoke, smoke_max_h=args.smoke_max_h)
