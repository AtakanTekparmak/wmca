"""Rollout probe for the drift-gated + MSDC combo experiment.

Loads checkpoints from `experiments/results/drift_gated_msdc_ckpts/`
(saved by `experiments/drift_gated_msdc_ablation.py`) and rolls each
autoregressively at H ∈ {15, 50, 100}.

Reports per-step MSE / cos_div + per-step mean gate value — same as
the multistep sibling probe; the load-bearing question for MSDC is
whether the gate develops a meaningful drift-discrimination signal
after coherence-aware training.
"""
from __future__ import annotations

import gc
import json
import time
from pathlib import Path
from statistics import median

import numpy as np
import torch

from wmca.benchmarks import generate_gray_scott, generate_ks
from wmca.model_registry import create_model


BENCHMARKS = {"gs": generate_gray_scott, "ks": generate_ks}
SEEDS = [42, 43, 44]
MODEL = "rescor_mamba_gated_rand"
N_STEPS = 105
N_TRAJECTORIES = 200
GRID = 16
CONTEXT_K = 4
N_ROLLOUT_TRAJS = 20
HORIZONS = [15, 50, 100]

CKPT_DIR = Path("experiments/results/drift_gated_msdc_ckpts")


def cosine_div(pred, true):
    p = pred.reshape(-1)
    t = true.reshape(-1)
    num = float((p * t).sum())
    den = float(np.linalg.norm(p) * np.linalg.norm(t)) + 1e-8
    return 1.0 - num / den


def ckpt_path_for(bench, seed):
    return CKPT_DIR / f"{bench}_seed{seed}.pt"


def probe_one(bench, gen_fn, seed):
    t0 = time.time()
    data = gen_fn(grid_size=GRID, seed=seed, n_steps=N_STEPS,
                  n_trajectories=N_TRAJECTORIES, context_k=1)
    meta = data.meta
    model = create_model(MODEL, in_channels=meta["in_channels"],
                         out_channels=meta["out_channels"],
                         grid_size=GRID, seed=seed)
    pc = model.param_count() if hasattr(model, "param_count") else None
    ckpt = ckpt_path_for(bench, seed)
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    gate_bias_loaded = float(model.gate_bias.detach().cpu().item())
    gate_scale_loaded = float(model.gate_scale.detach().cpu().item())

    T = N_STEPS
    X_test = data.X_test
    Y_test = data.Y_test
    n_test = X_test.shape[0] // T
    n_use = min(N_ROLLOUT_TRAJS, n_test)
    max_h = max(HORIZONS)

    per_step_mse = np.zeros(max_h)
    per_step_cos = np.zeros(max_h)
    per_step_gate = np.zeros(max_h)

    model.eval()
    device = next(model.parameters()).device
    K = CONTEXT_K
    with torch.no_grad():
        for ti in range(n_use):
            seed_frames = [X_test[ti * T]]
            for k in range(K - 1):
                seed_frames.append(Y_test[ti * T + k])
            buf = torch.stack(seed_frames, dim=0).unsqueeze(0).to(device)
            for s in range(max_h):
                pred = model(buf)
                # Use the model's own compute_gate (same source-of-truth as
                # forward()) on the current state buffer for telemetry.
                with torch.no_grad():
                    gate = model.compute_gate(buf)  # accepts rank-5
                gt_idx = ti * T + K - 1 + s
                if gt_idx >= Y_test.shape[0] or (gt_idx // T) != ti:
                    break
                gt = Y_test[gt_idx].unsqueeze(0).to(device)
                per_step_mse[s] += float(((pred - gt) ** 2).mean().item()) / n_use
                per_step_cos[s] += cosine_div(pred.cpu().numpy(),
                                              gt.cpu().numpy()) / n_use
                per_step_gate[s] += float(gate.mean().item()) / n_use
                buf = torch.cat([buf[:, 1:],
                                  pred.clamp(0.0, 1.0).unsqueeze(1)], dim=1)

    horizons_out = {}
    mse1 = float(per_step_mse[0])
    for h in HORIZONS:
        m_h = float(per_step_mse[h - 1])
        horizons_out[f"H={h}"] = {
            "mse_final": m_h,
            "cosine_div_final": float(per_step_cos[h - 1]),
            "mse_ratio_vs_step1": (m_h / mse1) if mse1 > 0 else float("inf"),
            "gate_mean_final": float(per_step_gate[h - 1]),
        }
    return {
        "bench": bench, "seed": seed, "params": pc,
        "probe_time_s": time.time() - t0,
        "mse_per_step": per_step_mse.tolist(),
        "cosine_div_per_step": per_step_cos.tolist(),
        "gate_mean_per_step": per_step_gate.tolist(),
        "horizons": horizons_out,
        "ckpt": str(ckpt),
        "gate_scale_loaded": gate_scale_loaded,
        "gate_bias_loaded": gate_bias_loaded,
    }


def main():
    out_path = Path("experiments/results/drift_gated_msdc_rollout_probe.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = {bn: {} for bn in BENCHMARKS}
    if out_path.exists():
        try:
            with open(out_path, "r") as f:
                prior = json.load(f)
            src = prior.get("per_cell", prior)
            for bn in BENCHMARKS:
                if bn in src:
                    for sk, cell in src[bn].items():
                        if cell and cell.get("mse_per_step"):
                            results[bn][sk] = cell
        except Exception as e:
            print(f"[resume] failed ({e})")

    for bn, gf in BENCHMARKS.items():
        for seed in SEEDS:
            sk = str(seed)
            if results[bn].get(sk):
                r = results[bn][sk]
                h15 = r.get("horizons", {}).get("H=15", {})
                print(f"[{bn} s={seed}] resumed H15_ratio={h15.get('mse_ratio_vs_step1', float('nan')):.2f}")
                continue
            ckpt = ckpt_path_for(bn, seed)
            if not ckpt.exists():
                print(f"[{bn} s={seed}] MISSING {ckpt}; skipping")
                continue
            print("=" * 78)
            print(f"probe MSDC  {bn}  seed={seed}  context_k={CONTEXT_K}")
            print("=" * 78)
            try:
                r = probe_one(bn, gf, seed)
                results[bn][sk] = r
                print(f"  probe time: {r['probe_time_s']:.1f}s")
                print(f"  gate_scale={r['gate_scale_loaded']:.3f} gate_bias={r['gate_bias_loaded']:.3f}")
                print(f"  step 1 MSE = {r['mse_per_step'][0]:.4e}")
                for hh in HORIZONS:
                    c = r["horizons"][f"H={hh}"]
                    print(f"  H={hh:3d}  MSE={c['mse_final']:.4e}  "
                          f"ratio={c['mse_ratio_vs_step1']:.2f}  "
                          f"cos_div={c['cosine_div_final']:.4f}  "
                          f"gate_mean={c['gate_mean_final']:.3f}")
            except Exception as e:
                import traceback; traceback.print_exc()
                results[bn][sk] = {"error": str(e), "bench": bn, "seed": seed}
            with open(out_path, "w") as f:
                json.dump({"per_cell": results}, f, indent=2)
            gc.collect()

    print()
    print("=" * 100)
    print("SUMMARY — drift-gated + MSDC combo (median across seeds)")
    print("=" * 100)
    hdr = (f"{'bench':6s}  {'H':>3s}  {'MSE median':>12s}  {'ratio median':>14s}  "
           f"{'cos_div med':>12s}  {'gate_mean med':>13s}")
    print(hdr)
    print("-" * len(hdr))
    summary = {bn: {} for bn in BENCHMARKS}
    for bn in BENCHMARKS:
        for hh in HORIZONS:
            mses, ratios, cosds, gates = [], [], [], []
            for sk, cell in results[bn].items():
                if "horizons" not in cell:
                    continue
                c = cell["horizons"].get(f"H={hh}")
                if not c:
                    continue
                mses.append(c["mse_final"])
                ratios.append(c["mse_ratio_vs_step1"])
                cosds.append(c["cosine_div_final"])
                gates.append(c["gate_mean_final"])
            if mses:
                summary[bn][f"H={hh}"] = {
                    "mse_median": median(mses),
                    "ratio_median": median(ratios),
                    "cos_div_median": median(cosds),
                    "gate_mean_median": median(gates),
                    "n_seeds": len(mses),
                }
                s = summary[bn][f"H={hh}"]
                print(f"{bn:6s}  {hh:>3d}  "
                      f"{s['mse_median']:>12.4e}  "
                      f"{s['ratio_median']:>14.2f}  "
                      f"{s['cos_div_median']:>12.4f}  "
                      f"{s['gate_mean_median']:>13.3f}")

    print()
    print("=" * 100)
    print("DECISION GATE — stable to H=15 iff median(MSE_15/MSE_1) < 2.0")
    print("=" * 100)
    for bn in BENCHMARKS:
        s15 = summary[bn].get("H=15", {})
        r = s15.get("ratio_median")
        stable = (r is not None) and (r < 2.0)
        rs = f"{r:.2f}" if r is not None else "n/a"
        print(f"{bn:6s}  median_ratio_H15={rs:>6s}   "
              f"{'STABLE' if stable else 'unstable'}")

    with open(out_path, "w") as f:
        json.dump({"per_cell": results, "summary_median": summary}, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
