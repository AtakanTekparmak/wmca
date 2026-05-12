"""Diagnostic: init gate at sweep-optimal (eps, beta) per benchmark.

Tests whether the learned gate stays near sweep-optimal init or drifts away.
If it stays -> init was the problem (default init in wrong basin).
If it drifts -> gradient through CML actively pushes gate wrong (Mikhaeil et al.).

Usage:
    uv run --with scikit-learn,scipy python experiments/gate_init_diagnostic.py
"""
import json
import time
from pathlib import Path

import torch

from wmca.benchmarks import (
    generate_gol,
    generate_gray_scott,
    generate_heat,
    generate_ks,
    generate_rule110,
    generate_wireworld,
)
from wmca.model_registry import create_model, train_model


BENCHMARKS = {
    "heat": generate_heat,
    "gol": generate_gol,
    "gray_scott": generate_gray_scott,
    "ks": generate_ks,
    "rule110": generate_rule110,
    "wireworld": generate_wireworld,
}

# Sweep-optimal (eps, beta) per benchmark from completed sweeps
SWEEP_OPTIMAL = {
    "heat":       {"eps": 0.05, "beta": 0.15},
    "gol":        {"eps": 0.30, "beta": 0.50},
    "gray_scott": {"eps": 0.15, "beta": 0.15},
    "ks":         {"eps": 0.15, "beta": 0.01},
    "rule110":    {"eps": 0.30, "beta": 0.15},  # invariant, use defaults
    "wireworld":  {"eps": 0.30, "beta": 0.15},
}

# Also test default init for comparison
DEFAULT_INIT = {"eps": 0.30, "beta": 0.15}


def evaluate_model(model, X_test, Y_test, meta):
    model.eval()
    with torch.no_grad():
        preds = model(X_test)
    if meta["metric"] == "mse":
        return ((preds - Y_test) ** 2).mean().item()
    elif meta["metric"] == "accuracy":
        if meta["loss_type"] in ("ce", "cross_entropy"):
            pred_classes = preds.argmax(dim=1)
            true_classes = Y_test.argmax(dim=1) if Y_test.dim() == 4 and Y_test.shape[1] > 1 else Y_test.long().squeeze(1)
            return (pred_classes == true_classes).float().mean().item()
        else:
            pred_binary = (preds > 0.5).float()
            return (pred_binary == Y_test).float().mean().item()
    return float("nan")


def get_gate_values(model, X_test, meta):
    """Extract learned eps/beta from the gate."""
    cml = model.cml_2d
    if not hasattr(cml, "gate"):
        return None
    with torch.no_grad():
        sample = X_test[:10, :meta["out_channels"]]
        eb = torch.sigmoid(cml.gate(sample))
        eps_mean = (eb[:, 0] * cml.eps_max).mean().item()
        beta_mean = (eb[:, 1] * cml.beta_max).mean().item()
        eps_std = (eb[:, 0] * cml.eps_max).std().item()
        beta_std = (eb[:, 1] * cml.beta_max).std().item()
    return {"eps": eps_mean, "eps_std": eps_std, "beta": beta_mean, "beta_std": beta_std}


def run_one(bench_name, gen_fn, eps_init, beta_init, seed=42):
    """Train rescor_gate_static with custom init."""
    data = gen_fn(grid_size=16, seed=seed)
    meta = data.meta

    # Create model with custom eps/beta init via the CML gate
    model = create_model(
        "rescor_gate_static",
        in_channels=meta["in_channels"],
        out_channels=meta["out_channels"],
        grid_size=16, seed=seed,
    )

    # Override the gate init to sweep-optimal values
    cml = model.cml_2d
    with torch.no_grad():
        cml.gate.weight.zero_()
        eps_logit = torch.log(torch.tensor(eps_init / cml.eps_max) / (1 - eps_init / cml.eps_max))
        beta_logit = torch.log(torch.tensor(beta_init / cml.beta_max) / (1 - beta_init / cml.beta_max))
        cml.gate.bias[0] = eps_logit
        cml.gate.bias[1] = beta_logit

    # Verify init
    init_vals = get_gate_values(model, data.X_test, meta)

    t0 = time.time()
    model = train_model(
        model, data.X_train, data.Y_train,
        X_val=data.X_val, Y_val=data.Y_val,
        loss_type=meta["loss_type"],
        epochs=30, batch_size=64, lr=1e-3,
    )
    elapsed = time.time() - t0

    score = evaluate_model(model, data.X_test, data.Y_test, meta)
    final_vals = get_gate_values(model, data.X_test, meta)

    return {
        "score": score,
        "metric": meta["metric"],
        "init_eps": eps_init,
        "init_beta": beta_init,
        "init_gate": init_vals,
        "final_gate": final_vals,
        "eps_drift": abs(final_vals["eps"] - eps_init),
        "beta_drift": abs(final_vals["beta"] - beta_init),
        "time": elapsed,
    }


def main():
    seed = 42
    results = {}

    for bench_name, gen_fn in BENCHMARKS.items():
        results[bench_name] = {}
        opt = SWEEP_OPTIMAL[bench_name]

        print(f"\n{'='*60}")
        print(f"BENCHMARK: {bench_name}")
        print(f"Sweep-optimal: eps={opt['eps']}, beta={opt['beta']}")
        print(f"{'='*60}")

        # 1. Default init (eps=0.30, beta=0.15)
        print(f"\n  Default init (eps=0.30, beta=0.15):")
        r_default = run_one(bench_name, gen_fn, 0.30, 0.15, seed)
        results[bench_name]["default_init"] = r_default
        print(f"    score={r_default['score']:.6e}  "
              f"final eps={r_default['final_gate']['eps']:.3f}  "
              f"final beta={r_default['final_gate']['beta']:.3f}  "
              f"eps_drift={r_default['eps_drift']:.3f}  "
              f"beta_drift={r_default['beta_drift']:.3f}")

        # 2. Sweep-optimal init
        print(f"\n  Sweep-optimal init (eps={opt['eps']}, beta={opt['beta']}):")
        r_optimal = run_one(bench_name, gen_fn, opt["eps"], opt["beta"], seed)
        results[bench_name]["optimal_init"] = r_optimal
        print(f"    score={r_optimal['score']:.6e}  "
              f"final eps={r_optimal['final_gate']['eps']:.3f}  "
              f"final beta={r_optimal['final_gate']['beta']:.3f}  "
              f"eps_drift={r_optimal['eps_drift']:.3f}  "
              f"beta_drift={r_optimal['beta_drift']:.3f}")

        # Compare
        default_score = r_default["score"]
        optimal_score = r_optimal["score"]
        if r_default["metric"] == "mse":
            improvement = default_score / max(optimal_score, 1e-15)
            better = "optimal" if optimal_score < default_score else "default"
        else:
            improvement = optimal_score / max(default_score, 1e-15)
            better = "optimal" if optimal_score > default_score else "default"
        print(f"\n    {better} init wins ({improvement:.1f}x)")
        stayed = r_optimal["eps_drift"] < 0.05 and r_optimal["beta_drift"] < 0.05
        print(f"    Gate {'STAYED near optimal' if stayed else 'DRIFTED from optimal'}")

    # Summary
    print(f"\n{'='*80}")
    print("DIAGNOSTIC SUMMARY")
    print(f"{'='*80}")
    print(f"{'Benchmark':15s}  {'Default Score':>15s}  {'Optimal Score':>15s}  "
          f"{'Winner':>8s}  {'eps drift':>10s}  {'beta drift':>10s}  {'Verdict':>10s}")
    print("-" * 95)
    for bench_name in BENCHMARKS:
        d = results[bench_name]["default_init"]
        o = results[bench_name]["optimal_init"]
        if d["metric"] == "mse":
            winner = "optimal" if o["score"] < d["score"] else "default"
        else:
            winner = "optimal" if o["score"] > d["score"] else "default"
        stayed = o["eps_drift"] < 0.05 and o["beta_drift"] < 0.05
        verdict = "STAYED" if stayed else "DRIFTED"
        print(f"{bench_name:15s}  {d['score']:>15.6e}  {o['score']:>15.6e}  "
              f"{winner:>8s}  {o['eps_drift']:>10.3f}  {o['beta_drift']:>10.3f}  {verdict:>10s}")

    out_path = Path("experiments/results/gate_init_diagnostic.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
