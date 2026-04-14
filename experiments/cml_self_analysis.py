"""CML Self-Analysis Experiment.

Characterizes the CML reservoir across operating regimes:
  1. Lyapunov exponent vs r
  2. State fidelity (memory horizon) vs (r, M)
  3. Precision comparison (f32 / bf16 / int8)
  4. Feature richness (effective rank via SVD)

Usage:
    uv run --with scikit-learn,matplotlib python experiments/cml_self_analysis.py
    uv run --with scikit-learn,matplotlib python experiments/cml_self_analysis.py --no-wandb
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Ensure the project root is importable
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wmca.modules.cml import CML
from wmca.utils import pick_device

# Lazy imports for optional deps
def _get_ridge():
    from sklearn.linear_model import Ridge
    return Ridge

def _get_plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


# ===== Constants ==========================================================
R_SWEEP = [2.5, 3.0, 3.2, 3.4, 3.57, 3.6, 3.69, 3.8, 3.9, 3.99]
M_SWEEP = [1, 3, 5, 10, 15, 20, 30]
C = 256
BATCH_FIDELITY = 64
BATCH_SVD = 256
LYAP_ITERS = 1000
LYAP_X0 = 0.4
EPS = 0.3
BETA = 0.15
KERNEL_SIZE = 3
DEFAULT_R = 3.69
DEFAULT_M = 15
PLOTS_DIR = PROJECT_ROOT / "experiments" / "plots"


# ===== Analysis 1: Lyapunov exponent vs r ==================================
def lyapunov_exponents(r_values: list[float], n_iter: int = LYAP_ITERS,
                       x0: float = LYAP_X0) -> dict[float, float]:
    """Compute Lyapunov exponent for the logistic map at each r."""
    results = {}
    for r in r_values:
        x = x0
        log_sum = 0.0
        for _ in range(n_iter):
            deriv = abs(r * (1 - 2 * x))
            log_sum += np.log(max(deriv, 1e-30))
            x = r * x * (1 - x)
        results[r] = log_sum / n_iter
    return results


# ===== Analysis 2: State fidelity (memory horizon) =========================
def state_fidelity(r_values: list[float], m_values: list[int],
                   device: torch.device) -> dict[tuple[float, int], float]:
    """Reconstruction MSE of original drive from CML output via Ridge."""
    Ridge = _get_ridge()
    results = {}
    for r in r_values:
        for m in m_values:
            rng = torch.Generator().manual_seed(42)
            cml = CML(C=C, steps=m, kernel_size=KERNEL_SIZE,
                       r=r, eps=EPS, beta=BETA, rng=rng).to(device)
            cml.eval()

            drive = torch.rand(BATCH_FIDELITY, C, generator=torch.Generator().manual_seed(0),
                               device=device)
            with torch.no_grad():
                out = cml(drive)

            X = out.cpu().numpy()
            Y = drive.cpu().numpy()
            reg = Ridge(alpha=1.0)
            reg.fit(X, Y)
            pred = reg.predict(X)
            mse = float(np.mean((pred - Y) ** 2))
            results[(r, m)] = mse
            print(f"  fidelity  r={r:<5.2f}  M={m:<3d}  MSE={mse:.6f}")
    return results


# ===== Analysis 3: Precision comparison ====================================
def _quantize_int8(t: torch.Tensor) -> torch.Tensor:
    """Simulate int8 by rounding to 128 uniform levels in [0,1]."""
    return (t * 127).round() / 127


class CMLInt8Sim(torch.nn.Module):
    """CML with simulated int8 quantization after each step."""
    def __init__(self, base_cml: CML):
        super().__init__()
        self.base = base_cml

    def forward(self, drive: torch.Tensor) -> torch.Tensor:
        r = self.base.r.unsqueeze(0)
        eps = self.base.eps.unsqueeze(0)
        beta = self.base.beta.unsqueeze(0)
        one_minus_eps = 1.0 - eps
        one_minus_beta = 1.0 - beta
        k = self.base.kernel_size
        pad = k // 2

        grid = drive
        for _ in range(self.base.steps):
            mapped = r * grid * (1.0 - grid)
            m3 = mapped.unsqueeze(1)
            m_pad = torch.cat([m3[:, :, -pad:], m3, m3[:, :, :pad]], dim=2)
            local = torch.nn.functional.conv1d(m_pad, self.base.K_local).squeeze(1)
            global_cc = mapped @ self.base.W_cc
            coupled = 0.5 * (local + global_cc)
            physics = one_minus_eps * mapped + eps * coupled
            grid = one_minus_beta * physics + beta * drive
            grid = _quantize_int8(grid)

        return grid.clamp(1e-4, 1.0 - 1e-4)


def precision_comparison(device: torch.device) -> dict[str, dict[str, float]]:
    """Compare f32, bf16, and simulated int8 outputs and reconstruction."""
    Ridge = _get_ridge()
    rng = torch.Generator().manual_seed(42)
    cml_f32 = CML(C=C, steps=DEFAULT_M, kernel_size=KERNEL_SIZE,
                   r=DEFAULT_R, eps=EPS, beta=BETA, rng=rng).to(device).float()
    cml_f32.eval()

    drive = torch.rand(BATCH_FIDELITY, C, generator=torch.Generator().manual_seed(0),
                       device=device)

    # Float32 baseline
    with torch.no_grad():
        out_f32 = cml_f32(drive)

    # BFloat16
    cml_bf16 = CML(C=C, steps=DEFAULT_M, kernel_size=KERNEL_SIZE,
                    r=DEFAULT_R, eps=EPS, beta=BETA,
                    rng=torch.Generator().manual_seed(42)).to(device).to(torch.bfloat16)
    cml_bf16.eval()
    with torch.no_grad():
        out_bf16 = cml_bf16(drive.to(torch.bfloat16)).float()

    # Simulated int8
    cml_int8 = CMLInt8Sim(cml_f32)
    cml_int8.eval()
    with torch.no_grad():
        out_int8 = cml_int8(drive)

    mse_bf16 = float(torch.mean((out_f32 - out_bf16) ** 2).item())
    mse_int8 = float(torch.mean((out_f32 - out_int8) ** 2).item())

    # Reconstruction fidelity per precision
    Y = drive.cpu().numpy()
    fidelity = {}
    for label, out in [("f32", out_f32), ("bf16", out_bf16), ("int8", out_int8)]:
        X = out.cpu().numpy()
        reg = Ridge(alpha=1.0)
        reg.fit(X, Y)
        pred = reg.predict(X)
        fidelity[label] = float(np.mean((pred - Y) ** 2))

    return {
        "output_mse": {"bf16_vs_f32": mse_bf16, "int8_vs_f32": mse_int8},
        "recon_mse": fidelity,
    }


# ===== Analysis 4: Feature richness (effective rank) =======================
def feature_richness(r_values: list[float], device: torch.device) -> dict[float, int]:
    """Effective rank (# singular values > 1% of max) of CML output."""
    results = {}
    for r in r_values:
        rng = torch.Generator().manual_seed(42)
        cml = CML(C=C, steps=DEFAULT_M, kernel_size=KERNEL_SIZE,
                   r=r, eps=EPS, beta=BETA, rng=rng).to(device)
        cml.eval()

        drive = torch.rand(BATCH_SVD, C, generator=torch.Generator().manual_seed(0),
                           device=device)
        with torch.no_grad():
            out = cml(drive)

        S = torch.linalg.svdvals(out.cpu().float())
        threshold = 0.01 * S[0].item()
        eff_rank = int((S > threshold).sum().item())
        results[r] = eff_rank
        print(f"  rank  r={r:<5.2f}  eff_rank={eff_rank}")
    return results


# ===== Plotting ============================================================
def make_plots(lyap: dict, fidelity: dict, precision: dict, ranks: dict,
               log_wandb: bool):
    plt = _get_plt()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    wandb = None
    if log_wandb:
        import wandb as _wandb
        wandb = _wandb

    # --- Figure 1: Lyapunov exponent vs r ---
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    rs = sorted(lyap.keys())
    ax1.plot(rs, [lyap[r] for r in rs], "o-", color="tab:red")
    ax1.axhline(0, color="gray", ls="--", lw=0.8)
    ax1.set_xlabel("r")
    ax1.set_ylabel("Lyapunov exponent λ")
    ax1.set_title("Logistic Map: Lyapunov Exponent vs r")
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    fig1.savefig(PLOTS_DIR / "lyapunov_vs_r.png", dpi=150)
    if wandb:
        wandb.log({"plots/lyapunov_vs_r": wandb.Image(fig1)})
    plt.close(fig1)

    # --- Figure 2: Heatmap of (r, M) vs reconstruction MSE ---
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    mse_matrix = np.zeros((len(R_SWEEP), len(M_SWEEP)))
    for i, r in enumerate(R_SWEEP):
        for j, m in enumerate(M_SWEEP):
            mse_matrix[i, j] = fidelity[(r, m)]
    im = ax2.imshow(mse_matrix, aspect="auto", origin="lower",
                    cmap="viridis_r")
    ax2.set_xticks(range(len(M_SWEEP)))
    ax2.set_xticklabels([str(m) for m in M_SWEEP])
    ax2.set_yticks(range(len(R_SWEEP)))
    ax2.set_yticklabels([f"{r:.2f}" for r in R_SWEEP])
    ax2.set_xlabel("CML steps M")
    ax2.set_ylabel("r")
    ax2.set_title("State Fidelity: Reconstruction MSE (lower = better memory)")
    fig2.colorbar(im, ax=ax2, label="MSE")
    # Annotate cells
    for i in range(len(R_SWEEP)):
        for j in range(len(M_SWEEP)):
            val = mse_matrix[i, j]
            ax2.text(j, i, f"{val:.4f}", ha="center", va="center",
                     fontsize=6, color="white" if val > mse_matrix.max() * 0.5 else "black")
    fig2.tight_layout()
    fig2.savefig(PLOTS_DIR / "fidelity_heatmap.png", dpi=150)
    if wandb:
        wandb.log({"plots/fidelity_heatmap": wandb.Image(fig2)})
    plt.close(fig2)

    # --- Figure 3: Precision comparison bar chart ---
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(10, 5))
    # Output MSE vs f32
    labels_out = list(precision["output_mse"].keys())
    vals_out = [precision["output_mse"][k] for k in labels_out]
    ax3a.bar(labels_out, vals_out, color=["tab:blue", "tab:orange"])
    ax3a.set_ylabel("MSE vs float32")
    ax3a.set_title("Output Divergence from f32")
    ax3a.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))
    # Reconstruction MSE per precision
    labels_rec = list(precision["recon_mse"].keys())
    vals_rec = [precision["recon_mse"][k] for k in labels_rec]
    ax3b.bar(labels_rec, vals_rec, color=["tab:green", "tab:blue", "tab:orange"])
    ax3b.set_ylabel("Reconstruction MSE")
    ax3b.set_title("Reconstruction Fidelity by Precision")
    ax3b.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))
    fig3.suptitle(f"Precision Comparison (r={DEFAULT_R}, M={DEFAULT_M})")
    fig3.tight_layout()
    fig3.savefig(PLOTS_DIR / "precision_comparison.png", dpi=150)
    if wandb:
        wandb.log({"plots/precision_comparison": wandb.Image(fig3)})
    plt.close(fig3)

    # --- Figure 4: Effective rank vs r ---
    fig4, ax4 = plt.subplots(figsize=(8, 5))
    rs_rank = sorted(ranks.keys())
    ax4.plot(rs_rank, [ranks[r] for r in rs_rank], "s-", color="tab:purple")
    ax4.set_xlabel("r")
    ax4.set_ylabel("Effective rank")
    ax4.set_title(f"Feature Richness: Effective Rank vs r (M={DEFAULT_M})")
    ax4.set_ylim(bottom=0)
    ax4.grid(True, alpha=0.3)
    fig4.tight_layout()
    fig4.savefig(PLOTS_DIR / "effective_rank_vs_r.png", dpi=150)
    if wandb:
        wandb.log({"plots/effective_rank_vs_r": wandb.Image(fig4)})
    plt.close(fig4)

    print(f"\nPlots saved to {PLOTS_DIR}/")


# ===== Summary =============================================================
def print_summary(lyap: dict, fidelity: dict, precision: dict, ranks: dict):
    print("\n" + "=" * 72)
    print("CML SELF-ANALYSIS SUMMARY")
    print("=" * 72)

    print("\n--- Lyapunov Exponents ---")
    print(f"{'r':>6s}  {'lambda':>10s}  {'regime'}")
    for r in sorted(lyap):
        lam = lyap[r]
        regime = "chaotic" if lam > 0 else "stable"
        print(f"{r:6.2f}  {lam:10.4f}  {regime}")

    print(f"\n--- State Fidelity (reconstruction MSE, lower=better) ---")
    header = f"{'r':>6s}" + "".join(f"  M={m:<3d}" for m in M_SWEEP)
    print(header)
    for r in R_SWEEP:
        row = f"{r:6.2f}"
        for m in M_SWEEP:
            row += f"  {fidelity[(r, m)]:.4f}"
        print(row)

    print(f"\n--- Precision Comparison (r={DEFAULT_R}, M={DEFAULT_M}) ---")
    print(f"  Output MSE bf16 vs f32: {precision['output_mse']['bf16_vs_f32']:.8f}")
    print(f"  Output MSE int8 vs f32: {precision['output_mse']['int8_vs_f32']:.8f}")
    for k, v in precision["recon_mse"].items():
        print(f"  Reconstruction MSE ({k}): {v:.6f}")

    print(f"\n--- Feature Richness (effective rank, M={DEFAULT_M}) ---")
    print(f"{'r':>6s}  {'eff_rank':>8s}")
    for r in sorted(ranks):
        print(f"{r:6.2f}  {ranks[r]:8d}")

    print("=" * 72)


# ===== Wandb logging =======================================================
def log_metrics_to_wandb(lyap: dict, fidelity: dict, precision: dict,
                         ranks: dict):
    import wandb
    for r, lam in lyap.items():
        wandb.log({"lyapunov/r": r, "lyapunov/lambda": lam})
    for (r, m), mse in fidelity.items():
        wandb.log({"fidelity/r": r, "fidelity/M": m, "fidelity/mse": mse})
    for k, v in precision["output_mse"].items():
        wandb.log({f"precision/output_mse_{k}": v})
    for k, v in precision["recon_mse"].items():
        wandb.log({f"precision/recon_mse_{k}": v})
    for r, rank in ranks.items():
        wandb.log({"rank/r": r, "rank/effective_rank": rank})


# ===== Main ================================================================
def main():
    parser = argparse.ArgumentParser(description="CML Self-Analysis Experiment")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Skip wandb logging")
    args = parser.parse_args()

    os.environ.setdefault("FORCE_CPU", "1")
    device = pick_device()

    config = dict(
        r_sweep=R_SWEEP, m_sweep=M_SWEEP, C=C,
        batch_fidelity=BATCH_FIDELITY, batch_svd=BATCH_SVD,
        lyap_iters=LYAP_ITERS, lyap_x0=LYAP_X0,
        eps=EPS, beta=BETA, kernel_size=KERNEL_SIZE,
        default_r=DEFAULT_R, default_m=DEFAULT_M,
    )

    log_wandb = not args.no_wandb
    if log_wandb:
        from wmca.training import init_wandb
        init_wandb("cml-self-analysis", config=config,
                   tags=["cml", "self-analysis", "reservoir"])

    print("=" * 72)
    print("CML SELF-ANALYSIS EXPERIMENT")
    print("=" * 72)

    # 1. Lyapunov
    print("\n[1/4] Computing Lyapunov exponents ...")
    lyap = lyapunov_exponents(R_SWEEP)
    for r in sorted(lyap):
        print(f"  r={r:<5.2f}  lambda={lyap[r]:.4f}")

    # 2. State fidelity
    print("\n[2/4] Measuring state fidelity (memory horizon) ...")
    fidelity = state_fidelity(R_SWEEP, M_SWEEP, device)

    # 3. Precision
    print("\n[3/4] Precision comparison ...")
    prec = precision_comparison(device)
    print(f"  bf16 vs f32 output MSE: {prec['output_mse']['bf16_vs_f32']:.8f}")
    print(f"  int8 vs f32 output MSE: {prec['output_mse']['int8_vs_f32']:.8f}")
    for k, v in prec["recon_mse"].items():
        print(f"  recon MSE ({k}): {v:.6f}")

    # 4. Feature richness
    print("\n[4/4] Computing feature richness (effective rank) ...")
    ranks = feature_richness(R_SWEEP, device)

    # Plots
    print("\nGenerating plots ...")
    make_plots(lyap, fidelity, prec, ranks, log_wandb)

    # Wandb
    if log_wandb:
        log_metrics_to_wandb(lyap, fidelity, prec, ranks)
        import wandb
        wandb.finish()

    # Summary
    print_summary(lyap, fidelity, prec, ranks)


if __name__ == "__main__":
    main()
