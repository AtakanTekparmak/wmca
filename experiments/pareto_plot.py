"""Pareto frontier plot: parameter efficiency across all WMCA experiments."""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ── Style config ──────────────────────────────────────────────────────
MODEL_STYLES = {
    "ResCor(D)":   dict(color="red",    marker="*", s=120, zorder=5),
    "PureNCA":     dict(color="blue",   marker="o", s=80,  zorder=5),
    "Conv2D":      dict(color="green",  marker="D", s=70,  zorder=4),
    "CML+Ridge":   dict(color="gray",   marker="s", s=70,  zorder=3),
    "MLP":         dict(color="orange",  marker="^", s=70,  zorder=3),
    "NCA":         dict(color="dodgerblue", marker="o", s=70, zorder=4),
}

def style(model):
    return MODEL_STYLES.get(model, dict(color="black", marker="x", s=50, zorder=2))

# ── Data ──────────────────────────────────────────────────────────────
# (model, params, metric, benchmark_label)

continuous_data = [
    # KS equation N=128 h=100
    ("ResCor(D)", 321,   0.000253, "KS-128"),
    ("Conv2D",    2625,  0.000277, "KS-128"),
    ("PureNCA",   177,   0.005919, "KS-128"),
    # Heat 64x64 h=50
    ("ResCor(D)", 321,   0.407,    "Heat-64"),
    ("Conv2D",    2625,  0.373,    "Heat-64"),
    ("PureNCA",   177,   0.492,    "Heat-64"),
    # Heat 16x16 h=10
    ("ResCor(D)", 321,   1e-6,     "Heat-16"),   # ~0
    ("Conv2D",    2625,  1e-6,     "Heat-16"),   # ~0
    ("PureNCA",   177,   6.8e-3,   "Heat-16"),
    ("CML+Ridge", 65792, 2.1e-2,  "Heat-16"),
    # Gray-Scott h=50
    ("CML+Ridge", 526336, 1.2e-4, "GS"),
    ("NCA",       338,    3.3e-4, "GS"),
    ("Conv2D",    2914,   1.3e-4, "GS"),
]

discrete_data = [
    # GoL 32x32
    ("NCA",       449,     97.23,  "GoL-32"),
    ("Conv2D",    2625,    97.91,  "GoL-32"),
    ("CML+Ridge", 1049600, 78.02, "GoL-32"),
    ("MLP",       1050112, 74.57, "GoL-32"),
    # GoL 64x64
    ("Conv2D",    2625,    98.95,  "GoL-64"),
    ("ResCor(D)", 321,     98.66,  "GoL-64"),
    ("PureNCA",   177,     98.64,  "GoL-64"),
    # Rule 110
    ("PureNCA",   81,      99.06,  "Rule110"),
    ("Conv2D",    897,     99.24,  "Rule110"),   # "Conv"
    ("CML+Ridge", 4160,    68.11,  "Rule110"),
    # Wireworld
    ("PureNCA",   1316,    99.90,  "Wireworld"),
    ("Conv2D",    11588,   99.89,  "Wireworld"), # "Conv"
    ("CML+Ridge", 1049600, 93.72, "Wireworld"),
]


def pareto_frontier_min(xs, ys):
    """Return indices on the Pareto frontier (minimise both x and y)."""
    pts = sorted(range(len(xs)), key=lambda i: xs[i])
    frontier = []
    best_y = float("inf")
    for i in pts:
        if ys[i] < best_y:
            frontier.append(i)
            best_y = ys[i]
    return frontier


def pareto_frontier_max(xs, ys):
    """Return indices on the Pareto frontier (minimise x, maximise y)."""
    pts = sorted(range(len(xs)), key=lambda i: xs[i])
    frontier = []
    best_y = -float("inf")
    for i in pts:
        if ys[i] > best_y:
            frontier.append(i)
            best_y = ys[i]
    return frontier


# ── Plot ──────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# --- Left panel: continuous (MSE, lower=better) ---
for model, params, mse, label in continuous_data:
    st = style(model)
    ax1.scatter(params, mse, **st, edgecolors="k", linewidths=0.5)
    ax1.annotate(label, (params, mse), fontsize=6, ha="left",
                 xytext=(5, 3), textcoords="offset points", alpha=0.8)

# Pareto frontier (minimise params AND mse)
xs = [d[1] for d in continuous_data]
ys = [d[2] for d in continuous_data]
pf = pareto_frontier_min(xs, ys)
pf_sorted = sorted(pf, key=lambda i: xs[i])
ax1.plot([xs[i] for i in pf_sorted], [ys[i] for i in pf_sorted],
         "k--", linewidth=1.2, alpha=0.5, label="Pareto frontier")

ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_xlabel("Trainable Parameters", fontsize=12)
ax1.set_ylabel("Rollout MSE (lower = better)", fontsize=12)
ax1.set_title("Continuous Dynamics", fontsize=13, fontweight="bold")
ax1.grid(True, alpha=0.3, which="both")

# --- Right panel: discrete (accuracy, higher=better) ---
for model, params, acc, label in discrete_data:
    st = style(model)
    ax2.scatter(params, acc, **st, edgecolors="k", linewidths=0.5)
    ax2.annotate(label, (params, acc), fontsize=6, ha="left",
                 xytext=(5, 3), textcoords="offset points", alpha=0.8)

# Pareto frontier (minimise params, maximise accuracy)
xs2 = [d[1] for d in discrete_data]
ys2 = [d[2] for d in discrete_data]
pf2 = pareto_frontier_max(xs2, ys2)
pf2_sorted = sorted(pf2, key=lambda i: xs2[i])
ax2.plot([xs2[i] for i in pf2_sorted], [ys2[i] for i in pf2_sorted],
         "k--", linewidth=1.2, alpha=0.5, label="Pareto frontier")

ax2.set_xscale("log")
ax2.set_xlabel("Trainable Parameters", fontsize=12)
ax2.set_ylabel("Accuracy % (higher = better)", fontsize=12)
ax2.set_title("Discrete Dynamics", fontsize=13, fontweight="bold")
ax2.grid(True, alpha=0.3, which="both")

# ── Shared legend ─────────────────────────────────────────────────────
handles = []
for name, st in MODEL_STYLES.items():
    h = ax1.scatter([], [], color=st["color"], marker=st["marker"],
                    s=st["s"], edgecolors="k", linewidths=0.5, label=name)
    handles.append(h)
# Add frontier line to legend
from matplotlib.lines import Line2D
handles.append(Line2D([0], [0], color="k", linestyle="--", linewidth=1.2,
                       alpha=0.5, label="Pareto frontier"))

fig.legend(handles=handles, loc="lower center", ncol=len(handles),
           fontsize=10, frameon=True, bbox_to_anchor=(0.5, -0.02))

fig.suptitle("Parameter Efficiency: Pareto Frontier", fontsize=15, fontweight="bold", y=1.01)
plt.tight_layout(rect=[0, 0.04, 1, 0.98])

out = Path(__file__).parent / "plots" / "pareto_plot.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"Saved to {out}")
