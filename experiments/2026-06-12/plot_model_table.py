#!/usr/bin/env python3
"""One-shot: 6 bar charts (3×2 grid) comparing 7 topology VAE models.
DeepSeek blue = best, gray = second best per metric."""
import matplotlib.pyplot as plt
import numpy as np

# ── Data (test split, same order as NOTE 06-10) ──────────────────────
models = [
    "KL=0.1",
    "KL=0.001",
    "+WL",
    "+N_EDGE",
    "+prefix\ncorruption",
    "+degree",
    "+prefix\ncorruption\n+degree",
]

metrics = {
    "ar_f1 ↑":             [0.7811, 0.9185, 0.9295, 0.9340, 0.9479, 0.9147, 0.9265],
    "exact_adj_acc ↑":     [0.4187, 0.7463, 0.7706, 0.7797, 0.7826, 0.7496, 0.7562],
    "ar_valid_strict ↑":   [0.9950, 0.9959, 0.9955, 0.9942, 0.9922, 0.9942, 0.9942],
    # "prior_valid_strict ↑":[0.9961, 0.9648, 0.9473, 0.8047, 0.7070, 0.7891, 0.7871],
    # "tf_f1 ↑":             [0.9143, 0.9663, 0.9693, 0.9723, 0.9618, 0.9699, 0.9565],
    # "KL":                  [0.1151, 2.1209, 1.7691, 1.7064, 1.8556, 1.8269, 1.9812],
}

# ── Colors ────────────────────────────────────────────────────────────
BLUE   = "#4C6EF5"   # DeepSeek blue (best)
GRAY   = "#ADB5BD"   # second best
LIGHT  = "#DEE2E6"   # remaining bars
BG     = "#F8F9FA"   # figure background

# ── Layout ────────────────────────────────────────────────────────────
n_metrics = len(metrics)
fig, axes = plt.subplots(1, 3, figsize=(16, 9))
fig.patch.set_facecolor(BG)
axes = axes.flatten()

x = np.arange(len(models))
bar_w = 0.55

for ax, (title, values) in zip(axes, metrics.items()):
    arr = np.array(values)
    # Determine ranks (higher=better for all 6; KL is neutral but we rank by
    # proximity to a "reasonable" value, not strictly min. For consistency:
    # ar_f1/exact/ar_valid/prior_valid/tf_f1: higher=better
    # KL: no clear "best", treat as higher-to-lower for ranking
    is_up = "↑" in title

    if is_up:
        order = np.argsort(arr)[::-1]  # descending
    else:
        order = np.argsort(arr)        # ascending

    best_idx   = order[0]
    second_idx = order[1]

    colors = [LIGHT] * len(models)
    colors[best_idx]   = BLUE
    colors[second_idx] = GRAY

    bars = ax.bar(x, arr, bar_w, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=8, linespacing=1.2)
    ax.set_title(title, fontsize=11, fontweight="bold", color="#212529", pad=8)

    # Value labels on bars
    for i, (bar, val) in enumerate(zip(bars, arr)):
        color = "#495057"
        fontweight = "bold" if i == best_idx else "normal"
        offset = 0.015 * (arr.max() - arr.min() + 1e-6)
        va = "bottom" if val > arr.mean() else "top"
        y_pos = val + offset if va == "bottom" else val - offset
        ax.text(bar.get_x() + bar.get_width() / 2, y_pos, f"{val:.4f}",
                ha="center", va=va, fontsize=7.5, color=color, fontweight=fontweight)

    # Y-axis: tight range to show differences
    if is_up:
        lo = arr.min() - 0.04
        hi = arr.max() + 0.04
    else:
        lo = arr.min() - 0.2
        hi = arr.max() + 0.2
    ax.set_ylim(lo, hi)
    ax.set_ylabel(title, fontsize=8, color="#6C757D")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CED4DA")
    ax.spines["bottom"].set_color("#CED4DA")
    ax.tick_params(colors="#6C757D", labelsize=8)
    ax.set_facecolor(BG)
    ax.grid(axis="y", color="white", linewidth=1.2)

fig.suptitle("Topology VAE — Test Split Metrics (2424 samples)", fontsize=14,
             fontweight="bold", color="#212529", y=0.985)

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=BLUE,  label="Best"),
    Patch(facecolor=GRAY,  label="2nd best"),
    Patch(facecolor=LIGHT, label="Rest"),
]
fig.legend(handles=legend_elements, loc="lower center", ncol=3,
           fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.01))

plt.tight_layout(rect=[0, 0.04, 1, 0.94])
out = "/mnt/d/python/experiments/2026-06-12/model_bars.png"
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=BG, edgecolor="none")
plt.close()
print(f"Saved → {out}")
