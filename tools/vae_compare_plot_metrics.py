"""Plot recorded comparison summaries without recomputing or selecting results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {"legacy": "#497ca5", "joint": "#d36b42"}


def read(path):
    return json.loads(path.read_text())


def save(fig, path):
    fig.savefig(path, dpi=160, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.comparison_root
    output = root / "visualizations"
    output.mkdir(parents=True, exist_ok=True)
    raw = read(root / "raw_metrics/posterior_mean/summary.json")
    archived = read(root / "raw_metrics/archived_latents/summary.json")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), layout="constrained")
    ax = axes[0]
    keys = ("face_mae", "face_cd_sq", "edge_cd_sq")
    ratios = [raw["paired"][key]["joint"]["mean"] / raw["paired"][key]["legacy"]["mean"] for key in keys]
    ax.bar(np.arange(3), ratios, color=COLORS["joint"], width=.6)
    ax.axhline(1, color=COLORS["legacy"], linestyle="--", label="Legacy = 1")
    ax.set_xticks(np.arange(3), ["Face MAE", "Face CD", "Edge CD"])
    ax.set_ylabel("Joint error / Legacy error (lower is better)")
    ax.set_ylim(0, max(ratios) * 1.2)
    ax.set_title("Same posterior mean: geometry")
    for i, value in enumerate(ratios):
        ax.text(i, value + .25, f"{value:.2f}x", ha="center")
    ax.legend(frameon=False)
    ax = axes[1]
    for index, method in enumerate(("legacy", "joint")):
        values = [raw["methods"][method]["metrics"]["topology_" + key]["mean"] * 100
                  for key in ("precision", "recall", "f1")]
        positions = np.arange(3) + (index - .5) * .36
        bars = ax.bar(positions, values, width=.34, color=COLORS[method], label=method.title())
        ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=9)
    ax.set_xticks(np.arange(3), ["Precision", "Recall", "F1"])
    ax.set_ylim(0, 112)
    ax.set_ylabel("Macro average (%)")
    ax.set_title("Distinct face pairs (duplicates removed)")
    ax.legend(frameon=True, facecolor="white", framealpha=.95, loc="lower right")
    ax = axes[2]
    legacy = raw["methods"]["legacy"]["metrics"]
    joint = raw["methods"]["joint"]["metrics"]
    base = [joint["gt_boundary_count"]["mean"], legacy["unique_pair_count"]["mean"], joint["unique_pair_count"]["mean"]]
    duplicates = joint["joint_duplicate_pair_count"]["mean"]
    invalid = joint["raw_edge_count"]["mean"] - base[2] - duplicates
    bars = ax.bar(np.arange(3), base, color=["#a5b2bb", COLORS["legacy"], COLORS["joint"]], label="Distinct pairs")
    ax.bar_label(bars, fmt="%.2f", label_type="center", color="white")
    ax.bar(2, duplicates, bottom=base[2], color="#efb491", label="Duplicate queries")
    ax.bar(2, invalid, bottom=base[2] + duplicates, color="#8a3844", label="Invalid/self pair")
    ax.text(2, base[2] + duplicates / 2, f"+{duplicates:.2f}\nduplicates", ha="center", va="center")
    ax.text(2, base[2] + duplicates + invalid + 1, f"{joint['raw_edge_count']['mean']:.2f} total", ha="center")
    ax.set_xticks(np.arange(3), ["GT", "Legacy\nunique pairs", "Joint\nall queries"])
    ax.set_ylabel("Mean count per object")
    ax.set_title("More output does not mean more boundaries")
    ax.set_ylim(0, (base[2] + duplicates + invalid) * 1.2)
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    fig.suptitle(f"VAE reconstruction | {raw['intended_count']:,} fixed test objects | original split 2,424", fontsize=14)
    save(fig, output / "raw_comparison.png")

    fig, ax = plt.subplots(figsize=(7, 3.8), layout="constrained")
    for index, (data, name) in enumerate(((raw, "Posterior mean"), (archived, "Archived diffusion latent"))):
        values = [data["paired"][key]["joint"]["mean"] / data["paired"][key]["legacy"]["mean"]
                  for key in ("face_cd_sq", "edge_cd_sq")]
        bars = ax.bar(np.arange(2) + (index - .5) * .36, values, width=.34,
                      color=(COLORS["joint"] if index == 0 else "#638d70"), label=name)
        ax.bar_label(bars, fmt="%.3fx", padding=3)
    ax.axhline(1, color=COLORS["legacy"], linestyle="--", label="Legacy = 1")
    ax.set_xticks(np.arange(2), ["Face squared CD", "Edge squared CD"])
    ax.set_ylabel("Joint error / Legacy error")
    ax.set_ylim(0, 20)
    ax.set_title("Decoder replacement depends on the upstream latent")
    ax.legend(frameon=False)
    save(fig, output / "diffusion_replay.png")

    paths = [root / "post" / mode / "summary.json" for mode in ("fit_only", "optimized")]
    if all(path.is_file() for path in paths):
        summaries = [read(path) for path in paths]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
        for ax, key, title in zip(axes, ("native_valid_solid", "geometry_computable"),
                                  ("As-read valid solid", "All faces/edges sampled successfully")):
            for index, method in enumerate(("legacy", "joint")):
                values = [data["methods"][method]["rates_over_intended"][key] * 100 for data in summaries]
                bars = ax.bar(np.arange(2) + (index - .5) * .36, values, width=.34,
                              color=COLORS[method], label=method.title())
                ax.bar_label(bars, fmt="%.1f%%", padding=3)
            ax.set_xticks(np.arange(2), ["Fit only", "Geometry optimized"])
            ax.set_ylim(0, 108)
            ax.set_ylabel("Fraction of all 2,417 intended objects (%)")
            ax.set_title(title)
            ax.legend(frameon=False)
        fig.suptitle("OCC output coverage | failures and timeouts remain in the denominator")
        save(fig, output / "post_comparison.png")


if __name__ == "__main__":
    main()
