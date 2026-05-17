import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path("/mnt/d/data/new_cond_results/exp_plan_0502_depth")
ASSET_ROOT = Path("src/brepnet/experiments/condition_debug/report_assets")


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def savefig(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_overview():
    e0 = load_json(ROOT / "E0_baseline_summary.json")["summary"]
    e1 = load_json(ROOT / "E1_ksample/oracle_summary.json")["summary"]
    e2n = load_json(ROOT / "E2_condition_sensitivity/normal_summary.json")["summary"]
    e2s = load_json(ROOT / "E2_condition_sensitivity/shuffle_summary.json")["summary"]
    e2z = load_json(ROOT / "E2_condition_sensitivity/zero_summary.json")["summary"]

    labels = ["E0", "E1 best-of-8", "E2 normal", "E2 shuffle", "E2 zero"]
    face_cd = [
        e0["face_cd_mean"],
        e1["best_face_cd_mean"],
        e2n["face_cd_mean"],
        e2s["face_cd_mean"],
        e2z["face_cd_mean"],
    ]
    valid = [
        e0["valid_rate"],
        e1["oracle_any_valid_rate"],
        e2n["valid_rate"],
        e2s["valid_rate"],
        e2z["valid_rate"],
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    axes[0].bar(labels, face_cd, color=["#3b6ea8", "#4d9f6f", "#3b6ea8", "#b65d5d", "#c49a3a"])
    axes[0].set_title("Face Chamfer Mean")
    axes[0].set_ylabel("lower is better")
    axes[0].tick_params(axis="x", rotation=25)
    for idx, value in enumerate(face_cd):
        axes[0].text(idx, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8)

    axes[1].bar(labels, valid, color=["#3b6ea8", "#4d9f6f", "#3b6ea8", "#b65d5d", "#c49a3a"])
    axes[1].set_title("Valid Rate")
    axes[1].set_ylim(0, 1)
    axes[1].tick_params(axis="x", rotation=25)
    for idx, value in enumerate(valid):
        axes[1].text(idx, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    savefig(ASSET_ROOT / "metric_overview.png")


def plot_denoise():
    summary = load_json(ROOT / "E3_denoise_probe/denoise_errors.json")["summary"]["by_mode_timestep"]
    timesteps = [50, 100, 200, 500, 800]
    colors = {"normal": "#3b6ea8", "zero": "#c49a3a", "shuffle": "#b65d5d"}
    for mode in ["normal", "zero", "shuffle"]:
        y = [summary[f"{mode}_t{t}"]["x0_l1_mean"] for t in timesteps]
        plt.plot(timesteps, y, marker="o", label=mode, color=colors[mode])
    plt.title("E3 Denoise Probe: x0 L1 Mean")
    plt.xlabel("timestep")
    plt.ylabel("x0_l1 mean")
    plt.legend()
    plt.grid(alpha=0.25)
    savefig(ASSET_ROOT / "e3_denoise_x0_l1.png")


def plot_face_count():
    summary = load_json(ROOT / "E5_face_count/face_count_summary.json")["summary"]
    bins = ["0", "1", "2", "3-5", ">5", "missing"]
    values = [summary["by_abs_face_count_bin"][key]["valid_rate"] for key in bins]
    counts = [summary["by_abs_face_count_bin"][key]["count"] for key in bins]
    plt.figure(figsize=(7, 3.8))
    bars = plt.bar(bins, values, color="#6f7f9f")
    plt.title("E5 Valid Rate by Absolute Face Count Error")
    plt.xlabel("|num_recon_face - num_gt_face|")
    plt.ylabel("valid rate")
    plt.ylim(0, 1)
    for bar, value, count in zip(bars, values, counts):
        plt.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2f}\nn={count}", ha="center", va="bottom", fontsize=8)
    savefig(ASSET_ROOT / "e5_face_count_validity.png")


def plot_oracle_gain():
    gains = []
    with (ROOT / "E1_ksample/oracle_summary.csv").open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["oracle_face_cd_gain"]:
                gains.append(float(row["oracle_face_cd_gain"]))
    plt.figure(figsize=(7, 3.8))
    plt.hist(gains, bins=25, color="#4d9f6f", edgecolor="white")
    plt.axvline(0, color="#333333", linewidth=1)
    plt.title("E1 Best-of-8 Face Chamfer Gain")
    plt.xlabel("baseline_face_cd - best_face_cd")
    plt.ylabel("count")
    savefig(ASSET_ROOT / "e1_oracle_gain_hist.png")


def main():
    ASSET_ROOT.mkdir(parents=True, exist_ok=True)
    plot_overview()
    plot_denoise()
    plot_face_count()
    plot_oracle_gain()


if __name__ == "__main__":
    main()
