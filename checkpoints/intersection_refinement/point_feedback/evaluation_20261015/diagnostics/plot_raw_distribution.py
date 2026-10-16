"""Plot already-audited raw summary statistics; no geometry recomputation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FONT = Path("/root/.cache/hola-browser-libs/fonts/NotoSansCJKsc-Regular.otf")
BASELINE, CANDIDATE = "conditioned", "point_feedback"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative_change(record, statistic):
    old, new = record[BASELINE][statistic], record[CANDIDATE][statistic]
    if old is None or old <= 0 or new is None:
        raise ValueError(f"Cannot form plotted ratio for {statistic}")
    return 100 * (new / old - 1)


def main():
    source_paths = [ROOT / "diagnostics" / name for name in ("raw_distribution.json", "adjacency_subset.json")]
    source_hashes = {str(path): digest(path) for path in source_paths}
    raw, adjacency = (json.loads(path.read_text()) for path in source_paths)
    if raw["methods"] != [BASELINE, CANDIDATE] or adjacency["method_names"] != [BASELINE, CANDIDATE]:
        raise ValueError("Unexpected figure method identities")
    posterior = raw["modes"]["posterior_mean"]
    exact = adjacency["cohorts"]["both_adjacency_exact"]
    metrics = ("face_mae", "face_cd_sq", "edge_cd_sq")
    metric_labels = ("面 XYZ MAE", "面平方 CD", "边平方 CD")
    fields = (("mean", "均值", "#397992"), ("median", "中位数", "#d48249"), ("p95", "P95", "#876caf"))
    if posterior["intended_count"] != 2417 or adjacency["intended_count"] != 2417:
        raise ValueError("Figure requires the fixed 2417-ID protocol")
    for metric in metrics:
        if posterior["metrics"][metric]["common_count"] != 2417:
            raise ValueError("This figure requires complete paired posterior geometry")
    if exact["metrics"]["edge_cd_sq"]["common_count"] != exact["count"]:
        raise ValueError("Adjacency subset geometry is incomplete")
    paths = {suffix: ROOT / "visualizations" / f"raw_geometry_distribution.{suffix}" for suffix in ("png", "svg")}
    audit_path = ROOT / "diagnostics/raw_distribution_figure.json"
    if audit_path.exists() or any(path.exists() for path in paths.values()):
        raise FileExistsError("Refusing to overwrite completed raw distribution figure")
    font_manager.fontManager.addfont(str(FONT))
    family = font_manager.FontProperties(fname=str(FONT)).get_name()
    audit = {"source_sha256": source_hashes, "script_sha256": digest(Path(__file__)),
             "methods": [BASELINE, CANDIDATE], "posterior_intended_count": 2417,
             "both_adjacency_exact_count": exact["count"], "plotted_values": {}}
    with plt.rc_context({"font.family": family, "font.size": 11, "axes.unicode_minus": False,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.hashsalt": "hola-point-feedback-20261015"}):
        fig, axes = plt.subplots(1, 3, figsize=(18.4, 6.2))
        fig.subplots_adjust(left=.055, right=.986, top=.75, bottom=.28, wspace=.30)
        fig.suptitle("逐点几何反馈与曲线条件调制：Raw 几何配对分布", fontsize=20, y=.97)
        fig.text(.5, .875, "相同 posterior mean · 固定 2417 件 CAD · 误差相对变化越低越好", ha="center", fontsize=12, color="#44515e")
        ax = axes[0]
        x = np.arange(3)
        audit["plotted_values"]["posterior_relative_percent"] = {}
        for index, (field, label, color) in enumerate(fields):
            values = [relative_change(posterior["metrics"][metric], field) for metric in metrics]
            audit["plotted_values"]["posterior_relative_percent"][field] = dict(zip(metrics, values))
            bars = ax.bar(x + (index - 1) * .25, values, width=.23, color=color, label=label)
            ax.bar_label(bars, labels=[f"{value:+.1f}%" if abs(value) >= .1 else f"{value:+.3f}%" for value in values], padding=4, fontsize=9)
        ax.axhline(0, color="#6e7781", linewidth=1)
        ax.set_xticks(x, metric_labels)
        ax.set_ylim(-29, 20)
        ax.set_ylabel("相对上一版曲线条件调制 的误差变化（%）")
        ax.set_title("A  分布统计量：全体 2417 件", pad=17)
        ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(.5, -.23), fontsize=10)
        ax.grid(axis="y", alpha=.12)
        ax.set_axisbelow(True)

        ax = axes[1]
        records = [posterior["metrics"][metric] for metric in metrics]
        values = [100 * record["win_fraction"] for record in records]
        audit["plotted_values"]["posterior_improved_count"] = {metric: record["win_count"] for metric, record in zip(metrics, records)}
        bars = ax.barh(np.arange(3), values, height=.56, color=("#629b95", "#629b95", "#d48249"))
        ax.bar_label(bars, labels=[f"{value:.2f}%\n{record['win_count']} / 2417" for value, record in zip(values, records)],
                     padding=7, fontsize=11, bbox={"facecolor": "white", "edgecolor": "none", "pad": .4})
        ax.set_yticks(np.arange(3), metric_labels)
        ax.invert_yaxis()
        ax.set_xlim(0, 102)
        ax.axvline(50, color="#77848f", linestyle="--", linewidth=1)
        ax.set_xticks((0, 25, 50, 75, 100))
        ax.set_xlabel("同一 CAD 的误差下降比例（%）")
        ax.set_title("B  逐件改善：全体 2417 件", pad=17)
        ax.text(.5, -.20, "改善阈值：误差下降 > 1e-12", transform=ax.transAxes, ha="center", fontsize=10, color="#596673")
        ax.grid(axis="x", alpha=.12)
        ax.set_axisbelow(True)

        ax = axes[2]
        audit["plotted_values"]["edge_cohort_relative_percent"] = {}
        for index, (name, record, label, color) in enumerate((
            ("all", posterior["metrics"]["edge_cd_sq"], "全体 2417 件", "#397992"),
            ("both_adjacency_exact", exact["metrics"]["edge_cd_sq"], f"双方邻接全对 {exact['count']} 件", "#d48249"),
        )):
            values = [relative_change(record, field) for field, _, _ in fields]
            audit["plotted_values"]["edge_cohort_relative_percent"][name] = dict(zip((f[0] for f in fields), values))
            bars = ax.bar(x + (index - .5) * .34, values, width=.31, color=color, label=label)
            ax.bar_label(bars, labels=[f"{value:+.1f}%" if abs(value) >= .1 else f"{value:+.3f}%" for value in values], padding=4, fontsize=9)
        ax.axhline(0, color="#6e7781", linewidth=1)
        ax.set_xticks(x, [label for _, label, _ in fields])
        ax.set_ylim(-12, 19)
        ax.set_ylabel("边平方 CD 的相对变化（%）")
        ax.set_title("C  邻接均正确时的边几何", pad=17)
        ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(.5, -.33), fontsize=10)
        ax.grid(axis="y", alpha=.12)
        ax.set_axisbelow(True)
        fig.text(.055, .034,
                 "变化 =（本版统计量 / 上一版统计量 − 1）× 100%；中位数、P95 是各方法的分布统计量。\n"
                 "双方邻接全对仅指无向面 pair 与 GT 一致；不保证 coedge、wire 或 solid 正确，也不构成因果消融。",
                 fontsize=10, color="#596673", linespacing=1.6)
        for path, expected in source_hashes.items():
            if digest(Path(path)) != expected:
                raise ValueError("Source changed while plotting")
        paths["png"].parent.mkdir(parents=True, exist_ok=True)
        for suffix, path in paths.items():
            fig.savefig(path, dpi=180, facecolor="white", metadata={"Creator": "HoLa raw diagnostic"})
        plt.close(fig)
    audit["artifact_sha256"] = {str(path): digest(path) for path in paths.values()}
    with audit_path.open("x") as handle:
        json.dump(audit, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"files": {key: str(path) for key, path in paths.items()}, "audit": str(audit_path)}, indent=2))


if __name__ == "__main__":
    main()
