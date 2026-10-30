"""Export separate Chinese presentation figures from a saved preference summary.

This reader uses the existing paired_common means, denominators, and confidence
intervals. It never evaluates geometry or changes the metric scale. Squared
Chamfer values are shown directly in scientific notation, without a x1000 factor.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np


METHODS = ("baseline", "dpo_5000")
LABELS = ("原模型", "DPO · 5,000 步")
COLORS = ("#71839B", "#168578")
INK, MUTED = "#182B3A", "#576777"
PANELS = ([.115, .29, .35, .44], [.605, .29, .35, .44])


def scientific(value: float, digits: int = 2) -> str:
    """Include the complete magnitude on each label and tick."""
    if value == 0:
        return "0"
    mantissa, exponent = f"{value:.{digits}e}".split("e")
    return rf"${mantissa.rstrip('0').rstrip('.')}\times10^{{{int(exponent)}}}$"


def _canvas(title: str, subtitle: str, notes: list[str]) -> plt.Figure:
    figure = plt.figure(figsize=(11, 6.2), facecolor="white")
    figure.text(.06, .925, title, fontsize=25, weight="bold", color=INK)
    figure.text(.06, .865, subtitle, fontsize=12, color=MUTED)
    figure.add_artist(plt.Line2D([.06, .94], [.193, .193], color="#DFE5EA", lw=1))
    for index, note in enumerate(notes):
        figure.text(.06, .14 - index * .043, note, fontsize=10.5, color=MUTED)
    return figure


def _bars(
    figure: plt.Figure, bounds: list[float], paired: dict[str, Any], *,
    title: str, percentage: bool = False,
) -> None:
    ax = figure.add_axes(bounds)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#E8EDF1", linewidth=.9)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#CFD8E1")
    values = [paired[f"{method}_mean"] for method in METHODS]
    scale = 100 if percentage else 1
    measured = [value * scale for value in values if value is not None]
    upper = 115 if percentage else max(measured, default=0) * 1.32 or 1
    ax.set_ylim(0, upper)
    ax.set_xlim(-.55, 1.55)
    ax.set_xticks([0, 1], LABELS)
    ax.tick_params(axis="both", length=0, labelcolor=MUTED, labelsize=11, pad=8)
    ax.set_title(title, loc="left", fontsize=16, pad=17, color=INK)
    if percentage:
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}%"))
    else:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: scientific(value, 1)))
    heights = [value * scale if value is not None else np.nan for value in values]
    ax.bar([0, 1], heights, width=.53, color=COLORS, zorder=3)
    for position, value in enumerate(values):
        if value is None:
            ax.text(position, .35, "不可计算", ha="center", color=MUTED,
                    transform=ax.get_xaxis_transform(), fontsize=17)
            continue
        label = f"{value * 100:.2f}%" if percentage else scientific(value)
        ax.text(position, value * scale + upper * .035, label, ha="center", va="bottom",
                fontsize=21 if percentage else 18, color=INK)


def _export(figure: plt.Figure, output: Path, stem: str) -> None:
    for suffix in ("png", "svg"):
        figure.savefig(output / f"{stem}.{suffix}", dpi=300, facecolor="white")
    plt.close(figure)


def plot_figures(summary: dict[str, Any], output: Path, *, font: Path) -> list[dict[str, Any]]:
    """Write four figures and the unscaled source values to a new directory."""
    total, validity = summary["intended_count"], summary["validity_paired"]
    if validity["n"] != total:
        raise ValueError("Validity must retain the complete intended population")
    font_manager.fontManager.addfont(font)
    family = font_manager.FontProperties(fname=font).get_name()
    plt.rcParams.update({
        "font.family": [family, "DejaVu Sans"], "font.size": 12,
        "mathtext.fontset": "dejavusans", "axes.unicode_minus": False, "svg.fonttype": "path",
    })
    output.mkdir(parents=True, exist_ok=False)
    metrics = summary["metrics"]
    counts = [summary["methods"][method]["counts"] for method in METHODS]
    transitions = summary["validity_transitions"]
    notes = [
        f"原模型：{counts[0]['validity_valid']}/{total} 件合法；DPO：{counts[1]['validity_valid']}/{total} 件合法。",
        f"{transitions['invalid_to_valid']['n']} 件从不合法变为合法；{transitions['valid_to_invalid']['n']} 件从合法变为不合法。",
    ]
    if any(count["validity_unknown"] for count in counts):
        notes.append(f"未知：原模型 {counts[0]['validity_unknown']} 件，DPO {counts[1]['validity_unknown']} 件，均保留在总数中。")
    figure = _canvas("合法实体生成率", f"固定验证集 {total:,} 件 · 每张图片生成 1 次 · 越高越好", notes)
    _bars(figure, [.12, .29, .78, .46], validity, title="", percentage=True)
    _export(figure, output, "01_validity")

    for stage, title, stem, description in (
        ("raw", "后处理前的几何误差", "02_geometry_before", "VAE 直接解码 · 双方共同可测图片"),
        ("post", "后处理后的几何误差", "03_geometry_after", "最终 STEP · 双方均合法且完成几何评估的共同图片"),
    ):
        figure = _canvas(title, f"{description} · 越低越好", [
            "双向平方 Chamfer 距离，使用归一化坐标；直接显示原始数量级。",
            "每栏以同一组图片比较两模型，样本数分别标注；差值的 95% 置信区间见附表。",
        ])
        for index, (entity, label) in enumerate((("face", "面几何误差"), ("edge", "边几何误差"))):
            paired = metrics[f"{stage}_{entity}_cd_sq"]["paired_common"]
            _bars(figure, PANELS[index], paired, title=f"{label} · 共同 {paired['n']:,} 件")
        _export(figure, output, stem)

    figure = _canvas("面匹配 F1", "衡量生成面与真实面的匹配情况 · 越高越好", [
        "F1 综合反映面匹配的精确率和召回率；图中以百分比展示。",
        "两栏样本集合可能不同，只比较每栏内的两个模型，不据此判断后处理本身的收益。",
    ])
    for index, (stage, label) in enumerate((("raw", "后处理前"), ("post", "后处理后"))):
        paired = metrics[f"{stage}_face_f1"]["paired_common"]
        _bars(figure, PANELS[index], paired, title=f"{label} · 共同 {paired['n']:,} 件", percentage=True)
    _export(figure, output, "04_face_matching")

    plotted = {"validity": validity}
    plotted.update({key: metrics[key]["paired_common"] for key in (
        "raw_face_cd_sq", "raw_edge_cd_sq", "post_face_cd_sq", "post_edge_cd_sq",
        "raw_face_f1", "post_face_f1",
    )})
    rows = []
    for key, paired in plotted.items():
        interval = paired["ci95"]
        rows.append({
            "metric": key, "n": paired["n"],
            **{method: paired[f"{method}_mean"] for method in METHODS},
            "dpo_minus_baseline": paired["dpo_minus_baseline"],
            "difference_ci95_low": interval[0] if interval is not None else None,
            "difference_ci95_high": interval[1] if interval is not None else None,
        })
    with (output / "plotted_values.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True, help="Existing preference_report summary.json")
    parser.add_argument("--output-dir", type=Path, required=True, help="New output directory; existing reports are not overwritten")
    parser.add_argument("--font", type=Path, required=True, help="CJK font file, such as Source Han Sans SC")
    args = parser.parse_args()
    summary_path = args.summary.resolve()
    source = summary_path.read_bytes()
    rows = plot_figures(json.loads(source), args.output_dir, font=args.font)
    metadata = {
        "source_summary": str(summary_path), "source_summary_sha256": hashlib.sha256(source).hexdigest(),
        "font": str(args.font.resolve()), "font_sha256": hashlib.sha256(args.font.read_bytes()).hexdigest(),
        "source_module_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "metric_count": len(rows), "aggregation": "existing paired_common fields only; no recomputation",
        "geometry_display_scale": 1, "percentage_display_scale": 100,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
