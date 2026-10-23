"""Export five presentation figures from the completed paired evaluation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter


COLORS = ("#71839B", "#168578")
INK = "#182B3A"
MUTED = "#576777"
METHODS = ("原模型", "DPO 微调 5,000 步")


def scientific(value: float, digits: int = 2) -> str:
    """Write the complete magnitude instead of an axis-wide scale factor."""
    if value == 0:
        return "0"
    mantissa, exponent = f"{value:.{digits}e}".split("e")
    mantissa = mantissa.rstrip("0").rstrip(".") if "." in mantissa else mantissa
    return rf"${mantissa}\times10^{{{int(exponent)}}}$"


def canvas(title: str, subtitle: str, notes: list[str]) -> plt.Figure:
    fig = plt.figure(figsize=(11, 6.2), facecolor="white")
    fig.text(.06, .925, title, fontsize=25, weight="bold", color=INK)
    fig.text(.06, .865, subtitle, fontsize=12, color=MUTED)
    fig.add_artist(plt.Line2D([.06, .94], [.193, .193], color="#DFE5EA", lw=1))
    for index, note in enumerate(notes):
        fig.text(.06, .14 - index * .043, note, fontsize=10.5, color=MUTED)
    return fig


def bars(fig: plt.Figure, bounds: list[float], values: list[float], *,
         title: str, ticks: list[float], upper: float,
         scientific_values: bool = False, percent: bool = False,
         decimals: int = 2) -> None:
    ax = fig.add_axes(bounds)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#E8EDF1", linewidth=.9)
    for name in ("top", "right", "left"):
        ax.spines[name].set_visible(False)
    ax.spines["bottom"].set_color("#CFD8E1")
    ax.set_ylim(0, upper)
    ax.set_xlim(-.55, 1.55)
    ax.set_yticks(ticks)
    ax.tick_params(axis="both", length=0, labelcolor=MUTED, labelsize=11, pad=8)
    ax.set_xticks([0, 1], METHODS)
    ax.set_title(title, loc="left", fontsize=16, pad=17, color=INK)
    if scientific_values:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: scientific(value, 1)))
    elif percent:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}%"))
    else:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))

    ax.bar([0, 1], values, width=.53, color=COLORS, zorder=3)
    for x, value in enumerate(values):
        label = scientific(value) if scientific_values else f"{value:.{decimals}f}"
        if percent:
            label += "%"
        ax.text(x, value + upper * .035, label, ha="center", va="bottom",
                fontsize=18 if scientific_values else 21, color=INK)


def export(fig: plt.Figure, output: Path, stem: str) -> None:
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(output / f"{stem}.{suffix}", dpi=300, facecolor="white")
    plt.close(fig)


def plot(summary: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    metrics = summary["metrics"]
    total = summary["intended_count"]

    def paired(key: str) -> dict:
        return metrics[key]["paired_common"]

    def means(key: str, scale: float = 1) -> list[float]:
        row = paired(key)
        return [row[f"{method}_mean"] * scale for method in ("baseline", "dpo_5000")]

    # Validity always retains the complete intended denominator.
    valid = summary["validity_paired"]
    counts = [summary["methods"][method]["counts"]["validity_valid"]
              for method in ("baseline", "dpo_5000")]
    transitions = summary["validity_transitions"]
    rescued = transitions["invalid_to_valid"]["n"]
    regressed = transitions["valid_to_invalid"]["n"]
    fig = canvas("合法实体生成率", f"固定验证集 {total} 件 · 每张图片生成 1 次 · 越高越好", [
        f"原模型：{counts[0]}/{total} 件合法；DPO：{counts[1]}/{total} 件合法。",
        f"{rescued} 件从不合法变为合法；{regressed} 件从合法变为不合法。两模型使用相同图片、随机种子和后处理。",
    ])
    bars(fig, [.12, .29, .78, .46],
         [valid[f"{method}_mean"] * 100 for method in ("baseline", "dpo_5000")],
         title="", ticks=[0, 25, 50, 75, 100], upper=112, percent=True, decimals=0)
    export(fig, output, "01_validity")

    panels = ([.115, .29, .35, .44], [.605, .29, .35, .44])
    for stage, title, stem, ticks, upper in (
        ("raw", "后处理前的几何误差", "02_geometry_before",
         ([0, .002, .004, .006], [0, .005, .010, .015]), (.0078, .020)),
        ("post", "后处理后的几何误差", "03_geometry_after",
         ([0, .002, .004, .006], [0, .003, .006, .009]), (.0082, .012)),
    ):
        n = paired(f"{stage}_face_cd_sq")["n"]
        subset = f"验证集 {n} 件 · VAE 直接解码" if stage == "raw" else f"双方均合法且可完整评估的同一组 {n} 件 · 最终 STEP"
        fig = canvas(title, f"{subset} · 越低越好", [
            "误差为双向平方 Chamfer 距离，使用归一化坐标；图中直接显示原始数值。",
            "展示各模型的样本均值；两项误差差值的 95% 置信区间均跨 0。",
        ])
        for index, (entity, label) in enumerate((("face", "面几何误差"), ("edge", "边几何误差"))):
            bars(fig, panels[index], means(f"{stage}_{entity}_cd_sq"), title=label,
                 ticks=ticks[index], upper=upper[index], scientific_values=True)
        export(fig, output, stem)

    fig = canvas("面匹配 F1", "衡量生成面与真实面的匹配情况 · 越高越好", [
        "F1 综合反映面匹配的精确率和召回率。",
        "两栏样本集合不同，只比较各栏内的两个模型，不用于判断后处理本身的收益。",
    ])
    for index, (stage, label) in enumerate((("raw", "后处理前"), ("post", "后处理后"))):
        n = paired(f"{stage}_face_f1")["n"]
        subset = f"共同 {n} 件" if stage == "post" else f"{n} 件"
        bars(fig, panels[index], means(f"{stage}_face_f1", 100), title=f"{label} · {subset}",
             ticks=[0, 25, 50, 75, 100], upper=115, percent=True)
    export(fig, output, "04_face_matching")

    n = paired("raw_face_count_abs_error")["n"]
    fig = canvas("面数量误差", f"平均每件模型相差多少个面 · 验证集 {n} 件 · 越低越好", [
        "每件先取生成面数与真实面数之差的绝对值，再对所有样本取平均。",
        "使用 VAE 直接解码的面数量；纵轴单位为“个面”。",
    ])
    bars(fig, [.12, .29, .78, .46], means("raw_face_count_abs_error"), title="",
         ticks=[0, .5, 1, 1.5], upper=1.6)
    export(fig, output, "05_face_count")

    # Unscaled means and denominators make every displayed number traceable.
    rows = [{"metric": "validity", "n": total,
             **{method: valid[f"{method}_mean"] for method in ("baseline", "dpo_5000")}}]
    rows.extend({"metric": key, "n": row["paired_common"]["n"],
                 **{method: row["paired_common"][f"{method}_mean"]
                    for method in ("baseline", "dpo_5000")}}
                for key, row in metrics.items())
    with (output / "plotted_values.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["metric", "n", "baseline", "dpo_5000"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=Path(__file__).with_name("summary.json"))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("weekly"))
    parser.add_argument("--font", type=Path, required=True, help="Source Han Sans SC OTF path")
    args = parser.parse_args()
    font_manager.fontManager.addfont(args.font)
    family = font_manager.FontProperties(fname=args.font).get_name()
    plt.rcParams.update({
        "font.family": [family, "DejaVu Sans"], "font.size": 12,
        "mathtext.fontset": "dejavusans", "axes.unicode_minus": False,
        "svg.fonttype": "path", "pdf.fonttype": 42,
    })
    plot(json.loads(args.summary.read_text()), args.output)


if __name__ == "__main__":
    main()
