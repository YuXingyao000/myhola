"""Validated intersection-refinement/local-attention comparison, with Chinese plots.

Consumes completed fixed-protocol results; never runs models/post or selects a
checkpoint from test performance. Historical comparison output stays untouched.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

from tools.intersection_evaluate_report import (
    EXPECTED_INTENDED, EXPECTED_ORIGINAL, POST_MODES, change, load_comparison,
    number, stat_mean, stat_value,
)

METHODS = ("intersection", "local")
LABELS = {"intersection": "上一版 Intersection refinement", "local": "本版局部注意力"}
COLORS = {"intersection": "#47749b", "local": "#ce7547"}
MODE_LABELS = {"fit_only": "仅拟合／裁剪／缝合", "optimized": "加入几何优化"}


def raw_table(summary: dict, names: tuple[tuple[str, str], ...]) -> list[str]:
    lines = ["| 指标 | 共同可计算 CAD | 上一版 refinement | 本版局部注意力 | 变化 |",
             "|---|---:|---:|---:|---|"]
    for metric, label in names:
        pair = summary["paired"][metric]
        old, new = (stat_mean(pair[method]) for method in METHODS)
        higher = metric.startswith("topology_")
        lines.append(f"| {label} | {pair['common_count']} | {number(old, higher)} | {number(new, higher)} | {change(old, new, higher=higher)} |")
    return lines


def build_report(data: dict) -> str:
    raw, post, audit = (data[name] for name in ("raw", "post", "audit"))
    selected = audit["checkpoint_recommendation"]
    last = audit["last_checkpoint"]
    lines = ["# 局部注意力 Intersection 与上一版 refinement 对比", "",
             "本报告比较上一版 Intersection refinement（`intersection`）与本版局部注意力（`local`）。原 HoLa 仅用于冻结 encoder 与既有 latent 的来源核验，不是本报告的质量对照。",
             "固定测试 2417 件 CAD；原清单另有 7 件合同不支持样本，仍保留在 2424 的总体分母中。", "",
             "![原始几何、连接、solid 与归档 latent 对比](visualizations/local_comparison.png)", "",
             "## 权重与训练状态", "",
             f"本版权重只按验证集最小 loss 固定：`{Path(selected['path']).name}`；epoch {selected['epoch']}、step {selected['global_step']}、val/loss={selected['best_model_score']:.8f}。未按 test 更换权重。",
             f"最后保存权重为 epoch {last['epoch']}、step {last['global_step']}。训练曲线、回调与哈希见 [training_audit.json](training_audit.json)。",
             "验证使用 posterior 随机采样；本次重建使用既存 posterior mean，总 loss 数值不能直接对照。本版有额外训练曝光，本次不是等预算的结构消融。", "",
             "## 原始解码：相同 posterior mean", "",
             "以下按各指标共同可计算的同一批 CAD 求均值。CD 是归一化 XYZ 下两个方向均方距离之和；连接评分针对无向不同面 pair。互反半边不算多余重复边。", ""]
    lines.extend(raw_table(raw["posterior_mean"], (("face_mae", "面 XYZ MAE ↓"), ("face_cd_sq", "面平方 CD ↓"),
                                                  ("edge_cd_sq", "边平方 CD ↓"), ("topology_precision", "连接精确率 ↑"),
                                                  ("topology_recall", "连接召回率 ↑"), ("topology_f1", "连接 F1 ↑"))))
    lines.extend(["", "逐件分布、中位数/P90/P95、改善比例、极端ID及按复杂度分组见 [独立 raw 分布诊断](diagnostics/RAW_DISTRIBUTION.md)。其中去尾均值只用于敏感性分析，不替代此处全部2417件的主指标。", ""])
    lines.extend(["", "| Raw 覆盖 | 上一版 refinement | 本版局部注意力 |", "|---|---:|---:|"])
    for key, label in (("files_present", "预测存在"), ("face_geometry_computable", "面几何可计算"),
                       ("edge_geometry_computable", "边几何可计算"), ("topology_scores_computable", "连接评分可计算")):
        counts = [raw["posterior_mean"]["methods"][method]["coverage"][key]["count"] for method in METHODS]
        lines.append(f"| {label} | {counts[0]}/2417 | {counts[1]}/2417 |")
    lines.extend(["", "## 后处理有效性", "",
                  "主口径为原读取精度的 TopAbs_SOLID + BRepCheck。STEP 存在、success marker 与有效 solid 分别记录；失败、缺失与超时留在分母。容差 0.1 的兼容口径仅单列参考。", "",
                  "| 后处理 | 方法 | 主口径 solid | /2417 | /2424 | 兼容口径 solid |", "|---|---|---:|---:|---:|---:|"])
    for mode in POST_MODES:
        for method in METHODS:
            counts = post[mode]["methods"][method]["counts"]
            count = counts["native_valid_solid"]
            lines.append(f"| {MODE_LABELS[mode]} | {LABELS[method]} | {count} | {count / EXPECTED_INTENDED:.2%} | {count / EXPECTED_ORIGINAL:.2%} | {counts['compat_valid_solid']} |")
    lines.extend(["", "| 后处理 | 旧失败→新成功 | 旧成功→新失败 | 都成功 | 都失败 | 净增 solid |", "|---|---:|---:|---:|---:|---:|"])
    for mode in POST_MODES:
        counts = data["transitions"][mode]["counts"]
        gained, lost = counts["old_failure_new_success"], counts["old_success_new_failure"]
        lines.append(f"| {MODE_LABELS[mode]} | {gained} | {lost} | {counts['both_success']} | {counts['both_failure']} | {gained - lost:+d} |")
    lines.extend(["", "四类仅覆盖已测试的 2417 件，7 件合同不支持对象不伪装成双失败。逐 ID 清单见 [solid_transitions.json](solid_transitions.json)。", "",
                  "| 后处理 | 方法 | 全阶段超时 | 其中已确认 solid | 几何可计算 |", "|---|---|---:|---:|---:|"])
    for mode in POST_MODES:
        for method in METHODS:
            timeout = data["timeout_diagnostics"][mode][method]
            count = post[mode]["methods"][method]["counts"]["geometry_computable"]
            lines.append(f"| {MODE_LABELS[mode]} | {LABELS[method]} | {timeout['all_stages']} | {timeout['already_valid_solid']} | {count}/2417 |")
    lines.extend(["", "若在通过 solid 检查后才发生几何采样超时，仍计入已确认 solid；因此超时数和 solid 失败数不是互斥分类。", "",
                  "## 原生 STEP 几何", "",
                  "共同可计算集合不要求 solid 有效；与各自可计算集合分开列出。均值可能受少数极端误差影响，因此同时列中位数。", "",
                  "| 后处理 | 统计集合 | 数量（旧/新） | 指标 | 统计量 | 上一版 refinement | 本版局部注意力 |", "|---|---|---:|---|---|---:|---:|"])
    for mode in POST_MODES:
        common = post[mode]["paired_common_geometry"]
        for metric, label in (("face_cd", "面平方 CD"), ("edge_cd", "边平方 CD")):
            for field, field_label in (("mean", "均值"), ("median", "中位数")):
                values = [stat_value(common["metrics"][metric][method], field) for method in METHODS]
                lines.append(f"| {MODE_LABELS[mode]} | 共同可计算 | {common['count']}/{common['count']} | {label} | {field_label} | {number(values[0])} | {number(values[1])} |")
                stats = [post[mode]["methods"][method]["geometry_all_computable"][metric] for method in METHODS]
                values = [stat_value(stat, field) for stat in stats]
                lines.append(f"| {MODE_LABELS[mode]} | 各自可计算 | {stats[0]['count']}/{stats[1]['count']} | {label} | {field_label} | {number(values[0])} | {number(values[1])} |")
    lines.extend(["", "| 后处理 | 方法 | 对象总耗时中位数（秒） | P90（秒） |", "|---|---|---:|---:|"])
    for mode in POST_MODES:
        for method in METHODS:
            timing = post[mode]["methods"][method].get("latency_seconds", {}).get("wall_seconds")
            if timing:
                lines.append(f"| {MODE_LABELS[mode]} | {LABELS[method]} | {number(timing['median'])} | {number(timing['p90'])} |")
    lines.extend(["", "耗时包含该对象后处理、有效性和几何评估；上一版来自历史运行，不能据此作受控速度优劣结论。", ""])
    lines.extend(["", "## 固定归档 diffusion latent 重放", "",
                  "同一批既有 latent 分别经过两版 decoder；不重新采样 diffusion。没有 GT 面对应，故不报告面对应 MAE 或连接 F1；结果不代表完整照片流程已改善。", ""])
    lines.extend(raw_table(raw["archived_latents"], (("face_cd_sq", "面平方 CD ↓"), ("edge_cd_sq", "边平方 CD ↓"))))
    lines.extend(["", "## 互反半边一致性", "",
                  "两版都显式对称化 logits。单向 pair 为零由该规则保证，不代表学会环闭合。互反曲线 CD 不检查方向，且两版选中 pair 集合可能不同。", "",
                  "| 输入 | 指标 | 可计算（旧/新） | 上一版 refinement | 本版局部注意力 |", "|---|---|---:|---:|---:|"])
    for mode, label in (("posterior_mean", "Posterior mean"), ("archived_latents", "归档 latent")):
        for metric, metric_label in (("legacy_reciprocal_cd_sq_mean", "每件平均互反差距"),
                                     ("legacy_reciprocal_cd_sq_max", "每件最差互反差距的均值"),
                                     ("legacy_unilateral_pair_count", "每件单向 pair 数")):
            stats = [raw[mode]["methods"][method]["metrics"][metric] for method in METHODS]
            values = [stat_mean(stat) for stat in stats]
            lines.append(f"| {label} | {metric_label} | {stats[0]['count']}/{stats[1]['count']} | {number(values[0])} | {number(values[1])} |")
    lines.extend(["", "## 来源与限制", "",
                  "上一版 raw/STEP/终态为只读复用，经过 split、输入 raw SHA、全部后处理算法、预算、采样与状态协议审核。其耗时来自历史运行，不能与本次直接视为受控速度对比。STEP SHA 本次补记，不伪称存在历史 STEP 哈希证明。",
                  "局部注意力、额外微调及权重选择共同决定本次差异，不能只归因于单一 block；边面距离超阈值也是相关信号，不自动构成唯一根因。失败全集及救回/退步的成对诊断见 [失败统计](../failure_census_20261012/REPORT.md)。", "",
                  "- [posterior raw](raw_metrics/posterior_mean/summary.json) / [归档 latent raw](raw_metrics/archived_latents/summary.json)",
                  "- [fit-only 汇总](post/fit_only/summary.json) / [optimized 汇总](post/optimized/summary.json)",
                  "- [冻结协议](PLAN.md)", ""])
    return "\n".join(lines)


def make_charts(data: dict, directory: Path, font_path: Path) -> None:
    if not font_path.is_file():
        raise FileNotFoundError(f"Chinese chart font missing: {font_path}")
    font_manager.fontManager.addfont(str(font_path))
    family = font_manager.FontProperties(fname=str(font_path)).get_name()
    with plt.rc_context({"font.family": family, "font.size": 10, "axes.unicode_minus": False,
                         "axes.spines.top": False, "axes.spines.right": False}):
        fig, axes = plt.subplots(2, 2, figsize=(13.8, 9.5), layout="constrained")
        for ax, mode, metrics, title in (
            (axes[0, 0], "posterior_mean", (("face_mae", "面 MAE"), ("face_cd_sq", "面平方 CD"), ("edge_cd_sq", "边平方 CD")), "相同 posterior mean：原始几何"),
            (axes[1, 1], "archived_latents", (("face_cd_sq", "面平方 CD"), ("edge_cd_sq", "边平方 CD")), "固定归档 diffusion latent：仅替换 decoder"),
        ):
            ratios = []
            for metric, _ in metrics:
                pair = data["raw"][mode]["paired"][metric]
                old, new = (stat_mean(pair[method]) for method in METHODS)
                ratios.append(new / old if old is not None and old > 0 and new is not None else math.nan)
            bars = ax.bar(range(len(metrics)), ratios, color=COLORS["local"], width=.55)
            ax.axhline(1, color=COLORS["intersection"], linestyle="--", label="上一版 = 1")
            ax.bar_label(bars, labels=[f"{value:.3f} 倍" if math.isfinite(value) else "不可计算" for value in ratios], padding=4)
            ax.set_xticks(range(len(metrics)), [label for _, label in metrics])
            ax.set_ylim(0, max([1., *[x for x in ratios if math.isfinite(x)]]) * 1.28)
            ax.set_ylabel("本版 / 上一版平均误差（越低越好）")
            ax.set_title(title)
            ax.legend(frameon=False)
        ax = axes[0, 1]
        for index, method in enumerate(METHODS):
            means = [stat_mean(data["raw"]["posterior_mean"]["paired"]["topology_" + metric][method]) for metric in ("precision", "recall", "f1")]
            values = [100 * value if value is not None else math.nan for value in means]
            bars = ax.bar(np.arange(3) + (index - .5) * .37, values, width=.35, color=COLORS[method], label=LABELS[method])
            ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
        ax.set_xticks(range(3), ["精确率", "召回率", "F1"])
        ax.set_ylim(0, 116)
        ax.set_ylabel("无向面连接宏平均（%）")
        ax.set_title("原始连接预测（越高越好）")
        ax.legend(frameon=False, fontsize=9, loc="upper center")
        ax = axes[1, 0]
        for index, method in enumerate(METHODS):
            counts = [data["post"][mode]["methods"][method]["counts"]["native_valid_solid"] for mode in POST_MODES]
            values = np.asarray(counts) / EXPECTED_INTENDED * 100
            bars = ax.bar(np.arange(2) + (index - .5) * .37, values, width=.35, color=COLORS[method], label=LABELS[method])
            ax.bar_label(bars, labels=[f"{value:.2f}%\n{count}/2417" for value, count in zip(values, counts)], padding=4, fontsize=9)
        ax.set_xticks(range(2), ["仅拟合／裁剪／缝合", "加入几何优化"])
        ax.set_ylim(0, 117)
        ax.set_ylabel("原读取精度合法 solid / 2417（%）")
        ax.set_title("OCC 有效性：失败保留在分母")
        ax.legend(frameon=False, loc="upper center", fontsize=9)
        fig.suptitle("上一版 Intersection refinement → 本版局部注意力 | 固定 2417 件 CAD", fontsize=15)
        directory.mkdir(parents=True, exist_ok=True)
        for suffix in ("png", "svg"):
            fig.savefig(directory / f"local_comparison.{suffix}", dpi=180, facecolor="white", bbox_inches="tight")
        plt.close(fig)


def generate_report(root: Path, font_path: Path) -> dict:
    root = root.resolve()
    paths = {"report": root / "REPORT.md", "transitions": root / "solid_transitions.json",
             "png": root / "visualizations/local_comparison.png", "svg": root / "visualizations/local_comparison.svg"}
    if any(path.exists() for path in paths.values()):
        raise FileExistsError("Refusing to overwrite existing completed report artifacts")
    data = load_comparison(root, methods=METHODS)
    report = build_report(data)
    with tempfile.TemporaryDirectory(prefix="local-report-", dir=root) as temporary:
        temporary = Path(temporary)
        make_charts(data, temporary, font_path)
        paths["png"].parent.mkdir(parents=True, exist_ok=True)
        for suffix in ("png", "svg"):
            with paths[suffix].open("xb") as handle:
                handle.write((temporary / f"local_comparison.{suffix}").read_bytes())
        paths["transitions"].write_text(json.dumps(data["transitions"], ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        paths["report"].write_text(report)
    return {name: str(path) for name, path in paths.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-root", type=Path, required=True)
    parser.add_argument("--font-path", type=Path, default=Path("/root/.cache/hola-browser-libs/fonts/NotoSansCJKsc-Regular.otf"))
    args = parser.parse_args()
    print(json.dumps(generate_report(args.comparison_root, args.font_path), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
