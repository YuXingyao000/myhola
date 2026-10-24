"""Render frozen no-feature evaluation summaries as PNG and standalone SVG."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
DIAG = HERE.parents[1] / "diagnosis_20261021"
spec = importlib.util.spec_from_file_location("ablation_report_style", DIAG / "plot_report.py")
report_style = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report_style)
VARIANT = "no_edge_feature_last64"
BLUE, GREEN, GRAY, ORANGE = "#237f9e", "#138a85", "#a2a8b0", "#c87440"
MUTED, GRID = report_style.MUTED, report_style.GRID
PATH_LABELS = {"teacher": "Teacher：GT 边 → CNN → 曲线头", "coarse": "Coarse：intersection 输出", "refined": "Refined：最终逐边输出"}


def axis_style(ax, title, ylabel):
    ax.set_title(title, loc="left", pad=14)
    ax.set_ylabel(ylabel, labelpad=9)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, linewidth=.8)
    ax.tick_params(axis="both", length=0, pad=7)
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)


def bars(ax, values, labels, colors):
    values = np.asarray([np.nan if v is None else v for v in values], dtype=float)
    drawn = ax.bar(range(len(values)), values, color=colors, width=.58, zorder=3)
    for bar, value in zip(drawn, values):
        if np.isfinite(value):
            ax.annotate(f"{value:.3f}", (bar.get_x() + bar.get_width()/2, value),
                        xytext=(0, 6), textcoords="offset points", ha="center", fontsize=10)
    ax.set_xticks(range(len(values)), labels, fontsize=9)
    finite = values[np.isfinite(values)]
    ax.set_ylim(0, max(float(finite.max()) * 1.35, 1e-12) if len(finite) else 1)
    return drawn


def paired_note(ax, paired):
    if not paired["count"]:
        return
    change = paired.get("relative_change")
    delta = f"均值 {change:+.1%}" if change is not None else "来源均值为 0"
    ax.text(.02, .96, f"{delta} · {paired['improved']:,} 改善 / {paired['worsened']:,} 退步",
            transform=ax.transAxes, ha="left", va="top", fontsize=8.8, color=MUTED)


def geometry(summary, output):
    descriptors = (
        ("raw_face_cd_sq", "完整输出：面几何"),
        ("raw_edge_cd_sq", "完整输出：边几何"),
        ("path_teacher_matched_cd_sq", "Teacher：GT 对应逐边"),
        ("path_coarse_matched_cd_sq", "Coarse：GT 对应逐边"),
        ("path_refined_matched_cd_sq", "Refined：GT 对应逐边"),
        ("path_refined_bbox_l1", "Refined：位置 / 大小参数"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(15.8, 9.4))
    fig.subplots_adjust(left=.065, right=.98, top=.80, bottom=.17, wspace=.28, hspace=.45)
    report_style.header(fig, "01  关闭 edge feature 匹配后的固定 30k 终点",
                        "3,495 件 validation · 自身 encoder · posterior mean · FP32 · CAD 等权 · 无 OCC")
    report_style.footer(fig,
        "蓝 / 绿：同一来源权重与关闭特征匹配后新增 30,000 步；改善 / 退步按这两组逐 CAD 配对统计。\n"
        "灰色斜线：原 64 维约 50k 末权重，仅作不同训练预算背景；不能据此隔离关闭损失的因果效果。\n"
        "CD² 是归一化坐标中的双向最近点平方距离均值之和；完整点集与 GT 逐边指标不可直接比较幅度。")
    old = summary["background"]["late_last64"]
    for ax, (key, title) in zip(axes.flat, descriptors):
        scale, unit = (1e3, "bbox L1（×1e-3）") if key.endswith("_l1") else (1e5, "CD²（×1e-5）")
        axis_style(ax, title, unit)
        records = [summary["models"][v]["metrics"][key] for v in ("source32", VARIANT)]
        records.append(old["metrics"][key])
        values = [None if r["mean"] is None else r["mean"] * scale for r in records]
        drawn = bars(ax, values, ("来源 32 维", "无 feature 64\n30,000 步", f"旧 64（背景）\n{old['global_step']:,} 步"),
                     (BLUE, GREEN, GRAY))
        drawn[-1].set_hatch("///")
        drawn[-1].set_edgecolor("#747b85")
        paired_note(ax, summary["paired"]["metrics"][key])
    report_style.save(fig, output, "01_geometry_30k_with_budget_context")


def placement(summary, output):
    fig, axes = plt.subplots(3, 3, figsize=(15.8, 12.1))
    fig.subplots_adjust(left=.065, right=.98, top=.84, bottom=.13, wspace=.30, hspace=.45)
    report_style.header(fig, "02  局部形状、位置与尺度分别发生了什么",
                        "来源 32 维 → 无 feature 64 维 30k · 相同 GT 边对应 · 每个 CAD 等权 · 全部越低越好")
    for row_axes, path in zip(axes, ("teacher", "coarse", "refined")):
        for ax, (key, label) in zip(row_axes, (("local_xyz_l1", "局部形状"), ("center_l1", "位置 center"), ("scale_l1", "尺度 scale"))):
            axis_style(ax, f"{path.capitalize()} · {label}", "L1（×1e-3）")
            values = [summary["models"][v]["paths"][path]["stats"][key]["mean"] for v in ("source32", VARIANT)]
            bars(ax, [None if v is None else v * 1e3 for v in values], ("来源 32 维", "无 feature 64 · 30k"), (BLUE, GREEN))
            paired_note(ax, summary["paired"]["paths"][path]["stats"][key])
    report_style.footer(fig,
        "Teacher 直接编码 GT 边；Coarse / Refined 来自 face latent 推理。此图用 GT 对应检查几何，属于诊断。\n"
        "local 使用边局部坐标；center / scale 使用归一化物体坐标，不跨列比较幅度。完整 mean / median / P90 在 summary.json。")
    report_style.save(fig, output, "02_shape_center_scale")


def interventions(summary, output):
    cases = ("pred_all", "gt_center", "gt_scale", "gt_center_scale", "gt_local")
    labels = ("原预测", "GT 位置", "GT 尺度", "GT 位置+尺度", "GT 局部形状")
    fig, axes = plt.subplots(1, 3, figsize=(16.2, 6.2))
    fig.subplots_adjust(left=.065, right=.98, top=.75, bottom=.25, wspace=.28)
    report_style.header(fig, "03  用 GT 替换组件后的几何误差（oracle 诊断）",
                        "来源 32 维与无 feature 64 维 30k · 不用于实际推理 / 修复 / 候选选择")
    for ax, path in zip(axes, ("teacher", "coarse", "refined")):
        axis_style(ax, PATH_LABELS[path], "逐边 CD²（×1e-5）")
        for variant, label, color in (("source32", "来源 32 维", BLUE), (VARIANT, "无 feature 64 · 30k", GREEN)):
            values = [summary["models"][variant]["paths"][path]["cases"][case]["cd_sq"]["mean"] for case in cases]
            ax.plot(range(len(cases)), [np.nan if v is None else v * 1e5 for v in values],
                    marker="o", markersize=5, label=label, color=color, linewidth=1.7)
        ax.set_xticks(range(len(cases)), labels, rotation=25, ha="right", fontsize=9)
        ax.set_ylim(bottom=0)
    handles, labels_legend = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_legend, loc="upper left", bbox_to_anchor=(.06, .87), ncols=2, frameon=False)
    report_style.footer(fig,
        "每个替换仅改变指定 GT 组件，其他组件保留该模型预测。曲线表示不同干预类别，不是训练时间。\n"
        "GT 替换回答误差位置；不能单独证明因果机制。实际点集对齐及共同可对齐 CAD 子集统计另存 summary.json。")
    report_style.save(fig, output, "03_oracle_component_substitution")


def history_figure(audit, split, output):
    rows = audit["history"]["aligned"][split]["rows"]
    if not rows:
        raise ValueError(f"No aligned {split} history")
    for row in rows:
        new, old = row["d64_no_edge_feature"], row["d64_original"]
        if (new["epoch"], new["trainer/global_step"]) != (old["epoch"], old["trainer/global_step"]):
            raise ValueError("History epoch/step mismatch")
        if not new["full_epoch"] or not old["full_epoch"] or new["trainer/global_step"] > 30000:
            raise ValueError("Expected matched full epochs within the 30k budget")
    descriptors = (
        ("encoder_edge_bbox", "Teacher：位置 / 大小", 1e3, "bbox L1（×1e-3）"),
        ("edge_xyz", "最终边：全局 XYZ", 1e3, "XYZ L1（×1e-3）"),
        ("face_xyz", "最终面：全局 XYZ", 1e3, "XYZ L1（×1e-3）"),
        ("edge_feature", "原始特征差：关闭后仍记录", 1, "feature L1（未加权）"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.4))
    fig.subplots_adjust(left=.08, right=.975, top=.80, bottom=.17, wspace=.24, hspace=.43)
    title = "04  相同训练步数：原 64 与关闭 feature 的验证走势" if split == "val" else "补图  相同训练步数：训练走势"
    report_style.header(fig, title,
                        "相同 64 维初始化 · 全模型解冻 · 精确匹配 epoch / step · 完整 epoch 原始点，不平滑")
    for ax, (key, title, scale, ylabel) in zip(axes.flat, descriptors):
        axis_style(ax, title, ylabel)
        for name, label, color in (("d64_original", "原 64 · feature 权重 1", ORANGE),
                                    ("d64_no_edge_feature", "新 64 · feature 权重 0", GREEN)):
            x = [r[name]["trainer/global_step"] / 1000 for r in rows]
            y = [r[name][key] * scale for r in rows]
            ax.plot(x, y, label=label, color=color, linewidth=1.8)
            ax.scatter(x[-1], y[-1], color=color, s=22, zorder=4)
        ax.set_xlim(0, 30.2)
        ax.set_xticks((0, 10, 20, 30), ("0", "10k", "20k", "30k"))
        ax.set_xlabel("本次新增 optimizer steps")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(.07, .893), frameon=False, ncols=2)
    first, last = rows[0]["d64_original"]["trainer/global_step"], rows[-1]["d64_original"]["trainer/global_step"]
    protocol = "验证使用随机 posterior；Teacher 的直接重建不经过 posterior。" if split == "val" else "训练保留增强、dropout 与随机 posterior；最后不完整 epoch 未纳入。"
    report_style.footer(fig,
        f"{len(rows)} 个共同完整 epoch 点，step {first:,}–{last:,}。{protocol}\n"
        "新实验 weighted_edge_feature 全程为 0；原始 feature 差增大不代表几何必然变差。两组 total loss 定义不同，不直接比较。\n"
        "这些是同期日志，和固定 posterior mean 的终点评估口径不同；旧实验没有 30k 固定 mean 权重。")
    name = "04_aligned_validation_30k" if split == "val" else "05_aligned_training_30k_supplement"
    report_style.save(fig, output, name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=HERE / "summary.json")
    parser.add_argument("--output", type=Path, default=HERE / "figures")
    parser.add_argument("--audit", type=Path, default=HERE / "training_audit.json")
    parser.add_argument("--font", default=None)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text())
    if summary["checkpoint_global_step"] != 30000:
        raise ValueError("Expected the frozen 30k evaluation")
    if not summary["complete"]:
        raise ValueError("Main report plots require complete coverage; inspect failed CADs in summary first")
    args.output.mkdir(parents=True, exist_ok=True)
    stems = ("01_geometry_30k_with_budget_context", "02_shape_center_scale", "03_oracle_component_substitution",
             "04_aligned_validation_30k", "05_aligned_training_30k_supplement")
    if any((args.output / f"{name}.{ext}").exists() for name in stems for ext in ("png", "svg")):
        raise FileExistsError("Figure output exists; choose a new --output directory")
    font = report_style.style(args.font)
    geometry(summary, args.output)
    placement(summary, args.output)
    interventions(summary, args.output)
    audit = json.loads(args.audit.read_text())
    history_figure(audit, "val", args.output)
    history_figure(audit, "train", args.output)
    for path in args.output.glob("*.svg"):
        path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    print(json.dumps({"font": font, "figures": sorted(p.name for p in args.output.glob("*.png"))}, ensure_ascii=False))


if __name__ == "__main__":
    main()
