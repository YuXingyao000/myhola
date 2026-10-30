"""Fixed-denominator tables and figures for saved diffusion audit records."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path

import numpy as np

from src.eval.io import write_csv, write_json
from src.eval.preference_report import paired_statistics
from src.eval.metrics.conditional_diversity import summarize_fingerprints


STAGE_LABELS = {
    "pending": "尚未评估", "generation_failure": "生成失败", "raw_input_failure": "原始几何不可读",
    "budget_exhausted": "评估超时", "execution_error": "执行或读取异常",
    "incomplete_trimmed_faces": "部分面未能裁剪", "no_solid": "完整面未构成实体",
    "native_readback_or_boundary_rejected": "实体或边界检查拒绝", "valid_solid": "合法实体",
    "unresolved": "记录不足，未能定位",
}
FLAG_LABELS = {
    "invalid_connectivity": "连接索引无效", "self_connection": "存在面自连接",
    "one_direction_pair": "存在单向半边", "duplicate_directed_pair": "存在重复方向半边",
    "isolated_face": "存在孤立面", "disconnected_face_graph": "面连接图不连通",
    "reciprocal_curve_gap": "互反曲线偏差超过阈值", "edge_owner_mesh_gap": "边到所属面偏差超过阈值",
    "wire_assembly_failed": "出现边界环构造失败", "open_wire_rejected": "出现不闭合边界环",
    "wire_count_decreased_during_face_fix": "面修复减少了边界环", "native_readback_rejected": "出现 STEP 重读拒绝",
}
OBSERVATION_METRICS = {
    "silhouette_iou": "输入轮廓 IoU",
    "silhouette_boundary_distance": "输入轮廓边界距离",
    "visible_face_cd_sq": "可见表面平方 Chamfer",
    "visible_face_pred_to_gt_cd_sq": "可见表面：预测到 GT 平方距离",
    "visible_face_gt_to_pred_cd_sq": "可见表面：GT 到预测平方距离",
    "depth_mae": "共同前景深度绝对误差", "overlap_coverage": "共同前景占 GT 前景比例",
    "gt_input_silhouette_iou": "GT 重渲染的输入轮廓 IoU",
    "gt_input_silhouette_boundary_distance": "GT 重渲染的输入边界距离",
}
METHOD_LABELS = {"baseline": "原模型", "dpo_5000": "DPO 5,000 步"}
COLORS = ("#71839B", "#168578", "#BD7446", "#7C69A6")


def _finite(value) -> bool:
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def _distribution(values: list[float]) -> dict:
    return {"n": len(values), "mean": float(np.mean(values)) if values else None,
            "median": float(np.median(values)) if values else None,
            "p90": float(np.quantile(values, .9)) if values else None}


def observation_values(rows: list[dict], key: str, method: str) -> dict[str, float]:
    return {row["image_id"]: row["observation"][key] for row in rows
            if row["method"] == method and row.get("observation", {}).get("status") == "ok"
            and _finite(row["observation"].get(key))}


def summarize(rows: list[dict]) -> dict:
    identities = [(row["method"], row["image_id"], row["candidate_id"]) for row in rows]
    if not rows or len(identities) != len(set(identities)):
        raise ValueError("Audit must contain a nonempty, unique candidate population")
    image_keys = [(row["method"], row["image_id"]) for row in rows]
    if len(image_keys) != len(set(image_keys)):
        raise ValueError("This comparison report requires K=1; use conditional_diversity for K>1")
    methods = sorted({row["method"] for row in rows})
    thresholds = {row["raw_consistency"]["distance_threshold"] for row in rows
                  if "distance_threshold" in row.get("raw_consistency", {})}
    if len(thresholds) > 1:
        raise ValueError("Cannot combine raw consistency measurements with different thresholds")
    summary = {"version": "single_view_diffusion_audit_report_v1", "methods": {}, "observation_metrics": {},
               "distance_threshold": next(iter(thresholds), None)}
    for method in methods:
        records = [row for row in rows if row["method"] == method]
        flags = {}
        flag_names = set(FLAG_LABELS)
        for row in records:
            flag_names.update(row.get("trace", {}).get("observation_flags", {}))
            flag_names.update(row.get("raw_consistency", {}).get("observation_flags", {}))
        for name in sorted(flag_names):
            groups = {}
            for group in ("all", "valid", "invalid", "unknown"):
                cohort = records if group == "all" else [row for row in records if row.get("validity", "unknown") == group]
                values = []
                for row in cohort:
                    measured = {**row.get("trace", {}).get("observation_flags", {}),
                                **row.get("raw_consistency", {}).get("observation_flags", {})}
                    if isinstance(measured.get(name), bool):
                        values.append(measured[name])
                groups[group] = {"intended_count": len(cohort), "measured_count": len(values),
                                 "positive_count": sum(values),
                                 "positive_rate": sum(values) / len(values) if values else None}
            flags[name] = groups
        summary["methods"][method] = {
            "intended_count": len(records),
            "validity_counts": dict(Counter(row.get("validity", "unknown") for row in records)),
            "primary_stage_counts": dict(Counter(row.get("trace", {}).get("primary_stage", "unresolved") for row in records)),
            "audit_status_counts": dict(Counter(row["status"] for row in records)),
            "observation_status_counts": dict(Counter(row.get("observation", {}).get("status", "not_reached") for row in records)),
            "raw_status_counts": dict(Counter(row.get("raw_consistency", {}).get("status", "not_reached") for row in records)),
            "signals": flags,
            "saved_step_check_disagreement_count": sum(
                row.get("step_check", {}).get("status") == "ok"
                and row.get("validity") in ("valid", "invalid")
                and row["step_check"]["native"]["native_valid_solid"] != (row.get("validity") == "valid")
                for row in records),
            "saved_step_check_unknown_resolved_count": sum(
                row.get("step_check", {}).get("status") == "ok" and row.get("validity", "unknown") == "unknown"
                for row in records),
            "raw_distance_distributions": {
                name: {group: _distribution([
                    row["raw_consistency"][name][key]["mean"] for row in records
                    if (group == "all" or row.get("validity") == group)
                    and _finite(row.get("raw_consistency", {}).get(name, {}).get(key, {}).get("mean"))
                ]) for group in ("all", "valid", "invalid")}
                for name, key in (("owner_mesh_distance", "p95_per_edge_owner"),
                                  ("reciprocal_curve_distance", "p95_per_pair"))
            },
        }
    for key in OBSERVATION_METRICS:
        values = {method: observation_values(rows, key, method) for method in methods}
        item = {"own": {method: _distribution(list(value.values())) for method, value in values.items()}}
        if len(methods) == 2:
            paired = paired_statistics(values[methods[0]], values[methods[1]])
            item["paired_common"] = {**paired, "first_method": methods[0], "second_method": methods[1]}
        summary["observation_metrics"][key] = item
    summary["fingerprints"] = summarize_fingerprints(rows)
    summary["interpretation_limits"] = [
        "Recorded failure stages and co-occurrences are not established root causes.",
        "Visible 3D error/depth use paired GT; silhouette directly compares the input image.",
        "One candidate per condition cannot measure conditional diversity.",
        "Raw point-set fingerprints do not establish B-Rep geometric equivalence.",
        "Missing measurements remain in the intended population and are not replaced by zeros.",
    ]
    return summary


def _flat_row(row: dict) -> dict:
    result = {key: row.get(key) for key in ("method", "image_id", "candidate_id", "status", "validity", "inference_status", "evaluation_status")}
    trace = row.get("trace", {})
    result.update(primary_stage=trace.get("primary_stage"), furthest_recovery_stage=trace.get("furthest_recovery_stage"),
                  stage_evidence=json.dumps(trace.get("stage_evidence"), ensure_ascii=False))
    for key in OBSERVATION_METRICS:
        result[key] = row.get("observation", {}).get(key)
    flags = {**trace.get("observation_flags", {}), **row.get("raw_consistency", {}).get("observation_flags", {})}
    result.update({f"signal_{key}": value for key, value in flags.items()})
    for name, key in (("owner_mesh_distance", "p95_per_edge_owner"),
                      ("reciprocal_curve_distance", "p95_per_pair")):
        measurement = row.get("raw_consistency", {}).get(name, {})
        for statistic in ("count", "mean", "p95", "max"):
            result[f"{name}_{key}_{statistic}"] = measurement.get(key, {}).get(statistic)
    result.update({f"source_{key}": value["path"] for key, value in row.get("sources", {}).items()})
    result["errors"] = json.dumps(row.get("errors", {}), ensure_ascii=False)
    return result


def plot_figures(summary: dict, output: Path, font: Path | None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.ticker import FuncFormatter, MaxNLocator
    if font is not None:
        font_manager.fontManager.addfont(font)
        family = font_manager.FontProperties(fname=font).get_name()
        plt.rcParams["font.family"] = [family, "DejaVu Sans"]
    plt.rcParams.update({"font.size": 12, "svg.fonttype": "path", "axes.spines.top": False,
                         "axes.spines.right": False, "axes.spines.left": False})
    methods = list(summary["methods"])
    method_labels = summary.get("method_labels", METHOD_LABELS)
    labels = [method_labels.get(name, name) for name in methods]

    def save(fig, name):
        for suffix in ("png", "svg"):
            fig.savefig(output / f"{name}.{suffix}", dpi=240, facecolor="white")
        plt.close(fig)

    stages = [key for key in STAGE_LABELS if key != "valid_solid" and any(
        summary["methods"][method]["primary_stage_counts"].get(key, 0) for method in methods)]
    if stages:
        fig, ax = plt.subplots(figsize=(12, max(5.5, 2.5 + len(stages) * .65)))
        fig.subplots_adjust(left=.28, right=.9, top=.78, bottom=.18)
        y = np.arange(len(stages))
        width = .75 / len(methods)
        for index, method in enumerate(methods):
            values = [summary["methods"][method]["primary_stage_counts"].get(stage, 0) for stage in stages]
            bars = ax.barh(y + (index - (len(methods) - 1) / 2) * width, values, height=width,
                           color=COLORS[index % len(COLORS)], label=labels[index])
            ax.bar_label(bars, padding=4, fontsize=11)
        ax.set_yticks(y, [STAGE_LABELS[stage] for stage in stages])
        ax.invert_yaxis()
        ax.set_xlim(0, max(ax.get_xlim()[1], 1) * 1.15)
        ax.grid(axis="x", alpha=.16)
        ax.set_axisbelow(True)
        ax.set_xlabel("模型数量（件）")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend(frameon=False, loc="upper right")
        fig.suptitle("未得到合法实体：记录停在哪一步", x=.06, ha="left", fontsize=23, y=.95)
        counts = "；".join(f"{label}：{summary['methods'][method]['validity_counts'].get('valid',0)}/{summary['methods'][method]['intended_count']} 合法"
                           for method, label in zip(methods, labels))
        fig.text(.06, .87, counts, fontsize=12, color="#576777")
        fig.text(.06, .055, "每件只计一个最终阶段；这描述失败发生位置，不等于已经确定模型内部的根因。", fontsize=11)
        save(fig, "01_failure_stages")

    signals = ["one_direction_pair", "isolated_face", "disconnected_face_graph", "reciprocal_curve_gap", "edge_owner_mesh_gap"]
    fig, ax = plt.subplots(figsize=(12, 6.6))
    fig.subplots_adjust(left=.29, right=.94, top=.78, bottom=.18)
    y = np.arange(len(signals))
    count = 2 * len(methods)
    for index, (method, group) in enumerate((method, group) for method in methods for group in ("valid", "invalid")):
        values = [summary["methods"][method]["signals"][key][group]["positive_rate"] for key in signals]
        positions = y + (index - (count - 1) / 2) * .8 / count
        ax.barh(positions, [value * 100 if value is not None else np.nan for value in values], height=.8 / count,
                color=COLORS[index % len(COLORS)], label=f"{method_labels.get(method,method)} · {'合法' if group=='valid' else '不合法'}")
    ax.set_yticks(y, [FLAG_LABELS[key] for key in signals])
    ax.invert_yaxis()
    ax.set_xlim(0, 105)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}%"))
    ax.set_xlabel("各可测子集中出现该异常的比例")
    ax.grid(axis="x", alpha=.16)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(.5, 1.02), fontsize=10)
    fig.suptitle("自身几何异常与实体失败", x=.06, ha="left", fontsize=23, y=.96)
    threshold = summary.get("distance_threshold")
    fig.text(.06, .07, f"距离阈值：{threshold}（归一化坐标）；多个异常可同时出现。", fontsize=11)
    fig.text(.06, .025, "边到面的距离基于离散三角网格；相关性不是因果证明。", fontsize=11)
    save(fig, "02_internal_consistency")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.8))
    fig.subplots_adjust(left=.1, right=.96, top=.84, bottom=.14, hspace=.58, wspace=.32)
    keys = ["silhouette_iou", "visible_face_cd_sq", "silhouette_boundary_distance", "depth_mae"]
    titles = ["输入轮廓 IoU · 越高越好", "可见表面误差 · 越低越好", "输入轮廓边界距离 · 越低越好", "共同前景深度误差 · 越低越好"]
    for ax, key, title in zip(axes.flat, keys, titles):
        item = summary["observation_metrics"][key]
        pair = item.get("paired_common", {})
        values = [pair.get("baseline_mean"), pair.get("dpo_5000_mean")] if len(methods) == 2 else [item["own"][method]["mean"] for method in methods]
        scale = 100 if key == "silhouette_iou" else 1
        bars = ax.bar(labels, [value * scale if value is not None else np.nan for value in values],
                      color=[COLORS[index % len(COLORS)] for index in range(len(methods))], width=.52)
        for bar, value in zip(bars, values):
            if value is not None:
                label = f"{value*100:.2f}%" if scale == 100 else f"{value:.3g}"
                ax.annotate(label, (bar.get_x() + bar.get_width()/2, bar.get_height()), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=15)
        ax.set_title(f"{title}\n共同可测 {pair.get('n', '—')} 件", fontsize=13, pad=15)
        ax.set_ylim(0, max(ax.get_ylim()[1], .00001) * 1.25)
        if scale == 100:
            ax.set_ylim(0, 115)
            ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}%"))
        else:
            ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.2g}"))
        ax.grid(axis="y", alpha=.16)
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", length=0)
    fig.suptitle("单视图可见区域的一致性", x=.06, ha="left", fontsize=23, y=.96)
    fig.text(.06, .06, "可见表面为双向平方 Chamfer；轮廓边界距离按图像对角线归一化。", fontsize=10.5)
    fig.text(.06, .025, "轮廓直接比较输入图；可见三维与深度使用 GT。可测非法 STEP 也纳入，合法性另报。", fontsize=10.5)
    save(fig, "03_observation_consistency")


def write_report(rows: list[dict], output: Path, *, font: Path | None = None) -> dict:
    report = output / "report"
    report.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda row: (row["method"], row["image_id"], row["candidate_id"]))
    summary = summarize(rows)
    # Resolve display labels from the immutable comparison protocol; numerical
    # keys remain baseline/dpo_5000 for compatibility with paired statistics.
    audit_protocol_path = output / "protocol.json"
    if audit_protocol_path.is_file():
        audit_protocol = json.loads(audit_protocol_path.read_text())
        comparison_path = Path(audit_protocol["config"]["comparison_run"]) / "metadata/protocol.json"
        from tools.vae_compare_post import sha256_file
        if sha256_file(comparison_path) != audit_protocol["comparison_protocol_sha256"]:
            raise ValueError("Comparison protocol changed before writing the audit report")
        comparison = json.loads(comparison_path.read_text())
        if "method_labels_zh" in comparison:
            summary["method_labels"] = comparison["method_labels_zh"]
            summary["comparison"] = comparison["comparison"]
    method_labels = summary.get("method_labels", METHOD_LABELS)
    write_json(report / "summary.json", summary)
    write_csv(report / "cases.csv", [_flat_row(row) for row in rows])
    plot_figures(summary, report, font)
    lines = ["# 单视图 diffusion 失败与观测一致性", "", "所有统计以冻结清单为分母；逐件证据见 cases.csv 及 ../cases/。", "",
             "| 模型 | 总数 | 合法 | 不合法 | 未知 | 可见指标可测 |", "|---|---:|---:|---:|---:|---:|"]
    for method, item in summary["methods"].items():
        counts = item["validity_counts"]
        lines.append(f"| {method_labels.get(method,method)} | {item['intended_count']} | {counts.get('valid',0)} | {counts.get('invalid',0)} | {counts.get('unknown',0)} | {item['observation_status_counts'].get('ok',0)} |")
    if (report / "01_failure_stages.png").is_file():
        lines.extend(["", "![失败阶段](01_failure_stages.png)"])
    lines.extend(["", "![内部一致性](02_internal_consistency.png)", "",
                  "![可见一致性](03_observation_consistency.png)", "", "## 指标边界", "",
                  "- 失败阶段和异常共现不是diffusion/VAE根因结论；未记录的内部过程保留为不可判断。",
                  "- 可见三维误差仍使用GT；隐藏补全不进入该指标，但不能据此证明完整CAD正确。",
                  "- GT自身对输入轮廓的误差记录在summary中，原OCC轮廓线会带来光栅化差异。",
                  "- 完整集每图K=1，同图多样性不可计算；raw网格指纹重复不等于B-Rep几何等价。",
                  "- 可见指标按共同可测集合比较，另报各自覆盖率；失败和缺失不填零。", ""])
    (report / "README.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--font", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.audit_dir / "manifest.json").read_text())
    rows = [json.loads((args.audit_dir / "cases" / case["method"] / case["image_id"] / "audit.json").read_text())
            for case in manifest["cases"]]
    if any(row["status"] == "running" for row in rows):
        raise ValueError("Audit is still running")
    write_report(rows, args.audit_dir, font=args.font)


if __name__ == "__main__":
    main()
