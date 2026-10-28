"""Audit the fixed 3495-CAD validation and compare old XYZ, UV30k, and UV v2.

Reads completed compact evaluation records only; never runs a model, changes
metrics, or infers OCC validity from a geometric threshold.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


HELPER_PATH = Path(__file__).resolve().parents[2] / "evaluation_20261027/aggregate.py"
SPEC = importlib.util.spec_from_file_location("uv_evaluation_aggregate", HELPER_PATH)
A = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(A)

V2_SHA = "da666c7db3f66e5257de8d17238d7996154f223c046ac2c0006cb807443f677a"
V2_STEP = 51460
MODELS = ("old_xyz64", "uv30k", "uv_v2_51460")
LABELS = {"old_xyz64": "Previous XYZ 64D", "uv30k": "UV 30k", "uv_v2_51460": "UV v2 51,460"}
COLORS = {"old_xyz64": "#596b7c", "uv30k": "#d08046", "uv_v2_51460": "#1b8b89"}
PRIMARY = ("face_cd_sq", "edge_cd_sq", "face_mae")
DIAGNOSTICS = (
    "edge_side0_cd_sq", "edge_side1_cd_sq", "edge_side_mean_cd_sq", "edge_side_worst_cd_sq",
    "active_gap_mean", "active_gap_rms", "active_gap_p90", "active_gap_max",
    "gt_pair_gap_mean", "gt_pair_gap_rms", "gt_pair_pair_cd_sq_mean",
    "active_pair_cd_sq_mean", "gt_pair_curve_xyz_mse", "gt_pair_edge_cd_sq_mean",
    "gt_pair_edge_side_mean_cd_sq_mean", "gt_pair_edge_side_worst_cd_sq_mean",
)
PAIRED_METRICS = (*A.COMPARABLE, *DIAGNOSTICS)
# Fixed descriptive cutoffs, selected before seeing this run's records.
GAP_THRESHOLDS = (0.001, 0.005, 0.01, 0.02, 0.05)


def canonical_baseline(records):
    return {identifier: {**row, **{key: row.get(old_key) for key, old_key in A.COMPARABLE.items()}}
            for identifier, row in records.items()}


def audit_inputs(ids, previous, current):
    matched, unavailable = 0, []
    for identifier in ids:
        hashes = [records[identifier].get("input_source_sha256") for records in (previous, current)]
        if None in hashes:
            for records, value in zip((previous, current), hashes):
                A.require(value is not None or records[identifier]["status"] != "ok",
                          f"Successful CAD without input source hash: {identifier}")
            unavailable.append(identifier)
        else:
            A.require(hashes[0] == hashes[1], f"Input CAD changed between UV runs: {identifier}")
            matched += 1
    return {"matched_cads": matched, "unavailable_ids": unavailable, "mismatch_cads": 0}


def compare_pair(ids, current, previous):
    result = {}
    for key in PAIRED_METRICS:
        selected = [identifier for identifier in ids
                    if A.finite(current[identifier].get(key)) and A.finite(previous[identifier].get(key))]
        if not selected:
            continue
        new = np.asarray([current[i][key] for i in selected], dtype=np.float64)
        old = np.asarray([previous[i][key] for i in selected], dtype=np.float64)
        delta = new - old
        result[key] = {
            "lower_is_better": True, "paired_cads": len(selected), "planned_cads": len(ids),
            "unpaired_cads": len(ids) - len(selected), "tie_absolute_tolerance": A.TIE,
            "new": A.stats(new, len(ids)), "old": A.stats(old, len(ids)),
            "delta_new_minus_old": A.stats(delta, len(ids)),
            "improved_cads": int((delta < -A.TIE).sum()),
            "worsened_cads": int((delta > A.TIE).sum()),
            "tied_cads": int((np.abs(delta) <= A.TIE).sum()),
            "ratio_of_means_new_over_old": float(new.mean() / old.mean()) if old.mean() != 0 else None,
            "relative_change_of_mean": float(new.mean() / old.mean() - 1) if old.mean() != 0 else None,
        }
    return result


def descriptive_thresholds(records):
    result = {}
    for key in ("active_gap_mean", "active_gap_max", "gt_pair_gap_mean", "gt_pair_gap_max"):
        values = np.asarray([r[key] for r in records.values() if A.finite(r.get(key))])
        result[key] = [
            {"threshold": threshold, "below_or_equal_cads": int((values <= threshold).sum()),
             "valid_cads": len(values), "planned_cads": len(records),
             "fraction_of_all_planned_cads": float((values <= threshold).sum() / len(records))}
            for threshold in GAP_THRESHOLDS
        ]
    return {"meaning": "Descriptive CAD gap fractions; not OCC tolerances or solid validity.", "metrics": result}


def per_cad_rows(ids, models):
    for identifier in ids:
        row = {"id": identifier}
        for model, records in models.items():
            row.update({f"{model}_{key}": value for key, value in records[identifier].items()
                        if value is None or isinstance(value, (str, bool, int, float))})
        for label, current, previous in (("v2_minus_uv30k", MODELS[2], MODELS[1]),
                                         ("v2_minus_old_xyz64", MODELS[2], MODELS[0])):
            for key in PAIRED_METRICS:
                new, old = models[current][identifier].get(key), models[previous][identifier].get(key)
                row[f"{label}_{key}"] = new - old if A.finite(new) and A.finite(old) else None
        yield row


def draw_figures(output, models):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 150, "savefig.bbox": "tight"})

    def ecdf(axis, name, key):
        values = np.sort([r[key] for r in models[name].values() if A.finite(r.get(key))])
        if len(values):
            axis.step(values, np.arange(1, len(values) + 1) / len(values), where="post",
                      color=COLORS[name], label=f"{LABELS[name]} (n={len(values)})", linewidth=1.8)

    def save(fig, name):
        for suffix in ("png", "pdf"):
            fig.savefig(output / f"{name}.{suffix}")
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.2), constrained_layout=True)
    panels = (("face_cd_sq", "Surface point sets", "Squared Chamfer sum"),
              ("edge_cd_sq", "Curve point sets", "Squared Chamfer sum"),
              ("active_gap_mean", "Two-side corresponding-point gap", "Mean gap within each CAD"))
    for axis, (key, title, unit) in zip(axes, panels):
        for name in MODELS:
            ecdf(axis, name, key)
        axis.set_xscale("symlog", linthresh=1e-10 if "cd" in key else 1e-6)
        axis.set(xlabel=unit + " (dataset coordinates" + (" squared)" if "cd" in key else ")"),
                 ylabel="CAD fraction", title=title, ylim=(0, 1.01))
        axis.grid(alpha=.15)
        axis.legend(fontsize=8, loc="lower right")
    fig.suptitle("Fixed validation: equally weighted CADs; raw geometry; no OCC construction", fontsize=12)
    save(fig, "three_way_geometry_cdf")

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.2), constrained_layout=True)
    panels = (("edge_side_worst_cd_sq", "Worse side per CAD"),
              ("gt_pair_edge_cd_sq_mean", "GT-pair edge geometry (diagnostic)"),
              ("gt_pair_curve_xyz_mse", "GT-pair corresponding XYZ (diagnostic)"))
    for axis, (key, title) in zip(axes, panels):
        for name in MODELS[1:]:
            ecdf(axis, name, key)
        axis.set_xscale("symlog", linthresh=1e-10)
        axis.set(xlabel=("XYZ MSE" if "mse" in key else "Squared Chamfer sum") + " (dataset coordinates squared)",
                 ylabel="CAD fraction", title=title, ylim=(0, 1.01))
        axis.grid(alpha=.15)
        axis.legend(fontsize=8, loc="lower right")
    fig.suptitle("UV diagnostics: both-side union is not a unique shared edge", fontsize=12)
    save(fig, "uv_curve_diagnostics_cdf")


def fmt(value):
    return "不可计算" if value is None else f"{value:.6g}"


def report(summary):
    n = summary["planned_cads"]
    pair = summary["paired"]["v2_vs_uv30k"]
    old_pair = summary["paired"]["v2_vs_old_xyz64"]
    conclusion = "存在缺失几何指标，需结合各项有效分母解释，不能给出完整比较结论。"
    if "edge_cd_sq" in pair and "edge_cd_sq" in old_pair:
        edge = pair["edge_cd_sq"]
        conclusion = (f"**曲线 CD² 相对 UV30k 变化 {edge['relative_change_of_mean']:+.2%}，"
                      f"{edge['improved_cads']} / {n} 件改善；但均值仍为旧 XYZ64 的 "
                      f"{old_pair['edge_cd_sq']['ratio_of_means_new_over_old']:.2f} 倍。**")
    lines = ["# UV 监督 v2 停训评估", "", conclusion, "",
             f"固定 checkpoint **51,460 步**；validation **{n} CAD**。本次没有重启训练。",
             "使用固定最终快照，不据评估结果选择 best；训练历史不等，不是等预算单变量消融。", "",
             "## 原始几何", "",
             "CD² 是双向最近采样点平方距离均值之和。每件 CAD 等权，单位为数据坐标平方。",
             "曲线为模型实际概率筛选的输出点集，GT pair 只用于后面的诊断，不用于选择输出。", "",
             "| 指标 | 模型 | 均值 | 中位数 | P90 | 有效 / 计划 |",
             "|---|---|---:|---:|---:|---:|"]
    for key in PRIMARY:
        for model in MODELS:
            item = summary["models"][model]["macro"].get(key, A.stats([], n))
            lines.append(f"| {key} | {LABELS[model]} | {fmt(item['mean'])} | {fmt(item['median'])} | "
                         f"{fmt(item['p90'])} | {item['valid_cads']} / {n} |")
    lines += ["", "## 配对变化", "", "下降为改善；逐件相同容差为 1e-12。缺失值留在计划分母，不填零。", "",
              "| 比较 | 指标 | 均值相对变化 | 改善 / 退步 / 相同 | 配对数 |",
              "|---|---|---:|---:|---:|"]
    for label, comparisons in (("v2 对 UV30k", pair), ("v2 对旧 XYZ64", old_pair)):
        for key in (*PRIMARY, "active_gap_mean", "gt_pair_curve_xyz_mse"):
            if key not in comparisons:
                continue
            item = comparisons[key]
            change = item["relative_change_of_mean"]
            change_text = "不可计算" if change is None else f"{change:+.2%}"
            lines.append(f"| {label} | {key} | {change_text} | {item['improved_cads']} / "
                         f"{item['worsened_cads']} / {item['tied_cads']} | {item['paired_cads']} / {n} |")
    lines += ["", "## 双侧与真实配对诊断", "", "| 指标 | UV30k 均值 / 中位 / P90 | v2 均值 / 中位 / P90 |",
              "|---|---:|---:|"]
    for key in DIAGNOSTICS:
        cells = []
        for model in MODELS[1:]:
            item = summary["models"][model]["macro"].get(key, A.stats([], n))
            cells.append(" / ".join(fmt(item[name]) for name in ("mean", "median", "p90")))
        lines.append(f"| {key} | {cells[0]} | {cells[1]} |")
    lines += ["", "Gap 是双侧同参数点的欧氏间距，不是 OCC 裁剪距离；GT pair 指标衡量正确配对上的曲线几何，"
              "与实际推理点集指标分开。分布图的分位数按 CAD 统计，不是全体边点池化分位数。", "",
              "## 拓扑与覆盖", "", "| 模型 | Micro P | Micro R | Micro F1 | 完全正确 CAD / 计划 | 状态 |",
              "|---|---:|---:|---:|---:|---|"]
    for model in MODELS:
        item = summary["models"][model]
        topo = item["topology"]
        lines.append(f"| {LABELS[model]} | {fmt(topo['micro_precision'])} | {fmt(topo['micro_recall'])} | "
                     f"{fmt(topo['micro_f1'])} | {topo['perfect_cads']} / {n} "
                     f"({topo['perfect_fraction_of_all_planned_cads']:.2%}) | {item['status_counts']} |")
    lines += ["", "## 解释边界", "",
              "- **OCC 合法率未评估。** 当前没有原生 dual UV 到唯一共享边的构造接口，不能以阈值比例替代合法率。",
              "- `summary.json` 的间距阈值比例仅描述误差分布；不是裁剪、缝合或 solid 的判定。",
              "- UV30k 与 v2 的 total loss 不可直接比较。训练新增监督、L1 变化和更多训练步数尚未分离。",
              "- 几何改善说明当前监督对实际输出有效；是否值得继续此架构，还需看与旧 XYZ64 的剩余差距。",
              "- 现有资料未证明曲线误差完全源于某一层、某个 loss 或 latent 容量。", "",
              "## 文件", "", "`summary.json`、`summary.csv.gz`、`per_cad.csv.gz`；"
              "`three_way_geometry_cdf.png/pdf`、`uv_curve_diagnostics_cdf.png/pdf`。", "",
              f"远程完整评估及原始预测：`{summary['artifact_locations']['v2_original_output_root']}`。",
              f"本地聚合读取紧凑副本：`{summary['artifact_locations']['v2_compact_input_root']}`；"
              "未复制或重新扫描大型原始预测文件。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--uv30k-root", type=Path, default=HELPER_PATH.parent / "full_v1")
    parser.add_argument("--baseline-root", type=Path, default=A.BASELINE)
    parser.add_argument("--baseline-export-root", type=Path, default=A.BASELINE_EXPORT)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ids = args.split.read_text().split()
    A.require(len(ids) == len(set(ids)) == 3495 and A.sha256(args.split) == A.SPLIT_SHA,
              "Expected frozen 3495-CAD validation list")
    A.require(not args.output.exists(), f"Refusing to overwrite: {args.output}")
    new, new_audit = A.load_shards(args.input_root, ids, A.SPLIT_SHA,
                                 checkpoint_sha=V2_SHA, checkpoint_step=V2_STEP)
    previous, previous_audit = A.load_shards(args.uv30k_root, ids, A.SPLIT_SHA)
    old, old_audit = A.load_shards(args.baseline_root, ids, A.SPLIT_SHA, baseline=True)
    completion = {"v2": A.audit_run_completion(args.input_root, new_audit, checkpoint_sha=V2_SHA),
                  "uv30k": A.audit_run_completion(args.uv30k_root, previous_audit)}
    new_protocol = A.read_json(args.input_root / "shards/shard_000/protocol.json")
    old_protocol = A.read_json(args.uv30k_root / "shards/shard_000/protocol.json")
    for key in ("precision", "seed", "latent", "batch_size", "threshold", "autocast", "tf32"):
        A.require(key in new_protocol and new_protocol[key] == old_protocol.get(key),
                  f"UV inference protocol changed: {key}")
    A.require(new_protocol["dataset_config"]["is_aug"] == old_protocol["dataset_config"]["is_aug"] == 0,
              "Augmented input is not comparable with the frozen validation")
    commands = A.read_json(args.input_root / "launch.json")["commands"]
    producer_roots = {item["command"][item["command"].index("--output-root") + 1] for item in commands}
    A.require(len(producer_roots) == 1, "Evaluation shards wrote to different original output roots")
    metric_key = "checkpoints/uv_intersection/evaluation_20261027/metrics.py"
    A.require(new_audit["source_hashes"].get(metric_key) == previous_audit["source_hashes"].get(metric_key)
              and metric_key in new_audit["source_hashes"], "UV metric implementation changed")
    input_audit = audit_inputs(ids, previous, new)
    exports = A.audit_old_exports(args.baseline_export_root, ids, new, old)
    models = dict(zip(MODELS, (canonical_baseline(old), previous, new)))
    provenance = dict(zip(MODELS, (old_audit, previous_audit, new_audit)))
    summary = {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(), "command": sys.argv,
        "compare_source_sha256": A.sha256(__file__), "aggregate_helper_sha256": A.sha256(HELPER_PATH),
        "split_sha256": A.SPLIT_SHA, "planned_cads": len(ids), "original_split_cads": 3505,
        "existing_contract_exclusions": 10, "completion": completion,
        "artifact_locations": {"v2_original_output_root": next(iter(producer_roots)),
                               "v2_compact_input_root": str(args.input_root.resolve()),
                               "comparison_output_root": str(args.output.resolve()),
                               "large_raw_files_copied_or_rehashed": False},
        "uv_input_hash_audit": input_audit, "baseline_export_audit": exports,
        "models": {name: {"provenance": provenance[name], "macro": A.macro_metrics(records),
                          "topology": A.topology(records),
                          "status_counts": dict(Counter(r["status"] for r in records.values())),
                          "failure_ids": [i for i in ids if records[i]["status"] != "ok"]}
                   for name, records in models.items()},
        "paired": {"v2_vs_uv30k": compare_pair(ids, new, previous),
                   "v2_vs_old_xyz64": compare_pair(ids, new, models[MODELS[0]]),
                   "uv30k_vs_old_xyz64": compare_pair(ids, previous, models[MODELS[0]])},
        "pooled_gap_moments": {name: {prefix: A.pooled_gap_moments(models[name], prefix)
                                      for prefix in ("active_gap_", "gt_pair_gap_")} for name in MODELS[1:]},
        "descriptive_gap_thresholds": {name: descriptive_thresholds(models[name]) for name in MODELS[1:]},
        "occ": {"status": "not_evaluated_native_dual_uv_constructor_not_implemented", "valid_rate": None},
        "limitations": ["Unequal cumulative budgets and changed losses; not a controlled single-factor ablation.",
                        "Raw two-side geometry does not constitute a unique shared edge or a valid solid.",
                        "CDFs use per-CAD summaries; pooled point quantiles are unavailable."],
    }
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    A.write_csv_gz(args.output / "per_cad.csv.gz", per_cad_rows(ids, models))
    A.write_csv_gz(args.output / "summary.csv.gz", ({"model": name, "metric": metric, **values}
                   for name in MODELS for metric, values in summary["models"][name]["macro"].items()))
    draw_figures(args.output, models)
    (args.output / "REPORT.md").write_text(report(summary))
    print(json.dumps({"output": str(args.output), "status_counts": {name: item["status_counts"]
                      for name, item in summary["models"].items()},
                      "v2_vs_uv30k": {key: summary["paired"]["v2_vs_uv30k"].get(key) for key in PRIMARY}}, indent=2))


if __name__ == "__main__":
    main()
