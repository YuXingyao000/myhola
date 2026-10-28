"""Audit all 3495 validation records and compare UV geometry with the fixed 64D baseline.

Only compact JSONL/protocol files are required. Gap CDFs are distributions of
per-CAD means/RMS, not pooled point quantiles. No OCC or shared-edge synthesis.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import sys

import numpy as np

SPLIT_SHA = "ae607bfe5499e919d1e68a5110c0c211daf1158aedbe57a4756eda99f6f3f0ed"
NEW_SHA = "c4f409b1a904d431cf82cf4754001ca590b59dc44abbaf987181819977a32318"
OLD_SHA = "4262f4998fe047e19c3ca21a835d74c183b7430f7a656c709ec8ed045262ada4"
THEME = Path("/mnt/d/python_curve_eval/checkpoints/latent_expansion/no_edge_feature_20261023")
BASELINE = THEME / "evaluation_20261024/runs/full/no_edge_feature_last64"
BASELINE_EXPORT = THEME / "occ_evaluation_20261024"
TIE = 1e-12
COMPARABLE = {
    "face_cd_sq": "raw_face_cd_sq", "edge_cd_sq": "raw_edge_cd_sq",
    "face_mae": "raw_face_xyz_l1", "face_pred_to_gt_sq": "raw_face_pred_to_gt_sq",
    "face_gt_to_pred_sq": "raw_face_gt_to_pred_sq",
    "edge_pred_to_gt_sq": "raw_edge_pred_to_gt_sq", "edge_gt_to_pred_sq": "raw_edge_gt_to_pred_sq",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def read_jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def stats(values, denominator):
    values = np.asarray(values, dtype=np.float64)
    require(np.isfinite(values).all(), "Nonfinite metric passed to aggregation")
    result = {"valid_cads": len(values), "planned_cads": denominator,
              "unavailable_cads": denominator - len(values)}
    result.update({key: None for key in ("mean", "median", "p90", "p95", "p99", "min", "max")})
    if len(values):
        result.update(mean=float(values.mean()), median=float(np.median(values)),
                      p90=float(np.quantile(values, .9)), p95=float(np.quantile(values, .95)),
                      p99=float(np.quantile(values, .99)), min=float(values.min()), max=float(values.max()))
    return result


def insert_unique(records, rows, source):
    for row in rows:
        identifier = row.get("id")
        require(isinstance(identifier, str), f"Missing string CAD ID: {source}")
        require(identifier not in records, f"Duplicate CAD {identifier}: {source}")
        require(isinstance(row.get("status"), str), f"Missing status: {source}/{identifier}")
        for key, value in row.items():
            require(not isinstance(value, float) or math.isfinite(value),
                    f"Nonfinite scalar {key}: {source}/{identifier}")
        records[identifier] = row


def load_shards(root, ids, split_sha, *, baseline=False, checkpoint_sha=None, checkpoint_step=30000):
    expected_sha = checkpoint_sha or (OLD_SHA if baseline else NEW_SHA)
    shard_root = root if baseline else root / "shards"
    dirs = sorted(shard_root.glob("shard_[0-9][0-9][0-9]"))
    require(bool(dirs), f"No shards in {shard_root}")
    records, audits, protocols = {}, [], []
    for index, shard in enumerate(dirs):
        protocol_path, metrics_path = shard / "protocol.json", shard / "metrics.jsonl"
        protocol, rows = read_json(protocol_path), read_jsonl(metrics_path)
        require(protocol["shard_index"] == index, f"Missing/out-of-order shard: {shard}")
        require(protocol["shard_count"] == len(dirs), f"Incomplete shard set: {shard}")
        require(protocol["split_sha256"] == split_sha, f"Split mismatch: {shard}")
        require(protocol["checkpoint_sha256"] == expected_sha, f"Wrong checkpoint: {shard}")
        require(protocol.get("checkpoint_global_step") == checkpoint_step, f"Not fixed checkpoint step: {shard}")
        require(protocol.get("smoke_limit") is None, f"Smoke run cannot be a full evaluation: {shard}")
        expected = ids[index::len(dirs)]
        require(protocol["intended_ids"] == expected, f"Unexpected shard CAD order: {shard}")
        require([row["id"] for row in rows] == expected, f"Missing/extra/reordered records: {shard}")
        counts = dict(Counter(row["status"] for row in rows))
        if baseline:
            require(protocol["status"] == "complete", f"Baseline incomplete: {shard}")
            require(protocol["status_counts"] == counts, f"Baseline counts mismatch: {shard}")
            completion_hash = None
        else:
            completion_path = shard / "completion.json"
            completion = read_json(completion_path)
            require(completion["status"] == "completed", f"Incomplete shard: {shard}")
            require(completion["intended_count"] == len(expected), f"Completion denominator mismatch: {shard}")
            require(completion["counts"] == counts, f"Completion status counts mismatch: {shard}")
            require(completion["checkpoint_sha256"] == expected_sha, f"Completion checkpoint mismatch: {shard}")
            completion_hash = sha256(completion_path)
        require(protocol.get("latent") == "posterior_mean" and protocol.get("tf32") is False,
                f"Incompatible inference protocol: {shard}")
        insert_unique(records, rows, shard)
        protocols.append(protocol)
        audits.append({"path": str(shard), "count": len(rows), "status_counts": counts,
                       "protocol_sha256": sha256(protocol_path), "metrics_sha256": sha256(metrics_path),
                       "completion_sha256": completion_hash})
    require(set(records) == set(ids), f"Full CAD coverage mismatch: {root}")
    for protocol in protocols[1:]:
        for key in ("source_hashes", "source_git_commit", "precision", "seed", "latent", "batch_size"):
            require(protocol.get(key) == protocols[0].get(key), f"Mixed shard provenance: {key}")
    return records, {"shards": audits, "checkpoint_sha256": protocols[0]["checkpoint_sha256"],
                     "checkpoint_global_step": protocols[0]["checkpoint_global_step"],
                     "source_git_commit": protocols[0].get("source_git_commit"),
                     "source_hashes": protocols[0].get("source_hashes"),
                     "status_counts": dict(Counter(row["status"] for row in records.values())),
                     "unique_cads": len(records), "all_expected_ids_present": True}


def audit_old_exports(root, ids, new_records, baseline):
    """Read existing export receipts; do not claim large raw files were rehashed."""
    validation_path = root / "metadata/export/validation.json"
    transfer_path = root / "validation/export_transfer_receipt.json"
    validation, transfer = read_json(validation_path), read_json(transfer_path)
    require(validation.get("passed") is True and transfer.get("passed") is True, "Old export receipts did not pass")
    require(validation.get("ids_match") is True and validation.get("count") == len(ids), "Old export coverage invalid")
    require(transfer.get("count") == len(ids) and transfer.get("all_raw_sha256_match") is True,
            "Old transfer receipt invalid")
    records, shard_audits = {}, []
    shards = sorted((root / "metadata/export").glob("shard_[0-9][0-9][0-9]"))
    require(len(shards) == 8, "Expected 8 old export shards")
    for index, shard in enumerate(shards):
        protocol, completion = read_json(shard / "protocol.json"), read_json(shard / "completion.json")
        rows = read_jsonl(shard / "records.jsonl")
        require(protocol["checkpoint_sha256"] == OLD_SHA and protocol["split_sha256"] == SPLIT_SHA,
                f"Wrong old export source: {shard}")
        require(protocol["selected_ids"] == ids[index::8] and [r["id"] for r in rows] == ids[index::8],
                f"Old export ID mismatch: {shard}")
        require(completion.get("source_unchanged") is True, f"Old export source changed: {shard}")
        require(completion["counts"] == dict(Counter(r["status"] for r in rows)), f"Old export counts invalid: {shard}")
        canonical_digest = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
        for row in rows:
            require(row.get("status") == "ok" and row.get("all_metrics_match") and row.get("all_counts_match"),
                    f"Old export metric receipt failed: {row['id']}")
            require(row.get("encoder_checkpoint_sha256") == OLD_SHA, f"Old encoder mismatch: {row['id']}")
            require(row.get("protocol_sha256") == canonical_digest, f"Old protocol hash mismatch: {row['id']}")
            for key, old_key in COMPARABLE.items():
                require(row["metric_comparison"][key]["previous"] == baseline[row["id"]][old_key],
                        f"Old export receipt refers to different baseline metric: {row['id']}/{key}")
        insert_unique(records, rows, shard)
        shard_audits.append({"path": str(shard), **{f"{name}_sha256": sha256(shard / name)
                             for name in ("protocol.json", "completion.json", "records.jsonl")}})
    require(set(records) == set(ids), "Old export coverage mismatch")
    matched, unavailable = [], []
    for identifier in ids:
        old_hash = records[identifier].get("input_source_sha256")
        new_hash = new_records[identifier].get("input_source_sha256")
        require(isinstance(old_hash, str) and len(old_hash) == 64, f"Missing old input hash: {identifier}")
        if new_hash is None:
            require(new_records[identifier]["status"] != "ok", f"Missing successful new input hash: {identifier}")
            unavailable.append(identifier)
        else:
            require(new_hash == old_hash, f"New/old input CAD differs: {identifier}")
            matched.append(identifier)
    return {"export_root": str(root), "export_validation": validation,
            "transfer_receipt": transfer, "validation_sha256": sha256(validation_path),
            "transfer_receipt_sha256": sha256(transfer_path), "shards": shard_audits,
            "data_hash_matched_cads": len(matched), "data_hash_unavailable_ids": unavailable,
            "data_hash_mismatch_cads": 0, "raw_files_rehashed_this_run": False,
            "note": "Existing full export/transfer receipts checked; no new scan of large raw predictions."}



def audit_run_completion(root, provenance, *, checkpoint_sha=NEW_SHA):
    verification = read_json(root / "source_verification.json")
    exits = read_json(root / "process_exit_codes.json")
    launch = read_json(root / "launch.json")
    require(verification.get("source_unchanged") is True and all(verification["source_checks"].values()),
            "Evaluation source changed during run")
    require(set(verification["source_checks"]) == set(provenance["source_hashes"]), "Incomplete source end checks")
    require(verification["checkpoint_sha256"] == checkpoint_sha, "End checkpoint hash mismatch")
    require(verification["source_git_commit"] == launch["source_git_commit"] == provenance["source_git_commit"],
            "Launch/protocol/end source commits differ")
    expected = [{"shard": index, "exit_code": 0} for index in range(len(provenance["shards"]))]
    require(exits == expected and verification["exit_codes"] == expected, "A shard did not exit successfully")
    require([item["shard"] for item in launch["commands"]] == list(range(len(expected))), "Launch shard coverage differs")
    return {"source_verification": verification, "process_exit_codes": exits,
            "receipts_sha256": {name: sha256(root / name) for name in
                                 ("source_verification.json", "process_exit_codes.json", "launch.json")}}


def macro_metrics(records):
    rows = list(records.values())
    keys = sorted({key for row in rows for key, value in row.items() if finite(value) or value is None})
    return {key: stats([row[key] for row in rows if finite(row.get(key))], len(rows)) for key in keys}


def topology(records):
    valid = [row for row in records.values() if all(finite(row.get(f"topology_{key}")) for key in ("tp", "fp", "fn"))]
    sums = {key: sum(int(row[f"topology_{key}"]) for row in valid) for key in ("tp", "fp", "fn")}
    for row in valid:
        require(all(row[f"topology_{key}"] >= 0 and int(row[f"topology_{key}"]) == row[f"topology_{key}"]
                    for key in sums), f"Invalid topology counts: {row['id']}")
    tp, fp, fn = (sums[key] for key in ("tp", "fp", "fn"))
    perfect = sum(row["topology_fp"] == 0 and row["topology_fn"] == 0 for row in valid)
    return {**sums, "micro_precision": tp / (tp + fp) if tp + fp else None,
            "micro_recall": tp / (tp + fn) if tp + fn else None,
            "micro_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
            "valid_cads": len(valid), "planned_cads": len(records), "unavailable_cads": len(records) - len(valid),
            "perfect_cads": perfect, "perfect_fraction_of_all_planned_cads": perfect / len(records),
            "note": "Micro totals use observed CADs; unavailable CADs remain in planned denominator and are not perfect."}


def paired(records, baseline):
    comparisons = {}
    for key, old_key in COMPARABLE.items():
        ids = [identifier for identifier in records if finite(records[identifier].get(key))
               and finite(baseline[identifier].get(old_key))]
        new = np.asarray([records[identifier][key] for identifier in ids], dtype=np.float64)
        old = np.asarray([baseline[identifier][old_key] for identifier in ids], dtype=np.float64)
        delta = new - old
        comparisons[key] = {"old_key": old_key, "lower_is_better": True, "tie_absolute_tolerance": TIE,
                            "paired_cads": len(ids), "unpaired_cads": len(records) - len(ids),
                            "new_same_subset": stats(new, len(records)), "old_same_subset": stats(old, len(records)),
                            "delta_new_minus_old": stats(delta, len(records)),
                            "improved_cads": int(np.sum(delta < -TIE)), "worsened_cads": int(np.sum(delta > TIE)),
                            "tied_cads": int(np.sum(np.abs(delta) <= TIE)),
                            "relative_change_of_macro_mean": float(new.mean() / old.mean() - 1)
                            if len(ids) and old.mean() != 0 else None}
    return comparisons


def pooled_gap_moments(records, prefix):
    rows = [row for row in records.values() if finite(row.get(prefix + "count"))
            and row[prefix + "count"] > 0 and finite(row.get(prefix + "mean")) and finite(row.get(prefix + "rms"))]
    count = sum(int(row[prefix + "count"]) for row in rows)
    return {"point_count": count, "contributing_cads": len(rows),
            "mean": sum(row[prefix + "mean"] * row[prefix + "count"] for row in rows) / count if count else None,
            "rms": math.sqrt(sum(row[prefix + "rms"] ** 2 * row[prefix + "count"] for row in rows) / count) if count else None,
            "quantiles": None, "note": "Exact pooled moments reconstructed from per-CAD count/mean/RMS; pooled quantiles not available."}


def write_csv_gz(path, rows):
    rows = list(rows)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("xb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
        with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
            writer = csv.DictWriter(text, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


def comparison_rows(ids, records, baseline):
    for identifier in ids:
        row = {"id": identifier}
        row.update({f"new_{key}": value for key, value in records[identifier].items()
                    if value is None or isinstance(value, (str, int, float, bool))})
        row["old_status"] = baseline[identifier]["status"]
        for key, old_key in COMPARABLE.items():
            old, new = baseline[identifier].get(old_key), records[identifier].get(key)
            row[f"old_{key}"] = old
            row[f"delta_{key}"] = new - old if finite(new) and finite(old) else None
        yield row


def draw_figures(output, records, baseline):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none", "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 150, "savefig.bbox": "tight"})

    def ecdf(axis, values, label, color, style="-"):
        x = np.sort(np.asarray(values, dtype=np.float64))
        if x.size:
            axis.step(x, np.arange(1, len(x) + 1) / len(x), where="post", label=f"{label} (n={len(x)})",
                      color=color, linestyle=style, linewidth=1.7)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1), constrained_layout=True)
    for axis, key, title in zip(axes, ("face_cd_sq", "edge_cd_sq"), ("Surface point sets", "Curve point sets (both sides)")):
        old_key = COMPARABLE[key]
        ecdf(axis, [r[old_key] for r in baseline.values() if finite(r.get(old_key))], "Previous 64D", "#576879")
        ecdf(axis, [r[key] for r in records.values() if finite(r.get(key))], "UV decoder 64D", "#c05436")
        axis.set_xscale("symlog", linthresh=1e-10)
        axis.set(xlabel="Bidirectional squared Chamfer sum\n(dataset coordinates squared)", ylabel="CAD fraction", title=title, ylim=(0, 1.01))
        axis.grid(alpha=.15)
        if axis.get_legend_handles_labels()[0]:
            axis.legend(fontsize=8, loc="lower right")
    fig.suptitle("Raw reconstruction: one equally weighted value per CAD", fontsize=12)
    for suffix in ("png", "svg"):
        fig.savefig(output / f"raw_geometry_cdf.{suffix}")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1), constrained_layout=True)
    for axis, prefix, title in zip(axes, ("active_gap_", "gt_pair_gap_"),
                                  ("Probability-selected pairs", "GT pairs (diagnostic only)")):
        for metric, label, color in (("mean", "CAD mean gap", "#c05436"), ("rms", "CAD RMS gap", "#2a7890")):
            key = prefix + metric
            ecdf(axis, [r[key] for r in records.values() if finite(r.get(key))], label, color)
        axis.set_xscale("symlog", linthresh=1e-6)
        axis.set(xlabel="Corresponding-point gap\n(dataset coordinates, not mm)", ylabel="CAD fraction", title=title, ylim=(0, 1.01))
        axis.grid(alpha=.15)
        if axis.get_legend_handles_labels()[0]:
            axis.legend(fontsize=8, loc="lower right")
    fig.suptitle("Two-side mismatch: distribution of CAD summaries, not all points", fontsize=12)
    for suffix in ("png", "svg"):
        fig.savefig(output / f"dual_side_gap_cdf.{suffix}")
    plt.close(fig)


def report(summary):
    def fmt(value):
        return "不可计算" if value is None else f"{value:.6g}"
    face_change = summary["paired_comparison"]["face_cd_sq"]["relative_change_of_macro_mean"]
    edge_comparison = summary["paired_comparison"]["edge_cd_sq"]
    edge_change = edge_comparison["relative_change_of_macro_mean"]
    result_line = (f"**面 CD² 改善 {-face_change:.1%}；曲线 CD² 为旧版 {1 + edge_change:.1f} 倍，"
                   f"{edge_comparison['worsened_cads']}/3495 件曲线退步。**"
                   if face_change is not None and edge_change is not None else
                   "主几何指标存在缺测，不能给出完整的新旧对照结论。")
    lines = ["# UV 全量训练 30k：原始几何验收", "", result_line,
             "训练 loss 收敛尚未转化成曲线重建收益；不能仅凭这一轮判定整个 UV 架构方向无效。", "",
             "固定 validation **3495 CAD**；原始 3505 件中既有 10 件合同排除，本次未追加过滤。",
             "固定最终 30,000 步权重；posterior mean、无增广、FP32、无 TF32、单 CAD 前向、概率阈值 >0.5，不去重。",
             "VAE 输入含 CAD 几何及拓扑，本报告衡量重建，不是照片生成。", "",
             f"新模型状态：`{summary['new_provenance']['status_counts']}`；全部分片完成，3495 ID 唯一覆盖。",
             f"与旧导出逐 CAD 输入 SHA 对照：{summary['baseline_export_audit']['data_hash_matched_cads']}/3495 相同，"
             f"{len(summary['baseline_export_audit']['data_hash_unavailable_ids'])} 件因失败无法核对。", "",
             "## 与旧 64D 的逐件对照", "",
             "以下均为 CAD 等权宏平均；下降为改善。CD² 是双向最近采样点平方距离均值之和，不是毫米或平均距离。", "",
             "| 指标 | 旧 64D | UV 64D | 相对变化 | 配对有效数 | 改善 / 退步 / 相同 |",
             "|---|---:|---:|---:|---:|---:|"]
    for key in ("face_cd_sq", "edge_cd_sq", "face_mae"):
        item = summary["paired_comparison"][key]
        change = item["relative_change_of_macro_mean"]
        change_text = "不可计算" if change is None else f"{change:+.2%}"
        lines.append(f"| {key} | {fmt(item['old_same_subset']['mean'])} | {fmt(item['new_same_subset']['mean'])} | "
                     f"{change_text} | {item['paired_cads']}/3495 | {item['improved_cads']} / {item['worsened_cads']} / {item['tied_cads']} |")
    lines += ["", "逐件相同的绝对容差为 1e-12。每项均在新旧同时有效的 CAD 上配对；缺失/失败留在总分母，不填零。",
              "新曲线项使用激活双侧轨迹的合并点集，旧曲线项使用全部互反半边；两侧分别统计见下表，避免合并点集掩盖较差一侧。", "",
              "| 新模型指标 | CAD 宏平均 | CAD 中位数 | CAD P90 | 有效 CAD |", "|---|---:|---:|---:|---:|"]
    for key in ("edge_side0_cd_sq", "edge_side1_cd_sq", "edge_side_mean_cd_sq", "edge_side_worst_cd_sq",
                "active_gap_mean", "active_gap_rms", "gt_pair_gap_mean", "gt_pair_gap_rms",
                "active_pair_cd_sq_mean", "gt_pair_pair_cd_sq_mean", "gt_pair_curve_xyz_mse"):
        item = summary["new_cad_macro"].get(key)
        if item:
            lines.append(f"| {key} | {fmt(item['mean'])} | {fmt(item['median'])} | {fmt(item['p90'])} | {item['valid_cads']}/3495 |")
    lines += ["", "Gap 为两侧同参数位置 XYZ 的欧氏距离；与不依赖点对应的双侧 CD² 分开。GT pair 指标只做诊断，未用于预测选择。",
              "CDF 显示每件 CAD 的 mean/RMS 分布，不是把全体边点合并后的分布。全点 mean/RMS 可由计数精确加权，"
              "汇总 JSON 单列；全点分位数未计算。", "", "## 拓扑", "",
              "| 模型 | Micro P | Micro R | Micro F1 | 完全匹配 CAD / 3495 | 有效 CAD |",
              "|---|---:|---:|---:|---:|---:|"]
    for name, key in (("旧 64D", "baseline_topology"), ("UV 64D", "new_topology")):
        item = summary[key]
        lines.append(f"| {name} | {fmt(item['micro_precision'])} | {fmt(item['micro_recall'])} | {fmt(item['micro_f1'])} | "
                     f"{item['perfect_cads']} ({item['perfect_fraction_of_all_planned_cads']:.2%}) | {item['valid_cads']} |")
    lines += ["", "## 结论边界", "",
              "- 当前输出仍是双侧曲线点集，尚未构造出唯一共享边；贴在各自连续面上是表示的构造性质，不是几何准确性证据。",
              "- **OCC 合法率未评估：原生 dual_uv_surface_v1 构造接口尚未实现。** 不能填 0、沿用旧合法率，或把两侧平均后冒充正式输出。",
              "- 新模型由旧 64D 权重初始化并增加训练，且网络、损失和边表示改变；这是系统结果对照，不是等预算单变量消融。",
              "- 原始几何结果与后续修复后的结果应分开报告；本次不调容差、不根据 test 样本修改规则。", "",
              "## 产物", "", "- `summary.json`：完整溯源、有效分母、宏统计、配对差值及拓扑统计。",
              "- `per_cad.csv.gz`、`summary.csv.gz`：逐件对照及紧凑统计。",
              "- `raw_geometry_cdf.png/svg`、`dual_side_gap_cdf.png/svg`：原始几何和两侧间距分布。",
              "- 大型 raw NPZ 留在原推理机器；本聚合仅读取分片指标和历史导出收据。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, default=BASELINE)
    parser.add_argument("--baseline-export-root", type=Path, default=BASELINE_EXPORT)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Must be a new directory")
    args = parser.parse_args()
    ids = args.split.read_text().split()
    require(len(ids) == 3495 and len(set(ids)) == 3495 and sha256(args.split) == SPLIT_SHA,
            "Expected frozen 3495-CAD validation list")
    require(not args.output.exists(), f"Output already exists; refusing overwrite: {args.output}")
    records, new_audit = load_shards(args.input_root, ids, SPLIT_SHA)
    completion_audit = audit_run_completion(args.input_root, new_audit)
    baseline, old_audit = load_shards(args.baseline_root, ids, SPLIT_SHA, baseline=True)
    export_audit = audit_old_exports(args.baseline_export_root, ids, records, baseline)
    summary = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
               "command": sys.argv, "aggregate_source_sha256": sha256(__file__), "split_sha256": SPLIT_SHA,
               "planned_cads": 3495, "original_split_cads": 3505, "existing_contract_exclusions": 10,
               "new_failure_ids": [key for key, value in records.items() if value["status"] != "ok"],
               "new_field_status_counts": {key: dict(Counter(row.get(key, "unavailable") for row in records.values()))
                                           for key in ("face_status", "edge_status", "topology_status", "gt_pair_status")},
               "run_completion_audit": completion_audit,
               "new_provenance": new_audit, "baseline_provenance": old_audit, "baseline_export_audit": export_audit,
               "new_cad_macro": macro_metrics(records), "baseline_cad_macro": macro_metrics(baseline),
               "paired_comparison": paired(records, baseline), "new_topology": topology(records),
               "baseline_topology": topology(baseline),
               "pooled_gap_moments": {prefix: pooled_gap_moments(records, prefix)
                                       for prefix in ("active_gap_", "gt_pair_gap_")},
               "occ": {"status": "not_evaluated_native_dual_uv_constructor_not_implemented", "valid_rate": None},
               "limitations": ["Dual-side point sets are not a constructed unique shared edge.",
                               "Different architectures, losses and cumulative training histories; not an equal-budget ablation.",
                               "CDFs are CAD-weighted summary distributions; pooled point quantiles not computed."]}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    write_csv_gz(args.output / "per_cad.csv.gz", comparison_rows(ids, records, baseline))
    write_csv_gz(args.output / "summary.csv.gz", ({"model": name, "metric": key, **value}
                 for name, section in (("uv64", "new_cad_macro"), ("previous64", "baseline_cad_macro"))
                 for key, value in summary[section].items()))
    draw_figures(args.output, records, baseline)
    (args.output / "REPORT.md").write_text(report(summary))
    print(json.dumps({"output": str(args.output), "new_status_counts": new_audit["status_counts"],
                      "paired_comparison": summary["paired_comparison"], "new_topology": summary["new_topology"]}, indent=2))


if __name__ == "__main__":
    main()
