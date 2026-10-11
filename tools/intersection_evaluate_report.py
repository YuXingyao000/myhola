"""Report completed legacy/intersection comparisons without choosing checkpoints.

Frozen execution contract: evaluation_20261011/PLAN.md. All input modes must
cover the same 2,417 accepted IDs; the original 2,424 denominator is retained.
This tool never recomputes geometry, retries failures, or replaces reports.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METHODS = ("legacy", "intersection")
POST_MODES = ("fit_only", "optimized")
MODE_NAMES = {"fit_only": "仅拟合/裁剪/缝合", "optimized": "加入几何优化"}
COLORS = {"legacy": "#47749b", "intersection": "#d07a39"}
TERMINAL = {
    "ok", "missing_raw", "invalid_raw", "post_error", "no_step", "step_read_error",
    "geometry_error", "timeout", "crash", "interrupted",
}
EXPECTED_INTENDED = 2417
EXPECTED_ORIGINAL = 2424


def read_json(path: Path) -> dict:
    def reject_constant(value):
        raise ValueError(f"Nonfinite JSON constant {value} in {path}")

    with path.open(encoding="utf-8") as handle:
        return json.load(handle, parse_constant=reject_constant)


def read_rows(path: Path, id_key: str) -> dict[str, dict[str, dict]]:
    result = {method: {} for method in METHODS}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            method, sample_id = row.get("method"), row.get(id_key)
            if method not in result or not sample_id:
                raise ValueError(f"Unexpected method or missing ID in {path}")
            if sample_id in result[method]:
                raise ValueError(f"Duplicate {method}/{sample_id} in {path}")
            result[method][sample_id] = row
    for method in METHODS:
        if len(result[method]) != EXPECTED_INTENDED:
            raise ValueError(f"Incomplete {path}: {method} has {len(result[method])}/{EXPECTED_INTENDED} rows")
    if set(result["legacy"]) != set(result["intersection"]):
        raise ValueError(f"Method ID sets disagree in {path}")
    return result


def csv_bool(value: str | None) -> bool:
    if value in ("True", "true", "1"):
        return True
    if value in (None, "", "False", "false", "0"):
        return False
    raise ValueError(f"Invalid CSV boolean: {value!r}")


def stat_value(statistic: dict, field: str) -> float | None:
    count, value = statistic["count"], statistic[field]
    if count < 0:
        raise ValueError("Metric count cannot be negative")
    if count == 0:
        if value is not None:
            raise ValueError(f"An empty metric must have {field}=null")
        return None
    if value is None or not math.isfinite(value) or value < 0:
        raise ValueError(f"A computable geometry/topology metric requires a finite nonnegative {field}")
    return float(value)


def stat_mean(statistic: dict) -> float | None:
    return stat_value(statistic, "mean")


def validate_statistics(pair: dict, count: int, context: str) -> None:
    if not 0 <= count <= EXPECTED_INTENDED:
        raise ValueError(f"Invalid common metric count for {context}")
    for method in METHODS:
        if pair[method]["count"] != count:
            raise ValueError(f"Paired metric counts disagree for {context}/{method}")
        stat_mean(pair[method])


def load_numerical_audit(root: Path) -> dict | None:
    """Optional provenance, never a replacement for exact decoder-input hashes."""
    path = root / "numerical_audit.json"
    if not path.exists():
        return None
    audit = read_json(path)
    sample_ids = audit["sample_ids"]
    repeats = audit["repeats_per_method_per_id"]
    if not sample_ids or len(set(sample_ids)) != len(sample_ids) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError("Invalid numerical audit sample IDs or repeat count")
    summary, tolerance = audit["summary"], audit["numerical_audit_tolerance"]
    for name in ("max_legacy_repeat", "max_legacy_vs_cache", "max_legacy_vs_new"):
        value = summary[name]
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid numerical audit residual: {name}")
    for name in ("previous_atol", "atol", "rtol"):
        value = tolerance[name]
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid numerical audit tolerance: {name}")
    attempts = len(sample_ids) * repeats
    for name in ("legacy_runs_pass_atol1e6", "new_runs_pass_atol1e6", "legacy_runs_pass_atol1e5", "new_runs_pass_atol1e5"):
        count = summary[name]
        if not isinstance(count, int) or not 0 <= count <= attempts:
            raise ValueError(f"Invalid numerical audit pass count: {name}")
    validation_path = root / "validation_summary.json"
    first_attempt = read_json(validation_path).get("first_full_inference") if validation_path.exists() else None
    return {
        "audit": audit,
        "first_attempt": first_attempt,
        "archive_present": (root / "metadata/initial_mean_audit_failed").is_dir(),
    }


def numerical_audit_paragraphs(record: dict | None) -> list[str]:
    if record is None:
        return []
    audit = record["audit"]
    summary, tolerance = audit["summary"], audit["numerical_audit_tolerance"]
    attempts = len(audit["sample_ids"]) * audit["repeats_per_method_per_id"]
    first = record["first_attempt"]
    initial = ""
    if first is not None:
        failures = sum(code != 0 for code in first["exit_codes"])
        initial = f"首轮 mean 复核有 {failures}/{len(first['exit_codes'])} 个分片被拦下。"
    cpu_checks = audit["cpu_algebra_check"]
    cpu_equal = sum(item["exact_equal"] is True and item["max_abs"] == 0 for item in cpu_checks)
    paragraphs = [
        "## 数值复核与相同输入", "",
        f"{initial}对固定 {len(audit['sample_ids'])} 件 CAD、每种实现各重复 {audit['repeats_per_method_per_id']} 次 GPU FP32 编码后，"
        f"原模型自身重复的最大差为 {summary['max_legacy_repeat']:.3g}，原模型与旧缓存最大差为 {summary['max_legacy_vs_cache']:.3g}，"
        f"新旧 encoder 实现间最大差为 {summary['max_legacy_vs_new']:.3g}。"
        f"CPU 逐 bit 一致为 {cpu_equal}/{len(cpu_checks)} 件，encoder/posterior 共有 {audit['encoder_equal_tensor_count']} 个张量完全相同。",
        f"原绝对容差 {tolerance['previous_atol']:.0e} 下，原/新实现分别只有 {summary['legacy_runs_pass_atol1e6']}/{attempts}、"
        f"{summary['new_runs_pass_atol1e6']}/{attempts} 次通过；审计绝对容差改为 {tolerance['atol']:.0e}、相对容差保持 {tolerance['rtol']:.0e} 后，"
        f"分别为 {summary['legacy_runs_pass_atol1e5']}/{attempts}、{summary['new_runs_pass_atol1e5']}/{attempts}。"
        "这里调整的是重新编码结果的数值复核容差；两种 decoder 实际读取的缓存 latent 数组和 SHA 校验始终严格相同，模型、预测阈值和指标均未改变。", "",
    ]
    if record["archive_present"]:
        paragraphs.extend([
            "首轮部分输出、协议与日志完整保留在 `metadata/initial_mean_audit_failed/`；最终两种输入模式按统一修订协议全量重新推理，初轮成功子集没有混入最终统计。", "",
        ])
    paragraphs.extend(["详细记录见 [numerical_audit.json](numerical_audit.json)。这些数值检查不等同于模型质量改善。", ""])
    return paragraphs


def load_comparison(root: Path) -> dict:
    """Validate completion and fixed denominators before publishing anything."""
    raw = {}
    expected_ids = None
    split_digest = None
    for mode in ("posterior_mean", "archived_latents"):
        directory = root / "raw_metrics" / mode
        summary = read_json(directory / "summary.json")
        rows = read_rows(directory / "per_object.csv", "model_id")
        if summary["intended_count"] != EXPECTED_INTENDED or summary["original_count"] != EXPECTED_ORIGINAL:
            raise ValueError(f"Unexpected fixed denominators for raw/{mode}")
        if summary["mode"] != mode or set(summary["methods"]) != set(METHODS):
            raise ValueError(f"Wrong mode/methods in raw/{mode}")
        if summary.get("candidate_method", "intersection") != "intersection" or not summary["protocol_valid"]:
            raise ValueError(f"Unverified comparison protocol for raw/{mode}")
        digest = summary.get("split_sha256")
        if not digest or (split_digest is not None and digest != split_digest):
            raise ValueError("Raw modes must use the same recorded split hash")
        split_digest = digest
        ids = set(rows["legacy"])
        if expected_ids is not None and ids != expected_ids:
            raise ValueError("Raw modes have different object sets")
        expected_ids = ids
        for method in METHODS:
            actual_status = dict(Counter(row["status"] for row in rows[method].values()))
            if actual_status != summary["methods"][method]["status_counts"]:
                raise ValueError(f"Raw summary/CSV status counts disagree for {mode}/{method}")
            if any(row["status"] in ("not_run", "running", "pending", "") for row in rows[method].values()):
                raise ValueError(f"Raw results are not complete for {mode}/{method}")
        shared = Counter(row.get("shared_input_status") for row in rows["legacy"].values())
        if dict(shared) != summary["shared_input_status_counts"] or shared.get("mismatch", 0) or shared.get("missing_digest", 0):
            raise ValueError(f"Raw input digest status disagrees for {mode}")
        metrics = ("face_cd_sq", "edge_cd_sq")
        if mode == "posterior_mean":
            metrics += ("face_mae", "topology_precision", "topology_recall", "topology_f1")
        for metric in metrics:
            pair = summary["paired"][metric]
            validate_statistics(pair, pair["common_count"], f"{mode}/{metric}")
        for method in METHODS:
            for metric in ("legacy_reciprocal_cd_sq_mean", "legacy_reciprocal_cd_sq_max", "legacy_unilateral_pair_count"):
                stat_mean(summary["methods"][method]["metrics"][metric])
        raw[mode] = summary

    post, transitions, timeout_diagnostics = {}, {}, {}
    for mode in POST_MODES:
        directory = root / "post" / mode
        summary = read_json(directory / "summary.json")
        rows = read_rows(directory / "samples.csv", "sample_id")
        if summary["intended_count_per_method"] != EXPECTED_INTENDED or summary["original_test_count"] != EXPECTED_ORIGINAL:
            raise ValueError(f"Unexpected fixed denominators for post/{mode}")
        if set(summary["methods"]) != set(METHODS) or summary["protocol"]["mode"] != mode:
            raise ValueError(f"Wrong mode/methods in post/{mode}")
        if summary["protocol"]["split_sha256"] != split_digest or set(rows["legacy"]) != expected_ids:
            raise ValueError(f"Post/raw object sets or split hashes disagree for {mode}")
        timeout_diagnostics[mode] = {}
        for method in METHODS:
            method_rows = list(rows[method].values())
            if any(row.get("status") not in TERMINAL for row in method_rows):
                raise ValueError(f"Unfinished post results for {mode}/{method}")
            recorded = summary["methods"][method]
            if recorded["counts"]["terminal_records"] != EXPECTED_INTENDED:
                raise ValueError(f"Incomplete post summary for {mode}/{method}")
            if dict(Counter(row["status"] for row in method_rows)) != recorded["status_counts"]:
                raise ValueError(f"Post summary/CSV statuses disagree for {mode}/{method}")
            timed_out = [row for row in method_rows if row["status"] == "timeout"]
            timeout_diagnostics[mode][method] = {
                "all_stages": len(timed_out),
                "already_valid_solid": sum(csv_bool(row.get("native_valid_solid")) for row in timed_out),
                "geometry_stage": sum(row.get("stage") == "geometry" for row in timed_out),
            }
            for key in ("native_valid_solid", "compat_valid_solid", "step_exists", "step_readable", "success_marker"):
                count = sum(csv_bool(row.get(key)) for row in method_rows)
                if recorded["counts"][key] != count:
                    raise ValueError(f"Post summary/CSV {key} disagree for {mode}/{method}")
                for rate_key, denominator in (("rates_over_intended", EXPECTED_INTENDED), ("rates_over_original_test", EXPECTED_ORIGINAL)):
                    if not math.isclose(recorded[rate_key][key], count / denominator, rel_tol=1e-10, abs_tol=1e-12):
                        raise ValueError(f"Incorrect {rate_key}/{key} for {mode}/{method}")
            count = sum(row.get("geometry_status") == "ok" for row in method_rows)
            if recorded["counts"]["geometry_computable"] != count:
                raise ValueError(f"Geometry coverage disagrees for {mode}/{method}")

        common = [item for item in sorted(expected_ids) if all(rows[method][item].get("geometry_status") == "ok" for method in METHODS)]
        paired = summary["paired_common_geometry"]
        if paired["count"] != len(common):
            raise ValueError(f"Common geometry coverage disagrees for {mode}")
        for metric in ("face_cd", "edge_cd"):
            validate_statistics(paired["metrics"][metric], len(common), f"post/{mode}/{metric}")
            for method in METHODS:
                values = [float(rows[method][item][metric]) for item in common]
                if any(not math.isfinite(value) or value < 0 for value in values):
                    raise ValueError(f"Invalid common geometry value in {mode}/{method}/{metric}")
                if values and not math.isclose(float(np.mean(values)), stat_mean(paired["metrics"][metric][method]), rel_tol=1e-9, abs_tol=1e-12):
                    raise ValueError(f"Common geometry summary/CSV means disagree for {mode}/{method}/{metric}")
                median = stat_value(paired["metrics"][metric][method], "median")
                if values and not math.isclose(float(np.median(values)), median, rel_tol=1e-9, abs_tol=1e-12):
                    raise ValueError(f"Common geometry summary/CSV medians disagree for {mode}/{method}/{metric}")

        groups = {name: [] for name in ("old_failure_new_success", "old_success_new_failure", "both_success", "both_failure")}
        for item in sorted(expected_ids):
            old = csv_bool(rows["legacy"][item].get("native_valid_solid"))
            new = csv_bool(rows["intersection"][item].get("native_valid_solid"))
            key = ("both_success" if old else "old_failure_new_success") if new else ("old_success_new_failure" if old else "both_failure")
            groups[key].append(item)
        transitions[mode] = {
            "intended_count": EXPECTED_INTENDED, "original_count": EXPECTED_ORIGINAL,
            "contract_unsupported_count": EXPECTED_ORIGINAL - EXPECTED_INTENDED,
            "counts": {key: len(value) for key, value in groups.items()}, "ids": groups,
        }
        post[mode] = summary
    audit = read_json(root / "training_audit.json")
    if not audit["checkpoint_recommendation"]["path"] or not audit["validation_history"]:
        raise ValueError("Missing checkpoint selection evidence in training_audit.json")
    return {
        "raw": raw, "post": post, "transitions": transitions, "audit": audit,
        "split_sha256": split_digest, "numerical_audit": load_numerical_audit(root),
        "timeout_diagnostics": timeout_diagnostics,
    }


def number(value: float | None, percentage: bool = False) -> str:
    if value is None:
        return "不可计算"
    return f"{value * 100:.4f}%" if percentage else f"{value:.6g}"


def change(old: float | None, new: float | None, *, higher: bool = False) -> str:
    if old is None or new is None:
        return "不可判断"
    difference = new - old
    if math.isclose(old, new, rel_tol=1e-8, abs_tol=1e-12):
        return "持平"
    magnitude = f"{abs(difference) * 100:.4f} 个百分点" if higher else (f"{abs(difference / old) * 100:.2f}%" if old else f"绝对差 {abs(difference):.6g}")
    if higher:
        return f"{'提高' if difference > 0 else '降低'} {magnitude}"
    return f"误差{'降低' if difference < 0 else '增加'} {magnitude}"


def comparison_table(summary: dict, names: tuple[tuple[str, str], ...]) -> list[str]:
    lines = ["| 指标 | 共同可计算 CAD | 原版 | Intersection | 变化 |", "|---|---:|---:|---:|---|"]
    for key, label in names:
        pair = summary["paired"][key]
        old, new = (stat_mean(pair[method]) for method in METHODS)
        higher = key.startswith("topology_")
        lines.append(f"| {label} | {pair['common_count']} | {number(old, higher)} | {number(new, higher)} | {change(old, new, higher=higher)} |")
    return lines


def reciprocal_comparison(raw: dict) -> list[str]:
    lines = [
        "", "## 同一条边的两份半边：是否画在一起", "",
        "每对互反半边先计算 XYZ 轨迹的平方 CD，再在每件 CAD 内取平均或最大值，最后对 CAD 取平均。"
        "该 CD 不检查半边方向，也不证明环闭合。两模型各自选中的面对集合可能不同，本表不是固定相同 pair 的消融。", "",
        "| 输入 | 指标 | 各自可计算 CAD（原/新） | 原版 | Intersection | 变化 |",
        "|---|---|---:|---:|---:|---|",
    ]
    for mode, title in (("posterior_mean", "Posterior mean"), ("archived_latents", "归档 diffusion latent")):
        for metric, label in (("legacy_reciprocal_cd_sq_mean", "每件平均互反差距 ↓"), ("legacy_reciprocal_cd_sq_max", "每件最差互反差距 ↓")):
            stats = [raw[mode]["methods"][method]["metrics"][metric] for method in METHODS]
            old, new = (stat_mean(item) for item in stats)
            lines.append(f"| {title} | {label} | {stats[0]['count']}/{stats[1]['count']} | {number(old)} | {number(new)} | {change(old, new)} |")
    lines.extend(["", "“每件最差”一行是各件最大差距的**均值**，不是全数据集最大的那一对。平均差距降低时，较差的边仍可能没有改善。", ""])
    new_is_zero = True
    for mode, title in (("posterior_mean", "Posterior mean"), ("archived_latents", "归档 latent")):
        old, new = (stat_mean(raw[mode]["methods"][method]["metrics"]["legacy_unilateral_pair_count"]) for method in METHODS)
        lines.append(f"- {title} 的每件单向 pair 数：{number(old)} → {number(new)}。")
        new_is_zero = new_is_zero and new == 0
    if new_is_zero:
        lines.extend(["", "新模型的单向 pair 为 0，来自 logits 的显式对称化及相同阈值筛选，不能解读为模型已经学会拓扑闭合。", ""])
    else:
        lines.extend(["", "单向 pair 计数需要结合 logits 对称化与导出规则解释，不能代替拓扑闭合检查。", ""])
    return lines


def build_report(data: dict) -> str:
    raw, post, transitions, audit = (data[key] for key in ("raw", "post", "transitions", "audit"))
    best, last = audit["checkpoint_recommendation"], audit["last_checkpoint"]
    lines = [
        "# Intersection refinement 测试对比", "",
        "本报告比较原 HoLa light 与本次 Intersection 微调模型。固定测试 2417 件 CAD，另将原始清单中 7 件合同不支持样本保留在 2424 的总体分母中。", "",
        "![四组指标对比](visualizations/intersection_comparison.png)", "",
        "## 权重与训练状态", "",
        f"权重只按验证集最小 loss 选择：`{Path(best['path']).name}`，epoch {best['epoch']}、step {best['global_step']}，验证 loss={best['best_model_score']:.8f}。没有按 test 挑权重。",
        f"中断前最后保存为 epoch {last['epoch']}、step {last['global_step']}；最后记录的训练 step 为 {audit['last_logged_global_step']}。", "",
    ]
    trend = audit["val_trends"]["loss"]
    no_improvement = int(audit["validation_history"][-1]["epoch"]) - best["epoch"]
    lines.extend([
        f"验证总损失前 3 轮均值 {trend['first_three_mean']:.6f}，末 3 轮 {trend['last_three_mean']:.6f}，变化 {trend['last_vs_first_three_pct']:+.2f}%；最近 8 轮标准差 {trend['last_eight_std_population']:.6f}，最好值之后 {no_improvement} 轮未再刷新。此处描述训练现象，最终判断看下面的几何和 solid。",
        "原验证使用 posterior 随机采样，本次重建使用固定 posterior mean；两种总 loss 不直接比较。完整训练曲线审计见 [training_audit.json](training_audit.json)。", "",
    ])
    lines.extend(numerical_audit_paragraphs(data.get("numerical_audit")))
    lines.extend([
        "## 原始解码：同一 posterior mean", "",
        "下表在各指标共同可计算的 ID 上取平均。MAE/CD 越小越好；连接 F1 越大越好。连接指标针对不同面的无向连接关系，不把互反半边算成重复边。", "",
    ])
    lines.extend(comparison_table(raw["posterior_mean"], (("face_mae", "面 XYZ MAE ↓"), ("face_cd_sq", "面平方 CD ↓"), ("edge_cd_sq", "边平方 CD ↓"), ("topology_f1", "连接 F1 ↑"))))
    lines.extend(["", "| Raw 覆盖 | 原版 | Intersection |", "|---|---:|---:|"])
    for name, label in (("files_present", "预测文件存在"), ("face_geometry_computable", "面几何可计算"), ("edge_geometry_computable", "边几何可计算"), ("topology_scores_computable", "连接评分可计算")):
        values = [raw["posterior_mean"]["methods"][method]["coverage"][name]["count"] for method in METHODS]
        lines.append(f"| {label} | {values[0]}/2417 | {values[1]}/2417 |")
    lines.extend(["", "## 后处理：能否得到合法 solid", "", "主口径为原读取精度下的 `TopAbs_SOLID + BRepCheck`；成功写出 STEP 或出现 success marker 不等于合法 solid。失败、超时、缺失始终留在分母里。", "", "| 后处理 | 方法 | 合法 solid | /2417 | /2424 | 兼容容差口径 solid |", "|---|---|---:|---:|---:|---:|"])
    for mode in POST_MODES:
        for method in METHODS:
            counts = post[mode]["methods"][method]["counts"]
            count = counts["native_valid_solid"]
            lines.append(f"| {MODE_NAMES[mode]} | {method} | {count} | {count / EXPECTED_INTENDED:.2%} | {count / EXPECTED_ORIGINAL:.2%} | {counts['compat_valid_solid']} |")
    lines.extend(["", "兼容口径使用容差 0.1，单列用于衔接历史结果，不替代主口径。", "", "| 后处理 | 原失败→新成功 | 原成功→新失败 | 都成功 | 都失败 | 净增 solid |", "|---|---:|---:|---:|---:|---:|"])
    for mode in POST_MODES:
        counts = transitions[mode]["counts"]
        gained, lost = counts["old_failure_new_success"], counts["old_success_new_failure"]
        lines.append(f"| {MODE_NAMES[mode]} | {gained} | {lost} | {counts['both_success']} | {counts['both_failure']} | {gained - lost:+d} |")
    lines.extend(["", "四类合计仅为共同测试的 2417 件；7 件合同不支持样本单列，不伪装成经过测试的“双失败”。逐 ID 清单见 [solid_transitions.json](solid_transitions.json)。", "", "| 后处理 | 方法 | 未得到原生合法 solid | 所有阶段超时* | 预测缺失 | 几何可计算 |", "|---|---|---:|---:|---:|---:|"])
    for mode in POST_MODES:
        for method in METHODS:
            row = post[mode]["methods"][method]
            lines.append(f"| {MODE_NAMES[mode]} | {method} | {EXPECTED_INTENDED - row['counts']['native_valid_solid']} | {row['status_counts'].get('timeout', 0)} | {row['status_counts'].get('missing_raw', 0)} | {row['counts']['geometry_computable']}/2417 |")
    lines.extend(["", "*超时与 solid 失败不是互斥分组：先通过 solid 检查、随后几何采样超时的对象，仍计入已确认的合法 solid，但几何指标不可计算。", ""])
    for mode in POST_MODES:
        for method in METHODS:
            timeout = data["timeout_diagnostics"][mode][method]
            if timeout["all_stages"]:
                lines.append(f"- {MODE_NAMES[mode]} / {method}：共 {timeout['all_stages']} 件超时，其中 {timeout['geometry_stage']} 件发生在几何采样阶段、{timeout['already_valid_solid']} 件已通过 solid 检查；均保留在固定分母。")
    lines.extend(["", "## 后处理几何：两方法都能计算的同一批 CAD", "", "这些是原生 STEP 采样后的平方 CD；共同可计算不要求 solid 合法，因此不能把本表当作有效 solid 子集。共同集合的覆盖率同时报告，不能代替全分母成功率。", "", "均值和中位数分别列出：少数误差特别大的案例会拉高均值，平均误差降低不代表多数样本都更准。中位数变化也不是逐对象改善比例。", "", "| 后处理 | 共同数量（/2417；/2424） | 指标/统计量 | 原版 | Intersection | 变化 |", "|---|---|---|---:|---:|---|"])
    for mode in POST_MODES:
        common = post[mode]["paired_common_geometry"]
        count = common["count"]
        for metric, label in (("face_cd", "面平方 CD ↓"), ("edge_cd", "边平方 CD ↓")):
            for statistic, statistic_label in (("mean", "均值"), ("median", "中位数")):
                old, new = (stat_value(common["metrics"][metric][method], statistic) for method in METHODS)
                lines.append(f"| {MODE_NAMES[mode]} | {count}（{count / EXPECTED_INTENDED:.2%}；{count / EXPECTED_ORIGINAL:.2%}） | {label} · {statistic_label} | {number(old)} | {number(new)} | {change(old, new)} |")
    lines.extend(["", "## 固定归档 diffusion latent：仅替换解码器", "", "两方法读取同一批之前归档的 diffusion latent，不重新采样。本表只检验这些既定输入下的解码表现；没有可用的 GT 面顺序，因此不报告面对应 MAE 或 GT 拓扑 F1。", ""])
    lines.extend(comparison_table(raw["archived_latents"], (("face_cd_sq", "面平方 CD ↓"), ("edge_cd_sq", "边平方 CD ↓"))))
    lines.extend(reciprocal_comparison(raw))
    lines.extend(["", "## 数据支持的判断", ""])
    posterior = raw["posterior_mean"]
    for metric, label in (("face_mae", "原始面的坐标误差"), ("edge_cd_sq", "原始边的几何误差"), ("topology_f1", "原始连接 F1")):
        old, new = (stat_mean(posterior["paired"][metric][method]) for method in METHODS)
        lines.append(f"- {label}：{change(old, new, higher=metric.startswith('topology_'))}。")
    for mode in POST_MODES:
        old, new = (post[mode]["methods"][method]["counts"]["native_valid_solid"] for method in METHODS)
        lines.append(f"- {MODE_NAMES[mode]}后的合法 solid：{old} → {new}，{change(old / EXPECTED_INTENDED, new / EXPECTED_INTENDED, higher=True)}。")
    lines.extend(["", "本次同时包含旧权重初始化后的微调、新损失和候选规则改变，并非等预算的单模块消融，不能把差异单独归因于新增校对块。固定 diffusion latent 重放也不是重新训练 diffusion 后的端到端照片重建评估。", ""])
    reused_modes = [MODE_NAMES[mode] for mode in POST_MODES if post[mode]["protocol"].get("reference_legacy", {}).get("result_origin") == "reference_reused"]
    if reused_modes:
        lines.extend([f"原版的{'、'.join(reused_modes)}结果复用通过协议与输入哈希审核的既有记录，并非本次重新运行。", ""])
    lines.extend(["原始指标、采样/优化预算、失败状态与来源以以下机器可读记录为准：", "", "- [posterior raw summary](raw_metrics/posterior_mean/summary.json)", "- [archived latent raw summary](raw_metrics/archived_latents/summary.json)", "- [fit-only summary](post/fit_only/summary.json) / [逐对象状态](post/fit_only/samples.csv)", "- [optimized summary](post/optimized/summary.json) / [逐对象状态](post/optimized/samples.csv)", "- [冻结评估协议](PLAN.md)", ""])
    return "\n".join(lines)


def plot_ratios(ax, summary: dict, names: tuple[tuple[str, str], ...], title: str) -> None:
    values = []
    counts = []
    for metric, _ in names:
        pair = summary["paired"][metric]
        old, new = (stat_mean(pair[method]) for method in METHODS)
        values.append(new / old if old is not None and old > 0 and new is not None else float("nan"))
        counts.append(pair["common_count"])
    finite = [value for value in values if math.isfinite(value)]
    upper = max([1.0, *finite]) * 1.27
    ax.bar(np.arange(len(names)), values, color=COLORS["intersection"], width=0.55)
    ax.axhline(1.0, color=COLORS["legacy"], linestyle="--", linewidth=1.2, label="Legacy = 1")
    for index, value in enumerate(values):
        label = f"{value:.3f}x\nn={counts[index]}" if math.isfinite(value) else f"N/A\nn={counts[index]}"
        ax.text(index, value + upper * 0.035 if math.isfinite(value) else upper * 0.1, label, ha="center", fontsize=9)
    ax.set_xticks(np.arange(len(names)), [label for _, label in names])
    ax.set_ylim(0, upper)
    ax.set_ylabel("Intersection / Legacy mean error")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=9)


def build_figure(data: dict, path: Path) -> None:
    raw, post = data["raw"], data["post"]
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False}):
        figure, axes = plt.subplots(2, 2, figsize=(12.5, 8.8), layout="constrained")
        plot_ratios(axes[0, 0], raw["posterior_mean"], (("face_mae", "Face MAE"), ("face_cd_sq", "Face squared CD"), ("edge_cd_sq", "Edge squared CD")), "Posterior mean: raw geometry (lower is better)")
        ax = axes[0, 1]
        for index, method in enumerate(METHODS):
            values = [stat_mean(raw["posterior_mean"]["paired"]["topology_" + key][method]) for key in ("precision", "recall", "f1")]
            values = [100 * value if value is not None else float("nan") for value in values]
            bars = ax.bar(np.arange(3) + (index - 0.5) * 0.38, values, width=0.35, label=method.title(), color=COLORS[method])
            ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
        ax.set_xticks(np.arange(3), ["Precision", "Recall", "F1"])
        ax.set_ylim(0, 113)
        ax.set_ylabel("Paired macro score (%)")
        ax.set_title("Posterior mean: face connections (higher is better)")
        ax.legend(frameon=False, loc="upper center", ncol=2)
        ax = axes[1, 0]
        for index, method in enumerate(METHODS):
            counts = [post[mode]["methods"][method]["counts"]["native_valid_solid"] for mode in POST_MODES]
            values = np.asarray(counts) / EXPECTED_INTENDED * 100
            bars = ax.bar(np.arange(2) + (index - 0.5) * 0.38, values, width=0.35, color=COLORS[method], label=method.title())
            ax.bar_label(bars, labels=[f"{value:.2f}%\n{count}/{EXPECTED_INTENDED}" for value, count in zip(values, counts)], padding=3, fontsize=9)
        ax.set_xticks(np.arange(2), ["Fit only", "Geometry optimized"])
        ax.set_ylim(0, 115)
        ax.set_ylabel("As-read valid solid / 2,417 (%)")
        ax.set_title("OCC validity: failures stay in the denominator")
        ax.legend(frameon=False, loc="upper center", ncol=2)
        plot_ratios(axes[1, 1], raw["archived_latents"], (("face_cd_sq", "Face squared CD"), ("edge_cd_sq", "Edge squared CD")), "Fixed archived diffusion latents: decoder replay")
        figure.suptitle("Legacy vs Intersection refinement | 2,417 accepted CADs; original split 2,424", fontsize=13)
        figure.savefig(path, dpi=180, facecolor="white", bbox_inches="tight")
        plt.close(figure)


def generate_report(root: Path) -> dict[str, str]:
    root = root.resolve()
    paths = {
        "report": root / "REPORT.md",
        "transitions": root / "solid_transitions.json",
        "figure": root / "visualizations" / "intersection_comparison.png",
    }
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite existing report artifacts: {existing}")
    data = load_comparison(root)
    report = build_report(data)
    transitions = json.dumps(data["transitions"], ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    # Render in a temporary directory, so incomplete inputs/render failures do
    # not publish a REPORT.md that appears to describe a completed evaluation.
    with tempfile.TemporaryDirectory(prefix="intersection-report-", dir=root) as temporary:
        staged_figure = Path(temporary) / "intersection_comparison.png"
        build_figure(data, staged_figure)
        paths["figure"].parent.mkdir(parents=True, exist_ok=True)
        with paths["transitions"].open("x", encoding="utf-8") as handle:
            handle.write(transitions)
        with paths["figure"].open("xb") as handle:
            handle.write(staged_figure.read_bytes())
        with paths["report"].open("x", encoding="utf-8") as handle:
            handle.write(report)
    return {key: str(value) for key, value in paths.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = generate_report(args.comparison_root)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, f"Report not generated: {error}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
