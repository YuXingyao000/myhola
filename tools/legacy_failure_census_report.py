"""Aggregate the frozen legacy failure census; never run geometry or post-processing.

All mechanism frequencies retain the intended failure denominator. Positive,
negative and unknown are separate; overlapping observations are not root causes.
Charts use matplotlib and the offline page only displays recorded statistics.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import html
import json
import math
from pathlib import Path

from tools.legacy_failure_select import file_sha256, read_table


VERSION = "legacy_failure_census_report_v1"
THRESHOLDS = ("0.002", "0.006", "0.01", "0.08")
TERMINAL = {
    "trim_gate": ("Trim-count gate", "裁剪面数量门槛阻止 solid 尝试"),
    "solid_construction": ("Solid construction", "已进入 solid 构造，但没有返回 solid"),
    "saved_step_rejected": ("Legacy save check", "内存 solid 返回后被旧保存检查拒绝"),
    "native_readback_invalid": ("Native readback", "旧保存检查通过，但原生重读仍无效"),
    "unresolved": ("Unresolved terminal branch", "完整轨迹仍未定位到上述末端分支"),
    "unknown": ("Missing / unverified trace", "诊断缺失、不完整或重放未核验一致"),
}
TRACE_LABELS = {
    "wire_not_returned": ("Wire assembly returned None", "至少一次 wire 组装未返回"),
    "wire_skipped_open_after_fix": ("Open wire discarded", "开放 wire 被舍弃（不推断内环或孔）"),
    "all_wires_rejected": ("No wire admitted", "至少一次候选面没有接纳任何 wire"),
    "partial_wire_drop_with_face_returned": ("Wire discarded, face returned", "开放 wire 被舍弃后仍返回面"),
    "closed_wire_admitted_but_trim_failed": ("Closed wire admitted, trim failed", "接纳闭合 wire 后候选裁剪仍失败"),
    "face_context_brepcheck_error": ("Candidate BRepCheck error", "候选裁剪面或其上下文有 BRepCheck 错误"),
    "face_fix_exception": ("Face fix exception", "面修复分支发生异常"),
    "periodic_wire_substitution": ("Periodic wire substitution", "执行过周期曲面的 wire 替换（不等于错误）"),
    "sewing_compound_early_return": ("Sewing compound rejection", "sewing compound 分支提前返回"),
    "sewing_free_edges": ("Sewing has free edges", "至少一次 solid 尝试的 sewing 有自由边"),
    "sewing_multiple_edges": ("Sewing has multiple edges", "至少一次 sewing 有 multiple edges"),
    "no_free_edge_but_solid_failed": ("No free edges, no solid", "无自由边的尝试仍未返回 solid"),
    "trim_gate_blocked": ("Trim-count gate blocked", "裁剪面数量门槛阻断"),
    "saved_step_rejected": ("Legacy save check rejected", "旧保存检查拒绝过返回的 solid"),
    "saved_check_passed_native_failed": ("Save passed, native failed", "旧保存检查通过但原生重读无效"),
    "uv_specific_failure": ("Specific UV cause unresolved", "具体 UV 参数域原因尚未识别"),
}
UPSTREAM_LABELS = {
    "raw_extra_gt_pair": ("Raw extra GT pair", "raw 存在额外无向 GT face pair"),
    "raw_missing_gt_pair": ("Raw missing GT pair", "raw 缺失无向 GT face pair"),
    "raw_duplicate_directed_pair": ("Duplicate directed pair", "raw 同方向 pair 重复"),
    "raw_same_face_pair": ("Raw self pair", "raw 同一面自连接"),
    "selection_removed_gt_pair": ("Selection removed GT pair", "半边选择删除了原有的正确 GT pair"),
}
THRESHOLD_LABELS = {
    "reciprocal_excess_disagreement": ("Reciprocal disagreement beyond GT", "正常双向半边比 GT 更不一致"),
    **{f"{stage}_{suffix}": (f"{stage.title()}: {label}", f"{stage}：{chinese}")
       for stage in ("raw", "selected", "optimized")
       for suffix, label, chinese in (
           ("new_cross_pair_curve_nearness", "new cross-pair nearness", "跨 pair 曲线接近，且 GT 不接近"),
           ("new_face_grid_nearness", "new face-grid nearness", "面网格接近，且 GT 不接近"))},
    **{f"{stage}_endpoint_gap": (f"{stage.title()}: endpoint gap proxy", f"{stage}：端点残差代理超阈值")
       for stage in ("selected", "optimized")},
    **{f"{stage}_new_open_edge_endpoint_collapse": (f"{stage.title()}: open-edge endpoints newly close", f"{stage}：GT 开边首末点新接近")
       for stage in ("selected", "optimized")},
}
CHEAP_LABELS = {
    "extra_pair": ("Extra GT pair", "额外无向 GT pair"),
    "missing_pair": ("Missing GT pair", "缺失无向 GT pair"),
    "any_pair_error": ("Any pair mismatch", "任一无向 pair 错误"),
    "unilateral_pair": ("Unilateral raw pair", "raw 单向 pair"),
    "directed_duplicate": ("Directed duplicate", "同方向 pair 重复"),
    "face_count_mismatch": ("Face count mismatch", "face count 偏差"),
}
PERSISTENT_LABELS = {
    "never_wire": ("Failed face: no wire ever returned", "最终失败面：所有容差均未组装出 wire"),
    "never_closed_admitted": ("Failed face: no closed wire admitted", "最终失败面：有 wire，但从未接纳闭合 wire"),
    "closed_admitted_still_failed": ("Failed face: closed wire admitted", "最终失败面：曾接纳闭合 wire，仍未返回裁剪面"),
    "strict_closed_candidate_failed": ("Failed face: all candidate wires admitted", "最终失败面：存在所有候选 wire 均闭合并接纳的失败尝试"),
    "final_success_discarded_wire": ("Returned face discarded open wire", "最终成功裁剪的面：成功尝试舍弃了开放 wire"),
}
FIT_LABELS = {
    "faces": ("Input points -> fitted surface", "面输入采样点到拟合曲面残差"),
    "curves": ("Input points -> fitted curve", "边输入采样点到拟合曲线残差"),
    "curve_owner": ("Fitted curve -> owner surfaces", "周期替换后的曲线到两侧拟合曲面残差"),
}
LOSS_KEYS = ("loss", "adj_distance_loss", "corners_loss", "wire_connected_loss")
LIMITATIONS = [
    "固定 369 个原生无效 solid 的全集；未知诊断不移出分母。原 test 的 7/2424 个契约不支持对象另列。",
    "互斥图描述末端阻断分支，不是互斥根因。机制为多标签观察，总和可以超过 369。",
    "wire 被舍弃不能证明它是内环、外环或孔；闭合 wire 和小 3D 残差也不能证明 UV 裁剪正确。",
    "正常 reciprocal 是表示冗余；跨 pair 近曲线、GT 不近的面网格接近、STEP 近重合自由边是不同观察，不能混作模型重复。",
    "阈值距离是归一化 XYZ 长度；旧 reciprocal CD 是双向均值平方距离之和，不能与新长度阈值直接比较。",
    "端点最近邻残差只是一项必要条件代理，不是闭环或 B-Rep 有效性证明。",
    "raw FP/FN 与失败裁剪面的关联是证据共现；面局部无向邻接正确不代表几何、方向或多重边正确。",
    "成功组对照仅重读已有 CSV；未对成功组运行本次 OCC 溯源或新增几何诊断，描述性关联不是因果。",
]


def get(data, *keys, default=None):
    for key in keys:
        if not isinstance(data, dict) or key not in data:
            return default
        data = data[key]
    return data


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def tri(value):
    return value if type(value) is bool else None


def tally(pairs):
    """IDs retained for every state, including unknown; no known-only percentages."""
    groups = {"positive": [], "negative": [], "unknown": []}
    for sid, value in pairs:
        groups["positive" if value is True else "negative" if value is False else "unknown"].append(sid)
    denominator = sum(map(len, groups.values()))
    return {"denominator": denominator, **{key: len(ids) for key, ids in groups.items()},
            "positive_fraction_of_intended": len(groups["positive"]) / denominator if denominator else None,
            "ids": groups}


def distribution(values):
    import numpy as np
    values = [float(value) for value in values if finite(value)]
    return {"finite_count": len(values), "mean": float(np.mean(values)) if values else None,
            "median": float(np.median(values)) if values else None,
            "p90": float(np.quantile(values, .9)) if values else None}


def trace_verified(trace):
    return (trace.get("status") == "ok" and not trace.get("instrumentation_errors")
            and get(trace, "comparison", "same_recorded_outcome") is True
            and get(trace, "step_data_comparison", "equal") is True)


def terminal_stage(trace):
    if not trace_verified(trace):
        return "unknown"
    saves = trace.get("save_attempts", [])
    if any(item.get("legacy_saved_step_check") is True for item in saves):
        return "native_readback_invalid" if get(trace, "replay", "native_valid_solid") is False else "unresolved"
    if any(item.get("legacy_saved_step_check") is False for item in saves):
        return "saved_step_rejected"
    if get(trace, "trim_gate", "solid_attempt_allowed") is False:
        return "trim_gate"
    attempts = trace.get("solid_attempts", [])
    if attempts and all(get(item, "result", "present") is False for item in attempts):
        return "solid_construction"
    return "unresolved"


def local_failed_faces(case, trace, upstream, identity_contract):
    """Join decoder-indexed failed faces to raw pair endpoints, never STEP IDs."""
    unknown = {"status": "unknown", "failed_faces": None,
               "reason": "Verified trace, identity-preserving main trim and raw pair lists are required"}
    if not trace_verified(trace) or identity_contract.get("gt_order_preserved") is not True:
        return unknown
    runs = [run for run in trace.get("trim_runs", []) if run.get("context") == "main"]
    if len(runs) != 1 or runs[0].get("num_faces") != case["raw_legacy"].get("pred_face_count"):
        return unknown
    failed = runs[0].get("failed_face_indices")
    if not isinstance(failed, list) or any(type(index) is not int or not 0 <= index < runs[0]["num_faces"] for index in failed):
        return unknown
    structure = get(upstream, "structure", "raw", default={})
    known = structure.get("status") == "ok" and isinstance(structure.get("fp_pairs"), list) and isinstance(structure.get("fn_pairs"), list)
    endpoints = set()
    if known:
        for pair in structure["fp_pairs"] + structure["fn_pairs"]:
            if len(pair) != 2 or any(type(index) is not int or not 0 <= index < runs[0]["num_faces"] for index in pair):
                known = False
                break
            endpoints.update(pair)
    rows = [{"face_id": index, "category": ("raw_wrong_pair_incident" if index in endpoints else "raw_local_pair_exact")
             if known else "unknown"} for index in sorted(set(failed))]
    return {"status": "ok" if known else "partial", "failed_faces": rows,
            "note": "Main trim decoder IDs; unordered adjacency only; co-occurrence, not causal attribution"}


def raw_topology_class(raw):
    fp, fn = raw.get("topology_fp"), raw.get("topology_fn")
    if not finite(fp) or not finite(fn):
        return "unknown"
    return "exact" if fp == fn == 0 else "inexact"


def persistent_trim_observations(trace):
    result = {"status": "unknown", "failed_faces": None, "successful_face_count": None,
              "successful_faces_discarding_wire": [], "flags": {key: None for key in PERSISTENT_LABELS}}
    if not trace_verified(trace):
        return result
    runs = [run for run in trace.get("trim_runs", []) if run.get("context") == "main"]
    if len(runs) != 1 or not isinstance(runs[0].get("failed_face_indices"), list):
        return result
    run, rows = runs[0], []
    failed = set(run["failed_face_indices"])
    for face_id in sorted(failed):
        attempts = [a for a in run.get("attempts", []) if a.get("face_index") == face_id]
        known, closed_admitted, strict = bool(attempts), False, False
        for attempt in attempts:
            if attempt.get("wire_returned") is False:
                continue
            detail = attempt.get("trim_detail", {})
            wires = detail.get("wire_events")
            if not isinstance(wires, list) or detail.get("outcome", "unknown") == "unknown":
                known = False
                continue
            admitted = [wire.get("added") is True and wire.get("repaired_closed") is True for wire in wires]
            closed_admitted = closed_admitted or any(admitted)
            strict = strict or (bool(wires) and all(admitted) and attempt.get("trim_returned") is False)
        if closed_admitted:
            category = "closed_admitted_still_failed"
        elif not known:
            category = "unknown"
        elif all(attempt.get("wire_returned") is False for attempt in attempts):
            category = "never_wire"
        else:
            category = "never_closed_admitted"
        rows.append({"face_id": face_id, "category": category,
                     "strict_closed_candidate_failed": True if strict else False if known else None})
    successful, unknown_successes = [], []
    for face_id in range(run["num_faces"]):
        if face_id in failed:
            continue
        attempts = [a for a in run.get("attempts", []) if a.get("face_index") == face_id and a.get("trim_returned") is True]
        if len(attempts) != 1 or not isinstance(get(attempts[0], "trim_detail", "wire_events"), list):
            unknown_successes.append(face_id)
        elif any(w.get("skip_reason") for w in attempts[0]["trim_detail"]["wire_events"]):
            successful.append(face_id)
    for key in ("never_wire", "never_closed_admitted", "closed_admitted_still_failed"):
        result["flags"][key] = True if any(row["category"] == key for row in rows) else None if any(row["category"] == "unknown" for row in rows) else False
    stricts = [row["strict_closed_candidate_failed"] for row in rows]
    result["flags"]["strict_closed_candidate_failed"] = True if True in stricts else None if None in stricts else False
    result["flags"]["final_success_discarded_wire"] = True if successful else None if unknown_successes else False
    result.update(status="ok" if not unknown_successes and all(row["category"] != "unknown" for row in rows) else "partial",
                  failed_faces=rows, successful_face_count=run["num_faces"] - len(failed),
                  successful_faces_discarding_wire=successful, unknown_successful_faces=unknown_successes)
    return result


def cheap_flags(raw):
    def positive(key):
        return raw[key] > 0 if finite(raw.get(key)) else None
    fp, fn = positive("topology_fp"), positive("topology_fn")
    return {"extra_pair": fp, "missing_pair": fn,
            "any_pair_error": True if fp is True or fn is True else False if fp is fn is False else None,
            "unilateral_pair": positive("legacy_unilateral_pair_count"),
            "directed_duplicate": positive("legacy_directed_duplicate_count"),
            "face_count_mismatch": raw["pred_face_count"] != raw["gt_face_count"]
            if finite(raw.get("pred_face_count")) and finite(raw.get("gt_face_count")) else None}


def raw_curve_decomposition(cases):
    """Only claim an additive decomposition where original counts prove equality."""
    values = Counter()
    checked, unknown = [], []
    for case in cases:
        raw = case["raw_legacy"]
        keys = ("raw_edge_count", "legacy_reciprocal_pair_count", "legacy_unilateral_pair_count",
                "legacy_directed_duplicate_count", "same_face_count", "invalid_face_index_count",
                "invalid_edge_index_count", "repeated_edge_reference_count")
        if (not all(finite(raw.get(key)) for key in keys)
                or any(raw[key] != 0 for key in keys[3:])
                or raw[keys[0]] != 2 * raw[keys[1]] + raw[keys[2]]):
            unknown.append(case["id"])
            continue
        checked.append(case["id"])
        values["raw_curve_rows"] += int(raw[keys[0]])
        values["first_rows_of_reciprocal_pairs"] += int(raw[keys[1]])
        values["second_rows_of_reciprocal_pairs"] += int(raw[keys[1]])
        values["unilateral_rows"] += int(raw[keys[2]])
    return {"unit": "raw curve rows; normal reciprocal is NOT an error", "counts": dict(values),
            "intended_objects": len(cases), "verified_objects": len(checked), "verified_ids": checked,
            "unresolved_objects": len(unknown), "unresolved_ids": unknown,
            "note": "Decomposition used only when no invalid/self/repeated/directed-duplicate rows and E = 2R + U. Cross-pair geometric nearness is separate and may overlap."}


def population_comparison(manifest):
    """Re-read existing complete tables, checking their frozen hashes first."""
    sources = manifest["sources"]
    for name in ("raw_metrics", "optimized", "fit_only"):
        if file_sha256(Path(sources[name]["path"])) != sources[name]["sha256"]:
            raise ValueError(f"Frozen source hash changed: {name}")
    raw = read_table(Path(sources["raw_metrics"]["path"]), "model_id")
    post = read_table(Path(sources["optimized"]["path"]), "sample_id")
    fit = read_table(Path(sources["fit_only"]["path"]), "sample_id")
    from tools.legacy_failure_census_manifest import partition_population
    ids, failed, transitions = partition_population(raw, post, fit)
    if set(failed) != {case["id"] for case in manifest["cases"]} or len(ids) != manifest["population"]["accepted_test"]:
        raise ValueError("Source population differs from frozen manifest")
    groups = {name: [sid for sid in ids if post["legacy", sid]["native_valid_solid"] is value]
              for name, value in (("native_valid", True), ("native_failed", False))}
    def summarize(group):
        rows = [raw["legacy", sid] for sid in group]
        return {"denominator": len(group),
                "flags": {key: tally((sid, cheap_flags(raw["legacy", sid])[key]) for sid in group) for key in CHEAP_LABELS},
                "continuous": {key: distribution(row.get(key) for row in rows) for key in
                               ("gt_face_count", "face_mae", "edge_cd_sq", "legacy_reciprocal_cd_sq_mean")}}
    bins = (("1-9", 1, 9), ("10-19", 10, 19), ("20-30", 20, 30))
    result = {"groups": {name: summarize(group) for name, group in groups.items()},
              "by_face_count": {label: {name: summarize([sid for sid in group
                  if low <= raw["legacy", sid]["gt_face_count"] <= high]) for name, group in groups.items()}
                  for label, low, high in bins}, "fit_only_vs_optimized": transitions,
              "continuous_note": "Existing reciprocal_cd_sq_mean = mean over reciprocal pairs of symmetric squared nearest-sample CD sum; unlike new length distances."}
    result["face_standardized_flags"] = {}
    for key in CHEAP_LABELS:
        common = [label for label, _, _ in bins if all(result["by_face_count"][label][g]["denominator"] > 0 for g in groups)]
        total = sum(sum(result["by_face_count"][label][g]["denominator"] for g in groups) for label in common)
        weights = {label: sum(result["by_face_count"][label][g]["denominator"] for g in groups) / total for label in common} if total else {}
        result["face_standardized_flags"][key] = {"weights": weights, "weighting": "Pooled accepted count in bins shared by both groups; descriptive, not causal",
            "rates": {g: sum(weights[label] * result["by_face_count"][label][g]["flags"][key]["positive_fraction_of_intended"] for label in common)
                      if weights and all(result["by_face_count"][label][g]["flags"][key]["unknown"] == 0 for label in common) else None for g in groups}}
    return result


def read_diagnostic(path, sid):
    if not path.is_file():
        return {}, {"path": str(path), "sha256": None, "status": "missing"}
    try:
        payload = path.read_bytes()
        data = json.loads(payload)
        if not isinstance(data, dict) or data.get("sample_id", data.get("id")) != sid:
            raise ValueError("Diagnostic ID differs from manifest")
        return data, {"path": str(path), "sha256": hashlib.sha256(payload).hexdigest(), "status": data.get("status", "unknown")}
    except (ValueError, OSError) as exc:
        return {}, {"path": str(path), "sha256": file_sha256(path), "status": "parse_error", "error": str(exc)}


def saved_step_observations(data):
    complete = data.get("status") == "ok" and not data.get("sampling_errors")
    result = {"free_edge_count": get(data, "native", "incidence", "free_edges"),
              "thresholds": {}}
    for threshold in THRESHOLDS:
        row = get(data, "free_edge_proximity", "thresholds", threshold, default={})
        count = row.get("pair_count")
        result["thresholds"][threshold] = {
            "has_near_free_edges": True if finite(count) and count > 0 else False if complete and count == 0 else None,
            "observed_pair_count": count, "observed_involved_edges": row.get("involved_edge_count"),
            "sampling_complete": complete}
    return result


def fit_observations(trace):
    fit = trace.get("fit_residuals", {}) if trace_verified(trace) else {}
    result = {}
    expected = {"faces": len(get(trace, "fitting", "faces", default=[])),
                "curves": get(trace, "fitting", "num_curves"),
                "curve_owner": 2 * get(trace, "fitting", "num_curves") if finite(get(trace, "fitting", "num_curves")) else None}
    for key in FIT_LABELS:
        rows = fit.get(key, [])
        if key == "curve_owner":
            rows = [{"edge_index": row["edge_index"], **owner} for row in rows
                    for owner in row.get("after_periodic_replacement", [])]
        complete = (fit.get("status") in ("ok", "partial") and expected[key] is not None
                    and len(rows) == expected[key] and bool(rows)
                    and all(row.get("status") == "ok" and row.get("count", 0) > 0
                            and row.get("count") == row.get("finite_count")
                            and finite(row.get("max")) and not row.get("errors") for row in rows))
        maxima = [row["max"] for row in rows if finite(row.get("max"))]
        result[key] = {"status": "ok" if complete else "partial" if maxima else "unknown",
            "expected_entity_count": expected[key], "recorded_entity_count": len(rows),
            "complete_entity_count": sum(row.get("status") == "ok" and row.get("count", 0) > 0 and row.get("count") == row.get("finite_count") for row in rows),
            "observed_max": max(maxima) if maxima else None,
            "threshold_flags": {t: True if any(value > float(t) for value in maxima) else False if complete else None for t in THRESHOLDS},
            "entities": [{key: row.get(key) for key in ("face_index", "edge_index", "count", "finite_count", "status", "mean", "max", "rms")} for row in rows]}
    return result


def optimization_loss_observations(trace):
    detail = trace.get("optimization_detail", {}) if trace_verified(trace) else {}
    rows = detail.get("loss_records", [])
    return {"returned_original": detail.get("returned_original"),
            "loss_timing": detail.get("loss_timing", "First/last recorded values precede optimizer updates"),
            "first_iteration": rows[0].get("iteration") if rows else None,
            "last_iteration": rows[-1].get("iteration") if rows else None,
            "terms": {key: {"first": rows[0].get(key) if rows else None, "last": rows[-1].get(key) if rows else None,
                             "delta": rows[-1][key] - rows[0][key] if len(rows) >= 2 and finite(rows[0].get(key)) and finite(rows[-1].get(key)) else None}
                      for key in LOSS_KEYS}}


def loss_down_gt_up(row, loss, metric):
    value = get(row, "optimization_loss", "terms", loss, "delta")
    delta = row["optimization_delta"][metric]
    if row["optimization_loss"]["returned_original"] is not False or not finite(value) or not finite(delta):
        return None
    return value < -1e-12 and delta > 1e-12


def loss_geometry_cross(records, loss, metric):
    cells = {key: [] for key in ("loss_down_gt_up", "loss_down_gt_not_up", "loss_not_down_gt_up", "loss_not_down_gt_not_up", "unknown")}
    for row in records:
        value = get(row, "optimization_loss", "terms", loss, "delta")
        delta = row["optimization_delta"][metric]
        if row["optimization_loss"]["returned_original"] is not False or not finite(value) or not finite(delta):
            key = "unknown"
        else:
            key = ("loss_down" if value < -1e-12 else "loss_not_down") + ("_gt_up" if delta > 1e-12 else "_gt_not_up")
        cells[key].append(row["id"])
    return {"denominator": len(records), "counts": {key: len(ids) for key, ids in cells.items()}, "ids": cells}


def closed_failed_owner_residual(record, threshold):
    faces = record["persistent_trim"]["failed_faces"]
    if faces is None:
        return None
    face_ids = {row["face_id"] for row in faces if row["category"] == "closed_admitted_still_failed"}
    if not face_ids:
        return None if any(row["category"] == "unknown" for row in faces) else False
    rows = [row for row in record["fit_residuals"]["curve_owner"]["entities"] if row.get("face_index") in face_ids]
    if any(finite(row.get("max")) and row["max"] > float(threshold) for row in rows):
        return True
    return False if record["fit_residuals"]["curve_owner"]["status"] == "ok" and rows else None


def aggregate(manifest, root):
    cases, records = manifest["cases"], []
    for case in cases:
        sid = case["id"]
        trace, trace_source = read_diagnostic(root / "traces" / f"{sid}.json", sid)
        upstream, upstream_source = read_diagnostic(root / "upstream" / f"{sid}.json", sid)
        saved, saved_source = read_diagnostic(root / "saved_step" / f"{sid}.json", sid)
        verified = trace_verified(trace)
        stage_verified = verified and get(upstream, "sources", "trace", "sha256") == trace_source["sha256"] and trace_source["sha256"] is not None
        # Raw-only measurements do not require replay. Selected/optimized values do.
        flags = {key: tri(get(upstream, "mechanism_flags", key))
                 if key != "selection_removed_gt_pair" or stage_verified else None for key in UPSTREAM_LABELS}
        thresholds = {threshold: {key: tri(get(upstream, "mechanism_flags_by_threshold", threshold, key))
            if not key.startswith(("selected_", "optimized_")) or stage_verified else None for key in THRESHOLD_LABELS}
            for threshold in THRESHOLDS}
        deltas = get(upstream, "stage_deltas", default={}) if stage_verified else {}
        records.append({"id": sid, "gt_face_count": case["gt_face_count"],
            "original_shape_type": case["post_legacy"].get("shape_type"),
            "fit_only_native_valid": case["fit_legacy"].get("native_valid_solid"),
            "raw_metrics": case["raw_legacy"], "raw_topology": raw_topology_class(case["raw_legacy"]),
            "terminal_stage": terminal_stage(trace), "trace_verified": verified,
            "upstream_stage_source_verified": stage_verified,
            "trace_status": trace.get("status", "missing"), "upstream_status": upstream.get("status", "missing"),
            "saved_step_status": saved.get("status", "missing"), "saved_step": saved_step_observations(saved),
            "trace_mechanisms": {key: tri(get(trace, "mechanisms", key)) if verified else None for key in TRACE_LABELS},
            "upstream_mechanisms": flags, "threshold_mechanisms": thresholds,
            "failed_faces": local_failed_faces(case, trace, upstream, manifest["face_identity_contract"]),
            "persistent_trim": persistent_trim_observations(trace),
            "optimization_delta": {"face_xyz_mae": deltas.get("face_xyz_mae_change"),
                "matched_edge_mean_length": get(deltas, "edge_mean_distance_change", "mean")},
            "fit_residuals": fit_observations(trace), "optimization_loss": optimization_loss_observations(trace),
            "unresolved": trace.get("unresolved", []) + upstream.get("unknowns", []),
            "source_files": {"trace": trace_source, "upstream": upstream_source, "saved_step": saved_source}})
    denominator = len(records)
    terminal = {key: {"count": sum(row["terminal_stage"] == key for row in records),
                      "ids": [row["id"] for row in records if row["terminal_stage"] == key]} for key in TERMINAL}
    cross = {top: {stage: [row["id"] for row in records if row["raw_topology"] == top and row["terminal_stage"] == stage]
                   for stage in TERMINAL} for top in ("exact", "inexact", "unknown")}
    faces = {key: [] for key in ("raw_wrong_pair_incident", "raw_local_pair_exact", "unknown")}
    uncountable = []
    for row in records:
        details = row["failed_faces"]["failed_faces"]
        if details is None:
            uncountable.append(row["id"])
        else:
            for face in details:
                faces[face["category"]].append({"id": row["id"], "face_id": face["face_id"]})
    summary = {"version": VERSION, "population": manifest["population"], "intended_failures": denominator,
        "terminal_stages": terminal, "raw_topology_by_terminal": cross,
        "trace_mechanisms": {key: tally((row["id"], row["trace_mechanisms"][key]) for row in records) for key in TRACE_LABELS},
        "persistent_trim_mechanisms": {key: tally((row["id"], row["persistent_trim"]["flags"][key]) for row in records)
                                       for key in PERSISTENT_LABELS},
        "persistent_trim_faces": {"categories": {key: [dict(id=row["id"], **face) for row in records
            for face in (row["persistent_trim"]["failed_faces"] or []) if face["category"] == key]
            for key in ("never_wire", "never_closed_admitted", "closed_admitted_still_failed", "unknown")},
            "uncountable_cad_ids": [row["id"] for row in records if row["persistent_trim"]["failed_faces"] is None],
            "returned_faces_discarding_wire": [{"id": row["id"], "face_id": face} for row in records
                                                for face in row["persistent_trim"]["successful_faces_discarding_wire"]],
            "returned_face_denominator_observed": sum(row["persistent_trim"]["successful_face_count"] for row in records
                                                       if finite(row["persistent_trim"]["successful_face_count"]))},
        "upstream_mechanisms": {key: tally((row["id"], row["upstream_mechanisms"][key]) for row in records) for key in UPSTREAM_LABELS},
        "threshold_mechanisms": {t: {key: tally((row["id"], row["threshold_mechanisms"][t][key]) for row in records)
                                         for key in THRESHOLD_LABELS} for t in THRESHOLDS},
        "failed_face_cooccurrence": {"unit": "failed main-trim faces with decoder IDs; NOT STEP face IDs",
            "counted_faces": sum(map(len, faces.values())),
            "categories": {key: {"face_count": len(items), "cad_count": len({item['id'] for item in items}),
                                  "faces": items, "ids": sorted({item['id'] for item in items})} for key, items in faces.items()},
            "cad_denominator": denominator, "uncountable_cad_ids": uncountable,
            "uncountable_cad_count": len(uncountable)},
        "raw_curve_decomposition": raw_curve_decomposition(cases),
        "optimization_changes": {key: {**tally((row["id"], row["optimization_delta"][key] > 1e-12
                    if finite(row["optimization_delta"][key]) else None) for row in records),
                "improved_ids": [row["id"] for row in records if finite(row["optimization_delta"][key]) and row["optimization_delta"][key] < -1e-12],
                "unchanged_ids": [row["id"] for row in records if finite(row["optimization_delta"][key]) and abs(row["optimization_delta"][key]) <= 1e-12],
                "delta_distribution": distribution(row["optimization_delta"][key] for row in records),
                "positive_means": "optimized minus selected > 1e-12; deterioration of this metric only"}
            for key in ("face_xyz_mae", "matched_edge_mean_length")},
        "saved_step": {"thresholds": {t: {**tally((row["id"], row["saved_step"]["thresholds"][t]["has_near_free_edges"])
                                                   for row in records),
                "observed_pair_sum": sum(row["saved_step"]["thresholds"][t]["observed_pair_count"] for row in records
                                          if finite(row["saved_step"]["thresholds"][t]["observed_pair_count"])),
                "observed_involved_edge_sum": sum(row["saved_step"]["thresholds"][t]["observed_involved_edges"] for row in records
                                                  if finite(row["saved_step"]["thresholds"][t]["observed_involved_edges"]))}
            for t in THRESHOLDS}, "free_edge_distribution": distribution(row["saved_step"]["free_edge_count"] for row in records),
            "note": "Distinct topological free edges are geometrically close; excludes seam/degenerate/same-face. Neither all free edges nor close pairs prove they should be welded."},
        "fit_residuals": {key: {"thresholds": {t: tally((row["id"], row["fit_residuals"][key]["threshold_flags"][t]) for row in records) for t in THRESHOLDS},
            "observed_object_max_distribution": distribution(row["fit_residuals"][key]["observed_max"] for row in records),
            "status_counts": dict(Counter(row["fit_residuals"][key]["status"] for row in records)),
            "expected_entity_count": sum(row["fit_residuals"][key]["expected_entity_count"] for row in records if finite(row["fit_residuals"][key]["expected_entity_count"])),
            "recorded_entity_count": sum(row["fit_residuals"][key]["recorded_entity_count"] for row in records),
            "complete_entity_count": sum(row["fit_residuals"][key]["complete_entity_count"] for row in records)} for key in FIT_LABELS},
        "closed_failed_face_owner_residual": {t: tally((row["id"], closed_failed_owner_residual(row, t)) for row in records) for t in THRESHOLDS},
        "optimization_loss": {key: {"delta_distribution": distribution(get(row, "optimization_loss", "terms", key, "delta") for row in records),
            "loss_down": tally((row["id"], get(row, "optimization_loss", "terms", key, "delta") < -1e-12
                                if finite(get(row, "optimization_loss", "terms", key, "delta")) else None) for row in records),
            "loss_down_gt_up": {metric: tally((row["id"], loss_down_gt_up(row, key, metric)) for row in records)
                                for metric in ("face_xyz_mae", "matched_edge_mean_length")},
            "loss_geometry_cross": {metric: loss_geometry_cross(records, key, metric) for metric in ("face_xyz_mae", "matched_edge_mean_length")}} for key in LOSS_KEYS},
        "diagnostic_status": {name: dict(Counter(row[name] for row in records)) for name in ("trace_status", "upstream_status", "saved_step_status")},
        "verified_trace_count": sum(row["trace_verified"] for row in records), "limitations": LIMITATIONS}
    return records, summary


def make_charts(summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "savefig.facecolor": "white"})
    output.mkdir(parents=True, exist_ok=True)
    denominator = summary["intended_failures"]
    charts = []
    def save(fig, name, title, note):
        fig.tight_layout()
        for extension in ("png", "svg"):
            fig.savefig(output / f"{name}.{extension}", dpi=170)
        plt.close(fig)
        charts.append({"name": name, "title": title, "note": note})
    def bars(data, labels, name, title, note):
        keys = list(labels)
        fig, ax = plt.subplots(figsize=(11, max(3.2, .42 * len(keys) + 1.4)))
        y = np.arange(len(keys))
        positive = [data[key]["positive"] for key in keys]
        unknown = [data[key]["unknown"] for key in keys]
        ax.barh(y, positive, color="#35678a", label="Observed positive")
        ax.barh(y, unknown, left=positive, color="#d1d5db", label="Unknown")
        for index, (value, missing) in enumerate(zip(positive, unknown)):
            ax.text(value + missing + denominator * .008, index, f"{value}/{denominator}" + (f"; ? {missing}" if missing else ""), va="center", fontsize=9)
        ax.set_yticks(y, [labels[key][0] for key in keys])
        ax.invert_yaxis()
        ax.set_xlim(0, denominator * 1.3)
        ax.set_xlabel(f"CAD objects / {denominator} intended failures (overlapping labels)")
        ax.set_title({"persistent_trim": "Main faces still failed after all tolerances / actual returned faces",
                      "ever_observed": "Observed in any main/fallback attempt; may later recover",
                      "upstream_discrete": "Discrete upstream observations; not proven causes"}.get(name, name))
        ax.legend(loc="lower right")
        save(fig, name, title, note)
    fig, ax = plt.subplots(figsize=(10, 4.1))
    keys = list(TERMINAL)
    values = [summary["terminal_stages"][key]["count"] for key in keys]
    ax.barh(range(len(keys)), values, color=["#35678a"] * 4 + ["#a6adb6", "#d1d5db"])
    ax.set_yticks(range(len(keys)), [TERMINAL[key][0] for key in keys])
    ax.invert_yaxis()
    for index, value in enumerate(values):
        ax.text(value + 2, index, str(value), va="center")
    ax.set_xlim(0, max(values + [1]) * 1.18)
    ax.set_xlabel(f"CAD objects / {denominator}; mutually exclusive terminal branches, NOT root causes")
    save(fig, "terminal_stages", "最终阻断分支", "每件只计一个末端分支；完整重放需同时满足原字段一致、STEP DATA 一致和无观测异常。")
    bars(summary["persistent_trim_mechanisms"], PERSISTENT_LABELS, "persistent_trim", "最终裁剪状态", "只看 main 最终失败面；成功面丢弃 wire 只看返回该面的成功尝试。对象标签可重叠，不把闭合候选失败直接解释为 UV 原因。")
    bars(summary["upstream_mechanisms"], UPSTREAM_LABELS, "upstream_discrete", "更早出现的离散结构异常", "面身份保持下的无向 pair 对照；结构异常与后续失败的共现不等于已证明因果。")
    bars(summary["trace_mechanisms"], TRACE_LABELS, "ever_observed", "过程中出现过的记录（补充）", "ever-observed：main 和 fallback 的全部容差尝试。随后可能修复或改变，不能当作持续阻断率。")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    groups = (("reciprocal_excess_disagreement", "raw_new_cross_pair_curve_nearness", "selected_new_cross_pair_curve_nearness", "optimized_new_cross_pair_curve_nearness"),
              ("raw_new_face_grid_nearness", "selected_new_face_grid_nearness", "optimized_new_face_grid_nearness"),
              ("selected_endpoint_gap", "optimized_endpoint_gap", "selected_new_open_edge_endpoint_collapse", "optimized_new_open_edge_endpoint_collapse"))
    for ax, group in zip(axes, groups):
        for key in group:
            counts = [summary["threshold_mechanisms"][t][key]["positive"] for t in THRESHOLDS]
            ax.plot(range(4), counts, marker="o", label=THRESHOLD_LABELS[key][0].replace(" nearness", " near"))
        ax.set_xticks(range(4), THRESHOLDS)
        ax.set_ylim(-5, denominator + 5)
        ax.set_xlabel("Tolerance (normalized XYZ length)")
        ax.legend(fontsize=7, loc="best")
    axes[0].set_ylabel(f"Observed-positive CAD objects / {denominator}")
    save(fig, "threshold_sensitivity", "阈值敏感性", "线图仅显示已观察到的正例；未知数量可在下方标签表查看。near 要求双向各至少 95% 样本落在阈值内；new 要求 GT 在同阈值下不近。曲面项是网格顶点代理。")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    transition = summary["population_comparison"]["fit_only_vs_optimized"]
    matrix = np.asarray([[transition.get(f"fit_{a}_optimized_{b}", 0) for b in ("failed", "valid")] for a in ("failed", "valid")])
    axes[0].imshow(matrix, cmap="Blues")
    for i in range(2):
        for j in range(2):
            axes[0].text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=20)
    axes[0].set_xticks([0, 1], ["Failed", "Native valid"])
    axes[0].set_yticks([0, 1], ["Failed", "Native valid"])
    axes[0].set_xlabel("Optimized (all accepted objects)")
    axes[0].set_ylabel("Fit-only")
    axes[0].set_title(f"Paired native validity: N={int(matrix.sum())}")
    metric_keys = ("face_xyz_mae", "matched_edge_mean_length")
    left = np.zeros(2)
    for label, color, count_key in (("Improved", "#468273", "improved_ids"), ("Unchanged", "#adb5bd", "unchanged_ids"),
                                    ("Worsened", "#c37b52", "positive"), ("Unknown", "#e0e2e5", "unknown")):
        values = [len(summary["optimization_changes"][key][count_key]) if count_key.endswith("ids") else summary["optimization_changes"][key][count_key] for key in metric_keys]
        axes[1].barh([0, 1], values, left=left, color=color, label=label)
        left += values
    axes[1].set_yticks([0, 1], ["GT face XYZ MAE", "Matched curve-to-GT distance (XYZ length)"])
    axes[1].set_xlabel(f"CAD objects / {denominator} failures")
    axes[1].set_title("Selected -> optimized; per-metric changes")
    axes[1].legend(fontsize=8)
    save(fig, "optimization", "优化前后对照", "左图使用全部 2417 件的既有 fit-only 与 optimized 输出；右图只在失败全集看同阶段匹配量的距离变化。某项距离变差不等于整个几何或某个损失项造成失败。")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for key, label in FIT_LABELS.items():
        values = [summary["fit_residuals"][key]["thresholds"][t]["positive"] for t in THRESHOLDS]
        axes[0].plot(THRESHOLDS, values, marker="o", label=label[0])
    axes[0].set_xlabel("Sampled maximum residual > tolerance (XYZ length)")
    axes[0].set_ylabel(f"Observed-positive CAD objects / {denominator}")
    axes[0].set_ylim(-5, denominator + 5)
    axes[0].legend(fontsize=8)
    axes[0].set_title("Fitting / owner-surface residuals (fixed samples)")
    for shift, metric, color, label in ((-.18, "face_xyz_mae", "#35678a", "GT face MAE increases"),
                                      (.18, "matched_edge_mean_length", "#c37b52", "Matched curve-to-GT distance increases")):
        values = [summary["optimization_loss"][key]["loss_down_gt_up"][metric]["positive"] for key in LOSS_KEYS]
        axes[1].barh(np.arange(4) + shift, values, height=.34, color=color, label=label)
        for index, value in enumerate(values):
            axes[1].text(value + 1, index + shift, str(value), va="center", fontsize=8)
    axes[1].set_yticks(range(4), ["Total loss decreases", "Adjacency loss decreases", "Corners loss decreases", "Wire loss decreases"])
    axes[1].set_xlabel(f"Joint observation in CAD objects / {denominator}")
    axes[1].set_xlim(0, denominator * 1.15)
    axes[1].legend(fontsize=7)
    axes[1].set_title("First -> last recorded pre-update loss; GT delta of output")
    save(fig, "geometry_chain", "优化目标、拟合与几何闭合之间的证据", "左图：自拟合残差衡量曲面/曲线对各自输入的拟合；owner 残差衡量不同实体间的一致性。自拟合误差小不能排除边面不一致。面用 16 个网格子采样点、边用 16 个输入点、周期替换后曲线用 9 个原生参数样点到两侧有限拟合曲面；任一采样最大残差超阈值记正例，缺测而无正例记未知，不证明 UV 裁剪。本轮未对成功组计算这些新几何诊断，不能用失败组频率推断因果。右图：优化日志首末项下降且输出相应 GT 误差上升的共现；日志在每次更新前记录，末值不等于最终坐标损失，返回原坐标的分支记未知。匹配曲线误差排除无 GT pair 的预测边。")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    counts = summary["raw_curve_decomposition"]["counts"]
    row_keys = ("first_rows_of_reciprocal_pairs", "second_rows_of_reciprocal_pairs", "unilateral_rows")
    values = [counts.get(key, 0) for key in row_keys]
    axes[0].barh(range(3), values, color=["#35678a", "#7fa6bd", "#c37b52"])
    axes[0].set_yticks(range(3), ["Reciprocal: first half-edge", "Reciprocal: opposite half-edge", "Unilateral curve rows"])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Raw curve rows (reciprocal is normal representation)")
    for i, value in enumerate(values):
        axes[0].text(value, i, f" {value}", va="center")
    axes[0].set_xlim(0, max(values + [1]) * 1.2)
    near = summary["saved_step"]["thresholds"]
    axes[1].plot(THRESHOLDS, [near[t]["positive"] for t in THRESHOLDS], marker="o", color="#35678a")
    axes[1].set_ylim(0, denominator)
    axes[1].set_ylabel(f"CAD objects / {denominator}")
    axes[1].set_xlabel("Length tolerance for close distinct free edges")
    axes[1].set_title("Saved STEP: geometric nearness of free edges")
    save(fig, "curve_sources", "边附近多条曲线的来源不能混合", "左：正常双向半边与单向 rows；跨 pair 近曲线见阈值图。右：保存 STEP 中不同面上的自由拓扑边彼此接近，使用 128 个均匀原生参数样点的点到折线 p95，不是等弧长采样。它不代表一定应该焊合，也不把所有自由边叫作重复。")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.9))
    cross = summary["raw_topology_by_terminal"]
    matrix = np.asarray([[len(cross[top][stage]) for stage in TERMINAL] for top in ("exact", "inexact", "unknown")])
    axes[0].imshow(matrix, cmap="Blues", aspect="auto")
    axes[0].set_xticks(range(len(TERMINAL)), [TERMINAL[key][0] for key in TERMINAL], rotation=35, ha="right", fontsize=8)
    axes[0].set_yticks(range(3), ["Raw pair exact", "Raw pair inexact", "Raw pair unknown"])
    for i in range(3):
        for j in range(len(TERMINAL)):
            axes[0].text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=10)
    axes[0].set_title(f"Raw topology x terminal branch; N={denominator}")
    face_data = summary["failed_face_cooccurrence"]
    keys = ("raw_wrong_pair_incident", "raw_local_pair_exact", "unknown")
    counts = [face_data["categories"][key]["face_count"] for key in keys]
    axes[1].barh(range(3), counts, color=["#c37b52", "#35678a", "#d1d5db"])
    axes[1].set_yticks(range(3), ["Incident to raw FP/FN", "Local raw pair adjacency exact", "Unknown adjacency"])
    axes[1].invert_yaxis()
    for i, value in enumerate(counts):
        axes[1].text(value + .5, i, str(value), va="center")
    axes[1].set_xlim(0, max(counts + [1]) * 1.2)
    axes[1].set_xlabel(f"Countable failed main-trim faces: {sum(counts)}")
    axes[1].set_title(f"Uncountable CAD objects: {face_data['uncountable_cad_count']}")
    save(fig, "structure_cooccurrence", "上游结构与下游失败相接", "左按对象，右按 main 最终失败面；对象可能同时含两类失败面。局部 pair 正确只指无向邻接，不证明方向、几何或多重边正确；这是共现而非因果。")
    comparison = summary["population_comparison"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    labels = list(CHEAP_LABELS)
    for g, color, shift in (("native_valid", "#468273", -.18), ("native_failed", "#c37b52", .18)):
        rates = [100 * comparison["groups"][g]["flags"][key]["positive_fraction_of_intended"] for key in labels]
        axes[0].barh(np.arange(len(labels)) + shift, rates, height=.34, color=color, label=f"{g}: N={comparison['groups'][g]['denominator']}")
        rates = [100 * comparison["face_standardized_flags"][key]["rates"][g] if comparison["face_standardized_flags"][key]["rates"][g] is not None else float("nan") for key in labels]
        axes[1].barh(np.arange(len(labels)) + shift, rates, height=.34, color=color, label=g)
    for ax in axes:
        ax.set_yticks(range(len(labels)), [CHEAP_LABELS[key][0] for key in labels])
        ax.invert_yaxis()
        ax.set_xlabel("Objects with observation (%)")
        ax.legend(fontsize=8)
    axes[0].set_title("Existing raw metrics: unadjusted")
    axes[1].set_title("Same pooled weights in face-count bins")
    save(fig, "success_comparison", "已有成功组对照", "仅使用原 CSV，不重新诊断成功组；右图按 1–9 / 10–19 / 20–30 面分层，使用两组共有层的 pooled 面数权重。它控制粗粒度面数构成，不能排除其它混杂。")
    return charts


def page_payload(records, summary, atlas_ids):
    groups = []
    def add(label, english, result, scope):
        groups.append({"label": label, "english": english, "scope": scope, **result})
    for key, result in summary["terminal_stages"].items():
        add(TERMINAL[key][1], TERMINAL[key][0], {"positive": result["count"], "unknown": 0,
            "denominator": summary["intended_failures"], "ids": {"positive": result["ids"], "unknown": []}}, "末端分支（互斥）")
    for name, labels, scope in (("persistent_trim_mechanisms", PERSISTENT_LABELS, "最终 main 裁剪状态"),
                                ("upstream_mechanisms", UPSTREAM_LABELS, "上游离散结构"),
                                ("trace_mechanisms", TRACE_LABELS, "ever-observed：全部容差 / main+fallback")):
        for key, result in summary[name].items():
            add(labels[key][1], labels[key][0], result, scope)
    for threshold, flags in summary["threshold_mechanisms"].items():
        for key, result in flags.items():
            add(THRESHOLD_LABELS[key][1], THRESHOLD_LABELS[key][0], result, f"长度阈值 {threshold}")
        add("STEP 不同面自由边近重合", "Saved STEP near free edges", summary["saved_step"]["thresholds"][threshold], f"长度阈值 {threshold}")
        for key, label in FIT_LABELS.items():
            add(label[1], label[0], summary["fit_residuals"][key]["thresholds"][threshold], f"拟合最大残差 > {threshold}")
        add("闭合候选仍失败的同一面存在拟合边面残差", "Closed-admitted failed face has owner residual", summary["closed_failed_face_owner_residual"][threshold], f"同面共现；长度 > {threshold}")
    for key in LOSS_KEYS:
        for metric, result in summary["optimization_loss"][key]["loss_down_gt_up"].items():
            add(f"{key} 下降且 {metric} 上升", "Optimization loss down, GT error up", result, "首末日志 vs 输出；共现")
    for row in records:
        row["has_existing_atlas"] = row["id"] in atlas_ids
    return {"records": records, "groups": groups, "summary": summary}


PLAIN_CHART_GUIDES = {
    "terminal_stages": ("最后卡在哪一步？", "有的零件连面都没做齐，有的面做出来了却拼不成实体。这张图只说程序停在哪里，不等于已经找到了最早的原因。"),
    "persistent_trim": ("边没围成圈，还是围成圈后仍裁不出面？", "重点看：始终没有可用闭环的有 73 件；有了闭环却仍裁不出面的有 216 件；另有 143 件在做面时丢弃了一圈没闭合的边。这几组可以重叠。"),
    "upstream_discrete": ("模型一开始就把哪些面连错了？", "多连：本来不相邻的两个面被连上。漏连：本来相邻的两个面没有连上。它们在后处理开始前就已经存在。"),
    "ever_observed": ("修复过程中还出过什么问题？", "这些问题曾经发生过，但有些后来修好了。不要把这里的数量当成最终仍未解决的数量。"),
    "threshold_sensitivity": ("对“足够接近”的要求变严或变松，统计会怎么变？", "距离多小才算两条线靠在一起，需要选一个标准。这张图把几个标准都列出来，避免结论只依赖某一个数字。"),
    "optimization": ("后处理优化救回了多少零件，又弄坏了多少？", "优化把 998 件从失败变成成功，也让 8 件从成功变失败。所以不能简单说优化没用，但需要查清它什么时候会把形状改坏。"),
    "geometry_chain": ("单独的面和边看着挺准，拼起来能对上吗？", "左图比较两件事：每个面、每条边有没有拟合好自己，以及边有没有贴在所属的面上。右图显示：程序的优化分数变好了，结果却可能离正确零件更远。"),
    "curve_sources": ("为什么同一位置会看到好几条边？", "有些是原模型正常输出的两份半边，有些是最后留下的两条独立边。看起来都是多条线，来源却不一样。"),
    "structure_cooccurrence": ("知道哪两个面该相连，就够了吗？", "还不够。很多失败零件的面连接关系是对的，但边的形状、位置或后续裁剪仍然有问题。"),
    "success_comparison": ("成功零件和失败零件，哪些地方不同？", "失败零件更常出现多连面、漏连面。右图粗略考虑了面数量的差别；它说明两者有关，还不能证明一种错误解释了全部失败。"),
}


def plain_language_overview(summary):
    persistent = summary["persistent_trim_mechanisms"]
    upstream = summary["upstream_mechanisms"]
    count = lambda key: persistent[key]["positive"]
    owner = summary["fit_residuals"]["curve_owner"]["thresholds"]["0.01"]["positive"]
    nearby = summary["saved_step"]["thresholds"]["0.01"]["positive"]
    extra = upstream["raw_extra_gt_pair"]["positive"]
    missing = upstream["raw_missing_gt_pair"]["positive"]
    findings = [
        ("边没能围成一圈", f"{count('never_closed_admitted')} 件",
         "至少有一个面的边界，经过修复仍没有形成可用的闭环。就像沿着轮廓剪纸，轮廓还留着口子，程序没法据此剪出这个面。"),
        ("圈闭合了，面仍然裁不出来", f"{count('closed_admitted_still_failed')} 件",
         "首尾接上只是第一步。比如线交叉了，或者线没有贴在这个曲面上，闭合了也未必能裁出面。我们已确认这批有闭环仍失败，但还没把每件具体是哪一种情况分清。"),
        ("难处理的一圈边，被程序丢掉了", f"{count('final_success_discarded_wire')} 件",
         "程序舍弃没闭合的环，再用剩下的边界做面。面虽然做出来了，却没有保留全部输入边界。如果丢的是孔的边界，孔就可能被填平；目前还不能说这些件都丢了孔。"),
        ("边和它所属的面没有贴合", f"{owner} 件 · 距离标准 0.01",
         "单看面和线可能都挺像，但放在一起，线没有落在它应该属于的曲面上。就像两个零件尺寸各差一点，摆到一起就对不上。这是最终测到的偏差，还不能一概归责于模型或后处理。"),
        ("两条边挨得很近，在 CAD 里却仍各算一条", f"{nearby} 件 · 距离标准 0.01",
         "这可能解释你看到的“同一个位置有好几条线”：它们靠近，却没有共享同一条 CAD 边。不过靠得近也可能是正常薄壁，不能看见两条就直接合并。"),
        ("一开始就把面连错了", f"多连 {extra} 件 · 漏连 {missing} 件",
         "本来不相连的面被预测成相连，或者本来该相连的面没连上。后处理拿到的拼装关系一开始就有错。但多数失败零件的面连接关系其实是对的，问题不止在这里。"),
    ]
    items = "".join(f'<li><strong>{html.escape(title)}</strong><span class="finding-count">{html.escape(number)}</span><p>{html.escape(explanation)}</p></li>' for title, number, explanation in findings)
    return ('<section class="overview"><h2>先用零件本身来理解</h2>'
            '<p>可以把整个过程想成：先画出面和边，再用边裁出每个面的轮廓，最后把面拼成一个没有破口的实体。失败主要出在下面这些地方。</p>'
            f'<ol class="findings">{items}</ol>'
            '<p><strong>现在更值得追的是：这些面和边能不能互相对上，围出正确的边界，最后拼到一起。</strong>仅仅知道哪两个面相邻，或者把线头接上，都还不够。</p>'
            '<p class="muted">同一零件可能有好几个问题，数量不能直接相加。两项距离统计使用归一化坐标下的 0.01，不是毫米；它们是排查线索，不等于每件失败的唯一原因。</p></section>')


def write_page(root, payload, charts):
    page = root / "page"
    page.mkdir(parents=True, exist_ok=True)
    (page / "data.js").write_text("window.CENSUS_DATA=" + json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + ";\n")
    figures = "\n".join(f'<section><h2>{html.escape(PLAIN_CHART_GUIDES.get(chart["name"], (chart["title"], chart["note"]))[0])}</h2><p>{html.escape(PLAIN_CHART_GUIDES.get(chart["name"], (chart["title"], chart["note"]))[1])}</p><details><summary>查看具体计算方式</summary><p>{html.escape(chart["note"])}</p></details><a href="../charts/{chart["name"]}.svg"><img loading="lazy" src="../charts/{chart["name"]}.png" alt="{html.escape(chart["title"])}"></a></section>' for chart in charts)
    overview = plain_language_overview(payload["summary"])
    limitations = "".join(f"<li>{html.escape(note)}</li>" for note in LIMITATIONS)
    (page / "index.html").write_text('''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>原版 VAE 全失败统计</title><style>
:root{color-scheme:light dark;font-family:system-ui,sans-serif;background:#f5f6f8;color:#202d38}body{max-width:1200px;margin:auto;padding:24px}h1{font-size:26px}h2{font-size:21px}p,li{line-height:1.7}section{background:white;border:1px solid #dce1e5;border-radius:12px;padding:20px;margin:20px 0}img{width:100%;height:auto}button,input{font:inherit;border:1px solid #bac5ce;background:transparent;color:inherit;border-radius:6px;padding:6px 10px}button{cursor:pointer}a{color:#205e8a}table{width:100%;border-collapse:collapse;font-size:14px}th,td{text-align:left;border-bottom:1px solid #dce1e5;padding:9px}th{position:sticky;top:0;background:#edf1f5}.scroll{overflow:auto;max-height:580px}#ids{display:flex;flex-wrap:wrap;gap:6px}#case{white-space:pre-wrap;overflow-wrap:anywhere;max-height:500px;overflow:auto;background:#f4f6f8;padding:14px;font-size:12px}.muted{color:#5b6874}#counts{font-size:18px;font-weight:600}@media(prefers-color-scheme:dark){:root{background:#111820;color:#e2e7eb}section{background:#18232d;border-color:#354550}th,#case{background:#222f3b}a{color:#8fc6ed}td{border-color:#354550}.muted{color:#acbac5}}@media(max-width:600px){body{padding:10px}section{padding:12px}h1{font-size:22px}table{min-width:650px}}
.overview{border-left:4px solid #468273}.findings{padding-left:24px}.findings>li{padding:10px 0}.findings p{margin:6px 0}.finding-count{display:block;color:#5b6874;font-size:14px}summary{cursor:pointer;line-height:1.7}details p{font-size:14px}@media(prefers-color-scheme:dark){.finding-count{color:#acbac5}}
</style><h1>原版 VAE：零件为什么没做成功？</h1><p id="counts"></p><p>先看下面的通俗说明。需要核对数字时，再往下看图表和逐件记录。</p>''' + overview + '''<section><details><summary>这次统计覆盖了哪些数据，有哪些限制？</summary><ul>''' + limitations + '''</ul></details></section>''' + figures + '''<section><h2>查找具体零件</h2><p>下面保留诊断工具的原始术语，供查证使用。点击数量查看对应零件编号；“未知”表示这一项还判断不了。原先的 50 件可以打开已有三维页面。</p><input id="search" aria-label="筛选标签" placeholder="搜索问题或诊断字段"><div class="scroll"><table><thead><tr><th>检查位置</th><th>检查到的问题</th><th>查到的问题件数 / 全部失败件数</th><th>还判断不了</th></tr></thead><tbody id="labels"></tbody></table></div></section><section><h2 id="selection">选择一个问题</h2><div id="ids"></div><p id="caseLinks"></p><pre id="case">点击零件编号查看记录。</pre></section><p class="muted">数据：<a href="../summary.json">summary.json</a> · <a href="../cases.json">cases.json</a> · <a href="../cases.csv">cases.csv</a></p><script src="data.js"></script><script src="app.js"></script></html>''')
    (page / "app.js").write_text('''"use strict";(()=>{const data=window.CENSUS_DATA;const byId=new Map(data.records.map(r=>[r.id,r]));const p=data.summary.population;
document.getElementById("counts").textContent=`测试了 ${p.accepted_test} 个零件：${p.native_valid} 个做成有效实体，${p.native_failed} 个失败。本页分析全部失败件。另有 ${p.contract_unsupported} 件不符合本次输入要求，未进入这批测试。`;
function showCase(id){const row=byId.get(id);if(!row)return;document.getElementById("case").textContent=JSON.stringify(row,null,2);const links=document.getElementById("caseLinks");links.replaceChildren();if(row.has_existing_atlas){const a=document.createElement("a");a.href=`../../legacy_failures_20261011/page/index.html#${id}`;a.textContent=`${id}：打开既有 50 件 atlas`;a.target="_blank";links.append(a);}else links.textContent=`${id}：本页提供数值记录，未新增 3D 页面。`;}
function showIds(group,state){document.getElementById("selection").textContent=`${group.scope} · ${group.label} · ${state==="positive"?"正例":"未知"}`;const box=document.getElementById("ids");box.replaceChildren();for(const id of group.ids[state]){const b=document.createElement("button");b.textContent=id;b.onclick=()=>showCase(id);box.append(b);}if(!box.children.length)box.textContent="无对象。";document.getElementById("selection").scrollIntoView({behavior:"smooth",block:"start"});}
function render(){const q=document.getElementById("search").value.toLowerCase();const body=document.getElementById("labels");body.replaceChildren();for(const group of data.groups){if(!`${group.label} ${group.english} ${group.scope}`.toLowerCase().includes(q))continue;const tr=document.createElement("tr");for(const text of [group.scope,group.label]){const td=document.createElement("td");td.textContent=text;tr.append(td);}for(const state of ["positive","unknown"]){const td=document.createElement("td");const b=document.createElement("button");b.textContent=state==="positive"?`${group.positive} / ${group.denominator}`:String(group.unknown);b.onclick=()=>showIds(group,state);td.append(b);tr.append(td);}body.append(tr);}}document.getElementById("search").addEventListener("input",render);render();})();
''')


def write_outputs(root, manifest_path, manifest, records, summary):
    summary["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    summary["manifest_sha256"] = file_sha256(manifest_path)
    summary["script_sha256"] = file_sha256(Path(__file__))
    summary["source_csvs"] = manifest["sources"]
    summary["definitions"] = {"trace_mechanisms": "ever_observed across all main/fallback tolerances; not persistent blocking",
        "persistent_trim_mechanisms": "main final failed faces, plus open-wire discard in a returned face's actual successful attempt",
        "threshold_nearness": "both directed 64 arc-length sample-to-polyline sets have >=95% of distances <= threshold; new requires GT not near at same threshold",
        "face_nearness": "16x16 nearest sampled grid vertices, not actual surface overlap",
        "normal_reciprocal": "opposite directed half-edges for one unordered face pair; by design, not duplicates",
        "saved_step_nearness": "128 uniform native-parameter points; symmetric p95 point-to-polyline distance, not uniform arc length",
        "endpoint_counts": "endpoint occurrences per incident face, not unique vertices/edges; multiple nearby slots need not be multiple different edges",
        "matched_edge_delta": "only edges with the same GT pair; predicted FP curves are excluded from this matched delta",
        "fitting_residuals": "fixed samples: faces 16 subsampled input grid points, curves 16 input points, each fitted curve owner 9 uniform native-parameter points after periodic replacement. Positive if ANY measured residual maximum exceeds threshold; incomplete with no positive is unknown. Object distributions summarize observed maxima, not missing projections.",
        "optimization_loss": "First/last logged values precede optimizer updates; last log need not equal loss at returned geometry. Loss-down/GT-up event excludes returned-original branches as unknown.",
        "evidence_gate": "post-selection trajectory flags need status ok, no instrumentation errors, same recorded outcome, identical STEP DATA, and upstream source trace hash equal current trace hash"}
    charts = make_charts(summary, root / "charts")
    summary["charts"] = charts
    atlas_path = root.parent / "legacy_failures_20261011" / "selected_cases.json"
    atlas_ids = {case["id"] for case in json.loads(atlas_path.read_text())["cases"]} if atlas_path.is_file() else set()
    payload = page_payload(records, summary, atlas_ids)
    (root / "cases.json").write_text(json.dumps({"version": VERSION, "manifest_sha256": summary["manifest_sha256"], "cases": records}, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    (root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    fields = ["id", "gt_face_count", "original_shape_type", "fit_only_native_valid", "raw_topology", "terminal_stage", "trace_verified", "trace_status", "upstream_status", "saved_step_status"]
    mechanism_groups = (("persistent", "persistent_trim", PERSISTENT_LABELS), ("ever", "trace_mechanisms", TRACE_LABELS), ("upstream", "upstream_mechanisms", UPSTREAM_LABELS))
    extra = [f"{prefix}.{key}" for prefix, _, labels in mechanism_groups for key in labels]
    with (root / "cases.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields + extra, lineterminator="\n")
        writer.writeheader()
        for row in records:
            flat = {key: row[key] for key in fields}
            for prefix, source, labels in mechanism_groups:
                values = row[source]["flags"] if prefix == "persistent" else row[source]
                flat.update({f"{prefix}.{key}": "unknown" if values[key] is None else str(values[key]).lower() for key in labels})
            writer.writerow(flat)
    write_page(root, payload, charts)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    ids = [case["id"] for case in manifest["cases"]]
    if (len(ids) != 369 or len(set(ids)) != 369 or manifest["population"]["accepted_test"] != 2417
            or any(case["post_legacy"].get("native_valid_solid") is not False for case in manifest["cases"])):
        raise ValueError("This report requires the frozen 369 failures from 2417 accepted objects")
    records, summary = aggregate(manifest, root)
    summary["population_comparison"] = population_comparison(manifest)
    write_outputs(root, manifest_path, manifest, records, summary)
    print(json.dumps({"intended_failures": len(records), "verified_traces": summary["verified_trace_count"],
                      "diagnostic_status": summary["diagnostic_status"], "page": str(root / "page/index.html")}, indent=2))


if __name__ == "__main__":
    main()
