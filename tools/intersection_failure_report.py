"""Compare recorded failure trajectories without rerunning or repairing geometry.

The main denominator is the 304 failed intersection outputs. The three paired
cohorts have independent denominators, and missing evidence never becomes an
improvement. Historical diagnostics are read in place with their own hashes.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path

from tools.failure_census_contract import case_method, case_path, case_record
from tools import legacy_failure_census_report as legacy


VERSION = "intersection_failure_report_v1"
THRESHOLDS = legacy.THRESHOLDS
COHORTS = {"persistent": "旧新都失败", "regressed": "原成功、新失败", "rescued": "原失败、新成功"}
EXPECTED = {"persistent": 262, "regressed": 42, "rescued": 107}
TERMINALS = {
    "native_valid": "已得到合法实体", "trim_gate": "面没做齐，未进入实体拼接",
    "solid_construction": "进入拼接后仍未得到实体", "saved_step_rejected": "拼出了候选实体，但保存检查没通过",
    "native_readback_invalid": "保存检查通过，重读却不是合法实体", "unresolved": "重放一致，但末端分支未细分",
    "unknown": "原结果已知，失败路径暂时无法核验",
}
CONTEXT_ERRORS = {
    "BRepCheck_InvalidCurveOnSurface": "边在所属面的参数表示不合法",
    "BRepCheck_InvalidSameParameterFlag": "边的三维与面上参数对应检查未通过",
    "BRepCheck_SelfIntersectingWire": "候选边界环被检查为自相交",
    "BRepCheck_IntersectingWires": "候选面中不同边界环被检查为相交",
    "BRepCheck_UnorientableShape": "候选面的方向无法一致确定",
    "BRepCheck_NotClosed": "候选边界在所属面的检查中仍不闭合",
    "BRepCheck_BadOrientationOfSubshape": "候选面中部分边界的方向检查不一致",
    "BRepCheck_InvalidImbricationOfWires": "候选面中边界环的内外嵌套检查不合法",
    "BRepCheck_InvalidPointOnCurve": "候选顶点与所属曲线的对应检查不合法",
}
EARLIEST_STAGES = (
    ("raw", "网络输出时已观察到异常", ("raw_extra_gt_pair", "raw_missing_gt_pair", "raw_duplicate_directed_pair", "raw_same_face_pair", "face_count_mismatch", "reciprocal_excess_disagreement", "raw_new_cross_pair_curve_nearness", "raw_new_face_grid_nearness")),
    ("selection", "半边选择时首次观察到异常", ("selection_removed_gt_pair",)),
    ("selected_geometry", "选择后的几何首先暴露偏差", ("face_without_boundary", "selected_endpoint_gap", "selected_new_open_edge_endpoint_collapse")),
    ("optimization", "优化阶段首次观察到偏差", ("optimization_face_away", "optimization_edge_away", "optimized_endpoint_gap", "optimized_new_open_edge_endpoint_collapse")),
    ("independent_fitting", "独立拟合阶段首次观察到偏差", ("fit_face_residual", "fit_curve_residual", "owner_independent")),
    ("periodic_replacement", "周期替换后首次观察到贴合偏差", ("owner_after",)),
    ("trim", "裁剪阶段首次观察到阻断或丢边界", ("never_wire", "never_closed_admitted", "closed_admitted_still_failed", "final_success_discarded_wire")),
    ("sewing", "拼接/保存后首次观察到不共边", ("sewing_free_edges", "saved_near_free_edges")),
)

# Plain language is kept separate from exact recorded predicates. These are
# overlapping observations, not a mutually exclusive list of proven causes.
OBSERVATIONS = [
    ("face_without_boundary", "有面没有分到任何边，直接跳过裁剪", "Face has no assigned boundary", "完整核验的main裁剪中，最终失败面在选后connectivity两列face端点中度为0，且没有该面的裁剪尝试。", False),
    ("never_wire", "边没有拼成环", "No wire assembled", "最终失败面在全部已记录容差尝试中，wire 组装均未返回。", False),
    ("never_closed_admitted", "拼出了边界，但没有闭合环被接纳", "No closed wire admitted", "最终失败面有 wire 返回，但所有尝试均未接纳闭合 wire。", False),
    ("closed_admitted_still_failed", "至少有一个闭合环，面仍裁不出来", "Closed wire, trim still failed", "最终失败面至少一次接纳闭合 wire；不保证其他候选环也闭合。", False),
    ("strict_closed_candidate_failed", "某次所有候选环都闭合并接纳，面仍失败", "All wires admitted, trim failed", "同一个最终失败面的某次尝试：非空候选 wire 全部 repaired_closed=True 且 added=True，但 trim_returned=False。", False),
    ("final_success_discarded_wire", "面做出来了，但舍弃了部分边界", "Returned face dropped a wire", "最终返回面的成功尝试仍有 wire.skip_reason；不推断丢的是孔还是外边界。", False),
    ("raw_extra_gt_pair", "网络多连了不该相连的面", "Raw extra face pair", "raw 无向 face pair FP > 0；GT 仅用于离线诊断。", False),
    ("raw_missing_gt_pair", "网络漏连了应该相连的面", "Raw missing face pair", "raw 无向 face pair FN > 0。", False),
    ("raw_duplicate_directed_pair", "同一方向的连接被输出了多份", "Repeated directed pair", "raw 同方向 pair 重复；正常 i→j/j→i 两份半边不算重复。", False),
    ("raw_same_face_pair", "出现了一个面连向自己的边", "Self face pair", "raw 存在 i=j 的 pair。", False),
    ("face_count_mismatch", "网络输出的面数与 GT 不同", "Raw face count differs", "raw pred_face_count != gt_face_count；本轮未执行面 latent 去重。", False),
    ("selection_removed_gt_pair", "选半边时丢掉了原本正确的连接", "Selection removed a true pair", "选择前存在、选择后缺失的 GT 无向 pair；要求对应重放及 stage geometry 来源一致。", False),
    ("reciprocal_excess_disagreement", "同一条边的两份半边比 GT 更对不上", "Reciprocal excess disagreement", "互反半边的不一致超过 GT 基准与当前长度阈值，沿用 upstream 的固定采样判据。", True),
    ("raw_new_cross_pair_curve_nearness", "不同连接的边挤得很近，GT 中没有", "Raw extra curve nearness", "不同无向 pair 的预测曲线双向至少95%采样接近，GT 对应曲线不接近；不自动视作重复边。", True),
    ("raw_new_face_grid_nearness", "两个面挤得很近，GT 中没有", "Raw extra face-grid nearness", "预测16×16面采样网格接近而对应GT不接近；不是曲面相交或重复面的证明。", True),
    ("selected_endpoint_gap", "选边后，相邻边端点仍对不上", "Selected endpoint gap", "GT 开/闭边类型约束下的端点最近邻残差代理超过阈值；不是完整闭环判据。", True),
    ("optimized_endpoint_gap", "优化后，相邻边端点仍对不上", "Optimized endpoint gap", "与上一行同一测量，作用于返回的优化后坐标。", True),
    ("selected_new_open_edge_endpoint_collapse", "选边后，原本分开的边首尾挤到一起", "Selected endpoints collapse", "GT 开边的首末点在预测中接近而 GT 不接近；不称为合法闭环。", True),
    ("optimized_new_open_edge_endpoint_collapse", "优化后，原本分开的边首尾挤到一起", "Optimized endpoints collapse", "相同端点接近判据，作用于返回的优化后坐标。", True),
    ("fit_face_residual", "拟合的曲面偏离了网络给出的面点", "Surface fitting residual", "16个固定输入采样点到拟合有限曲面的最大距离超过阈值。", True),
    ("fit_curve_residual", "拟合的曲线偏离了网络给出的边点", "Curve fitting residual", "16个原输入边点到拟合有限曲线的最大距离超过阈值。", True),
    ("owner_independent", "独立拟合后，边贴不到它所属的面上", "Independent curve-to-owner gap", "独立拟合曲线的9个固定参数采样点到两侧拟合曲面的距离至少一个超过阈值。", True),
    ("owner_after", "周期替换后，边仍贴不到所属面上", "After-periodic curve-to-owner gap", "周期曲线替换后采用同一采样定义；不保证替换前后采样是相同空间点。", True),
    ("closed_failed_owner_gap", "裁不出的面同时存在边面贴合偏差", "Closed-wire failure + owner gap", "至少接纳闭环仍失败的同一个 decoder 面，有所属曲线到拟合面的残差超阈值；是共现。", True),
    ("optimization_face_away", "优化后的面反而离 GT 更远", "Optimization moves faces away", "返回优化坐标的 face_xyz_mae 比选择后增加 >1e-12。", False),
    ("optimization_edge_away", "优化后的边反而离 GT 更远", "Optimization moves edges away", "已匹配边的平均长度距离比选择后增加 >1e-12；配对缺失为未知。", False),
    ("sewing_free_edges", "拼接时仍有只属于一个面的边", "Sewing free edges", "至少一次原 solid 拼接尝试报告自由边；不要求该次是最终尝试。", False),
    ("saved_near_free_edges", "保存后仍有靠得很近、却没共用的边", "Saved STEP near free edges", "原STEP中不同面上的拓扑自由边，双向p95几何距离接近；排除seam/退化/同面；不证明应该焊合。", True),
]


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def fingerprint(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def owner_residual_stages(trace: dict) -> dict:
    """Same curve/owner identities, measured before and after replacement.

    Complete-negative assertions require all 2E owners, unique identities and
    complete projections. Positive residuals survive partial sampling.
    """
    fit = trace.get("fit_residuals", {}) if legacy.trace_verified(trace) else {}
    count = legacy.get(trace, "fitting", "num_curves")
    expected = 2 * count if type(count) is int and count >= 0 else None
    results, indexed = {}, {}
    for stage in ("independent_curve", "after_periodic_replacement"):
        rows = []
        for edge in fit.get("curve_owner", []):
            for owner in edge.get(stage, []):
                rows.append({"edge_index": edge.get("edge_index"), **owner})
        keys = [(row.get("edge_index"), row.get("face_index")) for row in rows]
        identities_valid = (all(type(e) is int and type(f) is int for e, f in keys)
                            and len(set(keys)) == len(keys))
        def complete_row(row):
            return (row.get("status") == "ok" and row.get("count", 0) > 0
                    and row.get("count") == row.get("finite_count")
                    and legacy.finite(row.get("max")) and not row.get("errors"))
        complete = (fit.get("status") in ("ok", "partial") and expected is not None
                    and len(rows) == expected and bool(rows) and identities_valid
                    and all(complete_row(row) for row in rows))
        maxima = [row["max"] for row in rows if legacy.finite(row.get("max"))] if identities_valid else []
        indexed[stage] = {key: row for key, row in zip(keys, rows) if complete_row(row)} if identities_valid else {}
        results[stage] = {
            "status": "ok" if complete else "partial" if maxima else "unknown",
            "expected_entity_count": expected, "recorded_entity_count": len(rows),
            "complete_entity_count": len(indexed[stage]), "identities_valid": identities_valid,
            "observed_max": max(maxima) if maxima else None,
            "threshold_flags": {t: True if any(v > float(t) for v in maxima) else False if complete else None for t in THRESHOLDS},
        }
    before, after = (indexed[k] for k in ("independent_curve", "after_periodic_replacement"))
    paired = [{"edge_index": key[0], "face_index": key[1], "before_max": before[key]["max"],
               "after_max": after[key]["max"], "delta_max": after[key]["max"] - before[key]["max"]}
              for key in sorted(before.keys() & after.keys())]
    results["matched_owners"] = paired
    complete = all(results[k]["status"] == "ok" for k in ("independent_curve", "after_periodic_replacement")) and before.keys() == after.keys()
    results["object_max_delta"] = (results["after_periodic_replacement"]["observed_max"] - results["independent_curve"]["observed_max"]) if complete else None
    results["comparison_complete"] = complete
    return results


def failed_face_context_errors(trace: dict) -> dict:
    """Deduplicate by final-failed decoder face, not tolerance/attempt/STEP ID."""
    unknown = {"status": "unknown", "faces": None}
    if not legacy.trace_verified(trace):
        return unknown
    runs = [run for run in trace.get("trim_runs", []) if run.get("context") == "main"]
    if len(runs) != 1 or not isinstance(runs[0].get("failed_face_indices"), list):
        return unknown
    run, rows = runs[0], []
    for face in sorted(set(run["failed_face_indices"])):
        attempts = [a for a in run.get("attempts", []) if a.get("face_index") == face]
        contextual, direct, inspected, incomplete = set(), set(), 0, not bool(attempts)
        for attempt in attempts:
            detail = attempt.get("trim_detail", {})
            check = detail.get("final_brepcheck")
            if not isinstance(check, dict):
                # No face candidate is built when wire assembly/admission fails.
                # This is no inspection opportunity, not a clean UV check.
                if attempt.get("wire_returned") is not False and detail.get("outcome") not in ("no_admitted_wires", "no_wires", "no_wire"):
                    incomplete = True
                continue
            if not isinstance(check.get("with_errors"), list):
                incomplete = True
                continue
            inspected += 1
            incomplete |= bool(check.get("inspection_errors"))
            for item in check["with_errors"]:
                direct.update(code for code in item.get("statuses", []) if code != "BRepCheck_NoError")
                for context in item.get("contexts", []):
                    contextual.update(code for code in context.get("statuses", []) if code != "BRepCheck_NoError")
        rows.append({"face_id": face, "codes": sorted(contextual | direct), "context_codes": sorted(contextual), "direct_codes": sorted(direct),
                     "inspected_attempts": inspected, "inspection_complete": inspected > 0 and not incomplete,
                     "inspection_state": "partial" if inspected and incomplete else "inspected" if inspected else "unknown" if incomplete else "no_candidate_inspected"})
    return {"status": "ok" if all(r["inspection_state"] != "unknown" for r in rows) else "partial", "faces": rows}


def context_flag(row: dict, code: str):
    faces = row["failed_face_context"]["faces"]
    if faces is None:
        return None
    if any(code in f["codes"] for f in faces):
        return True
    return False if all(f["inspection_complete"] for f in faces) else None


def faces_without_boundary(trace: dict) -> dict:
    unknown = {"flag": None, "face_ids": None}
    if not legacy.trace_verified(trace):
        return unknown
    runs = [r for r in trace.get("trim_runs", []) if r.get("context") == "main"]
    if len(runs) != 1:
        return unknown
    run = runs[0]
    count, connectivity, failed = run.get("num_faces"), run.get("connectivity"), run.get("failed_face_indices")
    if type(count) is not int or not isinstance(connectivity, list) or not isinstance(failed, list):
        return unknown
    if any(type(i) is not int or not 0 <= i < count for i in failed):
        return unknown
    if any(not isinstance(c, list) or len(c) != 3 or any(type(i) is not int or not 0 <= i < count for i in c[1:]) for c in connectivity):
        return unknown
    incident = {i for c in connectivity for i in c[1:]}
    unassigned = sorted(set(failed) - incident)
    attempted = {a.get("face_index") for a in run.get("attempts", [])}
    if set(unassigned) & attempted:
        return unknown  # Inconsistent trace, not a confirmed skip branch.
    return {"flag": bool(unassigned), "face_ids": unassigned}


def observations_at(row: dict, threshold: str) -> dict:
    if threshold not in THRESHOLDS:
        raise ValueError(f"Unknown threshold: {threshold}")
    flags = {**row["persistent_trim"]["flags"], **row["upstream_mechanisms"], **row["threshold_mechanisms"][threshold]}
    flags["face_without_boundary"] = row["faces_without_boundary"]["flag"]
    flags["face_count_mismatch"] = legacy.cheap_flags(row["raw_metrics"])["face_count_mismatch"]
    flags.update(
        fit_face_residual=row["fit_residuals"]["faces"]["threshold_flags"][threshold],
        fit_curve_residual=row["fit_residuals"]["curves"]["threshold_flags"][threshold],
        owner_independent=row["owner_stages"]["independent_curve"]["threshold_flags"][threshold],
        owner_after=row["owner_stages"]["after_periodic_replacement"]["threshold_flags"][threshold],
        closed_failed_owner_gap=legacy.closed_failed_owner_residual(row, threshold),
        sewing_free_edges=row["trace_mechanisms"]["sewing_free_edges"],
        saved_near_free_edges=row["saved_step"]["thresholds"][threshold]["has_near_free_edges"],
    )
    for key, metric in (("optimization_face_away", "face_xyz_mae"), ("optimization_edge_away", "matched_edge_mean_length")):
        delta = row["optimization_delta"][metric]
        flags[key] = delta > 1e-12 if legacy.finite(delta) else None
    return {key: flags.get(key) for key, *_ in OBSERVATIONS}


def paired_transitions(pairs: list[dict], key: str, threshold: str) -> dict:
    names = ("present_to_absent", "absent_to_present", "both_present", "both_absent", "unknown")
    ids = {name: [] for name in names}
    for pair in pairs:
        before = pair["legacy"]["flags"][threshold].get(key)
        after = pair["intersection"]["flags"][threshold].get(key)
        if type(before) is not bool or type(after) is not bool:
            name = "unknown"
        else:
            name = "both_present" if before and after else "both_absent" if not before and not after else "present_to_absent" if before else "absent_to_present"
        ids[name].append(pair["id"])
    return {"denominator": len(pairs), "counts": {name: len(values) for name, values in ids.items()}, "ids": ids,
            "note": "标签消失/出现是同一对象的观察变化，不等于已证明因果改善；任一侧未知单列。"}


def validate_cohorts(new_cases: list[dict], old_failed_cases: list[dict], old_success_cases: list[dict], *, expected=EXPECTED) -> dict:
    def index(cases, method, valid=None):
        result = {}
        for case in cases:
            sid = case["id"]
            if sid in result or case_method(case) != method:
                raise ValueError(f"Duplicate ID or wrong method: {method}/{sid}")
            observed = case_record(case, "post").get("native_valid_solid")
            if type(observed) is not bool or (valid is not None and observed is not valid):
                raise ValueError(f"Unknown/unexpected original validity: {method}/{sid}")
            result[sid] = case
        return result
    new = index(new_cases, "intersection")
    old_failed, old_success = index(old_failed_cases, "legacy", False), index(old_success_cases, "legacy", True)
    if old_failed.keys() & old_success.keys() or new.keys() != old_failed.keys() | old_success.keys():
        raise ValueError("Old failed/success controls must partition the identical new 411 identities")
    cohorts = {name: [] for name in COHORTS}
    for sid, case in sorted(new.items()):
        new_valid = case_record(case, "post")["native_valid_solid"]
        if sid in old_success and new_valid:
            raise ValueError(f"Both-success case is outside this failure census: {sid}")
        cohort = "regressed" if sid in old_success else "rescued" if new_valid else "persistent"
        cohorts[cohort].append(sid)
    if {key: len(ids) for key, ids in cohorts.items()} != expected:
        raise ValueError(f"Cohort counts differ from frozen protocol: { {key: len(ids) for key, ids in cohorts.items()} }")
    return cohorts


def validate_diagnostics(manifest: dict, root: Path, manifest_path: Path, *, saved_required=True, post_mode="optimized") -> dict:
    """Reject unfinished runs, method/source/protocol mixing and stale assets.

    Terminal error/timeout diagnostics remain legal unknown observations. Their
    missing optional measurements are never manufactured. Hashes are checked
    against each run's recorded protocol, including historical runner hashes.
    """
    names = ("traces", "upstream", "saved_step") if saved_required else ("traces", "upstream")
    protocols = {}
    for name, directory in (("traces", "provenance"), ("upstream", "upstream"), ("saved_step", "saved_step")):
        if name not in names:
            continue
        path = root / "metadata" / directory / "protocol.json"
        value = read_json(path)
        if value.get("manifest_sha256") != legacy.file_sha256(manifest_path):
            raise ValueError(f"Protocol manifest hash differs: {path}")
        if name == "traces" and value.get("post_mode") != post_mode:
            raise ValueError(f"Wrong replay mode: {path}")
        protocols[name] = {"path": str(path), "sha256": legacy.file_sha256(path), "fingerprint": fingerprint(value)}
    checked, terminal_errors = [], []
    for case in manifest["cases"]:
        sid, method = case["id"], case_method(case)
        raw_hash = legacy.file_sha256(case_path(case, "raw_npz"))
        original = case_record(case, "post" if post_mode == "optimized" else "fit")
        if original.get("raw_sha256") != raw_hash:
            raise ValueError(f"Original output raw hash differs: {method}/{sid}")
        for name in names:
            path = root / name / f"{sid}.json"
            data = read_json(path)  # Missing files mean a run is not complete.
            if data.get("sample_id", data.get("id")) != sid or data.get("method", "legacy") != method:
                raise ValueError(f"Diagnostic identity/method differs: {path}")
            if data.get("status") not in ("ok", "partial", "error", "post_error", "timeout", "crash", "missing_step"):
                raise ValueError(f"Incomplete/unrecognized diagnostic status: {path}: {data.get('status')}")
            no_header = data.get("status") in ("error", "post_error", "timeout", "crash") and data.get("protocol_fingerprint") is None
            if not no_header and data.get("protocol_fingerprint") != protocols[name]["fingerprint"]:
                raise ValueError(f"Diagnostic protocol differs: {path}")
            if data.get("status") not in ("ok", "partial"):
                terminal_errors.append({"id": sid, "diagnostic": name, "status": data["status"]})
            if data.get("status") in ("ok", "partial"):
                required = {"traces": "original", "upstream": "sources", "saved_step": "step_sha256"}[name]
                if data.get(required) is None:
                    raise ValueError(f"Completed diagnostic lacks source evidence: {path}: {required}")
            if name == "traces":
                if data.get("raw_sha256") != raw_hash and not (no_header and data.get("raw_sha256") is None):
                    raise ValueError(f"Replay raw hash differs: {path}")
                if data.get("original") is not None and data["original"] != original:
                    raise ValueError(f"Replay original record differs: {path}")
                if legacy.trace_verified(data):
                    if legacy.get(data, "replay", "native_valid_solid") is not original["native_valid_solid"]:
                        raise ValueError(f"Verified replay has a different native outcome: {path}")
                    source_step = case_path(case, "post_step" if post_mode == "optimized" else "fit_step")
                    contents = source_step.read_bytes()
                    start = contents.index(b"DATA;") + len(b"DATA;")
                    end = contents.index(b"ENDSEC;", start)
                    digest = hashlib.sha256(contents[start:end]).hexdigest()
                    if any(data["step_data_comparison"].get(key) != digest for key in ("original", "replay")):
                        raise ValueError(f"Verified replay STEP DATA hash differs: {path}")
            elif name == "upstream" and data.get("sources"):
                for key, expected_path in (("raw", case_path(case, "raw_npz")), ("gt", case_path(case, "gt_npz")),
                                           ("trace", root / "traces" / f"{sid}.json"), ("stage_geometry", root / "replay" / sid / "trace_geometry.npz")):
                    source = data["sources"].get(key, {})
                    actual_hash = legacy.file_sha256(expected_path) if expected_path.is_file() else None
                    if source.get("sha256") != actual_hash or Path(source.get("path", "")).resolve() != expected_path.resolve():
                        raise ValueError(f"Upstream source differs: {path}: {key}")
            elif name == "saved_step" and data.get("step_sha256") is not None:
                if data["step_sha256"] != legacy.file_sha256(case_path(case, "post_step")):
                    raise ValueError(f"Saved STEP source differs: {path}")
        checked.append(sid)
    return {"checked_count": len(checked), "checked_ids": checked, "protocols": protocols, "terminal_diagnostic_errors": terminal_errors}


def enrich_records(manifest: dict, root: Path) -> list[dict]:
    records, _ = legacy.aggregate(manifest, root)
    sources = {case["id"]: case for case in manifest["cases"]}
    for row in records:
        case = sources[row["id"]]
        trace = read_json(root / "traces" / f"{row['id']}.json")
        row.update(method=case_method(case), original_native_valid=case_record(case, "post")["native_valid_solid"],
                   owner_stages=owner_residual_stages(trace), failed_face_context=failed_face_context_errors(trace),
                   faces_without_boundary=faces_without_boundary(trace))
        row["flags"] = {t: observations_at(row, t) for t in THRESHOLDS}
    return records


def earliest_observation(flags: dict) -> dict:
    earlier_unknown = []
    for stage, _, keys in EARLIEST_STAGES:
        observed = [key for key in keys if flags.get(key) is True]
        if observed:
            return {"stage": stage, "confirmed_observations": observed, "earlier_unknown": earlier_unknown,
                    "interpretation": "最早已观测位置；不是错误开始发生的位置或唯一根因。"}
        earlier_unknown.extend(key for key in keys if flags.get(key) is None)
    return {"stage": "unknown" if earlier_unknown else "none_in_catalog", "confirmed_observations": [], "earlier_unknown": earlier_unknown}


def summarize_records(records: list[dict]) -> dict:
    context_codes = sorted(set(CONTEXT_ERRORS) | {code for r in records for f in (r["failed_face_context"]["faces"] or []) for code in f["codes"]})
    faces = [{"id": r["id"], **face} for r in records for face in (r["failed_face_context"]["faces"] or [])]
    return {
        "denominator": len(records), "ids": [r["id"] for r in records],
        "native_valid": sum(r["original_native_valid"] for r in records),
        "verified_traces": sum(r["trace_verified"] for r in records),
        "raw_topology_counts": dict(Counter(r["raw_topology"] for r in records)),
        "terminal": {key: {"count": sum(r["terminal_stage"] == key for r in records), "ids": [r["id"] for r in records if r["terminal_stage"] == key]} for key in TERMINALS},
        "observations": {t: {key: legacy.tally((r["id"], r["flags"][t][key]) for r in records) for key, *_ in OBSERVATIONS} for t in THRESHOLDS},
        "earliest_observation": {t: {stage: {"count": sum(earliest_observation(r["flags"][t])["stage"] == stage for r in records),
                                             "ids": [r["id"] for r in records if earliest_observation(r["flags"][t])["stage"] == stage],
                                             "with_earlier_unknown_count": sum(earliest_observation(r["flags"][t])["stage"] == stage and bool(earliest_observation(r["flags"][t])["earlier_unknown"]) for r in records)}
                                     for stage in [x[0] for x in EARLIEST_STAGES] + ["unknown", "none_in_catalog"]} for t in THRESHOLDS},
        "owner_stages": {
            "before_object_max": legacy.distribution(r["owner_stages"]["independent_curve"]["observed_max"] for r in records),
            "after_object_max": legacy.distribution(r["owner_stages"]["after_periodic_replacement"]["observed_max"] for r in records),
            "complete_object_max_change": legacy.distribution(r["owner_stages"]["object_max_delta"] for r in records),
            "matched_owner_change": legacy.distribution(owner["delta_max"] for r in records for owner in r["owner_stages"]["matched_owners"]),
            "change_counts": {key: sum(legacy.finite(r["owner_stages"]["object_max_delta"]) and (r["owner_stages"]["object_max_delta"] > 1e-12 if key == "increased" else r["owner_stages"]["object_max_delta"] < -1e-12 if key == "decreased" else abs(r["owner_stages"]["object_max_delta"]) <= 1e-12) for r in records) for key in ("increased", "decreased", "unchanged")},
        },
        "failed_face_context": {
            "counted_final_failed_faces": len(faces), "inspection_states": dict(Counter(f["inspection_state"] for f in faces)),
            "uncountable_ids": [r["id"] for r in records if r["failed_face_context"]["faces"] is None],
            "codes": {code: {"cad": legacy.tally((r["id"], context_flag(r, code)) for r in records),
                              "face_count": sum(code in f["codes"] for f in faces),
                              "direct_face_count": sum(code in f["direct_codes"] for f in faces),
                              "context_face_count": sum(code in f["context_codes"] for f in faces),
                              "faces": [{"id": f["id"], "face_id": f["face_id"]} for f in faces if code in f["codes"]]} for code in context_codes},
        },
        "failed_face_local_topology": {key: [{"id": r["id"], "face_id": f["face_id"]} for r in records for f in (r["failed_faces"]["failed_faces"] or []) if f["category"] == key] for key in ("raw_wrong_pair_incident", "raw_local_pair_exact", "unknown")},
    }


def paired_continuous(pairs: list[dict]) -> dict:
    getters = {
        "independent_owner_max": lambda r: r["owner_stages"]["independent_curve"]["observed_max"] if r["owner_stages"]["independent_curve"]["status"] == "ok" else None,
        "after_periodic_owner_max": lambda r: r["owner_stages"]["after_periodic_replacement"]["observed_max"] if r["owner_stages"]["after_periodic_replacement"]["status"] == "ok" else None,
        "raw_reciprocal_cd_sq_mean": lambda r: r["raw_metrics"].get("legacy_reciprocal_cd_sq_mean"),
        "raw_reciprocal_cd_sq_max": lambda r: r["raw_metrics"].get("legacy_reciprocal_cd_sq_max"),
        "optimization_face_mae_change": lambda r: r["optimization_delta"]["face_xyz_mae"],
        "optimization_edge_distance_change": lambda r: r["optimization_delta"]["matched_edge_mean_length"],
    }
    result = {}
    for key, getter in getters.items():
        observed = [{"id": p["id"], "legacy": getter(p["legacy"]), "intersection": getter(p["intersection"])} for p in pairs]
        complete = [r for r in observed if legacy.finite(r["legacy"]) and legacy.finite(r["intersection"])]
        result[key] = {"denominator": len(pairs), "paired_count": len(complete),
                       "legacy": legacy.distribution(r["legacy"] for r in complete),
                       "intersection": legacy.distribution(r["intersection"] for r in complete),
                       "intersection_minus_legacy": legacy.distribution(r["intersection"] - r["legacy"] for r in complete),
                       "paired_ids": [r["id"] for r in complete], "unknown_ids": [r["id"] for r in observed if r not in complete]}
    return result


def optimization_counterfactual(manifest: dict, root: Path, records: list[dict]) -> dict:
    import numpy as np
    ids = manifest.get("optimization_regression_ids")
    if not isinstance(ids, list) or len(ids) != 6 or len(set(ids)) != 6:
        raise ValueError("Expected the frozen six fit-valid / optimized-failed IDs")
    cases = {c["id"]: c for c in manifest["cases"]}
    subset = dict(manifest, cases=[cases[sid] for sid in ids])
    fit_root = root / "fit_only_counterfactual"
    audit = validate_diagnostics(subset, fit_root, root / "manifest.json", saved_required=False, post_mode="fit_only")
    fit_manifest = dict(subset, cases=[dict(c, post_record=case_record(c, "fit")) for c in subset["cases"]])
    fit_records = {r["id"]: r for r in enrich_records(fit_manifest, fit_root)}
    opt_records = {r["id"]: r for r in records}
    rows = []
    for sid in ids:
        case = cases[sid]
        if case_record(case, "fit").get("native_valid_solid") is not True or case_record(case, "post").get("native_valid_solid") is not False:
            raise ValueError(f"Counterfactual original outcomes differ: {sid}")
        fit, optimized = fit_records[sid], opt_records[sid]
        equal = {}
        if fit["trace_verified"] and optimized["trace_verified"]:
            before_path, after_path = [p / "replay" / sid / "trace_geometry.npz" for p in (fit_root, root)]
            if before_path.is_file() and after_path.is_file():
                with np.load(before_path, allow_pickle=False) as a, np.load(after_path, allow_pickle=False) as b:
                    for key in ("selected_faces", "selected_edges", "selected_connectivity"):
                        equal[key] = bool(np.array_equal(a[key], b[key])) if key in a and key in b else None
        rows.append({"id": sid, "fit_only": fit, "optimized": optimized,
                     "same_selected_arrays": {key: equal.get(key) for key in ("selected_faces", "selected_edges", "selected_connectivity")},
                     "same_raw_input_sha256": case_record(case, "fit")["raw_sha256"] == case_record(case, "post")["raw_sha256"]})
    return {"denominator": 6, "audit": audit, "cases": rows,
            "note": "同一原输入的完整 fit-only 与优化后路径比较。后续拟合/裁剪也会改变，不是单一优化 loss 的因果消融。"}


def build_report(root: Path, baseline_root: Path) -> tuple[list[dict], dict]:
    manifest_path = root / "manifest.json"
    manifest = read_json(manifest_path)
    controls_path = root / "legacy_control_manifest.json"
    controls = read_json(controls_path)
    old_path = baseline_root / "manifest.json"
    old_manifest = read_json(old_path)
    population = manifest.get("population", {})
    if any(population.get(k) != v for k, v in {"accepted_test": 2417, "original_test": 2424, "contract_unsupported": 7, "native_failed": 304, "native_valid": 2113}.items()):
        raise ValueError("Population differs from the frozen 2417/2424 comparison")
    cohorts = validate_cohorts(manifest["cases"], old_manifest["cases"], controls["cases"])
    if manifest.get("failed_ids") != sorted(cohorts["persistent"] + cohorts["regressed"]):
        raise ValueError("Main failure ID list differs from paired original outcomes")
    reuse_path = root / "baseline_reuse.json"
    reuse = read_json(reuse_path)
    reuse_cases = {r["id"]: r for r in reuse["cases"]}
    if len(reuse_cases) != 369 or set(reuse_cases) != {c["id"] for c in old_manifest["cases"]}:
        raise ValueError("Historical reuse audit must cover the exact 369 failed IDs")
    if legacy.get(reuse, "baseline_manifest", "sha256") != legacy.file_sha256(old_path):
        raise ValueError("Historical manifest changed after reuse audit")
    for row in reuse_cases.values():
        if type(row.get("eligible")) is not bool:
            raise ValueError(f"Unknown historical reuse eligibility: {row['id']}")
        if row["eligible"]:
            for source in row["sources"].values():
                if not source.get("exists") or legacy.file_sha256(Path(source["path"])) != source.get("sha256"):
                    raise ValueError(f"Historical diagnostic changed since reuse audit: {row['id']}")
    eligible_manifest = dict(old_manifest, cases=[c for c in old_manifest["cases"] if reuse_cases[c["id"]]["eligible"]])
    audit = {
        "intersection": validate_diagnostics(manifest, root, manifest_path),
        "legacy_controls": validate_diagnostics(controls, root / "legacy_controls", controls_path),
        "legacy_historical": validate_diagnostics(eligible_manifest, baseline_root, old_path),
        "baseline_reuse": {"path": str(reuse_path), "sha256": legacy.file_sha256(reuse_path), "eligible_count": sum(r["eligible"] for r in reuse_cases.values()),
                           "ineligible": [{"id": r["id"], "reasons": r["reasons"]} for r in reuse_cases.values() if not r["eligible"]]},
    }
    preservation_path = root / "validation/source_preservation.json"
    if preservation_path.is_file():
        preservation = read_json(preservation_path)
        if preservation.get("mismatch_count") != 0 or preservation.get("source_audit_sha256") != legacy.file_sha256(root / "source_asset_audit.json"):
            raise ValueError("Source preservation audit failed or refers to a different inventory")
        audit["source_preservation"] = {"path": str(preservation_path), "sha256": legacy.file_sha256(preservation_path), **preservation}
    new_records = enrich_records(manifest, root)
    old_records = enrich_records(old_manifest, baseline_root)
    # Unusable historical evidence remains unknown. Original outcome/cohort is
    # fixed independently; do not drop cases or call unknown an improvement.
    for row in old_records:
        if not reuse_cases[row["id"]]["eligible"]:
            row["trace_verified"] = False
            row["terminal_stage"] = "unknown"
            row["flags"] = {t: {key: None for key, *_ in OBSERVATIONS} for t in THRESHOLDS}
            row["owner_stages"] = owner_residual_stages({})
            row["failed_face_context"] = failed_face_context_errors({})
            row["failed_faces"] = {"status": "unknown", "failed_faces": None}
            row["raw_metrics"] = {}
            row["raw_topology"] = "unknown"
            row["optimization_delta"] = {"face_xyz_mae": None, "matched_edge_mean_length": None}
    control_records = enrich_records(controls, root / "legacy_controls")
    old = {r["id"]: r for r in old_records + control_records}
    new = {r["id"]: r for r in new_records}
    cohort_of = {sid: key for key, ids in cohorts.items() for sid in ids}
    pairs = [{"id": sid, "cohort": cohort_of[sid], "legacy": old[sid], "intersection": new[sid]} for sid in sorted(new)]
    failed = [r for r in new_records if not r["original_native_valid"]]
    summary = {
        "schema": VERSION, "created_at_utc": datetime.now(timezone.utc).isoformat(), "population": population,
        "main": summarize_records(failed), "historical_failures": summarize_records(old_records),
        "fit_only_vs_optimized_all_accepted": manifest["fit_only_vs_optimized_all_accepted"],
        "cohorts": {key: {"label": COHORTS[key], "count": len(ids), "ids": ids,
                          "legacy": summarize_records([old[s] for s in ids]), "intersection": summarize_records([new[s] for s in ids]),
                          "transitions": {t: {name: paired_transitions([p for p in pairs if p["cohort"] == key], name, t) for name, *_ in OBSERVATIONS} for t in THRESHOLDS},
                          "continuous": paired_continuous([p for p in pairs if p["cohort"] == key])} for key, ids in cohorts.items()},
        "audit": audit, "optimization_counterfactual": optimization_counterfactual(manifest, root, new_records),
        "sources": {"manifest": {"path": str(manifest_path), "sha256": legacy.file_sha256(manifest_path)},
                    "legacy_manifest": {"path": str(old_path), "sha256": legacy.file_sha256(old_path)},
                    "source_asset_audit": {"path": str(root / "source_asset_audit.json"), "sha256": legacy.file_sha256(root / "source_asset_audit.json")}},
        "limitations": [
            "主分母304只含新版失败；107救回对照和42退步对照按原结果选择，不代表全部成功件。7个合同不支持对象单列，接受集2417，原始集2424。",
            "369旧失败与304新失败是不同对象集合，其频次差仅作描述。262/42/107成对组按同一ID比较；未知不算改善。",
            "只有原终态、STEP DATA相等且无观测异常的完整重放用于阶段归因。原成功即使重放缺测仍保留成功；中间标签可未知。",
            "这是GT编码后验均值的VAE解码溯源；不是随机后验训练loss，也不是端到端照片或固定diffusion生成能力结论。",
            "本轮没有执行面latent去重；半边选择会保留一份互反曲线，二者是不同操作。",
            "阈值为归一化XYZ长度，不是毫米。面网格接近/端点最近邻/曲线有限采样均为代理指标，不证明拓扑或UV合法。",
            "丢边界不等于丢孔；至少一个闭环不等于所有环都闭合；边面距离小也不能证明UV裁剪正确。",
            "独立曲线与周期替换后曲线按相同edge/owner身份比较，但曲线参数域可改变；9点采样不是逐空间点追踪或Hausdorff距离。",
            "候选检查错误只统计最终仍失败的decoder面，每面每错误去重；保留候选自身和所属上下文两种来源，修复后成功的面不混入。错误名字/共现不能直接作为唯一UV根因。",
            "当前训练并非等预算消融。统计不能证明latent不平滑、网络太小、某块信息融合失效或refiner带来因果改进。",
        ],
    }
    summary["highlights"] = evidence_highlights(summary)
    return pairs, summary


def evidence_highlights(summary: dict) -> dict:
    main = summary["main"]
    obs = main["observations"]["0.006"]
    rescued = summary["cohorts"]["rescued"]["intersection"]
    optimized = summary["fit_only_vs_optimized_all_accepted"]["intersection"]
    return {"failure_count": main["denominator"],
            "persistent_count": summary["cohorts"]["persistent"]["count"],
            "regressed_count": summary["cohorts"]["regressed"]["count"],
            "rescued_count": summary["cohorts"]["rescued"]["count"],
            "all_closed_but_trim_failed": obs["strict_closed_candidate_failed"]["positive"],
            "no_closed_admitted": obs["never_closed_admitted"]["positive"],
            "returned_face_discarded_wire": obs["final_success_discarded_wire"]["positive"],
            "raw_unordered_pairs_exact": main["raw_topology_counts"].get("exact", 0),
            "failed_owner_gap": obs["owner_after"]["positive"],
            "rescued_owner_gap": rescued["observations"]["0.006"]["owner_after"]["positive"],
            "optimized_rescued_from_fit": optimized["fit_failed_optimized_valid"],
            "optimized_regressed_from_fit": optimized["fit_valid_optimized_failed"],
            "normalized_length_threshold": "0.006"}


def highlight_paragraphs(summary: dict) -> list[str]:
    h = summary["highlights"]
    return [
        f"{h['failure_count']}个新版失败件里，{h['all_closed_but_trim_failed']}件存在“某次所有候选环都闭合并接纳，面仍裁不出来”；{h['no_closed_admitted']}件有最终失败面从未接纳闭合环；{h['returned_face_discarded_wire']}件虽然返回了面，却在成功尝试中舍弃了部分边界。这三种情况可以重叠，不能都归为“环没闭合”。",
        f"{h['raw_unordered_pairs_exact']}/{h['failure_count']}件的原始无向面连接与GT完全一致，最后仍失败。因此只把“哪些面相连”预测正确还不够；面数、方向、边界与参数表示的有效性仍要分开检查。",
        f"{h['failed_owner_gap']}/{h['failure_count']}个失败件在周期替换后仍有边面采样残差>{h['normalized_length_threshold']}，但本轮救回的{h['rescued_count']}个成功件里，{h['rescued_owner_gap']}件也超这个阈值。它是贴合偏差的信号，不能当作失败的充分条件或唯一原因；数值变小也不保证边界合法。",
        f"失败组中，优化常使几何离GT更远；同时在全部接受集里，优化把{h['optimized_rescued_from_fit']}个fit-only失败件变成合法实体，只让{h['optimized_regressed_from_fit']}个原本合法的fit-only结果退步。这里体现的是贴合与几何保真的权衡，不能说“大多数零件被优化弄坏了”。",
    ]


def make_charts(summary: dict, directory: Path, font_path: Path | None = None) -> list[dict]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    import numpy as np
    font_path = font_path or Path("/root/.cache/hola-browser-libs/fonts/NotoSansCJKsc-Regular.otf")
    if not font_path.is_file():
        raise FileNotFoundError("A CJK font is required for Chinese charts; pass --font-path")
    font_manager.fontManager.addfont(str(font_path))
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=str(font_path)).get_name(), "font.size": 10, "axes.unicode_minus": False, "axes.spines.top": False, "axes.spines.right": False})
    directory.mkdir(parents=True, exist_ok=True)
    charts = []
    labels = {key: chinese for key, chinese, *_ in OBSERVATIONS}
    def save(fig, name, caption):
        fig.savefig(directory / f"{name}.png", dpi=180, bbox_inches="tight")
        fig.savefig(directory / f"{name}.svg", bbox_inches="tight")
        plt.close(fig)
        charts.append({"name": name, "png": f"visualizations/{name}.png", "svg": f"visualizations/{name}.svg", "caption": caption})
    threshold = "0.006"
    observations = summary["main"]["observations"][threshold]
    keys = [k for k, *_ in OBSERVATIONS]
    keys.sort(key=lambda k: observations[k]["positive"], reverse=True)
    fig, ax = plt.subplots(figsize=(11, 10))
    ys = np.arange(len(keys))
    positives = [observations[k]["positive"] for k in keys]
    ax.barh(ys, positives, color="#277e83", label="确认观察到")
    ax.barh(ys, [observations[k]["unknown"] for k in keys], left=positives, color="#d8dce2", label="未知")
    ax.set_yticks(ys, [labels[k] for k in keys]); ax.invert_yaxis()
    ax.set_xlim(0, 304); ax.set_xlabel("件数 / 新版304个失败件（同一件可有多种问题）")
    ax.set_title("哪些现象反复出现？距离项阈值为归一化长度0.006")
    ax.legend(loc="lower right"); ax.grid(axis="x", alpha=.15); fig.tight_layout()
    save(fig, "main_failure_observations", "新版304失败件的多标签观察；灰色为未知，留白为确认未观察到。标签可以重叠。距离项阈值0.006。")
    comparison_keys = ["never_wire", "never_closed_admitted", "closed_admitted_still_failed", "strict_closed_candidate_failed", "final_success_discarded_wire", "owner_after", "optimized_endpoint_gap", "raw_extra_gt_pair", "raw_missing_gt_pair"]
    fig, axes = plt.subplots(1, 3, figsize=(17, 6), sharey=True)
    for ax, (cohort, title) in zip(axes, COHORTS.items()):
        data = summary["cohorts"][cohort]
        transition = data["transitions"][threshold]
        y = np.arange(len(comparison_keys))
        ax.barh(y - .19, [transition[k]["counts"]["present_to_absent"] for k in comparison_keys], .36, color="#277e83", label="原有→新无")
        ax.barh(y + .19, [transition[k]["counts"]["absent_to_present"] for k in comparison_keys], .36, color="#c9714d", label="原无→新有")
        ax.set_yticks(y, [labels[k] for k in comparison_keys]); ax.set_xlabel("成对比较的件数")
        ax.set_title(f"{title} (n={data['count']})"); ax.grid(axis="x", alpha=.15)
    axes[0].invert_yaxis(); axes[0].legend(loc="lower right", fontsize=8); fig.tight_layout()
    save(fig, "paired_observation_changes", "同一对象的标签变化，三个分母分别262/42/107。任一侧未知不画成消失或出现；完整未知和稳定计数在页面及JSON中。")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
    for key, label, color in (("owner_independent", "独立拟合后", "#277e83"), ("owner_after", "周期曲线替换后", "#c9714d")):
        axes[0].plot([float(t) for t in THRESHOLDS], [summary["main"]["observations"][t][key]["positive"] for t in THRESHOLDS], "o-", color=color, label=label)
    axes[0].set_xscale("log"); axes[0].set_xticks([float(t) for t in THRESHOLDS], THRESHOLDS)
    axes[0].set_ylim(0, 304); axes[0].set_xlabel("归一化XYZ长度阈值")
    axes[0].set_ylabel("至少一条边贴合所属面残差超限的件数")
    axes[0].set_title("同一批304失败件：边面贴合前后对照"); axes[0].legend(fontsize=9)
    change = summary["main"]["owner_stages"]["change_counts"]
    unknown = 304 - sum(change.values())
    axes[1].bar(["减小", "基本不变", "增大", "未知"], [change["decreased"], change["unchanged"], change["increased"], unknown], color=["#277e83", "#9aacae", "#c9714d", "#d8dce2"])
    axes[1].set_ylabel("件数"); axes[1].set_title("周期替换后，每件最差贴合残差怎样变化")
    for ax in axes: ax.grid(axis="y", alpha=.15)
    fig.tight_layout()
    save(fig, "owner_residual_before_after", "独立拟合与周期替换后的边面贴合对照。左图按阈值统计超限件；右图只在两阶段全部边/owner采样完整且对应一致时统计对象最大值变化。")
    context = summary["main"]["failed_face_context"]
    codes = sorted(context["codes"], key=lambda c: context["codes"][c]["face_count"], reverse=True)
    fig, ax = plt.subplots(figsize=(10, max(3.8, .5 * len(codes) + 1.4)))
    ax.barh(np.arange(len(codes)), [context["codes"][c]["face_count"] for c in codes], color="#6377a2")
    ax.set_yticks(np.arange(len(codes)), [CONTEXT_ERRORS.get(c, c.removeprefix("BRepCheck_")) for c in codes]); ax.invert_yaxis()
    ax.set_xlabel("最终失败面数（同一面同一种错误只数一次）")
    ax.set_title(f"{context['counted_final_failed_faces']}个最终失败面：候选检查反复报什么")
    ax.grid(axis="x", alpha=.15); fig.tight_layout()
    save(fig, "persistent_face_context_errors", "只数最终仍失败的面，合并候选自身及所属上下文错误；同一面同一错误跨容差去重。未构造候选/未检查/检查不完整另列，不能称作唯一UV根因。")
    return charts


def format_tally(value: dict) -> str:
    return f"{value['positive']}/{value['denominator']}；未知 {value['unknown']}"


def format_number(value) -> str:
    return f"{value:.6g}" if legacy.finite(value) else "未知"


def report_markdown(summary: dict, charts: list[dict]) -> str:
    main = summary["main"]
    observations = main["observations"]["0.006"]
    lines = [
        "# Intersection refinement：失败到底卡在哪里", "",
        "主统计是 **304个新版失败件＝262个持续失败＋42个原成功、新失败**；另外加入107个原失败、新成功的救回对照。接受集为2417件，原始2424件中7件合同不支持。",
        "", f"新版304失败中有 **{main['verified_traces']}/304** 份轨迹通过完整重放核验；只有这些轨迹用于后处理阶段归因。原结果来自原始评估，缺测不会改变成功/失败，也不会移出分母。",
        "", "[打开统计页面](page/index.html) · [研究判断与下一步方向](NOTE.md) · [紧凑统计JSON](compact_summary.json) · [完整统计JSON](summary.json) · [逐件成对CSV](cases.csv) · [冻结协议](PLAN.md)",
        "", "## 先看最值得讨论的几件事", "", *[paragraph + "\n" for paragraph in highlight_paragraphs(summary)],
        "", "## 先用直白的话看反复发生的情况", "",
        "下面是多标签观察，一件可以同时出现在多行。距离项目先用0.006，页面可切换0.002/0.006/0.01/0.08；它们是归一化长度，不是毫米。",
        "", "| 观察到的问题 | 确认件数 / 304；未知 |", "|---|---:|",
    ]
    by_count = sorted(OBSERVATIONS, key=lambda item: observations[item[0]]["positive"], reverse=True)
    lines.extend(f"| {label} | {format_tally(observations[key])} |" for key, label, *_ in by_count)
    lines.extend(["", "“至少有一个闭合环”不保证其他环也闭合；“全部候选环都闭合并接纳”是更严格的另一行。面做出来却丢了边界，也不能直接称作丢孔。", "", "## 最后停在哪一步", "", "| 原运行的末端情况 | 件数 |", "|---|---:|"])
    lines.extend(f"| {label} | {main['terminal'][key]['count']} |" for key, label in TERMINALS.items() if key != "native_valid")
    lines.extend(["", "这是最后阻断的分支，不是最早的唯一原因。比如闭环失败之前，可能已经出现漏连接、半边选择损失或几何端点偏差。", "", "**按现有测量，最早在哪个阶段看到异常？**（阈值0.006）", "",
                  "| 最早已观察到的位置 | 件数 | 其中更早阶段存在缺测 |", "|---|---:|---:|"])
    for stage, label in [(key, label) for key, label, _ in EARLIEST_STAGES] + [("unknown", "缺测，尚无确认异常"), ("none_in_catalog", "现有目录未确认异常")]:
        row = main["earliest_observation"]["0.006"][stage]
        lines.append(f"| {label} | {row['count']} | {row['with_earlier_unknown_count']} |")
    lines.extend(["", "这只是最早取得证据的位置，不等于错误真正开始的位置：例如选边后首次测到端点残差，并不能证明偏差由选择操作造成。", "", "## 相同零件的旧新对照", "",
                  "原369失败和新304失败是不同集合，不能只比较百分比后说某机制被修好了。下面分别比较同一个ID；任一侧未知时都不算标签改善。", "",
                  "| 分组 | 件数 | 原版完整重放 | 新版完整重放 |", "|---|---:|---:|---:|"])
    for key, data in summary["cohorts"].items():
        lines.append(f"| {COHORTS[key]} | {data['count']} | {data['legacy']['verified_traces']} | {data['intersection']['verified_traces']} |")
    keys = ("never_wire", "never_closed_admitted", "closed_admitted_still_failed", "strict_closed_candidate_failed", "final_success_discarded_wire", "owner_after", "optimized_endpoint_gap")
    labels = {k: label for k, label, *_ in OBSERVATIONS}
    for cohort, data in summary["cohorts"].items():
        lines.extend(["", f"**{COHORTS[cohort]}：{data['count']}件**（距离阈值0.006）", "",
                      "| 同一件的观察变化 | 原有→新无 | 原无→新有 | 两边都有 | 两边都无 | 任一未知 |", "|---|---:|---:|---:|---:|---:|"])
        for key in keys:
            values = data["transitions"]["0.006"][key]["counts"]
            lines.append(f"| {labels[key]} | " + " | ".join(str(values[k]) for k in ("present_to_absent", "absent_to_present", "both_present", "both_absent", "unknown")) + " |")
    owner = main["owner_stages"]
    delta = owner["complete_object_max_change"]
    counts = owner["change_counts"]
    lines.extend(["", "## 边贴不到面：拟合时就存在，还是周期替换后出现", "",
                  "这里比较同一条被选择的曲线及其所属面，分别测量独立拟合后与周期曲线替换后的贴合距离。两阶段都用9个均匀参数采样点，但曲线参数域可能改变，所以不是同一空间点的运动轨迹。", "",
                  "| 归一化长度阈值 | 独立拟合后超限 | 周期替换后超限 |", "|---|---:|---:|"])
    for t in THRESHOLDS:
        lines.append(f"| {t} | {format_tally(main['observations'][t]['owner_independent'])} | {format_tally(main['observations'][t]['owner_after'])} |")
    lines.extend(["", f"两阶段完整且对应一致的 **{delta['finite_count']}/304** 件中，对象最大采样距离下降{counts['decreased']}件、增加{counts['increased']}件、基本不变{counts['unchanged']}件。最大值变化中位数 **{format_number(delta['median'])}**，均值 **{format_number(delta['mean'])}**（后减前）。其余保留未知。",
                  "", "这可以区分‘替换前就有贴合偏差’和‘替换后观测发生变化’，但不能仅凭距离小就确认UV裁剪正确。", "", "## 裁不出的那些面，检查器反复报什么", ""])
    context = main["failed_face_context"]
    states = context["inspection_states"]
    lines.extend([f"共有{context['counted_final_failed_faces']}个最终失败的decoder面可计数；{len(context['uncountable_ids'])}件的失败面集合无法核验。检查状态：{json.dumps(states, ensure_ascii=False)}。只统计最终仍失败的面，每面每错误只计一次，修复后成功的面不混入。", "",
                  "| 通俗描述 / 精确错误名 | 有该错误的面数 | 自身 / 上下文面数 | 确认件数 / 304；未知 |", "|---|---:|---:|---:|"])
    for code, values in sorted(context["codes"].items(), key=lambda x: x[1]["face_count"], reverse=True):
        lines.append(f"| {CONTEXT_ERRORS.get(code, '其他候选检查错误')}；`{code}` | {values['face_count']} | {values['direct_face_count']} / {values['context_face_count']} | {format_tally(values['cad'])} |")
    lines.extend(["", "计数合并候选自身statuses与其所属上下文contexts.statuses，并另列来源：同一面同一码两处都有时，并集仍只计一次。没有候选可检查不冒充‘检查通过’。IntersectingWires是不同环被检查为相交，SelfIntersectingWire是单个环被检查为自相交；它们都不直接证明3D曲线真实相交或唯一UV根因。", "", "## 6件 fit-only 成功、优化后失败的路径对照", "",
                  "| ID | fit-only 重放一致 | 优化重放一致 | 原输入相同 | 选择后面 / 边 / 连接逐元素相同 | 优化后停在哪里 |", "|---|---|---|---|---|---|"])
    for row in summary["optimization_counterfactual"]["cases"]:
        same = "/".join("是" if v is True else "否" if v is False else "未知" for v in row["same_selected_arrays"].values())
        lines.append(f"| {row['id']} | {row['fit_only']['trace_verified']} | {row['optimized']['trace_verified']} | {row['same_raw_input_sha256']} | {same} | {TERMINALS[row['optimized']['terminal_stage']]} |")
    lines.extend(["", summary["optimization_counterfactual"]["note"], "",
                  "精细审计中的00991826提醒我们：fit-only的合法实体已经缺一张输入面并舍弃了部分边界；优化后修复那张面，却丢失另一张。因此这里比较的是实体有效性路径，不能把原成功等同完整准确重建GT。", "",
                  "[6件优化路径的独立精细审计](optimization_regressions.json)", "", "## 图表", ""])
    for chart in charts:
        lines.extend([f"![{chart['caption']}]({chart['png']})", "", chart["caption"], ""])
    lines.extend(["## 统计边界与复核", ""])
    lines.extend(f"- {line}" for line in summary["limitations"])
    lines.extend(["", f"历史369份诊断中，资产/生产算法/测量合同审核允许复用 **{summary['audit']['baseline_reuse']['eligible_count']}** 份。保留原runner指纹，不伪装成新运行。具体源哈希在summary.audit与baseline_reuse.json中。", "",
                  "观察与判据逐项对应见页面展开项及 `cases.json`；不能据此直接判定latent空间不平滑或网络容量不足。", ""])
    return "\n".join(lines)


def page_payload(pairs: list[dict], summary: dict, charts: list[dict], atlas_ids: set[str]) -> dict:
    def compact(row):
        return {key: row[key] for key in ("id", "method", "original_native_valid", "terminal_stage", "trace_verified", "flags", "owner_stages", "failed_face_context", "persistent_trim", "faces_without_boundary", "optimization_delta", "source_files")}
    # The page recalculates exactly the selected cohort from these paired rows.
    # Expose the authoritative main summary for browser/JSON parity checks.
    return {"schema": VERSION, "population": summary["population"], "main": summary["main"],
            "highlights": highlight_paragraphs(summary),
            "cohort_labels": COHORTS, "thresholds": list(THRESHOLDS), "terminal_labels": TERMINALS,
            "observations": [{"key": key, "label": label, "english": english, "criterion": criterion, "uses_threshold": threshold} for key, label, english, criterion, threshold in OBSERVATIONS],
            "context_labels": CONTEXT_ERRORS, "charts": charts, "limitations": summary["limitations"],
            "pairs": [{"id": p["id"], "cohort": p["cohort"], "in_atlas": p["id"] in atlas_ids,
                       "legacy": compact(p["legacy"]), "intersection": compact(p["intersection"])} for p in pairs]}


PAGE_HTML = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>失败溯源 · Intersection refinement</title>
<style>
:root{font-family:Inter,"Noto Sans CJK SC","Microsoft Yahei",system-ui,sans-serif;color:#21303d;background:#f4f6f8;font-size:15px;--teal:#277e83;--orange:#bd6544;--line:#dce3e8}*{box-sizing:border-box}body{margin:0}a{color:#286b98}main{max-width:1320px;margin:auto;padding:32px 28px 80px}header{margin-bottom:24px}.eyebrow{font-size:12px;letter-spacing:.15em;color:#597681;font-weight:700}h1{font-size:32px;margin:10px 0 12px}h2{font-size:21px;margin:0 0 12px}p{line-height:1.8;color:#4b5b67}.lead{max-width:970px}.links{display:flex;gap:18px;flex-wrap:wrap}.controls{position:sticky;top:0;z-index:3;background:#f4f6f8f5;border-block:1px solid var(--line);padding:15px 0;display:flex;flex-wrap:wrap;align-items:center;gap:18px;backdrop-filter:blur(8px)}label{font-size:13px;font-weight:600}select,input{font:inherit;padding:9px;border:1px solid #c5d0d8;border-radius:6px;background:white;margin-left:7px}input{width:145px}.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:22px 0}.card,section{background:white;border:1px solid var(--line);border-radius:10px;padding:20px}.card b{font-size:30px;display:block;margin:7px 0}.muted,small{color:#697b87;font-size:12px}.card strong{font-size:13px}section{margin-top:18px}.legend{display:flex;gap:18px;align-items:center;flex-wrap:wrap;margin-bottom:14px;font-size:12px}.dot{width:11px;height:11px;display:inline-block;border-radius:2px;margin-right:5px}.scroll{overflow:auto}table{width:100%;border-collapse:collapse;font-size:13px}th{background:#f1f5f6;color:#506572;font-size:12px;text-align:left}th,td{padding:11px 10px;border-bottom:1px solid #e5eaee;vertical-align:top}td.num,th.num{text-align:right;white-space:nowrap;font-variant-numeric:tabular-nums}td.name{min-width:230px}button.count{border:0;background:transparent;font:inherit;text-decoration:underline;text-underline-offset:3px;color:#21616b;cursor:pointer;padding:0 3px}button.count:disabled{text-decoration:none;color:#7d8b93;cursor:default}.bar{display:flex;min-width:95px;height:8px;border-radius:4px;overflow:hidden;background:#edf1f3;margin-top:7px}.bar span{height:8px}.negative{background:#b9cfd0}.positive{background:var(--teal)}.unknown{background:#d8dce2}details{margin-top:5px;font-size:12px;line-height:1.7}summary{cursor:pointer;color:#5f7582}.ids{display:flex;flex-wrap:wrap;gap:6px;max-height:170px;overflow:auto;margin:12px 0}.id{font:12px ui-monospace,monospace;background:#eef4f5;border:1px solid #d2dfe1;border-radius:4px;padding:5px 8px;color:#1e6170;cursor:pointer}.id:hover{background:#dbecee}.pair-grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}.method{border:1px solid var(--line);padding:16px;border-radius:8px}.method h3{margin-top:0;font-size:17px}.badge{display:inline-block;padding:3px 8px;border-radius:4px;background:#e4f1ee;color:#2c7468;font-size:12px}.bad{background:#f7ece5;color:#985433}.gray{background:#edf0f4;color:#69727e}.notice{border-left:3px solid #83abae;padding-left:14px}.figures{display:grid;grid-template-columns:1fr 1fr;gap:18px}.figures figure{margin:0;border:1px solid var(--line);border-radius:6px;padding:10px}.figures img{width:100%;height:270px;object-fit:contain}.figures figcaption{font-size:12px;line-height:1.7;color:#5c6d77}code{font-size:11px;word-break:break-all}.empty{color:#77868f;padding:15px}ul{line-height:1.8;padding-left:20px}.context-table{max-height:370px;overflow:auto}#selection{scroll-margin-top:100px}#case-view{scroll-margin-top:100px}.title-line{display:flex;justify-content:space-between;gap:10px;align-items:center}.foot{font-size:12px}.count-note{color:#2d636f;font-weight:600}#active-caption{margin:0}.tri{font-size:12px}.tiny{font-size:11px}@media(max-width:850px){main{padding:20px 12px}.grid{grid-template-columns:1fr 1fr}.pair-grid,.figures{grid-template-columns:1fr}.controls{position:static}h1{font-size:26px}section{padding:15px}select{max-width:200px}}
</style></head><body><main>
<header><div class="eyebrow">HOLA-BREP / INTERSECTION REFINEMENT / 2026-10-11</div><h1>失败从哪里开始？</h1>
<p class="lead">先看网络有没有连错，再看选边、优化、拟合与裁剪。<b>304个新版失败件＝262个持续失败＋42个退步</b>，另加107个救回对照。下面每一行都能展开具体ID。</p>
<div class="links"><a href="../NOTE.md">研究判断与下一步方向</a><a href="../REPORT.md">中文报告</a><a href="../summary.json">完整统计</a><a href="../cases.csv">逐件CSV</a><a href="../optimization_regressions.json">6件优化路径审计</a><a href="../../../joint_boundary/legacy_failures_20261011/page/">原50件3D图库</a></div></header>
<section id="highlights"><h2>先看最值得讨论的几件事</h2><div id="highlight-text"></div><p class="muted">以上来自固定全集；下方可切换具体对象组与距离阈值。</p></section>
<div class="controls"><label>对象组<select id="cohort"><option value="main">新版全部失败 · 304</option><option value="persistent">旧新都失败 · 262</option><option value="regressed">原成功、新失败 · 42</option><option value="rescued">原失败、新成功 · 107</option></select></label>
<label>距离阈值<select id="threshold"><option>0.002</option><option selected>0.006</option><option>0.01</option><option>0.08</option></select></label>
<label>查看ID<input id="case-search" placeholder="例如 00598784" inputmode="numeric" maxlength="8"></label><span class="muted">归一化XYZ长度，不是毫米</span></div>
<div class="grid" id="cards"></div>
<section><h2>同一批对象，旧新分别发生了什么</h2><p id="active-caption"></p><div class="legend"><span><i class="dot positive"></i>确认观察到</span><span><i class="dot negative"></i>确认未观察到</span><span><i class="dot unknown"></i>未知/无法核验</span><span>点击数量展开ID；多标签可重叠。</span></div>
<div class="scroll"><table id="mechanisms"><thead><tr><th>通俗描述 / 可展开精确判据</th><th class="num">原版：有 / 无 / 未知</th><th class="num">新版：有 / 无 / 未知</th><th class="num">原有 → 新无</th><th class="num">原无 → 新有</th><th class="num">任一未知</th></tr></thead><tbody></tbody></table></div></section>
<section id="selection"><div class="title-line"><h2 id="selection-title">展开观察对应的ID</h2><span id="selection-count" class="count-note"></span></div><p class="muted" id="selection-note">在上表点击任意数量。未知不会被当成标签消失或改善。</p><div class="ids" id="selected-ids"></div></section>
<section id="case-view"><h2>把同一个零件放在一起看</h2><p id="case-hint">点选一个ID后，展示原终态、重放核验、该件观察、边面贴合与持续失败面的上下文错误。</p><div id="case-detail"></div></section>
<section><h2>边贴不到面：独立拟合时就有，还是周期替换后改变了？</h2><p>按相同边 / 所属面身份比较，采样定义保持一致。周期替换可能改变曲线参数域，因此这里比较采样残差，不能理解成同一空间点的移动。</p><div class="scroll"><table id="owner-summary"><thead><tr><th>当前组 / 每件最大采样距离</th><th>原版</th><th>新版</th></tr></thead><tbody></tbody></table></div></section>
<section><h2>只看最后仍裁不出来的面：候选自身与上下文反复报什么？</h2><p>排除后来修复成功的面。每个decoder面每类错误只数一次；直接检查与所属上下文分别保留。不同错误可以同时出现。没有候选可检查或检查不完整的面，不写成“检查通过”。</p><div id="context-coverage" class="muted"></div><div class="scroll context-table"><table id="context-summary"><thead><tr><th>错误的通俗含义 / 原始名称</th><th class="num">原版并集面数 / 件数 / 自身 / 上下文</th><th class="num">新版并集面数 / 件数 / 自身 / 上下文</th></tr></thead><tbody></tbody></table></div></section>
<section><h2>可导出的静态统计图</h2><p class="muted">以下是固定全集科研图，标题标明分母。上方筛选不会更改这些图片。</p><div class="figures" id="figures"></div></section>
<section><h2>这些统计能说明到哪里</h2><ul id="limitations"></ul><p class="foot">数据源保留原runner指纹；只在原终态一致、STEP DATA逐字节一致且没有观测异常时采用后处理阶段证据。原结果与诊断是否齐全分开记录。</p></section>
</main><script src="data.js"></script><script src="app.js"></script></body></html>
'''


PAGE_JS = r'''"use strict";
const D=window.INTERSECTION_CENSUS_DATA;
const $=id=>document.getElementById(id);
const esc=value=>String(value??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const names={positive:"确认有",negative:"确认无",unknown:"未知"};
let activePairs=[],activeCase=null,selection=null;
const number=v=>typeof v==="number"&&Number.isFinite(v)?Number(v.toPrecision(5)).toString():"未知";
const state=v=>v===true?"positive":v===false?"negative":"unknown";
const stat=(pairs,method,key,t)=>{const a={positive:[],negative:[],unknown:[]};for(const p of pairs)a[state(p[method].flags[t][key])].push(p.id);return a;};
const transition=(pairs,key,t)=>{const a={present_to_absent:[],absent_to_present:[],both_present:[],both_absent:[],unknown:[]};for(const p of pairs){const b=p.legacy.flags[t][key],n=p.intersection.flags[t][key];const k=typeof b!=="boolean"||typeof n!=="boolean"?"unknown":b&&n?"both_present":!b&&!n?"both_absent":b?"present_to_absent":"absent_to_present";a[k].push(p.id);}return a;};
const badge=(v,yes,no)=>`<span class="badge ${v===true?"":v===false?"bad":"gray"}">${esc(v===true?yes:v===false?no:"未知")}</span>`;
function countButton(count,key,method,group){return `<button class="count" data-key="${esc(key)}" data-method="${method}" data-group="${group}" ${count?"":"disabled"}>${count}</button>`;}
function triCell(values,key,method){const den=activePairs.length;return `<div>${["positive","negative","unknown"].map(s=>countButton(values[s].length,key,method,s)).join(" / ")}</div><div class="bar">${["positive","negative","unknown"].map(s=>`<span class="${s}" style="width:${100*values[s].length/den}%"></span>`).join("")}</div>`;}
function showIds(ids,title,note){selection={ids,title,note};$("selection-title").textContent=title;$("selection-count").textContent=`${ids.length}件`;$("selection-note").textContent=note;$("selected-ids").innerHTML=ids.map(id=>`<button class="id" data-id="${id}">${id}</button>`).join("")||'<span class="empty">本组没有对应对象。</span>';}
function displayPair(id){const p=activePairs.find(p=>p.id===id);activeCase=p?.id??null;if(!p){$("case-detail").innerHTML='<p class="empty">这个ID不在当前对象组中，可切换分组后查看。</p>';return;}const t=$("threshold").value;$("case-hint").innerHTML=`<b>${esc(id)}</b> · ${esc(D.cohort_labels[p.cohort])} ${p.in_atlas?`· <a href="../../../joint_boundary/legacy_failures_20261011/page/#${id}">查看原版3D案例</a>`:"· 未收录在原50件3D图库"}`;$("case-detail").innerHTML=`<div class="pair-grid">${["legacy","intersection"].map(method=>{const r=p[method],faces=r.failed_face_context.faces;const context=faces===null?"失败面集合未知":faces.length===0?"没有最终失败的裁剪面":faces.map(f=>`面${f.face_id}: ${f.codes.length?f.codes.map(c=>esc(D.context_labels[c]||c)).join("；"):f.inspection_complete?"未见这些上下文错误":"没有完整上下文检查"}`).join("<br>");const positive=D.observations.filter(o=>r.flags[t][o.key]===true);const unknown=D.observations.filter(o=>r.flags[t][o.key]===null);return `<article class="method"><h3>${method==="legacy"?"原版 HoLa":"Intersection refinement"}</h3>${badge(r.original_native_valid,"原结果：合法实体","原结果：失败")} ${badge(r.trace_verified,"重放完整一致","阶段证据未核验")}<p><b>${esc(D.terminal_labels[r.terminal_stage])}</b></p><p class="tiny">独立拟合→周期替换后最大边面残差：${number(r.owner_stages.independent_curve.observed_max)} → ${number(r.owner_stages.after_periodic_replacement.observed_max)}<br>优化后面MAE变化：${number(r.optimization_delta.face_xyz_mae)}；边距离变化：${number(r.optimization_delta.matched_edge_mean_length)}</p><b class="tri">当前阈值确认观察到</b><ul>${positive.map(o=>`<li>${esc(o.label)}</li>`).join("")||"<li>未确认上述观察</li>"}</ul><details><summary>未知的观察 ${unknown.length}项</summary><ul>${unknown.map(o=>`<li>${esc(o.label)}</li>`).join("")}</ul></details><details><summary>最终失败面的候选检查</summary><p>${context}</p></details><details><summary>轨迹文件及SHA256</summary>${Object.entries(r.source_files).map(([name,source])=>`<p>${esc(name)}<br><code>${esc(source.path)}</code><br><code>${esc(source.sha256)}</code></p>`).join("")}</details></article>`;}).join("")}</div>`;}
function median(values){const a=values.filter(v=>typeof v==="number"&&Number.isFinite(v)).sort((a,b)=>a-b);return a.length? a.length%2?a[(a.length-1)/2]:(a[a.length/2-1]+a[a.length/2])/2:null;}
function ownerTable(){const metrics=[["independent_curve","独立拟合后的最大残差中位数"],["after_periodic_replacement","周期替换后的最大残差中位数"]];$("owner-summary").querySelector("tbody").innerHTML=metrics.map(([key,label])=>`<tr><td>${label}</td>${["legacy","intersection"].map(m=>{const v=activePairs.map(p=>p[m].owner_stages[key]).filter(v=>v.status==="ok").map(v=>v.observed_max);return `<td>${number(median(v))} <small>完整测量${v.length}/${activePairs.length}件</small></td>`;}).join("")}</tr>`).join("")+`<tr><td>周期替换后对象最大值：下降 / 不变 / 增加 / 未知</td>${["legacy","intersection"].map(m=>{const v=activePairs.map(p=>p[m].owner_stages.object_max_delta);return `<td>${v.filter(x=>typeof x==="number"&&x< -1e-12).length} / ${v.filter(x=>typeof x==="number"&&Math.abs(x)<=1e-12).length} / ${v.filter(x=>typeof x==="number"&&x>1e-12).length} / ${v.filter(x=>x===null).length}</td>`;}).join("")}</tr>`;}
function contextTable(){const all={},coverage={};for(const m of ["legacy","intersection"]){const r=activePairs.map(p=>p[m]);const faces=r.flatMap(r=>r.failed_face_context.faces||[]);const codes=new Set(faces.flatMap(f=>f.codes));coverage[m]=`${faces.length}个最终失败面，${faces.filter(f=>f.inspection_complete).length}个有完整候选检查；${r.filter(r=>r.failed_face_context.faces===null).length}件的失败面集合未知`;for(const code of codes){all[code]??={legacy:{faces:0,ids:[],direct:0,context:0},intersection:{faces:0,ids:[],direct:0,context:0}};all[code][m]={faces:faces.filter(f=>f.codes.includes(code)).length,direct:faces.filter(f=>f.direct_codes.includes(code)).length,context:faces.filter(f=>f.context_codes.includes(code)).length,ids:activePairs.filter(p=>(p[m].failed_face_context.faces||[]).some(f=>f.codes.includes(code))).map(p=>p.id)};}}$("context-coverage").textContent=`原版：${coverage.legacy}；新版：${coverage.intersection}`;$("context-summary").querySelector("tbody").innerHTML=Object.entries(all).sort((a,b)=>b[1].intersection.faces-a[1].intersection.faces).map(([code,v])=>`<tr><td>${esc(D.context_labels[code]||"其他上下文错误")}<br><code>${esc(code)}</code></td>${["legacy","intersection"].map(m=>`<td class="num">${v[m].faces} / <button class="count" data-context="${esc(code)}" data-method="${m}">${v[m].ids.length}</button> / ${v[m].direct} / ${v[m].context}</td>`).join("")}</tr>`).join("")||'<tr><td colspan="3">当前组未确认这些上下文错误。检查缺失不表示检查通过。</td></tr>';}
function render(){const cohort=$("cohort").value,t=$("threshold").value;activePairs=D.pairs.filter(p=>cohort==="main"?!p.intersection.original_native_valid:p.cohort===cohort);const den=activePairs.length;const valid=m=>activePairs.filter(p=>p[m].original_native_valid).length;const verified=m=>activePairs.filter(p=>p[m].trace_verified).length;$("cards").innerHTML=`<div class="card"><strong>当前固定分母</strong><b id="active-denominator">${den}</b><span class="muted">${cohort==="main"?"新版失败全集":esc(D.cohort_labels[cohort])}</span></div><div class="card"><strong>原版 → 新版合法实体</strong><b>${valid("legacy")} → ${valid("intersection")}</b><span class="muted">原评估结果；缺测不改结果</span></div><div class="card"><strong>原版 / 新版完整重放</strong><b>${verified("legacy")} / ${verified("intersection")}</b><span class="muted">两者分母均为${den}</span></div><div class="card"><strong>全测试有效性</strong><b>2048 → 2113</b><span class="muted">接受集2417 / 原始集2424；7不支持</span></div>`;$("active-caption").textContent=cohort==="main"?"当前304件都在新版失败。其中262件旧版也失败，42件旧版成功。因此此处原版列也是这304件的原版观测，不是旧369失败全集。":"当前两列始终是相同ID集合。标签消失/出现只是观察变化，不能据此确定唯一根因。";$("mechanisms").querySelector("tbody").innerHTML=D.observations.map(o=>{const a=stat(activePairs,"legacy",o.key,t),b=stat(activePairs,"intersection",o.key,t),x=transition(activePairs,o.key,t);return `<tr data-observation="${o.key}"><td class="name"><b>${esc(o.label)}</b>${o.uses_threshold?' <small>随阈值</small>':""}<details><summary>具体怎么判断</summary>${esc(o.criterion)}</details></td><td class="num">${triCell(a,o.key,"legacy")}</td><td class="num">${triCell(b,o.key,"intersection")}</td><td class="num">${countButton(x.present_to_absent.length,o.key,"paired","present_to_absent")}</td><td class="num">${countButton(x.absent_to_present.length,o.key,"paired","absent_to_present")}</td><td class="num">${countButton(x.unknown.length,o.key,"paired","unknown")}</td></tr>`;}).join("");ownerTable();contextTable();showIds(activePairs.map(p=>p.id),"当前对象组的全部ID",`分母${den}；点击ID查看同一件的原版与新版。`);if(activeCase)displayPair(activeCase);window.INTERSECTION_CENSUS_VIEW={cohort,threshold:t,denominator:den,ids:activePairs.map(p=>p.id)};}
$("mechanisms").addEventListener("click",event=>{const b=event.target.closest("button[data-key]");if(!b)return;const {key,method,group}=b.dataset,o=D.observations.find(o=>o.key===key),t=$("threshold").value;const ids=method==="paired"?transition(activePairs,key,t)[group]:stat(activePairs,method,key,t)[group];const label=method==="paired"?{present_to_absent:"原有→新无",absent_to_present:"原无→新有",unknown:"任一侧未知"}[group]:(method==="legacy"?"原版":"新版")+" · "+names[group];showIds(ids,`${o.label} · ${label}`,o.criterion);$("selection").scrollIntoView({behavior:"smooth",block:"start"});});
$("selected-ids").addEventListener("click",event=>{const b=event.target.closest("button[data-id]");if(b){displayPair(b.dataset.id);$("case-view").scrollIntoView({behavior:"smooth",block:"start"});}});
$("context-summary").addEventListener("click",event=>{const b=event.target.closest("button[data-context]");if(!b)return;const {context,method}=b.dataset;const ids=activePairs.filter(p=>(p[method].failed_face_context.faces||[]).some(f=>f.codes.includes(context))).map(p=>p.id);showIds(ids,`${method==="legacy"?"原版":"新版"} · ${D.context_labels[context]||context}`,"同一最终失败面上的候选/上下文检查错误；跨容差尝试每面去重。");$("selection").scrollIntoView({behavior:"smooth"});});
for(const id of ["cohort","threshold"])$(id).addEventListener("change",render);
$("case-search").addEventListener("input",event=>{if(/^\d{8}$/.test(event.target.value))displayPair(event.target.value);});
$("figures").innerHTML=D.charts.map(c=>`<figure><a href="../${c.png}" target="_blank"><img src="../${c.png}" alt="${esc(c.caption)}" loading="lazy"></a><figcaption>${esc(c.caption)} <a href="../${c.svg}">SVG</a></figcaption></figure>`).join("");
$("limitations").innerHTML=D.limitations.map(line=>`<li>${esc(line)}</li>`).join("");render();
$("highlight-text").innerHTML=D.highlights.map(line=>`<p class="notice">${esc(line)}</p>`).join("");
'''


def compact_summary(summary: dict) -> dict:
    """Versioned count tables without repeated per-ID or per-face evidence.

    Ordered numeric columns avoid repeating long JSON field names thousands of
    times. Exact definitions and source references remain in this small file;
    full IDs/trajectories stay in summary.json and cases.json locally.
    """
    observation_columns = ("positive", "negative", "unknown")
    transition_columns = ("present_to_absent", "absent_to_present", "both_present", "both_absent", "unknown")
    def reduce_group(group):
        context = group["failed_face_context"]
        return {"denominator": group["denominator"], "native_valid": group["native_valid"],
                "verified_traces": group["verified_traces"], "raw_topology_counts": group["raw_topology_counts"],
                "terminal": {key: value["count"] for key, value in group["terminal"].items()},
                "observations": {t: {key: [value[c] for c in observation_columns] for key, value in obs.items()} for t, obs in group["observations"].items()},
                "earliest_observation": {t: {key: [v["count"], v["with_earlier_unknown_count"]] for key, v in values.items()} for t, values in group["earliest_observation"].items()},
                "owner_stages": group["owner_stages"],
                "failed_face_checks": {"counted_final_failed_faces": context["counted_final_failed_faces"],
                                       "inspection_states": context["inspection_states"], "uncountable_cad_count": len(context["uncountable_ids"]),
                                       "codes": {code: {"face_count": v["face_count"], "direct_face_count": v["direct_face_count"],
                                                        "context_face_count": v["context_face_count"], "cad_counts": [v["cad"][c] for c in observation_columns]}
                                                 for code, v in context["codes"].items()}},
                "failed_face_local_topology": {key: {"face_count": len(values), "cad_count": len({v["id"] for v in values})}
                                               for key, values in group["failed_face_local_topology"].items()}}
    audit = summary["audit"]
    six = summary["optimization_counterfactual"]["cases"]
    return {
        "schema": "intersection_failure_compact_summary_v1", "created_at_utc": summary["created_at_utc"],
        "population": summary["population"], "highlights": summary["highlights"],
        "columns": {"observations_and_cad_counts": list(observation_columns), "transitions": list(transition_columns),
                    "earliest_observation": ["count", "with_earlier_unknown_count"]},
        "definitions": {key: {"label": label, "criterion": criterion, "uses_threshold": threshold} for key, label, _, criterion, threshold in OBSERVATIONS},
        "context_error_labels": CONTEXT_ERRORS,
        "main": reduce_group(summary["main"]), "historical_failures": reduce_group(summary["historical_failures"]),
        "cohorts": {key: {"label": value["label"], "count": value["count"],
                          "legacy": reduce_group(value["legacy"]), "intersection": reduce_group(value["intersection"]),
                          "transitions": {t: {name: [v["counts"][c] for c in transition_columns] for name, v in rows.items()}
                                          for t, rows in value["transitions"].items()}}
                    for key, value in summary["cohorts"].items()},
        "fit_only_vs_optimized_all_accepted": summary["fit_only_vs_optimized_all_accepted"],
        "audit": {**{name: {"checked_count": audit[name]["checked_count"], "terminal_diagnostic_error_count": len(audit[name]["terminal_diagnostic_errors"])}
                     for name in ("intersection", "legacy_controls", "legacy_historical")},
                  "baseline_reuse_eligible_count": audit["baseline_reuse"]["eligible_count"],
                  "source_preservation": {k: audit.get("source_preservation", {}).get(k) for k in ("asset_count", "mismatch_count", "sha256")}},
        "optimization_counterfactual": {"count": len(six),
            "verified_fit_traces": sum(r["fit_only"]["trace_verified"] for r in six),
            "verified_optimized_traces": sum(r["optimized"]["trace_verified"] for r in six),
            "same_raw_input_count": sum(r["same_raw_input_sha256"] is True for r in six),
            "same_selected_arrays_count": sum(all(v is True for v in r["same_selected_arrays"].values()) for r in six),
            "ids": [r["id"] for r in six], "details": "optimization_regressions.json"},
        "sources": summary["sources"], "limitations": summary["limitations"],
    }


def write_outputs(root: Path, pairs: list[dict], summary: dict, charts: list[dict]) -> None:
    def write_json(name, value):
        (root / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    write_json("summary.json", summary)
    compact = compact_summary(summary)
    compact["full_summary_sha256"] = legacy.file_sha256(root / "summary.json")
    (root / "compact_summary.json").write_text(json.dumps(compact, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
    write_json("cases.json", {"schema": VERSION, "pairs": pairs})
    with (root / "cases.csv").open("w", newline="") as handle:
        fields = ["id", "cohort", "legacy_native_valid", "intersection_native_valid", "legacy_trace_verified", "intersection_trace_verified", "legacy_terminal", "intersection_terminal"]
        fields += [f"{method}__{threshold}__{key}" for method in ("legacy", "intersection") for threshold in THRESHOLDS for key, *_ in OBSERVATIONS]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for p in pairs:
            row = {"id": p["id"], "cohort": p["cohort"]}
            for method in ("legacy", "intersection"):
                source = p[method]
                row.update({f"{method}_native_valid": source["original_native_valid"], f"{method}_trace_verified": source["trace_verified"], f"{method}_terminal": source["terminal_stage"]})
                row.update({f"{method}__{t}__{key}": "true" if source["flags"][t][key] is True else "false" if source["flags"][t][key] is False else "unknown" for t in THRESHOLDS for key, *_ in OBSERVATIONS})
            writer.writerow(row)
    (root / "REPORT.md").write_text(report_markdown(summary, charts))
    atlas_list = Path(__file__).resolve().parents[1] / "checkpoints/joint_boundary/legacy_failures_20261011/selected_ids.txt"
    atlas_ids = set(atlas_list.read_text().split()) if atlas_list.is_file() else set()
    payload = page_payload(pairs, summary, charts, atlas_ids)
    page = root / "page"; page.mkdir(exist_ok=True)
    (page / "index.html").write_text(PAGE_HTML)
    (page / "app.js").write_text(PAGE_JS)
    (page / "data.js").write_text("window.INTERSECTION_CENSUS_DATA=" + json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False).replace("</", "<\\/") + ";\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--census-root", type=Path, required=True)
    parser.add_argument("--legacy-census-root", type=Path, required=True)
    parser.add_argument("--font-path", type=Path, help="CJK font file for the Chinese static charts")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate only report artifacts after validating the frozen diagnostic sources")
    args = parser.parse_args(argv)
    root, old = args.census_root.resolve(), args.legacy_census_root.resolve()
    if root == old or root.is_relative_to(old):
        raise ValueError("Report output must not write into the historical census")
    if not args.overwrite and any((root / name).exists() for name in ("REPORT.md", "summary.json", "cases.json", "cases.csv", "page/index.html")):
        raise FileExistsError("Report artifacts already exist; explicitly use --overwrite to regenerate")
    pairs, summary = build_report(root, old)
    charts = make_charts(summary, root / "visualizations", args.font_path)
    write_outputs(root, pairs, summary, charts)
    print(json.dumps({"root": str(root), "main_failed": summary["main"]["denominator"], "paired_cases": len(pairs),
                      "verified_main_traces": summary["main"]["verified_traces"], "cohorts": {key: item["count"] for key, item in summary["cohorts"].items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
