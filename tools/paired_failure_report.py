"""Audited paired census for previous refinement versus an explicit candidate."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from tools.failure_census_contract import case_method, case_path, case_record
from tools import intersection_failure_report as measures
from tools import legacy_failure_census_report as legacy
from tools.paired_failure_manifest import CANDIDATE_METHODS, LABELS as METHOD_LABELS, paired_methods, read_json

VERSION = "paired_failure_report_v1"
THRESHOLDS = measures.THRESHOLDS
OBSERVATIONS = measures.OBSERVATIONS
CONTEXT_ERRORS = {
    **measures.CONTEXT_ERRORS,
    "BRepCheck_NoCurveOnSurface": "候选边缺少所属面上的参数曲线",
    "BRepCheck_InvalidSameRangeFlag": "边的三维与面上曲线参数范围检查未通过",
}
TERMINALS = measures.TERMINALS
COHORTS = {"persistent": "两版都失败", "regressed": "上一版成功、本版失败", "rescued": "上一版失败、本版成功"}
GROUP_MAP = {"persistent": "both_failed", "regressed": "new_failure", "rescued": "rescued"}
ROLES = ("baseline", "candidate")
LABELS = {"baseline": "上一版 Intersection refinement", "candidate": "本版局部注意力"}


def validate_cohorts(candidate, baseline, frozen, candidate_method="local"):
    paired_methods(candidate_method)
    def index(cases, method):
        result = {}
        for case in cases:
            sid = case["id"]
            if sid in result or case_method(case) != method:
                raise ValueError(f"Duplicate ID or wrong method: {method}/{sid}")
            if type(case_record(case, "post").get("native_valid_solid")) is not bool:
                raise ValueError(f"Unknown native validity: {method}/{sid}")
            result[sid] = case
        return result
    new, old = index(candidate, candidate_method), index(baseline, "intersection")
    if new.keys() != old.keys():
        raise ValueError("Paired methods must have identical census IDs")
    groups = {key: [] for key in COHORTS}
    for sid in sorted(new):
        a, b = case_record(old[sid], "post")["native_valid_solid"], case_record(new[sid], "post")["native_valid_solid"]
        if a and b:
            raise ValueError(f"Both-success case outside failure census: {sid}")
        groups["regressed" if a else "rescued" if b else "persistent"].append(sid)
    for group, source_key in GROUP_MAP.items():
        if groups[group] != sorted(frozen[source_key]) or len(set(frozen[source_key])) != len(frozen[source_key]):
            raise ValueError(f"Frozen cohort differs: {group}")
    return groups


def paired_transitions(pairs, key, threshold):
    ids = {name: [] for name in ("present_to_absent", "absent_to_present", "both_present", "both_absent", "unknown")}
    for pair in pairs:
        a, b = [pair[role]["flags"][threshold].get(key) for role in ROLES]
        name = "unknown" if type(a) is not bool or type(b) is not bool else "both_present" if a and b else "both_absent" if not a and not b else "present_to_absent" if a else "absent_to_present"
        ids[name].append(pair["id"])
    return {"denominator": len(pairs), "counts": {key: len(value) for key, value in ids.items()}, "ids": ids}


def paired_continuous(pairs):
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
        rows = [{"id": p["id"], **{r: getter(p[r]) for r in ROLES}} for p in pairs]
        complete = [row for row in rows if all(legacy.finite(row[r]) for r in ROLES)]
        result[key] = {"denominator": len(pairs), "paired_count": len(complete),
                       **{r: legacy.distribution(row[r] for row in complete) for r in ROLES},
                       "candidate_minus_baseline": legacy.distribution(row["candidate"] - row["baseline"] for row in complete),
                       "paired_ids": [row["id"] for row in complete], "unknown_ids": [row["id"] for row in rows if row not in complete]}
    return result


def diagnostic_audit(manifest, root, manifest_path, **kwargs):
    if not manifest["cases"]:
        return {"checked_count": 0, "checked_ids": [], "protocols": {}, "terminal_diagnostic_errors": [], "empty_scope": True}
    return measures.validate_diagnostics(manifest, root, manifest_path, **kwargs)


def optimization_counterfactual(manifest, root, records):
    ids = manifest.get("optimization_regression_ids")
    if not isinstance(ids, list) or len(set(ids)) != len(ids):
        raise ValueError("Invalid frozen optimization-regression IDs")
    note = "相同原输入的fit-only与optimized完整路径比较；后续拟合/裁剪也会变化，不是单一优化loss因果消融。"
    if not ids:
        return {"denominator": 0, "audit": {"checked_count": 0, "empty_scope": True}, "cases": [], "note": note}
    import numpy as np
    cases = {case["id"]: case for case in manifest["cases"]}
    if set(ids) - cases.keys():
        raise ValueError("Optimization regression outside selected candidate census")
    subset = dict(manifest, cases=[cases[sid] for sid in ids])
    fit_root = root / "fit_only_counterfactual"
    audit = diagnostic_audit(subset, fit_root, root / "manifest.json", saved_required=False, post_mode="fit_only")
    fit_manifest = dict(subset, cases=[dict(case, post_record=case_record(case, "fit")) for case in subset["cases"]])
    fits = {row["id"]: row for row in measures.enrich_records(fit_manifest, fit_root)}
    optimized = {row["id"]: row for row in records}
    rows = []
    for sid in ids:
        case = cases[sid]
        if case_record(case, "fit")["native_valid_solid"] is not True or case_record(case, "post")["native_valid_solid"] is not False:
            raise ValueError(f"Counterfactual source outcomes differ: {sid}")
        equality = {key: None for key in ("selected_faces", "selected_edges", "selected_connectivity")}
        if fits[sid]["trace_verified"] and optimized[sid]["trace_verified"]:
            paths = [p / "replay" / sid / "trace_geometry.npz" for p in (fit_root, root)]
            if all(path.is_file() for path in paths):
                with np.load(paths[0], allow_pickle=False) as a, np.load(paths[1], allow_pickle=False) as b:
                    equality = {key: bool(np.array_equal(a[key], b[key])) if key in a and key in b else None for key in equality}
        rows.append({"id": sid, "fit_only": fits[sid], "optimized": optimized[sid], "same_selected_arrays": equality,
                     "same_raw_input_sha256": case_record(case, "fit")["raw_sha256"] == case_record(case, "post")["raw_sha256"]})
    return {"denominator": len(ids), "audit": audit, "cases": rows, "note": note}


def audit_source_preservation(root):
    source = root / "source_asset_audit.json"
    audit = read_json(source)
    checked, changed = [], []
    for row in audit["source_assets"]:
        path = Path(row["path"])
        actual = legacy.file_sha256(path) if path.is_file() else None
        same = row["exists"] == path.is_file() and row["sha256"] == actual
        checked.append({"path": str(path), "expected_sha256": row["sha256"], "actual_sha256": actual, "same": same})
        if not same:
            changed.append(str(path))
    result = {"schema": "source_preservation_v1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "source_audit_sha256": legacy.file_sha256(source), "asset_count": len(checked),
              "mismatch_count": len(changed), "changed_paths": changed, "assets": checked}
    (root / "source_preservation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    if changed:
        raise ValueError(f"Frozen source assets changed: {changed}")
    return {key: value for key, value in result.items() if key != "assets"}


def build_report(root, baseline_root):
    paths = {name: root / name for name in ("manifest.json", "baseline_manifest.json", "baseline_control_manifest.json", "baseline_reuse.json")}
    manifest, baseline, controls, reuse = [read_json(paths[name]) for name in paths]
    candidate_method = manifest.get("candidate_method")
    if manifest.get("baseline_method") != "intersection" or candidate_method not in CANDIDATE_METHODS:
        raise ValueError("Wrong paired methods")
    if manifest.get("method") != candidate_method or any(
            document.get("method") != "intersection" or document.get("candidate_method") != candidate_method
            for document in (baseline, controls)):
        raise ValueError("Paired manifest identities differ")
    labels = {"baseline": METHOD_LABELS["intersection"], "candidate": METHOD_LABELS[candidate_method]}
    groups = validate_cohorts(manifest["cases"], baseline["cases"], manifest["cohorts"], candidate_method)
    population = manifest["population"]
    if (population.get("accepted_test") != 2417 or population.get("original_test") != 2424
            or population.get("contract_unsupported") != 7
            or population.get("native_failed") != len(groups["persistent"]) + len(groups["regressed"])
            or population.get("native_valid") + population.get("native_failed") != population.get("accepted_test")):
        raise ValueError("Frozen population does not reconcile")
    original_path = baseline_root / "manifest.json"
    original = read_json(original_path)
    if (Path(reuse["baseline_manifest"]["path"]).resolve() != original_path.resolve()
            or reuse["baseline_manifest"]["sha256"] != legacy.file_sha256(original_path)):
        raise ValueError("Historical baseline manifest changed")
    reuse_by_id = {row["id"]: row for row in reuse["cases"]}
    selected = {case["id"] for case in manifest["cases"]}
    if set(reuse_by_id) != selected or len(reuse_by_id) != len(reuse["cases"]):
        raise ValueError("Reuse audit must cover exact paired census")
    eligible = {sid for sid, row in reuse_by_id.items() if row["eligible"]}
    if len(eligible) != reuse["eligible_count"]:
        raise ValueError("Reuse eligible count differs")
    control_ids = {case["id"] for case in controls["cases"]}
    if eligible & control_ids or eligible | control_ids != selected or len(control_ids) != len(controls["cases"]):
        raise ValueError("Reuse and newly replayed controls must partition baseline identities")
    for sid in eligible:
        for source in reuse_by_id[sid]["sources"].values():
            if not source.get("exists") or legacy.file_sha256(Path(source["path"])) != source.get("sha256"):
                raise ValueError(f"Historical diagnostic changed: {sid}/{source['path']}")
    old_cases = [case for case in original["cases"] if case["id"] in eligible]
    if len(old_cases) != len(eligible):
        raise ValueError("Historical reused IDs absent from original manifest")
    old_subset = dict(original, cases=old_cases)
    audit = {"candidate": diagnostic_audit(manifest, root, paths["manifest.json"]),
             "baseline_controls": diagnostic_audit(controls, root / "baseline_controls", paths["baseline_control_manifest.json"]),
             "baseline_historical": diagnostic_audit(old_subset, baseline_root, original_path),
             "baseline_reuse": {"eligible_count": len(eligible), "path": str(paths["baseline_reuse.json"]),
                                "sha256": legacy.file_sha256(paths["baseline_reuse.json"])},
             "source_preservation": audit_source_preservation(root)}
    new = {row["id"]: row for row in measures.enrich_records(manifest, root)}
    old_rows = measures.enrich_records(old_subset, baseline_root) + measures.enrich_records(controls, root / "baseline_controls")
    old = {row["id"]: row for row in old_rows}
    if old.keys() != new.keys() or len(old_rows) != len(old):
        raise ValueError("Enriched paired identities differ")
    cohort_of = {sid: key for key, ids in groups.items() for sid in ids}
    pairs = [{"id": sid, "cohort": cohort_of[sid], "baseline": old[sid], "candidate": new[sid]} for sid in sorted(new)]
    cohort_results = {}
    for key, ids in groups.items():
        subset = [pair for pair in pairs if pair["id"] in set(ids)]
        cohort_results[key] = {"label": COHORTS[key], "count": len(ids), "ids": ids,
            **{role: measures.summarize_records([pair[role] for pair in subset]) for role in ROLES},
            "transitions": {t: {name: paired_transitions(subset, name, t) for name, *_ in OBSERVATIONS} for t in THRESHOLDS},
            "continuous": paired_continuous(subset)}
    summary = {"schema": VERSION, "created_at_utc": datetime.now(timezone.utc).isoformat(),
               "population": population, "methods": {"baseline": "intersection", "candidate": candidate_method}, "method_labels": labels,
               "main": measures.summarize_records([row for row in new.values() if not row["original_native_valid"]]),
               "baseline_failures": measures.summarize_records([row for row in old.values() if not row["original_native_valid"]]),
               "cohorts": cohort_results, "audit": audit,
               "fit_only_vs_optimized_all_accepted": manifest["fit_only_vs_optimized_all_accepted"],
               "optimization_counterfactual": optimization_counterfactual(manifest, root, list(new.values())),
               "sources": {name: {"path": str(path), "sha256": legacy.file_sha256(path)} for name, path in paths.items()},
               "limitations": [
                   "主分母只含本版失败；救回对照按结果选择，不代表全部成功件。接受集2417，原始2424中7件合同不支持。",
                   "同一ID的观察变化不等于因果改善；任何一侧未知都不算标签消失。多标签可以重叠，不能相加为失败总数。",
                   "只在原终态、STEP DATA精确一致且没有观测错误时采用阶段轨迹；缺测保留原分母。",
                   "边面距离是固定采样的归一化XYZ长度，不是毫米、Hausdorff距离或可裁剪性的证明。",
                   "面级错误只统计最终失败的decoder面；候选自身与所属上下文去重合并，OCC错误码不是唯一根因。",
                   f"本次额外微调曝光与上一版不同，无法把全部收益或退步因果归于{labels['candidate']}；不代表端到端照片重建结果。",
                   "原HoLa的50件3D图库只作历史参考，页面不将其视作本版几何。"]}
    summary["highlights"] = highlights(summary)
    return pairs, summary


def highlights(summary):
    main, population = summary["main"], summary["population"]
    obs = main["observations"]["0.006"]
    rescued = summary["cohorts"]["rescued"]
    return [
        f"本版失败{main['denominator']}件：持续失败{summary['cohorts']['persistent']['count']}件，退步{summary['cohorts']['regressed']['count']}件；另有{rescued['count']}件救回。固定2417件中合法实体由{population['baseline_native_valid']}变为{population['native_valid']}。",
        f"失败件中，原始无向面连接完全正确有{main['raw_topology_counts'].get('exact', 0)}件。某次全部候选环均闭合并接纳，面仍裁不出来有{obs['strict_closed_candidate_failed']['positive']}件；该项未知{obs['strict_closed_candidate_failed']['unknown']}件。",
        f"阈值0.006下，本版失败件中{obs['owner_after']['positive']}件在周期替换后仍观察到边面贴合偏差（未知{obs['owner_after']['unknown']}件）；救回组也有{rescued['candidate']['observations']['0.006']['owner_after']['positive']}/{rescued['count']}件出现该观察。超阈值不是失败的充分条件。",
        f"本版存在{summary['optimization_counterfactual']['denominator']}件fit-only成功而优化后失败，均纳入完整路径对照；比较的是整体路径，不是某个loss的因果消融。",
    ]


def compact_summary(summary):
    columns = ("positive", "negative", "unknown")
    transition_columns = ("present_to_absent", "absent_to_present", "both_present", "both_absent", "unknown")
    def group(value):
        context = value["failed_face_context"]
        return {"denominator": value["denominator"], "native_valid": value["native_valid"],
                "verified_traces": value["verified_traces"], "raw_topology_counts": value["raw_topology_counts"],
                "terminal": {k: v["count"] for k, v in value["terminal"].items()},
                "observations": {t: {k: [v[c] for c in columns] for k, v in rows.items()} for t, rows in value["observations"].items()},
                "owner_stages": value["owner_stages"],
                "failed_face_checks": {"counted_final_failed_faces": context["counted_final_failed_faces"],
                    "inspection_states": context["inspection_states"], "uncountable_cad_count": len(context["uncountable_ids"]),
                    "codes": {k: {"face_count": v["face_count"], "direct_face_count": v["direct_face_count"],
                        "context_face_count": v["context_face_count"], "cad_counts": [v["cad"][c] for c in columns]} for k, v in context["codes"].items()}},
                "earliest_observation": {t: {k: [v["count"], v["with_earlier_unknown_count"]] for k, v in rows.items()} for t, rows in value["earliest_observation"].items()}}
    result = {key: summary[key] for key in ("schema", "created_at_utc", "population", "methods", "method_labels", "highlights", "sources", "limitations", "fit_only_vs_optimized_all_accepted")}
    result.update(schema="paired_failure_compact_summary_v1", columns={"observations": list(columns), "transitions": list(transition_columns)},
                  definitions={key: {"label": label, "criterion": criterion, "uses_threshold": threshold} for key, label, _, criterion, threshold in OBSERVATIONS},
                  main=group(summary["main"]), baseline_failures=group(summary["baseline_failures"]),
                  cohorts={key: {"count": value["count"], **{r: group(value[r]) for r in ROLES},
                     "transitions": {t: {k: [v["counts"][c] for c in transition_columns] for k, v in rows.items()} for t, rows in value["transitions"].items()},
                     "continuous": value["continuous"]} for key, value in summary["cohorts"].items()},
                  audit={key: {k: v for k, v in value.items() if k not in ("checked_ids", "assets")} for key, value in summary["audit"].items()},
                  optimization_counterfactual={"count": summary["optimization_counterfactual"]["denominator"],
                     "ids": [r["id"] for r in summary["optimization_counterfactual"]["cases"]],
                     "verified_fit_traces": sum(r["fit_only"]["trace_verified"] for r in summary["optimization_counterfactual"]["cases"]),
                     "verified_optimized_traces": sum(r["optimized"]["trace_verified"] for r in summary["optimization_counterfactual"]["cases"]),
                     "details": "optimization_regressions.json"})
    return result


def make_charts(summary: dict, directory: Path, font_path: Path | None = None) -> list[dict]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager, ticker
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
    denominator = summary["main"]["denominator"]
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
    ax.set_xlim(0, max(1, denominator)); ax.set_xlabel(f"件数 / 本版{denominator}个失败件（同一件可有多种问题）")
    ax.set_title("哪些现象反复出现？距离项阈值为归一化长度0.006")
    ax.legend(loc="lower right"); ax.grid(axis="x", alpha=.15); fig.tight_layout()
    save(fig, "main_failure_observations", f"本版{denominator}失败件的多标签观察；灰色为未知，留白为确认未观察到。标签可以重叠。距离项阈值0.006。")
    comparison_keys = ["never_wire", "never_closed_admitted", "closed_admitted_still_failed", "strict_closed_candidate_failed", "final_success_discarded_wire", "owner_after", "optimized_endpoint_gap", "raw_extra_gt_pair", "raw_missing_gt_pair"]
    fig, axes = plt.subplots(1, 3, figsize=(17, 6), sharey=True)
    for ax, (cohort, title) in zip(axes, COHORTS.items()):
        data = summary["cohorts"][cohort]
        transition = data["transitions"][threshold]
        y = np.arange(len(comparison_keys))
        ax.barh(y - .19, [transition[k]["counts"]["present_to_absent"] for k in comparison_keys], .36, color="#277e83", label="原有→新无")
        ax.barh(y + .19, [transition[k]["counts"]["absent_to_present"] for k in comparison_keys], .36, color="#c9714d", label="原无→新有")
        ax.set_yticks(y, [labels[k] for k in comparison_keys]); ax.set_xlabel("成对比较的件数")
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.set_title(f"{title} (n={data['count']})"); ax.grid(axis="x", alpha=.15)
    axes[0].invert_yaxis(); axes[0].legend(loc="lower right", fontsize=8); fig.tight_layout()
    save(fig, "paired_observation_changes", "相同ID的标签变化；各组分母见标题。任一侧未知不画成消失或出现；完整未知和稳定计数在页面及JSON中。")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
    for key, label, color in (("owner_independent", "独立拟合后", "#277e83"), ("owner_after", "周期曲线替换后", "#c9714d")):
        axes[0].plot([float(t) for t in THRESHOLDS], [summary["main"]["observations"][t][key]["positive"] for t in THRESHOLDS], "o-", color=color, label=label)
    axes[0].set_xscale("log"); axes[0].set_xticks([float(t) for t in THRESHOLDS], THRESHOLDS)
    axes[0].set_ylim(0, max(1, denominator * 1.04)); axes[0].set_xlabel("归一化XYZ长度阈值")
    axes[0].set_ylabel("至少一条边贴合所属面残差超限的件数")
    axes[0].set_title(f"同一批{denominator}失败件：边面贴合前后对照"); axes[0].legend(fontsize=9)
    change = summary["main"]["owner_stages"]["change_counts"]
    unknown = denominator - sum(change.values())
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



def report_markdown(summary, charts):
    main = summary["main"]
    den = main["denominator"]
    lines = [f"# {summary['method_labels']['candidate']}：与上一版 refinement 的失败对照", "",
             *summary["highlights"], "",
             "[交互统计页面](page/index.html) · [紧凑统计](compact_summary.json) · [逐件CSV](cases.csv) · [冻结协议](PLAN.md)", "",
             f"本版失败轨迹完整一致 {main['verified_traces']}/{den}。各行是可重叠观察，未知保留分母；阈值0.006为归一化长度。",
             "", "## 常见观察", "", f"| 观察 | 确认 / {den} | 未知 |", "|---|---:|---:|"]
    observations = main["observations"]["0.006"]
    for key, label, *_ in sorted(OBSERVATIONS, key=lambda row: observations[row[0]]["positive"], reverse=True):
        value = observations[key]
        lines.append(f"| {label} | {value['positive']} | {value['unknown']} |")
    lines.extend(["", "## 最终阻断与最早观察", "", "| 最终阶段 | 件数 |", "|---|---:|"])
    lines.extend(f"| {label} | {main['terminal'][key]['count']} |" for key, label in TERMINALS.items() if key != "native_valid")
    lines.extend(["", "| 最早已观察到的位置 | 件数 | 更早阶段有缺测 |", "|---|---:|---:|"])
    for key, label in [(k, label) for k, label, _ in measures.EARLIEST_STAGES] + [("unknown", "缺测且尚无确认异常"), ("none_in_catalog", "现有目录未确认异常")]:
        value = main["earliest_observation"]["0.006"][key]
        lines.append(f"| {label} | {value['count']} | {value['with_earlier_unknown_count']} |")
    lines.extend(["", "最早取得证据的位置不等于错误最早发生的位置；末端阻断也不等于唯一根因。", "", "## 相同ID的成对变化", "",
                  "| 分组 | 件数 | 上一版完整重放 | 本版完整重放 |", "|---|---:|---:|---:|"])
    for key, group in summary["cohorts"].items():
        lines.append(f"| {COHORTS[key]} | {group['count']} | {group['baseline']['verified_traces']} | {group['candidate']['verified_traces']} |")
    names = {key: label for key, label, *_ in OBSERVATIONS}
    for key, group in summary["cohorts"].items():
        lines.extend(["", f"**{COHORTS[key]}，n={group['count']}**", "", "| 观察 | 原有→新无 | 原无→新有 | 两边都有 | 两边都无 | 未知 |", "|---|---:|---:|---:|---:|---:|"])
        for name in ("raw_extra_gt_pair", "raw_missing_gt_pair", "never_wire", "never_closed_admitted", "strict_closed_candidate_failed", "final_success_discarded_wire", "owner_after", "optimized_endpoint_gap"):
            counts = group["transitions"]["0.006"][name]["counts"]
            lines.append(f"| {names[name]} | " + " | ".join(str(counts[c]) for c in ("present_to_absent", "absent_to_present", "both_present", "both_absent", "unknown")) + " |")
    lines.extend(["", "## 独立拟合与周期替换后的贴合", "", "| 阈值 | 独立拟合后：确认 / 未知 | 周期替换后：确认 / 未知 |", "|---|---:|---:|"])
    for t in THRESHOLDS:
        a, b = [main["observations"][t][key] for key in ("owner_independent", "owner_after")]
        lines.append(f"| {t} | {a['positive']} / {a['unknown']} | {b['positive']} / {b['unknown']} |")
    changes = main["owner_stages"]["change_counts"]
    lines.extend(["", f"同一边/所属面且两阶段测量完整时，对象最大残差下降{changes['decreased']}、增大{changes['increased']}、基本不变{changes['unchanged']}件；其余{den-sum(changes.values())}件未知。周期替换可能改变参数域，不能解释为同一空间采样点的运动。", "",
                  "## 最终失败面的候选检查", "", "| 通俗描述 / 原始错误名 | 面数 | 自身 / 上下文 | 件数 / 未知 |", "|---|---:|---:|---:|"])
    for code, value in sorted(main["failed_face_context"]["codes"].items(), key=lambda row: row[1]["face_count"], reverse=True):
        lines.append(f"| {CONTEXT_ERRORS.get(code, code)} / `{code}` | {value['face_count']} | {value['direct_face_count']} / {value['context_face_count']} | {value['cad']['positive']} / {value['cad']['unknown']} |")
    lines.extend(["", "同一最终失败面同种错误跨容差去重。没有候选可检查或检查不完整，不能称为检查通过。", "", "## fit-only成功、优化后失败", "",
                  "| ID | fit重放一致 | optimized重放一致 | 相同raw | 选择后面/边/连接相同 | 优化后阻断 |", "|---|---|---|---|---|---|"])
    for row in summary["optimization_counterfactual"]["cases"]:
        same = "/".join("是" if v is True else "否" if v is False else "未知" for v in row["same_selected_arrays"].values())
        lines.append(f"| {row['id']} | {row['fit_only']['trace_verified']} | {row['optimized']['trace_verified']} | {row['same_raw_input_sha256']} | {same} | {TERMINALS[row['optimized']['terminal_stage']]} |")
    lines.extend(["", summary["optimization_counterfactual"]["note"], "", "[完整反事实路径](optimization_regressions.json)", "", "## 图表", ""])
    for chart in charts:
        lines.extend([f"![{chart['caption']}]({chart['png']})", "", chart["caption"], ""])
    lines.extend(["## 来源与解释边界", "", f"复用已有intersection诊断{summary['audit']['baseline_reuse']['eligible_count']}件，新增baseline对照{summary['audit']['baseline_controls']['checked_count']}件；来源资产检查{summary['audit']['source_preservation']['asset_count']}个，变化0个。", ""])
    lines.extend("- " + value for value in summary["limitations"])
    return "\n".join(lines) + "\n"


def page_payload(pairs, summary, charts, atlas_ids):
    def compact(row):
        return {key: row[key] for key in ("id", "method", "original_native_valid", "terminal_stage", "trace_verified", "flags", "owner_stages", "failed_face_context", "persistent_trim", "faces_without_boundary", "optimization_delta", "source_files")}
    return {"schema": VERSION, "population": summary["population"], "main": summary["main"],
            "method_labels": summary["method_labels"], "methods": summary["methods"],
            "highlights": summary["highlights"], "cohort_labels": COHORTS,
            "cohort_counts": {key: value["count"] for key, value in summary["cohorts"].items()},
            "thresholds": list(THRESHOLDS), "terminal_labels": TERMINALS,
            "observations": [{"key": key, "label": label, "english": english, "criterion": criterion, "uses_threshold": threshold} for key, label, english, criterion, threshold in OBSERVATIONS],
            "context_labels": CONTEXT_ERRORS, "charts": charts, "limitations": summary["limitations"],
            "pairs": [{"id": pair["id"], "cohort": pair["cohort"], "in_atlas": pair["id"] in atlas_ids,
                       **{role: compact(pair[role]) for role in ROLES}} for pair in pairs]}


PAGE_HTML = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>失败溯源 · 局部注意力与上一版refinement</title>
<style>
:root{font-family:Inter,"Noto Sans CJK SC","Microsoft Yahei",system-ui,sans-serif;color:#21303d;background:#f4f6f8;font-size:15px;--teal:#277e83;--orange:#bd6544;--line:#dce3e8}*{box-sizing:border-box}body{margin:0}a{color:#286b98}main{max-width:1320px;margin:auto;padding:32px 28px 80px}header{margin-bottom:24px}.eyebrow{font-size:12px;letter-spacing:.15em;color:#597681;font-weight:700}h1{font-size:32px;margin:10px 0 12px}h2{font-size:21px;margin:0 0 12px}p{line-height:1.8;color:#4b5b67}.lead{max-width:970px}.links{display:flex;gap:18px;flex-wrap:wrap}.controls{position:sticky;top:0;z-index:3;background:#f4f6f8f5;border-block:1px solid var(--line);padding:15px 0;display:flex;flex-wrap:wrap;align-items:center;gap:18px;backdrop-filter:blur(8px)}label{font-size:13px;font-weight:600}select,input{font:inherit;padding:9px;border:1px solid #c5d0d8;border-radius:6px;background:white;margin-left:7px}input{width:145px}.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:22px 0}.card,section{background:white;border:1px solid var(--line);border-radius:10px;padding:20px}.card b{font-size:30px;display:block;margin:7px 0}.muted,small{color:#697b87;font-size:12px}.card strong{font-size:13px}section{margin-top:18px}.legend{display:flex;gap:18px;align-items:center;flex-wrap:wrap;margin-bottom:14px;font-size:12px}.dot{width:11px;height:11px;display:inline-block;border-radius:2px;margin-right:5px}.scroll{overflow:auto}table{width:100%;border-collapse:collapse;font-size:13px}th{background:#f1f5f6;color:#506572;font-size:12px;text-align:left}th,td{padding:11px 10px;border-bottom:1px solid #e5eaee;vertical-align:top}td.num,th.num{text-align:right;white-space:nowrap;font-variant-numeric:tabular-nums}td.name{min-width:230px}button.count{border:0;background:transparent;font:inherit;text-decoration:underline;text-underline-offset:3px;color:#21616b;cursor:pointer;padding:0 3px}button.count:disabled{text-decoration:none;color:#7d8b93;cursor:default}.bar{display:flex;min-width:95px;height:8px;border-radius:4px;overflow:hidden;background:#edf1f3;margin-top:7px}.bar span{height:8px}.negative{background:#b9cfd0}.positive{background:var(--teal)}.unknown{background:#d8dce2}details{margin-top:5px;font-size:12px;line-height:1.7}summary{cursor:pointer;color:#5f7582}.ids{display:flex;flex-wrap:wrap;gap:6px;max-height:170px;overflow:auto;margin:12px 0}.id{font:12px ui-monospace,monospace;background:#eef4f5;border:1px solid #d2dfe1;border-radius:4px;padding:5px 8px;color:#1e6170;cursor:pointer}.id:hover{background:#dbecee}.pair-grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}.method{border:1px solid var(--line);padding:16px;border-radius:8px}.method h3{margin-top:0;font-size:17px}.badge{display:inline-block;padding:3px 8px;border-radius:4px;background:#e4f1ee;color:#2c7468;font-size:12px}.bad{background:#f7ece5;color:#985433}.gray{background:#edf0f4;color:#69727e}.notice{border-left:3px solid #83abae;padding-left:14px}.figures{display:grid;grid-template-columns:1fr 1fr;gap:18px}.figures figure{margin:0;border:1px solid var(--line);border-radius:6px;padding:10px}.figures img{width:100%;height:270px;object-fit:contain}.figures figcaption{font-size:12px;line-height:1.7;color:#5c6d77}code{font-size:11px;word-break:break-all}.empty{color:#77868f;padding:15px}ul{line-height:1.8;padding-left:20px}.context-table{max-height:370px;overflow:auto}#selection{scroll-margin-top:100px}#case-view{scroll-margin-top:100px}.title-line{display:flex;justify-content:space-between;gap:10px;align-items:center}.foot{font-size:12px}.count-note{color:#2d636f;font-weight:600}#active-caption{margin:0}.tri{font-size:12px}.tiny{font-size:11px}@media(max-width:850px){main{padding:20px 12px}.grid{grid-template-columns:1fr 1fr}.pair-grid,.figures{grid-template-columns:1fr}.controls{position:static}h1{font-size:26px}section{padding:15px}select{max-width:200px}}
</style></head><body><main>
<header><div class="eyebrow">HOLA-BREP / LOCAL INTERSECTION / PAIRED FAILURE CENSUS</div><h1>失败从哪里开始？</h1>
<p class="lead" id="population-lead"></p>
<div class="links"><a href="../NOTE.md">研究判断与下一步方向</a><a href="../REPORT.md">中文报告</a><a href="../summary.json">完整统计</a><a href="../cases.csv">逐件CSV</a><a href="../optimization_regressions.json">全部优化退步路径</a><a href="../../../joint_boundary/legacy_failures_20261011/page/">原HoLa历史50件图库</a></div></header>
<section id="highlights"><h2>先看最值得讨论的几件事</h2><div id="highlight-text"></div><p class="muted">以上来自固定全集；下方可切换具体对象组与距离阈值。</p></section>
<div class="controls"><label>对象组<select id="cohort"></select></label>
<label>距离阈值<select id="threshold"><option>0.002</option><option selected>0.006</option><option>0.01</option><option>0.08</option></select></label>
<label>查看ID<input id="case-search" placeholder="例如 00598784" inputmode="numeric" maxlength="8"></label><span class="muted">归一化XYZ长度，不是毫米</span></div>
<div class="grid" id="cards"></div>
<section><h2>同一批对象，旧新分别发生了什么</h2><p id="active-caption"></p><div class="legend"><span><i class="dot positive"></i>确认观察到</span><span><i class="dot negative"></i>确认未观察到</span><span><i class="dot unknown"></i>未知/无法核验</span><span>点击数量展开ID；多标签可重叠。</span></div>
<div class="scroll"><table id="mechanisms"><thead><tr><th>通俗描述 / 可展开精确判据</th><th class="num">上一版：有 / 无 / 未知</th><th class="num">本版：有 / 无 / 未知</th><th class="num">原有 → 新无</th><th class="num">原无 → 新有</th><th class="num">任一未知</th></tr></thead><tbody></tbody></table></div></section>
<section id="selection"><div class="title-line"><h2 id="selection-title">展开观察对应的ID</h2><span id="selection-count" class="count-note"></span></div><p class="muted" id="selection-note">在上表点击任意数量。未知不会被当成标签消失或改善。</p><div class="ids" id="selected-ids"></div></section>
<section id="case-view"><h2>把同一个零件放在一起看</h2><p id="case-hint">点选一个ID后，展示原终态、重放核验、该件观察、边面贴合与持续失败面的上下文错误。</p><div id="case-detail"></div></section>
<section><h2>边贴不到面：独立拟合时就有，还是周期替换后改变了？</h2><p>按相同边 / 所属面身份比较，采样定义保持一致。周期替换可能改变曲线参数域，因此这里比较采样残差，不能理解成同一空间点的移动。</p><div class="scroll"><table id="owner-summary"><thead><tr><th>当前组 / 每件最大采样距离</th><th>上一版</th><th>本版</th></tr></thead><tbody></tbody></table></div></section>
<section><h2>只看最后仍裁不出来的面：候选自身与上下文反复报什么？</h2><p>排除后来修复成功的面。每个decoder面每类错误只数一次；直接检查与所属上下文分别保留。不同错误可以同时出现。没有候选可检查或检查不完整的面，不写成“检查通过”。</p><div id="context-coverage" class="muted"></div><div class="scroll context-table"><table id="context-summary"><thead><tr><th>错误的通俗含义 / 原始名称</th><th class="num">上一版并集面数 / 件数 / 自身 / 上下文</th><th class="num">本版并集面数 / 件数 / 自身 / 上下文</th></tr></thead><tbody></tbody></table></div></section>
<section><h2>可导出的静态统计图</h2><p class="muted">以下是固定全集科研图，标题标明分母。上方筛选不会更改这些图片。</p><div class="figures" id="figures"></div></section>
<section><h2>这些统计能说明到哪里</h2><ul id="limitations"></ul><p class="foot">数据源保留原runner指纹；只在原终态一致、STEP DATA逐字节一致且没有观测异常时采用后处理阶段证据。原结果与诊断是否齐全分开记录。</p></section>
</main><script src="data.js"></script><script src="app.js"></script></body></html>
'''

PAGE_JS = r'''"use strict";
const D=window.PAIRED_CENSUS_DATA;
const $=id=>document.getElementById(id);
const esc=value=>String(value??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const names={positive:"确认有",negative:"确认无",unknown:"未知"};
let activePairs=[],activeCase=null,selection=null;
const number=v=>typeof v==="number"&&Number.isFinite(v)?Number(v.toPrecision(5)).toString():"未知";
const state=v=>v===true?"positive":v===false?"negative":"unknown";
const stat=(pairs,method,key,t)=>{const a={positive:[],negative:[],unknown:[]};for(const p of pairs)a[state(p[method].flags[t][key])].push(p.id);return a;};
const transition=(pairs,key,t)=>{const a={present_to_absent:[],absent_to_present:[],both_present:[],both_absent:[],unknown:[]};for(const p of pairs){const b=p.baseline.flags[t][key],n=p.candidate.flags[t][key];const k=typeof b!=="boolean"||typeof n!=="boolean"?"unknown":b&&n?"both_present":!b&&!n?"both_absent":b?"present_to_absent":"absent_to_present";a[k].push(p.id);}return a;};
const badge=(v,yes,no)=>`<span class="badge ${v===true?"":v===false?"bad":"gray"}">${esc(v===true?yes:v===false?no:"未知")}</span>`;
function countButton(count,key,method,group){return `<button class="count" data-key="${esc(key)}" data-method="${method}" data-group="${group}" ${count?"":"disabled"}>${count}</button>`;}
function triCell(values,key,method){const den=activePairs.length;return `<div>${["positive","negative","unknown"].map(s=>countButton(values[s].length,key,method,s)).join(" / ")}</div><div class="bar">${["positive","negative","unknown"].map(s=>`<span class="${s}" style="width:${den?100*values[s].length/den:0}%"></span>`).join("")}</div>`;}
function showIds(ids,title,note){selection={ids,title,note};$("selection-title").textContent=title;$("selection-count").textContent=`${ids.length}件`;$("selection-note").textContent=note;$("selected-ids").innerHTML=ids.map(id=>`<button class="id" data-id="${id}">${id}</button>`).join("")||'<span class="empty">本组没有对应对象。</span>';}
function displayPair(id){const p=activePairs.find(p=>p.id===id);activeCase=p?.id??null;if(!p){$("case-detail").innerHTML='<p class="empty">这个ID不在当前对象组中，可切换分组后查看。</p>';return;}const t=$("threshold").value;$("case-hint").innerHTML=`<b>${esc(id)}</b> · ${esc(D.cohort_labels[p.cohort])} ${p.in_atlas?`· <a href="../../../joint_boundary/legacy_failures_20261011/page/#${id}">查看原HoLa历史3D案例</a>`:"· 未收录在原50件3D图库"}`;$("case-detail").innerHTML=`<div class="pair-grid">${["baseline","candidate"].map(method=>{const r=p[method],faces=r.failed_face_context.faces;const context=faces===null?"失败面集合未知":faces.length===0?"没有最终失败的裁剪面":faces.map(f=>`面${f.face_id}: ${f.codes.length?f.codes.map(c=>esc(D.context_labels[c]||c)).join("；"):f.inspection_complete?"未见这些上下文错误":"没有完整上下文检查"}`).join("<br>");const positive=D.observations.filter(o=>r.flags[t][o.key]===true);const unknown=D.observations.filter(o=>r.flags[t][o.key]===null);return `<article class="method"><h3>${esc(D.method_labels[method])}</h3>${badge(r.original_native_valid,"原结果：合法实体","原结果：失败")} ${badge(r.trace_verified,"重放完整一致","阶段证据未核验")}<p><b>${esc(D.terminal_labels[r.terminal_stage])}</b></p><p class="tiny">独立拟合→周期替换后最大边面残差：${number(r.owner_stages.independent_curve.observed_max)} → ${number(r.owner_stages.after_periodic_replacement.observed_max)}<br>优化后面MAE变化：${number(r.optimization_delta.face_xyz_mae)}；边距离变化：${number(r.optimization_delta.matched_edge_mean_length)}</p><b class="tri">当前阈值确认观察到</b><ul>${positive.map(o=>`<li>${esc(o.label)}</li>`).join("")||"<li>未确认上述观察</li>"}</ul><details><summary>未知的观察 ${unknown.length}项</summary><ul>${unknown.map(o=>`<li>${esc(o.label)}</li>`).join("")}</ul></details><details><summary>最终失败面的候选检查</summary><p>${context}</p></details><details><summary>轨迹文件及SHA256</summary>${Object.entries(r.source_files).map(([name,source])=>`<p>${esc(name)}<br><code>${esc(source.path)}</code><br><code>${esc(source.sha256)}</code></p>`).join("")}</details></article>`;}).join("")}</div>`;}
function median(values){const a=values.filter(v=>typeof v==="number"&&Number.isFinite(v)).sort((a,b)=>a-b);return a.length? a.length%2?a[(a.length-1)/2]:(a[a.length/2-1]+a[a.length/2])/2:null;}
function ownerTable(){const metrics=[["independent_curve","独立拟合后的最大残差中位数"],["after_periodic_replacement","周期替换后的最大残差中位数"]];$("owner-summary").querySelector("tbody").innerHTML=metrics.map(([key,label])=>`<tr><td>${label}</td>${["baseline","candidate"].map(m=>{const v=activePairs.map(p=>p[m].owner_stages[key]).filter(v=>v.status==="ok").map(v=>v.observed_max);return `<td>${number(median(v))} <small>完整测量${v.length}/${activePairs.length}件</small></td>`;}).join("")}</tr>`).join("")+`<tr><td>周期替换后对象最大值：下降 / 不变 / 增加 / 未知</td>${["baseline","candidate"].map(m=>{const v=activePairs.map(p=>p[m].owner_stages.object_max_delta);return `<td>${v.filter(x=>typeof x==="number"&&x< -1e-12).length} / ${v.filter(x=>typeof x==="number"&&Math.abs(x)<=1e-12).length} / ${v.filter(x=>typeof x==="number"&&x>1e-12).length} / ${v.filter(x=>x===null).length}</td>`;}).join("")}</tr>`;}
function contextTable(){const all={},coverage={};for(const m of ["baseline","candidate"]){const r=activePairs.map(p=>p[m]);const faces=r.flatMap(r=>r.failed_face_context.faces||[]);const codes=new Set(faces.flatMap(f=>f.codes));coverage[m]=`${faces.length}个最终失败面，${faces.filter(f=>f.inspection_complete).length}个有完整候选检查；${r.filter(r=>r.failed_face_context.faces===null).length}件的失败面集合未知`;for(const code of codes){all[code]??={baseline:{faces:0,ids:[],direct:0,context:0},candidate:{faces:0,ids:[],direct:0,context:0}};all[code][m]={faces:faces.filter(f=>f.codes.includes(code)).length,direct:faces.filter(f=>f.direct_codes.includes(code)).length,context:faces.filter(f=>f.context_codes.includes(code)).length,ids:activePairs.filter(p=>(p[m].failed_face_context.faces||[]).some(f=>f.codes.includes(code))).map(p=>p.id)};}}$("context-coverage").textContent=`上一版：${coverage.baseline}；本版：${coverage.candidate}`;$("context-summary").querySelector("tbody").innerHTML=Object.entries(all).sort((a,b)=>b[1].candidate.faces-a[1].candidate.faces).map(([code,v])=>`<tr><td>${esc(D.context_labels[code]||"其他上下文错误")}<br><code>${esc(code)}</code></td>${["baseline","candidate"].map(m=>`<td class="num">${v[m].faces} / <button class="count" data-context="${esc(code)}" data-method="${m}">${v[m].ids.length}</button> / ${v[m].direct} / ${v[m].context}</td>`).join("")}</tr>`).join("")||'<tr><td colspan="3">当前组未确认这些上下文错误。检查缺失不表示检查通过。</td></tr>';}
function render(){const cohort=$("cohort").value,t=$("threshold").value;activePairs=D.pairs.filter(p=>cohort==="main"?!p.candidate.original_native_valid:p.cohort===cohort);const den=activePairs.length;const valid=m=>activePairs.filter(p=>p[m].original_native_valid).length;const verified=m=>activePairs.filter(p=>p[m].trace_verified).length;$("cards").innerHTML=`<div class="card"><strong>当前固定分母</strong><b id="active-denominator">${den}</b><span class="muted">${cohort==="main"?"本版失败全集":esc(D.cohort_labels[cohort])}</span></div><div class="card"><strong>上一版 → 本版合法实体</strong><b>${valid("baseline")} → ${valid("candidate")}</b><span class="muted">原评估结果；缺测不改结果</span></div><div class="card"><strong>上一版 / 本版完整重放</strong><b>${verified("baseline")} / ${verified("candidate")}</b><span class="muted">两者分母均为${den}</span></div><div class="card"><strong>全测试有效性</strong><b>${D.population.baseline_native_valid} → ${D.population.native_valid}</b><span class="muted">接受集${D.population.accepted_test} / 原始集${D.population.original_test}；${D.population.contract_unsupported}不支持</span></div>`;$("active-caption").textContent=cohort==="main"?`当前${den}件都在本版失败：${D.cohort_counts.persistent}件上一版也失败，${D.cohort_counts.regressed}件上一版成功。两列始终比较相同ID。`:"当前两列始终是相同ID集合。标签消失/出现只是观察变化，不能据此确定唯一根因。";$("mechanisms").querySelector("tbody").innerHTML=D.observations.map(o=>{const a=stat(activePairs,"baseline",o.key,t),b=stat(activePairs,"candidate",o.key,t),x=transition(activePairs,o.key,t);return `<tr data-observation="${o.key}"><td class="name"><b>${esc(o.label)}</b>${o.uses_threshold?' <small>随阈值</small>':""}<details><summary>具体怎么判断</summary>${esc(o.criterion)}</details></td><td class="num">${triCell(a,o.key,"baseline")}</td><td class="num">${triCell(b,o.key,"candidate")}</td><td class="num">${countButton(x.present_to_absent.length,o.key,"paired","present_to_absent")}</td><td class="num">${countButton(x.absent_to_present.length,o.key,"paired","absent_to_present")}</td><td class="num">${countButton(x.unknown.length,o.key,"paired","unknown")}</td></tr>`;}).join("");ownerTable();contextTable();showIds(activePairs.map(p=>p.id),"当前对象组的全部ID",`分母${den}；点击ID查看同一件的上一版与本版。`);if(activeCase)displayPair(activeCase);window.PAIRED_CENSUS_VIEW={cohort,threshold:t,denominator:den,ids:activePairs.map(p=>p.id)};}
$("mechanisms").addEventListener("click",event=>{const b=event.target.closest("button[data-key]");if(!b)return;const {key,method,group}=b.dataset,o=D.observations.find(o=>o.key===key),t=$("threshold").value;const ids=method==="paired"?transition(activePairs,key,t)[group]:stat(activePairs,method,key,t)[group];const label=method==="paired"?{present_to_absent:"原有→新无",absent_to_present:"原无→新有",unknown:"任一侧未知"}[group]:(method==="baseline"?"上一版":"本版")+" · "+names[group];showIds(ids,`${o.label} · ${label}`,o.criterion);$("selection").scrollIntoView({behavior:"smooth",block:"start"});});
$("selected-ids").addEventListener("click",event=>{const b=event.target.closest("button[data-id]");if(b){displayPair(b.dataset.id);$("case-view").scrollIntoView({behavior:"smooth",block:"start"});}});
$("context-summary").addEventListener("click",event=>{const b=event.target.closest("button[data-context]");if(!b)return;const {context,method}=b.dataset;const ids=activePairs.filter(p=>(p[method].failed_face_context.faces||[]).some(f=>f.codes.includes(context))).map(p=>p.id);showIds(ids,`${method==="baseline"?"上一版":"本版"} · ${D.context_labels[context]||context}`,"同一最终失败面上的候选/上下文检查错误；跨容差尝试每面去重。");$("selection").scrollIntoView({behavior:"smooth"});});
for(const id of ["cohort","threshold"])$(id).addEventListener("change",render);
$("case-search").addEventListener("input",event=>{if(/^\d{8}$/.test(event.target.value))displayPair(event.target.value);});
$("figures").innerHTML=D.charts.map(c=>`<figure><a href="../${c.png}" target="_blank"><img src="../${c.png}" alt="${esc(c.caption)}" loading="lazy"></a><figcaption>${esc(c.caption)} <a href="../${c.svg}">SVG</a></figcaption></figure>`).join("");
$("population-lead").textContent=`先看连接，再看选边、优化、拟合与裁剪。本版失败${D.main.denominator}件＝持续失败${D.cohort_counts.persistent}件＋退步${D.cohort_counts.regressed}件，另有${D.cohort_counts.rescued}件救回对照。每一行可以展开具体ID。`;
$("cohort").innerHTML=[['main',`本版全部失败 · ${D.main.denominator}`],...Object.entries(D.cohort_labels).map(([k,v])=>[k,`${v} · ${D.cohort_counts[k]}`])].map(([k,v])=>`<option value="${k}">${esc(v)}</option>`).join('');
$("limitations").innerHTML=D.limitations.map(line=>`<li>${esc(line)}</li>`).join("");render();
$("highlight-text").innerHTML=D.highlights.map(line=>`<p class="notice">${esc(line)}</p>`).join("");
'''


def write_outputs(root, pairs, summary, charts):
    def write_json(name, value):
        (root / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    write_json("summary.json", summary)
    compact = compact_summary(summary)
    compact["full_summary_sha256"] = legacy.file_sha256(root / "summary.json")
    write_json("compact_summary.json", compact)
    write_json("cases.json", {"schema": VERSION, "pairs": pairs})
    write_json("optimization_regressions.json", summary["optimization_counterfactual"])
    with (root / "cases.csv").open("w", newline="") as handle:
        fields = ["id", "cohort"] + [f"{role}_{field}" for role in ROLES for field in ("method", "native_valid", "trace_verified", "terminal")]
        fields += [f"{role}__{t}__{key}" for role in ROLES for t in THRESHOLDS for key, *_ in OBSERVATIONS]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for pair in pairs:
            row = {"id": pair["id"], "cohort": pair["cohort"]}
            for role in ROLES:
                record = pair[role]
                row.update({f"{role}_method": record["method"], f"{role}_native_valid": record["original_native_valid"],
                            f"{role}_trace_verified": record["trace_verified"], f"{role}_terminal": record["terminal_stage"]})
                row.update({f"{role}__{t}__{key}": record["flags"][t][key] for t in THRESHOLDS for key, *_ in OBSERVATIONS})
            writer.writerow(row)
    (root / "REPORT.md").write_text(report_markdown(summary, charts))
    atlas_root = Path(__file__).resolve().parents[1] / "checkpoints/joint_boundary/legacy_failures_20261011"
    atlas = atlas_root / "selected_ids.txt"
    atlas_available = atlas.is_file() and all((atlas_root / name).is_file() for name in (
        "page/index.html", "page/data.js", "page/vendor/three-0.160.1.min.js"))
    payload = page_payload(pairs, summary, charts, set(atlas.read_text().split()) if atlas_available else set())
    page = root / "page"
    page.mkdir(exist_ok=True)
    label = summary["method_labels"]["candidate"]
    html = PAGE_HTML.replace("局部注意力与上一版refinement", label + "与上一版refinement")
    html = html.replace("LOCAL INTERSECTION", summary["methods"]["candidate"].upper())
    atlas_url = os.path.relpath(atlas_root / "page", page) + "/"
    old_url = "../../../joint_boundary/legacy_failures_20261011/page/"
    html = html.replace(old_url, atlas_url)
    if not atlas_available:
        html = html.replace(f'<a href="{atlas_url}">原HoLa历史50件图库</a>',
                            '<span class="muted">原HoLa历史图库为独立资源</span>')
    (page / "index.html").write_text(html)
    (page / "app.js").write_text(PAGE_JS.replace(old_url, atlas_url))
    (page / "data.js").write_text("window.PAIRED_CENSUS_DATA=" + json.dumps(payload, ensure_ascii=False, allow_nan=False).replace("</", "<\\/") + ";\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--census-root", type=Path, required=True)
    parser.add_argument("--baseline-census-root", type=Path, required=True)
    parser.add_argument("--font-path", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    root, baseline = args.census_root.resolve(), args.baseline_census_root.resolve()
    if (root / "summary.json").exists() and not args.overwrite:
        raise FileExistsError("Use --overwrite only to regenerate report artifacts after source checks")
    pairs, summary = build_report(root, baseline)
    charts = make_charts(summary, root / "visualizations", args.font_path)
    write_outputs(root, pairs, summary, charts)
    print(json.dumps({"paired_cases": len(pairs), "main_failures": summary["main"]["denominator"],
                      "cohort_counts": {k: v["count"] for k, v in summary["cohorts"].items()},
                      "charts": len(charts)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
