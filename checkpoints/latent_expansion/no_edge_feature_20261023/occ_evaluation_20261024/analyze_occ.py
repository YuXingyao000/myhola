"""Frozen validation OCC analysis for source32 and the 64-D no-feature endpoint.

Measurement functions are imported unchanged from the 2026-10-20 analysis.
See SPEC.md. Run ``measure`` only after raw export; ``report`` after post completes.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import csv
import importlib.util
import json
import os
from pathlib import Path
import sys

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
HISTORY = REPO / "checkpoints/shared_curve_uv/evaluation_20261020"
SOURCE = HISTORY / "best"
SPLIT = REPO / "checkpoints/joint_boundary/validation/val_contract_audit/accepted.txt"
RAW_SUMMARY = HERE.parent / "evaluation_20261024/summary.json"
SIGNATURE = "704227ad75de74bd531ae8641c4060818dcfac02089bf6f43053eb95b9c4c77a"
ROLES = ("source32", "no_edge_feature64")
METHODS = {"source32": "shared_curve_uv", "no_edge_feature64": "no_edge_feature64"}
LABELS = {"source32": "Source 32D\n(42,471-step checkpoint)",
          "no_edge_feature64": "No edge-feature 64D\n(30,000 new steps)"}
SHORT_LABELS = {"source32": "Source 32D", "no_edge_feature64": "No edge-feature 64D"}
COLORS = {"source32": "#547c97", "no_edge_feature64": "#d47c3c"}
module_spec = importlib.util.spec_from_file_location(
    "shared_curve_validation_frozen", HISTORY / "analyze_validation_failures.py")
legacy = importlib.util.module_from_spec(module_spec)
sys.modules[module_spec.name] = legacy
module_spec.loader.exec_module(legacy)


def check_signature() -> str:
    actual = legacy.measurement_signature()
    if actual != SIGNATURE:
        raise ValueError(f"Frozen measurement implementation changed: {actual}")
    return actual


def measure(args) -> None:
    check_signature()
    ids = legacy.read_ids(args.split_list)
    raw_root = args.evaluation_root / "raw_predictions/posterior_mean/no_edge_feature64"
    present = {path.parent.name for path in raw_root.glob("*/data.npz")}
    if present != set(ids):
        raise ValueError(f"Expected all 3495 candidate exports; missing={len(set(ids) - present)}, "
                         f"extra={len(present - set(ids))}")
    if not 1 <= args.workers <= 8:
        raise ValueError("Frozen protocol allows 1 to 8 measurement workers")
    started = datetime.now(timezone.utc).isoformat()
    output = args.evaluation_root / "failure_analysis/raw/no_edge_feature64"
    legacy.measure(argparse.Namespace(
        split_list=args.split_list, raw_root=raw_root, output_root=output,
        data_root=args.data_root, workers=args.workers, only_ids=None))
    legacy.write_json(args.evaluation_root / "failure_analysis/measurement_execution.json", {
        "started_utc": started, "completed_utc": datetime.now(timezone.utc).isoformat(),
        "argv": sys.argv, "workers": args.workers, "threads_per_worker": 1,
        "script_sha256": legacy.sha(__file__), "measurement_fingerprint": check_signature(),
        "measurement_source_path": str(HISTORY / "analyze_validation_failures.py"),
        "measurement_source_sha256": legacy.sha(HISTORY / "analyze_validation_failures.py"),
        "source32_cache": str(HISTORY / "failure_analysis/raw/best"),
        "candidate_cache": str(output), "intended_count": len(ids),
    })


def paired_groups(post: dict, ids: list[str]) -> dict:
    groups = {name: [] for name in ("both_valid", "rescued", "regressed", "both_failed")}
    for sid in ids:
        old, new = (post[role][sid]["valid"] for role in ROLES)
        key = "both_valid" if old and new else "rescued" if new else "regressed" if old else "both_failed"
        groups[key].append(sid)
    if sum(map(len, groups.values())) != len(ids):
        raise ValueError("Paired validity groups do not partition the fixed population")
    return groups


def paired_geometry(post: dict, ids: list[str]) -> dict:
    result = {}
    common_valid = [sid for sid in ids if all(post[role][sid]["valid"] for role in ROLES)]
    for label, intended in (("common_computable", ids), ("common_native_valid", common_valid)):
        computable = [sid for sid in intended if all(
            post[role][sid]["row"].get("geometry_status") == "ok"
            and all(legacy.finite_number(post[role][sid]["row"].get(key)) is not None
                    for key in ("face_cd", "edge_cd")) for role in ROLES)]
        result[label] = {"intended_count": len(intended), "computable_count": len(computable),
                         "missing_count": len(intended) - len(computable), "ids": computable,
                         "missing_ids": sorted(set(intended) - set(computable)), "metrics": {}}
        for metric in ("face_cd", "edge_cd"):
            values = {role: np.asarray([float(post[role][sid]["row"][metric]) for sid in computable])
                      for role in ROLES}
            delta = values[ROLES[1]] - values[ROLES[0]]
            base_mean = float(values[ROLES[0]].mean()) if computable else None
            new_mean = float(values[ROLES[1]].mean()) if computable else None
            result[label]["metrics"][metric] = {
                **{role: legacy.distribution(value) for role, value in values.items()},
                "candidate_minus_baseline": legacy.distribution(delta),
                "mean_relative_change": new_mean / base_mean - 1 if base_mean else None,
                "candidate_lower_count": int((delta < 0).sum()),
                "equal_count": int((delta == 0).sum()), "candidate_higher_count": int((delta > 0).sum()),
            }
    return result


def load_measurements(args, ids: list[str]) -> tuple[dict, dict]:
    caches = {"source32": HISTORY / "failure_analysis/raw/best",
              "no_edge_feature64": args.evaluation_root / "failure_analysis/raw/no_edge_feature64"}
    raw_roots = {"source32": SOURCE / "raw_predictions/posterior_mean/shared_curve_uv",
                 "no_edge_feature64": args.evaluation_root / "raw_predictions/posterior_mean/no_edge_feature64"}
    gt_hashes = {sid: legacy.sha(args.data_root / sid / "data.npz") for sid in ids}
    if any(value is None for value in gt_hashes.values()):
        raise ValueError("Missing canonical GT during measurement provenance verification")
    result = {}
    for role in ROLES:
        result[role] = {}
        for sid in ids:
            path = caches[role] / f"{sid}.json"
            if not path.is_file():
                raise ValueError(f"Measurement is not complete: {role}/{sid}")
            record = legacy.read_json(path)
            if record.get("id") != sid or record.get("protocol_fingerprint") != SIGNATURE:
                raise ValueError(f"Measurement identity or fingerprint differs: {role}/{sid}")
            if record.get("sources", {}).get("gt", {}).get("sha256") != gt_hashes[sid]:
                raise ValueError(f"GT bytes differ: {role}/{sid}")
            expected_raw = raw_roots[role] / sid / "data.npz"
            if record.get("sources", {}).get("raw", {}).get("sha256") != legacy.sha(expected_raw):
                raise ValueError(f"Raw bytes differ from measurement: {role}/{sid}")
            if record.get("status") not in {"ok", "error"}:
                raise ValueError(f"Unknown raw measurement status: {role}/{sid}")
            for flags in record.get("flags", {}).values():
                pred, truth = flags.get("owner_mesh_gap"), flags.get("gt_owner_mesh_gap")
                flags["owner_gap_gt_clear"] = (True if pred is True and truth is False else
                                              False if pred is False or truth is True else None)
            result[role][sid] = record
    return result, {role: str(path) for role, path in caches.items()}


def raw_summary(args) -> dict:
    source = legacy.read_json(args.raw_summary)
    if (not source.get("complete") or source.get("intended_count") != 3495
            or source.get("split_sha256") != legacy.SPLIT_SHA
            or source.get("checkpoint_global_step") != 30000
            or source.get("checkpoint_sha256") != "4262f4998fe047e19c3ca21a835d74c183b7430f7a656c709ec8ed045262ada4"):
        raise ValueError("Raw summary does not identify the frozen complete 30k evaluation")
    export_check = legacy.read_json(args.export_validation)
    required = {"passed": True, "count": 3495, "expected_count": 3495,
                "unique": True, "ids_match": True,
                "all_metric_comparisons_match": True, "all_count_comparisons_match": True,
                "comparison_atol": 1e-9, "comparison_rtol": 1e-3,
                "checkpoint_sha256": source["checkpoint_sha256"], "split_sha256": legacy.SPLIT_SHA}
    mismatches = [key for key, expected in required.items() if export_check.get(key) != expected]
    if mismatches:
        raise ValueError(f"Export identity, coverage or raw metric check failed: {mismatches}")
    result = {}
    for role, source_name in (("source32", "source32"), ("no_edge_feature64", "no_edge_feature_last64")):
        model = source["models"][source_name]
        if model["coverage"]["ok"] != 3495:
            raise ValueError("Raw summary does not cover the entire frozen population")
        result[role] = {"metrics": {name: model["metrics"][name] for name in (
            "raw_face_xyz_l1", "raw_face_cd_sq", "raw_edge_cd_sq",
            "raw_face_pred_to_gt_sq", "raw_face_gt_to_pred_sq",
            "raw_edge_pred_to_gt_sq", "raw_edge_gt_to_pred_sq")}}
    return result


def aggregate(args) -> None:
    check_signature()
    completion_path = args.evaluation_root / "metadata/post_launch/completion.json"
    completion = legacy.read_json(completion_path)
    unchanged = completion.get("source_unchanged")
    if (completion.get("returncode") != 0 or not isinstance(unchanged, dict)
            or not unchanged or any(value is not True for value in unchanged.values())):
        raise ValueError("Post launch must complete successfully with every frozen source unchanged")
    ids = legacy.read_ids(args.split_list)
    raw, caches = load_measurements(args, ids)
    post = {"source32": legacy.load_post(SOURCE, "shared_curve_uv", ids),
            "no_edge_feature64": legacy.load_post(args.evaluation_root, "no_edge_feature64", ids)}
    if sum(value["valid"] for value in post["source32"].values()) != 3180:
        raise ValueError("Source32 historical 3180 native solids do not reproduce")
    groups = paired_groups(post, ids)
    summary = {
        "version": "no_edge_feature_occ_validation_v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "complete": True, "protocol": legacy.PROTOCOL,
        "population": {"accepted_validation": len(ids), "original_validation": 3505, "unsupported": 10},
        "roles": METHODS, "checkpoint_steps": {"source32": 42471, "no_edge_feature64": 30000},
        "checkpoint_sha256": {"source32": "ce749416c1635dda1563161b971bd4260d064cfdb0c31f1fa9d06ad0e70961ff",
                              "no_edge_feature64": "4262f4998fe047e19c3ca21a835d74c183b7430f7a656c709ec8ed045262ada4"},
        "provenance": {"script_sha256": legacy.sha(__file__), "measurement_fingerprint": SIGNATURE,
                       "measurement_script_sha256": legacy.sha(HISTORY / "analyze_validation_failures.py"),
                       "split_sha256": legacy.sha(args.split_list), "measurement_cache_roots": caches,
                       "post_csv_sha256": {role: legacy.sha(root / "post/optimized/samples.csv")
                                           for role, root in (("source32", SOURCE), ("no_edge_feature64", args.evaluation_root))},
                       "post_completion_sha256": legacy.sha(completion_path),
                       "raw_summary_path": str(args.raw_summary), "raw_summary_sha256": legacy.sha(args.raw_summary),
                       "export_validation_path": str(args.export_validation), "export_validation_sha256": legacy.sha(args.export_validation)},
        "raw_metrics": raw_summary(args), "methods": {},
        "paired": {"counts": {key: len(value) for key, value in groups.items()}, "ids": groups,
                   "post_geometry": paired_geometry(post, ids), "observations": {}},
        "limitations": [
            "Validation only; single seed; source32 and the new 64D 30k checkpoint differ in training budget and encoder state.",
            "Failure stop stages locate pipeline termination, not unique physical causes; raw observations may overlap.",
            "Owner distances use a sampled triangle mesh; GT has the same approximation floor.",
            "Twin agreement is guaranteed by the shared-curve representation, not an independent learned improvement.",
            "Raw and STEP geometry use different representations and sampling; their CDs cannot be subtracted as a repair gain.",
            "Common-valid geometry excludes failures and missing geometry; it does not replace full-population validity.",
            "Native stored tolerances are not measured geometry errors; historical timings are not a controlled speed comparison.",
        ],
    }
    rows = []
    cohort = {sid: group for group, members in groups.items() for sid in members}
    for role in ROLES:
        records = post[role]
        failed = [sid for sid in ids if not records[sid]["valid"]]
        valid_ids = [sid for sid in ids if records[sid]["valid"]]
        measurable = [sid for sid in valid_ids if records[sid]["row"].get("geometry_status") == "ok"
                      and all(legacy.finite_number(records[sid]["row"].get(k)) is not None for k in ("face_cd", "edge_cd"))]
        stages = Counter(records[sid]["stage"] for sid in failed)
        if sum(stages.values()) != len(failed):
            raise ValueError("Failure-stage partition does not reconcile")
        summary["methods"][role] = {
            "valid": len(valid_ids), "failed": len(failed), "valid_fraction": len(valid_ids) / len(ids),
            "valid_original_fraction": len(valid_ids) / 3505,
            "step_exists": sum(records[sid]["step_exists"] for sid in ids),
            "step_readable": sum(records[sid]["step_readable"] for sid in ids),
            "success_marker": sum(legacy.boolean(records[sid]["row"].get("success_marker")) for sid in ids),
            "compat_valid_solid": sum(legacy.boolean(records[sid]["row"].get("compat_valid_solid")) for sid in ids),
            "stages": dict(stages), "runner_statuses": dict(Counter(records[sid]["status"] for sid in ids)),
            "furthest_recovery_stages_failed": dict(Counter(records[sid]["furthest_recovery_stage"] for sid in failed)),
            "native_validity_statuses": dict(Counter(records[sid]["validity_status"] for sid in ids)),
            "timeouts_after_native_valid": sum(records[sid]["status"] == "timeout" for sid in valid_ids),
            "raw_diagnostic_statuses": dict(Counter(raw[role][sid]["status"] for sid in ids)),
            "accepted_path_counts": dict(Counter(records[sid]["chosen_path"] or "unrecorded" for sid in valid_ids)),
            "geometry_statuses": dict(Counter(records[sid]["row"].get("geometry_status", "not_run") for sid in ids)),
            "native_valid_geometry_count": len(measurable),
            "native_valid_geometry_missing": len(valid_ids) - len(measurable),
            "native_valid_geometry_missing_statuses": dict(Counter(records[sid]["status"] for sid in set(valid_ids) - set(measurable))),
            "native_valid_geometry": {metric: legacy.distribution(records[sid]["row"].get(metric) for sid in measurable)
                                      for metric in ("face_cd", "edge_cd")},
            "native_valid_max_tolerance": legacy.distribution(max((value for key in (
                "native_edge_tolerance_max", "native_face_tolerance_max", "native_vertex_tolerance_max")
                if (value := legacy.finite_number(records[sid]["row"].get(key))) is not None), default=None) for sid in valid_ids),
            "latency": {key: legacy.distribution(records[sid]["row"].get(key) for sid in ids)
                        for key in ("wall_seconds", "post_elapsed_seconds", "geometry_elapsed_seconds")},
            "owner_p95_max": legacy.distribution(raw[role][sid].get("owner", {}).get("p95_max") for sid in ids),
            "observations": {scope: {format(t, "g"): {flag: legacy.tally([raw[role][sid] for sid in group], format(t, "g"), flag)
                              for flag in legacy.REPORT_FLAG_LABELS} for t in legacy.THRESHOLDS}
                             for scope, group in (("all", ids), ("failed", failed), ("valid", valid_ids))},
        }
        for sid in ids:
            item, observation = records[sid], raw[role][sid]
            row = {"id": sid, "role": role, "method": METHODS[role], "valid": item["valid"],
                   "cohort_vs_source32": cohort[sid], "terminal_stage": item["stage"],
                   "native_validity_status": item["validity_status"], "runner_status": item["status"],
                   "raw_diagnostic_status": observation["status"], "last_path": item["last_path"],
                   "chosen_path": item["chosen_path"], "furthest_recovery_stage": item["furthest_recovery_stage"],
                   "geometry_status": item["row"].get("geometry_status"),
                   "face_cd": legacy.finite_number(item["row"].get("face_cd")),
                   "edge_cd": legacy.finite_number(item["row"].get("edge_cd")),
                   "raw_sha256": observation.get("sources", {}).get("raw", {}).get("sha256"),
                   "gt_sha256": observation.get("sources", {}).get("gt", {}).get("sha256"),
                   "owner_p95_max": observation.get("owner", {}).get("p95_max"),
                   "gt_owner_p95_max": observation.get("gt_owner", {}).get("p95_max"),
                   "twin_p95_max": max((x["p95"] for x in observation.get("twins", [])), default=None),
                   "endpoint_max": observation.get("endpoints", {}).get("max")}
            for t in legacy.THRESHOLDS:
                row.update({f"{flag}@{t:g}": observation.get("flags", {}).get(format(t, "g"), {}).get(flag)
                            for flag in legacy.REPORT_FLAG_LABELS})
            rows.append(row)
    for group, members in groups.items():
        summary["paired"]["observations"][group] = {
            role: {format(t, "g"): {flag: legacy.tally([raw[role][sid] for sid in members], format(t, "g"), flag)
                                   for flag in legacy.REPORT_FLAG_LABELS} for t in legacy.THRESHOLDS} for role in ROLES}
    delta = summary["methods"][ROLES[1]]["valid"] - summary["methods"][ROLES[0]]["valid"]
    if delta != len(groups["rescued"]) - len(groups["regressed"]):
        raise ValueError("Net validity change differs from paired rescues minus regressions")
    summary["paired"].update(net_valid_change=delta, percentage_point_change=100 * delta / len(ids))
    output = args.evaluation_root / "failure_analysis"
    output.mkdir(parents=True, exist_ok=True)
    legacy.write_json(output / "summary.json", summary)
    with (output / "samples.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    charts(summary, output / "figures")
    legacy.write_json(output / "aggregate_validation.json", {
        "passed": True, "intended_count": len(ids), "source32_native_valid": 3180,
        "paired_count_sum": sum(summary["paired"]["counts"].values()), "measurement_fingerprint": SIGNATURE,
        "checks": ["fixed split", "raw/GT cache SHA", "measurement fingerprint", "post raw/diagnostic SHA",
                   "native validity flags", "source32 3180", "exclusive failure stages", "paired partition",
                   "net rescue-minus-regression", "raw summary identity", "export passed", "common-valid geometry cohorts"],
        "figure_count": len(list((output / "figures").glob("*.png"))),
        "summary_sha256": legacy.sha(output / "summary.json"), "samples_csv_sha256": legacy.sha(output / "samples.csv"),
    })
    print(json.dumps({"methods": {role: {key: summary["methods"][role][key] for key in (
        "valid", "failed", "stages", "native_valid_geometry_count", "native_valid_geometry_missing")}
        for role in ROLES}, "paired": summary["paired"]["counts"]}, indent=2))


def charts(summary: dict, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import fontManager
    output.mkdir(parents=True, exist_ok=True)
    font = "Archivo" if any(item.name == "Archivo" for item in fontManager.ttflist) else "DejaVu Sans"
    plt.rcParams.update({"font.family": font, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
    labels = [LABELS[role] for role in ROLES]
    colors = [COLORS[role] for role in ROLES]

    def save(fig, name):
        fig.tight_layout()
        for extension in ("png", "svg"):
            fig.savefig(output / f"{name}.{extension}", bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    n = summary["population"]["accepted_validation"]
    values = [summary["methods"][role]["valid"] for role in ROLES]
    bars = axes[0].bar(labels, values, color=colors)
    axes[0].bar_label(bars, labels=[f"{value}/{n}\n{value/n:.2%}" for value in values], padding=4)
    axes[0].set_ylim(0, n * 1.16)
    axes[0].set_title("Saved STEP: native valid solid")
    keys = ("rescued", "regressed", "both_failed")
    transitions = [summary["paired"]["counts"][key] for key in keys]
    bars = axes[1].bar(("Rescued", "Regressed", "Both failed"), transitions,
                       color=("#6c9c75", "#d47c3c", "#939393"))
    axes[1].bar_label(bars, padding=3)
    axes[1].set_ylim(0, max(1, *transitions) * 1.22)
    axes[1].set_title(f"Paired: {summary['paired']['counts']['both_valid']} both valid; "
                      f"net {summary['paired']['net_valid_change']:+d}")
    fig.suptitle("Validation only | recover_v1 | 3495 accepted + 10 historically unsupported", fontsize=11)
    save(fig, "01_validity_and_transitions")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax, metric, title in zip(axes, ("raw_face_xyz_l1", "raw_face_cd_sq", "raw_edge_cd_sq"),
                                 ("Raw face XYZ L1", "Raw face CD squared", "Raw edge CD squared")):
        values = [summary["raw_metrics"][role]["metrics"][metric]["mean"] for role in ROLES]
        bars = ax.bar(labels, values, color=colors)
        ax.bar_label(bars, labels=[f"{value:.5g}" for value in values], padding=4)
        ax.set_ylim(0, max(values) * 1.22)
        ax.set_title(title)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
    fig.suptitle("Raw own-posterior-mean reconstruction | all 3495 CADs | object-weighted mean", fontsize=11)
    save(fig, "02_raw_geometry_comparison")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    keys = [key for key in legacy.STAGE_LABELS if any(summary["methods"][role]["stages"].get(key, 0) for role in ROLES)]
    for offset, role in zip((-.2, .2), ROLES):
        bars = axes[0].barh(np.arange(len(keys)) + offset,
                           [summary["methods"][role]["stages"].get(key, 0) for key in keys],
                           height=.36, color=COLORS[role], label=SHORT_LABELS[role])
        axes[0].bar_label(bars, padding=3, fontsize=8)
        bars = axes[1].barh(np.arange(len(legacy.FLAG_LABELS)) + offset,
                           [summary["methods"][role]["observations"]["failed"]["0.006"][key]["positive"] for key in legacy.FLAG_LABELS],
                           height=.36, color=COLORS[role], label=f"{SHORT_LABELS[role]}: n={summary['methods'][role]['failed']}")
        axes[1].bar_label(bars, padding=3, fontsize=8)
    axes[0].set_yticks(range(len(keys)), [legacy.STAGE_LABELS[key] for key in keys])
    axes[1].set_yticks(range(len(legacy.FLAG_LABELS)), list(legacy.FLAG_LABELS.values()))
    axes[0].set_title("Final recovery-path stop stage (exclusive)")
    axes[1].set_title("Raw observations among failures (overlap)")
    for ax in axes:
        ax.invert_yaxis()
        ax.set_xlabel("Number of validation objects")
        ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("Threshold 0.006 | observed termination and geometry, not unique failure causes", fontsize=11)
    unknown = "; ".join(SHORT_LABELS[role] + ": " + "/".join(str(
        summary["methods"][role]["observations"]["failed"]["0.006"][key]["unknown"])
        for key in legacy.FLAG_LABELS) for role in ROLES)
    fig.supxlabel("Unknown (owner / twin / endpoint / pair): " + unknown, fontsize=8)
    save(fig, "03_failure_stages_and_observations")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, flag in zip(axes, tuple(legacy.FLAG_LABELS)[:3]):
        for role in ROLES:
            values = [summary["methods"][role]["observations"]["failed"][format(t, "g")][flag] for t in legacy.THRESHOLDS]
            ax.plot(range(len(legacy.THRESHOLDS)), [v["positive"] / v["denominator"] if v["denominator"] else 0 for v in values],
                    "o-", color=COLORS[role], label=SHORT_LABELS[role])
        ax.set_xticks(range(len(legacy.THRESHOLDS)), [format(t, "g") for t in legacy.THRESHOLDS])
        ax.set_ylim(0, 1.04)
        ax.set_title(legacy.FLAG_LABELS[flag])
        ax.set_xlabel("Length threshold")
        ax.set_ylabel("Fraction of all failed objects")
    axes[0].legend()
    fig.suptitle("Fixed thresholds | unknown remains in denominator | twin agreement is by construction", fontsize=11)
    save(fig, "04_failure_threshold_sensitivity")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
    for ax, scope in zip(axes, ("all", "failed")):
        maximum = 1
        for offset, flag, color, label in ((-.25, "owner_mesh_gap", "#bb713e", "Prediction gap"),
                                         (0, "gt_owner_mesh_gap", "#8c98a2", "GT mesh gap"),
                                         (.25, "owner_gap_gt_clear", "#527f9b", "Pred gap; GT within threshold")):
            values = [summary["methods"][role]["observations"][scope]["0.006"][flag]["positive"] for role in ROLES]
            maximum = max(maximum, *values)
            bars = ax.bar(np.arange(2) + offset, values, width=.23, color=color, label=label)
            ax.bar_label(bars, padding=3, fontsize=8)
        ax.set_xticks(range(2), labels)
        ax.set_title("All validation objects" if scope == "all" else "Failed validation objects")
        ax.set_ylabel("Objects with owner P95 > 0.006")
        ax.set_ylim(0, maximum * 1.4)
        ax.legend(fontsize=8)
    fig.suptitle("GT discretization control | conservative count remains an observation, not proof of cause", fontsize=11)
    save(fig, "05_owner_gap_gt_control")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7))
    paired = summary["paired"]["post_geometry"]["common_native_valid"]
    for ax, metric in zip(axes, ("face_cd", "edge_cd")):
        values = [paired["metrics"][metric][role]["mean"] for role in ROLES]
        if all(value is not None for value in values):
            bars = ax.bar(labels, values, color=colors)
            ax.bar_label(bars, labels=[f"{value:.5g}" for value in values], padding=4)
            ax.set_ylim(0, max(values) * 1.22)
        ax.set_title(f"STEP {metric}: same {paired['computable_count']} CADs")
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
    fig.suptitle(f"Common native-valid and computable | {paired['computable_count']}/{paired['intended_count']} | "
                 f"geometry missing: {paired['missing_count']} | squared CD", fontsize=11)
    save(fig, "06_post_geometry_common_valid")


def self_test(_args) -> None:
    check_signature()
    legacy.self_test()
    post = {role: {} for role in ROLES}
    fixtures = (("both", True, True, True), ("regressed", True, False, True),
                ("missing", True, True, False), ("rescued", False, True, True),
                ("neither", False, False, True))
    for sid, old, new, computable in fixtures:
        for role, valid, available in ((ROLES[0], old, True), (ROLES[1], new, computable)):
            post[role][sid] = {"valid": valid, "row": {"geometry_status": "ok" if available else "timeout",
                              "face_cd": .01 if available else "", "edge_cd": .02 if available else ""}}
    ids = list(post[ROLES[0]])
    groups = paired_groups(post, ids)
    assert {key: len(value) for key, value in groups.items()} == {
        "both_valid": 2, "rescued": 1, "regressed": 1, "both_failed": 1}
    geometry = paired_geometry(post, ids)
    assert geometry["common_native_valid"]["ids"] == ["both"]
    assert geometry["common_native_valid"]["intended_count"] == 2
    assert geometry["common_native_valid"]["missing_count"] == 1
    assert geometry["common_computable"]["computable_count"] == 4
    print("Two-role checks passed: paired partition; invalid-but-computable excluded from common-valid; valid missing preserved")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--evaluation-root", type=Path, default=HERE)
    common.add_argument("--split-list", type=Path, default=SPLIT)
    common.add_argument("--data-root", type=Path, default=Path("/mnt/d/data/deepcad_v7"))
    measure_parser = sub.add_parser("measure", parents=[common])
    measure_parser.add_argument("--workers", type=int, default=8)
    report_parser = sub.add_parser("report", parents=[common])
    report_parser.add_argument("--raw-summary", type=Path, default=RAW_SUMMARY)
    report_parser.add_argument("--export-validation", type=Path, required=True)
    sub.add_parser("self-test")
    args = parser.parse_args()
    {"measure": measure, "report": aggregate, "self-test": self_test}[args.command](args)


if __name__ == "__main__":
    main()
