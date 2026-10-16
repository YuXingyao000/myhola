"""Independent census audit: frozen identities, direct trace predicates, summaries."""
from __future__ import annotations

from collections import Counter
import csv
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
EVAL = ROOT.parent / "evaluation_20261013"
HISTORICAL = REPO / "checkpoints/intersection_refinement/failure_census_20261011"
SPLIT = REPO / "checkpoints/joint_boundary/validation/test_contract_audit/accepted.txt"
THRESHOLDS = {"0.002", "0.006", "0.01", "0.08"}
OBSERVATIONS = {
    "face_without_boundary", "never_wire", "never_closed_admitted",
    "closed_admitted_still_failed", "strict_closed_candidate_failed",
    "final_success_discarded_wire", "raw_extra_gt_pair", "raw_missing_gt_pair",
    "raw_duplicate_directed_pair", "raw_same_face_pair", "face_count_mismatch",
    "selection_removed_gt_pair", "reciprocal_excess_disagreement",
    "raw_new_cross_pair_curve_nearness", "raw_new_face_grid_nearness",
    "selected_endpoint_gap", "optimized_endpoint_gap",
    "selected_new_open_edge_endpoint_collapse", "optimized_new_open_edge_endpoint_collapse",
    "fit_face_residual", "fit_curve_residual", "owner_independent", "owner_after",
    "closed_failed_owner_gap", "optimization_face_away", "optimization_edge_away",
    "sewing_free_edges", "saved_near_free_edges",
}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def csv_rows(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


def frozen_identities():
    manifest = read(ROOT / "manifest.json")
    baseline = read(ROOT / "baseline_manifest.json")
    controls = read(ROOT / "baseline_control_manifest.json")
    reuse = read(ROOT / "baseline_reuse.json")
    ids = set(SPLIT.read_text().split())
    assert len(ids) == 2417 and sha(SPLIT) == "2864849e8240134c6033cc6cd64efa19337f67fea94b89a493690ae84355b7b3"
    rows = csv_rows(EVAL / "post/optimized/samples.csv")
    old = {row["sample_id"]: row for row in rows if row["method"] == "intersection"}
    new = {row["sample_id"]: row for row in rows if row["method"] == "local"}
    assert set(old) == set(new) == ids
    groups = {name: [] for name in ("both_failed", "new_failure", "rescued", "both_valid")}
    for sid in sorted(ids):
        assert old[sid]["native_valid_solid"] in ("True", "False") and new[sid]["native_valid_solid"] in ("True", "False")
        a, b = old[sid]["native_valid_solid"] == "True", new[sid]["native_valid_solid"] == "True"
        name = "both_valid" if a and b else "new_failure" if a else "rescued" if b else "both_failed"
        groups[name].append(sid)
    assert groups == manifest["cohorts"]
    failed = sorted(groups["both_failed"] + groups["new_failure"])
    selected = set(failed + groups["rescued"])
    assert failed == manifest["failed_ids"]
    for document, method in ((manifest, "local"), (baseline, "intersection")):
        assert len(document["cases"]) == len(selected) and {case["id"] for case in document["cases"]} == selected
        assert all(case["method"] == method for case in document["cases"])
        outcomes = new if method == "local" else old
        assert all(case["post_record"]["native_valid_solid"] is (outcomes[case["id"]]["native_valid_solid"] == "True") for case in document["cases"])
    reuse_ids = {case["id"] for case in reuse["cases"] if case["eligible"]}
    control_ids = {case["id"] for case in controls["cases"]}
    assert len(control_ids) == len(controls["cases"]) and not (reuse_ids & control_ids) and reuse_ids | control_ids == selected
    assert reuse["eligible_count"] == len(reuse_ids)
    historical = read(HISTORICAL / "manifest.json")
    historical_ids = {case["id"] for case in historical["cases"]}
    assert reuse_ids <= historical_ids and all(case["method"] == "intersection" for case in historical["cases"])
    checked_reused_files = 0
    for case in reuse["cases"]:
        if case["eligible"]:
            assert case["method"] == "intersection" and not case["reasons"]
            for kind, source in case["sources"].items():
                path = Path(source["path"])
                assert path.is_relative_to(HISTORICAL) and sha(path) == source["sha256"]
                if kind != "stage_geometry":
                    assert read(path)["method"] == "intersection"
                checked_reused_files += 1
    fit = {row["sample_id"]: row for row in csv_rows(EVAL / "post/fit_only/samples.csv") if row["method"] == "local"}
    regressions = sorted(sid for sid in ids if fit[sid]["native_valid_solid"] == "True" and new[sid]["native_valid_solid"] == "False")
    assert regressions == manifest["optimization_regression_ids"]
    contract = REPO / "checkpoints/joint_boundary/validation/test_contract_audit"
    contract_summary = read(contract / "summary.json")
    original = set(Path(contract_summary["split_list"]).read_text().split())
    assert sha(Path(contract_summary["split_list"])) == contract_summary["split_sha256"]
    excluded = {row["model_id"] for row in csv_rows(contract / "rejected.csv")}
    assert len(original) == 2424 and original - ids == excluded and len(excluded) == 7
    result = {"accepted": len(ids), "original": len(original), "contract_excluded_ids": sorted(excluded),
              "contract_exclusion_source": str(contract / "rejected.csv"), "contract_exclusion_sha256": sha(contract / "rejected.csv"),
              "cohorts": {name: len(values) for name, values in groups.items()}, "candidate_failure_denominator": len(failed),
              "selected_paired_ids": len(selected), "excluded_both_valid_ids": len(groups["both_valid"]),
              "baseline_reuse": len(reuse_ids), "baseline_new_controls": len(control_ids),
              "historical_reuse_files_checked": checked_reused_files, "reuse_partition_verified": True,
              "optimization_regression_ids": regressions}
    baseline_cases = {case["id"]: case for case in baseline["cases"]}
    result["raw_topology_profile"] = {}
    for label, selected_cases in (
        ("all_candidate_failures", [case for case in manifest["cases"] if not case["post_record"]["native_valid_solid"]]),
        ("regressed", [case for case in manifest["cases"] if case["cohort"] == "new_failure"]),
        ("persistent", [case for case in manifest["cases"] if case["cohort"] == "both_failed"]),
        ("rescued", [case for case in manifest["cases"] if case["cohort"] == "rescued"]),
    ):
        result["raw_topology_profile"][label] = {
            "denominator": len(selected_cases),
            "candidate_exact": sum(case["raw_record"]["topology_fp"] == case["raw_record"]["topology_fn"] == 0 for case in selected_cases),
            "baseline_exact": sum(baseline_cases[case["id"]]["raw_record"]["topology_fp"] == baseline_cases[case["id"]]["raw_record"]["topology_fn"] == 0 for case in selected_cases),
            "candidate_extra": sum(case["raw_record"]["topology_fp"] > 0 for case in selected_cases),
            "candidate_missing": sum(case["raw_record"]["topology_fn"] > 0 for case in selected_cases),
            "candidate_extra_and_missing": sum(case["raw_record"]["topology_fp"] > 0 and case["raw_record"]["topology_fn"] > 0 for case in selected_cases),
        }
    return result, manifest, baseline, reuse_ids


def protected_sources():
    sources = read(ROOT / "source_asset_audit.json")["source_assets"]
    checked = 0
    for item in sources:
        path = Path(item["path"])
        assert path.is_file() == item["exists"], str(path)
        assert (sha(path) if path.is_file() else None) == item["sha256"], str(path)
        checked += 1
    result = {"count": checked, "changed": 0, "source_audit_sha256": sha(ROOT / "source_asset_audit.json")}
    report_path = ROOT / "source_preservation.json"
    if report_path.is_file():
        report = read(report_path)
        assert report["asset_count"] == checked and report["mismatch_count"] == 0 and report["changed_paths"] == []
        assert report["source_audit_sha256"] == result["source_audit_sha256"]
        result["report_sha256"] = sha(report_path)
        result["independent_rehash_matches_report"] = True
    return result


def verified(trace):
    return trace.get("status") == "ok" and not trace.get("instrumentation_errors") and trace.get("comparison", {}).get("same_recorded_outcome") is True and trace.get("step_data_comparison", {}).get("equal") is True


def step_data_sha(path):
    data = Path(path).read_bytes()
    start = data.index(b"DATA;") + len(b"DATA;")
    stop = data.index(b"ENDSEC;", start)
    return hashlib.sha256(data[start:stop]).hexdigest()


def terminal(trace, valid):
    if valid:
        return "native_valid"
    if not verified(trace):
        return "unknown"
    checks = [item.get("legacy_saved_step_check") for item in trace.get("save_attempts", [])]
    if True in checks:
        return "native_readback_invalid" if trace.get("replay", {}).get("native_valid_solid") is False else "unresolved"
    if False in checks:
        return "saved_step_rejected"
    if trace.get("trim_gate", {}).get("solid_attempt_allowed") is False:
        return "trim_gate"
    attempts = trace.get("solid_attempts", [])
    return "solid_construction" if attempts and all(item.get("result", {}).get("present") is False for item in attempts) else "unresolved"


def owner_flag(trace, stage, threshold):
    if not verified(trace):
        return None
    fit = trace.get("fit_residuals", {})
    count = trace.get("fitting", {}).get("num_curves")
    rows = [(edge.get("edge_index"), owner) for edge in fit.get("curve_owner", []) for owner in edge.get(stage, [])]
    identities = [(edge, owner.get("face_index")) for edge, owner in rows]
    if len(set(identities)) != len(identities) or any(type(e) is not int or type(f) is not int for e, f in identities):
        return None
    finite = lambda value: isinstance(value, (int, float)) and math.isfinite(value)
    if any(finite(owner.get("max")) and owner["max"] > threshold for _, owner in rows):
        return True
    complete = bool(rows) and type(count) is int and count >= 0 and len(rows) == 2 * count and fit.get("status") in ("ok", "partial")
    complete = complete and all(owner.get("status") == "ok" and owner.get("count", 0) > 0 and owner["count"] == owner.get("finite_count") and finite(owner.get("max")) and not owner.get("errors") for _, owner in rows)
    return False if complete else None


def trim_flags(trace):
    names = ("never_wire", "never_closed_admitted", "closed_admitted_still_failed", "strict_closed_candidate_failed")
    unknown = {name: None for name in names}
    if not verified(trace):
        return unknown
    runs = [run for run in trace.get("trim_runs", []) if run.get("context") == "main"]
    if len(runs) != 1 or not isinstance(runs[0].get("failed_face_indices"), list):
        return unknown
    run = runs[0]
    categories, strict_values = [], []
    for face in set(run["failed_face_indices"]):
        attempts = [attempt for attempt in run.get("attempts", []) if attempt.get("face_index") == face]
        complete, admitted, strict = bool(attempts), False, False
        for attempt in attempts:
            if attempt.get("wire_returned") is False:
                continue
            detail = attempt.get("trim_detail", {})
            wires = detail.get("wire_events")
            if not isinstance(wires, list) or detail.get("outcome", "unknown") == "unknown":
                complete = False
                continue
            ok = [wire.get("added") is True and wire.get("repaired_closed") is True for wire in wires]
            admitted |= any(ok)
            strict |= bool(ok) and all(ok) and attempt.get("trim_returned") is False
        category = "closed_admitted_still_failed" if admitted else "unknown" if not complete else "never_wire" if all(a.get("wire_returned") is False for a in attempts) else "never_closed_admitted"
        categories.append(category)
        strict_values.append(True if strict else False if complete else None)
    result = {name: True if name in categories else None if "unknown" in categories else False for name in names[:3]}
    result[names[3]] = True if True in strict_values else None if None in strict_values else False
    return result


def stratified_failure_mechanisms(pairs):
    """Post-hoc description only; preserve the original evaluation denominators."""
    failed = [pair for pair in pairs if not pair["candidate"]["original_native_valid"]]
    exact = lambda row: row["raw_metrics"]["topology_fp"] == row["raw_metrics"]["topology_fn"] == 0
    strata = {
        "all_failed_adjacency_exact": [pair["candidate"] for pair in failed if exact(pair["candidate"])],
        "all_failed_adjacency_inexact": [pair["candidate"] for pair in failed if not exact(pair["candidate"])],
        "regressed_adjacency_exact": [pair["candidate"] for pair in failed if pair["cohort"] == "regressed" and exact(pair["candidate"])],
    }
    assert len(strata["all_failed_adjacency_exact"]) + len(strata["all_failed_adjacency_inexact"]) == len(failed)
    assert {row["id"] for row in strata["regressed_adjacency_exact"]} <= {row["id"] for row in strata["all_failed_adjacency_exact"]}
    result = {
        "interpretation": "Post-hoc descriptive stratification, not a preregistered causal test; full evaluation metrics and the dynamically derived main failure denominator are unchanged. Exact adjacency checks only face-pair adjacency, not complete B-Rep topology or boundary geometry.",
        "main_failure_denominator": len(failed),
        "threshold": 0.006,
        "strata": {},
    }
    for label, rows in strata.items():
        observations = {}
        for key in ("owner_after", "strict_closed_candidate_failed", "never_closed_admitted", "final_success_discarded_wire"):
            counts = {"positive": 0, "negative": 0, "unknown": 0}
            for row in rows:
                value = row["flags"]["0.006"][key]
                counts["positive" if value is True else "negative" if value is False else "unknown"] += 1
            assert sum(counts.values()) == len(rows)
            observations[key] = counts
        result["strata"][label] = {
            "denominator": len(rows),
            "ids": sorted(row["id"] for row in rows),
            "terminal_counts": dict(sorted(Counter(row["terminal_stage"] for row in rows).items())),
            "observations": observations,
        }
    return result


def optimization_counterfactual_review(manifest, pairs):
    document = read(ROOT / "optimization_regressions.json")
    expected_ids = manifest["optimization_regression_ids"]
    assert document["denominator"] == len(expected_ids)
    assert [row["id"] for row in document["cases"]] == expected_ids
    cases = {case["id"]: case for case in manifest["cases"]}
    candidates = {pair["id"]: pair["candidate"] for pair in pairs}
    fit_root = ROOT / "fit_only_counterfactual"
    evidence = []
    for row in document["cases"]:
        sid = row["id"]
        case = cases[sid]
        assert case["fit_record"]["native_valid_solid"] is True
        assert case["post_record"]["native_valid_solid"] is False
        assert row["optimized"] == candidates[sid]
        fit = row["fit_only"]
        trace = read(fit_root / "traces" / (sid + ".json"))
        assert fit["method"] == trace["method"] == "local"
        assert trace["post_mode"] == "fit_only"
        assert trace["original"] == case["fit_record"]
        assert fit["original_native_valid"] is True
        assert fit["trace_verified"] is bool(verified(trace))
        assert fit["terminal_stage"] == terminal(trace, True)
        assert set(fit["flags"]) == THRESHOLDS
        assert all(set(flags) == OBSERVATIONS for flags in fit["flags"].values())
        assert all(value is None or type(value) is bool for flags in fit["flags"].values() for value in flags.values())
        for kind in ("trace", "upstream"):
            directory = "traces" if kind == "trace" else "upstream"
            path = fit_root / directory / (sid + ".json")
            assert Path(fit["source_files"][kind]["path"]) == path
            assert fit["source_files"][kind]["sha256"] == sha(path)
        if verified(trace):
            original_hash = step_data_sha(case["paths"]["fit_step"])
            replay_hash = step_data_sha(fit_root / "replay" / sid / "recon_brep.step")
            assert original_hash == replay_hash == trace["step_data_comparison"]["original"] == trace["step_data_comparison"]["replay"]
        for threshold, flags in fit["flags"].items():
            for flag, value in trim_flags(trace).items():
                assert flags[flag] is value, (sid, "fit_only", flag, threshold)
            for flag, stage in (("owner_independent", "independent_curve"), ("owner_after", "after_periodic_replacement")):
                assert flags[flag] is owner_flag(trace, stage, float(threshold)), (sid, "fit_only", flag, threshold)
        equality = {key: None for key in ("selected_faces", "selected_edges", "selected_connectivity")}
        if fit["trace_verified"] and candidates[sid]["trace_verified"]:
            paths = [location / "replay" / sid / "trace_geometry.npz" for location in (fit_root, ROOT)]
            if all(path.is_file() for path in paths):
                with np.load(paths[0], allow_pickle=False) as first, np.load(paths[1], allow_pickle=False) as second:
                    equality = {key: bool(np.array_equal(first[key], second[key])) if key in first and key in second else None for key in equality}
        assert row["same_selected_arrays"] == equality
        same_input = case["fit_record"]["raw_sha256"] == case["post_record"]["raw_sha256"]
        assert row["same_raw_input_sha256"] is same_input
        evidence.append({"id": sid, "fit_trace_verified": fit["trace_verified"],
                         "optimized_trace_verified": candidates[sid]["trace_verified"],
                         "same_raw_input_sha256": same_input, "same_selected_arrays": equality})
    return {"denominator": len(expected_ids), "ids": expected_ids, "all_records_verified_against_source": True,
            "fit_verified_trace_count": sum(row["fit_trace_verified"] for row in evidence),
            "cases": evidence, "report_sha256": sha(ROOT / "optimization_regressions.json")}


def final_results(manifest, baseline, reuse_ids):
    pairs = read(ROOT / "cases.json")["pairs"]
    summary = read(ROOT / "summary.json")
    compact = read(ROOT / "compact_summary.json")
    assert compact["full_summary_sha256"] == sha(ROOT / "summary.json")
    assert set(summary["cohorts"]) == {"persistent", "regressed", "rescued"}
    assert set(compact["definitions"]) == OBSERVATIONS
    assert compact["population"] == summary["population"] == manifest["population"]
    assert compact["columns"]["observations"] == ["positive", "negative", "unknown"]
    assert compact["columns"]["transitions"] == ["present_to_absent", "absent_to_present", "both_present", "both_absent", "unknown"]
    cases = {"candidate": {case["id"]: case for case in manifest["cases"]}, "baseline": {case["id"]: case for case in baseline["cases"]}}
    assert len(pairs) == len(cases["candidate"]) and {pair["id"] for pair in pairs} == set(cases["candidate"])
    trace_count, predicates = 0, 0
    for pair in pairs:
        sid = pair["id"]
        for role in ("candidate", "baseline"):
            location = ROOT if role == "candidate" else HISTORICAL if sid in reuse_ids else ROOT / "baseline_controls"
            path = location / "traces" / (sid + ".json")
            trace = read(path)
            row, case = pair[role], cases[role][sid]
            assert trace.get("method") == row["method"] == case["method"]
            assert row["original_native_valid"] is case["post_record"]["native_valid_solid"]
            for key in ("sample_id", "method", "mode", "status", "post_status", "raw_sha256", "native_valid_solid", "step_exists", "step_readable"):
                assert trace["original"].get(key) == case["post_record"].get(key), (sid, role, key)
            assert row["trace_verified"] == bool(verified(trace))
            assert row["terminal_stage"] == terminal(trace, case["post_record"]["native_valid_solid"])
            assert set(row["flags"]) == THRESHOLDS
            assert all(set(flags) == OBSERVATIONS for flags in row["flags"].values())
            assert all(value is None or type(value) is bool for flags in row["flags"].values() for value in flags.values())
            # Historical reuse preserves the original path; current baseline
            # manifests point at a read-only symlink to the identical NPZ.
            assert {key: value for key, value in row["raw_metrics"].items() if key != "prediction_file"} == {
                key: value for key, value in case["raw_record"].items() if key != "prediction_file"
            }
            assert row["source_files"]["trace"]["sha256"] == sha(path)
            if verified(trace):
                original_hash = step_data_sha(case["paths"]["post_step"])
                replay_hash = step_data_sha(location / "replay" / sid / "recon_brep.step")
                assert original_hash == replay_hash == trace["step_data_comparison"]["original"] == trace["step_data_comparison"]["replay"]
            for diagnostic, directory in (("upstream", "upstream"), ("saved_step", "saved_step")):
                assert row["source_files"][diagnostic]["sha256"] == sha(location / directory / (sid + ".json"))
            direct_trim = trim_flags(trace)
            for threshold, flags in row["flags"].items():
                for flag, value in direct_trim.items():
                    assert flags[flag] is value, (sid, role, flag, threshold)
                    predicates += 1
                for flag, stage in (("owner_independent", "independent_curve"), ("owner_after", "after_periodic_replacement")):
                    assert flags[flag] is owner_flag(trace, stage, float(threshold)), (sid, role, flag, threshold)
                    predicates += 1
            trace_count += 1
    group_lists = [("main", [p["candidate"] for p in pairs if not p["candidate"]["original_native_valid"]], summary["main"]),
                   ("baseline_failures", [p["baseline"] for p in pairs if not p["baseline"]["original_native_valid"]], summary["baseline_failures"])]
    for cohort, values in summary["cohorts"].items():
        subset = [pair for pair in pairs if pair["cohort"] == cohort]
        assert len(subset) == values["count"]
        for role in ("candidate", "baseline"):
            group_lists.append((cohort + ":" + role, [p[role] for p in subset], values[role]))
        for threshold, observations in values["transitions"].items():
            assert set(values["transitions"]) == THRESHOLDS and set(observations) == OBSERVATIONS
            for key, recorded in observations.items():
                actual = {name: [] for name in recorded["counts"]}
                for pair in subset:
                    a, b = (pair[role]["flags"][threshold][key] for role in ("baseline", "candidate"))
                    name = "unknown" if type(a) is not bool or type(b) is not bool else "both_present" if a and b else "both_absent" if not a and not b else "present_to_absent" if a else "absent_to_present"
                    actual[name].append(pair["id"])
                assert actual == recorded["ids"] and {name: len(ids) for name, ids in actual.items()} == recorded["counts"]
                assert compact["cohorts"][cohort]["transitions"][threshold][key] == [len(actual[name]) for name in compact["columns"]["transitions"]]
    groups_review = {}
    for label, records, group in group_lists:
        compact_group = compact[label] if ":" not in label else compact["cohorts"][label.split(":")[0]][label.split(":")[1]]
        assert len(records) == group["denominator"]
        assert compact_group["denominator"] == len(records)
        assert set(group["observations"]) == THRESHOLDS
        assert all(set(observations) == OBSERVATIONS for observations in group["observations"].values())
        assert sum(row["trace_verified"] for row in records) == group["verified_traces"]
        assert dict(Counter(row["raw_topology"] for row in records)) == group["raw_topology_counts"]
        for name, recorded in group["terminal"].items():
            ids = [row["id"] for row in records if row["terminal_stage"] == name]
            assert ids == recorded["ids"] and len(ids) == recorded["count"]
            assert compact_group["terminal"][name] == len(ids)
        for threshold, observations in group["observations"].items():
            for key, expected in observations.items():
                actual = {"positive": [], "negative": [], "unknown": []}
                for row in records:
                    value = row["flags"][threshold][key]
                    actual["positive" if value is True else "negative" if value is False else "unknown"].append(row["id"])
                for category, ids in actual.items():
                    assert len(ids) == expected[category] and sorted(ids) == sorted(expected["ids"][category])
                assert compact_group["observations"][threshold][key] == [len(actual[name]) for name in compact["columns"]["observations"]]
        context = group["failed_face_context"]
        faces = [face for row in records for face in row["failed_face_context"]["faces"] or []]
        assert len(faces) == context["counted_final_failed_faces"]
        for code, expected in context["codes"].items():
            for category, field in (("face_count", "codes"), ("direct_face_count", "direct_codes"), ("context_face_count", "context_codes")):
                assert expected[category] == sum(code in face[field] for face in faces)
        groups_review[label] = {"denominator": len(records), "verified_traces": group["verified_traces"],
                                "terminal": {key: item["count"] for key, item in group["terminal"].items()},
                                "raw_topology_counts": group["raw_topology_counts"],
                                "key_observations_at_0.006": {key: {name: values[name] for name in ("positive", "negative", "unknown")} for key, values in group["observations"]["0.006"].items()}}
    return {"paired_cases": len(pairs), "direct_trace_records_checked": trace_count,
            "primary_display_tally_grid": {"groups": 4, "thresholds": len(THRESHOLDS), "observations": len(OBSERVATIONS),
                                           "combinations_verified": 4 * len(THRESHOLDS) * len(OBSERVATIONS)},
            "direct_primary_trace_predicate_comparisons": predicates, "all_group_threshold_tallies_and_pair_transitions_verified": True,
            "compact_report_tallies_match_independent_counts": True,
            "failed_face_context_count_aggregation_verified": True, "groups": groups_review,
            "stratified_failure_mechanisms": stratified_failure_mechanisms(pairs),
            "optimization_counterfactual": optimization_counterfactual_review(manifest, pairs),
            "cases_sha256": sha(ROOT / "cases.json"), "summary_sha256": sha(ROOT / "summary.json"),
            "compact_summary_sha256": sha(ROOT / "compact_summary.json")}


def main():
    identities, manifest, baseline, reused = frozen_identities()
    result = {"created_utc": datetime.now(timezone.utc).isoformat(), "reviewer": "audit_training_local",
              "created_display_time": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
              "display_timezone": "Asia/Shanghai",
              "review_builder_sha256": sha(Path(__file__).resolve()),
              "independent_method": "No production census helper imports: identities from original CSVs, source file hashes, direct trim/owner/terminal predicates from traces, all tally/transition counts from cases.",
              "frozen_identities": identities, "source_preservation": protected_sources()}
    if (ROOT / "cases.json").is_file() and (ROOT / "summary.json").is_file():
        result["diagnostics"] = final_results(manifest, baseline, reused)
        result["status"] = "passed"
    else:
        result["status"] = "pending_diagnostics"
    (ROOT / "independent_review.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "frozen_identities": identities, "source_preservation": result["source_preservation"]}, indent=2))


if __name__ == "__main__":
    main()
