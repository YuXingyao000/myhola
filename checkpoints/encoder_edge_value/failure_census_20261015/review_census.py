"""Recheck frozen census evidence without importing the reporting pipeline.

The independent 28-predicate implementation is reused unchanged from the prior
audited evaluation, with its source SHA recorded. Geometry is not recomputed.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    root = args.root.resolve()
    repo = Path(__file__).resolve().parents[3]
    prior = repo / "checkpoints/intersection_refinement/point_feedback/evaluation_20261015/build_independent_review.py"
    spec = importlib.util.spec_from_file_location("prior_independent_predicates", prior)
    independent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(independent)
    digest = independent.digest
    manifest, baseline = read(root / "manifest.json"), read(root / "baseline_manifest.json")
    summary, compact = read(root / "summary.json"), read(root / "compact_summary.json")
    pairs = read(root / "cases.json")["pairs"]
    assert manifest["input_contract"] == summary["input_contract"] == compact["input_contract"] == "same_geometry_source"
    assert compact["full_summary_sha256"] == digest(root / "summary.json")
    cases = {"candidate": {r["id"]: r for r in manifest["cases"]},
             "baseline": {r["id"]: r for r in baseline["cases"]}}
    assert len(pairs) == len(cases["candidate"]) == len(cases["baseline"])
    assert {p["id"] for p in pairs} == cases["candidate"].keys() == cases["baseline"].keys()
    predicates = 0
    unknown_traces = []
    for pair in pairs:
        sid = pair["id"]
        for role in ("candidate", "baseline"):
            row, case = pair[role], cases[role][sid]
            assert row["method"] == case["method"] == summary["methods"][role]
            assert row["original_native_valid"] is case["post_record"]["native_valid_solid"]
            sources = row["source_files"]
            data = {}
            for kind in ("trace", "upstream", "saved_step"):
                path = Path(sources[kind]["path"])
                assert sources[kind]["sha256"] == digest(path)
                data[kind] = read(path)
                assert data[kind]["method"] == row["method"]
            trace = data["trace"]
            verified = bool(independent.verified(trace))
            assert row["trace_verified"] is verified
            assert row["terminal_stage"] == independent.terminal(trace, row["original_native_valid"])
            if verified:
                original_step = independent.step_data_digest(case["paths"]["post_step"])
                replay_step = independent.step_data_digest(Path(sources["trace"]["path"]).parents[1] / "replay" / sid / "recon_brep.step")
                assert original_step == replay_step == trace["step_data_comparison"]["original"] == trace["step_data_comparison"]["replay"]
            else:
                unknown_traces.append({"id": sid, "role": role})
            assert set(row["flags"]) == independent.THRESHOLDS
            for threshold, flags in row["flags"].items():
                calculated = independent.all_observation_flags(trace, data["upstream"], data["saved_step"],
                    case["raw_record"], sources["trace"]["sha256"], threshold)
                assert calculated.keys() == flags.keys() == independent.OBSERVATIONS
                for key, value in calculated.items():
                    assert value is flags[key], (sid, role, threshold, key, value, flags[key])
                    predicates += 1

    groups = [("main", [p["candidate"] for p in pairs if not p["candidate"]["original_native_valid"]], summary["main"]),
              ("baseline_failures", [p["baseline"] for p in pairs if not p["baseline"]["original_native_valid"]], summary["baseline_failures"])]
    for cohort, group in summary["cohorts"].items():
        subset = [pair for pair in pairs if pair["cohort"] == cohort]
        assert len(subset) == group["count"]
        for role in ("candidate", "baseline"):
            groups.append((cohort + ":" + role, [p[role] for p in subset], group[role]))
        for threshold, observations in group["transitions"].items():
            for key, expected in observations.items():
                actual = {name: [] for name in expected["counts"]}
                for pair in subset:
                    a, b = (pair[role]["flags"][threshold][key] for role in ("baseline", "candidate"))
                    category = "unknown" if type(a) is not bool or type(b) is not bool else "both_present" if a and b else "both_absent" if not a and not b else "present_to_absent" if a else "absent_to_present"
                    actual[category].append(pair["id"])
                assert actual == expected["ids"]
                assert {k: len(v) for k, v in actual.items()} == expected["counts"]
                assert compact["cohorts"][cohort]["transitions"][threshold][key] == [len(actual[k]) for k in compact["columns"]["transitions"]]
    for label, rows, group in groups:
        assert len(rows) == group["denominator"]
        assert sum(row["trace_verified"] for row in rows) == group["verified_traces"]
        actual_terminal = Counter(row["terminal_stage"] for row in rows)
        assert all(actual_terminal[key] == value["count"] for key, value in group["terminal"].items())
        assert sum(actual_terminal.values()) == len(rows)
        for threshold, observations in group["observations"].items():
            for key, expected in observations.items():
                actual = {name: [] for name in ("positive", "negative", "unknown")}
                for row in rows:
                    flag = row["flags"][threshold][key]
                    actual["positive" if flag is True else "negative" if flag is False else "unknown"].append(row["id"])
                for category, ids in actual.items():
                    assert len(ids) == expected[category] and sorted(ids) == sorted(expected["ids"][category])
        faces = [face for row in rows for face in (row["failed_face_context"]["faces"] or [])]
        context = group["failed_face_context"]
        assert len(faces) == context["counted_final_failed_faces"]
        for code, expected in context["codes"].items():
            for category, field in (("face_count", "codes"), ("direct_face_count", "direct_codes"), ("context_face_count", "context_codes")):
                assert expected[category] == sum(code in face[field] for face in faces)

    controls = read(root / "optimization_regressions.json")
    assert [r["id"] for r in controls["cases"]] == manifest["optimization_regression_ids"]
    fit_predicates = 0
    for row in controls["cases"]:
        sid = row["id"]
        case, fit = cases["candidate"][sid], row["fit_only"]
        fit_root = root / "fit_only_counterfactual"
        trace, upstream = read(fit_root / "traces" / (sid + ".json")), read(fit_root / "upstream" / (sid + ".json"))
        assert trace["method"] == fit["method"] == manifest["candidate_method"]
        assert trace["post_mode"] == "fit_only"
        assert case["fit_record"]["raw_sha256"] == case["post_record"]["raw_sha256"]
        assert row["same_raw_input_sha256"] is True
        for threshold, flags in fit["flags"].items():
            calculated = independent.all_observation_flags(trace, upstream, {}, case["raw_record"], digest(fit_root / "traces" / (sid + ".json")), threshold)
            for key, value in calculated.items():
                assert value is flags[key], (sid, "fit_only", threshold, key)
                fit_predicates += 1
        if independent.verified(trace) and row["optimized"]["trace_verified"]:
            with np.load(fit_root / "replay" / sid / "trace_geometry.npz", allow_pickle=False) as a, np.load(root / "replay" / sid / "trace_geometry.npz", allow_pickle=False) as b:
                for key, expected in row["same_selected_arrays"].items():
                    assert bool(np.array_equal(a[key], b[key])) is expected

    audit = read(root / "source_asset_audit.json")
    for source in audit["source_assets"]:
        path = Path(source["path"])
        assert source["exists"] == path.is_file()
        assert source["sha256"] == (digest(path) if path.is_file() else None)
    output = {"schema": "encoder_census_independent_review_v1", "status": "passed",
              "created_at_utc": datetime.now(timezone.utc).isoformat(), "methods": summary["methods"],
              "scope": "Independent old audit predicates reused; no reporting-pipeline helpers imported. Measured geometry itself is not rerun.",
              "reviewer_sha256": digest(Path(__file__)), "predicate_source": {"path": str(prior), "sha256": digest(prior)},
              "paired_cases": len(pairs), "trace_records": 2 * len(pairs), "unknown_traces": unknown_traces,
              "main_predicate_comparisons": predicates, "fit_predicate_comparisons": fit_predicates,
              "fit_path_controls": len(controls["cases"]), "source_assets_unchanged": len(audit["source_assets"]),
              "group_threshold_tallies_and_paired_transitions": "passed", "failed_face_count_aggregation": "passed",
              "summary_sha256": digest(root / "summary.json"), "cases_sha256": digest(root / "cases.json")}
    (root / "independent_review.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(output, ensure_ascii=False))


if __name__ == "__main__":
    main()
