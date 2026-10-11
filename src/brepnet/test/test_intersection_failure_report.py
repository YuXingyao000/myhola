import copy
import json

import pytest

from tools import legacy_failure_census_report as legacy
from tools.intersection_failure_report import (
    CONTEXT_ERRORS, OBSERVATIONS, THRESHOLDS, compact_summary, context_flag, earliest_observation, evidence_highlights,
    enrich_records, failed_face_context_errors, faces_without_boundary, owner_residual_stages,
    paired_transitions, summarize_records, validate_cohorts, validate_diagnostics,
)


def trace():
    return {"sample_id": "00000001", "status": "ok", "instrumentation_errors": [],
            "comparison": {"same_recorded_outcome": True},
            "step_data_comparison": {"equal": True}, "replay": {"native_valid_solid": False}}


def case(sid="00000001", method="intersection", valid=False):
    return {"id": sid, "method": method, "gt_face_count": 2,
            "raw_record": {"model_id": sid, "method": method, "pred_face_count": 2,
                           "gt_face_count": 2, "topology_fp": 0, "topology_fn": 0},
            "post_record": {"sample_id": sid, "method": method, "native_valid_solid": valid},
            "fit_record": {"sample_id": sid, "method": method, "native_valid_solid": valid}}


def projection(face, maximum=.001, *, partial=False):
    return {"face_index": face, "count": 9, "finite_count": 8 if partial else 9,
            "status": "partial" if partial else "ok", "max": maximum, "mean": maximum / 2,
            "rms": maximum / 2, "errors": ["one missing point"] if partial else []}


def fitted_trace():
    t = trace()
    t["fitting"] = {"num_curves": 1, "faces": [{}, {}]}
    t["fit_residuals"] = {"status": "ok", "curve_owner": [{"edge_index": 0, "face_pair": [0, 1],
        "independent_curve": [projection(0, .001), projection(1, .003)],
        "after_periodic_replacement": [projection(0, .002), projection(1, .001)]}]}
    return t


def test_success_controls_are_explicit_even_when_replay_unknown():
    assert legacy.terminal_stage({}, True) == "native_valid"
    assert legacy.terminal_stage({}, False) == "unknown"
    t = trace(); t["replay"]["native_valid_solid"] = True
    assert legacy.terminal_stage(t) == "native_valid"
    # Verification does not become true just because the original was successful.
    assert not legacy.trace_verified({})


def test_generic_aggregate_retains_old_measurements_and_supports_success(tmp_path):
    c = case(valid=True)
    manifest = {"cases": [c], "population": {}, "face_identity_contract": {"gt_order_preserved": True}}
    rows, summary = legacy.aggregate(manifest, tmp_path)
    assert rows[0]["terminal_stage"] == "native_valid"
    assert rows[0]["trace_verified"] is False
    assert all(v is None for v in rows[0]["trace_mechanisms"].values())
    assert summary["terminal_stages"]["native_valid"]["count"] == 1
    assert summary["terminal_stages"]["unknown"]["count"] == 0
    assert rows[0]["raw_metrics"] == c["raw_record"]


def test_owner_stages_match_identities_and_keep_direction_of_change():
    result = owner_residual_stages(fitted_trace())
    assert result["independent_curve"]["threshold_flags"]["0.002"] is True
    assert result["after_periodic_replacement"]["threshold_flags"]["0.002"] is False
    assert result["object_max_delta"] == pytest.approx(-.001)
    assert len(result["matched_owners"]) == 2
    assert result["matched_owners"][0]["delta_max"] == pytest.approx(.001)


def test_partial_owner_measurement_preserves_positive_but_not_negative():
    t = fitted_trace()
    t["fit_residuals"]["curve_owner"][0]["independent_curve"][1] = projection(1, .003, partial=True)
    result = owner_residual_stages(t)
    assert result["independent_curve"]["threshold_flags"]["0.002"] is True
    assert result["independent_curve"]["threshold_flags"]["0.006"] is None
    assert result["object_max_delta"] is None
    assert len(result["matched_owners"]) == 1


def test_duplicate_owner_identity_cannot_satisfy_complete_coverage():
    t = fitted_trace()
    t["fit_residuals"]["curve_owner"][0]["independent_curve"][1]["face_index"] = 0
    result = owner_residual_stages(t)
    assert result["independent_curve"]["status"] == "unknown"
    assert result["independent_curve"]["threshold_flags"]["0.08"] is None


def test_owner_residual_requires_identical_replay():
    t = fitted_trace(); t["step_data_comparison"]["equal"] = False
    result = owner_residual_stages(t)
    assert result["independent_curve"]["observed_max"] is None
    assert result["matched_owners"] == []


def candidate(face, direct=(), contextual=(), *, errors=()):
    return {"face_index": face, "wire_returned": True, "trim_returned": False,
            "trim_detail": {"outcome": "invalid_face", "wire_events": [{"added": True, "repaired_closed": True}],
                            "final_brepcheck": {"with_errors": [{"statuses": list(direct), "contexts": [{"statuses": list(contextual)}]}],
                                                "inspection_errors": list(errors)}}}


def test_only_final_failed_faces_count_and_direct_context_are_both_retained():
    t = trace()
    a = candidate(0, ["BRepCheck_IntersectingWires"], ["BRepCheck_InvalidCurveOnSurface"])
    t["trim_runs"] = [{"context": "main", "num_faces": 2, "failed_face_indices": [0],
                      "attempts": [a, copy.deepcopy(a), candidate(1, ["BRepCheck_SelfIntersectingWire"])]}]
    result = failed_face_context_errors(t)
    assert len(result["faces"]) == 1
    row = result["faces"][0]
    assert row["codes"] == ["BRepCheck_IntersectingWires", "BRepCheck_InvalidCurveOnSurface"]
    assert row["direct_codes"] == ["BRepCheck_IntersectingWires"]
    assert row["context_codes"] == ["BRepCheck_InvalidCurveOnSurface"]
    assert row["inspected_attempts"] == 2
    assert "BRepCheck_IntersectingWires" in CONTEXT_ERRORS


def test_no_candidate_inspection_is_unknown_not_clean_check():
    t = trace()
    t["trim_runs"] = [{"context": "main", "num_faces": 1, "failed_face_indices": [0],
                      "attempts": [{"face_index": 0, "wire_returned": False}]}]
    result = failed_face_context_errors(t)
    assert result["faces"][0]["inspection_state"] == "no_candidate_inspected"
    assert context_flag({"failed_face_context": result}, "BRepCheck_IntersectingWires") is None


def test_face_without_boundary_is_confirmed_without_guessing_missing_attempts():
    t = trace()
    t["trim_runs"] = [{"context": "main", "num_faces": 3, "connectivity": [[0, 0, 1]],
                      "failed_face_indices": [2], "attempts": []}]
    assert faces_without_boundary(t) == {"flag": True, "face_ids": [2]}
    t["trim_runs"][0]["connectivity"].append([1, 1, 2])
    assert faces_without_boundary(t) == {"flag": False, "face_ids": []}
    t["trim_runs"][0].pop("connectivity")
    assert faces_without_boundary(t)["flag"] is None


def test_any_observed_error_survives_partial_candidate_check():
    t = trace(); a = candidate(0, ["BRepCheck_IntersectingWires"], errors=["one inspection failed"])
    t["trim_runs"] = [{"context": "main", "num_faces": 1, "failed_face_indices": [0], "attempts": [a]}]
    result = failed_face_context_errors(t)
    assert context_flag({"failed_face_context": result}, "BRepCheck_IntersectingWires") is True
    assert context_flag({"failed_face_context": result}, "BRepCheck_SelfIntersectingWire") is None


def pair(sid, before, after):
    return {"id": sid, "legacy": {"flags": {"0.006": {"x": before}}},
            "intersection": {"flags": {"0.006": {"x": after}}}}


def test_paired_unknown_is_never_improvement():
    pairs = [pair("1", True, None), pair("2", None, False), pair("3", True, False), pair("4", False, True), pair("5", True, True), pair("6", False, False)]
    result = paired_transitions(pairs, "x", "0.006")
    assert result["denominator"] == 6
    assert result["counts"] == {"present_to_absent": 1, "absent_to_present": 1, "both_present": 1, "both_absent": 1, "unknown": 2}
    assert result["ids"]["unknown"] == ["1", "2"]


def test_cohort_partition_is_exact_and_validity_requires_bool():
    new = [case("1"), case("2"), case("3", valid=True)]
    failed = [case("1", "legacy"), case("3", "legacy")]
    controls = [case("2", "legacy", True)]
    expected = {"persistent": 1, "regressed": 1, "rescued": 1}
    assert validate_cohorts(new, failed, controls, expected=expected) == {"persistent": ["1"], "regressed": ["2"], "rescued": ["3"]}
    new[0]["post_record"]["native_valid_solid"] = 0
    with pytest.raises(ValueError, match="validity"):
        validate_cohorts(new, failed, controls, expected=expected)


@pytest.mark.parametrize("change", ["missing", "duplicate", "both_success"])
def test_cohort_rejects_changed_population(change):
    new = [case("1"), case("2"), case("3", valid=True)]
    failed = [case("1", "legacy"), case("3", "legacy")]
    controls = [case("2", "legacy", True)]
    if change == "missing": new.pop()
    elif change == "duplicate": new.append(copy.deepcopy(new[0]))
    else: new[1]["post_record"]["native_valid_solid"] = True
    with pytest.raises(ValueError):
        validate_cohorts(new, failed, controls, expected={"persistent": 1, "regressed": 1, "rescued": 1})


def test_earliest_observation_reports_unknown_earlier_stages():
    flags = {key: False for key, *_ in OBSERVATIONS}
    flags["raw_extra_gt_pair"] = None
    flags["owner_independent"] = True
    result = earliest_observation(flags)
    assert result["stage"] == "independent_fitting"
    assert result["earlier_unknown"] == ["raw_extra_gt_pair"]
    flags["raw_missing_gt_pair"] = True
    assert earliest_observation(flags)["stage"] == "raw"


def test_new_summary_has_fixed_denominators_and_deduplicated_checks(tmp_path):
    c = case()
    manifest = {"cases": [c], "population": {}, "face_identity_contract": {"gt_order_preserved": True}}
    (tmp_path / "traces").mkdir()
    t = fitted_trace()
    a = candidate(0, ["BRepCheck_IntersectingWires"], ["BRepCheck_IntersectingWires"])
    t["trim_runs"] = [{"context": "main", "num_faces": 2, "failed_face_indices": [0], "attempts": [a, a]}]
    (tmp_path / "traces/00000001.json").write_text(json.dumps(t))
    rows = enrich_records(manifest, tmp_path)
    summary = summarize_records(rows)
    assert summary["denominator"] == 1
    for threshold in THRESHOLDS:
        for count in summary["observations"][threshold].values():
            assert count["denominator"] == 1
            assert sum(count[key] for key in ("positive", "negative", "unknown")) == 1
        assert sum(row["count"] for row in summary["earliest_observation"][threshold].values()) == 1
    errors = summary["failed_face_context"]["codes"]["BRepCheck_IntersectingWires"]
    assert errors["face_count"] == errors["direct_face_count"] == errors["context_face_count"] == 1
    json.dumps(summary, allow_nan=False)


def test_missing_protocol_refuses_unfinished_results(tmp_path):
    manifest = {"cases": [case()]}
    path = tmp_path / "manifest.json"; path.write_text(json.dumps(manifest))
    with pytest.raises(FileNotFoundError):
        validate_diagnostics(manifest, tmp_path, path)


def test_evidence_highlights_are_derived_not_hardcoded():
    observation = {key: {"positive": value} for key, value in {
        "strict_closed_candidate_failed": 2, "never_closed_admitted": 1,
        "final_success_discarded_wire": 3, "owner_after": 4}.items()}
    summary = {"main": {"denominator": 5, "raw_topology_counts": {"exact": 3},
                         "observations": {"0.006": observation}},
               "cohorts": {"persistent": {"count": 3}, "regressed": {"count": 2},
                           "rescued": {"count": 7, "intersection": {"observations": {"0.006": {"owner_after": {"positive": 6}}}}}},
               "fit_only_vs_optimized_all_accepted": {"intersection": {"fit_failed_optimized_valid": 11, "fit_valid_optimized_failed": 2}}}
    result = evidence_highlights(summary)
    assert result["failure_count"] == 5
    assert result["raw_unordered_pairs_exact"] == 3
    assert result["rescued_owner_gap"] == 6
    assert result["optimized_rescued_from_fit"] == 11


def test_compact_group_preserves_threshold_counts_without_repeated_ids(tmp_path):
    c = case()
    manifest = {"cases": [c], "population": {}, "face_identity_contract": {"gt_order_preserved": True}}
    (tmp_path / "traces").mkdir()
    (tmp_path / "traces/00000001.json").write_text(json.dumps(trace()))
    group = summarize_records(enrich_records(manifest, tmp_path))
    observation = group["observations"]["0.006"]["never_wire"]
    summary = {"created_at_utc": "fixed", "population": {}, "highlights": {}, "main": group,
               "historical_failures": group, "cohorts": {}, "fit_only_vs_optimized_all_accepted": {},
               "optimization_counterfactual": {"cases": []}, "sources": {}, "limitations": [],
               "audit": {name: {"checked_count": 1, "terminal_diagnostic_errors": []} for name in ("intersection", "legacy_controls", "legacy_historical")}}
    summary["audit"]["baseline_reuse"] = {"eligible_count": 1}
    compact = compact_summary(summary)
    assert compact["main"]["observations"]["0.006"]["never_wire"] == [observation[k] for k in ("positive", "negative", "unknown")]
    assert "ids" not in compact["main"]
    assert compact["columns"]["observations_and_cad_counts"] == ["positive", "negative", "unknown"]
    assert set(compact["main"]["observations"]) == set(THRESHOLDS)
    json.dumps(compact, allow_nan=False)
