import copy
import json

from tools.legacy_failure_census_report import (
    aggregate, cheap_flags, fit_observations, local_failed_faces, loss_down_gt_up, persistent_trim_observations,
    raw_curve_decomposition, saved_step_observations, tally, terminal_stage,
)


def verified_trace():
    return {"sample_id": "00000001", "status": "ok", "instrumentation_errors": [],
            "comparison": {"same_recorded_outcome": True},
            "step_data_comparison": {"equal": True}, "replay": {"native_valid_solid": False}}


def wire_attempt(face, returned=False, *, admitted=False, skipped=False, wire=True):
    return {"face_index": face, "wire_returned": wire, "trim_returned": returned,
            "trim_detail": {"outcome": "success" if returned else "invalid_face" if admitted else "no_admitted_wires",
                "wire_events": [{"added": admitted, "repaired_closed": admitted,
                                 "skip_reason": "not_closed" if skipped else None}]}}


def case():
    return {"id": "00000001", "gt_face_count": 2,
            "raw_legacy": {"pred_face_count": 2, "gt_face_count": 2, "topology_fp": 0, "topology_fn": 0},
            "post_legacy": {"shape_type": 3, "native_valid_solid": False},
            "fit_legacy": {"native_valid_solid": False}}


def test_unknown_preserves_fixed_denominator_and_ids():
    result = tally([("1", True), ("2", False), ("3", None), ("4", "False")])
    assert result["denominator"] == 4
    assert result["positive_fraction_of_intended"] == .25
    assert result["ids"]["unknown"] == ["3", "4"]


def test_native_readback_is_separate_from_legacy_save_rejection():
    trace = verified_trace()
    trace["save_attempts"] = [{"legacy_saved_step_check": True}]
    assert terminal_stage(trace) == "native_readback_invalid"
    trace["save_attempts"][0]["legacy_saved_step_check"] = False
    assert terminal_stage(trace) == "saved_step_rejected"
    trace["step_data_comparison"]["equal"] = False
    assert terminal_stage(trace) == "unknown"


def test_no_solid_and_trim_gate_are_different_branches():
    trace = verified_trace()
    trace["trim_gate"] = {"solid_attempt_allowed": False}
    assert terminal_stage(trace) == "trim_gate"
    trace["trim_gate"]["solid_attempt_allowed"] = True
    trace["solid_attempts"] = [{"result": {"present": False}}]
    assert terminal_stage(trace) == "solid_construction"
    trace["solid_attempts"][0]["result"] = {}
    assert terminal_stage(trace) == "unresolved"


def test_failed_faces_are_joined_using_decoder_ids_only():
    trace = verified_trace()
    trace["trim_runs"] = [{"context": "main", "num_faces": 2, "failed_face_indices": [0, 1]}]
    upstream = {"structure": {"raw": {"status": "ok", "fp_pairs": [[0, 2]], "fn_pairs": []}}}
    result = local_failed_faces(case(), trace, upstream, {"gt_order_preserved": True})
    assert all(row["category"] == "unknown" for row in result["failed_faces"])
    upstream["structure"]["raw"]["fp_pairs"] = []
    result = local_failed_faces(case(), trace, upstream, {"gt_order_preserved": True})
    assert all(row["category"] == "raw_local_pair_exact" for row in result["failed_faces"])
    trace["trim_runs"][0]["num_faces"] = 3
    assert local_failed_faces(case(), trace, upstream, {"gt_order_preserved": True})["failed_faces"] is None


def test_failed_face_pair_cooccurrence_can_have_both_categories_in_one_object():
    source = case()
    source["raw_legacy"]["pred_face_count"] = 3
    trace = verified_trace()
    trace["trim_runs"] = [{"context": "main", "num_faces": 3, "failed_face_indices": [0, 2]}]
    upstream = {"structure": {"raw": {"status": "ok", "fp_pairs": [[0, 1]], "fn_pairs": []}}}
    rows = local_failed_faces(source, trace, upstream, {"gt_order_preserved": True})["failed_faces"]
    assert [row["category"] for row in rows] == ["raw_wrong_pair_incident", "raw_local_pair_exact"]


def test_ever_discard_does_not_mean_final_returned_face_discarded():
    trace = verified_trace()
    trace["trim_runs"] = [{"context": "main", "num_faces": 1, "failed_face_indices": [],
                          "attempts": [wire_attempt(0, skipped=True), wire_attempt(0, returned=True, admitted=True)]}]
    result = persistent_trim_observations(trace)
    assert result["flags"]["final_success_discarded_wire"] is False
    assert result["failed_faces"] == []
    trace["trim_runs"][0]["attempts"][1]["trim_detail"]["wire_events"].append(
        {"added": False, "repaired_closed": False, "skip_reason": "not_closed"})
    result = persistent_trim_observations(trace)
    assert result["flags"]["final_success_discarded_wire"] is True
    assert result["successful_faces_discarding_wire"] == [0]


def test_persistent_failed_face_categories_and_partial_information():
    trace = verified_trace()
    trace["trim_runs"] = [{"context": "main", "num_faces": 4, "failed_face_indices": [0, 1, 2, 3],
        "attempts": [wire_attempt(0, wire=False), wire_attempt(1, skipped=True),
                     wire_attempt(2, admitted=True), {"face_index": 3, "wire_returned": True}]}]
    result = persistent_trim_observations(trace)
    assert [row["category"] for row in result["failed_faces"]] == [
        "never_wire", "never_closed_admitted", "closed_admitted_still_failed", "unknown"]
    assert result["flags"]["strict_closed_candidate_failed"] is True
    assert result["status"] == "partial"


def test_normal_reciprocal_is_additive_representation_not_duplicate_error():
    source = case()
    source["raw_legacy"].update(raw_edge_count=7, legacy_reciprocal_pair_count=3, legacy_unilateral_pair_count=1,
        legacy_directed_duplicate_count=0, same_face_count=0, invalid_face_index_count=0,
        invalid_edge_index_count=0, repeated_edge_reference_count=0)
    result = raw_curve_decomposition([source])
    assert result["counts"] == {"raw_curve_rows": 7, "first_rows_of_reciprocal_pairs": 3,
                                "second_rows_of_reciprocal_pairs": 3, "unilateral_rows": 1}
    source["raw_legacy"]["legacy_directed_duplicate_count"] = 1
    assert raw_curve_decomposition([source])["unresolved_objects"] == 1


def test_incomplete_saved_step_sampling_zero_is_unknown_but_positive_is_evidence():
    data = {"status": "partial", "sampling_errors": ["one edge"], "free_edge_proximity": {"thresholds": {
        "0.002": {"pair_count": 0, "involved_edge_count": 0}, "0.006": {"pair_count": 1, "involved_edge_count": 2}}}}
    result = saved_step_observations(data)
    assert result["thresholds"]["0.002"]["has_near_free_edges"] is None
    assert result["thresholds"]["0.006"]["has_near_free_edges"] is True
    data.update(status="ok", sampling_errors=[])
    assert saved_step_observations(data)["thresholds"]["0.002"]["has_near_free_edges"] is False


def test_cheap_metrics_unknown_is_not_negative():
    assert cheap_flags({})["any_pair_error"] is None
    assert cheap_flags({"topology_fp": 1})["any_pair_error"] is True
    assert cheap_flags({"topology_fp": 0, "topology_fn": 0})["any_pair_error"] is False


def test_missing_sources_stay_in_summary_and_json_serializes(tmp_path):
    manifest = {"cases": [case()], "population": {"native_failed": 1}, "face_identity_contract": {"gt_order_preserved": True}}
    rows, summary = aggregate(manifest, tmp_path)
    assert len(rows) == 1
    assert summary["terminal_stages"]["unknown"]["count"] == 1
    assert summary["failed_face_cooccurrence"]["uncountable_cad_count"] == 1
    assert summary["upstream_mechanisms"]["raw_extra_gt_pair"]["unknown"] == 1
    json.dumps({"cases": rows, "summary": summary}, allow_nan=False)


def test_stale_upstream_trace_invalidates_stage_only(tmp_path):
    manifest = {"cases": [case()], "population": {"native_failed": 1}, "face_identity_contract": {"gt_order_preserved": True}}
    (tmp_path / "traces").mkdir()
    (tmp_path / "upstream").mkdir()
    (tmp_path / "traces/00000001.json").write_text(json.dumps(verified_trace()))
    upstream = {"sample_id": "00000001", "sources": {"trace": {"sha256": "stale"}},
                "mechanism_flags": {"raw_extra_gt_pair": True, "selection_removed_gt_pair": True},
                "stage_deltas": {"face_xyz_mae_change": .1}}
    (tmp_path / "upstream/00000001.json").write_text(json.dumps(upstream))
    rows, _ = aggregate(manifest, tmp_path)
    assert rows[0]["upstream_mechanisms"]["raw_extra_gt_pair"] is True
    assert rows[0]["upstream_mechanisms"]["selection_removed_gt_pair"] is None
    assert rows[0]["optimization_delta"]["face_xyz_mae"] is None


def test_partial_fit_projection_is_unknown_unless_observed_excess():
    trace = verified_trace()
    trace["fitting"] = {"faces": [{}], "num_curves": 1}
    trace["fit_residuals"] = {"status": "partial", "faces": [{"face_index": 0, "count": 16,
        "finite_count": 15, "status": "partial", "max": .003, "mean": .001, "rms": .002}], "curves": [], "curve_owner": []}
    result = fit_observations(trace)
    assert result["faces"]["threshold_flags"]["0.002"] is True
    assert result["faces"]["threshold_flags"]["0.006"] is None
    assert result["curves"]["threshold_flags"]["0.002"] is None
    assert result["faces"]["observed_max"] == .003


def test_loss_down_gt_up_requires_returned_optimized_coordinates():
    row = {"optimization_loss": {"returned_original": False, "terms": {"loss": {"delta": -.1}}},
           "optimization_delta": {"face_xyz_mae": .001}}
    assert loss_down_gt_up(row, "loss", "face_xyz_mae") is True
    row["optimization_loss"]["returned_original"] = True
    assert loss_down_gt_up(row, "loss", "face_xyz_mae") is None
