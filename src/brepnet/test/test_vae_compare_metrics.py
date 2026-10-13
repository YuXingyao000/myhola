"""Behavior tests for raw VAE comparison; these establish no model-quality claim."""

import csv
import json
import sys

import numpy as np
import pytest

from tools.vae_compare_metrics import (
    evaluate_prediction, evaluate_sample, pair_scores, read_frozen_ids, squared_chamfer, summarize,
)


def geometry(face_count=2):
    faces = np.zeros((face_count, 16, 16, 3), dtype=np.float32)
    faces[..., 0] = np.arange(face_count)[:, None, None]
    curve = np.zeros((1, 16, 3), dtype=np.float32)
    curve[0, :, 0] = np.linspace(0, 1, 16)
    return faces, curve, np.array([[0, 1]], dtype=np.int64)


def prediction(method="joint", face_count=2):
    faces, curves, _ = geometry(face_count)
    if method == "legacy":
        curves = np.concatenate((curves, curves[:, ::-1]))
        conn = np.array([[0, 0, 1], [1, 1, 0]], dtype=np.int64)
    else:
        conn = np.array([[0, 0, 1]], dtype=np.int64)
    result = {"pred_face": faces, "pred_edge": curves, "pred_edge_face_connectivity": conn,
              "input_latent_sha256": np.asarray("a" * 64),
              "representation": np.asarray("shared_boundary_v1" if method == "joint" else "legacy_half_edges")}
    if method == "joint":
        result["pred_edge_confidence"] = np.ones(len(curves), dtype=np.float32)
    return result


def evaluate(pred, method="joint", mode="posterior_mean", gt=None):
    return evaluate_prediction(pred, *(geometry() if gt is None else gt), method=method, mode=mode)


def test_chamfer_is_sum_of_mean_squared_directions_and_empty_is_explicit():
    point = np.zeros((1, 3))
    assert squared_chamfer(point, point + [2, 0, 0]) == (4.0, 4.0, 8.0)
    assert squared_chamfer(np.empty((0, 3)), np.empty((0, 3))) == (0.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="exactly one"):
        squared_chamfer(point, np.empty((0, 3)))
    with pytest.raises(ValueError, match="finite"):
        squared_chamfer(point, np.full((1, 3), np.nan))


def test_perfect_reciprocal_legacy_and_shared_joint_are_not_extra_duplicates():
    old, new = evaluate(prediction("legacy"), "legacy"), evaluate(prediction())
    for row in (old, new):
        assert row["status"] == "ok"
        assert row["face_mae"] == row["face_cd_sq"] == row["edge_cd_sq"] == 0
        assert row["topology_f1"] == 1
        assert row["unique_pair_count"] == 1
    assert old["raw_edge_count"] == 2 and new["raw_edge_count"] == 1
    assert old["legacy_directed_duplicate_count"] == 0
    assert old["legacy_reciprocal_pair_count"] == 1
    assert old["legacy_reciprocal_cd_sq_mean"] == 0
    assert new["joint_duplicate_pair_count"] == 0


def test_raw_cd_keeps_all_curves_and_duplicate_queries_remain_visible():
    pred = prediction()
    pred["pred_edge"] = np.repeat(pred["pred_edge"], 3, axis=0)
    pred["pred_edge_face_connectivity"] = np.array([[0, 0, 1], [1, 1, 0], [2, 0, 1]])
    pred["pred_edge_confidence"] = np.ones(3)
    row = evaluate(pred)
    assert row["edge_cd_sq"] == 0  # CD itself cannot count coincident duplicates.
    assert row["joint_duplicate_pair_count"] == row["joint_query_count_error"] == 2
    assert row["topology_f1"] == 1
    pred["pred_edge"][2, :, 2] += 5
    assert evaluate(pred)["edge_pred_to_gt_sq"] == pytest.approx(25 / 3)


def test_wrong_pairs_do_not_receive_topology_credit_from_good_geometry():
    pred = prediction(face_count=3)
    pred["pred_edge_face_connectivity"][0, 2] = 2
    row = evaluate(pred, gt=geometry(3))
    assert row["edge_cd_sq"] == 0
    assert row["topology_precision"] == row["topology_recall"] == row["topology_f1"] == 0


def test_invalid_indices_and_self_assignment_are_reported_without_losing_faces():
    pred = prediction()
    pred["pred_edge"] = np.repeat(pred["pred_edge"], 3, axis=0)
    pred["pred_edge_face_connectivity"] = np.array([[0, 0, 0], [1, 0, 99], [99, 0, 1]])
    pred["pred_edge_confidence"] = np.ones(3)
    row = evaluate(pred)
    assert row["face_mae"] == 0
    assert row["same_face_count"] == row["invalid_edge_index_count"] == row["invalid_face_index_count"] == 1
    assert row["topology_f1"] == 0 and row["unique_pair_count"] == 0
    assert row["internal_topology_status"] == "invalid_indices_or_references"
    assert row["status"] != "ok"


def test_empty_prediction_edges_preserve_face_scores_and_have_no_fake_cd():
    pred = prediction()
    pred.update(pred_edge=np.empty((0, 16, 3)), pred_edge_face_connectivity=np.empty((0, 3), dtype=np.int64),
                pred_edge_confidence=np.empty(0))
    row = evaluate(pred)
    assert row["face_mae"] == row["face_cd_sq"] == 0
    assert row["edge_cd_sq"] is None and row["edge_status"] == "empty_prediction"
    assert row["topology_f1"] == 0 and row["topology_precision"] == 1 and row["topology_recall"] == 0
    empty_gt = (geometry()[0], np.empty((0, 16, 3)), np.empty((0, 2), dtype=np.int64))
    row = evaluate(pred, gt=empty_gt)
    assert row["edge_cd_sq"] == 0 and row["edge_status"] == "empty_both"
    assert row["topology_f1"] == 1 and row["status"] == "ok"
    assert pair_scores({(0, 1)}, set())["topology_f1"] == 0


@pytest.mark.parametrize("corruption", ["nonfinite_curve", "float_connectivity", "bad_confidence"])
def test_partial_failure_does_not_erase_finite_face_geometry(corruption):
    pred = prediction()
    if corruption == "nonfinite_curve":
        pred["pred_edge"][0, 0, 0] = np.nan
    elif corruption == "float_connectivity":
        pred["pred_edge_face_connectivity"] = pred["pred_edge_face_connectivity"].astype(float)
    else:
        pred["pred_edge_confidence"] = np.array([np.nan])
    row = evaluate(pred)
    assert row["face_mae"] == row["face_cd_sq"] == 0
    assert row["status"] != "ok"
    if corruption == "nonfinite_curve":
        assert row["edge_cd_sq"] is None


def test_archived_mode_never_assumes_gt_face_correspondence_even_with_same_count():
    for count in (2, 3):
        row = evaluate(prediction(face_count=count), mode="archived_latents")
        assert np.isfinite(row["face_cd_sq"]) and np.isfinite(row["edge_cd_sq"])
        assert row["face_mae"] is None and row["topology_f1"] is None
        assert row["unique_pair_count_error"] is None and row["joint_query_count_error"] is None
        assert row["unique_pair_count"] == 1
    row = evaluate(prediction(face_count=3))
    assert row["face_mae"] is None and row["topology_f1"] is None
    assert row["topology_status"] == "face_count_mismatch"


def test_legacy_unilateral_and_directed_duplicates_are_not_shared_query_duplicates():
    pred = prediction("legacy")
    pred["pred_edge_face_connectivity"][1] = [1, 0, 1]
    row = evaluate(pred, "legacy")
    assert row["legacy_directed_duplicate_count"] == 1
    assert row["legacy_unilateral_pair_count"] == 1
    assert row["legacy_reciprocal_pair_count"] == 0
    assert row["legacy_reciprocal_cd_sq_mean"] is None
    assert row["joint_duplicate_pair_count"] is None


def write_gt(path):
    faces, curves, _ = geometry()
    path.mkdir(parents=True)
    faces = np.concatenate((faces, np.zeros_like(faces)), axis=-1)
    curves = np.concatenate((curves, np.zeros_like(curves)), axis=-1)
    np.savez(path / "data.npz", sample_points_faces=faces,
             sample_points_lines=np.concatenate((curves, curves[:, ::-1])),
             edge_face_connectivity=np.array([[0, 0, 1], [1, 1, 0]]))


def test_missing_predictions_and_input_mismatch_remain_in_fixed_denominators(tmp_path):
    write_gt(tmp_path / "gt" / "present")
    write_gt(tmp_path / "gt" / "missing")
    for method in ("legacy", "joint"):
        folder = tmp_path / "pred" / method / "present"
        folder.mkdir(parents=True)
        item = prediction(method)
        if method == "joint":
            item["input_latent_sha256"] = np.asarray("b" * 64)
        # Deliberately wrong exported GT must never influence independent labels.
        item["gt_face"] = np.full_like(item["pred_face"], 999)
        np.savez(folder / "data.npz", **item)
    rows = []
    for sample in ("present", "missing"):
        rows.extend(evaluate_sample(sample, data_root=tmp_path / "gt", prediction_root=tmp_path / "pred", mode="posterior_mean"))
    result = summarize(rows, ["present", "missing"], mode="posterior_mean", original_count=3)
    assert not result["protocol_valid"]
    assert result["paired"]["face_mae"]["common_count"] == 0
    coverage = result["methods"]["joint"]["coverage"]["face_geometry_computable"]
    assert coverage == {"count": 1, "intended_fraction": .5, "original_fraction": 1 / 3}
    assert result["methods"]["joint"]["metrics"]["face_mae"]["mean"] == 0


def test_paired_statistics_use_metric_specific_shared_input_intersection():
    rows = []
    for sample in ("a", "b"):
        for method in ("legacy", "joint"):
            row = evaluate(prediction(method), method)
            row.update(model_id=sample, method=method, prediction_file_exists=True, shared_input_status="verified")
            row["face_mae"] = {("a", "legacy"): 2.0, ("a", "joint"): 1.0,
                               ("b", "legacy"): 100.0, ("b", "joint"): None}[sample, method]
            rows.append(row)
    result = summarize(rows, ["a", "b"], mode="posterior_mean", original_count=3)
    paired = result["paired"]["face_mae"]
    assert paired["common_count"] == 1
    assert paired["legacy"]["mean"] == 2 and paired["joint"]["mean"] == 1
    assert paired["joint_minus_legacy"]["mean"] == -1 and paired["joint_win_fraction"] == 1
    assert "raw_edge_count" not in result["paired"]
    assert result["paired_topology_micro"]["joint"]["f1"] == 1


def test_intersection_uses_half_edge_contract_and_named_paired_results(tmp_path):
    write_gt(tmp_path / "gt" / "present")
    write_gt(tmp_path / "gt" / "missing")
    for method in ("legacy", "intersection"):
        folder = tmp_path / "pred" / method / "present"
        folder.mkdir(parents=True)
        np.savez(folder / "data.npz", **prediction("legacy"))
    rows = []
    for sample in ("present", "missing"):
        rows.extend(evaluate_sample(sample, data_root=tmp_path / "gt", prediction_root=tmp_path / "pred",
                                    mode="posterior_mean", candidate_method="intersection"))
    candidate = next(row for row in rows if row["method"] == "intersection" and row["model_id"] == "present")
    assert candidate["status"] == "ok"
    assert candidate["legacy_reciprocal_pair_count"] == 1
    assert candidate["legacy_directed_duplicate_count"] == 0
    assert candidate["joint_duplicate_pair_count"] is None
    assert candidate["joint_query_count_error"] is None
    assert "confidence_status" not in candidate
    result = summarize(rows, ["present", "missing"], mode="posterior_mean", original_count=3,
                       candidate_method="intersection")
    paired = result["paired"]["face_cd_sq"]
    assert paired["common_count"] == 1 and paired["intersection_minus_legacy"]["mean"] == 0
    assert paired["intersection_win_fraction"] == 0 and "joint" not in result["methods"]
    assert result["methods"]["intersection"]["coverage"]["face_geometry_computable"]["original_fraction"] == 1 / 3
    assert evaluate_prediction(prediction(), *geometry(), method="intersection", mode="posterior_mean")["prediction_status"] == "invalid_representation"


def test_cli_rejects_nonfrozen_split_instead_of_claiming_2417_coverage(tmp_path):
    path = tmp_path / "ids.txt"
    path.write_text("one\ntwo\n")
    with pytest.raises(ValueError, match="2417-ID"):
        read_frozen_ids(path)
    path.write_text("one\none\n")
    with pytest.raises(ValueError, match="unique"):
        read_frozen_ids(path)


def test_local_comparison_preserves_intersection_identity_and_pair_direction(tmp_path):
    write_gt(tmp_path / "gt" / "example")
    for method in ("intersection", "local"):
        folder = tmp_path / "pred" / method / "example"
        folder.mkdir(parents=True)
        value = prediction("legacy")
        if method == "local":
            value["pred_face"] += 0.1
        np.savez(folder / "data.npz", **value)
    rows = evaluate_sample("example", data_root=tmp_path / "gt", prediction_root=tmp_path / "pred",
                           mode="posterior_mean", baseline_method="intersection", candidate_method="local")
    result = summarize(rows, ["example"], mode="posterior_mean", original_count=2,
                       baseline_method="intersection", candidate_method="local")
    assert set(result["methods"]) == {"intersection", "local"}
    assert result["baseline_method"] == "intersection"
    assert result["paired"]["face_mae"]["local_minus_intersection"]["mean"] == pytest.approx(.1)
    assert result["paired"]["face_mae"]["local_win_fraction"] == 0
    assert all(row["legacy_reciprocal_pair_count"] == 1 for row in rows)
    assert "legacy" not in result["methods"]


@pytest.mark.parametrize("mode", ["posterior_mean", "archived_latents"])
@pytest.mark.parametrize("candidate_method", ["joint", "intersection"])
def test_complete_cli_writes_json_summary_and_matching_per_object_csv(tmp_path, monkeypatch, mode, candidate_method):
    """Exercise NPZ -> scalar metrics -> coverage counts -> actual JSON output."""
    from tools import vae_compare_metrics as metrics

    sample = "present"
    write_gt(tmp_path / "gt" / sample)
    split = tmp_path / "ids.txt"
    split.write_text(sample + "\n")
    comparison = tmp_path / "comparison"
    for method in ("legacy", candidate_method):
        folder = comparison / "raw_predictions" / mode / method / sample
        folder.mkdir(parents=True)
        np.savez(folder / "data.npz", **prediction("legacy" if method == "intersection" else method))
    # The frozen-list validation has its own test; this checks the real CLI outputs.
    monkeypatch.setattr(metrics, "read_frozen_ids", lambda path: [sample])
    monkeypatch.setattr(sys, "argv", ["vae_compare_metrics", "--data-root", str(tmp_path / "gt"),
                                     "--split-list", str(split), "--comparison-root", str(comparison),
                                     "--mode", mode, "--workers", "1", "--candidate-method", candidate_method])
    assert metrics.main() == 0
    output = comparison / "raw_metrics" / mode
    summary = json.loads((output / "summary.json").read_text())
    with (output / "per_object.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    for method in ("legacy", candidate_method):
        coverage = summary["methods"][method]["coverage"]
        assert type(coverage["face_geometry_computable"]["count"]) is int
        assert coverage["face_geometry_computable"]["count"] == 1
        assert coverage["edge_geometry_computable"]["count"] == 1
        assert coverage["aligned_face_mae_computable"]["count"] == (mode == "posterior_mean")
        row = next(row for row in rows if row["method"] == method)
        assert summary["methods"][method]["metrics"]["face_cd_sq"]["mean"] == float(row["face_cd_sq"])
    assert summary["paired"]["face_cd_sq"]["common_count"] == 1
