"""Append-only independent review of the fixed two-object feedback probe.

No model or production metric helpers are imported and no inference is rerun.
"""
from __future__ import annotations

import ast
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DIAG = ROOT / "diagnostics"
CASES = {"00335203": (0, 6), "00326127": (3, 11)}


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def close(actual, expected, label, *, tolerance=1e-15):
    assert math.isclose(float(actual), float(expected), rel_tol=1e-12, abs_tol=tolerance), (label, actual, expected)


def cd_sq(left, right):
    matrix = np.sum((left.astype(np.float64)[:, None] - right.astype(np.float64)[None]) ** 2, axis=-1)
    return math.fsum(matrix.min(0)) / len(right) + math.fsum(matrix.min(1)) / len(left)


def check_delta(delta, report):
    values = sorted(np.linalg.norm(delta.astype(np.float64), axis=-1).reshape(-1).tolist())
    assert report["point_count"] == len(values)
    position = .9 * (len(values) - 1)
    low, high = math.floor(position), math.ceil(position)
    actual = {"mean": math.fsum(values) / len(values), "median": statistics.median(values),
              "p90": values[low] + (values[high] - values[low]) * (position - low), "max": values[-1]}
    for name, value in actual.items():
        close(value, report[name], "displacement:" + name)
    return actual


def owner_distance(points, surface):
    """Independent triangle distance via plane projection and oriented side tests."""
    surface = surface.astype(np.float64)
    a, b = surface[:-1, :-1].reshape(-1, 3), surface[1:, :-1].reshape(-1, 3)
    c, d = surface[:-1, 1:].reshape(-1, 3), surface[1:, 1:].reshape(-1, 3)
    triangles = np.concatenate((np.stack((a, b, d), 1), np.stack((a, d, c), 1)))
    a, b, c = (triangles[:, index] for index in range(3))
    p = points.astype(np.float64)[:, None]
    segment_distances = []
    for start, end in ((a, b), (b, c), (c, a)):
        v = end - start
        length_sq = np.sum(v * v, -1)
        fraction = np.sum((p - start) * v, -1) / np.where(length_sq > 0, length_sq, 1)
        nearest = start + np.clip(fraction, 0, 1)[..., None] * v
        segment_distances.append(np.sum((p - nearest) ** 2, -1))
    normal = np.cross(b - a, c - a)
    normal_sq = np.sum(normal * normal, -1)
    signed = np.sum((p - a) * normal, -1)
    projection = p - (signed / np.where(normal_sq > 0, normal_sq, 1))[..., None] * normal
    inside = np.broadcast_to(normal_sq > 0, signed.shape).copy()
    for start, end in ((a, b), (b, c), (c, a)):
        inside &= np.sum(np.cross(end - start, projection - start) * normal, -1) >= 0
    plane_sq = signed ** 2 / np.where(normal_sq > 0, normal_sq, 1)
    closest_sq = np.minimum.reduce(segment_distances + [np.where(inside, plane_sq, np.inf)])
    return np.sqrt(closest_sq.min(1))


def main():
    probe_path = DIAG / "feedback_branch_probe.json"
    probe, protocol = read(probe_path), read(ROOT / "metadata/inference/posterior_mean/protocol.json")
    main_review = read(ROOT / "independent_review.json")
    assert main_review["status"] == "passed" and not main_review["pending"]
    assert main_review["review_builder_sha256"] == digest(ROOT / "build_independent_review.py")
    assert probe["status"] == "passed" and probe["fixed_cases"] == {key: list(pair) for key, pair in CASES.items()}
    assert probe["checkpoint_sha256"] == protocol["checkpoint_sha256"] == main_review["checkpoint_sha256"]
    assert digest(probe["checkpoint"]) == probe["checkpoint_sha256"]
    assert (probe["model_epoch"], probe["global_step"]) == (34, 12705)
    for key in ("dtype", "autocast", "seed", "matmul_precision", "matmul_allow_tf32",
                "cudnn_allow_tf32", "cudnn_benchmark", "cudnn_deterministic", "deterministic_algorithms"):
        assert probe["numeric_settings"][key] == protocol["numeric_settings"][key]
    for path, expected in probe["source_sha256"].items():
        assert digest(path) == expected, path
    initial = read(DIAG / "feedback_branch_initial_check.json")
    assert initial["status"] == "stopped_on_exact_invariant_check"
    assert initial["mismatched_elements"] == 7 and initial["compared_elements"] == 288
    assert initial["max_absolute_difference"] == 2.384185791015625e-7
    assert initial["field_name"] is None
    tree = ast.parse((DIAG / "feedback_branch_probe.py").read_text())
    decode_calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute) and node.func.attr == "decode_state"]
    assert len(decode_calls) == 1
    # Static inspection additionally establishes that the stored pre-branch
    # geometry is captured by the pre-hook, rather than by a second decode.
    samples = {}
    for sid, pair in CASES.items():
        record = probe["samples"][sid]
        assert digest(record["arrays_path"]) == record["arrays_sha256"]
        with np.load(record["arrays_path"], allow_pickle=False) as saved:
            arrays = {key: saved[key] for key in saved.files}
        with np.load(ROOT / "raw_predictions/posterior_mean/point_feedback" / sid / "data.npz", allow_pickle=False) as raw:
            raw = {key: raw[key] for key in raw.files}
        with np.load(Path(protocol["reference_root"]) / "raw_predictions/posterior_mean/legacy" / sid / "data.npz", allow_pickle=False) as source:
            np.testing.assert_array_equal(arrays["latent"], source["face_latents"])
        np.testing.assert_array_equal(arrays["latent"], raw["face_latents"])
        assert record["input_latent_sha256"] == str(raw["input_latent_sha256"])
        assert record["exact_unchanged_tensor_checks"] == {"tensor_count": 22, "exact": True}
        assert record["captured_prefix_matches_final_state"] == {"tensor_count": 6, "exact": True}
        assert all(item == {"tensor_count": 3, "exact": True} for item in record["shared_prefix_inputs_not_mutated"].values())
        assert record["direction_channels_exact_unchanged"]
        for value in arrays.values():
            assert np.isfinite(value).all()
        before, after = arrays["curve_before"], arrays["curve_after"]
        delta = arrays["global_xyz_delta"]
        np.testing.assert_array_equal(after - before, delta)
        np.testing.assert_array_equal(arrays["local_curve_before"][..., 3:], arrays["local_curve_after"][..., 3:])
        pairs, probability, active = arrays["pairs"], arrays["probability"], arrays["active"]
        np.testing.assert_array_equal(active, probability > .5)
        np.testing.assert_array_equal(pairs, raw["all_candidate_pairs"])
        np.testing.assert_array_equal(pairs[active], raw["pred_edge_face_connectivity"][:, 1:])
        assert record["candidate_count"] == len(pairs) and record["active_halfedge_count"] == int(active.sum())
        for key, values in (("pred_face", arrays["face_points"]), ("pred_edge", after[active]), ("all_candidate_prob", probability)):
            error = float(np.abs(values.astype(np.float64) - raw[key].astype(np.float64)).max())
            close(error, record["raw_export_replay"][key]["max_abs_error"], "raw replay " + key)
            assert error <= 1e-5
        for key in ("pred_edge_face_connectivity", "pred_face_adj", "all_candidate_pairs"):
            assert record["raw_export_replay"][key] == {"exact": True, "max_abs_error": 0.0}
        bbox = arrays["curve_center_scale"].astype(np.float64)
        contract_errors = {}
        for stage in ("before", "after"):
            reconstructed = arrays["local_curve_" + stage][..., :3].astype(np.float64) * bbox[:, None, 3:] + bbox[:, None, :3]
            points = arrays["curve_" + stage].astype(np.float64)
            error = float(np.max(np.abs(reconstructed - points)))
            assert error <= 8 * np.finfo(np.float32).eps * max(1., float(np.max(np.abs(points))))
            contract_errors[stage] = error
        check_delta(delta, record["all_candidate_delta"])
        check_delta(delta[active], record["active_candidate_delta"])
        with np.load(Path(protocol["data_root"]) / sid / "data.npz", allow_pickle=False) as gt:
            matching = [int(edge) for edge, face_a, face_b in gt["edge_face_connectivity"] if tuple(sorted((face_a, face_b))) == pair]
            assert len(matching) == 2 and min(matching) == record["gt_source_edge_id"]
            target = gt["sample_points_lines"][min(matching), :, :3].astype(np.float32)
        rows = [i for i, owners in enumerate(pairs) if tuple(sorted(owners)) == pair]
        assert len(rows) == 2 and record["pair_is_exported_before_and_after"] == bool(active[rows].all())
        curves = []
        for curve, row in zip(record["curves"], rows):
            assert curve["candidate_row"] == row and curve["directed_pair"] == pairs[row].tolist()
            active_rows = np.flatnonzero(active).tolist()
            assert curve["raw_edge_id"] == (active_rows.index(row) if active[row] else None)
            close(probability[row], curve["probability_before_and_after"], "probability")
            assert curve["active_before_and_after"] == bool(active[row])
            check_delta(delta[row], curve["delta_norm"])
            for stage in ("before", "after"):
                values = arrays["curve_" + stage][row]
                np.testing.assert_array_equal(values.mean(0), curve["centroid_" + stage])
                close(cd_sq(values, target), curve["own_gt_cd_sq_" + stage], "own GT CD")
                for owner in pair:
                    distances = owner_distance(values, arrays["face_points"][owner])
                    expected = curve["owner_distances_" + stage][str(owner)]
                    assert len(distances) == expected["samples"] == 16
                    close(math.fsum(distances) / len(distances), expected["mean"], "owner mean", tolerance=1e-12)
                    close(max(distances), expected["max"], "owner max", tolerance=1e-12)
            curves.append({key: curve[key] for key in ("directed_pair", "raw_edge_id", "probability_before_and_after", "own_gt_cd_sq_before", "own_gt_cd_sq_after", "delta_norm")})
        for stage in ("before", "after"):
            close(cd_sq(arrays["curve_" + stage][rows[0]], arrays["curve_" + stage][rows[1]]),
                  record["selected_pair_twin_cd_sq_" + stage], "twin CD")
        samples[sid] = {"candidate_count": len(pairs), "all_displacement_statistics_recomputed": True,
                        "gt_and_twin_cd_recomputed": True, "owner_distances_independently_recomputed": True,
                        "local_global_contract_max_abs_error": contract_errors, "curves": curves,
                        "arrays_sha256": record["arrays_sha256"]}
    output = {"status": "passed", "created_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "Additional fixed two-case probe review; completed primary evaluation review remains unchanged.",
              "reviewer_script_sha256": digest(__file__), "probe_sha256": digest(probe_path),
              "main_review_sha256": digest(ROOT / "independent_review.json"),
              "checkpoint_sha256": probe["checkpoint_sha256"], "verified_source_hashes": len(probe["source_sha256"]),
              "fixed_inputs_and_threshold_selection_verified": True,
              "hook_implementation_and_exact_assertion_records_reviewed": True,
              "single_decode_call_in_probe_source": True,
              "initial_two_pass_stop_preserved": initial,
              "samples": samples,
              "limits": ["No second model execution: exact upstream hook invariants are verified by source inspection and recorded assertions; stored arrays independently validate coordinates and metrics.",
                         "Owner distances are independently recomputed with NumPy against the same 450 triangles, not an analytic CAD surface.",
                         "Final branch input/output only; does not isolate effects of training the branch on upstream weights.",
                         "Initial failed two-pass invariant did not identify the failing field; its reported numerical discrepancy cannot be attributed to a named tensor."]}
    target = ROOT / "validation/feedback_branch_review.json"
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": output["status"], "verified_sources": output["verified_source_hashes"],
                      "samples": samples}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
