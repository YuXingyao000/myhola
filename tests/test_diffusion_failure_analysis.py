"""Failure evidence stays distinct from validity and GT reconstruction error."""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.eval.failure_analysis import trace_outcome
from src.eval.failure_geometry import measure_raw_consistency, point_triangle_distances


class TraceOutcomeTests(unittest.TestCase):
    def test_absent_or_running_evaluation_is_pending(self):
        self.assertEqual(trace_outcome({}, None)["primary_stage"], "pending")
        self.assertEqual(trace_outcome({"status": "ok"}, {"status": "running"})["primary_stage"], "pending")

    def test_generation_error_and_timeout_stay_distinct(self):
        self.assertEqual(trace_outcome({"status": "generation_error"}, None)["primary_stage"], "generation_failure")
        result = trace_outcome({"status": "ok"}, {"status": "timeout", "validity": "unknown"})
        self.assertEqual(result["primary_stage"], "budget_exhausted")
        self.assertEqual(result["native_validity"], "unknown")

    def test_preference_input_error_remains_raw_stage_failure(self):
        # score_raw or raw preparation can fail before OCC is invoked.
        result = trace_outcome({"status": "ok"}, {
            "status": "input_error", "stage": "raw", "validity": "unknown",
            "error": "ValueError: Nonfinite raw geometry",
        })
        self.assertEqual(result["primary_stage"], "raw_input_failure")
        self.assertEqual(result["stage_evidence"]["status"], "input_error")
        self.assertEqual(result["native_validity"], "unknown")

    def test_validity_survives_geometry_error(self):
        result = trace_outcome({"status": "ok"}, {"status": "evaluation_error", "stage": "post_metrics", "validity": "valid"})
        self.assertEqual(result["primary_stage"], "valid_solid")
        self.assertTrue(result["observation_flags"]["geometry_metric_error"])
        self.assertTrue(result["observation_flags"]["evaluation_error_after_validity"])

    def test_last_path_does_not_hide_further_earlier_progress(self):
        # Mirrors baseline/00098632: fit-only reached solid, the last path did not.
        diagnostic = {"status": "no_accepted_solid", "paths": [
            {"label": "roundtrip_fit_only", "stage": "solid_candidates", "status": "no_accepted_solid",
             "solid_attempts": [{"status": "no_solid", "tolerance": .01}]},
            {"label": "fit_score_v2_roundtrip_optimized", "stage": "trimming", "status": "incomplete_faces",
             "input_face_count": 18, "used_face_count": 16, "faces": [
                 {"face_index": 2, "accepted_attempt": None, "attempts": [
                     {"status": "trim_rejected", "skipped_wires": ["open_wire"]}]},
             ]},
        ]}
        result = trace_outcome({"status": "ok"}, {"status": "ok", "validity": "invalid"}, diagnostic)
        self.assertEqual(result["primary_stage"], "incomplete_trimmed_faces")
        self.assertEqual(result["furthest_recovery_stage"], "solid_candidates")
        self.assertEqual(result["path_summaries"][-1]["failed_face_indices"], [2])
        self.assertTrue(result["observation_flags"]["open_wire_rejected"])
        self.assertEqual(result["unavailable_diagnostics"]["fitting_residuals"], "not_recorded")

    def test_solid_construction_and_native_rejection_are_distinct(self):
        path = {"label": "last", "status": "no_accepted_solid", "stage": "solid_candidates",
                "solid_attempts": [{"status": "no_solid"}]}
        record = {"status": "ok", "validity": "invalid", "post_diagnostics": {"paths": [path]}}
        self.assertEqual(trace_outcome({"status": "ok"}, record)["primary_stage"], "no_solid")
        # Mirrors dpo_5000/00059708: a saved candidate read back as a shell.
        path["solid_attempts"].append({"status": "native_rejected", "shape_type": 3})
        result = trace_outcome({"status": "ok"}, record)
        self.assertEqual(result["primary_stage"], "native_readback_or_boundary_rejected")
        self.assertEqual(result["furthest_recovery_stage"], "native_readback")

    def test_missing_trace_does_not_mean_clean_or_trim_failure(self):
        result = trace_outcome({"status": "ok"}, {"status": "ok", "validity": "invalid"})
        self.assertEqual(result["primary_stage"], "unresolved")
        self.assertIsNone(result["observation_flags"]["trim_rejected"])

    def test_interrupted_face_trials_are_not_final_trim_failures(self):
        diagnostic = {"status": "budget_exhausted", "paths": [
            {"status": "budget_exhausted", "stage": "trimming", "faces": [
                {"face_index": 3, "accepted_attempt": None, "attempts": [{"status": "running"}]}]}]}
        result = trace_outcome({"status": "ok"}, {"status": "timeout"}, diagnostic)
        self.assertEqual(result["path_summaries"][0]["failed_face_indices"], [])
        self.assertEqual(result["path_summaries"][0]["unaccepted_face_indices"], [3])


class RawConsistencyTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="hola-failure-geometry-")
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "data.npz"
        u, v = np.meshgrid(np.linspace(0, 1, 3), np.linspace(0, 1, 3), indexing="ij")
        self.faces = np.stack([np.stack((u, v, np.zeros_like(u)), -1),
                               np.stack((u, np.zeros_like(u), v), -1)])
        curve = np.stack((np.linspace(0, 1, 16), np.zeros(16), np.zeros(16)), -1)
        self.curves = np.stack((curve, curve[::-1]))
        self.conn = np.array([[0, 0, 1], [1, 1, 0]], dtype=np.int64)

    def measure(self, faces=None, curves=None, conn=None):
        np.savez(self.path, pred_face=self.faces if faces is None else faces,
                 pred_edge=self.curves if curves is None else curves,
                 pred_edge_face_connectivity=self.conn if conn is None else conn)
        return measure_raw_consistency(self.path)

    def test_reversed_twin_is_geometrically_equal_and_on_both_owners(self):
        result = self.measure()
        self.assertAlmostEqual(result["reciprocal_curve_distance"]["max_distance"], 0)
        self.assertAlmostEqual(result["owner_mesh_distance"]["max_distance"], 0)
        self.assertFalse(result["observation_flags"]["reciprocal_curve_gap"])
        self.assertEqual(result["structure"]["connected_component_count"], 1)

    def test_shifted_twin_and_owner_gap_are_detected_without_gt(self):
        curves = self.curves.copy()
        curves[1, :, 2] += .05
        result = self.measure(curves=curves)
        self.assertAlmostEqual(result["reciprocal_curve_distance"]["p95_per_pair"]["max"], .05)
        self.assertEqual(result["owner_mesh_distance"]["above_threshold_count"], 1)
        self.assertTrue(result["observation_flags"]["reciprocal_curve_gap"])

    def test_isolated_face_duplicates_self_pairs_and_invalid_rows(self):
        faces = np.concatenate((self.faces, self.faces[:1] + 3))
        conn = np.array([[0, 0, 1], [1, 0, 1], [0, 0, 0], [0, 8, 0]])
        result = self.measure(faces=faces, conn=conn)
        structure = result["structure"]
        self.assertEqual(structure["one_direction_pairs"], [[0, 1]])
        self.assertEqual(structure["duplicate_directed_count"], 1)
        self.assertEqual(structure["self_connection_rows"], [2])
        self.assertEqual(structure["invalid_connectivity_rows"], [3])
        self.assertEqual(structure["isolated_face_indices"], [2])
        self.assertEqual(structure["connected_component_count"], 2)
        self.assertIsNone(result["observation_flags"]["reciprocal_curve_gap"])

    def test_unavailable_edges_are_not_zero_distance(self):
        result = self.measure(curves=np.empty((0, 16, 3)), conn=np.empty((0, 3), dtype=int))
        self.assertIsNone(result["owner_mesh_distance"]["max_distance"])
        self.assertIsNone(result["observation_flags"]["edge_owner_mesh_gap"])
        self.assertEqual(result["structure"]["connected_component_count"], 2)

    def test_legacy_integral_float_connectivity_preserves_ids(self):
        result = self.measure(conn=self.conn.astype(np.float32))
        self.assertEqual(result["reciprocal_curve_distance"]["records"][0]["edge_ids"], [0, 1])
        fractional = self.conn.astype(np.float32)
        fractional[0, 1] = .2
        with self.assertRaisesRegex(ValueError, "integral connectivity"):
            self.measure(conn=fractional)

    def test_degenerate_triangle_and_finite_surface_boundaries(self):
        tris = np.array([[[0., 0, 0], [1, 0, 0], [0, 1, 0]]])
        np.testing.assert_allclose(point_triangle_distances(np.array([[.2, .2, .3], [2., 0, 0]]), tris), [.3, 1])
        degenerate = np.array([[[0., 0, 0], [1, 0, 0], [1, 0, 0]]])
        np.testing.assert_allclose(point_triangle_distances(np.array([[.5, .2, 0]]), degenerate), [.2])

    def test_wrong_representation_and_nonfinite_data_are_not_silently_measured(self):
        curves = self.curves.copy()
        curves[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            self.measure(curves=curves)
        np.savez(self.path, representation="shared_boundary_v1")
        with self.assertRaisesRegex(ValueError, "legacy_half_edges"):
            measure_raw_consistency(self.path)


if __name__ == "__main__":
    unittest.main()
