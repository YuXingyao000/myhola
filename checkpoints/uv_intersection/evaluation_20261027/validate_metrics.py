"""Analytic sanity checks; run directly from the repository root."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from metrics import measure


def fixture() -> dict:
    line = np.zeros((16, 3), dtype=np.float64)
    line[:, 0] = np.linspace(0, 1, 16)
    return {
        "representation": np.asarray("dual_uv_surface_v1"),
        "pred_face": np.zeros((2, 16, 16, 3)),
        "gt_face": np.zeros((2, 16, 16, 3)),
        "pred_edge_xyz_by_face": np.tile(line, (1, 2, 1, 1)),
        "diagnostic_gt_pair_xyz": np.tile(line, (1, 2, 1, 1)),
        "gt_edge": line[None],
        "pred_edge_owner_pairs": np.array([[0, 1]], dtype=np.int64),
        "gt_edge_owner_pairs": np.array([[0, 1]], dtype=np.int64),
    }


class FrozenMetricChecks(unittest.TestCase):
    def test_identity_is_zero_and_json_serializable(self):
        result = measure(fixture())
        for name in ("face_cd_sq", "edge_cd_sq", "face_mae", "gt_pair_curve_xyz_mse",
                     "active_gap_mean", "gt_pair_gap_rms", "active_pair_cd_sq_mean"):
            self.assertEqual(result[name], 0.0)
        self.assertEqual(result["topology_f1"], 1.0)
        self.assertEqual(result["active_gap_count"], 16)
        self.assertEqual(result["raw_edge_count"], 2)
        json.dumps(result, allow_nan=False)

    def test_known_perpendicular_translation(self):
        arrays = fixture()
        delta = np.array([0.0, 0.3, 0.4])
        for name in ("pred_face", "pred_edge_xyz_by_face", "diagnostic_gt_pair_xyz"):
            arrays[name] += delta
        result = measure(arrays)
        self.assertAlmostEqual(result["face_mae"], 0.7 / 3)
        for name in ("face_cd_sq", "edge_cd_sq", "edge_side0_cd_sq",
                     "edge_side_mean_cd_sq", "gt_pair_edge_cd_sq_mean"):
            self.assertAlmostEqual(result[name], 0.5)
        self.assertAlmostEqual(result["edge_pred_to_gt_sq"], 0.25)
        self.assertAlmostEqual(result["edge_gt_to_pred_sq"], 0.25)
        self.assertAlmostEqual(result["gt_pair_curve_xyz_mse"], 0.25 / 3)
        self.assertEqual(result["active_gap_mean"], 0.0)

    def test_one_bad_side_cannot_be_hidden(self):
        arrays = fixture()
        for name in ("pred_edge_xyz_by_face", "diagnostic_gt_pair_xyz"):
            arrays[name][:, 1, :, 1] += 2.0
        result = measure(arrays)
        self.assertEqual(result["edge_cd_sq"], 2.0)
        self.assertEqual(result["edge_side0_cd_sq"], 0.0)
        self.assertEqual(result["edge_side1_cd_sq"], 8.0)
        self.assertEqual(result["edge_side_mean_cd_sq"], 4.0)
        self.assertEqual(result["edge_side_worst_cd_sq"], 8.0)
        for stat in ("mean", "rms", "p50", "p90", "p95", "p99", "max"):
            self.assertEqual(result[f"active_gap_{stat}"], 2.0)
            self.assertEqual(result[f"gt_pair_gap_{stat}"], 2.0)
        self.assertEqual(result["active_pair_cd_sq_mean"], 8.0)

    def test_common_reversal_and_independent_reversal_differ(self):
        arrays = fixture()
        for name in ("pred_edge_xyz_by_face", "diagnostic_gt_pair_xyz"):
            arrays[name] = arrays[name][:, :, ::-1].copy()
        result = measure(arrays)
        self.assertEqual(result["gt_pair_curve_xyz_mse"], 0.0)
        self.assertEqual(result["edge_cd_sq"], 0.0)
        arrays["diagnostic_gt_pair_xyz"][:, 1] = arrays["gt_edge"]
        result = measure(arrays)
        self.assertGreater(result["gt_pair_curve_xyz_mse"], 0.0)
        self.assertGreater(result["gt_pair_gap_mean"], 0.0)
        self.assertEqual(result["gt_pair_pair_cd_sq_mean"], 0.0)

    def test_empty_predictions_keep_faces_and_gt_diagnostics(self):
        arrays = fixture()
        arrays["pred_edge_xyz_by_face"] = np.empty((0, 2, 16, 3))
        arrays["pred_edge_owner_pairs"] = np.empty((0, 2), dtype=np.int64)
        result = measure(arrays)
        self.assertEqual(result["face_cd_sq"], 0.0)
        self.assertEqual(result["edge_status"], "empty_prediction")
        self.assertIsNone(result["edge_cd_sq"])
        self.assertIsNone(result["active_gap_mean"])
        self.assertEqual(result["topology_fn"], 1)
        self.assertEqual(result["topology_recall"], 0.0)
        self.assertEqual(result["gt_pair_curve_xyz_mse"], 0.0)
        self.assertEqual(result["gt_pair_gap_count"], 16)
        json.dumps(result, allow_nan=False)

    def test_empty_gt_and_both_empty_are_explicit(self):
        arrays = fixture()
        arrays["gt_edge"] = np.empty((0, 16, 3))
        arrays["diagnostic_gt_pair_xyz"] = np.empty((0, 2, 16, 3))
        arrays["gt_edge_owner_pairs"] = np.empty((0, 2), dtype=np.int64)
        result = measure(arrays)
        self.assertIsNone(result["edge_cd_sq"])
        self.assertIsNone(result["gt_pair_curve_xyz_mse"])
        self.assertEqual(result["topology_fp"], 1)
        self.assertEqual(result["edge_status"], "empty_ground_truth")
        arrays["pred_edge_xyz_by_face"] = np.empty((0, 2, 16, 3))
        arrays["pred_edge_owner_pairs"] = np.empty((0, 2), dtype=np.int64)
        result = measure(arrays)
        self.assertEqual(result["edge_status"], "empty_both")
        self.assertIsNone(result["edge_cd_sq"])
        self.assertEqual(result["face_mae"], 0.0)
        self.assertEqual(result["topology_f1"], 1.0)

    def test_malformed_or_nonfinite_inputs_rejected(self):
        cases = (
            ("pred_face", np.zeros((2, 16, 15, 3))),
            ("pred_edge_owner_pairs", np.array([[1, 0]])),
            ("gt_edge_owner_pairs", np.array([[0, 2]])),
            ("diagnostic_gt_pair_xyz", np.empty((0, 2, 16, 3))),
            ("pred_face", np.full((2, 16, 16, 3), np.nan)),
        )
        for key, value in cases:
            with self.subTest(key=key):
                arrays = copy.deepcopy(fixture())
                arrays[key] = value
                with self.assertRaises(ValueError):
                    measure(arrays)


if __name__ == "__main__":
    unittest.main(verbosity=2)
