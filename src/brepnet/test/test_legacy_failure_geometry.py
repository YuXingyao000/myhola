"""Numerical checks for diagnostic geometry, without repairing any shape."""
import tempfile
from pathlib import Path
import unittest

import numpy as np

from tools.legacy_failure_geometry import (
    common_bounds, mesh_face, native_diagnostics, raw_panel_and_diagnostics, wire_gaps,
)
from tools.legacy_failure_extent import extent_diagnostics


class LegacyFailureGeometryTests(unittest.TestCase):
    def test_extent_ratios_are_scale_invariant(self):
        x, y = np.meshgrid(np.linspace(0, 2, 16), np.linspace(0, 1, 16))
        truth = np.stack((x, y, x * 0.005), -1)[None]
        prediction = truth + [0.02, 0.01, 0.001]
        first = extent_diagnostics(truth, prediction)
        second = extent_diagnostics(truth * 10, prediction * 10)
        np.testing.assert_allclose(first["raw_abs_error_mean_xyz_over_span"], [0.01, 0.01, 0.1])
        np.testing.assert_allclose(first["raw_abs_error_mean_xyz_over_span"], second["raw_abs_error_mean_xyz_over_span"])
        self.assertAlmostEqual(first["raw_xyz_mae_over_min_span"], second["raw_xyz_mae_over_min_span"])

    def test_zero_extent_remains_undefined(self):
        truth = np.zeros((1, 16, 16, 3))
        result = extent_diagnostics(truth, truth + [0, 0, 0.1])
        self.assertEqual(result["raw_abs_error_mean_xyz_over_span"], [None, None, None])
        self.assertIsNone(result["raw_xyz_mae_over_min_span"])
        self.assertIsNone(result["min_over_max_span"])

    def test_box_and_open_face_are_not_conflated(self):
        from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
        from OCC.Core.TopoDS import topods
        box = BRepPrimAPI_MakeBox(1, 1, 1).Shape()
        report, edges, maps = native_diagnostics(box)
        self.assertTrue(report["native_valid_solid"])
        self.assertEqual(report["counts"]["faces"], 6)
        self.assertEqual(report["counts"]["edges"], 12)
        self.assertEqual(report["incidence"]["free_edges"], 0)
        self.assertEqual(report["incidence"]["nonmanifold_edges"], 0)
        self.assertTrue(all(edge["incidence_degree"] == 2 for edge in edges))
        self.assertFalse(report["brepcheck_subshapes"]["with_errors"])
        self.assertFalse(report["brepcheck_subshapes"]["inspection_errors"])
        gaps = wire_gaps(maps)["wires"]
        self.assertEqual(len(gaps), 6)
        self.assertTrue(all(row["traversal_complete"] and row["max"] == 0 for row in gaps))
        face = topods.Face(maps["faces"].FindKey(1))
        open_report, _, _ = native_diagnostics(face)
        self.assertTrue(open_report["native_brepcheck_valid"])
        self.assertFalse(open_report["native_valid_solid"])
        self.assertEqual(open_report["counts"]["shells"], 0)
        self.assertEqual(open_report["incidence"]["free_edges"], 4)
        self.assertFalse(open_report["brepcheck_subshapes"]["inspection_errors"])
        mesh, measurement = mesh_face(face)
        self.assertEqual(measurement["status"], "ok")
        self.assertEqual(len(mesh["indices"]), 6)

    def test_seams_and_degenerate_edges_are_not_holes(self):
        from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeSphere
        shape = BRepPrimAPI_MakeSphere(1).Shape()
        report, _, _ = native_diagnostics(shape)
        self.assertTrue(report["native_valid_solid"])
        self.assertEqual(report["incidence"]["free_edges"], 0)
        self.assertEqual(report["incidence"]["nonmanifold_edges"], 0)
        self.assertGreater(report["incidence"]["seam_edges"], 0)
        self.assertGreater(report["incidence"]["degenerate_edges"], 0)
        self.assertFalse(report["brepcheck_subshapes"]["inspection_errors"])

    def test_known_invalid_saved_solid_has_named_diagnostics(self):
        from tools.vae_compare_post import read_step_at_precision
        path = Path("checkpoints/joint_boundary/comparison_20261011/post/optimized/legacy/00480996/recon_brep.step")
        if not path.is_file():
            self.skipTest("Saved comparison asset not mounted")
        report, _, _ = native_diagnostics(read_step_at_precision(path))
        self.assertTrue(report["native_is_solid_type"])
        self.assertFalse(report["native_brepcheck_valid"])
        self.assertFalse(report["brepcheck_subshapes"]["inspection_errors"])
        self.assertTrue(report["brepcheck_subshapes"]["with_errors"])
        names = [name for item in report["brepcheck_subshapes"]["with_errors"]
                 for name in item["statuses"] + [s for ctx in item["contexts"] for s in ctx["statuses"]]]
        self.assertTrue(all(name.startswith("BRepCheck_") and "UnknownStatus" not in name for name in names))

    def test_raw_reciprocal_reversal_and_gap(self):
        x, y = np.meshgrid(np.linspace(0, 1, 16), np.linspace(0, 1, 16))
        face = np.stack((x, y, np.zeros_like(x)), -1)
        curve = face[0]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "data.npz"
            def evaluate(second):
                np.savez(path, pred_face=np.stack((face, face)),
                         pred_edge=np.stack((curve, second)),
                         pred_edge_face_connectivity=np.array([[0, 0, 1], [1, 1, 0]]))
                return raw_panel_and_diagnostics(path)
            panel, report = evaluate(curve[::-1])
            self.assertEqual(report["reciprocal"]["open_endpoint_mean"]["mean"], 0)
            self.assertEqual(report["reciprocal"]["polyline_max"]["max"], 0)
            self.assertEqual(report["curve_to_face_grid"]["side_max"]["max"], 0)
            _, shifted = evaluate(curve[::-1] + [0, 0, 0.1])
            self.assertAlmostEqual(shifted["reciprocal"]["open_endpoint_mean"]["mean"], 0.1)
            self.assertAlmostEqual(shifted["reciprocal"]["polyline_max"]["max"], 0.1)
            self.assertAlmostEqual(shifted["curve_to_face_grid"]["side_max"]["max"], 0.1)
            bounds = common_bounds({"raw": panel})
            self.assertEqual(bounds["min"], [0, 0, 0])
            self.assertEqual(bounds["max"], [1, 1, 0])


if __name__ == "__main__":
    unittest.main()
