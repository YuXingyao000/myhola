"""Focused observational-equivalence and tri-state provenance checks."""
import random
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tools.legacy_failure_provenance import (
    ProvenanceTrace, curve_distances, mechanism_flags, step_data_hash, surface_distances,
)


class ProvenanceTests(unittest.TestCase):
    def test_step_hash_excludes_only_header(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "a.step"
            p.write_bytes(b"HEADER;abc ENDSEC;DATA;\n#1=X;\nENDSEC;")
            first = step_data_hash(p)
            p.write_bytes(b"HEADER;other ENDSEC;DATA;\n#1=X;\nENDSEC;")
            self.assertEqual(first, step_data_hash(p))
            p.write_bytes(b"HEADER;other ENDSEC;DATA;\n#1=Y;\nENDSEC;")
            self.assertNotEqual(first, step_data_hash(p))

    def test_unknown_is_not_absent_and_labels_overlap(self):
        self.assertIsNone(mechanism_flags({})["all_wires_rejected"])
        detail = {"outcome": "success", "wire_events": [
            {"repaired_closed": True, "added": True, "skip_reason": None},
            {"repaired_closed": False, "added": False, "skip_reason": "open_after_fix"}]}
        trace = {"status": "ok", "trim_runs": [{"attempts": [{"trim_detail": detail}]}]}
        flags = mechanism_flags(trace)
        self.assertTrue(flags["partial_wire_drop_with_face_returned"])
        self.assertTrue(flags["wire_skipped_open_after_fix"])
        self.assertFalse(flags["all_wires_rejected"])
        self.assertIsNone(flags["uv_specific_failure"])

    def test_projection_is_length_not_squared(self):
        from OCC.Core.Geom import Geom_Plane
        from OCC.Core.gp import gp_Pln, gp_Pnt, gp_Dir
        surface = Geom_Plane(gp_Pln(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)))
        result = surface_distances([[0, 0, .1], [1, 1, .2]], surface)
        self.assertEqual(result["errors"], [])
        self.assertAlmostEqual(result["mean"], .15)
        self.assertAlmostEqual(result["max"], .2)

    def test_curve_distance_includes_endpoint_minimum(self):
        from OCC.Core.Geom import Geom_Line
        from OCC.Core.gp import gp_Pnt, gp_Dir
        curve = Geom_Line(gp_Pnt(0, 0, 0), gp_Dir(1, 0, 0))
        result = curve_distances([[-.2, 0, 0], [.5, .1, 0]], curve, 0., 1.)
        self.assertEqual(result["status"], "ok")
        self.assertAlmostEqual(result["mean"], .15)
        self.assertAlmostEqual(result["max"], .2)

    def test_closed_and_open_wire_paths_preserve_rng(self):
        from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
        from OCC.Core.Geom import Geom_Plane
        from OCC.Core.gp import gp_Pnt, gp_Pln, gp_Dir
        from src.brepnet.post import construct_brep as cb
        points = [gp_Pnt(0, 0, 0), gp_Pnt(1, 0, 0), gp_Pnt(1, 1, 0), gp_Pnt(0, 1, 0)]
        def edges(closed):
            return [BRepBuilderAPI_MakeEdge(points[i], points[(i + 1) % 4]).Edge() for i in range(4 if closed else 1)]
        surface = Geom_Plane(gp_Pln(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)))
        for closed in (True, False):
            raw = edges(closed)
            random.seed(918)
            expected_wires = cb.create_wire_from_unordered_edges(raw.copy(), .002)
            expected = cb.create_trimmed_face_from_wire(surface, raw, expected_wires, .002)
            rng = random.getstate()
            random.seed(918)
            trace = {}
            with ProvenanceTrace(trace, lambda: None) as observer:
                observer.current_trim = {"attempts": []}
                raw = edges(closed)
                wires = cb.create_wire_from_unordered_edges(raw.copy(), .002)
                result = cb.create_trimmed_face_from_wire(surface, raw, wires, .002)
                detail = observer.current_trim["attempts"][0]["trim_detail"]
                self.assertEqual(random.getstate(), rng)
                self.assertEqual(expected is None, result is None)
                self.assertEqual(detail["outcome"], "success" if closed else "no_admitted_wires")
                self.assertEqual(detail["wire_events"][0]["added"], closed)
                self.assertEqual(detail["wire_events"][0]["repaired_closed"], closed)
                if closed:
                    self.assertEqual(detail["final_brepcheck"]["inspection_errors"], [])
                    self.assertEqual(detail["final_brepcheck"]["with_errors"], [])
            self.assertFalse(trace.get("instrumentation_errors"))

    def test_real_optimizer_losses_and_coordinates_unchanged(self):
        import torch
        from src.brepnet.post.geometry_optimization import optimize
        torch.set_num_threads(1)
        x, y = np.meshgrid(np.linspace(0, 1, 16), np.linspace(0, 1, 16))
        face = np.stack([x, y, np.zeros_like(x)], axis=-1).astype("float32")
        faces = np.stack([face, face + np.array([0, 0, .01], dtype="float32")])
        edges = np.stack([face[0], face[-1]])
        args = ([torch.from_numpy(f.reshape(-1, 3)) for f in faces], edges, faces,
                np.array([[0, 0, 1], [1, 0, 1]]), np.array([], dtype=bool), None, [[0, 1], [0, 1]])
        expected = optimize(*args, v_islog=False, v_max_iter=2, use_cuda=False)
        trace = {}
        with ProvenanceTrace(trace, lambda: None):
            result = optimize(*args, v_islog=False, v_max_iter=2, use_cuda=False)
        self.assertEqual(len(trace["optimization_detail"]["loss_records"]), 2)
        self.assertEqual(trace["optimization_detail"]["final_iteration_count"], 2)
        self.assertFalse(trace.get("instrumentation_errors"))
        for first, second in zip(expected, result):
            np.testing.assert_array_equal(first, second)


if __name__ == "__main__":
    unittest.main()
