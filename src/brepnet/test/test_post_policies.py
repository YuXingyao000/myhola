"""Focused real-OCC contracts for opt-in post-processing policies."""
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakePolygon, BRepBuilderAPI_MakeWire
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCC.Core.Geom import Geom_Plane
from OCC.Core.Interface import Interface_Static
from OCC.Core.ShapeFix import ShapeFix_ShapeTolerance
from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_WIRE
from OCC.Core.TopoDS import topods
from OCC.Core.gp import gp_Pln, gp_Pnt, gp_Dir

from shared.occ_utils import get_primitives
from src.brepnet.post import construct_brep as post
from src.brepnet.post.occ_fit import create_surface
from src.brepnet.post.occ_topology import copy_occ_shapes, create_trimmed_face_from_wire, get_edge_length


class PostPolicyTests(unittest.TestCase):
    def test_candidate_copy_isolated_and_preserves_shared_edges(self):
        faces = get_primitives(BRepPrimAPI_MakeBox(1, 1, 1).Shape(), TopAbs_FACE)
        copies = copy_occ_shapes(faces)
        first_edges = get_primitives(copies[0], TopAbs_EDGE)
        shared = [edge for edge in first_edges if any(edge.IsSame(other)
                  for face in copies[1:] for other in get_primitives(face, TopAbs_EDGE))]
        self.assertEqual(len(shared), 4)
        ShapeFix_ShapeTolerance().SetTolerance(copies[0], .05)
        self.assertLess(BRep_Tool.Tolerance(faces[0]), 1e-5)
        self.assertFalse(copies[0].IsSame(faces[0]))

    def test_native_roundtrip_restores_global_read_precision(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "box.step"
            post._save_step_checked(path, BRepPrimAPI_MakeBox(1, 1, 1).Shape())
            mode = Interface_Static.IVal("read.precision.mode")
            precision = Interface_Static.RVal("read.precision.val")
            _, summary = post._read_native_post_step(path)
            self.assertTrue(summary["native_valid_solid"])
            self.assertLess(summary["maximum_stored_tolerance"], .002)
        self.assertEqual(Interface_Static.IVal("read.precision.mode"), mode)
        self.assertEqual(Interface_Static.RVal("read.precision.val"), precision)

    def test_roundtrip_rejection_continues_and_trials_do_not_mutate_source(self):
        solid = BRepPrimAPI_MakeBox(1, 1, 1).Solid()
        faces = get_primitives(solid, TopAbs_FACE)
        original_read = post._read_native_post_step
        before = []
        reads = []

        def get_solid(candidate_faces, tolerance):
            before.append(BRep_Tool.Tolerance(candidate_faces[0]))
            ShapeFix_ShapeTolerance().SetTolerance(candidate_faces[0], .5)
            return solid

        def read(path):
            shape, summary = original_read(path)
            reads.append(path)
            if len(reads) == 1:
                summary["native_valid_solid"] = False
            return shape, summary

        with tempfile.TemporaryDirectory() as directory, patch.object(post, "get_solid", get_solid), \
                patch.object(post, "_read_native_post_step", read):
            report = {}
            self.assertTrue(post._try_save_complete_solid(faces, Path(directory), report, lambda: None, None))
            self.assertEqual([x["status"] for x in report["solid_attempts"]], ["native_rejected", "accepted"])
            self.assertTrue((Path(directory) / "success.txt").exists())
        self.assertEqual(len(before), 2)
        self.assertLess(max(before), 1e-5)

    def test_ordinary_open_wire_is_not_silently_discarded(self):
        plane = Geom_Plane(gp_Pln(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)))
        outer = BRepBuilderAPI_MakePolygon()
        for point in ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)):
            outer.Add(gp_Pnt(*point))
        outer.Close()
        inner = BRepBuilderAPI_MakePolygon()
        inner.Add(gp_Pnt(.2, .2, 0))
        inner.Add(gp_Pnt(.7, .2, 0))
        wires = [outer.Wire(), inner.Wire()]
        edges = [topods.Edge(edge) for wire in wires for edge in get_primitives(wire, TopAbs_EDGE)]
        diagnostics = {}
        result = create_trimmed_face_from_wire(plane, edges, wires, .002,
                                               require_all_wires=True, diagnostics=diagnostics)
        self.assertIsNone(result)
        self.assertEqual(diagnostics["rejection"], "unexplained_wire_skip")
        self.assertFalse(diagnostics["boundary_complete"])

    def test_fitted_periodic_surface_keeps_full_natural_loop(self):
        angles = np.linspace(0, 2 * np.pi, 16)
        heights = np.linspace(0, 1, 16)
        points = np.array([[[np.cos(a), np.sin(a), h] for a in angles] for h in heights])
        surface = create_surface(points)
        self.assertTrue(surface.IsUPeriodic() or surface.IsVPeriodic())
        natural = BRepBuilderAPI_MakeFace(surface, 1e-6).Face()
        self.assertEqual(len(get_primitives(natural, TopAbs_WIRE)), 1)
        edges = [topods.Edge(edge) for edge in get_primitives(natural, TopAbs_EDGE)
                 if get_edge_length(topods.Edge(edge)) > 2]
        self.assertEqual(len(edges), 2)
        wires = [BRepBuilderAPI_MakeWire(edge).Wire() for edge in edges]
        diagnostics = {}
        result = create_trimmed_face_from_wire(surface, edges, wires, .002,
                                               require_all_wires=True, diagnostics=diagnostics)
        self.assertIsNotNone(result)
        self.assertTrue(diagnostics["natural_wire_replacement"])
        self.assertEqual(diagnostics["input_wire_count"], 2)
        self.assertEqual(diagnostics["effective_wire_count"], 1)
        self.assertTrue(diagnostics["boundary_complete"])

    def test_expired_shared_budget_does_not_start_a_path(self):
        with tempfile.TemporaryDirectory() as directory:
            post.construct_brep_for_sample("unused", directory, "sample", post_policy="recover_v1",
                                           deadline_monotonic=time.monotonic() - 1)
            diagnostics = json.loads((Path(directory) / "sample/post_diagnostics.json").read_text())
            self.assertEqual(diagnostics["status"], "budget_exhausted")
            self.assertEqual(diagnostics["paths"], [])

    def test_recovery_preserves_native_legacy_without_running_fallback(self):
        original_construct = post.construct_brep_for_sample
        solid = BRepPrimAPI_MakeBox(1, 1, 1).Solid()

        def legacy(data_root, out_root, sample_id, **kwargs):
            path = Path(out_root) / sample_id
            path.mkdir(parents=True, exist_ok=True)
            post._save_step_checked(path / "recon_brep.step", solid)

        with tempfile.TemporaryDirectory() as directory, patch.object(post, "construct_brep_for_sample", legacy), \
                patch.object(post, "_run_complete_post_path", side_effect=AssertionError("Unexpected fallback")):
            original_construct("unused", directory, "sample", post_policy="recover_v1")
            diagnostics = json.loads((Path(directory) / "sample/post_diagnostics.json").read_text())
            self.assertEqual(diagnostics["chosen_path"], "legacy")
            self.assertIsNone(diagnostics["boundary_complete"])
            self.assertEqual(len(diagnostics["paths"]), 1)

    def test_recovery_preserves_legacy_step_after_auxiliary_exception(self):
        original_construct = post.construct_brep_for_sample

        def legacy(data_root, out_root, sample_id, **kwargs):
            path = Path(out_root) / sample_id
            path.mkdir(parents=True, exist_ok=True)
            post._save_step_checked(path / "recon_brep.step", BRepPrimAPI_MakeBox(1, 1, 1).Solid())
            raise RuntimeError("Simulated optional STL failure")

        with tempfile.TemporaryDirectory() as directory, patch.object(post, "construct_brep_for_sample", legacy), \
                patch.object(post, "_run_complete_post_path", side_effect=AssertionError("Unexpected fallback")):
            original_construct("unused", directory, "sample", post_policy="recover_v1")
            diagnostics = json.loads((Path(directory) / "sample/post_diagnostics.json").read_text())
            self.assertEqual(diagnostics["chosen_path"], "legacy")
            self.assertEqual(diagnostics["paths"][0]["status"], "accepted_with_auxiliary_error")

    def test_recovery_rejects_mislabeled_fit_only_mode(self):
        with self.assertRaisesRegex(ValueError, "optimized policy"):
            post.construct_brep_for_sample("unused", "unused", "sample", is_optimize_geom=False,
                                           post_policy="recover_v1")


if __name__ == "__main__":
    unittest.main()
