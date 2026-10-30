"""Fitting checks use XYZ distances and include the previously missed boundary."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeFace
from OCC.Core.GeomAPI import GeomAPI_PointsToBSpline, GeomAPI_PointsToBSplineSurface
from OCC.Core.TColgp import TColgp_Array1OfPnt, TColgp_Array2OfPnt
from OCC.Core.gp import gp_Pnt

from src.postprocess.constants import EDGE_FITTING_TOLERANCE, FACE_FITTING_TOLERANCE
from src.postprocess.occ_fit import _fitting_metrics_v2, create_edge, create_surface


def line_points(offset=0.0):
    return np.stack([np.linspace(0, 1, 16), np.full(16, offset), np.zeros(16)], axis=-1)


def plane_points(offset=0.0):
    u, v = np.meshgrid(np.linspace(0, 1, 16), np.linspace(0, 1, 16), indexing="ij")
    return np.stack([u, v, np.full_like(u, offset)], axis=-1)


def occ_line(offset):
    points = TColgp_Array1OfPnt(1, 16)
    for i, point in enumerate(line_points(offset), 1):
        points.SetValue(i, gp_Pnt(*point))
    return GeomAPI_PointsToBSpline(points).Curve()


def occ_plane(offset):
    points = TColgp_Array2OfPnt(1, 16, 1, 16)
    for i, row in enumerate(plane_points(offset), 1):
        for j, point in enumerate(row, 1):
            points.SetValue(i, j, gp_Pnt(*point))
    return GeomAPI_PointsToBSplineSurface(points).Surface()


class OccFittingTests(unittest.TestCase):
    def test_rms_is_a_distance_not_square_root_of_a_distance(self):
        shape = BRepBuilderAPI_MakeEdge(occ_line(0.0)).Edge()
        rms, maximum = _fitting_metrics_v2(shape, [[0.25, 0.003, 0], [0.75, 0.004, 0]])
        self.assertAlmostEqual(rms, np.sqrt((0.003 ** 2 + 0.004 ** 2) / 2), places=9)
        self.assertAlmostEqual(maximum, 0.004, places=9)

    def test_endpoint_error_prevents_early_stop_and_changes_curve_selection(self):
        points = line_points()
        points[-1, 1] = 0.004
        curves = [occ_line(offset) for offset in [0.0, 0.001, 0.002, 0.003]]
        seen = []

        def fit_candidate(*args):
            precision = args[-1]
            seen.append(precision)
            curve = curves[EDGE_FITTING_TOLERANCE.index(precision)]
            return SimpleNamespace(Curve=lambda: curve)

        with patch("src.postprocess.occ_fit.GeomAPI_PointsToBSpline", side_effect=fit_candidate):
            legacy = create_edge(points)
            self.assertEqual(seen, [EDGE_FITTING_TOLERANCE[0]])
            self.assertAlmostEqual(legacy.Value(legacy.FirstParameter()).Y(), 0.0)
            seen.clear()
            selected = create_edge(points, fitting_policy="fit_score_v2")
        self.assertEqual(seen, EDGE_FITTING_TOLERANCE)
        self.assertAlmostEqual(selected.Value(selected.FirstParameter()).Y(), 0.002)

    def test_last_face_row_and_column_affect_selection(self):
        points = plane_points()
        points[-1, :, 2] = 0.004
        points[:, -1, 2] = 0.004
        surfaces = [occ_plane(offset) for offset in [0.0, 0.001, 0.002, 0.003, 0.004]]
        seen = []

        def fit_candidate(*args):
            precision = args[-1]
            seen.append(precision)
            surface = surfaces[FACE_FITTING_TOLERANCE.index(precision)]
            return SimpleNamespace(Surface=lambda: surface)

        with patch("src.postprocess.occ_fit.GeomAPI_PointsToBSplineSurface", side_effect=fit_candidate):
            legacy = create_surface(points)
            self.assertEqual(seen, [FACE_FITTING_TOLERANCE[0]])
            self.assertAlmostEqual(legacy.Value(0.5, 0.5).Z(), 0.0)
            seen.clear()
            selected = create_surface(points, fitting_policy="fit_score_v2")
        self.assertEqual(seen, FACE_FITTING_TOLERANCE)
        self.assertAlmostEqual(selected.Value(0.5, 0.5).Z(), 0.002)

    def test_first_candidate_stops_when_all_checked_distances_are_small(self):
        curve = occ_line(0.0)
        with patch("src.postprocess.occ_fit.GeomAPI_PointsToBSpline",
                   return_value=SimpleNamespace(Curve=lambda: curve)) as fit:
            create_edge(line_points(), fitting_policy="fit_score_v2")
        self.assertEqual(fit.call_count, 1)

    def test_real_fits_retain_periodic_surface_handling(self):
        theta = np.linspace(0, 2 * np.pi, 16)[:, None]
        height = np.linspace(0, 1, 16)[None, :]
        points = np.stack([
            np.broadcast_to(0.4 * np.cos(theta), (16, 16)),
            np.broadcast_to(0.4 * np.sin(theta), (16, 16)),
            np.broadcast_to(height, (16, 16)),
        ], axis=-1)
        for policy in ("legacy", "fit_score_v2"):
            surface = create_surface(points, fitting_policy=policy)
            self.assertTrue(surface.IsUPeriodic())
            self.assertFalse(surface.IsVPeriodic())
            topo_face = BRepBuilderAPI_MakeFace(surface, 1e-6).Face()
            _, maximum = _fitting_metrics_v2(topo_face, points.reshape(-1, 3))
            self.assertLess(maximum, 1e-3)

    def test_unknown_policy_is_not_silently_treated_as_legacy(self):
        with self.assertRaisesRegex(ValueError, "Unknown fitting_policy"):
            create_edge(line_points(), fitting_policy="typo")
        with self.assertRaisesRegex(ValueError, "Unknown fitting_policy"):
            create_surface(plane_points(), fitting_policy="typo")


if __name__ == "__main__":
    unittest.main()
