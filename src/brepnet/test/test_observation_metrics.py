"""Analytic perspective, visibility and silhouette checks without OCC or datasets."""

import unittest
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np

from src.brepnet.eval.metrics.observation import (
    compare_depth_maps,
    _cached_gt_depth,
    legacy_svr_camera,
    project_points,
    rasterize_meshes,
    silhouette_metrics,
    unproject_points,
)


class ObservationMetricTests(unittest.TestCase):
    def setUp(self):
        self.camera = legacy_svr_camera(64)
        self.triangles = np.array([[0, 1, 2], [0, 2, 3]])
        self.xy = np.array([[10., 10.], [50., 10.], [50., 50.], [10., 50.]])

    def square(self, depth):
        vertices = unproject_points(self.xy, np.full(4, depth), self.camera)
        return vertices, self.triangles

    def render(self, *meshes):
        return rasterize_meshes(meshes, self.camera)

    def test_projection_roundtrip_preserves_image_y(self):
        points = np.array([[.2, -.3, .4], [-.4, .1, -.1], [0., 0., 0.]])
        xy, depth = project_points(points, self.camera)
        np.testing.assert_allclose(unproject_points(xy, depth, self.camera), points, atol=1e-14)
        np.testing.assert_allclose(project_points(np.zeros((1, 3)), self.camera)[0], [[31.5, 31.5]])

    def test_sloped_triangle_uses_perspective_depth(self):
        xy = np.array([[10., 10.], [40., 10.], [10., 40.]])
        depths = np.array([2., 4., 4.])
        vertices = unproject_points(xy, depths, self.camera)
        rendered = self.render((vertices, np.array([[0, 1, 2]])))
        self.assertAlmostEqual(rendered[20, 20], 3.)
        self.assertNotAlmostEqual(rendered[20, 20], depths.mean())
        normal = np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])
        rows, cols = np.nonzero(np.isfinite(rendered))
        visible = unproject_points(np.column_stack([cols, rows]), rendered[rows, cols], self.camera)
        np.testing.assert_allclose((visible - vertices[0]) @ normal, 0., atol=1e-13)

    def test_hidden_geometry_does_not_change_visible_measurement(self):
        front = self.render(self.square(2.))
        hidden_changed = self.render(self.square(4.), self.square(2.))
        np.testing.assert_allclose(front, hidden_changed, equal_nan=True)
        result = compare_depth_maps(hidden_changed, front, np.isfinite(front), self.camera)
        self.assertAlmostEqual(result["visible_face_cd_sq"], 0.)
        self.assertAlmostEqual(result["silhouette_iou"], 1.)
        self.assertAlmostEqual(result["silhouette_boundary_distance"], 0.)

    def test_foreground_depth_change_is_detected_without_silhouette_change(self):
        gt = self.render(self.square(3.))
        pred = self.render(self.square(2.), self.square(3.))
        result = compare_depth_maps(pred, gt, np.isfinite(gt), self.camera)
        self.assertGreater(result["visible_face_cd_sq"], 0.)
        self.assertAlmostEqual(result["silhouette_iou"], 1.)
        self.assertAlmostEqual(result["depth_mae"], 1.)
        self.assertAlmostEqual(result["overlap_coverage"], 1.)

    def test_disjoint_foreground_has_no_depth_mae(self):
        gt = np.full((64, 64), np.nan)
        pred = gt.copy()
        gt[2:8, 2:8] = 2.
        pred[50:56, 50:56] = 2.
        result = compare_depth_maps(pred, gt, np.isfinite(gt), self.camera)
        self.assertIsNone(result["depth_mae"])
        self.assertEqual(result["overlap_coverage"], 0.)
        self.assertEqual(result["silhouette_iou"], 0.)
        self.assertGreater(result["silhouette_boundary_distance"], 0.)

    def test_boundary_distance_has_fixed_diagonal_normalization(self):
        left = np.zeros((64, 64), dtype=bool)
        right = left.copy()
        left[10, 10], right[10, 13] = True, True
        iou, distance = silhouette_metrics(left, right)
        self.assertEqual(iou, 0.)
        self.assertAlmostEqual(distance, 6. / np.hypot(64, 64))

    def test_unverified_view_and_empty_foreground_fail_explicitly(self):
        with self.assertRaisesRegex(ValueError, "view_id=0"):
            legacy_svr_camera(view_id=1)
        with self.assertRaisesRegex(ValueError, "nonempty"):
            silhouette_metrics(np.zeros((3, 3), dtype=bool), np.ones((3, 3), dtype=bool))

    def test_gt_cache_keys_include_source_content_and_meshing_parameters(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "shape.step"
            source.write_text("first shape")
            depth = np.ones((64, 64))
            dependencies = {"observation_sha256": "v1", "triangulation_sha256": "v1", "pythonocc_version": "v1"}
            with patch("src.brepnet.eval.metrics.observation.render_step", return_value=depth) as render, \
                 patch("src.brepnet.eval.metrics.observation.renderer_dependencies", return_value=dependencies) as identity:
                _cached_gt_depth(source, self.camera, .002, root / "cache")
                np.testing.assert_array_equal(_cached_gt_depth(source, self.camera, .002, root / "cache"), depth)
                self.assertEqual(render.call_count, 1)
                _cached_gt_depth(source, self.camera, .001, root / "cache")
                source.write_text("second shape")
                _cached_gt_depth(source, self.camera, .002, root / "cache")
                self.assertEqual(render.call_count, 3)
                for key in dependencies:
                    identity.return_value = {**identity.return_value, key: "v2"}
                    _cached_gt_depth(source, self.camera, .002, root / "cache")
                self.assertEqual(render.call_count, 6)


if __name__ == "__main__":
    unittest.main()
