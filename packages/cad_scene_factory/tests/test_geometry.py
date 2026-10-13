"""解析几何用例；无需 Blender、CAD 文件或第三方测试框架。"""

from dataclasses import replace
import math
import unittest

from cad_scene_factory.geometry import (
    BoundingBox,
    PerspectiveCamera,
    check_bbox_framing,
    plan_support_placement,
    project_point,
)


class GeometryTests(unittest.TestCase):
    def setUp(self):
        # 沿 +Y 拍摄；世界 +X 向画面右，世界 +Z 向画面上。
        self.axis_camera = PerspectiveCamera(
            position=(0, -3, 0), target=(0, 0, 0),
            vertical_fov_degrees=90, width=200, height=100,
        )

    def test_default_camera_projects_origin_to_center(self):
        u, v, depth = project_point((0, 0, 0), PerspectiveCamera())
        self.assertAlmostEqual(u, 256)
        self.assertAlmostEqual(v, 256)
        self.assertAlmostEqual(depth, math.sqrt(12))

    def test_default_camera_has_expected_right_and_up_directions(self):
        camera = PerspectiveCamera()
        self.assertGreater(project_point((-0.5, 0.5, 0), camera)[0], 256)
        self.assertLess(project_point((0, 0, 0.5), camera)[1], 256)

    def test_rectangular_image_preserves_vertical_fov_and_square_pixels(self):
        # 深度 3、垂直 FOV 90°：画幅覆盖高度 6、宽度 12。
        u, v, depth = project_point((1.5, 0, 1.5), self.axis_camera)
        self.assertAlmostEqual(u, 125)
        self.assertAlmostEqual(v, 25)
        self.assertAlmostEqual(depth, 3)

    def test_support_translation_uses_each_actual_minimum(self):
        for h in (-0.9, -0.2, 1.5):
            with self.subTest(h=h):
                bounds = BoundingBox((-0.7, -0.2, h), (0.3, 0.5, h + 1.3))
                placement = plan_support_placement(bounds)
                self.assertEqual(placement.environment_translation, (0, 0, h))
                local = placement.target_bounds_in_support_frame
                self.assertEqual(local.minimum, (-0.7, -0.2, 0))
                self.assertEqual(local.maximum[:2], (0.3, 0.5))
                self.assertAlmostEqual(local.maximum[2], 1.3)
                self.assertEqual(bounds.minimum, (-0.7, -0.2, h))

    def test_support_frame_change_preserves_projection(self):
        bounds = BoundingBox((-0.4, -0.2, -0.7), (0.3, 0.5, 0.6))
        placement = plan_support_placement(bounds)
        h = placement.environment_translation[2]
        camera = PerspectiveCamera()
        local_camera = replace(
            camera, position=(2, 2, 2 - h), target=(0, 0, -h),
        )
        for world, local in zip(bounds.corners(), placement.target_bounds_in_support_frame.corners()):
            world_projection = project_point(world, camera)
            local_projection = project_point(local, local_camera)
            for expected, actual in zip(world_projection, local_projection):
                self.assertAlmostEqual(expected, actual)
        self.assertEqual(camera.position, (2, 2, 2))
        self.assertEqual(camera.target, (0, 0, 0))

    def test_bbox_inside_and_outside_frame(self):
        inside = BoundingBox((-0.5, -0.5, -0.5), (0.5, 0.5, 0.5))
        outside = BoundingBox((-10, -0.5, -0.5), (10, 0.5, 0.5))
        self.assertTrue(check_bbox_framing(inside, self.axis_camera, near_clip=0.1, far_clip=10))
        self.assertFalse(check_bbox_framing(outside, self.axis_camera, near_clip=0.1, far_clip=10))

    def test_depth_clipping_is_axial_and_includes_endpoints(self):
        # (3,0,0) 深度为 3，距离为 sqrt(18)，必须按前向深度处理。
        point = BoundingBox((3, 0, 0), (3, 0, 0))
        self.assertTrue(check_bbox_framing(point, self.axis_camera, near_clip=1, far_clip=3))
        self.assertTrue(check_bbox_framing(point, self.axis_camera, near_clip=3, far_clip=4))
        self.assertFalse(check_bbox_framing(point, self.axis_camera, near_clip=1, far_clip=2.9))

    def test_all_four_image_boundaries_are_included_symmetrically(self):
        # 深度 3：横向边界 ±6，纵向边界 ±3；避免舍入误差只拒收左／上边。
        for point in ((-6, 0, 0), (6, 0, 0), (0, 0, -3), (0, 0, 3)):
            with self.subTest(point=point):
                self.assertTrue(check_bbox_framing(
                    BoundingBox(point, point), self.axis_camera, near_clip=1, far_clip=4,
                ))
        for point in ((-6.001, 0, 0), (6.001, 0, 0), (0, 0, -3.001), (0, 0, 3.001)):
            with self.subTest(point=point):
                self.assertFalse(check_bbox_framing(
                    BoundingBox(point, point), self.axis_camera, near_clip=1, far_clip=4,
                ))

    def test_degenerate_camera_axes_are_rejected_when_projecting(self):
        for parameters in (
            {"position": (0, 0, 0)},
            {"up_reference": (0, 0, 0)},
            {"position": (0, 0, 0), "target": (1, 2, 3), "up_reference": (5, 10, 15)},
            {"position": (0, 0, 0), "target": (1, 2, 3), "up_reference": (-5, -10, -15)},
        ):
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                camera = PerspectiveCamera(**parameters)
                project_point(camera.target, camera)

    def test_margin_can_reject_an_otherwise_in_frame_bbox(self):
        point = BoundingBox((0, 0, 2), (0, 0, 2))
        self.assertTrue(check_bbox_framing(point, self.axis_camera, near_clip=1, far_clip=4))
        self.assertFalse(check_bbox_framing(
            point, self.axis_camera, near_clip=1, far_clip=4, margin_pixels=20,
        ))

    def test_bbox_behind_or_crossing_camera_plane_is_not_certified(self):
        for bounds in (
            BoundingBox((-1, -5, -1), (1, -4, 1)),
            BoundingBox((-1, -4, -1), (1, 0, 1)),
        ):
            with self.subTest(bounds=bounds):
                self.assertFalse(check_bbox_framing(bounds, self.axis_camera, near_clip=0.1, far_clip=10))
        for point in ((0, -3, 0), (0, -4, 0)):
            with self.subTest(point=point), self.assertRaises(ValueError):
                project_point(point, self.axis_camera)


if __name__ == "__main__":
    unittest.main()
