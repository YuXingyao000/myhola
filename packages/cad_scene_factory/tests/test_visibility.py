"""AABB 占用保护区域的解析用例，不依赖 Blender 或外部几何库。"""

from dataclasses import replace
import unittest

from cad_scene_factory.geometry import BoundingBox, PerspectiveCamera
from cad_scene_factory.visibility import find_blockers


class VisibilityTests(unittest.TestCase):
    def setUp(self):
        self.target = BoundingBox((-1, -1, -1), (1, 1, 1))
        self.camera = PerspectiveCamera(position=(0, 0, 4), up_reference=(0, 1, 0))

    def test_empty_scene(self):
        self.assertEqual(find_blockers(self.target, self.camera, {}), [])

    def test_center_obstacle_missed_by_all_eight_corner_rays(self):
        # 在 z=2..2.2 处，八条角点射线的 |x|、|y| 至少为 0.36。
        # 中央半宽 0.1 的物体不碰这些射线，但确实进入拍摄通道。
        scene = {"center": BoundingBox((-0.1, -0.1, 2), (0.1, 0.1, 2.2))}
        self.assertEqual(find_blockers(self.target, self.camera, scene), ["center"])

    def test_object_inside_target_and_object_enclosing_whole_region(self):
        scene = {
            "inside_target": BoundingBox((-0.2, -0.2, -0.2), (0.2, 0.2, 0.2)),
            "enclosing": BoundingBox((-5, -5, -5), (5, 5, 5)),
        }
        self.assertEqual(find_blockers(self.target, self.camera, scene), list(scene))

    def test_background_sides_and_behind_camera_are_allowed(self):
        scene = {
            "background": BoundingBox((-5, -5, -3), (5, 5, -2)),
            "side": BoundingBox((2, -1, 0), (3, 1, 2)),
            "behind_camera": BoundingBox((-1, -1, 5), (1, 1, 6)),
        }
        self.assertEqual(find_blockers(self.target, self.camera, scene), [])

    def test_support_contact_allowed_but_penetration_blocked(self):
        scene = {
            "table": BoundingBox((-5, -5, -2), (5, 5, -1)),
            "floor_plane": BoundingBox((-5, -5, -1), (5, 5, -1)),
            "raised_table": BoundingBox((-5, -5, -2), (5, 5, -0.99)),
        }
        self.assertEqual(find_blockers(self.target, self.camera, scene), ["raised_table"])

    def test_internal_plane_line_and_point_are_detected(self):
        scene = {
            "plane": BoundingBox((-0.2, -0.2, 2), (0.2, 0.2, 2)),
            "line": BoundingBox((0, -0.2, 2), (0, 0.2, 2)),
            "point": BoundingBox((0, 0, 2), (0, 0, 2)),
        }
        self.assertEqual(find_blockers(self.target, self.camera, scene), list(scene))

    def test_sloping_boundary_contact_and_small_penetration(self):
        # 目标上方的锥段，右边界为 3*x+z=4。
        scene = {
            "touching": BoundingBox((0.5, -0.1, 2.5), (0.8, 0.1, 2.8)),
            "penetrating": BoundingBox((0.49, -0.1, 2.5), (0.8, 0.1, 2.8)),
            "outside_slope": BoundingBox((0.8, -0.1, 2.8), (1, 0.1, 3)),
        }
        self.assertEqual(find_blockers(self.target, self.camera, scene), ["penetrating"])

    def test_default_diagonal_camera_and_boundary_contact(self):
        scene = {
            "central": BoundingBox((1.4, 1.4, 1.4), (1.5, 1.5, 1.5)),
            "background": BoundingBox((-3, -3, -3), (-2, -2, -2)),
            "floor": BoundingBox((-5, -5, -1), (5, 5, -1)),
        }
        self.assertEqual(find_blockers(self.target, PerspectiveCamera(), scene), ["central"])

    def test_common_translation_and_scale_preserve_result(self):
        scene = {
            "center": BoundingBox((-0.1, -0.1, 2), (0.1, 0.1, 2.2)),
            "background": BoundingBox((-5, -5, -3), (5, 5, -2)),
        }

        def transform(point):
            return tuple(10 * v + shift for v, shift in zip(point, (5, 7, -6)))

        def transform_box(box):
            return BoundingBox(transform(box.minimum), transform(box.maximum))

        camera = replace(self.camera, position=transform(self.camera.position),
                         target=transform(self.camera.target))
        transformed_scene = {name: transform_box(box) for name, box in scene.items()}
        self.assertEqual(find_blockers(transform_box(self.target), camera, transformed_scene),
                         find_blockers(self.target, self.camera, scene))


if __name__ == "__main__":
    unittest.main()
