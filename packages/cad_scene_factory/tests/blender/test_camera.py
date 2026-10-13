"""在真实 Blender 中核对投影；运行命令见 BLENDER_CAMERA.md。"""

from dataclasses import replace
import unittest

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

from cad_scene_factory.blender.render import configure_camera
from cad_scene_factory.geometry import PerspectiveCamera, project_point


class BlenderCameraTests(unittest.TestCase):
    def setUp(self):
        self.scene = bpy.context.scene
        self.previous_camera = self.scene.camera
        self.data = bpy.data.cameras.new("test_camera")
        self.camera = bpy.data.objects.new("test_camera", self.data)
        self.scene.collection.objects.link(self.camera)

    def tearDown(self):
        self.scene.camera = self.previous_camera
        bpy.data.objects.remove(self.camera, do_unlink=True)
        bpy.data.cameras.remove(self.data)

    def configure(self, profile):
        configure_camera(self.scene, self.camera, profile, near_clip=0.1, far_clip=100)
        bpy.context.view_layer.update()

    def test_projection_matches_math_for_aspects_and_up_directions(self):
        profiles = (
            PerspectiveCamera(),
            PerspectiveCamera(width=768, height=384),
            PerspectiveCamera(width=320, height=640, vertical_fov_degrees=60),
            PerspectiveCamera(position=(-3, 1, 2), target=(0.1, 0.2, -0.1),
                              up_reference=(1, 0, 1), vertical_fov_degrees=70),
        )
        points = ((0, 0, 0), (0.4, -0.3, 0.2), (-0.4, 0.3, -0.2), (0, 0, 0.5))
        for profile in profiles:
            with self.subTest(profile=profile):
                self.configure(profile)
                for point in points:
                    ndc = world_to_camera_view(self.scene, self.camera, Vector(point))
                    u, v, depth = project_point(point, profile)
                    self.assertAlmostEqual(ndc.x * profile.width, u, delta=0.001)
                    self.assertAlmostEqual((1 - ndc.y) * profile.height, v, delta=0.001)
                    self.assertAlmostEqual(ndc.z, depth, delta=1e-5)

    def test_axis_camera_has_expected_image_directions(self):
        self.configure(PerspectiveCamera(position=(0, 0, 4), up_reference=(0, 1, 0),
                                         vertical_fov_degrees=90))
        # z=0 时深度为 4；90 度方幅下，x/y 的一个单位对应 64 像素。
        for point, expected in (((0, 0, 0), (256, 256)),
                                ((1, 0, 0), (320, 256)),
                                ((0, 1, 0), (256, 192))):
            ndc = world_to_camera_view(self.scene, self.camera, Vector(point))
            self.assertAlmostEqual(ndc.x * 512, expected[0], delta=0.001)
            self.assertAlmostEqual((1 - ndc.y) * 512, expected[1], delta=0.001)

    def test_reconfiguration_resets_capture_settings_and_preserves_other_objects(self):
        self.data.type = "ORTHO"
        self.data.shift_x, self.data.shift_y = 0.2, -0.3
        self.data.dof.use_dof = True
        render = self.scene.render
        render.resolution_percentage = 50
        render.pixel_aspect_x, render.pixel_aspect_y = 2, 3
        render.use_border = render.use_crop_to_border = True
        before = {obj.name: obj.matrix_world.copy() for obj in self.scene.objects
                  if obj != self.camera}
        object_count = len(bpy.data.objects)
        self.configure(PerspectiveCamera(width=800, height=400))
        self.configure(PerspectiveCamera())
        self.assertEqual(len(bpy.data.objects), object_count)
        self.assertEqual(self.scene.camera, self.camera)
        self.assertEqual(self.data.type, "PERSP")
        self.assertEqual(self.data.sensor_fit, "VERTICAL")
        self.assertEqual((self.data.shift_x, self.data.shift_y), (0, 0))
        self.assertFalse(self.data.dof.use_dof)
        self.assertAlmostEqual(self.data.clip_start, 0.1)
        self.assertAlmostEqual(self.data.clip_end, 100)
        self.assertEqual((render.resolution_x, render.resolution_y), (512, 512))
        self.assertEqual(render.resolution_percentage, 100)
        self.assertEqual((render.pixel_aspect_x, render.pixel_aspect_y), (1, 1))
        self.assertFalse(render.use_border or render.use_crop_to_border)
        for name, transform in before.items():
            self.assertEqual(self.scene.objects[name].matrix_world, transform)

    def test_undefined_camera_frame_is_rejected(self):
        for profile in (replace(PerspectiveCamera(), position=(0, 0, 0)),
                        replace(PerspectiveCamera(), up_reference=(2, 2, 2)),
                        replace(PerspectiveCamera(), up_reference=(0, 0, 0))):
            with self.subTest(profile=profile), self.assertRaises(ValueError):
                self.configure(profile)


if __name__ == "__main__":
    unittest.main(argv=[__file__], verbosity=2)
