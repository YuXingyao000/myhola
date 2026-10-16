"""在真实 Blender 中构建手写模板；运行命令见 BLENDER_BUILD.md。"""

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import bpy
from mathutils import Matrix, Vector

from cad_scene_factory.blender.build import build_template
from cad_scene_factory.contracts import load_template, save_template
from cad_scene_factory.geometry import BoundingBox, plan_support_placement


# 手写测试模板：桌面、分组内的道具和灯光；不需要外部模型或素材。
BUILDER_SOURCE = '''
from __future__ import annotations
from dataclasses import dataclass
import random
import bpy


@dataclass
class TableFinish:
    roughness: float


def build_scene(*, scene, environment_root, assets_directory, parameters, seed):
    rng = random.Random(seed)
    finish = TableFinish(roughness=parameters["table_roughness"])
    material = bpy.data.materials.new("table_material")
    material.use_nodes = True
    shader = material.node_tree.nodes["Principled BSDF"]
    shader.inputs["Roughness"].default_value = finish.roughness

    bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, -0.1))
    support = bpy.context.object
    support.name = "workbench_top"
    support.dimensions = (4, 4, 0.2)
    support.parent = environment_root
    support.data.materials.append(material)

    group = bpy.data.objects.new("props", None)
    scene.collection.objects.link(group)
    group.parent = environment_root
    group.location = (1.5, 0, 0)
    bpy.ops.mesh.primitive_cube_add(size=0.2)
    prop = bpy.context.object
    prop.name = "prop"
    prop.parent = group
    prop.location = (0, rng.uniform(-0.5, 0.5), 0.1)

    light_data = bpy.data.lights.new("key_light", "AREA")
    light = bpy.data.objects.new("key_light", light_data)
    scene.collection.objects.link(light)
    light.parent = environment_root
    light.location = (1, 1, 3)
    light_data.energy = float((assets_directory / "light_energy.txt").read_text())

    scene.world = bpy.data.worlds.new("template_world")
    scene.world.color = (0.1, 0.1, 0.1)
'''


class BlenderBuildTests(unittest.TestCase):
    def setUp(self):
        bpy.ops.wm.read_factory_settings(use_empty=True)
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        directory = Path(temporary.name)
        self.path = directory / "template.json"
        self.builder_path = directory / "build_scene.py"
        self.builder_path.write_text(BUILDER_SOURCE, encoding="utf-8")
        assets = directory / "assets"
        assets.mkdir()
        (assets / "light_energy.txt").write_text("350", encoding="utf-8")
        example = Path(__file__).resolve().parents[2] / "examples/template_manifest.json"
        self.template = load_template(example)
        save_template(self.template, self.path)

    def test_build_uses_manifest_parameters_assets_and_hierarchy(self):
        result = build_template(self.path, seed=17, overrides={"table_roughness": 0.55})
        self.assertEqual(result.template, self.template)
        self.assertEqual(result.parameters, {"table_roughness": 0.55})
        self.assertEqual(result.seed, 17)
        self.assertEqual(result.environment_root.matrix_world, Matrix.Identity(4))
        self.assertEqual(result.environment_root.type, "EMPTY")
        self.assertEqual(result.support_object.name, "workbench_top")
        self.assertEqual(result.support_object.parent, result.environment_root)
        material = result.support_object.data.materials[0]
        roughness = material.node_tree.nodes["Principled BSDF"].inputs["Roughness"]
        self.assertAlmostEqual(roughness.default_value, 0.55)
        objects = {obj.name: obj for obj in result.environment_root.children_recursive}
        self.assertEqual(set(objects), {"workbench_top", "props", "prop", "key_light"})
        self.assertEqual(objects["prop"].parent, objects["props"])
        self.assertEqual(objects["key_light"].data.energy, 350)
        self.assertIsNone(bpy.context.scene.camera)

    def test_seed_reproduces_template_randomness(self):
        positions = []
        for seed in (17, 17, 18):
            bpy.ops.wm.read_factory_settings(use_empty=True)
            build_template(self.path, seed=seed)
            positions.append(tuple(bpy.context.scene.objects["prop"].location))
        self.assertEqual(positions[0], positions[1])
        self.assertNotEqual(positions[0], positions[2])

    def test_placement_moves_nested_objects_and_lights_together(self):
        result = build_template(self.path, seed=17)
        before = {obj.name: obj.matrix_world.translation.copy()
                  for obj in result.environment_root.children_recursive}
        world = bpy.context.scene.world
        world_color = tuple(world.color)
        bounds = BoundingBox((-0.5, -0.5, -0.35), (0.5, 0.5, 0.5))
        placement = plan_support_placement(bounds)
        delta = Vector(placement.environment_translation)
        result.environment_root.location = delta
        bpy.context.view_layer.update()
        for obj in result.environment_root.children_recursive:
            with self.subTest(object=obj.name):
                self.assertLess((obj.matrix_world.translation - before[obj.name] - delta).length,
                                1e-6)
        top = max((result.support_object.matrix_world @ Vector(corner)).z
                  for corner in result.support_object.bound_box)
        self.assertAlmostEqual(top, bounds.minimum[2], places=6)
        self.assertEqual(bpy.context.scene.world, world)
        self.assertEqual(tuple(world.color), world_color)

    def test_invalid_parameters_are_rejected_before_executing_source(self):
        self.builder_path.write_text('raise RuntimeError("builder was executed")\n',
                                     encoding="utf-8")
        for overrides, error in (({"unknown": 1}, KeyError),
                                 ({"table_roughness": 1.5}, ValueError)):
            with self.subTest(overrides=overrides), self.assertRaises(error):
                build_template(self.path, seed=17, overrides=overrides)
            self.assertEqual(len(bpy.context.scene.objects), 0)

    def test_nonempty_scene_is_preserved_and_rejected(self):
        existing = bpy.data.objects.new("existing", None)
        bpy.context.scene.collection.objects.link(existing)
        existing.location = (1, 2, 3)
        with self.assertRaisesRegex(ValueError, "empty scene"):
            build_template(self.path, seed=17)
        self.assertEqual(list(bpy.context.scene.objects), [existing])
        self.assertEqual(tuple(existing.location), (1, 2, 3))

    def test_support_lookup_accepts_descendants_and_rejects_unparented_objects(self):
        # prop 是二级后代：本片只定位引用，几何是否适合作为支撑另行检查。
        nested_path = self.path.with_name("nested.json")
        save_template(replace(self.template, support_object="prop"), nested_path)
        result = build_template(nested_path, seed=17)
        self.assertEqual(result.support_object.name, "prop")
        self.assertNotEqual(result.support_object.parent, result.environment_root)

        bpy.ops.wm.read_factory_settings(use_empty=True)
        self.builder_path.write_text(BUILDER_SOURCE + '\n    support.parent = None\n',
                                     encoding="utf-8")
        with self.assertRaises(KeyError):
            build_template(self.path, seed=17)


if __name__ == "__main__":
    unittest.main(argv=[__file__], verbosity=2)
