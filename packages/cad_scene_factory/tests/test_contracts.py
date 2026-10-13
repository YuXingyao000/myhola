"""模板清单的真实示例与往返保存；不执行示例中的文件引用。"""

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from cad_scene_factory.contracts import FloatParameter, load_template, save_template
from cad_scene_factory.geometry import BoundingBox, PerspectiveCamera, project_point


EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "template_manifest.json"


class TemplateManifestTests(unittest.TestCase):
    def test_example_restores_geometry_and_parameter_types(self):
        template = load_template(EXAMPLE)
        self.assertEqual(template.target_envelope,
                         BoundingBox((-0.5, -0.5, 0.0), (0.5, 0.5, 1.0)))
        self.assertEqual(template.placement_range, (-0.5, 0.0))
        self.assertEqual(template.camera_profile, PerspectiveCamera())
        self.assertEqual(project_point((0, 0, 0), template.camera_profile)[:2], (256, 256))
        self.assertEqual(template.parameters["table_roughness"], FloatParameter(0.4, 0.2, 0.6))

    def test_round_trip_preserves_custom_profile_and_relative_references(self):
        source = load_template(EXAMPLE)
        camera = replace(source.camera_profile, position=(3, -2, 1), up_reference=(0, 1, 0),
                         vertical_fov_degrees=60, width=640, height=320)
        template = replace(source, template_id="车间示例", version=2, camera_profile=camera,
                           builder="scripts/build_scene.py", parameters={})
        with TemporaryDirectory() as directory:
            path = Path(directory) / "template.json"
            save_template(template, path)
            self.assertEqual(load_template(path), template)
            self.assertIn("车间示例", path.read_text(encoding="utf-8"))

    def test_save_does_not_overwrite_existing_manifest(self):
        template = load_template(EXAMPLE)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "template.json"
            save_template(template, path)
            with self.assertRaises(FileExistsError):
                save_template(replace(template, version=2), path)
            self.assertEqual(load_template(path), template)


if __name__ == "__main__":
    unittest.main()
