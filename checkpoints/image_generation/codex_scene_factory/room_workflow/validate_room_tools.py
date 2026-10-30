"""用真实 Blender 验证环境工具；全部 blend、图片和日志写入外部新目录。

不调用 Codex，不修改 Shed，不把小场景通过解释为真实感通过。
"""

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time

from PIL import Image


FIXTURE = '''import bpy
from pathlib import Path

directory = Path(__file__).parent / "template" / "assets"
bpy.ops.wm.read_factory_settings(use_empty=True)
for tile in (1001, 1002):
    image = bpy.data.images.new(f"tile{tile}", width=4, height=4)
    image.generated_color = (0.5, 0.25, 0.1, 1)
    image.filepath_raw = str(directory / f"surface.{tile}.png")
    image.file_format = "PNG"
    image.save()
    bpy.data.images.remove(image)
image = bpy.data.images.load(str(directory / "surface.1001.png"))
image.source = "TILED"
image.filepath = "//surface.<UDIM>.png"
image.tiles.new(1002)
material = bpy.data.materials.new("PBR")
material.use_nodes = True
texture = material.node_tree.nodes.new("ShaderNodeTexImage")
texture.image = image
shader = material.node_tree.nodes.get("Principled BSDF")
material.node_tree.links.new(texture.outputs["Color"], shader.inputs["Base Color"])
bpy.ops.mesh.primitive_cube_add()
bpy.context.object.data.materials.append(material)
bpy.ops.object.light_add(type="AREA", location=(1, 1, 4))
bpy.context.object.data.energy = 400
world = bpy.data.worlds.new("World")
world.use_nodes = True
bpy.context.scene.world = world
bpy.context.scene.view_settings.exposure = 0.3
bpy.context.scene.render.image_settings.media_type = "MULTI_LAYER_IMAGE"
bpy.context.scene.render.image_settings.file_format = "OPEN_EXR_MULTILAYER"
bpy.ops.wm.save_as_mainfile(filepath=str(directory / "base.blend"))
'''

BUILDER = '''from pathlib import Path
import bpy


def build_scene(template_dir: Path):
    bpy.ops.wm.open_mainfile(filepath=str(template_dir / "assets" / "base.blend"))
'''


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    package = Path(__file__).resolve().parents[4] / "packages" / "cad_scene_factory"
    parser.add_argument("--blender", type=Path, required=True)
    parser.add_argument("--room_tools", type=Path, default=package / "room_tools.py")
    parser.add_argument("--workflow", type=Path, default=package / "room_workflow.py")
    parser.add_argument("--shed_blend", type=Path)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--results_file", type=Path)
    args = parser.parse_args()
    directory = args.output_dir.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    results = {"status": "running", "scope": "真实 Blender 工具及六视角接口；不评价场景真实感",
               "output_dir": str(directory), "checks": [],
               "room_tools_sha256": digest(args.room_tools), "workflow_sha256": digest(args.workflow)}

    def check(name: str, condition: bool, evidence=None):
        results["checks"].append({"name": name, "pass": bool(condition), "evidence": evidence})
        if not condition:
            raise AssertionError(f"{name}: {evidence}")

    def blender(label: str, *arguments: str, expected_code: int = 0, script: Path | None = None):
        command = [str(args.blender), "--background", "--threads", "2", "--python-exit-code", "1",
                   "--python", str(script or args.room_tools)]
        if arguments:
            command += ["--", *arguments]
        write_json(directory / f"{label}.command.json", command)
        with (directory / f"{label}.log").open("w", encoding="utf-8") as log:
            process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, text=True)
        check(f"{label}_exit_code", process.returncode == expected_code, process.returncode)

    try:
        template = directory / "template"
        (template / "assets").mkdir(parents=True)
        (template / "build_scene.py").write_text(BUILDER, encoding="utf-8")
        (directory / "create_fixture.py").write_text(FIXTURE, encoding="utf-8")
        views = {"views": [
            {"id": "landscape", "position": [3, 3, 2], "target": [0, 0, 0],
             "vertical_fov_degrees": 45, "resolution": [64, 48]},
            {"id": "portrait", "position": [3, 3, 2], "target": [0, 0, 0],
             "vertical_fov_degrees": 45, "resolution": [48, 64]},
        ]}
        write_json(template / "views.json", views)
        blender("create_fixture", script=directory / "create_fixture.py")
        blender("build", "build", "--template_dir", str(template), "--output_dir", str(template / "build"))
        report = read_json(template / "build" / "build_report.json")
        check("build_udim", report["status"] == "pass" and not report["missing_dependencies"],
              [item["tile"] for item in report["dependencies"] if "tile" in item])

        # 移走原路径，使旧绝对路径不可能悄悄帮助搬移测试通过。
        moved = directory / "moved"
        shutil.copytree(template, moved)
        template.rename(directory / "original_hidden")
        blender("render_moved", "render", "--blend", str(moved / "build" / "room.blend"),
                "--views_file", str(moved / "views.json"), "--output_dir", str(directory / "renders"),
                "--samples", "1", "--device", "CPU")
        report = read_json(directory / "renders" / "render_report.json")
        check("moved_render", (report["planned"], report["success"], report["failed"]) == (2, 2, 0))
        for item in report["results"]:
            camera = item["camera"]
            fov = math.degrees(2 * math.atan(1 / camera["projection_matrix"][1][1]))
            check(f"{item['id']}_vertical_fov", abs(fov - 45) < 1e-5, fov)
            with Image.open(item["image_file"]) as image:
                check(f"{item['id']}_image_size", list(image.size) == camera["resolution"], list(image.size))
        check("source_exposure_preserved", abs(report["color_management"]["exposure"] - 0.3) < 1e-6)
        blender("rebuild_moved", "build", "--template_dir", str(moved), "--output_dir", str(moved / "rebuild"))

        missing = directory / "missing"
        shutil.copytree(moved, missing)
        (missing / "assets" / "surface.1002.png").unlink()
        blender("missing_tile", "build", "--template_dir", str(missing), "--output_dir", str(missing / "failed_build"), expected_code=1)
        report = read_json(missing / "failed_build" / "build_report.json")
        check("missing_udim_rejected", len(report["missing_dependencies"]) == 1
              and report["missing_dependencies"][0]["tile"] == 1002)

        external = directory / "external"
        external.mkdir()
        (external / "build_scene.py").write_text(
            "import bpy\ndef build_scene(template_dir):\n"
            f"    bpy.ops.wm.open_mainfile(filepath={str(moved / 'assets' / 'base.blend')!r})\n")
        blender("external_dependency", "build", "--template_dir", str(external), "--output_dir", str(external / "build"), expected_code=1)
        report = read_json(external / "build" / "build_report.json")
        check("external_dependency_rejected", bool(report["external_dependencies"]))

        views["views"][0]["position"] = views["views"][0]["target"]
        write_json(directory / "partial_views.json", views)
        blender("partial_render", "render", "--blend", str(moved / "build" / "room.blend"),
                "--views_file", str(directory / "partial_views.json"), "--output_dir", str(directory / "partial"),
                "--samples", "1", "--device", "CPU", expected_code=1)
        report = read_json(directory / "partial" / "render_report.json")
        check("partial_failure_denominator", (report["planned"], report["success"], report["failed"]) == (2, 1, 1))

        # 通过普通 Python 的真实 preview_room 接口，核对报告字段及绝对图片路径。
        sys.path.insert(0, str(args.workflow.parent))
        spec = importlib.util.spec_from_file_location("validation_room_workflow", args.workflow)
        workflow = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(workflow)
        six_views = {"views": [dict(views["views"][1], id=name) for name in sorted(workflow.VIEW_IDS)]}
        write_json(moved / "views.json", six_views)
        write_json(moved / "assets.json", {"files": []})
        (moved / "scene_plan.md").write_text("工具接口验证；不代表正式资产清单或视觉质量。\n", encoding="utf-8")
        request = {"output_dir": str(moved), "samples": 1,
                   "runtime": {"blender": str(args.blender), "room_tools": str(args.room_tools),
                               "seed": 8, "device": "CPU", "gpu_index": 0}}
        report = workflow.preview_room(request, directory / "workflow_preview")
        check("workflow_six_view_contract", report["status"] == "pass" and report["success"] == 6,
              {key: report[key] for key in ("status", "planned", "success", "failed", "errors")})

        six_views["views"][0]["position"] = six_views["views"][0]["target"]
        write_json(moved / "views.json", six_views)
        report = workflow.preview_room(request, directory / "workflow_partial")
        check("workflow_partial_failure_details", report["status"] == "failed"
              and (report["planned"], report["success"], report["failed"]) == (6, 5, 1)
              and "render" in report)
        for name in ("views.json", "assets.json", "scene_plan.md"):
            shutil.copyfile(moved / name, missing / name)
        request["output_dir"] = str(missing)
        report = workflow.preview_room(request, directory / "workflow_missing")
        check("workflow_missing_dependency_details", report["status"] == "failed"
              and len(report.get("build", {}).get("missing_dependencies", [])) == 1)

        if args.shed_blend:
            before = digest(args.shed_blend)
            blender("shed_inspect", "inspect", "--blend", str(args.shed_blend), "--output_dir", str(directory / "shed_inspect"))
            inventory = read_json(directory / "shed_inspect" / "scene_inventory.json")
            check("shed_readonly", digest(args.shed_blend) == before, before)
            check("shed_dependencies_present", all(item["exists"] for item in inventory["dependencies"]))
            results["shed"] = {"sha256": before, "blender_version": inventory["blender_version"],
                               **{key: len(inventory[key]) for key in ("objects", "materials", "lights", "dependencies")}}
        results["status"] = "pass"
    except Exception as error:
        results.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        results["elapsed_seconds"] = time.monotonic() - started
        write_json(directory / "results.json", results)
        if args.results_file:
            write_json(args.results_file, results)
        print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
