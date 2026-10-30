"""在 Blender 内构建、检查、拍摄独立环境，不导入或摆放 CAD。

例如：blender --background --python room_tools.py -- build
      --template_dir /path/template --output_dir /path/template/build
"""

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys
import time
import traceback

import bpy
from mathutils import Vector


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _node_trees():
    """材质、世界和嵌套节点组共享同一份遍历，避免重复统计。"""
    roots = [item.node_tree for item in (*bpy.data.materials, *bpy.data.worlds) if item.use_nodes]
    roots.extend(bpy.data.node_groups)
    for scene in bpy.data.scenes:
        roots.append(getattr(scene, "compositing_node_group", None))
        roots.append(getattr(scene, "node_tree", None))
    visited = set()
    while roots:
        tree = roots.pop()
        if tree is None or tree.as_pointer() in visited:
            continue
        visited.add(tree.as_pointer())
        yield tree
        roots.extend(node.node_tree for node in tree.nodes if node.type == "GROUP" and node.node_tree)


def dependencies(template_dir: Path | None = None) -> list[dict]:
    """检查实际文件依赖；历史素材库的 weak reference 不属于渲染依赖。

    UDIM 根据图像登记的瓦片逐一检查，不能仅用 glob 找到一张就认为完整。
    已打包图片无需外部文件；保留其记录，但不将原始文件路径当作缺失。
    """
    entries = []
    for collection in ("images", "libraries", "fonts", "movieclips", "sounds", "cache_files", "volumes"):
        for block in getattr(bpy.data, collection, []):
            filepath = getattr(block, "filepath", "")
            if not filepath or filepath == "<builtin>":
                continue
            if collection == "images" and block.source in {"GENERATED", "VIEWER"}:
                continue
            packed_files = list(getattr(block, "packed_files", []))
            packed = bool(getattr(block, "packed_file", None) or packed_files)
            absolute = Path(bpy.path.abspath(filepath, library=block.library)).resolve()
            tiles = [tile.number for tile in block.tiles] if collection == "images" and block.source == "TILED" else [None]
            for tile in tiles:
                filename = str(absolute)
                if tile is not None:
                    u, v = (tile - 1001) % 10 + 1, (tile - 1001) // 10 + 1
                    filename = filename.replace("<UDIM>", str(tile)).replace("<UVTILE>", f"u{u}_v{v}")
                tile_packed = any(item.tile_number == tile for item in packed_files) if tile is not None else packed
                path = Path(filename)
                entries.append({
                    "kind": collection, "name": block.name, "path": str(path), "tile": tile,
                    "packed": tile_packed, "exists": tile_packed or path.is_file(),
                    "inside_template": tile_packed or (path.is_relative_to(template_dir) if template_dir else None),
                })
    return entries


def scene_inventory() -> dict:
    """导出可阅读的对象、材质、灯光和成像设置，不输出顶点数组。"""
    scene = bpy.context.scene
    objects, lights = [], []
    for obj in sorted(scene.objects, key=lambda item: item.name):
        bounds = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
        entry = {
            "name": obj.name, "type": obj.type, "parent": obj.parent.name if obj.parent else None,
            "position": list(obj.matrix_world.translation), "dimensions": list(obj.dimensions),
            "rotation_euler": list(obj.rotation_euler), "scale": list(obj.scale),
            "hide_render": obj.hide_render, "collections": [group.name for group in obj.users_collection],
            "materials": [slot.material.name if slot.material else None for slot in obj.material_slots],
        }
        if obj.type == "MESH":
            entry.update(vertices=len(obj.data.vertices), polygons=len(obj.data.polygons),
                         bounds_min=[min(p[i] for p in bounds) for i in range(3)],
                         bounds_max=[max(p[i] for p in bounds) for i in range(3)])
        objects.append(entry)
        if obj.type == "LIGHT":
            light = obj.data
            lights.append(dict(name=obj.name, type=light.type, energy=light.energy,
                               color=list(light.color), position=list(obj.matrix_world.translation),
                               matrix_world=[list(row) for row in obj.matrix_world],
                               size=getattr(light, "size", None), size_y=getattr(light, "size_y", None)))
    materials = []
    for material in bpy.data.materials:
        nodes = list(material.node_tree.nodes) if material.use_nodes else []
        materials.append({
            "name": material.name, "users": material.users, "use_nodes": material.use_nodes,
            "node_types": sorted(set(node.bl_idname for node in nodes)),
            "images": [node.image.name for node in nodes if getattr(node, "image", None)],
        })
    return {
        "blend_file": bpy.data.filepath, "blender_version": bpy.app.version_string,
        "units": {"system": scene.unit_settings.system, "scale_length": scene.unit_settings.scale_length},
        "objects": objects, "materials": materials, "lights": lights,
        "world": {"name": scene.world.name, "use_nodes": scene.world.use_nodes,
                  "node_types": [node.bl_idname for node in scene.world.node_tree.nodes] if scene.world.use_nodes else []} if scene.world else None,
        "color_management": {"view_transform": scene.view_settings.view_transform,
                             "look": scene.view_settings.look, "exposure": scene.view_settings.exposure,
                             "gamma": scene.view_settings.gamma},
        "render": {"engine": scene.render.engine, "samples": scene.cycles.samples,
                   "resolution": [scene.render.resolution_x, scene.render.resolution_y]},
        "dependencies": dependencies(),
    }


def build(template_dir: Path, output_dir: Path) -> dict:
    """从空文件执行构建脚本，保存环境与依赖报告；不覆盖模板的源 blend。"""
    template_dir, output_dir = template_dir.resolve(), output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    blend_file = output_dir / "room.blend"
    if blend_file.exists():
        raise FileExistsError(f"Build output already exists: {blend_file}")
    report = {"status": "failed", "blend_file": str(blend_file),
              "inventory_file": str(output_dir / "scene_inventory.json"),
              "blender_version": bpy.app.version_string, "error": None}
    previous_cwd = Path.cwd()
    try:
        bpy.ops.wm.read_factory_settings(use_empty=True)
        os.chdir(template_dir)
        sys.path.insert(0, str(template_dir))
        specification = importlib.util.spec_from_file_location("room_template", template_dir / "build_scene.py")
        module = importlib.util.module_from_spec(specification)
        sys.modules[specification.name] = module
        specification.loader.exec_module(module)
        module.build_scene(template_dir)
        bpy.context.view_layer.update()
        # 先按源 blend 解析相对路径。保存到新位置后再相对化，避免路径基准漂移。
        bpy.ops.file.make_paths_absolute()
        items = dependencies(template_dir)
        if bpy.data.filepath:
            source_blend = Path(bpy.data.filepath).resolve()
            report["source_blend"] = str(source_blend)
            items.append({"kind": "source_blend", "name": source_blend.name, "path": str(source_blend),
                          "packed": False, "exists": source_blend.is_file(),
                          "inside_template": source_blend.is_relative_to(template_dir)})
        report.update(dependencies=items,
                      missing_dependencies=[item for item in items if not item["exists"]],
                      external_dependencies=[item for item in items if not item["inside_template"]])
        if report["missing_dependencies"] or report["external_dependencies"]:
            raise ValueError("Template has missing or external dependencies; see build_report.json")
        bpy.context.preferences.filepaths.save_version = 0
        bpy.ops.wm.save_as_mainfile(filepath=str(blend_file), relative_remap=False, check_existing=False)
        bpy.ops.file.make_paths_relative()
        bpy.ops.wm.save_as_mainfile(filepath=str(blend_file), relative_remap=False, check_existing=False)
        _write_json(output_dir / "scene_inventory.json", scene_inventory())
        report["status"] = "pass"
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        traceback.print_exc()
    finally:
        os.chdir(previous_cwd)
        _write_json(output_dir / "build_report.json", report)
    return report


def _configure_device(scene, device: str, gpu_index: int) -> list[str]:
    """显式选择一张设备，避免继承参考工程作者的显卡配置。"""
    scene.render.engine = "CYCLES"
    if device == "CPU":
        scene.cycles.device = "CPU"
        return ["CPU"]
    preferences = bpy.context.preferences.addons["cycles"].preferences
    preferences.compute_device_type = device
    preferences.get_devices()
    devices = [item for item in preferences.devices if item.type == device]
    if not 0 <= gpu_index < len(devices):
        raise ValueError(f"{device} GPU index {gpu_index} unavailable; found {len(devices)} GPUs")
    for item in preferences.devices:
        item.use = False
    devices[gpu_index].use = True
    scene.cycles.device = "GPU"
    return [devices[gpu_index].name]


def _set_camera(scene, camera, view: dict) -> dict:
    """按米制位置与垂直视场角设置观察相机，记录实际投影矩阵。"""
    width, height = view["resolution"]
    fov = float(view["vertical_fov_degrees"])
    position, target = Vector(view["position"]), Vector(view["target"])
    if not all(math.isfinite(value) for value in (*position, *target, fov)):
        raise ValueError("Camera values must be finite")
    if width <= 0 or height <= 0 or not 0 < fov < 180 or (target - position).length == 0:
        raise ValueError("Invalid resolution, field of view or camera direction")
    camera.location = position
    camera.rotation_euler = (target - position).to_track_quat("-Z", "Y").to_euler()
    camera.data.type = "PERSP"
    camera.data.sensor_fit = "VERTICAL"
    camera.data.sensor_height = 24.0
    camera.data.lens = 12.0 / math.tan(math.radians(fov) / 2)
    camera.data.clip_start, camera.data.clip_end = 0.01, 1000.0
    camera.data.dof.use_dof = False
    scene.camera = camera
    scene.render.resolution_x, scene.render.resolution_y = width, height
    scene.render.resolution_percentage = 100
    scene.render.pixel_aspect_x = scene.render.pixel_aspect_y = 1.0
    bpy.context.view_layer.update()
    projection = camera.calc_matrix_camera(bpy.context.evaluated_depsgraph_get(), x=width, y=height)
    return dict(view, matrix_world=[list(row) for row in camera.matrix_world],
                matrix_world_inverse=[list(row) for row in camera.matrix_world.inverted()],
                projection_matrix=[list(row) for row in projection], lens_mm=camera.data.lens,
                clip_start=camera.data.clip_start, clip_end=camera.data.clip_end, dof=False)


def render(blend: Path, views_file: Path, output_dir: Path, samples: int = 128,
           seed: int = 8, device: str = "OPTIX", gpu_index: int = 0) -> dict:
    """一次打开场景，完成全部视角；失败项保留在预定分母中。"""
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {"status": "failed", "planned": 0, "success": 0, "failed": 0, "results": [],
              "blend_file": str(blend.resolve()), "views_file": str(views_file.resolve()),
              "samples": samples, "seed": seed, "device": device, "gpu_index": gpu_index,
              "blender_version": bpy.app.version_string, "error": None}
    start = time.monotonic()
    try:
        views = json.loads(views_file.read_text(encoding="utf-8"))["views"]
        report["planned"] = len(views)
        names = [view["id"] for view in views]
        if not views or len(set(names)) != len(names) or any(not re.fullmatch(r"[A-Za-z0-9_-]+", name) for name in names):
            raise ValueError("Views require unique, nonempty filename-safe IDs")
        if samples <= 0:
            raise ValueError("samples must be positive")
        if any((output_dir / f"{name}.png").exists() for name in names):
            raise FileExistsError("Render output contains an existing requested image")
        bpy.ops.wm.open_mainfile(filepath=str(blend.resolve()))
        scene = bpy.context.scene
        report["devices"] = _configure_device(scene, device, gpu_index)
        report["color_management"] = scene_inventory()["color_management"]
        scene.cycles.samples, scene.cycles.seed = samples, seed
        scene.cycles.use_animated_seed = False
        # 5.2 的多层 EXR 工程需先切换媒体类型，才允许选择 PNG。
        scene.render.image_settings.media_type = "IMAGE"
        scene.render.image_settings.file_format = "PNG"
        scene.render.image_settings.color_mode = "RGB"
        scene.render.image_settings.color_depth = "8"
        scene.render.use_file_extension = True
        scene.render.use_multiview = False
        scene.render.use_border = False
        scene.render.film_transparent = False
        # 保留合成调色，但移除向任意外部路径写文件的输出节点。
        for tree in _node_trees():
            for node in list(tree.nodes):
                if node.bl_idname == "CompositorNodeOutputFile":
                    tree.nodes.remove(node)
        camera = bpy.data.objects.new("room_preview_camera", bpy.data.cameras.new("room_preview_camera"))
        scene.collection.objects.link(camera)
        for view in views:
            image_file = output_dir / f"{view['id']}.png"
            result = {"id": view["id"], "image_file": str(image_file), "status": "failed", "error": None}
            try:
                result["camera"] = _set_camera(scene, camera, view)
                _write_json(output_dir / f"{view['id']}.camera.json", result["camera"])
                scene.render.filepath = str(image_file)
                bpy.ops.render.render(write_still=True)
                if not image_file.is_file():
                    raise RuntimeError("Blender returned without writing the requested PNG")
                result["status"] = "success"
            except Exception as error:
                result["error"] = f"{type(error).__name__}: {error}"
                traceback.print_exc()
            report["results"].append(result)
            report["success"] = sum(item["status"] == "success" for item in report["results"])
            report["failed"] = report["planned"] - report["success"]
            _write_json(output_dir / "render_report.json", report)
        report["status"] = "pass" if report["failed"] == 0 else "failed"
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        report["failed"] = report["planned"] - report["success"]
        traceback.print_exc()
    report["elapsed_seconds"] = time.monotonic() - start
    _write_json(output_dir / "render_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser("build", help="执行模板并保存 room.blend")
    build_parser.add_argument("--template_dir", type=Path, required=True)
    build_parser.add_argument("--output_dir", type=Path, required=True)
    render_parser = subparsers.add_parser("render", help="拍摄 views.json 指定的环境视角")
    render_parser.add_argument("--blend", type=Path, required=True)
    render_parser.add_argument("--views_file", type=Path, required=True)
    render_parser.add_argument("--output_dir", type=Path, required=True)
    render_parser.add_argument("--samples", type=int, default=128)
    render_parser.add_argument("--seed", type=int, default=8)
    render_parser.add_argument("--device", choices=("CPU", "CUDA", "OPTIX"), default="OPTIX")
    render_parser.add_argument("--gpu_index", type=int, default=0)
    inspect_parser = subparsers.add_parser("inspect", help="只读导出场景概要")
    inspect_parser.add_argument("--blend", type=Path, required=True)
    inspect_parser.add_argument("--output_dir", type=Path, required=True)
    arguments = vars(parser.parse_args(sys.argv[sys.argv.index("--") + 1:]))
    command = arguments.pop("command")
    if command == "inspect":
        bpy.ops.wm.open_mainfile(filepath=str(arguments["blend"].resolve()))
        arguments["output_dir"].mkdir(parents=True, exist_ok=True)
        _write_json(arguments["output_dir"] / "scene_inventory.json", scene_inventory())
    else:
        report = build(**arguments) if command == "build" else render(**arguments)
        if report["status"] != "pass":
            raise RuntimeError(f"{command} failed; see {arguments['output_dir']}")


if __name__ == "__main__":
    main()
