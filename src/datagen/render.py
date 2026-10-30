"""在 Blender 内打开 .blend，逐件导入归一化的 STL/OBJ 并渲染。

输入网格已经使用右手坐标系、Z 向上；此脚本不归一化或改变 CAD 姿态。
环境按每个零件的最低点移动。输出图片及对应的输入、相机和 seed 记录。
"""

import argparse
import json
import math
from pathlib import Path
import sys
import traceback

import bpy
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))
from geometry import BoundingBox, PerspectiveCamera, check_bbox_framing
from visibility import find_blockers


SUPPORT_CONTACT_TOLERANCE = 1e-6
# STL 没有 B-Rep 曲面标签。暂用夹角区分曲面离散边与硬边，并记录在输出中。
STL_SMOOTH_ANGLE_DEGREES = 30.0


def render(
    model_files: list[Path],
    scene_file_path: Path,
    output_dir: Path,
    *,
    samples: int = 64,
    seed: int = 0,
    device: str = "OPTIX",
    gpu_index: int = 0,
    model_seeds: list[int] | None = None,
    continue_on_error: bool = False,
    empty_heights: list[float] | None = None,
) -> None:
    """一个场景打开一次，循环替换零件。output_dir 使用本次任务的新目录。

    model_seeds 用于固定预览组，让 quick 和 review 中同一件零件使用相同 seed。
    continue_on_error 会完成后续样本，最后仍以错误退出，失败不会被当作成功。
    empty_heights 额外渲染没有 CAD 的环境，用于观察支撑高度变化后的背景。
    """
    model_files = [path.resolve() for path in model_files]
    model_seeds = model_seeds if model_seeds is not None else [seed + i for i in range(len(model_files))]
    if len(model_seeds) != len(model_files):
        raise ValueError("Each model must have exactly one seed")
    scene_file_path = scene_file_path.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    # 1. 恢复 build.py 保存的环境、相机和零件材质。
    bpy.ops.wm.open_mainfile(filepath=str(scene_file_path))
    scene = bpy.context.scene
    material = bpy.data.materials[scene["part_material"]]
    environment = bpy.data.objects["environment_root"]
    camera = _camera_profile(scene)
    _configure_device(scene, device, gpu_index)
    scene.cycles.samples = samples
    scene.cycles.use_animated_seed = False

    # 保存完整输入列表；即使中途失败，也能知道哪些模型原本计划渲染。
    request = {
        "scene_file": str(scene_file_path),
        "models": [str(path) for path in model_files],
        "samples": samples,
        "seed": seed,
        "model_seeds": model_seeds,
        "continue_on_error": continue_on_error,
        "empty_heights": empty_heights or [],
        "device": device,
        "gpu_index": gpu_index,
        "support_contact_tolerance": SUPPORT_CONTACT_TOLERANCE,
        "stl_shading": {"method": "smooth_by_angle", "angle_degrees": STL_SMOOTH_ANGLE_DEGREES},
        "blender_version": bpy.app.version_string,
        "camera": {
            "position": camera.position,
            "target": camera.target,
            "up_reference": camera.up_reference,
            "vertical_fov_degrees": camera.vertical_fov_degrees,
            "width": camera.width,
            "height": camera.height,
            "near_clip": scene.camera.data.clip_start,
            "far_clip": scene.camera.data.clip_end,
        },
        "template_sources": {
            key: scene[key] for key in (
                "scene_script", "scene_script_sha256", "material_script", "material_script_sha256",
            )
        },
    }
    (output_dir / "request.json").write_text(
        json.dumps(request, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    jobs = [
        {"kind": "model", "model_file": str(path), "seed": model_seeds[index],
         "image_file": str(output_dir / f"{index:06d}_{path.stem}.png")}
        for index, path in enumerate(model_files)
    ]
    jobs.extend(
        {"kind": "empty", "environment_height": height, "seed": seed,
         "image_file": str(output_dir / f"empty_{index:03d}.png")}
        for index, height in enumerate(empty_heights or [])
    )
    failures = 0
    with (output_dir / "renders.jsonl").open("w", encoding="utf-8") as records:
        for index, job in enumerate(jobs):
            print(f"[{index + 1}/{len(jobs)}] {job['image_file']}", flush=True)
            previous_objects = set(scene.objects)
            record = dict(job, status="failed", geometry_checks="not_run")
            try:
                if job["kind"] == "model":
                    # 2. 一个 OBJ 可包含多个网格对象，统一赋予零件材质。
                    objects = _load_model(Path(job["model_file"]))
                    if not objects:
                        raise ValueError("The imported CAD contains no mesh objects")
                    for obj in objects:
                        obj.data.materials.clear()
                        obj.data.materials.append(material)
                        for polygon in obj.data.polygons:
                            polygon.material_index = 0

                    # 3. 保持 CAD 和相机不动，绝对设置环境高度。
                    bounds = _world_bounds(objects)
                    environment.location.z = bounds.minimum[2]
                    bpy.context.view_layer.update()
                    record.update(
                        bounds={"minimum": bounds.minimum, "maximum": bounds.maximum},
                        environment_height=environment.location.z,
                        geometry_checks="failed",
                    )
                    _check_visibility(scene, objects, bounds, camera)
                    record["geometry_checks"] = "pass"
                else:
                    environment.location.z = job["environment_height"]
                    bpy.context.view_layer.update()
                    record["geometry_checks"] = "not_applicable"

                # 4. 拍照。主程序随后独立检查 PNG 能否解码、尺寸是否正确。
                record["environment_height"] = environment.location.z
                scene.cycles.seed = job["seed"]
                scene.render.filepath = job["image_file"]
                bpy.ops.render.render(write_still=True)
                record["status"] = "success"
            except Exception as error:
                failures += 1
                record["error"] = f"{type(error).__name__}: {error}"
                traceback.print_exc()
                if not continue_on_error:
                    raise
            finally:
                records.write(json.dumps(record, ensure_ascii=False) + "\n")
                records.flush()
                # 5. 导入失败时也清理已经创建的网格，避免污染下一件。
                _remove_model([
                    obj for obj in scene.objects
                    if obj not in previous_objects and obj.type == "MESH"
                ])
    if failures:
        raise RuntimeError(f"{failures}/{len(jobs)} renders failed; see renders.jsonl")
    print(f"渲染完成：{output_dir}", flush=True)


def _load_model(file_path: Path) -> list[bpy.types.Object]:
    previous_objects = set(bpy.context.scene.objects)
    if file_path.suffix.lower() == ".stl":
        # STL 的逐面法线会让曲面出现分段高光；不导入它们作为自定义法线。
        bpy.ops.wm.stl_import(
            filepath=str(file_path), forward_axis="Y", up_axis="Z", use_facet_normal=False,
        )
    elif file_path.suffix.lower() == ".obj":
        bpy.ops.wm.obj_import(filepath=str(file_path), forward_axis="Y", up_axis="Z")
    else:
        raise ValueError(f"Expected an STL or OBJ file: {file_path}")
    objects = [
        obj for obj in bpy.context.scene.objects
        if obj not in previous_objects and obj.type == "MESH"
    ]
    if file_path.suffix.lower() == ".stl":
        for obj in objects:
            _smooth_stl_mesh(obj.data)
    return objects


def _smooth_stl_mesh(mesh: bpy.types.Mesh) -> None:
    """只改变着色法线，保留顶点、三角形和轮廓。

    小夹角边两侧插值法线，大于阈值的边保持硬边；共面的三角形仍呈平面。
    这是网格近似：小于阈值的设计折线也会被平滑，不能替代原始 CAD 法线。
    """
    for polygon in mesh.polygons:
        polygon.use_smooth = True
    mesh.set_sharp_from_angle(angle=math.radians(STL_SMOOTH_ANGLE_DEGREES))
    mesh.update()


def _world_bounds(objects: list[bpy.types.Object]) -> BoundingBox:
    """包含世界变换及已有 modifier 的 AABB；不修改网格。"""
    graph = bpy.context.evaluated_depsgraph_get()
    points = []
    for obj in objects:
        evaluated = obj.evaluated_get(graph)
        points.extend(evaluated.matrix_world @ Vector(corner) for corner in evaluated.bound_box)
    return BoundingBox(
        tuple(min(point[axis] for point in points) for axis in range(3)),
        tuple(max(point[axis] for point in points) for axis in range(3)),
    )


def _camera_profile(scene: bpy.types.Scene) -> PerspectiveCamera:
    transform = scene.camera.matrix_world
    position = transform.translation
    return PerspectiveCamera(
        position=tuple(position),
        target=tuple(position + transform.to_3x3() @ Vector((0, 0, -1))),
        up_reference=tuple(transform.to_3x3() @ Vector((0, 1, 0))),
        vertical_fov_degrees=math.degrees(scene.camera.data.angle_y),
        width=scene.render.resolution_x,
        height=scene.render.resolution_y,
    )


def _check_visibility(scene, objects, bounds, camera) -> None:
    if not check_bbox_framing(
        bounds, camera, near_clip=scene.camera.data.clip_start, far_clip=scene.camera.data.clip_end,
    ):
        raise ValueError("The CAD bounding box is outside the camera frame")
    geometry_types = {"MESH", "CURVE", "SURFACE", "FONT", "META"}
    scene_bounds = {
        obj.name: _world_bounds([obj]) for obj in scene.objects
        if obj not in objects and obj.type in geometry_types and not obj.hide_render
    }
    support_name = scene["support_object"]
    support = scene_bounds[support_name]
    contact_height = bounds.minimum[2]
    if not math.isclose(
        support.maximum[2], contact_height, rel_tol=0, abs_tol=SUPPORT_CONTACT_TOLERANCE,
    ):
        raise ValueError("The support surface does not meet the CAD's lowest point")
    if any(
        support.minimum[axis] > bounds.minimum[axis] + SUPPORT_CONTACT_TOLERANCE
        or support.maximum[axis] < bounds.maximum[axis] - SUPPORT_CONTACT_TOLERANCE
        for axis in (0, 1)
    ):
        raise ValueError("The support surface does not cover the CAD bounding box footprint")

    # Blender 世界变换使用 float32。仅对声明的支撑面吸收接触处的舍入误差；
    # 真实凸起超过此容差已被拒绝，其他道具的 AABB 保持原样。
    scene_bounds[support_name] = BoundingBox(
        (*support.minimum[:2], min(support.minimum[2], contact_height)),
        (*support.maximum[:2], contact_height),
    )
    blockers = find_blockers(bounds, camera, scene_bounds)
    if blockers:
        raise ValueError(f"Objects intrude into the camera-to-CAD protected region: {blockers}")


def _remove_model(objects: list[bpy.types.Object]) -> None:
    meshes = {obj.data for obj in objects}
    for obj in objects:
        bpy.data.objects.remove(obj, do_unlink=True)
    for mesh in meshes:
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)


def _configure_device(scene: bpy.types.Scene, device: str, gpu_index: int) -> None:
    if device == "CPU":
        scene.cycles.device = "CPU"
        return
    preferences = bpy.context.preferences.addons["cycles"].preferences
    preferences.compute_device_type = device
    preferences.get_devices()
    devices = [item for item in preferences.devices if item.type == device]
    if not 0 <= gpu_index < len(devices):
        raise ValueError(f"{device} GPU index {gpu_index} unavailable; detected {len(devices)} GPUs")
    for item in preferences.devices:
        item.use = False
    devices[gpu_index].use = True
    scene.cycles.device = "GPU"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene_file", type=Path, required=True)
    parser.add_argument("--model_files", type=Path, nargs="+", required=True)
    parser.add_argument("--output_dir", type=Path, required=True, help="本次渲染的新输出目录")
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("CPU", "CUDA", "OPTIX"), default="OPTIX")
    parser.add_argument("--gpu_index", type=int, default=0)
    parser.add_argument("--model_seeds", type=int, nargs="+")
    parser.add_argument("--continue_on_error", action="store_true")
    parser.add_argument("--empty_heights", type=float, nargs="*")
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    args = parser.parse_args(argv)
    render(
        args.model_files, args.scene_file, args.output_dir,
        samples=args.samples, seed=args.seed, device=args.device, gpu_index=args.gpu_index,
        model_seeds=args.model_seeds, continue_on_error=args.continue_on_error,
        empty_heights=args.empty_heights,
    )


if __name__ == "__main__":
    main()
