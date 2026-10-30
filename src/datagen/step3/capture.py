"""Blender 入口。按机位打开场景，主循环只做：导入 → 摆放 → 拍摄 → 删除。

直接输入外围 workflow.py 冻结的 request.json，不调用 Agent，也不保存改写场景。
"""

import argparse
import json
import math
from pathlib import Path
import sys
import time
import traceback

import bpy
from mathutils import Matrix
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "step2"))
from part import load_part, make_material, place_part, remove_part, check_support_contact
from preview import configure_render, make_camera
from scene import open_scene, set_world_matrices
from spatial import bounds_of, camera_spec
from geometry import check_bbox_framing, project_point


def capture_batch(request):
    output_dir = Path(request["output_dir"])
    config = request["config"]
    failures = 0
    with (output_dir / "renders.jsonl").open("w") as records:
        for index, placement in enumerate(request["placements"]):
            jobs = [job for job in request["jobs"] if job["placement_index"] == index]
            if not jobs:
                continue
            camera, outputs = prepare_placement(request, placement)
            material = make_material(config["material"])
            for job in jobs:
                print(f"[{job['index'] + 1}/{len(request['jobs'])}] {job['model_id']} · "
                      f"{placement['setup']['region_id']}", flush=True)
                sample_dir = output_dir / job["model_id"]
                sample_dir.mkdir()
                record = dict(job, status="failed", region_id=placement["setup"]["region_id"])
                part = None
                started = time.monotonic()
                # 唯一逐件异常边界：坏样本留在分母里，其余样本继续。
                try:
                    part = load_part(job["stl_file"], config["smooth_angle_degrees"])
                    record["geometry"] = place_part(
                        part, placement["setup"]["support_point_world"], job["longest_side_m"],
                        request["step2_config"]["proxy_size_m"])
                    record["contact"] = check_support_contact(part, placement["support_validation"])
                    record["framing"] = check_part_framing(record["geometry"], placement)
                    part.data.materials.clear()
                    part.data.materials.append(material)
                    part.pass_index = 1
                    record["images"], record["mask"] = take_photo(sample_dir, job["seed"], outputs)
                    record.update(status="success", camera=camera, material=config["material"],
                                  render_device=request["render_device"],
                                  smooth_angle_degrees=config["smooth_angle_degrees"],
                                  scene_hash=request["scene_hash"], blender_version=bpy.app.version_string,
                                  depth_convention="camera-axis Z in meters; full-scene first surface")
                except Exception as error:
                    failures += 1
                    record["error"] = f"{type(error).__name__}: {error}"
                    traceback.print_exc()
                finally:
                    if part is not None:
                        remove_part(part)
                    record["elapsed_seconds"] = time.monotonic() - started
                    (sample_dir / "metadata.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
                    records.write(json.dumps(record, ensure_ascii=False) + "\n")
                    records.flush()
    if failures:
        raise RuntimeError(f"{failures} 个样本失败，详见 renders.jsonl")


def prepare_placement(request, placement):
    """每个机位从同一个原场景开始，避免上个机位的移动累积。"""
    open_scene(request["base_scene"])
    set_world_matrices({name: Matrix(value["after"]) for name, value in placement["transforms"].items()})
    config = dict(request["step2_config"], samples=request["config"]["samples"],
                  threads=request["config"]["threads"])
    configure_render(config)
    request["render_device"] = configure_device(request)
    camera = make_camera(placement["setup"], config)
    camera.matrix_world = Matrix(placement["camera"]["matrix_world"])
    bpy.context.view_layer.update()
    bpy.context.scene.cycles.use_denoising = True
    bpy.context.scene.cycles.use_adaptive_sampling = False
    return camera_metadata(camera, config), add_geometry_outputs()


def configure_device(request):
    """外围进程已隐藏其他 GPU；这里仅启用唯一可见的 OptiX 设备。"""
    if request["device"] == "CPU":
        return {"type": "CPU", "threads": request["config"]["threads"]}
    preferences = bpy.context.preferences.addons["cycles"].preferences
    preferences.compute_device_type = "OPTIX"
    preferences.get_devices()
    devices = [device for device in preferences.devices if device.type == "OPTIX"]
    if len(devices) != 1:
        raise ValueError(f"预期唯一可见的 OPTIX GPU，实际发现 {len(devices)} 个")
    for device in preferences.devices:
        device.use = device == devices[0]
    bpy.context.scene.cycles.device = "GPU"
    return {"type": "OPTIX", "gpu_index": request["gpu_index"],
            "name": devices[0].name, "id": devices[0].id}


def camera_metadata(camera, config):
    width, height = config["resolution"]
    focal_px = height / (2 * math.tan(math.radians(config["vertical_fov_degrees"]) / 2))
    cv_from_blender = Matrix.Diagonal((1, -1, -1, 1))
    return {
        "resolution": [width, height], "vertical_fov_degrees": config["vertical_fov_degrees"],
        "K": [[focal_px, 0, width / 2], [0, focal_px, height / 2], [0, 0, 1]],
        "pixel_convention": "top-left image origin; pixel centers at (column+0.5, row+0.5)",
        "camera_to_world_blender": [list(row) for row in camera.matrix_world],
        "world_to_camera_cv": [list(row) for row in cv_from_blender @ camera.matrix_world.inverted()],
        "cv_axes": "X right, Y down, Z forward", "world_units": "meters",
        "near_clip_m": camera.data.clip_start, "far_clip_m": camera.data.clip_end,
    }


def check_part_framing(geometry, placement):
    bounds = bounds_of([geometry["world_bounds"]["minimum"], geometry["world_bounds"]["maximum"]])
    spec = placement["camera"]
    camera = camera_spec(placement["setup"]["camera_position_world"], placement["setup"]["look_at_world"], spec)
    if not check_bbox_framing(bounds, camera, near_clip=spec["near_clip_m"], far_clip=spec["far_clip_m"]):
        raise ValueError("零件包围盒未完整入镜")
    projected = np.array([project_point(corner, camera)[:2] for corner in bounds.corners()])
    return {"inside_frame": True, "bbox_px": [*projected.min(axis=0).tolist(), *projected.max(axis=0).tolist()]}


def add_geometry_outputs():
    """与 RGB 同一次渲染导出可见零件 mask 和全场景深度，不用生成模型猜掩码。"""
    scene = bpy.context.scene
    layer = bpy.context.view_layer
    layer.use_pass_object_index = layer.use_pass_z = True
    layer.update_render_passes()
    for obj in scene.objects:
        obj.pass_index = 0
    tree = scene.compositing_node_group
    if tree is None:
        tree = bpy.data.node_groups.new("step3_outputs", "CompositorNodeTree")
        scene.compositing_node_group = tree
        tree.interface.new_socket(name="Image", in_out="OUTPUT", socket_type="NodeSocketColor")
        output = tree.nodes.new("NodeGroupOutput")
        layers = tree.nodes.new("CompositorNodeRLayers")
        tree.links.new(layers.outputs["Image"], output.inputs["Image"])
    scene.render.use_compositing = True
    layers = tree.nodes.new("CompositorNodeRLayers")
    layers.layer = layer.name
    mask = tree.nodes.new("CompositorNodeIDMask")
    mask.inputs["Index"].default_value = 1
    mask.inputs["Anti-Alias"].default_value = False
    tree.links.new(layers.outputs["Object Index"], mask.inputs[0])
    outputs = {}
    for name, socket, file_format in (("mask", mask.outputs[0], "PNG"),
                                       ("depth", layers.outputs["Depth"], "OPEN_EXR")):
        output = tree.nodes.new("CompositorNodeOutputFile")
        output.file_name = name + "_"
        output.format.media_type = "IMAGE"
        output.format.file_format = file_format
        output.format.color_mode = "BW"
        output.format.color_depth = "32" if name == "depth" else "8"
        output.save_as_render = False
        output.file_output_items.new("FLOAT", name)
        tree.links.new(socket, output.inputs[0])
        outputs[name] = output
    return outputs


def take_photo(output_dir, seed, outputs):
    scene = bpy.context.scene
    scene.cycles.seed = seed
    scene.render.filepath = str(output_dir / "rgb.png")
    for node in outputs.values():
        node.directory = str(output_dir)
    bpy.ops.render.render(write_still=True)
    for name, suffix in (("mask", "png"), ("depth", "exr")):
        path, = output_dir.glob(f"{name}_*.{suffix}")
        path.rename(output_dir / f"{name}.{suffix}")
    image = bpy.data.images.load(str(output_dir / "mask.png"), check_existing=False)
    width, height = image.size
    pixels = np.array(image.pixels[:]).reshape(height, width, 4)
    rows, columns = np.nonzero(pixels[:, :, 0] > 0.5)
    bpy.data.images.remove(image)
    if len(rows) == 0:
        raise ValueError("零件可见掩码为空")
    mask = {"visible_pixels": len(rows), "image_fraction": len(rows) / (width * height),
            "bbox_px": [int(columns.min()), int(height - 1 - rows.max()),
                        int(columns.max() + 1), int(height - rows.min())]}
    images = {name: str(output_dir / filename) for name, filename in
              (("rgb", "rgb.png"), ("mask", "mask.png"), ("depth", "depth.exr"))}
    return images, mask


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    capture_batch(json.loads(args.request.read_text()))
