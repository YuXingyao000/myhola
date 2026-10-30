"""用实际渲染深度核对支撑面，补足普通网格 BVH 看不到的材质位移。

Blender 5.2.2 本机已验证：Z pass 是相机轴向米制深度。只评估代理周围
实际可见的指定支撑对象，不把被遮挡的部分宣称为连续表面证明。
"""

import math

import bpy
import numpy as np


def add_support_outputs(support_id, output_dir):
    """与方体 after.png 同一次渲染。30000 与保留物件组的小整数 ID 分开。"""
    scene = bpy.context.scene
    layer = bpy.context.view_layer
    layer.use_pass_object_index = layer.use_pass_z = True
    layer.update_render_passes()
    bpy.data.objects[support_id].pass_index = 30000
    # preview.add_group_masks 已准备合成树，保留它原有的 RGB 输出。
    tree = scene.compositing_node_group
    layers = tree.nodes.new("CompositorNodeRLayers")
    layers.layer = layer.name
    mask = tree.nodes.new("CompositorNodeIDMask")
    mask.inputs["Index"].default_value = 30000
    mask.inputs["Anti-Alias"].default_value = False
    tree.links.new(layers.outputs["Object Index"], mask.inputs[0])
    for name, socket, file_format in (("support_mask", mask.outputs[0], "PNG"),
                                       ("support_depth", layers.outputs["Depth"], "OPEN_EXR")):
        output = tree.nodes.new("CompositorNodeOutputFile")
        output.directory = str(output_dir)
        output.file_name = name + "_"
        output.format.media_type = "IMAGE"
        output.format.file_format = file_format
        output.format.color_mode = "BW"
        output.format.color_depth = "32" if file_format == "OPEN_EXR" else "8"
        output.save_as_render = False
        output.file_output_items.new("FLOAT", "value")
        tree.links.new(socket, output.inputs[0])


def read_channel(path):
    image = bpy.data.images.load(str(path), check_existing=False)
    width, height = image.size
    values = np.asarray(image.pixels[:]).reshape(height, width, 4)[::-1, :, 0].copy()
    bpy.data.images.remove(image)
    return values


def check_rendered_support(output_dir, point, camera, config):
    """反投影被支撑面 mask 覆盖的局部像素，直接与记录高度比较。"""
    mask_path, = output_dir.glob("support_mask_*.png")
    depth_path, = output_dir.glob("support_depth_*.exr")
    mask = read_channel(mask_path) > 0.5
    depth = read_channel(depth_path)
    height, width = depth.shape
    # 去掉 ID 边界一个像素，避免表面交界处的采样歧义。
    interior = np.zeros_like(mask)
    interior[1:-1, 1:-1] = (mask[1:-1, 1:-1] & mask[:-2, 1:-1] & mask[2:, 1:-1]
                            & mask[1:-1, :-2] & mask[1:-1, 2:])
    rows, columns = np.indices(depth.shape)
    focal = height / (2 * math.tan(math.radians(config["vertical_fov_degrees"]) / 2))
    # Blender 相机的本地坐标为 X 右、Y 上、-Z 前。
    rays = np.stack([(columns + .5 - width / 2) / focal,
                     (height / 2 - rows - .5) / focal, -np.ones_like(depth)], axis=-1)
    camera_points = rays * depth[:, :, None]
    transform = np.asarray(camera.matrix_world)
    world = camera_points @ transform[:3, :3].T + transform[:3, 3]
    radius = config["proxy_size_m"] / 2 + config["support_margin_m"]
    selected = (interior & np.isfinite(depth) & (depth > 0)
                & (abs(world[:, :, 0] - point[0]) <= radius)
                & (abs(world[:, :, 1] - point[1]) <= radius))
    heights = world[:, :, 2][selected]
    report = {"pass": False, "visible_surface_pixels": int(heights.size),
              "expected_height_m": point[2], "tolerance_m": config["height_tolerance_m"],
              "mask": str(mask_path), "depth": str(depth_path), "depth_convention": "camera_axis_z_m"}
    if not heights.size:
        return dict(report, reason="no_visible_support_pixels")
    error = float(np.max(np.abs(heights - point[2])))
    report.update(rendered_height_m={"min": float(heights.min()), "median": float(np.median(heights)),
                                      "max": float(heights.max())}, max_height_error_m=error,
                  **{"pass": error <= config["height_tolerance_m"]})
    return report
