"""Blender 内的零件操作：导入、等比摆放、程序化材质、删除。

保留 STL 坐标轴和三角形。物理尺寸由拍摄任务定义，不从无单位的 STL 猜测。
"""

import math

import bpy
from mathutils import Matrix, Vector
import numpy as np


def make_material(config):
    material = bpy.data.materials.new(config["name"])
    material.use_nodes = True
    shader = material.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = config["base_color_linear"]
    shader.inputs["Metallic"].default_value = config["metallic"]
    shader.inputs["Roughness"].default_value = config["roughness"]
    return material


def load_part(path, smooth_angle_degrees):
    bpy.ops.wm.stl_import(filepath=str(path), forward_axis="Y", up_axis="Z", use_facet_normal=False)
    part = bpy.context.object
    part.name = "step3_part"
    # 只修改着色法线，不加 bevel/remesh，不改变孔槽、顶点或轮廓。
    for polygon in part.data.polygons:
        polygon.use_smooth = True
    part.data.set_sharp_from_angle(angle=math.radians(smooth_angle_degrees))
    part.data.update()
    return part


def place_part(part, support_point, longest_side_m, proxy_size_m):
    points = np.array([vertex.co[:] for vertex in part.data.vertices])
    if points.size == 0 or not np.isfinite(points).all():
        raise ValueError("STL 顶点为空或包含非有限数值")
    low, high = points.min(axis=0), points.max(axis=0)
    longest = float((high - low).max())
    if longest <= 0:
        raise ValueError("STL 包围盒没有正长度")
    scale = longest_side_m / longest
    # XY 居中，Z 使用最低点；保持原有姿态，不把零件质心当作接触点。
    anchor = np.array([(low[0] + high[0]) / 2, (low[1] + high[1]) / 2, low[2]])
    translation = np.asarray(support_point) - scale * anchor
    part.matrix_world = Matrix.Translation(Vector(translation)) @ Matrix.Scale(scale, 4)
    bpy.context.view_layer.update()
    world_low, world_high = low * scale + translation, high * scale + translation
    proxy_low = np.asarray(support_point) - [proxy_size_m / 2, proxy_size_m / 2, 0]
    proxy_high = proxy_low + proxy_size_m
    if np.any(world_low < proxy_low - 1e-6) or np.any(world_high > proxy_high + 1e-6):
        raise ValueError("当前零件超出了 Step 2 验收的代理包络")
    return {
        "source_bounds": {"minimum": low.tolist(), "maximum": high.tolist()},
        "world_bounds": {"minimum": world_low.tolist(), "maximum": world_high.tolist()},
        "model_to_world": [list(row) for row in part.matrix_world],
        "scale": scale, "longest_side_m": longest_side_m,
        "vertex_count": len(part.data.vertices), "triangle_count": len(part.data.polygons),
        "inside_step2_proxy": True, "support_height_m": support_point[2],
    }


def remove_part(part):
    mesh = part.data
    bpy.data.objects.remove(part, do_unlink=True)
    bpy.data.meshes.remove(mesh)


def check_support_contact(part, support_validation):
    """实际变换后的底端，对比 Step 2 在同一场景中独立测得的渲染表面。"""
    bottom = min((part.matrix_world @ vertex.co).z for vertex in part.data.vertices)
    surface = support_validation["rendered_height_m"]["median"]
    gap = bottom - surface
    if abs(gap) > support_validation["tolerance_m"]:
        raise ValueError(f"零件底端与验收表面不齐平：{gap * 1000:.3f} mm")
    return {"bottom_z_m": bottom, "rendered_surface_z_m": surface, "gap_m": gap, "pass": True}
