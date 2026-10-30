"""Blender 内的空间计算：真实表面采样、保守 AABB、相机投影。

坐标为右手系，Z 向上，1 Blender unit = 1 m。支撑来自求交，包围盒只用于
碰撞近似。表面网格采样不证明连续支撑；AABB 相交也不等于真实网格相交。
"""

from itertools import product
import math
from pathlib import Path
import sys

import bpy
from mathutils import Vector
from mathutils.bvhtree import BVHTree
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
from geometry import BoundingBox, PerspectiveCamera, check_bbox_framing, project_point
from visibility import find_blockers


def bounds_of(points) -> BoundingBox:
    points = np.asarray(points)
    return BoundingBox(tuple(points.min(axis=0).tolist()), tuple(points.max(axis=0).tolist()))


def overlaps(a: BoundingBox, b: BoundingBox, tolerance: float) -> bool:
    # 允许接触；无厚度的墙面落入体积内部仍算碰撞。
    return all(a.minimum[i] + tolerance < b.maximum[i]
               and b.minimum[i] < a.maximum[i] - tolerance for i in range(3))


def proxy_bounds(point, config, margin=0.0) -> BoundingBox:
    x, y, z = point
    half = config["proxy_size_m"] / 2 + margin
    return BoundingBox((x - half, y - half, z), (x + half, y + half, z + config["proxy_size_m"]))


def above_contact(bounds, contact_z, config) -> BoundingBox:
    """碰撞/遮挡查询略去底部接触容差；薄贴花或微小起伏不要求被挪走。

    支撑与完整入镜仍用原始包络。这里允许的穿入量明确是 height_tolerance_m，
    并非宣称代理底面附近严格零误差。
    """
    low = list(bounds.minimum)
    low[2] = min(bounds.maximum[2], max(low[2], contact_z + config["height_tolerance_m"]))
    return BoundingBox(tuple(low), bounds.maximum)


class SceneGeometry:
    """缓存当前姿态的渲染网格。挪物后重新构造，避免混用旧位置的 BVH。

    普通对象 ID 是 Blender 名称；集合实例带 parent/persistent_id 后缀，只作为
    障碍，首版不移动实例。大网格按共面片区拆碰撞盒，避免把合并房间当实心盒。
    """

    def __init__(self, config):
        self.config = config
        self.items = {}
        depsgraph = bpy.context.evaluated_depsgraph_get()
        def collection_objects(collection):
            if collection.hide_render:
                return set()
            names = {obj.name for obj in collection.objects if not obj.hide_render}
            for child in collection.children:
                names.update(collection_objects(child))
            return names

        visible = collection_objects(bpy.context.scene.collection)
        visible &= {obj.name for obj in bpy.context.view_layer.objects}
        for instance in depsgraph.object_instances:
            obj = instance.object
            if obj.type not in {"MESH", "CURVE", "SURFACE", "FONT"} or obj.hide_render:
                continue
            if not instance.is_instance and obj.name not in visible:
                continue
            name = obj.original.name
            if instance.is_instance:
                if instance.parent.original.name not in visible:
                    continue
                name = f"{instance.parent.original.name}::{name}::{tuple(instance.persistent_id)}"
            mesh = obj.to_mesh()
            matrix = instance.matrix_world
            vertices = [matrix @ vertex.co for vertex in mesh.vertices]
            polygons = [tuple(face.vertices) for face in mesh.polygons]
            if polygons:
                bounds = bounds_of(vertices)
                parts = [bounds]
                if max(b - a for a, b in zip(bounds.minimum, bounds.maximum)) >= config["split_bounds_above_m"]:
                    parts = planar_bounds(vertices, polygons)
                self.items[name] = {
                    "bounds": bounds, "parts": parts,
                    "tree": BVHTree.FromPolygons(vertices, polygons),
                    "instance": instance.is_instance,
                }
            obj.to_mesh_clear()

    def collisions(self, bounds, exclude=()) -> list[str]:
        return [name for name, item in self.items.items() if name not in exclude
                and any(overlaps(bounds, part, self.config["collision_tolerance_m"]) for part in item["parts"])]

    def blockers(self, bounds, camera) -> list[str]:
        parts = {f"{name}#{i}": box for name, item in self.items.items()
                 for i, box in enumerate(item["parts"])}
        return sorted({key.rsplit("#", 1)[0] for key in find_blockers(bounds, camera, parts)})

    def support_height(self, support_id, x, y, expected_z):
        tolerance = self.config["height_tolerance_m"]
        hit, normal, _, _ = self.items[support_id]["tree"].ray_cast(
            Vector((x, y, expected_z + 2 * tolerance)), Vector((0, 0, -1)), 4 * tolerance)
        if hit is None or normal.z < math.cos(math.radians(self.config["surface_slope_degrees"])):
            return None
        return float(hit.z) if abs(hit.z - expected_z) <= tolerance else None

    def supported(self, support_id, bounds, margin=0.0) -> dict:
        """在整个轴对齐占地上采样，包括边缘。不能用四个角替代内部孔洞检查。"""
        step = self.config["surface_grid_step_m"]
        axes = [np.linspace(bounds.minimum[i] - margin, bounds.maximum[i] + margin,
                            math.ceil((bounds.maximum[i] - bounds.minimum[i] + 2 * margin) / step) + 1)
                for i in (0, 1)]
        heights = [self.support_height(support_id, x, y, bounds.minimum[2]) for x, y in product(*axes)]
        missing = sum(value is None for value in heights)
        valid = [value for value in heights if value is not None]
        return {"pass": missing == 0, "samples": len(heights), "unsupported_samples": missing,
                "height_range_m": [min(valid), max(valid)] if valid else None}


def planar_bounds(vertices, polygons) -> list[BoundingBox]:
    """仅给大对象使用：按世界平面合并面片；斜面仍使用保守 AABB。"""
    groups = {}
    for indices in polygons:
        points = [vertices[i] for i in indices]
        normal = (points[1] - points[0]).cross(points[2] - points[0]).normalized()
        # 正反面不必合并；共面但不相连的部分允许保守合并。
        key = tuple(round(v, 4) for v in normal) + (round(normal.dot(points[0]), 4),)
        box = bounds_of(points)
        if key in groups:
            previous = groups[key]
            box = BoundingBox(tuple(min(a, b) for a, b in zip(previous.minimum, box.minimum)),
                              tuple(max(a, b) for a, b in zip(previous.maximum, box.maximum)))
        groups[key] = box
    return list(groups.values())


def camera_spec(position, target, config) -> PerspectiveCamera:
    return PerspectiveCamera(tuple(position), tuple(target), vertical_fov_degrees=config["vertical_fov_degrees"],
                             width=config["resolution"][0], height=config["resolution"][1])


def pixel_ray(view, pixel):
    """views.json 的原图像素 → 世界射线；不缩放图像坐标。"""
    origin = Vector(view["position"])
    forward = (Vector(view["target"]) - origin).normalized()
    right = forward.cross(Vector((0, 0, 1))).normalized()
    up = right.cross(forward)
    width, height = view["resolution"]
    focal = height / (2 * math.tan(math.radians(view["vertical_fov_degrees"]) / 2))
    direction = forward + right * ((pixel[0] - width / 2) / focal) - up * ((pixel[1] - height / 2) / focal)
    return origin, direction.normalized()


def check_framing(bounds, camera, config) -> dict:
    fits = check_bbox_framing(bounds, camera, near_clip=config["near_clip_m"],
                             far_clip=config["far_clip_m"], margin_pixels=config["frame_margin_px"])
    if not fits:
        return {"pass": False, "reason": "proxy_outside_frame_or_clip"}
    pixels = np.array([project_point(corner, camera)[:2] for corner in bounds.corners()])
    low, high = pixels.min(axis=0), pixels.max(axis=0)
    fraction = float(max(high - low) / min(camera.width, camera.height))
    offset = float(np.linalg.norm((high + low) / 2 - np.array([camera.width, camera.height]) / 2)
                   / min(camera.width, camera.height))
    minimum, maximum = config["projected_size_range"]
    return {"pass": minimum <= fraction <= maximum and offset <= config["center_offset_fraction"],
            "bbox_px": [*low.tolist(), *high.tolist()], "long_side_fraction": fraction,
            "center_offset_fraction": offset}
