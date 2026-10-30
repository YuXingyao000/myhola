"""Blender 内读取场景、测量候选区域、按完整物件组施加有限移动。"""

import math

import bpy
from mathutils import Matrix
import numpy as np

from spatial import SceneGeometry, above_contact, camera_spec, pixel_ray, project_point, proxy_bounds, bounds_of


def open_scene(path):
    bpy.ops.wm.open_mainfile(filepath=str(path))
    if bpy.context.scene.unit_settings.scale_length != 1:
        raise ValueError("Step 2 输入需要 1 Blender unit = 1 m；请先转换场景单位")


def prepare_support_scene(regions, output_path):
    """支撑面的宏观形状以网格为准；微纹理保留为 Bump，另存本轮场景。

    对象级材质槽避免连带修改其他共用网格/材质的物体。原 .blend 不回写。
    真正需要起伏的支撑面必须提前做成网格，不能只在 Cycles 渲染时出现。
    """
    changes = []
    support_ids = sorted({region["support_object_id"] for region in regions})
    for name in support_ids:
        obj = bpy.data.objects[name]
        for index, slot in enumerate(obj.material_slots):
            original = slot.material
            if original is None or original.displacement_method == "BUMP":
                continue
            material = original.copy()
            material.name = f"{original.name} · support"
            material.displacement_method = "BUMP"
            slot.link = "OBJECT"
            slot.material = material
            changes.append({"object_id": name, "slot": index, "source_material": original.name,
                            "prepared_material": material.name, "before": original.displacement_method,
                            "after": material.displacement_method})
    bpy.ops.wm.save_as_mainfile(filepath=str(output_path), check_existing=False, copy=True, relative_remap=True)
    return {"base_scene": str(output_path), "support_object_ids": support_ids,
            "policy": "mesh_geometry_with_bump", "material_changes": changes}


def describe_scene(geometry, views) -> dict:
    """编号与名称对应；投影框只用于定位，不声明遮挡或支撑正确。"""
    objects = []
    for index, (name, item) in enumerate(sorted(geometry.items.items()), 1):
        bounds = item["bounds"]
        obj = bpy.data.objects.get(name)
        objects.append({
            "id": name, "label": index, "bounds_min": bounds.minimum, "bounds_max": bounds.maximum,
            "parent": obj.parent.name if obj and obj.parent else None,
            "children": [child.name for child in obj.children] if obj else [],
            "instance": item["instance"], "collision_parts": len(item["parts"]),
        })
    projections = {}
    for view in views:
        camera = camera_spec(view["position"], view["target"], view)
        forward = (np.array(camera.target) - camera.position)
        visible = []
        for item in objects:
            center = (np.array(item["bounds_min"]) + item["bounds_max"]) / 2
            if np.dot(center - camera.position, forward) <= 0:
                continue
            u, v, _ = project_point(tuple(center), camera)
            if 0 <= u < camera.width and 0 <= v < camera.height:
                visible.append({"id": item["id"], "label": item["label"], "pixel": [float(u), float(v)]})
        projections[view["id"]] = visible
    hierarchy = [{"id": obj.name, "type": obj.type, "parent": obj.parent.name if obj.parent else None,
                  "children": [child.name for child in obj.children]}
                 for obj in bpy.context.scene.objects]
    return {"units": "meters", "axis": "right-handed Z-up", "objects": objects, "hierarchy": hierarchy,
            "views": views, "label_positions": projections}


def measure_region(geometry, region, view, config) -> dict:
    """用像素区域射向指定表面，再以 1 cm 网格测量；道具不会被当成支撑面。"""
    support_id = region["support_object_id"]
    support = geometry.items[support_id]
    x0, y0, x1, y1 = region["image_region_px"]
    hits = []
    for pixel in ((x, y) for x in np.linspace(x0, x1, 5) for y in np.linspace(y0, y1, 5)):
        origin, direction = pixel_ray(view, pixel)
        point, normal, _, _ = support["tree"].ray_cast(origin, direction)
        if point is not None and normal.z >= math.cos(math.radians(config["surface_slope_degrees"])):
            hits.append(list(point))
    if not hits:
        return {"region_id": region["region_id"], "status": "no_surface", "sites": []}
    bounds = bounds_of(hits)
    z = float(np.median(np.asarray(hits)[:, 2]))
    step = config["surface_grid_step_m"]
    margin = config["proxy_size_m"] / 2 + config["support_margin_m"]
    radius = math.ceil(margin / step)
    # 先采样一次高度场，随后复用窗口，不为每个落点重复数百条射线。
    xs = np.arange(bounds.minimum[0] - radius * step, bounds.maximum[0] + (radius + 1) * step, step)
    ys = np.arange(bounds.minimum[1] - radius * step, bounds.maximum[1] + (radius + 1) * step, step)
    supported = np.array([[geometry.support_height(support_id, x, y, z) is not None for y in ys] for x in xs])
    allowed = {name for group in region["movable_groups"] for name in group["members"]}
    sites = []
    for i in range(radius, len(xs) - radius):
        for j in range(radius, len(ys) - radius):
            if not supported[i-radius:i+radius+1, j-radius:j+radius+1].all():
                continue
            point = [float(xs[i]), float(ys[j]), z]
            envelope = proxy_bounds(point, config, config["support_margin_m"])
            blockers = geometry.collisions(above_contact(envelope, z, config))
            kind = "free" if not blockers else "movable" if set(blockers) <= allowed else "fixed_or_unknown"
            sites.append({"point": point, "occupants": blockers, "kind": kind})
    counts = {kind: sum(item["kind"] == kind for item in sites) for kind in ("free", "movable", "fixed_or_unknown")}
    return {"region_id": region["region_id"], "status": "measured" if sites else "no_supported_patch",
            "support_object_id": support_id, "surface_z_m": z,
            "sample_step_m": step, "counts": counts, "sites": sites,
            "requires_rearrangement": counts["movable"] > 0,
            "movability_is_agent_assumption": True}


def group_bounds(geometry, group):
    members = [geometry.items[name]["bounds"] for name in group["members"] if name in geometry.items]
    return bounds_of([point for box in members for point in (box.minimum, box.maximum)])


def validate_moves(region, proposal, config) -> list[str]:
    """核对有限动作合同。可动性的语义由 Agent 提议，程序不靠名字猜台钳是否固定。"""
    groups = {group["id"]: group for group in region["movable_groups"]}
    moves = proposal["moves"]
    reasons, seen = [], set()
    moved_ids = [move["group_id"] for move in moves]
    if len(moves) > config["max_moved_groups"] or len(set(moved_ids)) != len(moved_ids):
        reasons.append("too_many_or_duplicate_moves")
    if not set(moved_ids) <= set(proposal["retain_in_frame"]):
        reasons.append("all_moved_groups_must_remain_in_frame")
    if not set(proposal["retain_in_frame"]) <= groups.keys():
        reasons.append("unknown_retained_group")
    for move in moves:
        group = groups[move["group_id"]]
        members = set(group["members"])
        if members & seen:
            reasons.append(f"{group['id']}: overlapping_groups")
        seen.update(members)
        if group["support_object_id"] != region["support_object_id"] or region["support_object_id"] in members:
            reasons.append(f"{group['id']}: support_must_stay_fixed")
        for name in members:
            obj = bpy.data.objects[name]
            if obj.type not in {"MESH", "CURVE", "SURFACE", "FONT", "EMPTY"} or obj.instance_type != "NONE":
                reasons.append(f"{name}: unsupported_movable_type")
            if obj.constraints or any(modifier.type == "ARMATURE" for modifier in obj.modifiers):
                reasons.append(f"{name}: constrained_or_rigged_object")
            if not {child.name for child in obj.children_recursive} <= members:
                reasons.append(f"{name}: incomplete_parent_child_group")
        dx, dy = move["translation_xy_m"]
        if math.hypot(dx, dy) > config["max_translation_m"] or abs(move["yaw_degrees"]) > config["max_yaw_degrees"]:
            reasons.append(f"{group['id']}: movement_budget_exceeded")
    return reasons


def set_world_matrices(matrices):
    # 先父后子，给父子指定各自的最终世界矩阵，避免移动被重复叠加。
    def depth(name):
        parent, count = bpy.data.objects[name].parent, 0
        while parent is not None:
            parent, count = parent.parent, count + 1
        return count

    for name in sorted(matrices, key=depth):
        bpy.data.objects[name].matrix_world = matrices[name]
        bpy.context.view_layer.update()
    bpy.context.view_layer.update()


def apply_moves(region, proposal):
    """本轮完整方案相对刚打开的基础场景应用一次；返回原/新世界矩阵。"""
    groups = {group["id"]: group for group in region["movable_groups"]}
    before, after = {}, {}
    for move in proposal["moves"]:
        pivot = Matrix.Translation(move["pivot_world"])
        dx, dy = move["translation_xy_m"]
        transform = (Matrix.Translation((dx, dy, 0)) @ pivot
                     @ Matrix.Rotation(math.radians(move["yaw_degrees"]), 4, "Z") @ pivot.inverted())
        for name in groups[move["group_id"]]["members"]:
            before[name] = bpy.data.objects[name].matrix_world.copy()
            after[name] = transform @ before[name]
    set_world_matrices(after)
    return before, after
