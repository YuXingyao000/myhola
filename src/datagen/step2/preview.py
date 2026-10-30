"""Blender 内验收移动终态，拍摄同机位的整理前/后图片及物件 ID 掩码。"""

import math
from pathlib import Path

import bpy
from mathutils import Vector
import numpy as np

from scene import apply_moves, group_bounds, set_world_matrices, validate_moves
from spatial import SceneGeometry, above_contact, camera_spec, check_framing, proxy_bounds
from support import add_support_outputs, check_rendered_support


def configure_render(config):
    """只固定摄影规格与 CPU 预算；保留场景的灯光、世界及成像调色。"""
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = config["samples"]
    scene.cycles.seed = config["seed"]
    scene.cycles.use_animated_seed = False
    scene.cycles.denoiser = "OPENIMAGEDENOISE"
    scene.cycles.denoising_use_gpu = False
    scene.render.threads_mode = "FIXED"
    scene.render.threads = config["threads"]
    scene.render.compositor_device = "CPU"
    scene.render.compositor_denoise_device = "CPU"
    scene.render.image_settings.media_type = "IMAGE"
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    scene.render.resolution_x, scene.render.resolution_y = config["resolution"]
    scene.render.resolution_percentage = 100
    scene.render.pixel_aspect_x = scene.render.pixel_aspect_y = 1
    scene.render.use_border = scene.render.use_multiview = False
    scene.render.film_transparent = False
    # 去掉原工程的文件输出副作用，不改通向最终 Image 的调色节点。
    for tree in bpy.data.node_groups:
        if tree.bl_idname == "CompositorNodeTree":
            for node in list(tree.nodes):
                if node.bl_idname == "CompositorNodeOutputFile":
                    tree.nodes.remove(node)


def make_camera(proposal, config):
    camera = bpy.data.objects.new("step2_camera", bpy.data.cameras.new("step2_camera"))
    bpy.context.scene.collection.objects.link(camera)
    camera.location = proposal["camera_position_world"]
    camera.rotation_euler = (Vector(proposal["look_at_world"]) - camera.location).to_track_quat("-Z", "Y").to_euler()
    camera.data.type = "PERSP"
    camera.data.sensor_fit = "VERTICAL"
    camera.data.sensor_height = 24
    camera.data.lens = 12 / math.tan(math.radians(config["vertical_fov_degrees"]) / 2)
    camera.data.clip_start, camera.data.clip_end = config["near_clip_m"], config["far_clip_m"]
    camera.data.dof.use_dof = False
    bpy.context.scene.camera = camera
    bpy.context.view_layer.update()
    return camera


def make_proxy(point, config):
    bpy.ops.mesh.primitive_cube_add(size=config["proxy_size_m"],
                                    location=(point[0], point[1], point[2] + config["proxy_size_m"] / 2))
    proxy = bpy.context.object
    proxy.name = "step2_proxy"
    material = bpy.data.materials.new("step2_matte_gray")
    material.use_nodes = True
    shader = material.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = (0.32, 0.32, 0.32, 1)
    shader.inputs["Metallic"].default_value = 0
    shader.inputs["Roughness"].default_value = 0.65
    proxy.data.materials.append(material)
    return proxy


def add_group_masks(region, proposal, output_dir):
    """与 after 同一次 Cycles 渲染输出 IndexOB→IDMask，保留原合成 Image。

    掩码表示渲染可见像素，不是 AABB 投影。非抗锯齿 Object Index 不等于
    感知可辨认度；透明/折射表面也不能用此通道证明视觉清晰。
    """
    scene = bpy.context.scene
    layer = bpy.context.view_layer
    layer.use_pass_object_index = True
    layer.update_render_passes()
    scene.render.use_compositing = True
    tree = scene.compositing_node_group
    if tree is None:
        tree = bpy.data.node_groups.new("step2_compositor", "CompositorNodeTree")
        scene.compositing_node_group = tree
        tree.interface.new_socket(name="Image", in_out="OUTPUT", socket_type="NodeSocketColor")
        output = tree.nodes.new("NodeGroupOutput")
        layers = tree.nodes.new("CompositorNodeRLayers")
        tree.links.new(layers.outputs["Image"], output.inputs["Image"])
    layers = tree.nodes.new("CompositorNodeRLayers")
    layers.layer = layer.name
    groups = {group["id"]: group for group in region["movable_groups"]}
    for obj in scene.objects:
        obj.pass_index = 0
    masks = []
    for index, group_id in enumerate(proposal["retain_in_frame"], 1):
        for name in groups[group_id]["members"]:
            bpy.data.objects[name].pass_index = index
        mask = tree.nodes.new("CompositorNodeIDMask")
        mask.inputs["Index"].default_value = index
        mask.inputs["Anti-Alias"].default_value = False
        tree.links.new(layers.outputs["Object Index"], mask.inputs[0])
        output = tree.nodes.new("CompositorNodeOutputFile")
        output.directory = str(output_dir)
        output.file_name = f"mask_{index:02d}_"
        output.format.media_type = "IMAGE"
        output.format.file_format = "PNG"
        output.format.color_mode = "BW"
        output.save_as_render = False
        output.file_output_items.new("FLOAT", "mask")
        tree.links.new(mask.outputs[0], output.inputs[0])
        masks.append({"group_id": group_id, "prefix": output.file_name})
    return masks


def render_image(path):
    bpy.context.scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)


def preview_placement(region, proposal, config, output_dir) -> dict:
    """调用者每次先打开基础场景。本函数不保存/改写原 blend。"""
    reasons = validate_moves(region, proposal, config)
    report = {"region_id": region["region_id"], "numeric_pass": False, "reasons": reasons,
              "images": {}, "checks": {}, "moves": [], "visibility": []}
    if reasons:
        return report
    original_geometry = SceneGeometry(config)
    groups = {group["id"]: group for group in region["movable_groups"]}
    before, after = apply_moves(region, proposal)
    geometry = SceneGeometry(config)
    # Blender 父级非均匀缩放/约束可能让赋值后的世界矩阵偏离刚体变换。
    # 验收实际矩阵；不能只记录我们希望赋予的矩阵。
    for name, expected in after.items():
        actual = bpy.data.objects[name].matrix_world
        if max(abs(a - b) for row_a, row_b in zip(actual, expected) for a, b in zip(row_a, row_b)) > 1e-5:
            reasons.append(f"{name}: rigid_transform_not_preserved")
    for move in proposal["moves"]:
        group = groups[move["group_id"]]
        original = group_bounds(original_geometry, group)
        current = group_bounds(geometry, group)
        support = geometry.supported(group["support_object_id"], current)
        original_support = original_geometry.supported(group["support_object_id"], original)
        collisions = sorted({blocker for name in group["members"] if name in geometry.items
                             for blocker in geometry.collisions(
                                 above_contact(geometry.items[name]["bounds"], current.minimum[2], config), group["members"])})
        passed = support["pass"] and original_support["pass"] and not collisions
        report["moves"].append({"group_id": group["id"], "pass": passed, "support_before": original_support,
                                "support_after": support, "collision_ids": collisions})
        if not passed:
            reasons.append(f"{group['id']}: unsupported_or_colliding_move")
    bounds = proxy_bounds(proposal["support_point_world"], config)
    camera = camera_spec(proposal["camera_position_world"], proposal["look_at_world"], config)
    support = geometry.supported(region["support_object_id"], bounds, config["support_margin_m"])
    envelope = proxy_bounds(proposal["support_point_world"], config, config["support_margin_m"])
    collision_ids = geometry.collisions(above_contact(envelope, bounds.minimum[2], config))
    blocker_ids = geometry.blockers(above_contact(bounds, bounds.minimum[2], config), camera)
    framing = check_framing(bounds, camera, config)
    report["checks"] = {"support": support, "contact_band_m": config["height_tolerance_m"], "collision_ids": collision_ids,
                        "protected_volume_blockers": blocker_ids, "framing": framing}
    for name, passed in (("proxy_support", support["pass"]), ("proxy_collision", not collision_ids),
                         ("camera_protected_volume", not blocker_ids), ("framing", framing["pass"])):
        if not passed:
            reasons.append(name)
    report["transforms"] = {name: {"before": [list(row) for row in before[name]],
                                    "after": [list(row) for row in bpy.data.objects[name].matrix_world]} for name in before}
    if reasons:
        return report

    configure_render(config)
    actual_camera = make_camera(proposal, config)
    report["camera"] = {"matrix_world": [list(row) for row in actual_camera.matrix_world],
                        "vertical_fov_degrees": config["vertical_fov_degrees"], "resolution": config["resolution"],
                        "near_clip_m": config["near_clip_m"], "far_clip_m": config["far_clip_m"]}
    # 原图不放代理，避免在尚未腾位时把方体插进道具。两图用同一个相机。
    set_world_matrices(before)
    before_path, after_path = output_dir / "before.png", output_dir / "after.png"
    render_image(before_path)
    set_world_matrices(after)
    make_proxy(proposal["support_point_world"], config)
    masks = add_group_masks(region, proposal, output_dir)
    add_support_outputs(region["support_object_id"], output_dir)
    render_image(after_path)
    report["checks"]["rendered_support"] = check_rendered_support(
        output_dir, proposal["support_point_world"], actual_camera, config)
    if not report["checks"]["rendered_support"]["pass"]:
        reasons.append("rendered_support_height")
    for mask in masks:
        path, = output_dir.glob(mask["prefix"] + "*.png")
        image = bpy.data.images.load(str(path), check_existing=False)
        pixels = np.asarray(image.pixels[:]).reshape(-1, 4)
        count = int(np.count_nonzero(pixels[:, 0] > 0.5))
        bpy.data.images.remove(image)
        report["visibility"].append({"group_id": mask["group_id"], "visible_pixels": count, "mask": str(path)})
        if count == 0:
            reasons.append(f"{mask['group_id']}: no_visible_pixels")
    report["images"] = {"before": str(before_path), "after": str(after_path)}
    report["numeric_pass"] = not reasons
    report["blender_version"] = bpy.app.version_string
    return report
