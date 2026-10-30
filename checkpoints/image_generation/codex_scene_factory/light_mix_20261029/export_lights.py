"""Blender experiment: export linear light contributions at an existing CAD shot.

Only light-group assignments and output passes are added. Source scenes, light
powers, materials, CAD vertices, placement, and camera are never edited on disk.
The default is one pilot CAD; expanding the model list is an explicit next step.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import bpy
from mathutils import Matrix
import numpy as np


GROUPS = ("sun", "daylight", "world", "ceiling", "task", "emission", "other_lights")


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def socket_value(socket):
    value = getattr(socket, "default_value", None)
    return list(value) if hasattr(value, "__len__") and not isinstance(value, str) else value


def node_description(tree):
    """Record values and links, including nested shader/compositor groups."""
    if tree is None:
        return None
    return {
        "name": tree.name,
        "nodes": [{"name": node.name, "type": node.bl_idname, "mute": node.mute,
                   "inputs": {socket.name: socket_value(socket) for socket in node.inputs},
                   "image": node.image.filepath if getattr(node, "image", None) else None,
                   "group": node_description(node.node_tree) if node.type == "GROUP" else None}
                  for node in tree.nodes],
        "links": [[link.from_node.name, link.from_socket.name, link.to_node.name, link.to_socket.name]
                  for link in tree.links],
    }


def emission_nodes(tree, prefix=""):
    """Conservative emitter inventory; all objects are grouped regardless.

    A linked value is listed as a possible emitter, not evaluated as a constant.
    Unused material slots are omitted by the caller. Grouping every object also
    covers emitters hidden inside shader groups or procedural geometry.
    """
    found = []
    if tree is None:
        return found
    for node in tree.nodes:
        name = prefix + node.name
        if node.type in {"EMISSION", "BACKGROUND", "VOLUME_PRINCIPLED"}:
            found.append({"node": name, "type": node.bl_idname,
                          "inputs": {s.name: {"value": socket_value(s), "linked": s.is_linked}
                                     for s in node.inputs}})
        elif node.type == "BSDF_PRINCIPLED":
            color, strength = node.inputs["Emission Color"], node.inputs["Emission Strength"]
            if (color.is_linked or max(color.default_value[:3]) > 0) and \
                    (strength.is_linked or strength.default_value > 0):
                found.append({"node": name, "type": node.bl_idname,
                              "color": socket_value(color), "strength": socket_value(strength),
                              "color_linked": color.is_linked, "strength_linked": strength.is_linked})
        if node.type == "GROUP":
            found.extend(emission_nodes(node.node_tree, name + "/"))
    return found


def used_materials(obj):
    materials = getattr(obj.data, "materials", [])
    if obj.type == "MESH":
        indices = {polygon.material_index for polygon in obj.data.polygons}
        return [materials[i] for i in sorted(indices) if i < len(materials) and materials[i]]
    return [material for material in materials if material]


def group_name(obj):
    """Map the actual three rooms' source names to physical light roles."""
    text = (obj.name + " " + " ".join(m.name for m in used_materials(obj))).lower()
    if obj.type == "LIGHT":
        if obj.data.type == "SUN":
            return "sun"
        if "window" in text or "daylight" in text:
            return "daylight"
        if any(word in text for word in ("panel emitted", "work light", "overhead")):
            return "ceiling"
        if any(word in text for word in ("task", "checking reflector")):
            return "task"
        return "other_lights"
    if any(word in text for word in ("ceiling", "batten diffuser", "opal diffuser")):
        return "ceiling"
    if any(word in text for word in ("desk_lamp", "task light diffuser", "articulated lamp")):
        return "task"
    return "emission"


def world_centroid(obj):
    """Mean evaluated mesh vertex position; used only to match emitting fixtures."""
    evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh = evaluated.to_mesh()
    vertices = np.empty(len(mesh.vertices) * 3, dtype=np.float64)
    mesh.vertices.foreach_get("co", vertices)
    local = vertices.reshape(-1, 3).mean(axis=0)
    matrix = np.asarray(evaluated.matrix_world, dtype=np.float64)
    center = matrix[:3, :3] @ local + matrix[:3, 3]
    evaluated.to_mesh_clear()
    return center


def source_group_plan():
    """One group per light, with nearby ceiling diffuser emission kept together.

    Names are stable for a fixed scene: sort original object names, then number.
    Other emitting meshes get individual groups; all remaining geometry shares
    other_emission, ensuring unrecognized procedural emission is still included.
    """
    objects = sorted(bpy.data.objects, key=lambda obj: obj.name)
    lights = [obj for obj in objects if obj.type == "LIGHT"]
    assignments = {obj.name: "other_emission" for obj in objects}
    light_names = {obj.name: f"source_{index + 1:02d}" for index, obj in enumerate(lights)}
    assignments.update(light_names)
    ceiling_lights = [obj for obj in lights if obj.data.type == "AREA" and group_name(obj) == "ceiling"]
    groups = [*light_names.values(), "world", "other_emission"]
    matches = []
    emitter_count = 0
    for obj in objects:
        if obj.type == "LIGHT" or not any(emission_nodes(m.node_tree) for m in used_materials(obj)):
            continue
        if obj.type == "MESH" and group_name(obj) == "ceiling" and ceiling_lights:
            center = world_centroid(obj)
            closest = min(ceiling_lights, key=lambda lamp: np.linalg.norm(center - np.asarray(lamp.matrix_world.translation)))
            assignments[obj.name] = light_names[closest.name]
            matches.append({"emitter": obj.name, "group": assignments[obj.name],
                            "matched_light": closest.name, "centroid_world": center.tolist(),
                            "light_position_world": list(closest.matrix_world.translation),
                            "distance_m": float(np.linalg.norm(center - np.asarray(closest.matrix_world.translation))),
                            "method": "nearest ceiling AREA to evaluated mesh vertex-mean centroid"})
        else:
            emitter_count += 1
            group = f"emitter_{emitter_count:02d}"
            assignments[obj.name] = group
            groups.append(group)
    return assignments, groups, matches


def assign_light_groups(scene, mode="role"):
    layer = bpy.context.view_layer
    if mode == "source":
        assignment, groups, matches = source_group_plan()
    else:
        assignment = {obj.name: group_name(obj) for obj in bpy.data.objects}
        groups, matches = list(GROUPS), []
    existing = [group.name for group in layer.lightgroups]
    for name in groups:
        if name not in existing:
            bpy.ops.scene.view_layer_add_lightgroup(name=name)
    # Include objects inside collection instances, not just scene.objects.
    lamps, emitters = [], []
    for obj in bpy.data.objects:
        group = assignment[obj.name]
        obj.lightgroup = group
        if obj.type == "LIGHT":
            light = obj.data
            lamps.append({"name": obj.name, "group": group, "type": light.type,
                          "role": group_name(obj), "position_world": list(obj.matrix_world.translation),
                          "energy": light.energy, "color": list(light.color),
                          "matrix_world": [list(row) for row in obj.matrix_world],
                          "hide_render": obj.hide_render,
                          "size": getattr(light, "size", None), "size_y": getattr(light, "size_y", None),
                          "angle": getattr(light, "angle", None),
                          "spot_size": getattr(light, "spot_size", None),
                          "nodes": node_description(light.node_tree)})
        materials = [{"material": material.name, "nodes": emission_nodes(material.node_tree)}
                     for material in used_materials(obj)]
        materials = [material for material in materials if material["nodes"]]
        if materials:
            emitters.append({"object": obj.name, "group": group, "hide_render": obj.hide_render,
                             "materials": materials})
    worlds = []
    for world in bpy.data.worlds:
        world.lightgroup = "world"
        worlds.append({"name": world.name, "active": world == scene.world,
                       "group": "world", "color": list(world.color),
                       "nodes": node_description(world.node_tree)})
    layer.update_render_passes()
    return {"mode": mode, "groups": groups, "object_assignments": assignment,
            "ceiling_emission_matches": matches,
            "lights": lamps, "emissive_material_objects": emitters, "worlds": worlds,
            "unassigned_objects": [obj.name for obj in bpy.data.objects if not obj.lightgroup],
            "classification": "All objects receive a group. Nonemissive objects contribute zero emitted light; "
                              "emissive material listing is conservative. Group values include all bounces."}


def add_outputs(folder, groups):
    scene, layer = bpy.context.scene, bpy.context.view_layer
    layer.use_pass_normal = layer.use_pass_z = layer.use_pass_object_index = True
    layer.cycles.denoising_store_passes = True
    layer.update_render_passes()
    tree = scene.compositing_node_group
    if tree is None:
        tree = bpy.data.node_groups.new("light_mix_outputs", "CompositorNodeTree")
        scene.compositing_node_group = tree
        tree.interface.new_socket(name="Image", in_out="OUTPUT", socket_type="NodeSocketColor")
        output = tree.nodes.new("NodeGroupOutput")
        layers = tree.nodes.new("CompositorNodeRLayers")
        layers.layer = layer.name
        tree.links.new(layers.outputs["Image"], output.inputs["Image"])
    layers = tree.nodes.new("CompositorNodeRLayers")
    layers.layer = layer.name
    scene.render.use_compositing = True
    passes = {"combined_noisy": "Noisy Image", "combined_denoised": "Image", "normal": "Normal",
              "denoising_normal": "Denoising Normal", "denoising_albedo": "Denoising Albedo",
              "denoising_specular_albedo": "Denoising Specular Albedo", "depth": "Depth"}
    passes.update({"light_" + group: "Combined_" + group for group in groups})
    for name, socket in passes.items():
        output = tree.nodes.new("CompositorNodeOutputFile")
        output.directory = str(folder)
        output.file_name = name + "_"
        output.format.media_type = "IMAGE"
        output.format.file_format = "OPEN_EXR"
        output.format.color_mode = "RGBA"
        output.format.color_depth = "32"
        output.format.exr_codec = "ZIP"
        output.save_as_render = False
        output.file_output_items.new("RGBA", "Image")
        tree.links.new(layers.outputs[socket], output.inputs[0])
    mask = tree.nodes.new("CompositorNodeIDMask")
    mask.inputs["Index"].default_value = 1
    mask.inputs["Anti-Alias"].default_value = False
    tree.links.new(layers.outputs["Object Index"], mask.inputs[0])
    output = tree.nodes.new("CompositorNodeOutputFile")
    output.directory = str(folder)
    output.file_name = "mask_"
    output.format.media_type = "IMAGE"
    output.format.file_format = "PNG"
    output.format.color_mode = "BW"
    output.format.color_depth = "8"
    output.save_as_render = False
    output.file_output_items.new("FLOAT", "mask")
    tree.links.new(mask.outputs[0], output.inputs[0])
    return passes


def read_pixels(path):
    image = bpy.data.images.load(str(path), check_existing=False)
    pixels = np.empty(len(image.pixels), dtype=np.float32)
    image.pixels.foreach_get(pixels)
    pixels = pixels.reshape(image.size[1], image.size[0], 4)
    bpy.data.images.remove(image)
    return pixels


def run(request_file, output_dir, scene_id, model_id, probe_only, grouping_mode="role"):
    request = json.loads(request_file.read_text())
    source = Path(request["source_package"])
    sys.path[:0] = [str(source / "step3"), str(source / "step2"), str(source)]
    from part import load_part, make_material
    from preview import configure_render, make_camera

    original = Path(request["output_dir"]) / "raw" / scene_id
    source_blend = original / "prepared_scene.blend"
    metadata_file = original / model_id / "metadata.json"
    prior = json.loads(metadata_file.read_text())
    folder = output_dir / scene_id / model_id
    folder.mkdir(parents=True, exist_ok=False)
    bpy.ops.wm.open_mainfile(filepath=str(source_blend))
    scene = bpy.context.scene
    color = scene.view_settings
    original_settings = {
        "color_management": {"view_transform": color.view_transform, "look": color.look,
                             "exposure": color.exposure, "gamma": color.gamma,
                             "display_device": scene.display_settings.display_device},
        "use_compositing": scene.render.use_compositing,
        "compositor": node_description(scene.compositing_node_group),
    }
    config = dict(request["render_config"], samples=64, threads=8, seed=prior["seed"])
    configure_render(config)
    scene.cycles.use_denoising = True
    scene.cycles.use_adaptive_sampling = False
    part = load_part(str(Path(request["stl_dir"]) / (model_id + ".stl")), 30)
    part.matrix_world = Matrix(prior["geometry"]["model_to_world"])
    part.data.materials.clear()
    part.data.materials.append(make_material(prior["material"]))
    matrix = Matrix(prior["camera"]["camera_to_world_blender"])
    position = matrix.translation
    proposal = {"camera_position_world": list(position), "look_at_world": [position[0], position[1], position[2]-1]}
    camera = make_camera(proposal, config)
    camera.matrix_world = matrix
    bpy.context.view_layer.update()
    for obj in bpy.data.objects:
        obj.pass_index = 0
    part.pass_index = 1
    assert len(part.data.vertices) == prior["geometry"]["vertex_count"]
    assert len(part.data.polygons) == prior["geometry"]["triangle_count"]
    grouping = assign_light_groups(scene, grouping_mode)
    assert not grouping["unassigned_objects"]
    passes = add_outputs(folder, grouping["groups"])
    record = {
        "status": "prepared", "scene_id": scene_id, "model_id": model_id,
        "source_request": str(request_file), "source_request_sha256": sha256(request_file),
        "source_scene": str(source_blend), "source_scene_sha256": sha256(source_blend),
        "source_metadata": str(metadata_file), "source_metadata_sha256": sha256(metadata_file),
        "export_script_sha256": sha256(__file__), "blender": bpy.app.version_string,
        "device": "CPU", "threads": 8, "samples": 64, "seed": prior["seed"],
        "resolution": config["resolution"], "original_settings": original_settings,
        "geometry": prior["geometry"], "camera": prior["camera"], "material": prior["material"],
        "light_groups": grouping, "passes": passes,
        "pass_semantics": {"exr": "scene-linear Rec.709 RGB, float32, no display transform",
                           "light_groups": "Noisy linear radiance per source group, includes direct and indirect paths",
                           "combined_noisy": "Reference for sum(groups), not the denoised Combined",
                           "combined_denoised": "Original Cycles OIDN denoised Image, before display transform",
                           "normal": "Cycles Normal pass; not RGB albedo",
                           "mask": "Non-antialiased visible target object index, same contract as original shot"},
        "sum_condition": "Sum all light groups in scene-linear RGB with coefficient 1, before denoising, "
                         "tone mapping, clipping, or compositing; compare combined_noisy. Denoising is nonlinear.",
        "compositor_note": "Original scene compositor is recorded before configure_render. Source files are not saved; "
                           "existing FileOutput nodes are removed in memory by the frozen rendering helper.",
    }
    write_json(folder / "metadata.json", record)
    if probe_only:
        return
    scene.render.filepath = str(folder / "original.png")
    started = time.monotonic()
    bpy.ops.render.render(write_still=True)
    for name in passes:
        path, = folder.glob(name + "_*.exr")
        path.rename(folder / (name + ".exr"))
    path, = folder.glob("mask_*.png")
    path.rename(folder / "mask.png")
    old_mask = read_pixels(original / model_id / "mask.png")[:, :, 0]
    new_mask = read_pixels(folder / "mask.png")[:, :, 0]
    target = new_mask > 0.5
    group_radiance = {}
    total = np.zeros((*target.shape, 3), dtype=np.float64)
    for group in grouping["groups"]:
        pixels = read_pixels(folder / ("light_" + group + ".exr"))[:, :, :3].astype(np.float64)
        luminance = pixels @ np.array([0.2126, 0.7152, 0.0722])
        total += pixels
        group_radiance[group] = {"mean_rgb": pixels.mean(axis=(0, 1)).tolist(),
                                 "mean_luminance": float(luminance.mean()),
                                 "target_mean_luminance": float(luminance[target].mean()),
                                 "background_mean_luminance": float(luminance[~target].mean()),
                                 "max_luminance": float(luminance.max())}
    combined = read_pixels(folder / "combined_noisy.exr")[:, :, :3].astype(np.float64)
    error = np.abs(total - combined)
    record.update(status="completed", render_seconds=time.monotonic()-started,
                  group_radiance=group_radiance,
                  sum_check={"mean_abs_error": float(error.mean()), "max_abs_error": float(error.max()),
                             "relative_l1": float(error.sum()/max(np.abs(combined).sum(), 1e-12)),
                             "all_finite": bool(np.isfinite(total).all() and np.isfinite(combined).all())},
                  mask_changed_pixels=int(np.count_nonzero(old_mask != new_mask)),
                  source_scene_unchanged=sha256(source_blend)==record["source_scene_sha256"],
                  actual_camera_matrix=[list(row) for row in camera.matrix_world],
                  actual_model_to_world=[list(row) for row in part.matrix_world])
    write_json(folder / "metadata.json", record)
    print(json.dumps({key: record[key] for key in ("scene_id", "model_id", "status", "render_seconds",
                                                  "sum_check", "mask_changed_pixels")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--scene_id", choices=("office", "factory", "assembly"), required=True)
    parser.add_argument("--model_id", default="00301161")
    parser.add_argument("--grouping", choices=("role", "source"), default="role")
    parser.add_argument("--probe_only", action="store_true")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    run(args.request.resolve(), args.output_dir.resolve(), args.scene_id, args.model_id, args.probe_only, args.grouping)
