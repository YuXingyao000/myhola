"""只读核对 Step 2 支撑表面与 Step 3 实际网格；不改场景、不补拍数据集。"""

import json
from pathlib import Path
import sys

import bpy
from mathutils import Vector
from mathutils.bvhtree import BVHTree
import numpy as np


run_dir = Path(sys.argv[sys.argv.index("--") + 1])
output_dir = Path(sys.argv[sys.argv.index("--") + 2])
output_dir.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(run_dir / "source" / "step3"))
from capture import prepare_placement
from part import load_part, place_part, remove_part

request = json.loads((run_dir / "request.json").read_text())
records = [json.loads(line) for line in (run_dir / "renders.jsonl").read_text().splitlines()]
results = []
for record in records:
    placement = request["placements"][record["placement_index"]]
    prepare_placement(request, placement)
    scene = bpy.context.scene
    graph = bpy.context.evaluated_depsgraph_get()
    support_id = placement["region"]["support_object_id"]
    support = bpy.data.objects[support_id].evaluated_get(graph)
    mesh = support.to_mesh()
    vertices = [support.matrix_world @ vertex.co for vertex in mesh.vertices]
    support_tree = BVHTree.FromPolygons(vertices, [tuple(p.vertices) for p in mesh.polygons])
    support.to_mesh_clear()
    x, y, z = placement["setup"]["support_point_world"]
    heights, full_scene_hits = [], []
    # 独立从上方射向全部场景，不能只拿 Step 2 给出的 z 与自己比较。
    for dx in np.linspace(-.1, .1, 11):
        for dy in np.linspace(-.1, .1, 11):
            origin = Vector((x + dx, y + dy, z + .5))
            hit, normal, _, _ = support_tree.ray_cast(origin, Vector((0, 0, -1)), 1)
            if hit is not None:
                heights.append(float(hit.z))
            found, point, _, _, obj, _ = scene.ray_cast(graph, origin, Vector((0, 0, -1)), distance=1)
            if found:
                full_scene_hits.append({"object": obj.name, "z": float(point.z)})

    part = load_part(record["stl_file"], request["config"]["smooth_angle_degrees"])
    place_part(part, placement["setup"]["support_point_world"], record["longest_side_m"],
               request["step2_config"]["proxy_size_m"])
    graph = bpy.context.evaluated_depsgraph_get()
    evaluated = part.evaluated_get(graph)
    points = np.array([list(evaluated.matrix_world @ v.co) for v in evaluated.data.vertices])
    gaps = []
    for point in points:
        hit, _, _, _ = support_tree.ray_cast(Vector((point[0], point[1], z + .5)), Vector((0, 0, -1)), 1)
        if hit is not None:
            gaps.append(float(point[2] - hit.z))
    result = {
        "model_id": record["model_id"], "region_id": placement["setup"]["region_id"],
        "support_object_id": support_id, "recorded_support_z_m": z,
        "support_mesh_height_range_m": [min(heights), max(heights)],
        "full_scene_hit_objects": sorted({item["object"] for item in full_scene_hits}),
        "full_scene_hit_height_range_m": [min(item["z"] for item in full_scene_hits), max(item["z"] for item in full_scene_hits)],
        "actual_part_z_range_m": [float(points[:, 2].min()), float(points[:, 2].max())],
        "min_vertex_surface_gap_m": min(gaps),
        "vertices_below_surface_0_01mm": sum(gap < -1e-5 for gap in gaps),
        "vertex_count": len(points),
        "support_modifiers": [{"name": m.name, "type": m.type, "viewport": m.show_viewport, "render": m.show_render}
                              for m in bpy.data.objects[support_id].modifiers],
        "support_materials": [{"name": mat.name, "displacement_method": mat.displacement_method,
                               "displacement_connected": any(n.bl_idname == 'ShaderNodeOutputMaterial' and n.inputs['Displacement'].is_linked
                                                             for n in mat.node_tree.nodes) if mat.use_nodes else False}
                              for slot in bpy.data.objects[support_id].material_slots if (mat := slot.material) is not None],
    }
    # 渲染深度含 Cycles 材质位移；与 Blender 普通 depsgraph 网格独立比较。
    depth_image = bpy.data.images.load(record["images"]["depth"], check_existing=False)
    width, height = depth_image.size
    depth = np.array(depth_image.pixels[:]).reshape(height, width, 4)[::-1, :, 0]
    mask_image = bpy.data.images.load(record["images"]["mask"], check_existing=False)
    mask = np.array(mask_image.pixels[:]).reshape(height, width, 4)[::-1, :, 0] > .5
    rows, columns = np.indices((height, width))
    pixels = np.stack([columns + .5, rows + .5, np.ones_like(rows)], axis=-1)
    K = np.array(record["camera"]["K"])
    view = np.array(record["camera"]["world_to_camera_cv"])
    inverse_view = np.linalg.inv(view)
    camera_points = (pixels @ np.linalg.inv(K).T) * depth[:, :, None]
    world_points = camera_points @ inverse_view[:3, :3].T + inverse_view[:3, 3]
    nearby = ((abs(world_points[:, :, 0] - x) < .11) & (abs(world_points[:, :, 1] - y) < .11)
              & (abs(world_points[:, :, 2] - z) < .2) & ~mask)
    samples = np.argwhere(nearby)
    actual_surface_heights = []
    for row, column in samples[::max(1, len(samples) // 500)]:
        direction = Vector(world_points[row, column] - inverse_view[:3, 3]).normalized()
        found, _, _, _, obj, _ = scene.ray_cast(graph, Vector(inverse_view[:3, 3]), direction, distance=100)
        if found and obj.name == support_id:
            actual_surface_heights.append(float(world_points[row, column, 2]))
    result["rendered_surface_samples"] = len(actual_surface_heights)
    result["rendered_surface_height_range_m"] = [min(actual_surface_heights), max(actual_surface_heights)]
    result["rendered_surface_minus_recorded_mm"] = [float(value) for value in
                                                   (np.quantile(actual_surface_heights, [0, .5, 1]) - z) * 1000]
    result["displacement_nodes"] = []
    for slot in bpy.data.objects[support_id].material_slots:
        material = slot.material
        for node in material.node_tree.nodes:
            if node.bl_idname == "ShaderNodeDisplacement":
                result["displacement_nodes"].append({"node": node.name,
                                                     "space": node.space,
                                                     "inputs": {socket.name: float(socket.default_value) for socket in node.inputs
                                                                if socket.type == 'VALUE'},
                                                     "source": [{"input": link.to_socket.name,
                                                                 "node": link.from_node.name} for link in material.node_tree.links
                                                                if link.to_node == node]})
    results.append(result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    remove_part(part)

(output_dir / "support_audit.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
