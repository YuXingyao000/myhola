"""在 Blender 内核对少量真实产物：相机投影、变换、掩码、深度射线。无渲染。"""

import json
from pathlib import Path
import sys

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector
import numpy as np


run_dir = Path(sys.argv[sys.argv.index("--") + 1])
sys.path.insert(0, str(run_dir / "source" / "step3"))
from capture import prepare_placement
from part import load_part, place_part, remove_part

request = json.loads((run_dir / "request.json").read_text())
records = [json.loads(line) for line in (run_dir / "renders.jsonl").read_text().splitlines()]
assert len(records) == len(request["jobs"])
assert all(record["status"] == "success" for record in records)
assert len({job["model_id"] for job in request["jobs"]}) == len(records)
assert len({job["source"]["stl_sha256"] for job in request["jobs"]}) == len(records)
results = []
for record in records:
    placement = request["placements"][record["placement_index"]]
    prepare_placement(request, placement)
    part = load_part(record["stl_file"], request["config"]["smooth_angle_degrees"])
    geometry = place_part(part, placement["setup"]["support_point_world"], record["longest_side_m"],
                          request["step2_config"]["proxy_size_m"])
    assert geometry["triangle_count"] == record["source"]["triangle_count"]
    transform = np.array(geometry["model_to_world"])
    assert np.allclose(transform[:3, :3], np.eye(3) * geometry["scale"])
    assert np.allclose(transform, record["geometry"]["model_to_world"])
    points = np.array([list(part.matrix_world @ vertex.co) for vertex in part.data.vertices])
    assert abs(points[:, 2].min() - geometry["support_height_m"]) < 1e-6
    width, height = record["camera"]["resolution"]
    K = np.array(record["camera"]["K"])
    view = np.array(record["camera"]["world_to_camera_cv"])
    projection_errors = []
    for point in points[::max(1, len(points) // 100)]:
        projected = K @ (view @ [*point, 1])[:3]
        projected = projected[:2] / projected[2]
        reference = world_to_camera_view(bpy.context.scene, bpy.context.scene.camera, Vector(point))
        projection_errors.append(np.linalg.norm(projected - [reference.x * width, (1-reference.y) * height]))
    assert max(projection_errors) < 0.001

    mask_image = bpy.data.images.load(record["images"]["mask"], check_existing=False)
    depth_image = bpy.data.images.load(record["images"]["depth"], check_existing=False)
    assert tuple(mask_image.size) == tuple(depth_image.size) == (width, height)
    mask = np.array(mask_image.pixels[:]).reshape(height, width, 4)[::-1, :, 0] > 0.5
    depth = np.array(depth_image.pixels[:]).reshape(height, width, 4)[::-1, :, 0]
    assert int(mask.sum()) == record["mask"]["visible_pixels"]
    assert np.isfinite(depth[mask]).all() and (depth[mask] > 0).all()
    # 离轮廓至少两个像素，避开 Object Index / Z 非抗锯齿边缘。
    inner = mask.copy()
    for axis in (0, 1):
        for shift in (-2, -1, 1, 2):
            inner &= np.roll(mask, shift, axis)
    coordinates = np.argwhere(inner)
    assert len(coordinates)
    camera_world_cv = np.linalg.inv(view)
    origin = Vector(camera_world_cv[:3, 3])
    graph = bpy.context.evaluated_depsgraph_get()
    depth_errors, axial_errors = [], []
    for row, column in coordinates[np.linspace(0, len(coordinates)-1, min(30, len(coordinates)), dtype=int)]:
        direction_cv = np.linalg.inv(K) @ [column + 0.5, row + 0.5, 1]
        direction = Vector(camera_world_cv[:3, :3] @ direction_cv).normalized()
        hit, location, _, _, obj, _ = bpy.context.scene.ray_cast(graph, origin, direction, distance=100)
        assert hit and obj == part
        depth_errors.append(abs(float(depth[row, column]) - (location - origin).length))
        axial_errors.append(abs(float(depth[row, column]) - float((view @ [*location, 1])[2])))
    # 本机 Z pass 实测为轴向深度；与欧氏射线距离分开记录，不放宽阈值掩盖语义差异。
    assert max(axial_errors) < 1e-5, axial_errors
    results.append({"model_id": record["model_id"], "triangles": geometry["triangle_count"],
                    "max_projection_error_px": max(projection_errors),
                    "depth_rays": len(depth_errors), "depth_convention": "camera-axis Z in meters",
                    "max_ray_distance_difference_m": max(depth_errors),
                    "max_axial_depth_error_m": max(axial_errors),
                    "mask_pixels": int(mask.sum()), "rigid_pose_and_uniform_scale": "pass"})
    remove_part(part)
    assert not any(obj.name == "step3_part" for obj in bpy.context.scene.objects)

report = {"status": "pass", "samples": results, "blender_version": bpy.app.version_string}
(run_dir / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
