"""Blender 数值验证：固定12件实际 loader、几何不变、硬边/平面与浅角反例；不渲染。"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

import bpy
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "packages/cad_scene_factory"))
import render


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array(collection, field, width=1, dtype=np.float32):
    result = np.empty((len(collection), width), dtype=dtype)
    collection.foreach_get(field, result.ravel())
    return result


def geometry(obj):
    mesh = obj.data
    pieces = [array(mesh.vertices, "co", 3), array(mesh.loops, "vertex_index", dtype=np.int32),
              array(mesh.polygons, "loop_start", dtype=np.int32),
              array(mesh.polygons, "loop_total", dtype=np.int32)]
    bounds = render._world_bounds([obj])
    return {"vertices": len(mesh.vertices), "triangles": len(mesh.polygons),
            "all_triangles": bool(np.all(pieces[-1] == 3)),
            "coordinates_and_triangles_sha256": hashlib.sha256(b"".join(p.tobytes() for p in pieces)).hexdigest(),
            "world_transform": [list(row) for row in obj.matrix_world],
            "world_aabb": [list(bounds.minimum), list(bounds.maximum)]}


def normal_stats(mesh):
    mesh.update()
    face = array(mesh.polygons, "normal", 3).astype(np.float64)
    face /= np.linalg.norm(face, axis=1, keepdims=True)
    corner = array(mesh.corner_normals, "vector", 3)
    edge_faces = [[] for _ in mesh.edges]
    corner_map = []
    for polygon in mesh.polygons:
        corner_map.append({mesh.loops[i].vertex_index: i for i in polygon.loop_indices})
        for i in polygon.loop_indices:
            edge_faces[mesh.loops[i].edge_index].append(polygon.index)
    adjacency = [[] for _ in mesh.polygons]
    counts = {"gt30_edges": 0, "gt30_sharp": 0, "90_edges": 0, "90_sharp": 0,
              "60_edges": 0, "60_sharp": 0, "low_angle_edges": 0, "nonmanifold_edges": 0}
    low_jumps = []
    for edge, faces in zip(mesh.edges, edge_faces):
        if len(faces) != 2:
            counts["nonmanifold_edges"] += 1
            continue
        a, b = faces
        degrees = math.degrees(math.acos(float(np.clip(np.dot(face[a], face[b]), -1, 1))))
        for name, selected in [("gt30", degrees > 30.01), ("90", abs(degrees - 90) < 0.01),
                               ("60", abs(degrees - 60) < 0.01)]:
            counts[name + "_edges"] += int(selected)
            counts[name + "_sharp"] += int(selected and edge.use_edge_sharp)
        if 0.1 < degrees < 29.99:
            counts["low_angle_edges"] += 1
            low_jumps.extend(float(np.linalg.norm(corner[corner_map[a][v]] - corner[corner_map[b][v]]))
                             for v in edge.vertices)
        if not edge.use_edge_sharp:
            adjacency[a].append(b)
            adjacency[b].append(a)
    # 检查被硬边隔开的全共面连通组；不把任意局部三角形当成完整平面。
    seen, plane_errors, plane_faces = set(), [], 0
    for first in range(len(face)):
        if first in seen:
            continue
        queue, component = [first], []
        seen.add(first)
        while queue:
            current = queue.pop()
            component.append(current)
            for neighbor in adjacency[current]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
        if np.max(np.linalg.norm(face[component] - face[first], axis=1)) < 1e-4:
            loops = [i for f in component for i in mesh.polygons[f].loop_indices]
            plane_errors.append(float(np.max(np.linalg.norm(corner[loops] - face[first], axis=1))))
            plane_faces += len(component)
    return {"custom_normals": mesh.has_custom_normals,
            "smooth_faces": sum(p.use_smooth for p in mesh.polygons), **counts,
            "low_edge_normal_jump_max_vector": max(low_jumps, default=0),
            "planar_components": len(plane_errors), "planar_faces": plane_faces,
            "planar_normal_max_vector_error": max(plane_errors, default=0)}


def remove(obj):
    mesh = obj.data
    bpy.data.objects.remove(obj, do_unlink=True)
    bpy.data.meshes.remove(mesh)


def validate_file(path, expected_sha):
    source_sha = sha256(path)
    bpy.ops.wm.stl_import(filepath=str(path), forward_axis="Y", up_axis="Z", use_facet_normal=True)
    old = bpy.context.object
    old_geometry, old_normals = geometry(old), normal_stats(old.data)
    remove(old)
    objects = render._load_model(path)
    assert len(objects) == 1
    new_geometry, new_normals = geometry(objects[0]), normal_stats(objects[0].data)
    remove(objects[0])
    checks = {"source_matches_panel_and_unchanged": source_sha == expected_sha == sha256(path),
              "geometry_transform_aabb_identical": old_geometry == new_geometry,
              "all_triangles": new_geometry["all_triangles"],
              "no_custom_normals": not new_normals["custom_normals"],
              "all_faces_smooth": new_normals["smooth_faces"] == new_geometry["triangles"],
              "hard_edges_preserved": all(new_normals[k + "_edges"] == new_normals[k + "_sharp"]
                                           for k in ["gt30", "90", "60"]),
              "low_edge_normals_continuous": new_normals["low_edge_normal_jump_max_vector"] < 1e-5,
              "planar_components_preserved": new_normals["planar_normal_max_vector_error"] < 1e-4}
    new_geometry.pop("world_transform")  # 已逐值比较，报告省略单位矩阵。
    return {"model_id": path.stem, "stl_sha256": source_sha, "geometry": new_geometry,
            "old_custom_normals": old_normals["custom_normals"], "old_smooth_faces": old_normals["smooth_faces"],
            "old_low_edge_normal_jump_max_vector": old_normals["low_edge_normal_jump_max_vector"],
            "new_normals": new_normals, "checks": checks, "passed": all(checks.values())}


def shallow_fold():
    mesh = bpy.data.meshes.new("intentional_15_degree_fold")
    angle = math.radians(15)
    mesh.from_pydata([(0, -1, 0), (0, 1, 0), (-1, -1, 0), (-1, 1, 0),
                     (math.cos(angle), -1, math.sin(angle)), (math.cos(angle), 1, math.sin(angle))], [],
                    [(0, 1, 3), (0, 3, 2), (0, 4, 5), (0, 5, 1)])
    obj = bpy.data.objects.new(mesh.name, mesh)
    bpy.context.collection.objects.link(obj)
    before = geometry(obj)
    render._smooth_stl_mesh(mesh)
    stats = normal_stats(mesh)
    result = {"geometry_unchanged": before == geometry(obj), "new_normals": stats,
              "known_limitation_observed": sum(e.use_edge_sharp for e in mesh.edges) == 0,
              "expected_join_normal_tilt_degrees": 7.5}
    remove(obj)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stl-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("RESULTS_NORMALS.json"))
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    panel_path = ROOT / "packages/cad_scene_factory/review_models.json"
    panel, results = json.loads(panel_path.read_text()), []
    assert render.STL_SMOOTH_ANGLE_DEGREES == 30, "This validation protocol is fixed to 30 degrees"
    for model in panel["models"]:
        try:
            results.append(validate_file(args.stl_dir / f'{model["model_id"]}.stl', model["stl_sha256"]))
        except Exception as error:
            results.append({"model_id": model["model_id"], "passed": False, "error": repr(error)})
    fold = shallow_fold()
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "blender": bpy.app.version_string,
              "blender_build_hash": bpy.app.build_hash.decode(), "command": sys.argv,
              "render_source_sha256": sha256(Path(render.__file__)), "script_sha256": sha256(Path(__file__)),
              "panel_sha256": sha256(panel_path), "stl_dir": str(args.stl_dir), "threshold_degrees": 30,
              "counts": {"planned": len(results), "passed": sum(r["passed"] for r in results),
                         "failed": sum(not r["passed"] for r in results)},
              "models": results, "intentional_15_degree_fold": fold,
              "limitations": ["仅数值验证，不渲染，不评价照片真实感；固定12件不代表全库。",
                              "STL没有B-Rep曲面/硬边标签，30度以下设计折线可能被误平滑；15度反例已确认。",
                              "平面检查仅限硬边隔开的全共面连通组，浅角过渡旁单个平面三角形不保证面法线。",
                              "不恢复精确CAD法线，不改变低面数轮廓，不增加网格细节；30度非全库优化结果。"]}
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "counts": report["counts"]}, ensure_ascii=False))
    assert report["counts"]["failed"] == 0
    assert fold["geometry_unchanged"] and fold["known_limitation_observed"]


if __name__ == "__main__":
    main()
