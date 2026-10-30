"""Read-only STL corner-normal and screen-space diagnostics; run in Blender.

No scene file is saved and no renderer is invoked. Rasterization uses the saved
camera and object transforms and perspective-correct normal interpolation.
"""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import struct
import sys
import zlib

import bpy
from mathutils import Matrix
import numpy as np


def write_png(path, rgb):
    rgb = np.asarray(np.clip(rgb, 0, 255), dtype=np.uint8)
    height, width = rgb.shape[:2]

    def chunk(kind, data):
        return (struct.pack('!I', len(data)) + kind + data
                + struct.pack('!I', zlib.crc32(kind + data) & 0xffffffff))

    raw = b''.join(b'\0' + row.tobytes() for row in rgb)
    path.write_bytes(b'\x89PNG\r\n\x1a\n'
                     + chunk(b'IHDR', struct.pack('!2I5B', width, height, 8, 2, 0, 0, 0))
                     + chunk(b'IDAT', zlib.compress(raw)) + chunk(b'IEND', b''))


def rasterize(vertices, triangles, corner_normals, camera):
    width, height = camera['resolution']
    camera_points = np.c_[vertices, np.ones(len(vertices))] @ np.array(camera['world_to_camera_cv']).T
    projected = camera_points[:, :3] @ np.array(camera['K']).T
    screen = projected[:, :2] / projected[:, 2:3]
    depth = np.full((height, width), np.inf)
    face_ids = np.full((height, width), -1, dtype=int)
    normals = np.zeros((height, width, 3))
    for face, ids in enumerate(triangles):
        xy = screen[ids]
        z = camera_points[ids, 2]
        if (z <= 0).any():
            continue
        low = np.maximum(np.floor(xy.min(axis=0)).astype(int), 0)
        high = np.minimum(np.ceil(xy.max(axis=0)).astype(int), [width - 1, height - 1])
        if (high < low).any():
            continue
        matrix = np.vstack([xy.T, np.ones(3)])
        if abs(np.linalg.det(matrix)) < 1e-9:
            continue
        xx, yy = np.meshgrid(np.arange(low[0], high[0] + 1), np.arange(low[1], high[1] + 1))
        coordinates = np.stack([xx + .5, yy + .5, np.ones_like(xx)], axis=-1)
        weights = coordinates @ np.linalg.inv(matrix).T
        inverse_depth = (weights / z).sum(axis=-1)
        current_depth = 1 / np.maximum(inverse_depth, 1e-30)
        inside = (weights >= -1e-8).all(axis=-1) & (current_depth < depth[yy, xx])
        row, col = yy[inside], xx[inside]
        perspective = weights[inside] / z / inverse_depth[inside, None]
        current_normals = perspective @ corner_normals[face]
        current_normals /= np.linalg.norm(current_normals, axis=-1, keepdims=True)
        depth[row, col] = current_depth[inside]
        face_ids[row, col] = face
        normals[row, col] = current_normals
    return screen, depth, face_ids, normals


def inspect(request, model_id, output):
    sys.path.insert(0, str(Path(request['source_package']) / 'step3'))
    from part import load_part, remove_part

    source = Path(request['stl_dir']) / f'{model_id}.stl'
    raw = Path(request['output_dir']) / 'raw' / 'office' / model_id
    metadata = json.loads((raw / 'metadata.json').read_text())
    part = load_part(source, 30)
    part.matrix_world = Matrix(metadata['geometry']['model_to_world'])
    mesh = part.data
    points = np.array([vertex.co[:] for vertex in mesh.vertices])
    world = np.c_[points, np.ones(len(points))] @ np.array(part.matrix_world).T
    triangles = np.array([polygon.vertices[:] for polygon in mesh.polygons])
    face_normals = np.array([polygon.normal[:] for polygon in mesh.polygons], dtype=float)
    face_normals /= np.linalg.norm(face_normals, axis=-1, keepdims=True)
    loop_normals = np.array([normal.vector[:] for normal in mesh.corner_normals], dtype=float)
    corner_normals = np.array([loop_normals[list(polygon.loop_indices)] for polygon in mesh.polygons])
    corner_normals /= np.linalg.norm(corner_normals, axis=-1, keepdims=True)
    deviation = np.rad2deg(np.arccos(np.clip(np.sum(corner_normals * face_normals[:, None], axis=-1), -1, 1)))
    edge_faces = defaultdict(list)
    edge_sharp = {tuple(sorted(edge.vertices)): bool(edge.use_edge_sharp) for edge in mesh.edges}
    for face, triangle in enumerate(triangles):
        for index in range(3):
            edge_faces[tuple(sorted([triangle[index], triangle[(index + 1) % 3]]))].append(face)
    parents = list(range(len(triangles)))

    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    edge_records = []
    for edge, adjacent in edge_faces.items():
        if len(adjacent) != 2:
            continue
        angle = float(np.rad2deg(np.arccos(np.clip(np.dot(*face_normals[adjacent]), -1, 1))))
        jumps = []
        for vertex in edge:
            corners = [corner_normals[face, list(triangles[face]).index(vertex)] for face in adjacent]
            jumps.append(float(np.rad2deg(np.arccos(np.clip(np.dot(*corners), -1, 1)))))
        edge_records.append({'vertices': list(map(int, edge)), 'faces': adjacent,
                             'dihedral_degrees': angle, 'sharp': edge_sharp[edge],
                             'corner_jump_degrees': max(jumps)})
        if angle < .01:
            parents[root(adjacent[0])] = root(adjacent[1])
    screen, depth, face_ids, normals = rasterize(world[:, :3], triangles, corner_normals, metadata['camera'])
    visible = face_ids >= 0
    mask_image = bpy.data.images.load(str(raw / 'mask.png'), check_existing=False)
    source_mask = np.array(mask_image.pixels[:]).reshape(mask_image.size[1], mask_image.size[0], 4)[::-1, :, 0] > .5
    mask_difference = int(np.count_nonzero(source_mask != visible))
    bpy.data.images.remove(mask_image)
    visible_counts = np.bincount(face_ids[visible], minlength=len(triangles))
    groups = defaultdict(list)
    for face in range(len(triangles)):
        groups[root(face)].append(face)
    plane_groups = []
    for group in groups.values():
        if len(group) < 2:
            continue
        values = deviation[group]
        plane_groups.append({'faces': group, 'triangle_count': len(group),
                             'visible_pixels': int(visible_counts[group].sum()),
                             'max_corner_deviation_degrees': float(values.max()),
                             'corners_deviating_above_1_degree': int((values > 1).sum()),
                             'normal': face_normals[group[0]].tolist()})
    plane_groups.sort(key=lambda value: value['visible_pixels'], reverse=True)
    image = bpy.data.images.load(str(raw / 'rgb.png'), check_existing=False)
    original = np.array(image.pixels[:]).reshape(image.size[1], image.size[0], 4)[::-1, :, :3]
    # Blender exposes this byte PNG buffer as encoded display values. Do not
    # apply another sRGB transfer function (checked against the source PNG).
    overlay = np.clip(original * 255, 0, 255).astype(np.uint8)
    for record in edge_records:
        ids = record['vertices']
        a, b = screen[ids]
        count = max(2, int(np.linalg.norm(b - a) * 2))
        samples = a + np.linspace(0, 1, count)[:, None] * (b - a)
        xy = np.floor(samples).astype(int)
        xy = xy[(xy[:, 0] >= 0) & (xy[:, 0] < overlay.shape[1])
                & (xy[:, 1] >= 0) & (xy[:, 1] < overlay.shape[0])]
        if len(xy) == 0:
            continue
        selected = np.isin(face_ids[xy[:, 1], xy[:, 0]], record['faces'])
        xy = xy[selected]
        overlay[xy[:, 1], xy[:, 0]] = [255, 40, 40] if record['sharp'] else [40, 255, 100]
    normal_rgb = (normals * .5 + .5) * 255
    normal_rgb[~visible] = 32
    flat_rgb = np.full_like(normal_rgb, 32)
    flat_rgb[visible] = (face_normals[face_ids[visible]] * .5 + .5) * 255
    pixel_deviation = np.zeros_like(depth)
    pixel_deviation[visible] = np.rad2deg(np.arccos(np.clip(np.sum(normals[visible] * face_normals[face_ids[visible]], axis=-1), -1, 1)))
    selected_regions = []
    if model_id == '00301161':
        for name, xy in [('large_slope', (350, 250)), ('rear_vertical_plate', (400, 230)),
                         ('base_top', (365, 330))]:
            face = int(face_ids[xy[1], xy[0]])
            assert face >= 0
            group = groups[root(face)]
            region = np.isin(face_ids, group)
            yy, xx = np.where(region)
            luma = original[region] @ np.array([.2126, .7152, .0722])
            selected_regions.append({'name': name, 'sample_pixel': list(xy), 'faces': group,
                                     'triangle_count': len(group), 'visible_pixels': int(region.sum()),
                                     'bbox_px': [int(xx.min()), int(yy.min()), int(xx.max()), int(yy.max())],
                                     'geometric_normal_world': face_normals[face].tolist(),
                                     'corner_deviation_max_degrees': float(deviation[group].max()),
                                     'pixel_deviation_p50_p90_p99_max_degrees': np.percentile(pixel_deviation[region], [50, 90, 99, 100]).tolist(),
                                     'encoded_luma_p5_p50_p95': np.percentile(luma, [5, 50, 95]).tolist()})
        selected_faces = set(sum((region['faces'] for region in selected_regions), []))
        region_boundaries = [dict(record, projected_endpoints_px=screen[record['vertices']].tolist())
                             for record in edge_records
                             if record['sharp'] and all(face in selected_faces for face in record['faces'])]
    else:
        region_boundaries = []
    heatmap = np.zeros_like(normal_rgb)
    heatmap[..., 0] = np.minimum(pixel_deviation / 10, 1) * 255
    heatmap[..., 1] = np.minimum(pixel_deviation / 30, 1) * 120
    heatmap[~visible] = 32
    for name, rgb in [('edges', overlay), ('smooth_normals', normal_rgb), ('face_normals', flat_rgb), ('deviation', heatmap)]:
        write_png(output / f'{model_id}_{name}.png', rgb)
    result = {'model_id': model_id, 'stl_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
              'vertices': len(mesh.vertices), 'triangles': len(mesh.polygons),
              'all_faces_smooth': all(polygon.use_smooth for polygon in mesh.polygons),
              'has_custom_normals': mesh.has_custom_normals, 'sharp_edges': sum(edge_sharp.values()),
              'boundary_edges': sum(len(value) == 1 for value in edge_faces.values()),
              'nonmanifold_edges': sum(len(value) > 2 for value in edge_faces.values()),
              'maximum_corner_deviation_degrees': float(deviation.max()),
              'visible_pixels': int(visible.sum()),
              'source_mask_differing_pixels': mask_difference,
              'visible_normal_deviation_percentiles_degrees': np.percentile(pixel_deviation[visible], [50, 90, 99, 100]).tolist(),
              'edges_dihedral_above_30': sum(record['dihedral_degrees'] > 30 for record in edge_records),
              'smooth_edges_with_normal_jump_above_point1': sum(not record['sharp'] and record['corner_jump_degrees'] > .1 for record in edge_records),
              'largest_visible_coplanar_groups': plane_groups[:7],
              'noncoplanar_smooth_edges_count': sum(not record['sharp'] and record['dihedral_degrees'] > 1 for record in edge_records),
              'selected_regions': selected_regions, 'selected_region_boundaries': region_boundaries,
              'diagnostics': 'World-space normals; edges: red=sharp, green=smooth; deviation heatmap red saturates at 10 degrees.'}
    bpy.data.images.remove(image)
    remove_part(part)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    args.output.mkdir(parents=True, exist_ok=True)
    request = json.loads(args.request.read_text())
    reports = [inspect(request, model_id, args.output) for model_id in request['model_ids']]
    (args.output / 'mesh_normals.json').write_text(json.dumps({'blender': bpy.app.version_string, 'cases': reports}, indent=2) + '\n')
    print(json.dumps([{key: value for key, value in report.items() if key not in ('selected_regions', 'selected_region_boundaries', 'largest_visible_coplanar_groups')} for report in reports], indent=2))
