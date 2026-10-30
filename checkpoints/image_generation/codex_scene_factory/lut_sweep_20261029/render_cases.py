"""Blender-only diagnostic: same three parts in one existing room, CPU only."""

import argparse
import json
from pathlib import Path
import sys
import time

import bpy
from mathutils import Vector
import numpy as np


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def fit_camera(part, config, direction, fraction):
    """Choose distance from actual vertex projections, retaining view direction."""
    points = np.array([part.matrix_world @ vertex.co for vertex in part.data.vertices])
    center = (points.min(axis=0) + points.max(axis=0)) / 2
    backward = Vector(direction).normalized()
    rotation = (-backward).to_track_quat('-Z', 'Y').to_matrix()
    local = (points - center) @ np.asarray(rotation)
    tangent = np.tan(np.deg2rad(config['vertical_fov_degrees']) / 2)

    def projected(distance):
        return .5 + local[:, :2] / (2 * tangent * (distance - local[:, 2, None]))

    low, high = float(local[:, 2].max()) + .011, 3.0
    for _ in range(36):
        distance = (low + high) / 2
        pixels = projected(distance)
        if np.ptp(pixels, axis=0).max() > fraction:
            low = distance
        else:
            high = distance
    pixels = projected(high)
    assert pixels.min() >= .025 and pixels.max() <= .975, pixels
    proposal = {'camera_position_world': list(Vector(center) + backward * high),
                'look_at_world': center.tolist()}
    return proposal, {'distance_m': high, 'projected_bbox': [*pixels.min(axis=0), *pixels.max(axis=0)],
                      'projected_long_side_fraction': float(np.ptp(pixels, axis=0).max())}


def run(request, scene_id):
    source = Path(request['source_package'])
    sys.path[:0] = [str(source / 'step3'), str(source / 'step2'), str(source)]
    from part import load_part, place_part, make_material, remove_part
    from scene import open_scene, prepare_support_scene
    from spatial import SceneGeometry, bounds_of, above_contact, camera_spec
    from preview import configure_render, make_camera
    from capture import add_geometry_outputs, take_photo, camera_metadata
    from support import add_support_outputs, check_rendered_support

    case = next(item for item in request['scenes'] if item['id'] == scene_id)
    folder = Path(request['output_dir']) / 'raw' / scene_id
    folder.mkdir(parents=True, exist_ok=True)
    config = request['render_config']
    start = time.monotonic()
    open_scene(case['blend'])
    preparation = prepare_support_scene([{'support_object_id': case['support_id']}], folder / 'prepared_scene.blend')
    geometry = SceneGeometry(config)
    point = list(case['point'])
    height = geometry.support_height(case['support_id'], point[0], point[1], point[2])
    assert height is not None, 'Candidate does not hit the declared support surface'
    point[2] = height
    material = make_material(request['material'])
    write_json(folder / 'preparation.json', {'scene': case, 'support_point': point,
               'preparation': preparation, 'elapsed_seconds': time.monotonic() - start,
               'placement_scope': 'Diagnostic numeric checks; not Agent Step2 approval'})
    for model_id in request['model_ids']:
        sample_dir = folder / model_id
        sample_dir.mkdir(exist_ok=False)
        part = load_part(str(Path(request['stl_dir']) / f'{model_id}.stl'), 30)
        placement = place_part(part, point, request['longest_side_m'], config['proxy_size_m'])
        bounds = bounds_of([placement['world_bounds']['minimum'], placement['world_bounds']['maximum']])
        support = geometry.supported(case['support_id'], bounds, .01)
        envelope = above_contact(bounds, point[2], config)
        collisions = geometry.collisions(envelope)
        proposal, framing = fit_camera(part, config, request['camera_direction'], request['frame_fraction'])
        blockers = geometry.blockers(envelope, camera_spec(proposal['camera_position_world'], proposal['look_at_world'], config))
        checks = {'support': support, 'collision_ids': collisions, 'blocker_ids': blockers, 'framing': framing}
        write_json(sample_dir / 'preflight.json', checks)
        assert support['pass'] and not collisions and not blockers, checks

        configure_render(config)
        bpy.context.scene.cycles.use_denoising = True
        bpy.context.scene.cycles.use_adaptive_sampling = False
        camera = make_camera(proposal, config)
        outputs = add_geometry_outputs()
        add_support_outputs(case['support_id'], sample_dir)
        part.data.materials.clear()
        part.data.materials.append(material)
        part.pass_index = 1
        shot_start = time.monotonic()
        images, mask = take_photo(sample_dir, config['seed'], outputs)
        rendered_support = check_rendered_support(sample_dir, point, camera, config)
        record = {'scene_id': scene_id, 'model_id': model_id, 'status': 'success' if rendered_support['pass'] else 'failed',
                  'geometry': placement, 'checks': checks, 'rendered_support': rendered_support,
                  'camera': camera_metadata(camera, config), 'images': images, 'mask': mask,
                  'material': request['material'], 'seed': config['seed'], 'samples': config['samples'],
                  'device': 'CPU', 'blender': bpy.app.version_string, 'render_seconds': time.monotonic() - shot_start,
                  'color_management': {'view_transform': bpy.context.scene.view_settings.view_transform,
                    'look': bpy.context.scene.view_settings.look, 'exposure': bpy.context.scene.view_settings.exposure}}
        write_json(sample_dir / 'metadata.json', record)
        assert rendered_support['pass'], rendered_support
        remove_part(part)
        print(f'CASE_COMPLETE {scene_id}/{model_id}', flush=True)
    write_json(folder / 'summary.json', {'status': 'completed', 'count': len(request['model_ids']),
               'elapsed_seconds': time.monotonic() - start})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--scene_id', required=True)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    run(json.loads(args.request.read_text()), args.scene_id)
