"""冻结十件的 I/B/F/G/S 渲染；只在 Blender 中执行，不修改原场景文件。"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

import bpy
from mathutils import Matrix
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from render_pairs import add_radiance_output, compare_source, read_json, read_numeric_image
from render_pairs import render_view, sha256, write_json

LAYERS = ('with_part', 'empty', 'foreground', 'interaction', 'shadow_only')
RAY_FLAGS = ('hide_render', 'visible_camera', 'visible_diffuse', 'visible_glossy',
             'visible_transmission', 'visible_volume_scatter', 'visible_shadow', 'is_holdout')
GEOMETRY_TYPES = {'MESH', 'CURVE', 'SURFACE', 'META', 'FONT', 'CURVES', 'POINTCLOUD', 'VOLUME'}


def geometry_signature(part):
    mesh = part.data
    arrays = {
        'vertices': np.array([v.co[:] for v in mesh.vertices], dtype=np.float32),
        'triangles': np.array([p.vertices[:] for p in mesh.polygons], dtype=np.int32),
        'corner_normals': np.array([n.vector[:] for n in mesh.corner_normals], dtype=np.float32),
        'smooth': np.array([p.use_smooth for p in mesh.polygons], dtype=np.uint8),
        'sharp': np.array([e.use_edge_sharp for e in mesh.edges], dtype=np.uint8),
        'model_to_world': np.array(part.matrix_world, dtype=np.float64),
    }
    return {key: hashlib.sha256(value.tobytes()).hexdigest() for key, value in arrays.items()}


def add_linear_outputs(outputs):
    """直接保存 Combined 与现有 Noisy Image；不改变 beauty 去噪设置。"""
    add_radiance_output(outputs)
    outputs['radiance'].format.color_mode = 'RGBA'
    scene = bpy.context.scene
    tree = scene.compositing_node_group
    layer = tree.nodes.new('CompositorNodeRLayers')
    layer.layer = bpy.context.view_layer.name
    socket = layer.outputs.get('Noisy Image')
    available = socket is not None and socket.enabled
    if available:
        output = tree.nodes.new('CompositorNodeOutputFile')
        output.file_name = 'noisy_'
        output.format.media_type = 'IMAGE'
        output.format.file_format = 'OPEN_EXR'
        output.format.color_mode = 'RGBA'
        output.format.color_depth = '32'
        output.format.exr_codec = 'ZIP'
        output.save_as_render = False
        output.file_output_items.new('RGBA', 'noisy')
        tree.links.new(socket, output.inputs[0])
        outputs['noisy'] = output
    return {'available': available, 'socket': 'Noisy Image' if available else None,
            'reason': 'Existing Cycles output' if available else 'No enabled Noisy Image socket',
            'beauty_denoising_changed': False,
            'denoising_store_passes': bpy.context.view_layer.cycles.denoising_store_passes}


def capture_layers(pair, request, record):
    from capture import prepare_placement, camera_metadata
    from part import load_part, make_material, check_support_contact

    capture, source = pair['capture_request'], pair['source_metadata']
    _, outputs = prepare_placement(capture, pair['placement'])
    scene = bpy.context.scene
    scene.camera.matrix_world = Matrix(source['camera']['camera_to_world_blender'])
    part = load_part(Path(pair['inputs']['stl']), source['smooth_angle_degrees'])
    part.matrix_world = Matrix(source['geometry']['model_to_world'])
    part.data.materials.clear()
    part.data.materials.append(make_material(source['material']))
    part.pass_index = 1
    bpy.context.view_layer.update()
    before = geometry_signature(part)
    directory = Path(request['output_dir']) / pair['model_id']
    part_flags = {name: getattr(part, name) for name in RAY_FLAGS}
    environment = [(obj, obj.is_holdout) for obj in scene.objects
                   if obj != part and obj.type in GEOMETRY_TYPES]
    film, png_mode = scene.render.film_transparent, scene.render.image_settings.color_mode
    record.update(
        geometry_before=before, material=source['material'],
        contact=check_support_contact(part, pair['placement']['support_validation']),
        camera=camera_metadata(scene.camera, capture['step2_config']),
        model_to_world=[list(row) for row in part.matrix_world],
        camera_matrix_max_abs_error=float(np.abs(np.array(scene.camera.matrix_world) -
            source['camera']['camera_to_world_blender']).max()),
        model_matrix_max_abs_error=float(np.abs(np.array(part.matrix_world) -
            source['geometry']['model_to_world']).max()),
        render_device=capture['render_device'], blender_version=bpy.app.version_string,
        render_settings={'samples': scene.cycles.samples, 'seed': pair['render_seed'],
                         'denoising': scene.cycles.use_denoising,
                         'adaptive_sampling': scene.cycles.use_adaptive_sampling},
        color_management={key: getattr(scene.view_settings, key)
                          for key in ['view_transform', 'look', 'exposure', 'gamma']},
        noisy_output=add_linear_outputs(outputs), render_seconds={}, layer_settings={},
        layers_completed=[], output_sha256={})

    def restore():
        for name, value in part_flags.items():
            setattr(part, name, value)
        for obj, holdout in environment:
            obj.is_holdout = holdout
        scene.render.film_transparent = film
        scene.render.image_settings.color_mode = png_mode

    for name in LAYERS:
        restore()
        if name == 'empty':
            part.hide_render = True
        elif name == 'foreground':
            scene.render.film_transparent = True
            scene.render.image_settings.color_mode = 'RGBA'
            for obj, _ in environment:
                obj.is_holdout = True
        elif name in {'interaction', 'shadow_only'}:
            part.visible_camera = False
            if name == 'shadow_only':
                for flag in ('visible_diffuse', 'visible_glossy', 'visible_transmission',
                             'visible_volume_scatter'):
                    setattr(part, flag, False)
                part.visible_shadow = True
        bpy.context.view_layer.update()
        assert geometry_signature(part) == before
        record['layer_settings'][name] = {
            'part_visibility': {key: getattr(part, key) for key in RAY_FLAGS},
            'film_transparent': scene.render.film_transparent,
            'png_color_mode': scene.render.image_settings.color_mode,
            'environment_holdout_count': sum(obj.is_holdout for obj, _ in environment),
        }
        print(f"{pair['model_id']} · {name}", flush=True)
        images, seconds = render_view(directory / name, pair['render_seed'], outputs)
        if 'noisy' in outputs:
            path, = (directory / name).glob('noisy_*.exr')
            path.rename(directory / name / 'noisy.exr')
            images['noisy'] = str(directory / name / 'noisy.exr')
        record[name] = images
        record['render_seconds'][name] = seconds
        record['layers_completed'].append(name)
        record['output_sha256'].update({str(Path(path).relative_to(directory)): sha256(path)
                                       for path in images.values()})
        write_json(directory / 'metadata.json', record)
    restore()
    bpy.context.view_layer.update()
    record['geometry_after'] = geometry_signature(part)
    assert record['geometry_after'] == before
    record['comparison_to_source'] = compare_source(pair, record['with_part'], record['empty'])
    alpha = read_numeric_image(record['foreground']['rgb'])[:, :, 3]
    if not np.isfinite(alpha).all() or not (alpha > 0).any() or (alpha > 0).all():
        raise ValueError('Foreground alpha must be finite, nonempty and not cover the entire frame')
    record['foreground_alpha'] = {'minimum': float(alpha.min()), 'maximum': float(alpha.max()),
        'area': float(alpha.sum()), 'nonzero_pixels': int((alpha > 0).sum()),
        'opaque_pixels': int((alpha == 1).sum()),
        'partial_coverage_pixels': int(((alpha > 0) & (alpha < 1)).sum())}
    record['layer_masks'] = {}
    record['linear_outputs'] = {}
    for name in LAYERS:
        mask = read_numeric_image(record[name]['mask'])[:, :, 0] > .5
        record['layer_masks'][name] = {'visible_pixels': int(mask.sum())}
        if name in {'empty', 'interaction', 'shadow_only'} and mask.any():
            raise ValueError(f'{name} unexpectedly contains CAD in its camera mask')
        for kind in ('radiance', 'noisy'):
            if kind in record[name]:
                values = read_numeric_image(record[name][kind])
                if not np.isfinite(values).all():
                    raise ValueError(f'{name}/{kind}.exr contains nonfinite values')
                record['linear_outputs'][f'{name}/{kind}'] = {
                    'minimum': float(values.min()), 'maximum': float(values.max()),
                    'alpha_min': float(values[:, :, 3].min()),
                    'alpha_max': float(values[:, :, 3].max()), 'finite': True}
    record['status'] = 'success'


def run_request(path, limit):
    request = read_json(path)
    output = Path(request['output_dir'])
    request_hash = sha256(path)
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '7'
    for filename, digest in request['input_sha256'].items():
        assert sha256(filename) == digest, filename
    for filename, digest in request['source_sha256'].items():
        assert sha256(output / 'source' / filename) == digest, filename
    for filename, digest in request['script_sha256'].items():
        assert sha256(output / filename) == digest, filename
    assert sha256(__file__) == request['script_sha256']['render_layers.py']
    sys.path[:0] = [str(output / 'source/step3'), str(output / 'source/step2'), str(output / 'source')]
    pairs = request['pairs'] if limit is None else request['pairs'][:limit]
    with (output / 'renders.jsonl').open('a') as stream:
        for pair in pairs:
            directory = output / pair['model_id']
            metadata_path = directory / 'metadata.json'
            if metadata_path.exists():
                existing = read_json(metadata_path)
                assert existing['request_sha256'] == request_hash, metadata_path
                if existing['status'] == 'success':
                    for filename, digest in existing['output_sha256'].items():
                        assert sha256(directory / filename) == digest, filename
                print(f"保留既有记录 {pair['model_id']} · {existing['status']}，不重试", flush=True)
                continue
            directory.mkdir()
            record = {key: pair[key] for key in ('model_id', 'render_seed', 'enhancement_seed',
                                                'transform_sha256', 'material_sha256')}
            record.update(status='running', request_sha256=request_hash)
            write_json(metadata_path, record)
            started = time.perf_counter()
            try:
                capture_layers(pair, request, record)
            except Exception as error:
                record.update(status='failed', error=f'{type(error).__name__}: {error}')
                traceback.print_exc()
            record['elapsed_seconds'] = time.perf_counter() - started
            write_json(metadata_path, record)
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
            stream.flush()
    records = [read_json(output / pair['model_id'] / 'metadata.json') for pair in request['pairs']
               if (output / pair['model_id'] / 'metadata.json').exists()]
    summary = {'planned': len(request['pairs']), 'success': sum(r['status'] == 'success' for r in records),
               'failed': sum(r['status'] == 'failed' for r in records),
               'incomplete': sum(r['status'] == 'running' for r in records),
               'missing': len(request['pairs']) - len(records), 'limit': limit,
               'request_sha256': request_hash, 'human_review': 'pending'}
    write_json(output / 'render_summary.json', summary)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--limit', type=int)
    arguments = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else sys.argv[1:]
    args = parser.parse_args(arguments)
    run_request(args.request, args.limit)
