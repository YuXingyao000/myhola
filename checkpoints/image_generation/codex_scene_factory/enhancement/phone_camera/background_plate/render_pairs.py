"""冻结六件输入；在 Blender 中按原机位拍摄有 CAD / 无 CAD 的成对图片。

准备：普通 Python --prepare_only --output_dir NEW_DIRECTORY。
执行：Blender --python NEW_DIRECTORY/render_pairs.py -- --request NEW_DIRECTORY/request.json。
本脚本不启动 Blender、不调用 diffusion，也不修改原场景或生产 Step3。
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
PACKAGE = Path('/mnt/d/python_room_generation/packages/cad_scene_factory')
SAMPLES = HERE.parent / 'samples.json'


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def object_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def prepare_request(args):
    """只读现有输入，冻结复用源码及每件的真实来源，不启动 Blender。"""
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = output / 'source'
    for directory in ['step2', 'step3']:
        (source / directory).mkdir(parents=True)
        for path in (PACKAGE / directory).glob('*.py'):
            shutil.copy2(path, source / directory / path.name)
    for name in ['geometry.py', 'visibility.py']:
        shutil.copy2(PACKAGE / name, source / name)
    shutil.copy2(__file__, output / 'render_pairs.py')
    shutil.copy2(HERE / 'SPEC.md', output / 'SPEC.md')

    input_hashes, pairs = {}, []
    for sample in read_json(args.samples_file):
        metadata_path = Path(sample['metadata_file'])
        source_request_path = metadata_path.parent.parent / 'request.json'
        metadata = read_json(metadata_path)
        original = read_json(source_request_path)
        placement = original['placements'][metadata['placement_index']]
        inputs = {
            'rgb': sample['rgb_file'], 'mask': sample['mask_file'],
            'metadata': str(metadata_path), 'request': str(source_request_path),
            'stl': metadata['stl_file'], 'scene': original['base_scene'],
        }
        for path in inputs.values():
            if path not in input_hashes:
                input_hashes[path] = sha256(path)
        assert input_hashes[inputs['rgb']] == sample['rgb_sha256']
        assert input_hashes[inputs['mask']] == sample['mask_sha256']
        assert input_hashes[inputs['stl']] == metadata['source']['stl_sha256']
        assert input_hashes[inputs['scene']] == original['scene_hash']
        transform = {
            'camera_to_world': metadata['camera']['camera_to_world_blender'],
            'model_to_world': metadata['geometry']['model_to_world'],
            'scene_object_moves': placement['transforms'],
        }
        capture = {
            'base_scene': original['base_scene'], 'scene_hash': original['scene_hash'],
            'step2_config': original['step2_config'],
            'config': dict(original['config'], samples=256, threads=8),
            'device': 'OPTIX', 'gpu_index': 7,
        }
        pairs.append({
            'model_id': sample['model_id'], 'inputs': inputs,
            'source_metadata': metadata, 'placement': placement, 'capture_request': capture,
            'transform_sha256': object_sha256(transform),
            'material_sha256': object_sha256(metadata['material']),
            'render_seed': metadata['seed'], 'enhancement_seed': sample['experiment_seed'],
        })
    request = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'output_dir': str(output), 'samples_file': str(args.samples_file.resolve()),
        'samples_sha256': sha256(args.samples_file), 'planned_pairs': len(pairs),
        'pairs': pairs, 'input_sha256': input_hashes,
        'source_sha256': {str(path.relative_to(source)): sha256(path)
                          for path in source.rglob('*.py')},
        'script_sha256': sha256(output / 'render_pairs.py'),
        'spec_sha256': sha256(output / 'SPEC.md'),
        'settings': {'samples': 256, 'threads': 8, 'device': 'OPTIX', 'gpu_index': 7},
        'color_contract': {
            'rgb': 'scene display transform, RGB PNG, 8 bit',
            'radiance': 'Render Layers Image, scene-linear RGB EXR, 32 bit, Cycles denoised',
            'depth': 'camera-axis Z meters, first surface',
        },
        'human_review': 'pending', 'status': 'prepared_not_rendered',
    }
    write_json(output / 'request.json', request)
    print(output / 'request.json', flush=True)
    return request


def add_radiance_output(outputs):
    """从 Render Layers 直接保存线性颜色，不从已显示映射的 PNG 反推。"""
    import bpy
    scene = bpy.context.scene
    tree = scene.compositing_node_group
    layer = tree.nodes.new('CompositorNodeRLayers')
    layer.layer = bpy.context.view_layer.name
    output = tree.nodes.new('CompositorNodeOutputFile')
    output.file_name = 'radiance_'
    output.format.media_type = 'IMAGE'
    output.format.file_format = 'OPEN_EXR'
    output.format.color_mode = 'RGB'
    output.format.color_depth = '32'
    output.format.exr_codec = 'ZIP'
    output.save_as_render = False
    output.file_output_items.new('RGBA', 'radiance')
    tree.links.new(layer.outputs['Image'], output.inputs[0])
    outputs['radiance'] = output


def render_view(directory, seed, outputs):
    """保存一次完整渲染；empty 的合法全零 mask 不在这里报错。"""
    import bpy
    directory.mkdir()
    scene = bpy.context.scene
    scene.cycles.seed = seed
    scene.render.filepath = str(directory / 'rgb.png')
    for output in outputs.values():
        output.directory = str(directory)
    started = time.perf_counter()
    bpy.ops.render.render(write_still=True)
    for name, suffix in [('mask', 'png'), ('depth', 'exr'), ('radiance', 'exr')]:
        path, = directory.glob(f'{name}_*.{suffix}')
        path.rename(directory / f'{name}.{suffix}')
    files = {name: str(directory / filename) for name, filename in [
        ('rgb', 'rgb.png'), ('mask', 'mask.png'), ('depth', 'depth.exr'), ('radiance', 'radiance.exr')]}
    return files, time.perf_counter() - started


def read_numeric_image(path):
    """关闭载入时颜色转换，读取 PNG 数值通道用于同域误差比较。"""
    import bpy
    import numpy as np
    image = bpy.data.images.load(str(path), check_existing=False)
    image.colorspace_settings.name = 'Non-Color'
    width, height = image.size
    values = np.asarray(image.pixels[:], dtype=np.float32).reshape(height, width, 4)[::-1].copy()
    bpy.data.images.remove(image)
    return values


def compare_source(pair, with_part, empty):
    import numpy as np
    original = read_numeric_image(pair['inputs']['rgb'])[:, :, :3]
    current = read_numeric_image(with_part['rgb'])[:, :, :3]
    old_mask = read_numeric_image(pair['inputs']['mask'])[:, :, 0] > 0.5
    new_mask = read_numeric_image(with_part['mask'])[:, :, 0] > 0.5
    empty_mask = read_numeric_image(empty['mask'])[:, :, 0] > 0.5
    if not new_mask.any() or empty_mask.any():
        raise ValueError('with_part mask 必须非空，empty mask 必须全零')
    difference = np.abs(current - original) * 255
    return {
        'mask_equal': bool(np.array_equal(old_mask, new_mask)),
        'mask_iou': float((old_mask & new_mask).sum() / (old_mask | new_mask).sum()),
        'mask_differing_pixels': int((old_mask != new_mask).sum()),
        'with_part_visible_pixels': int(new_mask.sum()), 'empty_visible_pixels': int(empty_mask.sum()),
        'rgb_mae_255': float(difference.mean()),
        'cad_rgb_mae_255': float(difference[old_mask].mean()),
        'background_rgb_mae_255': float(difference[~old_mask].mean()),
        'rgb_p99_abs_difference_255': float(np.percentile(difference, 99)),
        'rgb_max_abs_difference_255': float(difference.max()),
        'status': 'review_required',
        'note': 'Source used 64 samples; this pair uses 256. Device and denoising differences require visual review.',
    }


def capture_pair(pair, request):
    import bpy
    import numpy as np
    from mathutils import Matrix
    from capture import prepare_placement, camera_metadata
    from part import load_part, make_material, remove_part, check_support_contact

    metadata = pair['source_metadata']
    capture = pair['capture_request']
    _, outputs = prepare_placement(capture, pair['placement'])
    scene = bpy.context.scene
    scene.camera.matrix_world = Matrix(metadata['camera']['camera_to_world_blender'])
    part = load_part(Path(pair['inputs']['stl']), metadata['smooth_angle_degrees'])
    part.matrix_world = Matrix(metadata['geometry']['model_to_world'])
    part.data.materials.clear()
    part.data.materials.append(make_material(metadata['material']))
    part.pass_index = 1
    bpy.context.view_layer.update()
    contact = check_support_contact(part, pair['placement']['support_validation'])
    add_radiance_output(outputs)
    directory = Path(request['output_dir']) / pair['model_id']
    actual_camera = camera_metadata(scene.camera, capture['step2_config'])
    matrix_error = float(np.abs(np.array(part.matrix_world) - metadata['geometry']['model_to_world']).max())
    camera_error = float(np.abs(np.array(scene.camera.matrix_world) - metadata['camera']['camera_to_world_blender']).max())
    with_part, with_seconds = render_view(directory / 'with_part', pair['render_seed'], outputs)
    remove_part(part)
    empty, empty_seconds = render_view(directory / 'empty', pair['render_seed'], outputs)
    return {
        'status': 'success', 'with_part': with_part, 'empty': empty,
        'render_seconds': {'with_part': with_seconds, 'empty': empty_seconds},
        'comparison_to_source': compare_source(pair, with_part, empty),
        'camera': actual_camera, 'model_to_world': metadata['geometry']['model_to_world'],
        'camera_matrix_max_abs_error': camera_error, 'model_matrix_max_abs_error': matrix_error,
        'contact': contact, 'material': metadata['material'], 'render_device': capture['render_device'],
        'blender_version': bpy.app.version_string,
        'color_management': {key: getattr(scene.view_settings, key)
                             for key in ['view_transform', 'look', 'exposure', 'gamma']},
        'output_sha256': {str(Path(path).relative_to(directory)): sha256(path)
                          for files in [with_part, empty] for path in files.values()},
    }


def run_request(path):
    """主循环：恢复一件 → with_part → 移除 CAD → empty → 保存证据。"""
    request = read_json(path)
    output = Path(request['output_dir'])
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '7':
        raise ValueError('请在启动 Blender 前明确设置 CUDA_VISIBLE_DEVICES=7')
    for filename, digest in request['input_sha256'].items():
        assert sha256(filename) == digest, filename
    for filename, digest in request['source_sha256'].items():
        assert sha256(output / 'source' / filename) == digest, filename
    assert sha256(__file__) == request['script_sha256']
    sys.path.insert(0, str(output / 'source' / 'step3'))
    records = []
    with (output / 'pairs.jsonl').open('x') as stream:
        for pair in request['pairs']:
            directory = output / pair['model_id']
            directory.mkdir()
            record = {key: pair[key] for key in ['model_id', 'render_seed', 'enhancement_seed',
                                                  'transform_sha256', 'material_sha256']}
            record['status'] = 'failed'
            started = time.perf_counter()
            print(f"成对拍摄 {pair['model_id']} · render seed {pair['render_seed']}", flush=True)
            try:
                record.update(capture_pair(pair, request))
            except Exception as error:
                record['error'] = f'{type(error).__name__}: {error}'
                traceback.print_exc()
            record['elapsed_seconds'] = time.perf_counter() - started
            write_json(directory / 'metadata.json', record)
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
            stream.flush()
            records.append(record)
    summary = {
        'planned': request['planned_pairs'], 'success': sum(row['status'] == 'success' for row in records),
        'failed': sum(row['status'] != 'success' for row in records),
        'missing': request['planned_pairs'] - len(records),
        'human_review': 'pending', 'source_reproduction': 'review_required',
        'training_acceptance': 'not_accepted_development_inputs',
    }
    write_json(output / 'summary.json', summary)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare_only', action='store_true')
    parser.add_argument('--samples_file', type=Path, default=SAMPLES)
    parser.add_argument('--output_dir', type=Path)
    parser.add_argument('--request', type=Path)
    arguments = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else sys.argv[1:]
    args = parser.parse_args(arguments)
    if args.prepare_only:
        prepare_request(args)
    else:
        run_request(args.request)
