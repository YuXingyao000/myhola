"""复用固定六件场景，仅替换目标金属材质；准备与 Blender 渲染分开执行。"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import traceback


HERE = Path(__file__).resolve().parent


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


def prepare_request(args):
    """冻结两种表面假设及既有背景，不下载模型、不调用生成器。"""
    pairs_request = read_json(args.pairs_dir / 'request.json')
    canny_request = read_json(args.canny_dir / 'request.json')
    for path, digest in pairs_request['source_sha256'].items():
        assert sha256(args.pairs_dir / 'source' / path) == digest, path
    assert sha256(args.pairs_dir / 'render_pairs.py') == pairs_request['script_sha256']
    assert sha256(args.canny_dir / 'composite.py') == canny_request['research_source_sha256']['composite.py']
    pairs = [json.loads(line) for line in (args.pairs_dir / 'pairs.jsonl').read_text().splitlines()]
    assert len(pairs) == 6 and all(pair['status'] == 'success' for pair in pairs)
    profiles = read_json(HERE / 'profiles.json')
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(args.pairs_dir / 'source', output / 'source')
    shutil.copy2(args.pairs_dir / 'render_pairs.py', output / 'render_pairs.py')
    shutil.copy2(args.canny_dir / 'composite.py', output / 'composite.py')
    for name in ['render_finishes.py', 'surface_material.py', 'compare_finishes.py', 'profiles.json', 'SPEC.md']:
        shutil.copy2(HERE / name, output / name)

    hashes = dict(pairs_request['input_sha256'])
    dependencies = read_json(args.dependency_audit)['dependency_audit']['files']
    hashes.update({item['path']: item['sha256'] for item in dependencies})
    for pair in pairs:
        for paths in (pair['with_part'], pair['empty']):
            for path in paths.values():
                relative = str(Path(path).relative_to(args.pairs_dir / pair['model_id']))
                hashes[path] = pair['output_sha256'][relative]
        directory = args.canny_dir / pair['model_id']
        previous = read_json(directory / 'metadata.json')
        for name in ['enhanced_empty.png', 'relative_darkening.png']:
            path = str(directory / name)
            hashes[path] = previous['output_sha256'][name]
        pair['enhanced_empty'] = str(directory / 'enhanced_empty.png')
        pair['previous_composite'] = str(directory / 'relative_darkening.png')
    for path, digest in hashes.items():
        assert sha256(path) == digest, path
    source_hashes = {str(path.relative_to(output)): sha256(path)
                     for path in output.rglob('*') if path.is_file()}
    request = dict(
        output_dir=str(output), pairs_request=pairs_request, pairs=pairs,
        profiles=profiles, planned=len(pairs) * len(profiles),
        input_sha256=hashes, source_sha256=source_hashes,
        parent_pair_request_sha256=sha256(args.pairs_dir / 'request.json'),
        parent_canny_request_sha256=sha256(args.canny_dir / 'request.json'),
        command=[sys.executable, *sys.argv], gpu_index=7,
        status='prepared_not_rendered', human_review='pending',
    )
    write_json(output / 'request.json', request)
    print(output / 'request.json', flush=True)


def geometry_signature(part):
    """网格、sharp/smooth及角法线一起核对，材质节点不应改写它们。"""
    import numpy as np
    mesh = part.data
    arrays = {
        'vertices': np.array([v.co[:] for v in mesh.vertices], dtype=np.float32),
        'triangles': np.array([p.vertices[:] for p in mesh.polygons], dtype=np.int32),
        'corner_normals': np.array([n.vector[:] for n in mesh.corner_normals], dtype=np.float32),
        'smooth': np.array([p.use_smooth for p in mesh.polygons], dtype=np.uint8),
        'sharp': np.array([e.use_edge_sharp for e in mesh.edges], dtype=np.uint8),
        'model_to_world': np.array(part.matrix_world, dtype=np.float64),
    }
    return {name: hashlib.sha256(values.tobytes()).hexdigest() for name, values in arrays.items()}


def compare_geometry_files(images, old_images):
    import numpy as np
    from render_pairs import read_numeric_image
    mask = read_numeric_image(images['mask'])[:, :, 0]
    old_mask = read_numeric_image(old_images['mask'])[:, :, 0]
    depth = read_numeric_image(images['depth'])[:, :, 0]
    old_depth = read_numeric_image(old_images['depth'])[:, :, 0]
    result = dict(mask_differing_pixels=int(np.count_nonzero(mask != old_mask)),
                  depth_differing_pixels=int(np.count_nonzero(depth != old_depth)),
                  depth_max_abs_error_m=float(np.max(np.abs(depth - old_depth))))
    assert result['mask_differing_pixels'] == result['depth_differing_pixels'] == 0, result
    return result


def render_pair_finishes(pair, previous, request, stream):
    import bpy
    import numpy as np
    from mathutils import Matrix
    from capture import prepare_placement
    from part import load_part, check_support_contact
    from render_pairs import add_radiance_output, render_view
    from surface_material import make_finish

    capture = pair['capture_request']
    metadata = pair['source_metadata']
    _, outputs = prepare_placement(capture, pair['placement'])
    scene = bpy.context.scene
    scene.camera.matrix_world = Matrix(metadata['camera']['camera_to_world_blender'])
    part = load_part(Path(pair['inputs']['stl']), metadata['smooth_angle_degrees'])
    part.matrix_world = Matrix(metadata['geometry']['model_to_world'])
    part.pass_index = 1
    bpy.context.view_layer.update()
    geometry_before = geometry_signature(part)
    contact = check_support_contact(part, pair['placement']['support_validation'])
    add_radiance_output(outputs)
    records = []
    for name, config in request['profiles'].items():
        directory = Path(request['output_dir']) / pair['model_id'] / name
        directory.mkdir(parents=True)
        record = dict(model_id=pair['model_id'], profile=name, status='failed', render_seed=pair['render_seed'])
        try:
            material, material_metadata = make_finish(part, config)
            part.data.materials.clear()
            part.data.materials.append(material)
            assert geometry_signature(part) == geometry_before
            print(f"表面拍摄 {pair['model_id']} · {name}", flush=True)
            images, seconds = render_view(directory / 'render', pair['render_seed'], outputs)
            geometry_after = geometry_signature(part)
            assert geometry_before == geometry_after
            checks = compare_geometry_files(images, previous['with_part'])
            record.update(status='success', images=images, render_seconds=seconds,
                geometry_checks=checks, geometry_signature=geometry_after, contact=contact,
                material=material_metadata, camera_to_world=[list(row) for row in scene.camera.matrix_world],
                camera_max_error_to_pair=float(np.abs(np.array(scene.camera.matrix_world) -
                    previous['camera']['camera_to_world_blender']).max()),
                model_to_world=[list(row) for row in part.matrix_world], render_device=capture['render_device'],
                color_management={key: getattr(scene.view_settings, key)
                    for key in ['view_transform', 'look', 'exposure', 'gamma']},
                blender_version=bpy.app.version_string,
                output_sha256={name: sha256(path) for name, path in images.items()})
        except Exception as error:
            record['error'] = f'{type(error).__name__}: {error}'
            traceback.print_exc()
        write_json(directory / 'metadata.json', record)
        stream.write(json.dumps(record, ensure_ascii=False) + '\n')
        stream.flush()
        records.append(record)
    return records


def run_request(path):
    request = read_json(path)
    output = Path(request['output_dir'])
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '7'
    for path, digest in request['input_sha256'].items():
        assert sha256(path) == digest, path
    for path, digest in request['source_sha256'].items():
        assert sha256(output / path) == digest, path
    # Blender --python 不会像普通 Python 一样把脚本目录加入模块搜索路径。
    sys.path.insert(0, str(output))
    sys.path.insert(0, str(output / 'source' / 'step3'))
    previous = {pair['model_id']: pair for pair in request['pairs']}
    records = []
    with (output / 'renders.jsonl').open('x') as stream:
        for pair in request['pairs_request']['pairs']:
            records.extend(render_pair_finishes(pair, previous[pair['model_id']], request, stream))
    write_json(output / 'render_summary.json', dict(
        planned=request['planned'], success=sum(r['status'] == 'success' for r in records),
        failed=sum(r['status'] != 'success' for r in records), missing=request['planned'] - len(records),
        human_review='pending', training_acceptance='not_accepted',
    ))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare_only', action='store_true')
    parser.add_argument('--pairs_dir', type=Path)
    parser.add_argument('--canny_dir', type=Path)
    parser.add_argument('--dependency_audit', type=Path)
    parser.add_argument('--output_dir', type=Path)
    parser.add_argument('--request', type=Path)
    arguments = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else sys.argv[1:]
    args = parser.parse_args(arguments)
    if args.prepare_only:
        prepare_request(args)
    else:
        run_request(args.request)
