"""固定十件输入和依赖，生成可独立执行的渲染/增强请求。"""

import argparse
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'background_plate'))
from render_pairs import prepare_request, read_json, sha256, write_json

PACKAGE = Path('/mnt/d/python_room_generation/packages/cad_scene_factory')
OLD_PAIRS = Path('/mnt/d/data/cad_scene_renders/office_phone_pairs_20261028_001')
OLD_CANNY = Path('/mnt/d/data/cad_scene_enhancements/office_phone_plate_canny_20261028_001')
CAPTURE = Path('/mnt/d/data/cad_scene_renders/office_step3_100_support_fix_20261027_001')


def prepare(output):
    samples = read_json(HERE.parent / 'samples.json')
    for seed, model_id in enumerate(['00985631', '00062376', '00121033', '00199547'], 14):
        directory = CAPTURE / model_id
        samples.append(dict(model_id=model_id, index=len(samples),
                            rgb_file=str(directory / 'rgb.png'), mask_file=str(directory / 'mask.png'),
                            metadata_file=str(directory / 'metadata.json'),
                            rgb_sha256=sha256(directory / 'rgb.png'),
                            mask_sha256=sha256(directory / 'mask.png'), experiment_seed=seed))
    write_json(HERE / 'samples.json', samples)
    request = prepare_request(SimpleNamespace(output_dir=output, samples_file=HERE / 'samples.json'))
    output = Path(request['output_dir'])

    # 原场景的外部贴图也是输入；在本轮渲染前冻结，而非事后补记。
    audit = read_json(HERE.parent / 'background_plate/pair_capture_audit_20261028.json')
    for asset in audit['dependency_audit']['files']:
        assert sha256(asset['path']) == asset['sha256'], asset['path']
        request['input_sha256'][asset['path']] = asset['sha256']
    for pair in request['pairs'][:6]:
        pair['cached_background'] = dict(
            empty=str(OLD_PAIRS / pair['model_id'] / 'empty/rgb.png'),
            enhanced=str(OLD_CANNY / pair['model_id'] / 'enhanced_empty.png'))
        for path in pair['cached_background'].values():
            request['input_sha256'][path] = sha256(path)

    (output / 'source/step4').mkdir()
    shutil.copy2(PACKAGE / 'step4/model.py', output / 'source/step4/model.py')
    shutil.copy2(HERE.parent / 'background_plate/composite.py', output / 'legacy_composite.py')
    shutil.copy2(HERE.parent / 'krea_office_surfaces.md', output / 'prompt.md')
    for path in HERE.iterdir():
        if path.suffix in ['.py', '.md', '.json']:
            shutil.copy2(path, output / path.name)
    previous = read_json(OLD_CANNY / 'request.json')
    request.update(
        config=previous['config'], model_dir=previous['model_dir'],
        transformer_file=previous['transformer_file'],
        prompt=(output / 'prompt.md').read_text().strip(),
        model_sources=dict(parent_request=str(OLD_CANNY / 'request.json'),
                           parent_sha256=sha256(OLD_CANNY / 'request.json'),
                           controlnet_files=previous['controlnet_files'],
                           model_files=previous['model_files'], versions=previous['versions']),
        git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=HERE, text=True).strip(),
        command=[sys.executable, *sys.argv],
        source_sha256={str(p.relative_to(output / 'source')): sha256(p)
                       for p in (output / 'source').rglob('*.py')},
        script_sha256={p.name: sha256(p) for p in output.glob('*.py')},
        spec_sha256=sha256(output / 'SPEC.md'),
        color_contract=dict(png='AgX display RGB; foreground PNG is straight RGBA',
                            exr='scene-linear RGBA; foreground is premultiplied',
                            compositing='display-linear approximation; neutral lighting gain'),
    )
    (output / 'git.diff').write_bytes(subprocess.check_output(['git', 'diff', 'HEAD'], cwd=HERE))
    write_json(output / 'request.json', request)
    print(f'固定 {len(samples)} 件：{output}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output_dir', type=Path, required=True)
    prepare(parser.parse_args().output_dir)
