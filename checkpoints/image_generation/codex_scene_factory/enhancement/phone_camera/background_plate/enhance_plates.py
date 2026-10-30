"""增强不含CAD的背景，再转移成对渲染中的零件与接触光影。"""

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import traceback

import numpy as np
from PIL import Image, ImageDraw, ImageFont


HERE = Path(__file__).resolve().parent
CODE = Path('/mnt/d/python_room_generation/packages/cad_scene_factory/step4')
EXR_READER = '/mnt/d/model/Flux1_ControlNet_Depth/tools/python'


def prepare(args):
    sys.path.insert(0, str(CODE))
    from workflow import save_request, sha256

    pair_summary = json.loads((args.pairs_dir / 'summary.json').read_text())
    pairs = [json.loads(row) for row in (args.pairs_dir / 'pairs.jsonl').read_text().splitlines()]
    assert pair_summary['success'] == len(pairs) == 6
    assert all(pair['status'] == 'success' for pair in pairs)
    profile = json.loads((HERE / 'profiles.json').read_text())[args.profile]
    config = dict(pipeline='flux_control', processing_size=512, resident_model=True,
                  max_sequence_length=512, strengths=[0.65], num_inference_steps=40,
                  guidance_scale=4.5, **profile)
    request = dict(
        profile=args.profile, output_dir=str(args.output_dir.resolve()),
        pairs_dir=str(args.pairs_dir.resolve()), pairs=pairs, config=config,
        pairs_request_sha256=sha256(args.pairs_dir / 'request.json'),
        model_dir='/mnt/d/model/Flux1_Krea_dev',
        transformer_file='/mnt/d/model/Flux1_Krea_dev_GGUF/flux1-krea-dev-Q8_0.gguf',
        prompt=(HERE.parent / 'krea_office_surfaces.md').read_text().strip(),
        gpu_index=7, command=[sys.executable, *sys.argv], human_review='pending',
    )
    save_request(request)
    output = Path(request['output_dir'])
    for name in ['enhance_plates.py', 'composite.py', 'PLAN.md', 'COMPOSITE_SPEC.md', 'profiles.json']:
        shutil.copy2(HERE / name, output / name)
    shutil.copy2(HERE.parent / 'prepare_depth.py', output / 'prepare_depth.py')
    request['research_source_sha256'] = {p.name: sha256(p) for p in output.glob('*.py')}
    request['input_sha256'] = {path: sha256(path) for pair in pairs
                               for kind in ['with_part', 'empty']
                               for path in pair[kind].values()}
    request['controlnet_files'] = {p.name: dict(bytes=p.stat().st_size, sha256=sha256(p))
                                   for p in Path(config['controlnet_dir']).iterdir() if p.is_file()}
    (output / 'request.json').write_text(json.dumps(request, indent=2) + '\n')
    return request


def make_control(pair, config):
    if config['control_type'] == 'depth':
        sys.path.insert(0, EXR_READER)
        import OpenEXR
        from prepare_depth import inverse_depth_control
        depth = OpenEXR.File(pair['empty']['depth'], separate_channels=True).channels()['V'].pixels
        pixels, _, metrics = inverse_depth_control(depth)
        return Image.fromarray(pixels), metrics
    import cv2
    gray = np.asarray(Image.open(pair['empty']['rgb']).convert('L'))
    edges = cv2.Canny(gray, *config['canny_thresholds'])
    return Image.fromarray(edges).convert('RGB'), {
        'canny_thresholds': config['canny_thresholds'],
        'edge_pixels': int(np.count_nonzero(edges)), 'opencv_version': cv2.__version__,
    }


def save_comparison(directory, with_part, empty, enhanced, composites):
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 18)
    items = [('Original pair', with_part), ('Empty background', empty),
             ('Enhanced background', enhanced), *composites.items()]
    # 两行可直接看全尺寸，也避免五列长条过度缩小。
    canvas = Image.new('RGB', (3 * 512, 2 * 548), 'white')
    draw = ImageDraw.Draw(canvas)
    for index, (label, pixels) in enumerate(items):
        x, y = (index % 3) * 512, (index // 3) * 548
        draw.text((x + 8, y + 8), label, font=font, fill='black')
        canvas.paste(pixels, (x, y + 36))
    canvas.save(directory / 'comparison.jpg', quality=96)


def run(request):
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    output = Path(request['output_dir'])
    sys.path.insert(0, str(output / 'source'))
    from model import FluxEnhancer
    from workflow import sha256
    from composite import compose

    engine = FluxEnhancer(Path(request['model_dir']), Path(request['transformer_file']),
                          request['prompt'], request['config'])
    records = []
    with (output / 'results.jsonl').open('x') as stream:
        for pair in request['pairs']:
            directory = output / pair['model_id']
            directory.mkdir()
            record = dict(model_id=pair['model_id'], seed=pair['enhancement_seed'], status='failed')
            try:
                for kind in ['with_part', 'empty']:
                    for path in pair[kind].values():
                        assert sha256(path) == request['input_sha256'][path], path
                with_part = Image.open(pair['with_part']['rgb']).convert('RGB')
                empty = Image.open(pair['empty']['rgb']).convert('RGB')
                mask = Image.open(pair['with_part']['mask']).convert('L')
                condition, record['condition'] = make_control(pair, request['config'])
                condition.save(directory / 'control.png')
                enhanced, record['inference'] = engine.enhance(
                    empty, Image.new('L', empty.size, 255), 0.65, record['seed'],
                    control_image=condition)
                enhanced.save(directory / 'enhanced_empty.png')
                composites, checks = {}, {}
                for mode in ['additive', 'relative_darkening']:
                    pixels, checks[mode], diagnostics = compose(with_part, empty, enhanced, mask, mode)
                    pixels.save(directory / f'{mode}.png')
                    composites[mode] = pixels
                    for name, value in diagnostics.items():
                        if isinstance(value, Image.Image):
                            value.save(directory / f'{mode}_{name}.png')
                save_comparison(directory, with_part, empty, enhanced, composites)
                record.update(status='success', checks=checks,
                    output_sha256={p.name: sha256(p) for p in directory.glob('*.png')})
            except Exception as error:
                record['error'] = f'{type(error).__name__}: {error}'
                traceback.print_exc()
            (directory / 'metadata.json').write_text(json.dumps(record, indent=2) + '\n')
            stream.write(json.dumps(record) + '\n')
            stream.flush()
            records.append(record)
    summary = dict(planned=len(records), success=sum(r['status']=='success' for r in records),
                   failed=sum(r['status']!='success' for r in records), human_review='pending',
                   training_acceptance='not_accepted')
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(summary, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pairs_dir', type=Path)
    parser.add_argument('--output_dir', type=Path)
    parser.add_argument('--profile', choices=['depth', 'canny'])
    parser.add_argument('--run_request', type=Path)
    args = parser.parse_args()
    if args.run_request:
        run(json.loads(args.run_request.read_text()))
    else:
        request = prepare(args)
        destination = Path(request['output_dir'])
        os.execv(sys.executable, [sys.executable, str(destination / 'enhance_plates.py'),
                                '--run_request', str(destination / 'request.json')])
