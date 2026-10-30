"""固定六件：加载一个模型 → 保存整图编辑/零件回贴两版 → 逐件比较。"""

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

from PIL import Image, ImageDraw, ImageFont
import numpy as np

HERE = Path(__file__).resolve().parent
CODE = Path('/mnt/d/python_room_generation/packages/cad_scene_factory/step4')


def prepare_run(profile_name, output_dir):
    sys.path.insert(0, str(CODE))
    from workflow import save_request
    profile = json.loads((HERE / 'profiles.json').read_text())[profile_name]
    samples = json.loads((HERE / 'samples.json').read_text())
    prompt_file = HERE / profile['prompt']
    config = {key: profile[key] for key in ('pipeline', 'processing_size',
              'num_inference_steps', 'guidance_scale', 'strengths')}
    for key in ('controlnet_dir', 'controlnet_conditioning_scale', 'depth_input_dir',
                'sampling_mask', 'protect_padding_px', 'feather_px', 'spatial_plan'):
        if key in profile:
            config[key] = profile[key]
    config.update(resident_model=True, max_sequence_length=512,
                  protect_padding_px=profile.get('protect_padding_px', 2),
                  feather_px=profile.get('feather_px', 4))
    request = dict(profile=profile_name, scene_id='office_20261023_001',
                   output_dir=str(output_dir), samples=samples, config=config,
                   model_dir=profile['model_dir'], transformer_file=profile['transformer_file'],
                   prompt=prompt_file.read_text().strip(), gpu_index=7,
                   command=[sys.executable, *sys.argv], human_review='pending')
    save_request(request)
    if 'controlnet_dir' in config:
        from workflow import sha256
        request['controlnet_files'] = {str(path): sha256(path) for path in
            Path(config['controlnet_dir']).glob('*') if path.is_file()}
        request['depth_files'] = {}
        for sample in samples:
            path = Path(config['depth_input_dir']) / sample['model_id'] / 'depth.png'
            request['depth_files'][sample['model_id']] = dict(
                path=str(path), sha256=sha256(path),
                preparation=json.loads((path.parent / 'metadata.json').read_text()))
        (output_dir / 'request.json').write_text(json.dumps(request, indent=2)+'\n')
    for name in ('compare_diffusion.py', 'profiles.json', 'samples.json', 'PLAN.md'):
        shutil.copy2(HERE / name, output_dir / name)
    if config.get('sampling_mask') == 'spatial':
        from workflow import sha256
        for name in ('spatial_sampling.py', 'SPATIAL_SPEC.md'):
            shutil.copy2(HERE / name, output_dir / name)
        request['spatial_source_sha256'] = sha256(output_dir / 'spatial_sampling.py')
        request['spatial_spec_sha256'] = sha256(output_dir / 'SPATIAL_SPEC.md')
        (output_dir / 'request.json').write_text(json.dumps(request, indent=2)+'\n')
    return request


def make_comparison(sample, records, output_dir):
    successful = [r for r in records if r['model_id'] == sample['model_id']]
    width = 512
    header = 32
    canvas = Image.new('RGB', (3*width, len(successful)*(width+header)), 'white')
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 18)
    for row, record in enumerate(successful):
        items = [('Original', sample['rgb_file']),
                 ('Generated | CAD may change', record.get('generated_file')),
                 ('Original CAD pasted back', record.get('protected_file'))]
        for col, (title, path) in enumerate(items):
            x, y = col*width, row*(width+header)
            draw.text((x+8,y+5), f'{title} | {record["strength"]:g}', fill='black', font=font)
            if path:
                canvas.paste(Image.open(path).convert('RGB'), (x,y+header))
            else:
                draw.text((x+10,y+80), 'FAILED', fill='red', font=font)
    canvas.save(output_dir / sample['model_id'] / 'comparison.jpg', quality=95)


def run_experiment(request):
    output_dir = Path(request['output_dir'])
    os.environ['CUDA_VISIBLE_DEVICES'] = str(request['gpu_index'])
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    sys.path.insert(0, str(output_dir / 'source'))
    from model import FluxEnhancer
    from masks import read_inputs, make_masks, composite_and_check
    from workflow import sha256

    engine = FluxEnhancer(Path(request['model_dir']), Path(request['transformer_file']),
                          request['prompt'], request['config'])
    records = []
    with (output_dir / 'results.jsonl').open('x') as stream:
        for sample in request['samples']:
            for strength in request['config']['strengths']:
                destination = output_dir / sample['model_id'] / f'strength_{strength:g}'
                destination.mkdir(parents=True)
                record = dict(model_id=sample['model_id'], strength=strength,
                              seed=sample['experiment_seed'], status='failed')
                start = time.perf_counter()
                # 唯一逐件异常边界；保留失败，继续剩余固定清单。
                try:
                    for kind in ('rgb', 'mask'):
                        assert sha256(sample[f'{kind}_file']) == sample[f'{kind}_sha256']
                    original, mask = read_inputs(sample['rgb_file'], sample['mask_file'])
                    region = 'convex_hull' if request['config'].get('sampling_mask') == 'convex_hull' else 'silhouette'
                    protected, edit, blend = make_masks(
                        mask, request['config']['protect_padding_px'],
                        request['config']['feather_px'], region=region)
                    protected.save(destination / 'protected_mask.png')
                    if request['config'].get('sampling_mask') == 'full_frame':
                        edit = Image.new('L', original.size, 255)
                    # 区域释放的机制实验；默认批量生产路径不启用此分支。
                    plan, callback, sampling_log = None, None, None
                    if request['config'].get('sampling_mask') == 'spatial':
                        from spatial_sampling import build_spatial_plan, make_callback
                        plan = build_spatial_plan(original, mask, request['config']['spatial_plan'])
                        protected = plan['strict_protected_mask']
                        blend = plan['strict_blend_mask']
                        edit = plan['initial_edit_mask']
                        callback, sampling_log = make_callback(engine.pipe, plan)
                        for name, value in plan.items():
                            if isinstance(value, Image.Image):
                                value.save(destination / f'{name}.png')
                        protected.save(destination / 'protected_mask.png')
                    depth = None
                    if 'depth_files' in request:
                        source = request['depth_files'][sample['model_id']]
                        assert sha256(source['path']) == source['sha256']
                        depth = Image.open(source['path']).convert('RGB')
                    generated, inference = engine.enhance(
                        original, edit, strength, record['seed'], depth_image=depth,
                        step_callback=callback)
                    final, protected_checks = composite_and_check(original, generated, protected, blend)
                    generated.save(destination / 'generated.png')
                    final.save(destination / 'protected.png')
                    if plan is not None:
                        # 仅对开放的面内部采用生成像素，其余与 strict 版相同。
                        # 此图会改变 CAD 像素，不能借 strict 检查宣称几何正确。
                        material = Image.composite(generated, original, plan['material_blend_mask'])
                        material.save(destination / 'material.png')
                        (destination / 'sampling.json').write_text(json.dumps(sampling_log, indent=2)+'\n')
                        record.update(spatial_plan=plan['summary'],
                                      material_file=str(destination / 'material.png'),
                                      material_sha256=sha256(destination / 'material.png'),
                                      material_geometry_status='unverified')
                    raw = np.asarray(generated)
                    target = np.asarray(mask) > 0
                    record.update(status='success', inference=inference,
                        generated_file=str(destination / 'generated.png'),
                        protected_file=str(destination / 'protected.png'),
                        generated_sha256=sha256(destination / 'generated.png'),
                        protected_sha256=sha256(destination / 'protected.png'),
                        protected_checks=protected_checks,
                        raw_target_mae=float(np.abs(raw.astype(float)-np.asarray(original))[target].mean()),
                        raw_target_channel_clip_fraction=float(((raw[target]==0)|(raw[target]==255)).mean()))
                except Exception as error:
                    record['error'] = f'{type(error).__name__}: {error}'
                    traceback.print_exc()
                record['elapsed_seconds'] = time.perf_counter() - start
                (destination / 'metadata.json').write_text(json.dumps(record,indent=2)+'\n')
                stream.write(json.dumps(record)+'\n'); stream.flush()
                records.append(record)
            make_comparison(sample, records, output_dir)
    summary = dict(planned=len(request['samples'])*len(request['config']['strengths']),
                   success=sum(r['status']=='success' for r in records),
                   failed=sum(r['status']=='failed' for r in records),
                   human_review='pending', training_acceptance='not_accepted')
    (output_dir / 'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile')
    parser.add_argument('--output_dir', type=Path)
    parser.add_argument('--run_request', type=Path)
    parser.add_argument('--prepare_only', action='store_true')
    args = parser.parse_args()
    if args.run_request:
        run_experiment(json.loads(args.run_request.read_text()))
    else:
        output_dir = args.output_dir.resolve()
        prepare_run(args.profile, output_dir)
        if args.prepare_only:
            return
        os.execv(sys.executable, [sys.executable, str(output_dir / 'compare_diffusion.py'),
                                 '--run_request', str(output_dir / 'request.json')])


if __name__ == '__main__':
    main()
