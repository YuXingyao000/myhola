"""复用/增强空背景 → 分层合成 → 保存十件前后对照。"""

import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from compose_layers import compose_layers, encode
from render_pairs import read_json, sha256, write_json
from legacy_composite import compose as legacy_compose


def comparisons(output, model_id, original, old, final, alpha):
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 19)
    page = Image.new('RGB', (1536, 552), '#fafafa')
    detail = Image.new('RGB', (1536, 552), '#fafafa')
    rows, columns = np.nonzero(alpha > 0)
    cx, cy = (columns.min() + columns.max()) // 2, (rows.min() + rows.max()) // 2
    box = (int(cx - 128), int(cy - 128), int(cx + 128), int(cy + 128))
    for column, (label, image) in enumerate([('Original', original), ('Old: hard 2px ring', old),
                                            ('New: alpha + neutral light', final)]):
        for canvas in (page, detail):
            ImageDraw.Draw(canvas).text((column * 512 + 8, 9), f'{model_id} | {label}', fill='#222', font=font)
        page.paste(image, (column * 512, 40))
        detail.paste(image.crop(box).resize((512, 512), Image.Resampling.NEAREST), (column * 512, 40))
    page.save(output / 'comparison.png')
    detail.save(output / 'edge_comparison.png')


def run(request):
    output = Path(request['output_dir'])
    for filename, digest in request['input_sha256'].items():
        assert sha256(filename) == digest, filename
    for filename, digest in request['script_sha256'].items():
        assert sha256(output / filename) == digest, filename
    for filename, digest in request['source_sha256'].items():
        assert sha256(output / 'source' / filename) == digest, filename
    for name, metadata in request['model_sources']['controlnet_files'].items():
        assert sha256(Path(request['config']['controlnet_dir']) / name) == metadata['sha256'], name
    for filename, metadata in request['model_sources']['model_files'].items():
        assert sha256(filename) == metadata['sha256'], filename

    sys.path.insert(0, str(output / 'source/step4'))
    engine = None
    records = []
    started = time.perf_counter()
    with (output / 'enhancements.jsonl').open('x') as stream:
        for pair in request['pairs']:
            model_id = pair['model_id']
            directory = output / model_id
            record = dict(model_id=model_id, seed=pair['enhancement_seed'], status='failed')
            try:
                assert read_json(directory / 'metadata.json')['status'] == 'success'
                i, b, f, g, s = [Image.open(directory / kind / 'rgb.png').convert(mode) for kind, mode in
                                 [('with_part', 'RGB'), ('empty', 'RGB'), ('foreground', 'RGBA'),
                                  ('interaction', 'RGB'), ('shadow_only', 'RGB')]]
                mask = Image.open(directory / 'with_part/mask.png').convert('L')
                tick = time.perf_counter()
                if 'cached_background' in pair:
                    cached = pair['cached_background']
                    difference = np.abs(np.asarray(b).astype(int) -
                                        np.asarray(Image.open(cached['empty']).convert('RGB')).astype(int))
                    assert difference.max() <= 1, model_id
                    e = Image.open(cached['enhanced']).convert('RGB')
                    record['background_source'] = dict(mode='reuse_quantization_tolerance', **cached,
                        input_max_abs_rgb_255=int(difference.max()), input_mae_255=float(difference.mean()),
                        input_changed_pixels=int(np.any(difference != 0, axis=-1).sum()))
                else:
                    if engine is None:
                        from model import FluxEnhancer
                        engine = FluxEnhancer(Path(request['model_dir']), Path(request['transformer_file']),
                                              request['prompt'], request['config'])
                    edges = cv2.Canny(np.asarray(b.convert('L')), *request['config']['canny_thresholds'])
                    condition = Image.fromarray(edges).convert('RGB')
                    condition.save(directory / 'control.png')
                    e, record['inference'] = engine.enhance(b, Image.new('L', b.size, 255), .65,
                                                          pair['enhancement_seed'], control_image=condition)
                    record['background_source'] = dict(mode='new_fixed_seed')
                record['background_seconds'] = time.perf_counter() - tick
                e.save(directory / 'enhanced_empty.png')
                final, reference, record['compositing'], diagnostics = compose_layers(f, b, g, s, e)
                old, _, _ = legacy_compose(i, b, e, mask, 'relative_darkening')
                final.save(directory / 'final.png')
                reference.save(directory / 'rgb_transfer_reference.png')
                old.save(directory / 'old_hard_ring.png')
                Image.fromarray(np.rint(diagnostics['alpha'] * 255).astype(np.uint8)).save(directory / 'alpha.png')
                encode(diagnostics['background']).save(directory / 'background_with_interactions.png')
                np.savez_compressed(directory / 'lighting.npz', **{name: diagnostics[name].astype(np.float32)
                                    for name in ['shadow_gain', 'indirect_gain', 'total_gain']})
                comparisons(directory, model_id, i, old, final, diagnostics['alpha'])
                record.update(status='success', outputs={p.name: sha256(p) for p in directory.glob('*.png')})
            except Exception as error:
                record['error'] = f'{type(error).__name__}: {error}'
                traceback.print_exc()
            write_json(directory / 'enhancement.json', record)
            stream.write(json.dumps(record) + '\n')
            stream.flush()
            records.append(record)
            print(f"{model_id}: {record['status']}", flush=True)
    summary = dict(planned=10, success=sum(r['status'] == 'success' for r in records),
                   failed=sum(r['status'] != 'success' for r in records), missing=10-len(records),
                   elapsed_seconds=time.perf_counter()-started, human_review='pending')
    write_json(output / 'enhancement_summary.json', summary)
    successes = [r for r in records if r['status'] == 'success']
    gallery = Image.new('RGB', (5*512, 2*544), '#fafafa')
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 21)
    for index, record in enumerate(successes):
        x, y = (index % 5)*512, (index // 5)*544
        ImageDraw.Draw(gallery).text((x+8,y+5), record['model_id'], font=font, fill='#222')
        gallery.paste(Image.open(output / record['model_id'] / 'final.png'), (x,y+32))
    gallery.save(output / 'gallery.png')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    args = parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '7'
    run(read_json(args.request))
