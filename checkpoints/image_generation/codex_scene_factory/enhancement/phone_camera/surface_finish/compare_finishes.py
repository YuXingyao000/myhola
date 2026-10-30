"""把新材质与同一增强空背景合成，并展示全部六件；本脚本仅使用CPU。"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from render_finishes import read_json, write_json, sha256


def draw_page(rows, path, crop=False):
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 18)
    page = Image.new('RGB', (1536, len(rows) * 552), 'white')
    draw = ImageDraw.Draw(page)
    for row, (model_id, images, mask) in enumerate(rows):
        ys, xs = np.nonzero(np.asarray(mask))
        x0 = min(max(round((xs.min() + xs.max()) / 2) - 128, 0), 256)
        y0 = min(max(round((ys.min() + ys.max()) / 2) - 128, 0), 256)
        for column, (label, image) in enumerate(images.items()):
            if crop:
                image = image.crop((x0, y0, x0 + 256, y0 + 256)).resize((512, 512), Image.Resampling.NEAREST)
            x, y = column * 512, row * 552
            draw.text((x + 8, y + 9), f'{model_id} | {label}' + (' | 2x' if crop else ''), font=font, fill='#17202a')
            page.paste(image, (x, y + 40))
    page.save(path, quality=96)


def compare(output):
    from composite import compose
    request = read_json(output / 'request.json')
    records = [json.loads(line) for line in (output / 'renders.jsonl').read_text().splitlines()]
    assert len(records) == request['planned'] == 12
    assert all(record['status'] == 'success' for record in records)
    renders = {(r['model_id'], r['profile']): r for r in records}
    rows, reports = [], []
    for pair in request['pairs']:
        model_id = pair['model_id']
        for path in [pair['empty']['rgb'], pair['enhanced_empty'], pair['previous_composite']]:
            assert sha256(path) == request['input_sha256'][path]
        empty = Image.open(pair['empty']['rgb'])
        enhanced = Image.open(pair['enhanced_empty'])
        mask = Image.open(pair['with_part']['mask']).convert('L')
        target = np.asarray(mask) > 0
        images = {'Previous material': Image.open(pair['previous_composite'])}
        for profile in request['profiles']:
            record = renders[model_id, profile]
            directory = output / model_id / profile
            assert sha256(record['images']['rgb']) == record['output_sha256']['rgb']
            original = Image.open(record['images']['rgb'])
            composite, checks, diagnostics = compose(original, empty, enhanced, mask, 'relative_darkening')
            composite.save(directory / 'composite.png')
            for name, value in diagnostics.items():
                if isinstance(value, Image.Image):
                    value.save(directory / f'{name}.png')
            values = np.asarray(composite)[target].astype(float)
            report = dict(model_id=model_id, profile=profile, checks=checks,
                          target_mean_rgb=values.mean(axis=0).tolist(),
                          target_std_rgb=values.std(axis=0).tolist(),
                          target_all_black_fraction=float(np.all(values == 0, axis=1).mean()),
                          target_any_white_fraction=float(np.any(values == 255, axis=1).mean()),
                          output_sha256=sha256(directory / 'composite.png'))
            write_json(directory / 'composite_metrics.json', report)
            reports.append(report)
            label = {'uniform_silver': 'Uniform silver', 'directional_machined': 'Directional finish'}[profile]
            images[label] = composite
        rows.append((model_id, images, mask))
    for start in range(0, len(rows), 2):
        draw_page(rows[start:start + 2], output / f'comparison_{start // 2:02d}.jpg')
        draw_page(rows[start:start + 2], output / f'target_{start // 2:02d}.jpg', crop=True)
    # 保留收益例和薄片局限例；完整分母仍由三张全量对照覆盖。
    draw_page([row for row in rows if row[0] in ['00301161', '00182999']], output / 'preview.jpg')
    write_json(output / 'composite_report.json', dict(
        planned=request['planned'], success=len(reports), failed=request['planned'] - len(reports),
        records=reports, human_review='pending', training_acceptance='not_accepted',
        note='All profiles reuse identical enhanced empty backgrounds; CAD/shadows are rerendered per material.',
    ))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output_dir', type=Path, required=True)
    compare(parser.parse_args().output_dir)
