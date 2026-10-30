"""将固定六件的全部结果排成等尺寸对照；不改变任何候选图片。"""

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def make_page(rows, output):
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 18)
    page = Image.new('RGB', (1536, 552 * len(rows)), 'white')
    draw = ImageDraw.Draw(page)
    sources = []
    for row_index, (model_id, columns) in enumerate(rows):
        for column_index, (label, path) in enumerate(columns):
            pixels = Image.open(path)
            assert pixels.size == (512, 512)
            x, y = column_index * 512, row_index * 552
            draw.text((x + 8, y + 9), f'{model_id} | {label}', fill='#17202a', font=font)
            page.paste(pixels, (x, y + 40))
            sources.append(dict(model_id=model_id, label=label, path=str(path),
                                sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    page.save(output, quality=96)
    return dict(image=str(output), sources=sources)


def main(args):
    args.output_dir.mkdir(parents=True, exist_ok=False)
    request = json.loads((args.canny_dir / 'request.json').read_text())
    pairs = request['pairs']
    pages = []
    for start in range(0, len(pairs), 2):
        rows = []
        for pair in pairs[start:start + 2]:
            model_id = pair['model_id']
            rows.append((model_id, [
                ('Rendered pair', Path(pair['with_part']['rgb'])),
                ('Background + Depth', args.depth_dir / model_id / 'relative_darkening.png'),
                ('Background + Canny', args.canny_dir / model_id / 'relative_darkening.png'),
            ]))
        pages.append(make_page(rows, args.output_dir / f'all_{start // 2:02d}.jpg'))

    # 这两个例子分别展示接触影收益及仍未解决的薄片感；不是成功率样本。
    selected = []
    for pair in pairs:
        model_id = pair['model_id']
        if model_id in ['00301161', '00182999']:
            selected.append((model_id, [
                ('Rendered pair', Path(pair['with_part']['rgb'])),
                ('Previous masked diffusion', args.previous_dir / model_id / 'strength_0.65/protected.png'),
                ('New background + Canny', args.canny_dir / model_id / 'relative_darkening.png'),
            ]))
    pages.append(make_page(selected, args.output_dir / 'progress.jpg'))
    (args.output_dir / 'manifest.json').write_text(json.dumps(dict(
        pages=pages, planned_models=[p['model_id'] for p in pairs],
        note='Previous masked diffusion uses the older 64-sample render; the new pair uses 256. '
             'This progress display is not a single-factor experiment. All images remain synthetic.',
    ), indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--depth_dir', type=Path, required=True)
    parser.add_argument('--canny_dir', type=Path, required=True)
    parser.add_argument('--previous_dir', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    main(parser.parse_args())
