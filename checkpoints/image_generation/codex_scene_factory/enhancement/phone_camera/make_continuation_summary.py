"""固定六件并列局部修图与背景增强；另展示旧失败例的变化。"""

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


HERE = Path(__file__).resolve().parent
ROOT = Path('/mnt/d/data/cad_scene_enhancements')
OUTPUT = ROOT / 'office_phone_comparison_20261027_002'
OUTPUT.mkdir(exist_ok=False)
font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 18)
samples = json.loads((HERE / 'samples.json').read_text())
manifest = []
for sample in samples:
    sid = sample['model_id']
    manifest.append({
        'model_id': sid,
        'original': sample['rgb_file'],
        'retouch': str(ROOT / 'office_phone_deeplpf_20261027_001' / sid / 'deeplpf.png'),
        'previous': str(ROOT / 'office_phone_depth_guard_matte_20261027_001' / sid / 'strength_0.65' / 'protected.png'),
        'current': str(ROOT / 'office_phone_depth_office_surfaces_20261027_001' / sid / 'strength_0.65' / 'protected.png'),
    })


def draw_rows(items, columns, size, filename):
    canvas = Image.new('RGB', (len(columns) * size, len(items) * (size + 36)), 'white')
    draw = ImageDraw.Draw(canvas)
    for row, item in enumerate(items):
        for col, (key, label) in enumerate(columns):
            x, y = col * size, row * (size + 36)
            draw.text((x + 8, y + 7), label, fill='#182231', font=font)
            pixels = Image.open(item[key]).convert('RGB')
            canvas.paste(pixels.resize((size, size), Image.Resampling.LANCZOS), (x, y + 36))
    canvas.save(OUTPUT / filename, quality=96)


for page in range(2):
    draw_rows(manifest[page * 3:(page + 1) * 3], [
        ('original', 'Original'), ('retouch', 'Local retouch: DeepLPF'),
        ('current', 'Background diffusion'),
    ], 384, f'comparison_{page:02d}.jpg')

draw_rows([manifest[0], manifest[3]], [
    ('original', 'Original'), ('previous', 'Previous diffusion'),
    ('current', 'Background-only description'),
], 512, 'preview.jpg')
(OUTPUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(OUTPUT)
