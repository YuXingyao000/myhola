"""Assemble static diagnostic panels without changing any source image."""

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


HERE = Path(__file__).resolve().parent
SOURCE = Path('/mnt/d/data/cad_scene_enhancements/lut_sweep_20261029_001/raw/office/00301161/rgb.png')
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'


def main():
    report = json.loads((HERE / 'mesh_normals.json').read_text())['cases'][0]
    canvas = Image.new('RGB', (1064, 1220), '#f1f3f6')
    draw = ImageDraw.Draw(canvas)
    title_font = ImageFont.truetype(FONT, 23)
    font = ImageFont.truetype(FONT, 17)
    small = ImageFont.truetype(FONT, 15)
    draw.text((20, 12), '00301161 / fixed geometry and camera / normal diagnostic', fill='#101821', font=title_font)
    panels = [('Original RGB + region labels', SOURCE),
              ('Triangle edges: red=sharp; green=smooth', HERE / '00301161_edges.png'),
              ('Geometric face normals (world XYZ)', HERE / '00301161_face_normals.png'),
              ('Interpolated corner normals (world XYZ)', HERE / '00301161_smooth_normals.png')]
    for index, (label, path) in enumerate(panels):
        col, row = index % 2, index // 2
        x, y = 20 + col * 532, 74 + row * 554
        draw.text((x, y - 25), label, fill='#101821', font=font)
        panel = Image.open(path).convert('RGB')
        if index in (0, 2, 3):
            mark = ImageDraw.Draw(panel)
            for letter, region in zip('ABC', report['selected_regions']):
                px, py = region['sample_pixel']
                mark.ellipse((px - 12, py - 12, px + 12, py + 12), fill='white', outline='black', width=1)
                mark.text((px - 6, py - 10), letter, fill='black', font=font)
        canvas.paste(panel, (x, y))
    lines = [
        'A: large slope, 2 triangles; max corner deviation 1.475 deg (smooth transition to the curved top).',
        'B: rear plate, 12 triangles; max corner deviation 0.000374 deg, while brightness varies visibly.',
        'C: base top, 2 triangles; zero normal deviation. A-B is a real 90 deg crease; A-C is 56.145 deg.',
        'No rendering, lighting, material, geometry or source-file edits. Normal RGB encodes XYZ; it is not illumination.',
    ]
    for index, line in enumerate(lines):
        draw.text((20, 1151 + index * 17), line, fill='#101821', font=small)
    canvas.save(HERE / '00301161_evidence.png')


if __name__ == '__main__':
    main()
