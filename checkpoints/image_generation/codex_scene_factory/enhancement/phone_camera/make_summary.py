"""固定六件原图 / LUT / 最后diffusion候选；失败也占原来的位置。"""
from pathlib import Path
import json
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
ROOT = Path('/mnt/d/data/cad_scene_enhancements')
SAMPLES = json.loads((HERE / 'samples.json').read_text())
OUTPUT = ROOT / 'office_phone_comparison_20261027_001'
OUTPUT.mkdir(exist_ok=False)
LUT = ROOT / 'office_phone_lut_20261027_001'
DIFFUSION = ROOT / 'office_phone_depth_guard_matte_20261027_001'
font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 18)
small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 15)
labels = ['Blender original', 'Adaptive LUT | mix 0.5', 'Depth-guided diffusion']
observations = {
    '00301161': 'More photographic surroundings; unchanged CAD material. Candidate, not accepted.',
    '00489180': 'Less artificial glare; surroundings redrawn. Candidate, not accepted.',
    '00495463': 'FAILURE: extra transparent / metallic structure next to the thin part.',
    '00182999': 'FAILURE: a new metal base and sheet appear underneath the cross.',
    '00269315': 'More photographic surroundings; target opening kept. Candidate, not accepted.',
    '00320493': 'More natural surface texture; props change. Candidate, not accepted.',
}
manifest = []
for sample in SAMPLES:
    sid=sample['model_id']
    paths=[Path(sample['rgb_file']), LUT/sid/'paired_mix_0.5.png', DIFFUSION/sid/'strength_0.65'/'protected.png']
    manifest.append(dict(model_id=sid,files=list(map(str,paths)),note=observations[sid]))
for page in range(2):
    canvas=Image.new('RGB',(1152,1320),'#f6f7f8'); draw=ImageDraw.Draw(canvas)
    for row, item in enumerate(manifest[page*3:(page+1)*3]):
        y=row*440
        for col,(title,path) in enumerate(zip(labels,item['files'])):
            draw.text((col*384+8,y+7),title,fill='#151c28',font=font)
            canvas.paste(Image.open(path).convert('RGB').resize((384,384),Image.Resampling.LANCZOS),(col*384,y+32))
        draw.text((8,y+420),item['model_id']+' | '+item['note'],fill='#a22626' if 'FAILURE' in item['note'] else '#334155',font=small)
    canvas.save(OUTPUT/f'comparison_{page:02d}.jpg',quality=96)
# 正文保留一个普通收益例和一个明确失败；完整六件仍按固定顺序在上面两页。
hero=Image.new('RGB',(1536,1100),'white');draw=ImageDraw.Draw(hero)
for row, sid in enumerate(['00301161','00182999']):
    item=next(x for x in manifest if x['model_id']==sid);y=row*550
    for col,(title,path) in enumerate(zip(labels,item['files'])):
        draw.text((col*512+10,y+7),title,fill='#151c28',font=font)
        hero.paste(Image.open(path).convert('RGB'),(col*512,y+32))
hero.save(OUTPUT/'preview.jpg',quality=96)
(OUTPUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(OUTPUT)
