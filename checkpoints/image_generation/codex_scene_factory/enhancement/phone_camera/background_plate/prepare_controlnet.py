"""下载固定版本的公开 Union Pro 2 ControlNet，不安装依赖或改变运行环境。"""

import hashlib
import json
from pathlib import Path
import time

import requests


REPO = 'Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0'
REVISION = '5d700aaad96c5ddcdf8a38ef9b22a82aac2c38e5'
OUTPUT = Path('/mnt/d/model/Flux1_ControlNet_Union_Pro_2')
OUTPUT.mkdir(parents=True, exist_ok=False)
response = requests.get(f'https://huggingface.co/api/models/{REPO}/revision/{REVISION}',
                        params={'blobs': 'true'}, timeout=30)
response.raise_for_status()
metadata = response.json()
(OUTPUT / 'repository.json').write_text(json.dumps(metadata, indent=2) + '\n')
names = ['config.json', 'README.md', 'LICENSE', 'diffusion_pytorch_model.safetensors']
known = {item['rfilename']: item for item in metadata['siblings']}
records = []
for name in names:
    if name not in known:
        continue
    started = time.monotonic()
    temporary = OUTPUT / (name + '.partial')
    digest, size, next_progress = hashlib.sha256(), 0, 256 * 1024**2
    url = f'https://huggingface.co/{REPO}/resolve/{REVISION}/{name}'
    with requests.get(url, stream=True, timeout=(30, 120)) as response:
        response.raise_for_status()
        with temporary.open('xb') as stream:
            for block in response.iter_content(chunk_size=4 * 1024**2):
                stream.write(block)
                digest.update(block)
                size += len(block)
                if size >= next_progress:
                    print(f'{name}: {size / 1024**3:.2f} GiB', flush=True)
                    next_progress += 256 * 1024**2
    expected = known[name]
    assert size == expected['size'], (name, size, expected['size'])
    actual = digest.hexdigest()
    if 'lfs' in expected:
        assert actual == expected['lfs']['sha256'], name
    temporary.rename(OUTPUT / name)
    records.append(dict(file=name, bytes=size, sha256=actual,
                        seconds=time.monotonic() - started, url=url))
    print(name, size, actual, flush=True)
(OUTPUT / 'source.json').write_text(json.dumps({
    'repository': REPO, 'revision': REVISION, 'files': records,
    'license': metadata.get('cardData', {}).get('license'),
    'combination_with_krea': 'research candidate; not claimed officially validated',
}, indent=2) + '\n')
