"""Download pinned runtime wheels concurrently; verify official PyPI SHA256.

Only downloads installers. No environment is copied or mutated.
"""
from __future__ import annotations
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import subprocess
import time
import urllib.request

ROOT = Path('/mnt/d/migration/wheels')
ROOT.mkdir(parents=True, exist_ok=True)
PACKAGES = {
    'nvidia-cudnn-cu12': '9.10.2.21',
    'nvidia-cublas-cu12': '12.8.4.1',
    'nvidia-cufft-cu12': '11.3.3.83',
    'nvidia-cusolver-cu12': '11.7.3.90',
    'nvidia-cusparse-cu12': '12.5.8.93',
    'nvidia-cusparselt-cu12': '0.7.1',
    'nvidia-nccl-cu12': '2.27.3',
    'triton': '3.4.0',
}
parser = argparse.ArgumentParser()
parser.add_argument('--package', action='append', default=[])
parser.add_argument('--label', default='runtime')
args = parser.parse_args()
if args.package:
    PACKAGES = dict(spec.split('==', 1) for spec in args.package)
CHUNK_SIZE = 16 * 1024**2
metadata = []
for package, version in PACKAGES.items():
    with urllib.request.urlopen(f'https://pypi.org/pypi/{package}/{version}/json', timeout=30) as response:
        info = json.load(response)
    files = [x for x in info['urls'] if x['filename'].endswith('.whl')
             and 'x86_64' in x['filename'] and 'manylinux' in x['filename']
             and ('py3-none' in x['filename'] or 'cp310-cp310' in x['filename'])]
    assert len(files) == 1, (package, [x['filename'] for x in files])
    metadata.append(files[0])
(ROOT / f'{args.label}-download-manifest.json').write_text(json.dumps(metadata, indent=2))

def fetch_part(info, index):
    size = info['size']
    start = index * CHUNK_SIZE
    end = min(start + CHUNK_SIZE, size) - 1
    part = ROOT / '.parts' / info['filename'] / str(index)
    part.parent.mkdir(parents=True, exist_ok=True)
    url = info['url'].replace('https://files.pythonhosted.org/', 'https://mirrors.aliyun.com/pypi/')
    for attempt in range(12):
        have = part.stat().st_size if part.exists() else 0
        if have == end-start+1:
            return part
        if have > end-start+1:
            raise RuntimeError(f'Oversized part: {part}')
        tmp = part.with_suffix('.partial')
        subprocess.run(['curl', '-fsSL', '--connect-timeout', '20', '--max-time', '120',
                        '--retry', '2', '--range', f'{start+have}-{end}', url, '-o', str(tmp)], check=False)
        if tmp.exists():
            with part.open('ab') as stream:
                stream.write(tmp.read_bytes())
            tmp.unlink()
        time.sleep(0.2)
    raise RuntimeError(f'Incomplete part: {part}')

with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
    pending = {info['filename']: [pool.submit(fetch_part, info, i)
               for i in range((info['size']+CHUNK_SIZE-1)//CHUNK_SIZE)] for info in metadata}
    for info in metadata:
        out = ROOT / (info['filename'] + '.partial')
        digest = hashlib.sha256()
        with out.open('wb') as stream:
            for future in pending[info['filename']]:
                with future.result().open('rb') as source:
                    while block := source.read(4 * 1024**2):
                        digest.update(block)
                        stream.write(block)
        assert digest.hexdigest() == info['digests']['sha256'], info['filename']
        out.rename(ROOT / info['filename'])
        print('verified', info['filename'], digest.hexdigest(), flush=True)
(ROOT / f'{args.label}-download.complete').touch()
