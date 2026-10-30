"""下载固定版本的IC-Light背景条件权重及所需SD1.5组件，不执行第三方代码。"""

import hashlib
import json
from pathlib import Path
import time

import requests


OUTPUT = Path('/mnt/d/model/ICLight_SD15')
BASE_REPO = 'stablediffusionapi/realistic-vision-v51'
BASE_REVISION = '19e3643d7d963c156d01537188ec08f0b79a514a'
RELIGHT_REPO = 'lllyasviel/ic-light'
RELIGHT_REVISION = '9cad1878695f546a7fb9eaca14e2a89131ba5ffe'
CODE_REVISION = 'bcf3f29ca85be8a4686215f477b546f5030be8b7'


def download(url, path, expected=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.partial')
    started = time.monotonic()
    digest, size, next_progress = hashlib.sha256(), 0, 256 * 1024**2
    with requests.get(url, stream=True, timeout=(30, 120)) as response:
        response.raise_for_status()
        with temporary.open('xb') as stream:
            for block in response.iter_content(4 * 1024**2):
                stream.write(block)
                digest.update(block)
                size += len(block)
                if size >= next_progress:
                    print(f'{path.name}: {size / 1024**3:.2f} GiB', flush=True)
                    next_progress += 256 * 1024**2
    sha = digest.hexdigest()
    if expected:
        assert size == expected['size'], (path, size, expected['size'])
        if 'lfs' in expected:
            assert sha == expected['lfs']['sha256'], path
    temporary.rename(path)
    print(f'完成 {path.relative_to(OUTPUT)} · {size} bytes', flush=True)
    return dict(path=str(path), url=url, bytes=size, sha256=sha, seconds=time.monotonic()-started)


def repository(repo, revision, folder, names):
    response = requests.get(f'https://huggingface.co/api/models/{repo}/revision/{revision}',
                            params={'blobs': 'true'}, timeout=30)
    response.raise_for_status()
    metadata = response.json()
    directory = OUTPUT / folder
    directory.mkdir()
    (directory / 'repository.json').write_text(json.dumps(metadata, indent=2) + '\n')
    known = {item['rfilename']: item for item in metadata['siblings']}
    files = [download(f'https://huggingface.co/{repo}/resolve/{revision}/{name}',
                      directory / name, known[name]) for name in names]
    return dict(repository=repo, revision=revision, files=files)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=False)
    base_names = ['README.md', 'model_index.json', 'scheduler/scheduler_config.json',
                  'tokenizer/merges.txt', 'tokenizer/special_tokens_map.json',
                  'tokenizer/tokenizer_config.json', 'tokenizer/vocab.json']
    for component in ['text_encoder', 'vae', 'unet']:
        weights = 'model.safetensors' if component == 'text_encoder' else 'diffusion_pytorch_model.safetensors'
        base_names += [f'{component}/config.json', f'{component}/{weights}']
    base = repository(BASE_REPO, BASE_REVISION, 'base', base_names)
    relight = repository(RELIGHT_REPO, RELIGHT_REVISION, 'relight', ['README.md', 'iclight_sd15_fbc.safetensors'])
    code = [download(f'https://raw.githubusercontent.com/lllyasviel/IC-Light/{CODE_REVISION}/{name}',
                     OUTPUT / 'official' / name)
            for name in ['gradio_demo_bg.py', 'README.md', 'LICENSE', 'requirements.txt']]
    license_record = download('https://huggingface.co/spaces/CompVis/stable-diffusion-license/raw/main/license.txt',
                              OUTPUT / 'OpenRAIL-M-license.txt')
    (OUTPUT / 'source.json').write_text(json.dumps(dict(
        base=base, relight=relight, official_code_revision=CODE_REVISION, official_files=code,
        code_license='Apache-2.0', weight_license='CreativeML OpenRAIL-M as declared in model cards',
        license_record=license_record, status='downloaded_and_hashed_not_inference_validated',
    ), indent=2) + '\n')


if __name__ == '__main__':
    main()
