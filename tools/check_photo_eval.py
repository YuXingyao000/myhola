"""Check real photo checkpoint mapping and one full sampling/decode chain."""

from pathlib import Path
import json
import time

import numpy as np
import torch

from src.brepnet.diffusion.photo_baseline import PhotoBaselineDiffusion, load_photo_svr
from src.brepnet.diffusion.legacy import sample_candidates
from src.brepnet.data.photo_dataset import photo_transform
from src.brepnet.data.preference_generate import candidate_seed


def main():
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    root = Path('/mnt/d/python/checkpoints/diffusion/photo_baseline/evaluation_20261026')
    ckpt = root / 'weights/photo_latest.ckpt'
    original = '/mnt/d/ckpt/HoLa/Diffusion_svr_sq30_1500k.ckpt'
    model, provenance = load_photo_svr(original, ckpt)
    saved = torch.load(ckpt, map_location='cpu', mmap=True, weights_only=False)
    wrapper = PhotoBaselineDiffusion({**saved['hyper_parameters']['model'], 'restoring_checkpoint': True}).eval()
    wrapper.load_state_dict({k.removeprefix('model.'): v for k,v in saved['state_dict'].items()}, strict=True)
    torch.manual_seed(42)
    features = torch.tensor(np.load('checkpoints/diffusion/photo_baseline/cache_v1/validation_features.npy')[:2])
    noisy, timestep = torch.randn(2,30,32), torch.tensor([0,731])
    with torch.no_grad():
        expected = wrapper.policy(noisy, timestep, wrapper.policy.image_condition(features))
        actual = model.diffuse(noisy, timestep, model.img_fc(features)[:,None])
    error = (expected-actual).abs().max().item()
    assert torch.equal(actual, expected), error
    del wrapper, saved
    mid = Path('checkpoints/diffusion/svr_comparison_20261017/validation100.txt').read_text().split()[0]
    with np.load(f'/mnt/d/data/deepcad_v7_cond/{mid}/real_photo.npz') as data:
        images = photo_transform()(data['flux'])[None]
    started = time.monotonic()
    result = sample_candidates(model.cuda(), images, [candidate_seed(42,mid,0)])[0]
    dest=root/'preflight'
    dest.mkdir(exist_ok=True)
    np.savez_compressed(dest/'data.npz',**result['raw'])
    np.save(dest/'latent.npy',result['latent'].numpy())
    report={'actual_checkpoint_step':provenance['photo']['global_step'],'fp32_forward_bitwise_equal':True,
            'max_abs_forward_error':error,'image_id':mid,'sampling_seconds':time.monotonic()-started,
            'latent_shape':list(result['latent'].shape),
            'raw_shapes':{k:list(v.shape) for k,v in result['raw'].items()},'original_ae_retained':True}
    (root/'PREFLIGHT.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':
    main()
