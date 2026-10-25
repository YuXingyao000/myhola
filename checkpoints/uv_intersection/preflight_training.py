"""Before launching: check the real source, packed CAD batch and optimizer path."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import time

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import torch

from src.brepnet.dataset import AutoEncoder_joint_dataset
from src.brepnet.models import build_model
from src.brepnet.models.vae.intersection import AutoEncoder_intersection


def to_device(value):
    if isinstance(value, torch.Tensor):
        return value.cuda()
    if isinstance(value, dict):
        return {key: to_device(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_device(item) for item in value]
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    torch.manual_seed(42)
    torch.set_float32_matmul_precision('highest')
    with initialize_config_dir(config_dir=str(root / 'configs'), version_base=None):
        cfg = compose(config_name='train_vae_uv_intersection', overrides=[
            f'model.init_uv_decoder_checkpoint={args.source}',
            'dataset.raw_data_root=/mnt/d/data/deepcad_v7',
        ])
    dataset = AutoEncoder_joint_dataset('validation', cfg.dataset)
    batch = to_device(dataset.collate_fn([dataset[0], dataset[1]]))
    model = build_model(cfg.model).cuda().eval()
    report = {
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'Implementation/source migration check, not reconstruction-quality evaluation.',
        'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0), 'seed': 42,
        'sample_ids': list(batch['v_prefix']), 'face_counts': batch['face_counts'].tolist(),
        'initialization': model.initialization_report,
        'source_sha_expected': cfg.model.init_uv_decoder_sha256,
        'config': OmegaConf.to_container(cfg, resolve=True),
    }

    # Compare retained computations to the actual trained source, before any update.
    checkpoint = torch.load(args.source, map_location='cpu', weights_only=False, mmap=True)
    old_cfg = OmegaConf.to_container(OmegaConf.create(checkpoint['hyper_parameters']['model']), resolve=True)
    for key in list(old_cfg):
        if key.startswith('init_'):
            old_cfg[key] = None
    old_cfg.update(checkpoint=None, freeze_encoder=False, mode='train')
    reference = AutoEncoder_intersection(old_cfg).cuda().eval()
    reference.load_state_dict({key.removeprefix('model.'): value for key, value in checkpoint['state_dict'].items()}, strict=True)
    with torch.no_grad():
        old_encoded = reference.encode(batch)['face_features']
        new_encoded = model.encode(batch)['face_features']
        old_z = reference.sample(old_encoded, v_is_test=True)[0]
        new_z = model.sample(new_encoded, v_is_test=True)[0]
        old_f = reference.decoder.face_features(old_z, batch['face_counts'])
        ids = torch.repeat_interleave(torch.arange(len(batch['face_counts']), device='cuda'), batch['face_counts'])
        new_f = model.decoder.global_attention(new_z, ids)
        old_face = reference.decoder.face_geometry(old_f)
        new_face = model.decoder.face_decoder(new_f)
        pairs = {'encoder': (old_encoded, new_encoded), 'posterior_mean': (old_z, new_z),
                 'global_attention': (old_f, new_f), 'face_local': (old_face.local_points, new_face['local']),
                 'face_bbox': (old_face.center_scale, new_face['placement']), 'face_xyz': (old_face.points, new_face['xyz'])}
        report['retained_forward_max_abs'] = {key: float((old-new).abs().max()) for key, (old,new) in pairs.items()}
        for old, new in pairs.values():
            torch.testing.assert_close(old, new, rtol=1e-6, atol=1e-7)
    del reference, checkpoint, pairs, old_encoded, new_encoded, old_z, new_z, old_f, new_f, old_face, new_face
    gc.collect()
    torch.cuda.empty_cache()

    model.train()
    assert not model.encoder.training and not model.posterior.training
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=2e-5)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        losses, data = model(batch, v_test=False)
    assert data == {}
    losses['total_loss'].backward()
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for _,p in trainable)
    assert all(p.grad is None for p in model.encoder.parameters())
    assert all(p.grad is None for p in model.posterior.parameters())
    gradient_norm = torch.nn.utils.clip_grad_norm_([p for _,p in trainable], 1., error_if_nonfinite=True)
    uv_head = model.decoder.intersection.uv_head
    assert sum(p.grad.abs().sum().item() for p in uv_head.parameters()) > 0
    optimizer.step()
    torch.cuda.synchronize()
    report.update(losses={key: float(value.detach()) for key,value in losses.items()},
                  trainable_parameter_count=sum(p.numel() for _,p in trainable),
                  frozen_parameter_count=sum(p.numel() for p in model.parameters() if not p.requires_grad),
                  decoder_gradients_all_finite=True, frozen_encoder_gradients_none=True,
                  gradient_norm_before_clip=float(gradient_norm),
                  peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
                  training_step_seconds=time.perf_counter()-started)
    model.eval()
    with torch.no_grad():
        first, exported = model(batch, v_test=True)
        second, no_export = model(batch, v_test=False, v_use_mean=True)
    assert no_export == {}
    for key in first:
        torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)
    assert all(str(item['representation']) == 'dual_uv_surface_v1' and 'pred_edge' not in item for item in exported['items'])
    report.update(validation_mean_repeat_exact=True, dual_surface_export=True, passed=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({key: report[key] for key in ('passed','face_counts','retained_forward_max_abs','trainable_parameter_count','peak_allocated_GiB','training_step_seconds')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
