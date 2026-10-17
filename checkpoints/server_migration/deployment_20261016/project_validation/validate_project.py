"""在迁移目标机验证真实 training 小批次；不启动正式训练、不读取 ABC。"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

import numpy as np
import torch

REPO = Path('/mnt/d/python')
sys.path.insert(0, str(REPO))
OUT = Path('/mnt/d/migration/project-validation')
IDS = ['00000061', '00000070']


def on_gpu(value):
    if isinstance(value, torch.Tensor):
        return value.cuda()
    if isinstance(value, dict):
        return {k: on_gpu(v) for k, v in value.items()}
    return value


def shapes(value):
    if isinstance(value, (torch.Tensor, np.ndarray)):
        return list(value.shape)
    if isinstance(value, dict):
        return {k: shapes(v) for k, v in value.items()}
    return value


def check_gradients(module):
    params = [(name, p) for name, p in module.named_parameters() if p.requires_grad]
    present = [(name, p) for name, p in params if p.grad is not None]
    assert present, 'No trainable gradients'
    bad = [name for name, p in present if not torch.isfinite(p.grad).all()]
    assert not bad, bad[:10]
    nonzero = sum(bool(p.grad.count_nonzero()) for _, p in present)
    assert nonzero > 0
    return dict(trainable_tensors=len(params), gradients_present=len(present),
                nonzero_gradient_tensors=nonzero, all_present_gradients_finite=True,
                missing=[name for name, p in params if p.grad is None])


def inputs(report):
    split = REPO / 'src/brepnet/data/list/deduplicated_deepcad_training_7_30.txt'
    assert set(IDS) <= set(split.read_text().splitlines())
    report['input_ids'] = IDS
    report['source_split'] = str(split)
    report['source_split_sha256'] = hashlib.sha256(split.read_bytes()).hexdigest()
    report['input_split_role'] = 'training'
    subset = OUT / 'training_subset.txt'
    subset.write_text('\n'.join(IDS) + '\n')
    report['input_files'] = {}
    for model_id in IDS:
        assets = [Path('/mnt/d/data/deepcad_v7') / model_id / 'data.npz',
                  Path('/mnt/d/data/deepcad_v7_cond') / model_id / 'imgs.npz',
                  Path('/mnt/d/data/ae_cache/1119_deepcad_aug1_11k_24') / (model_id + '_0') / 'features.npy']
        for p in assets:
            assert p.is_file(), str(p)
            report['input_files'][str(p)] = {'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
    return subset


def validate_training(report):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from src.brepnet.train import UnifiedTrainer
    from src.brepnet.dataset import Diffusion_dataset, AutoEncoder_dataset3
    from src.brepnet.models.vae import build_autoencoder

    subset = inputs(report)
    with initialize_config_dir(version_base=None, config_dir=str(REPO / 'configs')):
        cfg = compose(config_name='train_diffusion_layerwise_svr', overrides=[
            f'dataset.train_dataset={subset}', 'dataset.epoch_scale_factor=1',
            'trainer.devices=1', 'trainer.batch_size=2', 'trainer.num_workers=0',
            f'trainer.output_dir={OUT / "trainer-runtime"}',
        ])
    dataset = Diffusion_dataset('training', cfg.dataset)
    assert len(dataset) == 2
    batch = on_gpu(next(iter(torch.utils.data.DataLoader(dataset, batch_size=2, num_workers=0, collate_fn=dataset.collate_fn))))
    report['diffusion_batch_shapes'] = shapes(batch)
    assert batch['cached_latent_stats'].shape == (2, 30, 64)
    assert batch['conditions']['imgs'].shape == (2, 3, 224, 224)
    assert batch['rotation_id'].tolist() == [0, 0]
    model = UnifiedTrainer(cfg).model.cuda().train()
    report['diffusion_num_layers'] = cfg.model.denoiser.num_layers
    report['diffusion_trainable_parameters'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    report['autoencoder_checkpoint'] = cfg.model.autoencoder.checkpoint
    report['diffusion_checkpoint'] = None
    report['diffusion_checkpoint_note'] = '当前 diffusion 使用随机初始化，仅验证完整架构运行；源机在训权重按用户要求暂不复制。'
    hook = model.condition_encoder.register_forward_hook(lambda module, args, result: report.update(condition_tokens_shape=list(result.shape)))
    with torch.autocast('cuda', dtype=torch.bfloat16):
        losses = model(batch)
    assert torch.isfinite(losses['total_loss'])
    losses['total_loss'].backward()
    torch.cuda.synchronize()
    report['diffusion_loss'] = float(losses['total_loss'].detach())
    report['diffusion_gradients'] = check_gradients(model.denoiser)
    report['image_projection_gradients'] = check_gradients(model.condition_encoder.image_encoder.projection)
    assert all(p.grad is None and not p.requires_grad for p in model.autoencoder.parameters())
    assert all(p.grad is None and not p.requires_grad for p in model.condition_encoder.image_encoder.img_model.parameters())
    report['frozen_dino_and_vae_no_gradients'] = True
    report['diffusion_peak_gpu_bytes'] = torch.cuda.max_memory_allocated()
    hook.remove()
    ae_cfg = OmegaConf.to_container(cfg.model.autoencoder, resolve=True)
    del losses, model, batch
    gc.collect()
    torch.cuda.empty_cache()

    vae_data_cfg = dict(raw_data_root='/mnt/d/data/deepcad_v7', train_dataset=str(subset),
                        val_dataset=str(subset), test_dataset=str(subset), epoch_scale_factor=1,
                        is_aug=0, is_overfit=False, condition_name=[], condition_data_root=None,
                        cached_condition=False, num_points=10000)
    vae_dataset = AutoEncoder_dataset3('training', vae_data_cfg)
    assert len(vae_dataset) == 2
    vae_batch = on_gpu(vae_dataset.collate_fn([vae_dataset[0]]))
    report['vae_batch_shapes'] = shapes(vae_batch)
    ae_cfg['mode'] = 'train'
    vae = build_autoencoder(ae_cfg).cuda().train()
    # Exercise the real trainer's trusted Lightning-init path under PyTorch 2.8.
    init_cfg = OmegaConf.create({'trainer': {'init_from_checkpoint': ae_cfg['checkpoint'],
                                           'resume_from_checkpoint': None}})
    UnifiedTrainer._init_model_from_checkpoint(SimpleNamespace(cfg=init_cfg, model=vae, stage='vae'))
    report['trainer_init_trusted_checkpoint'] = True
    with torch.autocast('cuda', dtype=torch.bfloat16):
        vae_losses, _ = vae(vae_batch)
    assert torch.isfinite(vae_losses['total_loss'])
    vae_losses['total_loss'].backward()
    torch.cuda.synchronize()
    report['vae_loss'] = float(vae_losses['total_loss'].detach())
    report['vae_gradients'] = check_gradients(vae)
    vae.zero_grad(set_to_none=True)
    vae.eval()
    latent = np.load('/mnt/d/data/ae_cache/1119_deepcad_aug1_11k_24/00000061_0/features.npy')[:, :32]
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        decoded = vae.decode_latents(torch.as_tensor(latent, device='cuda'))
    for key in ('pred_face', 'pred_edge'):
        assert np.isfinite(decoded[key]).all(), key
    report['vae_cached_mean_decode_shapes'] = shapes(decoded)
    report['post_imports'] = []
    for name in ['src.brepnet.post.construct_brep', 'src.brepnet.post.occ_fit', 'src.brepnet.post.occ_topology', 'src.brepnet.eval.run', 'src.brepnet.data.extract_brep']:
        importlib.import_module(name)
        report['post_imports'].append(name)
    report['scope'] = '真实 training batch 的 CUDA/BF16 前向、反向、冻结与缓存解码、OCC/post 入口；无优化器更新和正式训练。'


def validate_flux(report):
    from PIL import Image
    from src.brepnet.data.datagen.external.generation.config import Img2BrepConfig
    from src.brepnet.data.datagen.external.generation.pipeline import Img2BrepPipeline

    inputs(report)
    cfg = Img2BrepConfig(pre_encode_text=False, num_steps=2, seed=42)
    start = time.monotonic()
    wrapper = Img2BrepPipeline(cfg)
    report['full_weight_load_seconds'] = time.monotonic() - start
    with np.load('/mnt/d/data/deepcad_v7_cond/00000061/imgs.npz') as data:
        raw = data['svr_imgs'][0]
    image = Image.fromarray(raw).convert('RGB').resize((256, 256))
    report['input_image_shape'] = [256, 256, 3]
    generator = torch.Generator(device='cuda').manual_seed(42)
    with torch.inference_mode():
        output = wrapper.pipeline(image=image, prompt=cfg.pos_prompt, num_inference_steps=2,
                                  guidance_scale=3.5, true_cfg_scale=1.0, width=256, height=256,
                                  max_sequence_length=128, max_area=256 * 256, generator=generator, output_type='pt')
    values = output.images[0]
    assert torch.isfinite(values).all(), 'Nonfinite FLUX output'
    out = (values.clamp(0, 1).permute(1, 2, 0) * 255).round().to(torch.uint8).cpu().numpy()
    Image.fromarray(out).save(OUT / 'flux_training_smoke.png')
    report.update(output_shape=list(out.shape), output_finite=True,
                  output_path=str(OUT / 'flux_training_smoke.png'), seed=42, steps=2,
                  output_sha256=hashlib.sha256((OUT / 'flux_training_smoke.png').read_bytes()).hexdigest(),
                  gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  transformer_path=str(cfg.transformer_path), base_model_path=str(cfg.base_model_path),
                  scope='仓库 Img2BrepPipeline 完整加载 Q8 GGUF+CLIP/T5/VAE，256px/2步真实 training 渲染条件编辑；仅环境连通验证，不判断合成图质量或几何一致性。')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['training', 'flux'])
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    os.chdir(REPO)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    np.random.seed(42)
    git = subprocess.run(['git', 'rev-parse', 'HEAD'], text=True, capture_output=True)
    report = dict(phase=args.phase, torch=torch.__version__, device=torch.cuda.get_device_name(0), seed=42,
                  git_commit=git.stdout.strip() if git.returncode == 0 else None,
                  git_metadata_ready=git.returncode == 0,
                  validation_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    started = time.monotonic()
    try:
        (validate_training if args.phase == 'training' else validate_flux)(report)
        report['status'] = 'passed'
    except Exception as exc:
        report.update(status='failed', error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        report['elapsed_seconds'] = time.monotonic() - started
        (OUT / (args.phase + '.json')).write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        print(json.dumps({k: report[k] for k in ('phase', 'status', 'elapsed_seconds')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
