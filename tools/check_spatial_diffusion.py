"""单 GPU 真实照片/原 VAE 合同检查；不把短运行当作重建效果证据。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader

from src.brepnet.data.spatial_photo_dataset import SpatialPhotoDataset
from src.brepnet.diffusion.spatial_model import SpatialPhotoDiffusion


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    with initialize_config_dir(config_dir=str(Path("configs").resolve()), version_base=None):
        cfg = compose(config_name="train_diffusion_spatial_photo")
    started = time.monotonic()
    training = SpatialPhotoDataset("training", cfg.dataset)
    validation = SpatialPhotoDataset("validation", cfg.dataset)
    batch = next(iter(DataLoader(training, batch_size=args.batch_size,
                                 collate_fn=training.collate_fn)))
    batch = {key: value.cuda() if torch.is_tensor(value) else value for key, value in batch.items()}
    model = SpatialPhotoDiffusion(OmegaConf.to_container(cfg.model, resolve=True)).cuda().train()
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=1e-4)
    torch.cuda.reset_peak_memory_stats()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        losses = model(batch)
    losses["total_loss"].backward()
    missing, nonfinite = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            if parameter.grad is None:
                missing.append(name)
            elif not torch.isfinite(parameter.grad).all():
                nonfinite.append(name)
        else:
            assert parameter.grad is None, name
    assert not missing and not nonfinite, (missing, nonfinite)
    coordinate_grad = model.condition_encoder.coordinate_projection[0].weight.grad.norm().item()
    cross_grads = [block.cross_attention.attention.q_proj_weight.grad.norm().item()
                   for block in model.denoiser.blocks]
    assert coordinate_grad > 0 and min(cross_grads) > 0
    torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    peak = torch.cuda.max_memory_allocated() / 1024 ** 3
    del optimizer
    model.eval()
    val = validation[0]
    image = val["gt_image"][None].cuda()
    calls = []
    hook = model.condition_encoder.register_forward_hook(lambda *args: calls.append(1))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        sample = model.sample_latents(image, generator=torch.Generator("cuda").manual_seed(43), num_steps=4)
    hook.remove()
    assert calls == [1] and torch.isfinite(sample).all()
    # 未训练样本不能代表几何能力；用真实 GT mean 检查固定原 VAE 解码接口。
    decoded = model.decode_latents(val["gt_latent"][None].cuda())[0]
    assert decoded["raw"]["pred_face"].ndim == 4
    with torch.no_grad():
        memory = model.extract_condition(image)
    report = {"status": "implementation_checks_passed_not_quality_evaluation",
              "torch": torch.__version__, "device": torch.cuda.get_device_name(),
              "seed": 42, "training_count": len(training.ids), "training_items_per_epoch": len(training),
              "validation_count": len(validation),
              "training_ids": batch["v_prefix"], "validation_id": val["v_prefix"],
              "batch_size": len(batch["gt_image"]), "loss_bf16": losses["total_loss"].item(),
              "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
              "frozen_parameters": sum(p.numel() for p in model.parameters() if not p.requires_grad),
              "memory_shape": list(memory.shape), "sample_shape": list(sample.shape),
              "coordinate_gradient_norm": coordinate_grad, "min_cross_attention_gradient_norm": min(cross_grads),
              "all_trainable_parameters_have_finite_gradient": True,
              "frozen_dino_vae_have_no_gradient": True, "sampling_memory_evaluations": len(calls),
              "sampling_steps_for_interface_check": 4, "gt_decoded_face_shape": list(decoded["raw"]["pred_face"].shape),
              "peak_training_allocated_gib": peak, "seconds": time.monotonic() - started,
              "data_provenance": training.provenance}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
