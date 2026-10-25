"""原 HoLa concat 网络的照片 DDPM baseline；不含偏好或几何辅助目标。"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .legacy import build_legacy_svr, load_legacy_svr
from .preference import LegacySVRDenoiser, per_sample_mse


class PhotoBaselineDiffusion(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        options = {"source_cache": cfg.get("legacy_source_cache"), "dino_repo": cfg.get("dino_repo")}
        if cfg.get("restoring_checkpoint", False):
            legacy, self.provenance = build_legacy_svr(cfg["legacy_model_config"], **options)
        else:
            legacy, self.provenance = load_legacy_svr(cfg["legacy_checkpoint"], **options)
        self.legacy_model_config = self.provenance["model_config"]
        self.noise_scheduler = legacy.noise_scheduler
        self.policy = LegacySVRDenoiser(legacy).requires_grad_(True)
        # 保存原 DINO 权重使新 checkpoint 自含；训练读其离线 FP32 输出。
        self.image_encoder = legacy.img_model.requires_grad_(False).eval()
        del legacy

    def train(self, mode: bool = True):
        super().train(mode)
        self.image_encoder.eval()
        return self

    def forward(self, batch: dict, v_test: bool = False) -> dict[str, Tensor]:
        clean = batch["gt_latent"]
        time = torch.randint(self.noise_scheduler.config.num_train_timesteps,
                             (len(clean),), device=clean.device)
        noise = torch.randn_like(clean)
        noisy = self.noise_scheduler.add_noise(clean, noise, time)
        condition = self.policy.image_condition(batch["image_features"])
        prediction = self.policy(noisy, time, condition)
        loss = per_sample_mse(prediction, noise).mean()
        return {"ddpm_loss": loss, "total_loss": loss}

    @torch.no_grad()
    def sample_latents(self, images: Tensor, *, initial_generator=None, step_generator=None) -> Tensor:
        """照片归一化张量 → 原 DDPM latent；后续使用匹配的原版 AE 解码。"""
        if self.training:
            raise ValueError("Call eval() before sampling")
        condition = self.policy.image_condition(self.image_encoder(images))
        latent = torch.randn((len(images), 30, 32), generator=initial_generator).to(images.device)
        self.noise_scheduler.set_timesteps(1000, device=images.device)
        for time in self.noise_scheduler.timesteps:
            prediction = self.policy(latent, time.expand(len(images)), condition)
            latent = self.noise_scheduler.step(prediction, time, latent, generator=step_generator).prev_sample
        return latent
