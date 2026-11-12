"""纯 GT 照片训练：冻结完整 AdaLN，训练 patch ControlNet 支路。"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn

from src.common.io import sha256_file

from .adaln_zero_photo import AdaLNZeroPhotoDiffusion
from .controlnet import AdaLNControlNetDenoiser
from .preference import per_sample_mse


def backbone_config(checkpoint: dict) -> dict:
    """仅从完整纯 GT AdaLN checkpoint 继承结构，不重新访问其旧权重源。"""
    saved = checkpoint["hyper_parameters"]["model"]
    if saved["name"] != "AdaLNZeroPhotoDiffusion":
        raise ValueError("ControlNet initialization requires a complete GT AdaLN-Zero checkpoint")
    return {key: saved[key] for key in (
        "name", "modulation_hidden_width", "photo_global_step", "legacy_model_config",
    )}


def model_state(checkpoint: dict) -> dict:
    if any(not key.startswith("model.") for key in checkpoint["state_dict"]):
        raise ValueError("Expected a complete model.* checkpoint")
    return {key[6:]: value for key, value in checkpoint["state_dict"].items()}


def check_controlnet_checkpoint(config: dict, checkpoint: dict) -> None:
    """结构语义也必须一致；strict tensor loading 无法区分所有同形状改动。"""
    saved = checkpoint["hyper_parameters"]["model"]
    fields = ("name", "architecture_version", "backbone_config")
    if any(config[key] != saved[key] for key in fields):
        raise ValueError("ControlNet checkpoint architecture differs")


class ControlNetPhotoDiffusion(nn.Module):
    """单次冻结 DINO 同时提供 CLS/patch；唯一训练目标是 GT epsilon MSE。"""

    def __init__(self, cfg: dict):
        super().__init__()
        if cfg["architecture_version"] != "adaln_patch_control_v1":
            raise ValueError("Unsupported AdaLN ControlNet architecture version")
        restoring = cfg.get("restoring_checkpoint", False)
        source = None
        if restoring:
            self.backbone_config = cfg["backbone_config"]
            self.initialization = cfg["initialization"]
        else:
            if not cfg.get("init_adaln_checkpoint") or cfg.get("checkpoint"):
                raise ValueError("Choose AdaLN initialization or a complete ControlNet checkpoint")
            source_path = Path(cfg["init_adaln_checkpoint"]).resolve()
            source = torch.load(source_path, map_location="cpu", mmap=True, weights_only=False)
            self.backbone_config = backbone_config(source)
            self.initialization = {
                "checkpoint": str(source_path), "sha256": sha256_file(source_path),
                "global_step": source["global_step"], "optimizer_restored": False,
                "initial_function_preserved": True,
            }
        baseline = AdaLNZeroPhotoDiffusion({
            **self.backbone_config, "restoring_checkpoint": True,
            "dino_repo": cfg.get("dino_repo"), "legacy_source_cache": cfg.get("legacy_source_cache"),
        })
        if source is not None:
            # 先恢复全部已学门值，再复制控制层；绝不重新初始化原 AdaLN。
            baseline.load_state_dict(model_state(source), strict=True)
        self.image_encoder = baseline.image_encoder.requires_grad_(False).eval()
        self.noise_scheduler = baseline.noise_scheduler
        patch_width = baseline.policy.img_fc[0].in_features
        self.denoiser = AdaLNControlNetDenoiser(baseline.policy, patch_width)
        if cfg.get("checkpoint") and not cfg.get("resuming_training", False):
            saved = torch.load(cfg["checkpoint"], map_location="cpu", mmap=True, weights_only=False)
            check_controlnet_checkpoint(cfg, saved)
            self.load_state_dict(model_state(saved), strict=True)

    def train(self, mode: bool = True):
        super().train(mode)
        self.image_encoder.eval()
        return self

    def image_condition(self, images: Tensor) -> tuple[Tensor, Tensor]:
        # 同一张图、同一次前向；register tokens 不进入 patch memory。
        with torch.no_grad():
            features = self.image_encoder.forward_features(images)
            condition = self.denoiser.backbone.image_condition(features["x_norm_clstoken"])
        return condition, features["x_norm_patchtokens"]

    def forward(self, batch: dict, v_test: bool = False) -> dict[str, Tensor]:
        clean = batch["gt_latent"]
        time = torch.randint(self.noise_scheduler.config.num_train_timesteps,
                             (len(clean),), device=clean.device)
        noise = torch.randn_like(clean)
        noisy = self.noise_scheduler.add_noise(clean, noise, time)
        condition, patches = self.image_condition(batch["gt_image"])
        prediction = self.denoiser(noisy, time, condition, patches)
        loss = per_sample_mse(prediction, noise).mean()
        return {"ddpm_loss": loss, "total_loss": loss}

    @torch.no_grad()
    def sample_latents(self, images: Tensor, *, initial_generator=None, step_generator=None) -> Tensor:
        """保留原 CPU 初始噪声与 DDPM1000；图像仅编码一次。"""
        if self.training:
            raise ValueError("Call eval() before sampling")
        condition, patches = self.image_condition(images)
        latent = torch.randn((len(images), 30, 32), generator=initial_generator).to(images.device)
        self.noise_scheduler.set_timesteps(1000, device=images.device)
        for time in self.noise_scheduler.timesteps:
            prediction = self.denoiser(latent, time.expand(len(images)), condition, patches)
            latent = self.noise_scheduler.step(prediction, time, latent, generator=step_generator).prev_sample
        return latent


def prepare_controlnet_model_config(cfg, model_cfg, resume: str | None) -> None:
    from omegaconf import OmegaConf

    if cfg.trainer.init_from_checkpoint:
        raise ValueError("Use init_adaln_checkpoint, complete model.checkpoint, or resume_from_checkpoint")
    if cfg.eval.enabled or cfg.trainer.evaluate or cfg.trainer.diffusion_validation_inference:
        raise ValueError("Use sample_latents with the explicit original VAE evaluation pipeline")
    full = model_cfg.get("checkpoint")
    if full and model_cfg.get("init_adaln_checkpoint") and not resume:
        raise ValueError("Choose AdaLN initialization or a complete ControlNet checkpoint")
    restore_path = resume or full
    source_path = restore_path or model_cfg.get("init_adaln_checkpoint")
    if not source_path:
        raise ValueError("ControlNet requires explicit AdaLN initialization or a complete checkpoint")
    saved = torch.load(str(source_path), map_location="cpu", mmap=True, weights_only=False)
    previous = saved["hyper_parameters"]
    if restore_path:
        if dict(cfg.dataset.provenance) != previous["dataset"]["provenance"]:
            raise ValueError("Cannot restore ControlNet with changed photo GT data/provenance")
        expected = previous["model"]["backbone_config"]
        if model_cfg.get("backbone_config") is not None and model_cfg.backbone_config != expected:
            raise ValueError("ControlNet backbone config differs from checkpoint")
        OmegaConf.update(model_cfg, "backbone_config", expected, force_add=True)
        check_controlnet_checkpoint(model_cfg, saved)
        OmegaConf.update(model_cfg, "initialization", previous["model"]["initialization"], force_add=True)
    else:
        if dict(cfg.dataset.provenance.gt) != previous["dataset"]["provenance"]:
            raise ValueError("AdaLN initialization and ControlNet use different GT data/provenance")
        OmegaConf.update(model_cfg, "backbone_config", backbone_config(saved), force_add=True)
    OmegaConf.update(model_cfg, "restoring_checkpoint", bool(restore_path), force_add=True)
    OmegaConf.update(model_cfg, "resuming_training", bool(resume), force_add=True)


def load_controlnet_photo(checkpoint: str | Path, device: str | torch.device = "cpu", *,
                          dino_repo: str | None = None,
                          legacy_source_cache: str | None = None) -> ControlNetPhotoDiffusion:
    """完整权重独立加载，供照片采样和原 VAE/OCC 评估调用。"""
    saved = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False)
    cfg = dict(saved["hyper_parameters"]["model"])
    if cfg["name"] != "ControlNetPhotoDiffusion":
        raise ValueError("Expected a ControlNetPhotoDiffusion checkpoint")
    cfg.update(checkpoint=str(checkpoint), init_adaln_checkpoint=None,
               restoring_checkpoint=True, resuming_training=False)
    if dino_repo is not None:
        cfg["dino_repo"] = dino_repo
    if legacy_source_cache is not None:
        cfg["legacy_source_cache"] = legacy_source_cache
    return ControlNetPhotoDiffusion(cfg).to(device).eval()
