"""照片 → 空间 memory → 24-block face latent DDPM；冻结原 HoLa VAE/DINO。"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import torch
from torch import Tensor, nn
from diffusers import DDPMScheduler

from .legacy import build_legacy_svr, decode_candidate, load_legacy_svr
from .spatial_blocks import SpatialMemoryDenoiser
from .spatial_conditioning import SpatialImageMemory


def architecture_contract(cfg: dict) -> dict:
    """除 tensor 形状之外，也校验坐标、memory 组成与噪声语义。"""
    return deepcopy({key: cfg[key] for key in ("architecture_version", "spatial", "denoiser", "noise")})


class SpatialPhotoDiffusion(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        if cfg["architecture_version"] != "spatial_photo_v1":
            raise ValueError("Unsupported spatial photo architecture version")
        self.contract = architecture_contract(cfg)
        options = {"source_cache": cfg.get("legacy_source_cache"), "dino_repo": cfg["dino_repo"]}
        if cfg.get("restoring_checkpoint", False):
            legacy, self.provenance = build_legacy_svr(
                cfg["legacy_model_config"], with_autoencoder=True, **options)
        else:
            from src.datasets.preference import sha256_file
            if sha256_file(Path(cfg["legacy_checkpoint"])) != cfg["frozen_source_sha256"]:
                raise ValueError("Frozen DINO/VAE source checkpoint SHA256 changed")
            legacy, self.provenance = load_legacy_svr(cfg["legacy_checkpoint"], **options)
        self.legacy_model_config = self.provenance["model_config"]
        self.image_encoder = legacy.img_model.requires_grad_(False).eval()
        self.ae_model = legacy.ae_model.requires_grad_(False).eval()
        del legacy
        # 仅保留预训练 DINO/VAE；空间编码器和新去噪器均从头训练。
        self.condition_encoder = SpatialImageMemory(cfg["spatial"])
        self.denoiser = SpatialMemoryDenoiser(cfg["denoiser"], cfg["spatial"]["width"])
        if cfg["noise"]["prediction_type"] != "epsilon":
            raise ValueError("Spatial photo v1 trains epsilon MSE only")
        self.noise_scheduler = DDPMScheduler(**cfg["noise"], clip_sample=False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.image_encoder.eval()
        self.ae_model.eval()
        return self

    def get_extra_state(self):
        return self.contract

    def set_extra_state(self, state):
        if state != self.contract:
            raise ValueError("Spatial checkpoint architecture/config differs; cannot resume")

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        if not strict:
            raise ValueError("Spatial diffusion requires strict=True; old denoisers are incompatible")
        return super().load_state_dict(state_dict, strict=True, assign=assign)

    def extract_condition(self, images: Tensor) -> Tensor:
        if images.ndim != 4 or images.shape[1:] != (3, 224, 224):
            raise ValueError("Expected normalized photo [B,3,224,224]")
        # 冻结特征固定 FP32；可训练 projection/lifting 遵循外层 AMP。
        with torch.no_grad(), torch.autocast(device_type=images.device.type, enabled=False):
            features = self.image_encoder.forward_features(images.float())
            tokens = torch.cat((features["x_norm_clstoken"][:, None], features["x_norm_patchtokens"]), 1)
        return self.condition_encoder(tokens)

    def forward(self, batch: dict, v_test: bool = False) -> dict[str, Tensor]:
        clean = batch["gt_latent"]
        time = torch.randint(self.noise_scheduler.config.num_train_timesteps,
                             (len(clean),), device=clean.device)
        noise = torch.randn_like(clean)
        noisy = self.noise_scheduler.add_noise(clean, noise, time)
        prediction = self.denoiser(noisy, time, self.extract_condition(batch["gt_image"]))
        loss = (prediction.float() - noise.float()).square().mean()
        return {"ddpm_loss": loss, "total_loss": loss}

    @torch.no_grad()
    def sample_latents(self, images: Tensor, *, generator: torch.Generator | None = None,
                       num_steps: int = 1000) -> Tensor:
        """只需照片；memory 一次计算，所有反向时间步复用；默认完整 DDPM1000。"""
        if self.training:
            raise ValueError("Call eval() before sampling")
        if not 1 <= num_steps <= self.noise_scheduler.config.num_train_timesteps:
            raise ValueError("num_steps must be within the training noise schedule")
        memory = self.extract_condition(images)
        latent = torch.randn((len(images), 30, 32), device=images.device, generator=generator)
        # 每次采样使用独立 scheduler，避免覆盖训练对象的 timesteps。
        scheduler = DDPMScheduler.from_config(self.noise_scheduler.config)
        scheduler.set_timesteps(num_steps, device=images.device)
        for timestep in scheduler.timesteps:
            prediction = self.denoiser(latent, timestep.expand(len(images)), memory)
            latent = scheduler.step(prediction.float(), timestep, latent, generator=generator).prev_sample
        return latent

    @torch.no_grad()
    def decode_latents(self, latent: Tensor) -> list[dict]:
        """沿用原 baseline 两级去重及 VAE raw 输出；OCC 后处理仍在模型外。"""
        if self.training or latent.ndim != 3 or latent.shape[1:] != (30, 32):
            raise ValueError("Decode requires eval() and latent [B,30,32]")
        with torch.autocast(device_type=latent.device.type, enabled=False):
            return [decode_candidate(self, item.float()) for item in latent]


def prepare_spatial_model_config(cfg, model_cfg, resume: str | None):
    """Lightning 的严格恢复接口；完整 checkpoint 自带冻结 DINO 与 VAE。"""
    from omegaconf import OmegaConf
    from src.datasets.spatial_photo import spatial_data_provenance

    provenance = spatial_data_provenance(cfg.dataset)
    if cfg.dataset.get("provenance") is not None and dict(cfg.dataset.provenance) != provenance:
        raise ValueError("Spatial data changed after prepare_run")
    OmegaConf.update(cfg.dataset, "provenance", provenance, force_add=True)
    if model_cfg.frozen_source_sha256 != provenance["original_checkpoint_sha256"]:
        raise ValueError("Spatial model and GT cache use different frozen VAE sources")
    if cfg.trainer.init_from_checkpoint:
        raise ValueError("Spatial v1 starts fresh or resumes its own complete checkpoint")
    if cfg.eval.enabled or cfg.trainer.evaluate or cfg.trainer.diffusion_validation_inference:
        raise ValueError("Use sample_latents/decode_latents with an explicit downstream evaluation protocol")
    if resume:
        saved = torch.load(resume, map_location="cpu", mmap=True, weights_only=False)
        previous = saved["hyper_parameters"]
        if (previous["model"]["name"] != "SpatialPhotoDiffusion" or
                architecture_contract(previous["model"]) != architecture_contract(model_cfg) or
                previous["model"]["frozen_source_sha256"] != model_cfg.frozen_source_sha256 or
                previous["dataset"]["provenance"] != provenance):
            raise ValueError("Cannot resume a different spatial architecture or data protocol")
        OmegaConf.update(model_cfg, "legacy_model_config", previous["model"]["legacy_model_config"], force_add=True)
    OmegaConf.update(model_cfg, "restoring_checkpoint", bool(resume), force_add=True)


def load_spatial_photo(checkpoint: str | Path, *, device="cpu", dino_repo=None, source_cache=None):
    """推理严格加载完整新 checkpoint；不需要再次读取原 SVR 权重。"""
    saved = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False)
    cfg = deepcopy(saved["hyper_parameters"]["model"])
    if cfg["name"] != "SpatialPhotoDiffusion":
        raise ValueError("Expected a SpatialPhotoDiffusion checkpoint")
    cfg["restoring_checkpoint"] = True
    if dino_repo is not None:
        cfg["dino_repo"] = str(dino_repo)
    if source_cache is not None:
        cfg["legacy_source_cache"] = str(source_cache)
    state = saved["state_dict"]
    if any(not key.startswith("model.") for key in state):
        raise ValueError("Unexpected state outside the spatial diffusion model")
    model = SpatialPhotoDiffusion(cfg)
    model.load_state_dict({key[len("model."):]: value for key, value in state.items()}, strict=True)
    return model.to(device).eval()
