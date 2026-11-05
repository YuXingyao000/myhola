"""合成照片 AdaLN-Zero：独立 GT epsilon MSE 训练，不持有偏好模型。"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn

from .adaln_zero import AdaLNZeroDenoiser
from .legacy import build_legacy_svr
from .photo_baseline import load_photo_svr
from .preference import LegacySVRDenoiser, per_sample_mse


def check_complete_checkpoint(config: dict, checkpoint: dict) -> None:
    """严格恢复还核对无法由 tensor shape 判断的架构与来源语义。"""
    saved = checkpoint["hyper_parameters"]["model"]
    fields = ("name", "modulation_hidden_width", "photo_global_step", "legacy_model_config")
    if any(config[key] != saved[key] for key in fields):
        raise ValueError("Complete AdaLN-Zero checkpoint architecture/source differs")


class AdaLNZeroPhotoDiffusion(nn.Module):
    """冻结原 DINO；可训练 policy 直接预测 GT 加噪所用 epsilon。"""

    def __init__(self, cfg: dict):
        super().__init__()
        options = {"source_cache": cfg.get("legacy_source_cache"), "dino_repo": cfg.get("dino_repo")}
        restoring = cfg.get("restoring_checkpoint", False)
        if restoring:
            legacy, self.provenance = build_legacy_svr(cfg["legacy_model_config"], **options)
        else:
            legacy, self.provenance = load_photo_svr(
                cfg["legacy_checkpoint"], cfg["init_photo_checkpoint"], **options,
            )
            if self.provenance["photo"]["global_step"] != cfg["photo_global_step"]:
                raise ValueError("Photo initialization step differs from the selected baseline")
        self.legacy_model_config = self.provenance["model_config"]
        self.noise_scheduler = legacy.noise_scheduler
        self.image_encoder = legacy.img_model.requires_grad_(False).eval()
        baseline = LegacySVRDenoiser(legacy)
        # These are the only pretrained tensors deliberately discarded (24 * 4).
        discarded = [f"net1.layers.{index}.norm{norm}.{field}"
                     for index in range(len(baseline.net1.layers))
                     for norm in (1, 2) for field in ("weight", "bias")]
        retained = set(baseline.state_dict()) - set(discarded)
        self.policy = AdaLNZeroDenoiser(baseline, cfg["modulation_hidden_width"]).requires_grad_(True)
        self.initialization = {
            "source_photo": self.provenance.get("photo"), "optimizer_restored": False,
            "discarded_block_ln_tensors": discarded,
            "retained_policy_tensors": sorted(retained),
            "new_policy_tensors": sorted(set(self.policy.state_dict()) - retained),
            "initial_function_preserved": False,
        }
        del legacy
        if cfg.get("checkpoint") and not cfg.get("resuming_training", False):
            saved = torch.load(cfg["checkpoint"], map_location="cpu", mmap=True, weights_only=False)
            check_complete_checkpoint(cfg, saved)
            if any(not key.startswith("model.") for key in saved["state_dict"]):
                raise ValueError("Complete AdaLN-Zero checkpoint contains state outside model.*")
            self.load_state_dict({key[6:]: value for key, value in saved["state_dict"].items()}, strict=True)

    def train(self, mode: bool = True):
        super().train(mode)
        self.image_encoder.eval()
        return self

    def image_features(self, batch: dict) -> Tensor:
        if "image_features" in batch:
            return batch["image_features"].detach()
        with torch.no_grad():
            return self.image_encoder(batch["gt_image"])

    def forward(self, batch: dict, v_test: bool = False) -> dict[str, Tensor]:
        clean = batch["gt_latent"]
        time = torch.randint(self.noise_scheduler.config.num_train_timesteps,
                             (len(clean),), device=clean.device)
        noise = torch.randn_like(clean)
        noisy = self.noise_scheduler.add_noise(clean, noise, time)
        condition = self.policy.image_condition(self.image_features(batch))
        prediction = self.policy(noisy, time, condition)
        loss = per_sample_mse(prediction, noise).mean()
        return {"ddpm_loss": loss, "total_loss": loss}

    @torch.no_grad()
    def sample_latents(self, images: Tensor, *, initial_generator=None, step_generator=None) -> Tensor:
        """原 CPU 初始噪声 + device DDPM 噪声；冻结 VAE 解码在显式评估管线中。"""
        if self.training:
            raise ValueError("Call eval() before sampling")
        condition = self.policy.image_condition(self.image_encoder(images))
        latent = torch.randn((len(images), 30, 32), generator=initial_generator).to(images.device)
        self.noise_scheduler.set_timesteps(1000, device=images.device)
        for time in self.noise_scheduler.timesteps:
            prediction = self.policy(latent, time.expand(len(images)), condition)
            latent = self.noise_scheduler.step(prediction, time, latent, generator=step_generator).prev_sample
        return latent


def prepare_adaln_zero_model_config(cfg, model_cfg, resume: str | None):
    """完整恢复只依赖自身权重；初始化只接受健康的纯 GT photo baseline。"""
    from omegaconf import OmegaConf
    from src.common.io import sha256_file

    if cfg.trainer.init_from_checkpoint:
        raise ValueError("Use init_photo_checkpoint, complete model.checkpoint, or resume_from_checkpoint")
    if cfg.eval.enabled or cfg.trainer.evaluate or cfg.trainer.diffusion_validation_inference:
        raise ValueError("Use sample_latents and the explicit original VAE evaluation pipeline")
    full = model_cfg.get("checkpoint")
    if full and model_cfg.get("init_photo_checkpoint") and not resume:
        raise ValueError("Choose photo initialization or a complete AdaLN-Zero checkpoint")
    restore_path = resume or full
    if restore_path:
        saved = torch.load(str(restore_path), map_location="cpu", mmap=True, weights_only=False)
        previous = saved["hyper_parameters"]
        if dict(cfg.dataset.provenance) != previous["dataset"]["provenance"]:
            raise ValueError("Cannot restore AdaLN-Zero with changed GT data/provenance")
        OmegaConf.update(model_cfg, "legacy_model_config", previous["model"]["legacy_model_config"], force_add=True)
        check_complete_checkpoint(model_cfg, saved)
        OmegaConf.update(cfg.model, "initialization", previous["model"]["initialization"], force_add=True)
    else:
        if not model_cfg.get("init_photo_checkpoint"):
            raise ValueError("AdaLN-Zero requires explicit photo initialization or a complete checkpoint")
        if sha256_file(Path(model_cfg.legacy_checkpoint)) != cfg.dataset.provenance.original_checkpoint_sha256:
            raise ValueError("Photo GT/DINO cache and initialization use different original checkpoints")
        if (sha256_file(Path(model_cfg.init_photo_checkpoint)) != cfg.dataset.provenance.photo_checkpoint_sha256 or
                model_cfg.photo_global_step != cfg.dataset.provenance.photo_global_step):
            raise ValueError("Photo initialization differs from the audited GT source baseline")
    OmegaConf.update(model_cfg, "restoring_checkpoint", bool(restore_path), force_add=True)
    OmegaConf.update(model_cfg, "resuming_training", bool(resume), force_add=True)
