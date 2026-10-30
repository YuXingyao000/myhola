"""原 HoLa concat 网络的照片 DDPM baseline；不含偏好或几何辅助目标。"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .legacy import build_legacy_svr, load_legacy_svr
from .preference import LegacySVRDenoiser, per_sample_mse


def apply_photo_state(legacy: nn.Module, state: dict[str, Tensor]) -> None:
    """严格映射训练 wrapper 的全部状态；原 AE 不在映射中，DINO 必须未变。"""
    modules = {f"model.policy.{name}.": getattr(legacy, name)
               for name in ("p_embed", "time_embed", "net1", "fc_out", "img_fc")}
    modules["model.image_encoder."] = legacy.img_model
    expected = {prefix + key: value for prefix, module in modules.items()
                for key, value in module.state_dict().items()}
    if state.keys() != expected.keys():
        raise ValueError("Photo checkpoint state keys differ from the complete original architecture")
    for key, value in state.items():
        reference = expected[key]
        if value.shape != reference.shape or value.dtype != reference.dtype or not torch.isfinite(value).all():
            raise ValueError(f"Photo checkpoint shape/dtype/value mismatch: {key}")
        if key.startswith("model.image_encoder.") and not torch.equal(value, reference):
            raise ValueError("Frozen photo DINO differs from the original SVR DINO")
    for prefix, module in modules.items():
        if prefix.startswith("model.policy."):
            module.load_state_dict({key[len(prefix):]: value for key, value in state.items()
                                    if key.startswith(prefix)}, strict=True)


def load_photo_svr(original_checkpoint, photo_checkpoint, *, device="cpu", source_cache=None,
                   dino_repo=None):
    """照片去噪器 + 严格保留原 AE，供既有 FP32 sampler / decoder 评价。"""
    from pathlib import Path
    from src.brepnet.data.preference_dataset import sha256_file

    saved = torch.load(photo_checkpoint, map_location="cpu", mmap=True, weights_only=False)
    config = saved["hyper_parameters"]
    if config["model"]["name"] != "PhotoBaselineDiffusion":
        raise ValueError("Expected a PhotoBaselineDiffusion checkpoint")
    original_hash = sha256_file(Path(original_checkpoint))
    if config["dataset"]["provenance"]["source_checkpoint_sha256"] != original_hash:
        raise ValueError("Photo training used a different original DINO / AE")
    legacy, provenance = load_legacy_svr(original_checkpoint, source_cache=source_cache, dino_repo=dino_repo)
    if config["model"]["legacy_model_config"] != provenance["model_config"]:
        raise ValueError("Photo and original architecture configurations differ")
    apply_photo_state(legacy, saved["state_dict"])
    provenance["photo"] = {"checkpoint": str(Path(photo_checkpoint).resolve()),
                            "sha256": sha256_file(Path(photo_checkpoint)),
                            "global_step": saved["global_step"], "strict_load": True,
                            "dino_equal_original": True, "original_ae_retained": True}
    return legacy.to(device).eval(), provenance


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
