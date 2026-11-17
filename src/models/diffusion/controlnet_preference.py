"""完整 ControlNet 初始化的 GT + DPO；仅更新控制分支。"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import torch
from torch import Tensor, nn

from src.common.io import sha256_file
from .adaln_zero_preference import check_initialization_data
from .controlnet_photo import ControlNetPhotoDiffusion, model_state
from .preference import deterministic_policy, diffusion_dpo_loss, per_sample_mse


def control_config(checkpoint: dict) -> dict:
    """保存推理结构及原主干来源；恢复时不依赖旧初始化文件。"""
    saved = checkpoint["hyper_parameters"]["model"]
    if saved["name"] != "ControlNetPhotoDiffusion":
        raise ValueError("Initialize DPO from a complete GT ControlNet checkpoint")
    return {key: saved[key] for key in (
        "name", "architecture_version", "backbone_config", "initialization",
    )}


def check_complete_checkpoint(config: dict, checkpoint: dict) -> None:
    saved = checkpoint["hyper_parameters"]["model"]
    fields = ("name", "control_config", "beta_dpo", "preference_weight", "initialization")
    if any(config[key] != saved[key] for key in fields):
        raise ValueError("ControlNet DPO architecture/reference/objective differs")


class ControlNetPreferenceDiffusion(ControlNetPhotoDiffusion):
    """共享冻结 DINO；policy 主干冻结，reference 整体冻结，采样只用 policy。"""

    def __init__(self, cfg: dict):
        nn.Module.__init__(self)
        saved = None
        if cfg.get("restoring_checkpoint", False):
            source_config = cfg["control_config"]
        else:
            saved = torch.load(cfg["init_controlnet_checkpoint"], map_location="cpu", mmap=True, weights_only=False)
            source_config = control_config(saved)
        baseline = ControlNetPhotoDiffusion({
            **source_config, "checkpoint": None, "restoring_checkpoint": True,
            "dino_repo": cfg.get("dino_repo"), "legacy_source_cache": cfg.get("legacy_source_cache"),
        })
        if saved is not None:
            baseline.load_state_dict(model_state(saved), strict=True)
        self.image_encoder = baseline.image_encoder
        self.noise_scheduler = baseline.noise_scheduler
        self.denoiser = baseline.denoiser                       # 已学 zero_out 不重新归零。
        self.reference = deepcopy(self.denoiser).requires_grad_(False).eval()
        self.beta = float(cfg["beta_dpo"])
        self.preference_weight = float(cfg["preference_weight"])
        if self.beta <= 0 or self.preference_weight < 0:
            raise ValueError("beta_dpo must be positive and preference_weight nonnegative")
        if cfg.get("checkpoint") and not cfg.get("resuming_training", False):
            saved = torch.load(cfg["checkpoint"], map_location="cpu", mmap=True, weights_only=False)
            check_complete_checkpoint(cfg, saved)
            self.load_state_dict(model_state(saved), strict=True)

    def train(self, mode: bool = True):
        super().train(mode)
        self.reference.eval()
        return self

    def _noise(self, clean: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        time = torch.randint(self.noise_scheduler.config.num_train_timesteps,
                             (len(clean),), device=clean.device)
        noise = torch.randn_like(clean)
        return self.noise_scheduler.add_noise(clean, noise, time), time, noise

    def forward(self, batch: dict, v_test: bool = False) -> dict[str, Tensor]:
        has_pairs = not v_test and "winner_latent" in batch
        images = torch.cat((batch["gt_image"], batch["pair_image"])) if has_pairs else batch["gt_image"]
        condition, patches = self.image_condition(images)       # 冻结 CLS [*,1,256] / patches [*,256,1024]
        size = len(batch["gt_latent"])
        noisy_gt, gt_time, gt_noise = self._noise(batch["gt_latent"])
        prediction = self.denoiser(noisy_gt, gt_time, condition[:size], patches[:size])
        gt_loss = per_sample_mse(prediction, gt_noise).mean()
        result = {"ddpm_loss": gt_loss, "total_loss": gt_loss}
        if not has_pairs:
            return result

        # 同图 winner/loser 使用同一 t 和 epsilon；不改变生成时的 slot 排列。
        winner, time, noise = self._noise(batch["winner_latent"])
        loser = self.noise_scheduler.add_noise(batch["loser_latent"], noise, time)
        noisy_pairs = torch.cat((winner, loser))                 # [2B,30,32]
        pair_time, targets = time.repeat(2), noise.repeat(2, 1, 1)
        pair_condition = condition[size:].repeat(2, 1, 1)
        pair_patches = patches[size:].repeat(2, 1, 1)
        with deterministic_policy(self.denoiser):
            prediction = self.denoiser(noisy_pairs, pair_time, pair_condition, pair_patches)
            winner_mse, loser_mse = per_sample_mse(prediction, targets).chunk(2)
        with torch.no_grad():
            prediction = self.reference(noisy_pairs, pair_time, pair_condition, pair_patches)
            ref_winner, ref_loser = per_sample_mse(prediction, targets).chunk(2)
        preference_loss, logits = diffusion_dpo_loss(winner_mse, loser_mse, ref_winner, ref_loser, self.beta)
        result.update({
            "preference_loss": preference_loss,
            "preference_logit": logits.detach().mean(),
            "preference_accuracy": (logits.detach() > 0).float().mean(),
            "winner_mse": winner_mse.detach().mean(), "loser_mse": loser_mse.detach().mean(),
            "reference_winner_mse": ref_winner.mean(), "reference_loser_mse": ref_loser.mean(),
            "total_loss": gt_loss + self.preference_weight * preference_loss,
        })
        return result


def prepare_controlnet_preference_config(cfg, model_cfg, resume: str | None) -> None:
    """pair 生成身份与新 reference 分开记录；新 run 不恢复 GT optimizer。"""
    from omegaconf import OmegaConf

    if cfg.trainer.init_from_checkpoint:
        raise ValueError("Use init_controlnet_checkpoint, complete model.checkpoint, or resume_from_checkpoint")
    if cfg.eval.enabled or cfg.trainer.evaluate or cfg.trainer.diffusion_validation_inference:
        raise ValueError("Use sample_latents with the explicit original VAE evaluation pipeline")
    full, initial = model_cfg.get("checkpoint"), model_cfg.get("init_controlnet_checkpoint")
    if full and initial and not resume:
        raise ValueError("Choose ControlNet initialization or a complete DPO checkpoint")
    restore_path = resume or full
    source_path = restore_path or initial
    if not source_path:
        raise ValueError("ControlNet DPO requires an explicit complete checkpoint")
    saved = torch.load(str(source_path), map_location="cpu", mmap=True, weights_only=False)
    previous = saved["hyper_parameters"]
    if restore_path:
        if dict(cfg.dataset.provenance) != previous["dataset"]["provenance"]:
            raise ValueError("Cannot restore ControlNet DPO with changed pair/GT/image provenance")
        source_config = previous["model"]["control_config"]
        initialization = previous["model"]["initialization"]
        for key, value in (("control_config", source_config), ("initialization", initialization)):
            if model_cfg.get(key) is not None and dict(model_cfg[key]) != value:
                raise ValueError(f"Cannot restore ControlNet DPO with changed {key}")
    else:
        source_config = control_config(saved)
        source_data = previous["dataset"]["provenance"]
        current = cfg.dataset.provenance
        check_initialization_data(source_data["gt"], dict(current.preference), cfg.dataset.condition_data_root)
        if source_data["inputs_sha256"] != current.inputs_sha256:
            raise ValueError("ControlNet initialization and DPO photos differ")
        initialization = {
            "checkpoint": str(Path(initial).resolve()), "sha256": sha256_file(Path(initial)),
            "model_name": source_config["name"], "global_step": saved["global_step"],
            "policy_and_reference_same_checkpoint": True, "learned_control_preserved": True,
            "trainable_scope": "control_blocks", "optimizer_restored": False,
        }
    for target in (cfg.model, model_cfg):
        OmegaConf.update(target, "control_config", source_config, force_add=True)
        OmegaConf.update(target, "initialization", initialization, force_add=True)
    if restore_path:
        check_complete_checkpoint(model_cfg, saved)
    OmegaConf.update(model_cfg, "restoring_checkpoint", bool(restore_path), force_add=True)
    OmegaConf.update(model_cfg, "resuming_training", bool(resume), force_add=True)


def load_controlnet_preference(checkpoint: str | Path, device: str | torch.device = "cpu", *,
                              dino_repo: str | None = None,
                              legacy_source_cache: str | None = None) -> ControlNetPreferenceDiffusion:
    """完整 DPO 权重可独立采样；不访问旧的 GT 初始化 checkpoint。"""
    saved = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False)
    cfg = dict(saved["hyper_parameters"]["model"])
    if cfg["name"] != "ControlNetPreferenceDiffusion":
        raise ValueError("Expected a ControlNetPreferenceDiffusion checkpoint")
    cfg.update(checkpoint=str(checkpoint), init_controlnet_checkpoint=None,
               restoring_checkpoint=True, resuming_training=False)
    if dino_repo is not None:
        cfg["dino_repo"] = dino_repo
    if legacy_source_cache is not None:
        cfg["legacy_source_cache"] = legacy_source_cache
    return ControlNetPreferenceDiffusion(cfg).to(device).eval()
