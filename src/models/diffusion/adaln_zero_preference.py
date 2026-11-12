"""完整 GT AdaLN-Zero → 同架构冻结 reference；保留 GT 去噪的离线 DPO。"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import torch
from torch import nn

from .adaln_zero_photo import AdaLNZeroPhotoDiffusion
from .preference import PreferenceDiffusion
from .preference_geometry import geometry_settings


def check_initialization_data(source: dict, pairs: dict, condition_root: str) -> None:
    """共享表示、图像与 split；pair 生成 checkpoint 可以不同于新 reference。"""
    shared = (("photo_cache", "photo_cache"),
              ("original_checkpoint_sha256", "original_checkpoint_sha256"),
              ("train_split_sha256", "train_list_sha256"),
              ("validation_split_sha256", "validation_list_sha256"))
    if (source["version"] != "photo_gt_only_v1" or
            pairs["protocol_version"] != "photo_preference_generation_v1" or
            any(source[left] != pairs[right] for left, right in shared) or
            Path(source["condition_data_root"]).resolve() != Path(condition_root).resolve()):
        raise ValueError("AdaLN initialization and pairs must share DINO/AE, GT cache, images and splits")


def check_complete_checkpoint(config: dict, checkpoint: dict) -> None:
    """完整恢复固定训练目标和 reference 身份，避免同 shape 的静默语义变更。"""
    saved = checkpoint["hyper_parameters"]["model"]
    fields = ("name", "modulation_hidden_width", "legacy_model_config", "beta_dpo",
              "preference_weight", "initialization")
    if any(config[key] != saved[key] for key in fields):
        raise ValueError("Complete AdaLN-Zero DPO checkpoint architecture/reference/objective differs")


class AdaLNZeroPreferenceDiffusion(PreferenceDiffusion):
    """继承相同 GT + DPO 与采样路径；只替换初始化，不改推理网络。"""

    def __init__(self, cfg: dict):
        nn.Module.__init__(self)
        restoring = cfg.get("restoring_checkpoint", False)
        if restoring:
            backbone_cfg = {**cfg, "name": "AdaLNZeroPhotoDiffusion", "checkpoint": None,
                            "restoring_checkpoint": True, "resuming_training": True}
        else:
            saved = torch.load(cfg["init_adaln_checkpoint"], map_location="cpu", mmap=True, weights_only=False)
            backbone_cfg = {**saved["hyper_parameters"]["model"],
                            "checkpoint": cfg["init_adaln_checkpoint"],
                            "restoring_checkpoint": True, "resuming_training": False}
            for key in ("dino_repo", "legacy_source_cache"):
                backbone_cfg[key] = cfg.get(key)
        # 完整恢复已学 gate/scale/shift；构造时的零初始化随后被严格权重加载覆盖。
        backbone = AdaLNZeroPhotoDiffusion(backbone_cfg)
        self.legacy_model_config = backbone.legacy_model_config
        self.provenance = backbone.provenance
        self.noise_scheduler = backbone.noise_scheduler
        self.image_encoder = backbone.image_encoder
        self.policy = backbone.policy
        self.reference = deepcopy(self.policy).requires_grad_(False).eval()
        self.geometry = geometry_settings(None)
        self.beta = float(cfg["beta_dpo"])
        self.preference_weight = float(cfg["preference_weight"])
        if cfg.get("checkpoint") and not cfg.get("resuming_training", False):
            saved = torch.load(cfg["checkpoint"], map_location="cpu", mmap=True, weights_only=False)
            check_complete_checkpoint(cfg, saved)
            if any(not key.startswith("model.") for key in saved["state_dict"]):
                raise ValueError("Complete AdaLN-Zero DPO checkpoint contains state outside model.*")
            self.load_state_dict({key[6:]: value for key, value in saved["state_dict"].items()}, strict=True)


def prepare_adaln_zero_preference_config(cfg, model_cfg, resume: str | None) -> None:
    """pair provenance 原样保留；新 policy/reference 来源单独写入 initialization。"""
    from omegaconf import OmegaConf
    from src.common.io import sha256_file

    if cfg.trainer.init_from_checkpoint:
        raise ValueError("Use init_adaln_checkpoint, complete model.checkpoint, or resume_from_checkpoint")
    if cfg.eval.enabled or cfg.trainer.evaluate or cfg.trainer.diffusion_validation_inference:
        raise ValueError("Use sample_latents and the explicit original VAE evaluation pipeline")
    full, initial = model_cfg.get("checkpoint"), model_cfg.get("init_adaln_checkpoint")
    if full and initial and not resume:
        raise ValueError("Choose AdaLN GT initialization or a complete DPO checkpoint")
    restore_path = resume or full
    if restore_path:
        saved = torch.load(str(restore_path), map_location="cpu", mmap=True, weights_only=False)
        previous = saved["hyper_parameters"]
        if dict(cfg.dataset.provenance) != previous["dataset"]["provenance"]:
            raise ValueError("Cannot restore AdaLN-Zero DPO with changed pair/GT provenance")
        previous_model = previous["model"]
        if model_cfg.get("initialization") is not None and dict(model_cfg.initialization) != previous_model["initialization"]:
            raise ValueError("Cannot restore AdaLN-Zero DPO with changed reference identity")
        legacy_config = previous_model["legacy_model_config"]
        initialization = previous_model["initialization"]
    else:
        if not initial:
            raise ValueError("AdaLN-Zero DPO requires an explicit complete GT AdaLN initialization")
        saved = torch.load(str(initial), map_location="cpu", mmap=True, weights_only=False)
        previous = saved["hyper_parameters"]
        source = previous["model"]
        if (source["name"] != "AdaLNZeroPhotoDiffusion" or
                source["modulation_hidden_width"] != model_cfg.modulation_hidden_width):
            raise ValueError("Initialize from the same complete GT AdaLN-Zero architecture")
        check_initialization_data(previous["dataset"]["provenance"], dict(cfg.dataset.provenance),
                                  cfg.dataset.condition_data_root)
        legacy_config = source["legacy_model_config"]
        initialization = {
            "checkpoint": str(Path(initial).resolve()), "sha256": sha256_file(Path(initial)),
            "model_name": source["name"], "global_step": saved["global_step"],
            "policy_and_reference_same_checkpoint": True, "learned_modulation_preserved": True,
            "optimizer_restored": False,
        }
    for target in (cfg.model, model_cfg):
        OmegaConf.update(target, "legacy_model_config", legacy_config, force_add=True)
        OmegaConf.update(target, "initialization", initialization, force_add=True)
    if restore_path:
        check_complete_checkpoint(model_cfg, saved)
    OmegaConf.update(model_cfg, "restoring_checkpoint", bool(restore_path), force_add=True)
    OmegaConf.update(model_cfg, "resuming_training", bool(resume), force_add=True)
