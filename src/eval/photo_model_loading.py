"""照片 DPO / 空间模型的严格评价加载；共用原版 FP32 sampler 与 VAE decode。"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from src.models.diffusion.photo_baseline import load_photo_svr
from src.models.diffusion.spatial_model import load_spatial_photo
from tools.vae_compare_post import sha256_file


def apply_photo_preference_state(legacy, state: dict) -> dict:
    """reference 必须等于已加载照片 baseline；只替换 policy 的五个模块。"""
    names = ("p_embed", "time_embed", "net1", "fc_out", "img_fc")
    reference = {f"{name}.{key}": value for name in names
                 for key, value in getattr(legacy, name).state_dict().items()}
    expected = {**{f"model.policy.{key}": value for key, value in reference.items()},
                **{f"model.reference.{key}": value for key, value in reference.items()},
                **{f"model.image_encoder.{key}": value for key, value in legacy.img_model.state_dict().items()}}
    if state.keys() != expected.keys():
        raise ValueError("Photo DPO state must contain the complete policy/reference/frozen DINO only")
    for key, original in expected.items():
        value = state[key]
        if value.shape != original.shape or value.dtype != original.dtype or not torch.isfinite(value).all():
            raise ValueError(f"Photo DPO tensor shape/dtype/value mismatch: {key}")
        if not key.startswith("model.policy.") and not torch.equal(value, original):
            raise ValueError(f"Frozen photo310k reference/DINO changed: {key}")
    for name in names:
        prefix = f"model.policy.{name}."
        getattr(legacy, name).load_state_dict({k[len(prefix):]: v for k, v in state.items()
                                             if k.startswith(prefix)}, strict=True, assign=True)
    return {"reference_equal_photo310k": True, "dino_equal_photo310k": True,
            "original_vae_retained": True, "policy_tensors": len(reference)}


def load_photo_dpo(original, photo, dpo, *, device="cpu", source_cache=None):
    saved = torch.load(dpo, map_location="cpu", mmap=True, weights_only=False)
    hp, model_cfg = saved["hyper_parameters"], saved["hyper_parameters"]["model"]
    data = hp["dataset"]["provenance"]
    if (model_cfg["name"] != "PreferenceDiffusion" or model_cfg.get("model_source") != "photo" or
            model_cfg.get("photo_global_step") != 310000 or saved["global_step"] != 5000 or
            model_cfg.get("geometry", {}).get("enabled", False) or
            data["source_checkpoint_sha256"] != sha256_file(Path(photo)) or
            data["original_checkpoint_sha256"] != sha256_file(Path(original))):
        raise ValueError("Expected photo310k -> DPO5000 with unchanged original AE and no geometry objective")
    legacy, provenance = load_photo_svr(original, photo, source_cache=source_cache)
    if model_cfg["legacy_model_config"] != provenance["model_config"]:
        raise ValueError("Photo DPO legacy architecture differs from its baseline")
    provenance["photo_dpo"] = {**apply_photo_preference_state(legacy, saved["state_dict"]),
                                "global_step": 5000, "checkpoint_sha256": sha256_file(Path(dpo))}
    return legacy.to(device).eval(), provenance


class SpatialSamplingAdapter(nn.Module):
    """只适配调用签名；沿用相同 CPU 初始噪声和 GPU step-noise 随机序列。"""

    num_max_faces = 30
    dim_input = 32

    def __init__(self, spatial):
        super().__init__()
        self.spatial = spatial

    @property
    def ae_model(self):
        return self.spatial.ae_model

    @property
    def noise_scheduler(self):
        return self.spatial.noise_scheduler

    def extract_condition(self, data):
        images = data["conditions"]["imgs"]
        if images.ndim != 5 or images.shape[1:] != (1, 3, 224, 224):
            raise ValueError("Sampling adapter requires one normalized photo per object")
        return self.spatial.extract_condition(images[:, 0])

    def diffuse(self, latent, time, condition):
        return self.spatial.denoiser(latent, time, condition)


def load_comparison_model(method: str, config: dict, gpu: int):
    cache = Path(config["run_dir"]) / "metadata" / f"legacy-gpu{gpu}"
    if method == "photo_dpo5000":
        return load_photo_dpo(config["original_checkpoint"], config["photo_checkpoint"],
                              config[method], device="cuda:0", source_cache=cache)
    if method in {"spatial_best", "spatial_last"}:
        model = load_spatial_photo(config[method], device="cuda:0", source_cache=cache)
        return SpatialSamplingAdapter(model).eval(), {
            "checkpoint_sha256": sha256_file(Path(config[method])),
            "sampling_adapter": "signature only; original CPU initial noise + GPU DDPM1000",
            "architecture_contract": model.contract,
        }
    raise ValueError(f"Unknown new comparison method: {method}")
