"""Photo DPO 的 AdaLN policy；reference 仍为原 photo baseline。"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from src.common.io import sha256_file
from .adaln import AdaLNSVRDenoiser
from .preference import PreferenceDiffusion
from .preference_geometry import geometry_settings


def check_initialization_data(current: dict, previous: dict) -> None:
    """内容身份必须相同；显式 relocation 只允许改变资产位置及协议身份。"""
    locations = {"pairs_manifest", "validation_pairs_manifest", "generation_protocol",
                 "validation_gt_cache", "generation_protocol_sha256", "generation_fingerprint"}

    def contents(provenance):
        result = {key: value for key, value in provenance.items() if key not in locations}
        result["photo_cache"] = {key: value for key, value in result["photo_cache"].items()
                                 if key != "manifest"}
        return result

    if contents(current) != contents(previous):
        raise ValueError("AdaLN initialization requires the same pairs, GT splits, cache and source weights")
    if current == previous:
        return
    protocol = json.loads(Path(current["generation_protocol"]).read_text())
    relocation = protocol.get("asset_relocation", {})
    if (relocation.get("source_protocol_sha256") != previous["generation_protocol_sha256"] or
            relocation.get("source_protocol_fingerprint") != previous["generation_fingerprint"]):
        raise ValueError("Changed initialization paths require an explicit source protocol relocation record")


def check_adaln_checkpoint(config: dict, checkpoint: dict) -> None:
    """完整恢复也核对不一定改变 tensor shape 的训练／架构语义。"""
    saved = checkpoint["hyper_parameters"]["model"]
    fields = ("name", "modulation_hidden_width", "model_source", "photo_global_step",
              "beta_dpo", "preference_weight", "legacy_model_config")
    if any(config[key] != saved[key] for key in fields):
        raise ValueError("Complete AdaLN checkpoint architecture/source/objective differs")
    if geometry_settings(config.get("geometry")) != geometry_settings(saved.get("geometry")):
        raise ValueError("Complete AdaLN checkpoint geometry objective differs")


class AdaLNPreferenceDiffusion(PreferenceDiffusion):
    """原 GT epsilon MSE + DPO 不变；仅 policy 新增条件调制。"""

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.policy = AdaLNSVRDenoiser(self.policy, cfg["modulation_hidden_width"])
        if cfg.get("checkpoint") and not cfg.get("resuming_training", False):
            saved = torch.load(cfg["checkpoint"], map_location="cpu", mmap=True, weights_only=False)
            check_adaln_checkpoint(cfg, saved)
            prefix = "model."
            if any(not key.startswith(prefix) for key in saved["state_dict"]):
                raise ValueError("Complete AdaLN checkpoint contains state outside model.*")
            self.load_state_dict({key[len(prefix):]: value for key, value in saved["state_dict"].items()},
                                 strict=True)

    def initialize_preference_weights(self, path: str | Path, data_provenance: dict) -> dict:
        """新 run：严格继承旧 DPO policy，验证 photo310k reference，增量零初始化。"""
        path = Path(path).expanduser().resolve()
        saved = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
        config = saved["hyper_parameters"]["model"]
        if (config["name"] != "PreferenceDiffusion" or config.get("model_source") != "photo" or
                config["legacy_model_config"] != self.legacy_model_config or
                geometry_settings(config.get("geometry"))["enabled"]):
            raise ValueError("AdaLN initialization requires the original photo DPO architecture without geometry loss")
        if float(config["beta_dpo"]) != self.beta or float(config["preference_weight"]) != self.preference_weight:
            raise ValueError("AdaLN initialization preference objective differs")
        check_initialization_data(data_provenance, saved["hyper_parameters"]["dataset"]["provenance"])

        # 原参数必须完整匹配；新增参数只能是 policy.modulations.*。
        expected = {f"model.{key}": value for key, value in self.state_dict().items()
                    if not key.startswith("policy.modulations.")}
        weights = saved["state_dict"]
        if set(weights) != set(expected):
            raise ValueError("AdaLN initialization state keys differ from the complete original DPO model")
        for key, current in expected.items():
            value = weights[key]
            if value.shape != current.shape or value.dtype != current.dtype:
                raise ValueError(f"AdaLN initialization shape/dtype differs: {key}")
            if not key.startswith("model.policy.") and not torch.equal(value, current):
                raise ValueError(f"Initialization frozen photo reference/DINO differs: {key}")
        policy = {key[len("model.policy."):]: value for key, value in weights.items()
                  if key.startswith("model.policy.")}
        policy.update({key: value for key, value in self.policy.state_dict().items()
                       if key.startswith("modulations.")})
        self.policy.load_state_dict(policy, strict=True)
        return {"checkpoint": str(path), "sha256": sha256_file(path),
                "source_global_step": saved["global_step"], "source_epoch": saved["epoch"],
                "policy_strict_load": True, "reference_and_dino_equal_photo_baseline": True,
                "optimizer_restored": False,
                "additional_trainable_parameters": sum(p.numel() for p in self.policy.modulations.parameters())}
