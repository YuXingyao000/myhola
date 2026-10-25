"""UV VAE 的训练装配、明确权重迁移和双侧几何导出。

新 decoder 的全部网络定义仍在 uv_intersection.py；此文件不增加网络层。
旧 encoder/posterior 与全局/面 decoder 逐 tensor 保留，可显式迁移同构曲线 attention。
双侧轨迹尚不是唯一拓扑边，dual_uv_surface_v1 禁止进入旧 OCC 路径。
"""
from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import asdict
import hashlib
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
import torch
from torch import Tensor

from .base import BaseVAE
from .encoders import FacePosterior, HolisticEncoder
from .legacy import AutoEncoder_light
from .uv_intersection import UVDecoderConfig, UVDecoderLoss, UVIntersectionDecoder


ENCODER_OPTIONS = (
    "in_channels", "latent_channels", "hidden_channels", "norm", "gaussian_weights",
    "sigmoid", "num_gat_layers", "bottleneck_dim", "num_encoder_layers", "nhead",
    "primitive_blocks_per_stage", "kl_reduction",
)
RETAINED_SOURCE_PREFIXES = (
    "encoder.", "posterior.", "decoder.face_features.", "decoder.face_geometry.",
)
REMOVED_SOURCE_PREFIXES = (
    "decoder.inter.", "decoder.classifier.", "decoder.edge_geometry.",
    "decoder.refiner.", "decoder.shared_curve.", "decoder.uv_boundary.",
)


def _encoder_options(cfg: Mapping) -> dict:
    values = {**AutoEncoder_light.DEFAULTS, **dict(cfg)}
    options = {name: values[name] for name in ENCODER_OPTIONS}
    options["face_encoder_kind"] = (cfg.get("face_encoder") or {}).get("kind", "cnn")
    options["encoder_edge_value_enabled"] = bool((cfg.get("encoder_edge_value") or {}).get("enabled", False))
    return options


def _decoder_options(cfg: Mapping) -> UVDecoderConfig:
    options = {"latent_dim": 4 * int(cfg["latent_channels"]), **dict(cfg.get("uv_intersection") or {})}
    if options["latent_dim"] != 4 * int(cfg["latent_channels"]):
        raise ValueError("UV decoder latent_dim differs from the retained posterior.")
    return UVDecoderConfig(**options)


def _read_checkpoint(path: str | Path) -> tuple[dict, dict[str, Tensor], str]:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    checkpoint = torch.load(str(path), map_location="cpu", weights_only=False, mmap=True)
    weights = {key.removeprefix("model."): value for key, value in checkpoint["state_dict"].items()}
    if len(weights) != len(checkpoint["state_dict"]):
        raise ValueError("Ambiguous checkpoint keys after removing model prefix.")
    return checkpoint, weights, digest.hexdigest()


def _checkpoint_config(checkpoint: Mapping) -> dict:
    return OmegaConf.to_container(OmegaConf.create(checkpoint["hyper_parameters"]["model"]), resolve=True)


def _check_retained_semantics(cfg: Mapping, source_cfg: Mapping) -> None:
    expected, recorded = _encoder_options(cfg), _encoder_options(source_cfg)
    differences = {key: (recorded[key], value) for key, value in expected.items() if recorded[key] != value}
    if differences:
        raise ValueError(f"Retained encoder/posterior configuration differs: {differences}")
    if int(recorded["latent_channels"]) * 4 != 64:
        raise ValueError("This migration requires the trained 64-D source posterior.")
    if recorded["face_encoder_kind"] != "cnn" or recorded["encoder_edge_value_enabled"]:
        raise ValueError("Source encoder differs from the retained CNN/GAT encoder.")
    decoder = _decoder_options(cfg)
    source = {**AutoEncoder_light.DEFAULTS, **source_cfg}
    same = {
        "global_width": (source["bottleneck_dim"], decoder.face_width),
        "global_layers": (source["num_decoder_layers"], decoder.face_layers),
        "global_heads": (source["nhead"], decoder.face_heads),
        "global_ffn": (2048, decoder.face_ffn),
        "global_dropout": (0.1, decoder.dropout),
        "face_width": (source["hidden_channels"], decoder.face_width),
        "face_blocks_per_stage": (source["point_decoder_blocks_per_stage"], 1),
        "face_bbox_blocks": (source["face_bbox_decoder_blocks"], 7),
        "face_norm": (source["norm"], "layer"),
        "face_output_channels": (source["in_channels"], 6),
    }
    if mismatched := {key: value for key, value in same.items() if value[0] != value[1]}:
        raise ValueError(f"Retained global/face decoder configuration differs: {mismatched}")
    if cfg.get("init_uv_attention_from_intersection", False):
        if (source_cfg.get("intersection") or {}).get("kind", "legacy") != "legacy":
            raise ValueError("UV attention migration requires the legacy intersection backbone.")
        attention = {
            "width": (source["intersection_dim"], decoder.curve_width),
            "layers": (source["intersection_layers"], decoder.curve_layers),
            "heads": (8, decoder.curve_heads),
            "ffn": (2048, decoder.curve_ffn),
            "dropout": (0.1, decoder.dropout),
        }
        if mismatched := {key: value for key, value in attention.items() if value[0] != value[1]}:
            raise ValueError(f"Retained curve attention configuration differs: {mismatched}")


def _curve_attention_key(name: str) -> str | None:
    """The explicit pre-LN block is the same operator as TransformerDecoderLayer."""
    prefix = "decoder.intersection.curve_attention.blocks."
    if not name.startswith(prefix):
        return None
    layer, component, suffix = name[len(prefix):].split(".", 2)
    old_names = {
        "norm_self": "norm1", "norm_cross": "norm2", "norm_ffn": "norm3",
        "self_attention": "self_attn", "cross_attention": "multihead_attn",
    }
    if component == "ffn":
        index, suffix = suffix.split(".", 1)
        component = {"0": "linear1", "3": "linear2"}[index]
    else:
        component = old_names[component]
    return f"decoder.inter.layers.{layer}.{component}.{suffix}"


def _retained_key(name: str, include_curve_attention: bool = False) -> str | None:
    """Target key -> old source key; no suffix matching or permissive load."""
    if include_curve_attention and (source_key := _curve_attention_key(name)) is not None:
        return source_key
    if name.startswith(("encoder.", "posterior.")):
        return name
    global_names = {"input": "input_projection", "layers": "backbone", "output": "output_projection"}
    prefix = "decoder.global_attention."
    if name.startswith(prefix):
        branch, suffix = name[len(prefix):].split(".", 1)
        return f"decoder.face_features.{global_names[branch]}.{suffix}"
    point_names = {
        "stem": "1", "to_4.upsample": "2", "to_4.refine": "3",
        "to_8.upsample": "4", "to_8.refine": "5", "to_16.upsample": "6",
        "to_16.refine": "7", "output": "8",
    }
    prefix = "decoder.face_decoder.points."
    if name.startswith(prefix):
        relative = name[len(prefix):]
        for branch, index in point_names.items():
            if relative.startswith(branch + "."):
                suffix = relative[len(branch) + 1:].replace("skip.", "downsample.")
                return f"decoder.face_geometry.points_decoder.{index}.{suffix}"
    prefix = "decoder.face_decoder.placement."
    if name.startswith(prefix):
        relative = name[len(prefix):]
        if relative.startswith("input."):
            index, suffix = 0, relative[len("input."):]
        elif relative.startswith("blocks."):
            block, suffix = relative[len("blocks."):].split(".", 1)
            index = int(block) + 1
        else:
            index, suffix = 7, relative[len("output."):]
        suffix = suffix.replace("linear1.", "conv1.").replace("linear2.", "conv2.").replace("skip.", "downsample.")
        return f"decoder.face_geometry.bbox_decoder.{index}.{suffix}"
    return None


def _initialize_retained(model: BaseVAE, cfg: Mapping, path: str) -> dict:
    checkpoint, source, digest = _read_checkpoint(path)
    expected_digest = cfg.get("init_uv_decoder_sha256")
    if expected_digest and digest != expected_digest:
        raise ValueError("UV decoder initialization checkpoint SHA256 differs from the configured source.")
    source_cfg = _checkpoint_config(checkpoint)
    if source_cfg.get("name") != "AutoEncoder_intersection":
        raise ValueError("UV decoder migration requires an AutoEncoder_intersection checkpoint.")
    _check_retained_semantics(cfg, source_cfg)
    include_attention = bool(cfg.get("init_uv_attention_from_intersection", False))
    initialized, mapping = model.state_dict(), {}
    for target, fresh in initialized.items():
        source_key = _retained_key(target, include_attention)
        if source_key is None:
            continue
        value = source[source_key]
        if value.shape != fresh.shape or value.dtype != fresh.dtype:
            raise ValueError(f"Retained checkpoint tensor differs: {source_key} -> {target}.")
        initialized[target] = value
        mapping[target] = source_key
    consumed = set(mapping.values())
    retained_prefixes = RETAINED_SOURCE_PREFIXES + (("decoder.inter.layers.",) if include_attention else ())
    retained_source = {key for key in source if key.startswith(retained_prefixes)}
    if consumed != retained_source:
        raise ValueError(f"Incomplete retained checkpoint mapping: missing={sorted(retained_source-consumed)}, unexpected={sorted(consumed-retained_source)}")
    ignored = sorted(source.keys() - consumed)
    if unexpected := [key for key in ignored if not key.startswith(REMOVED_SOURCE_PREFIXES)]:
        raise ValueError(f"Unknown source tensors cannot be silently discarded: {unexpected}")
    new_keys = sorted(initialized.keys() - mapping.keys())
    if unexpected := [key for key in new_keys if not key.startswith(("decoder.face_tokens.", "decoder.intersection.", "decoder.surface."))]:
        raise ValueError(f"Unknown new tensors cannot be silently initialized: {unexpected}")
    model.load_state_dict(initialized, strict=True)
    actual = model.state_dict()
    if any(not torch.equal(actual[target].cpu(), source[old]) for target, old in mapping.items()):
        raise ValueError("Retained parameters changed during initialization.")
    return {
        "source_checkpoint": str(Path(path).resolve()), "source_sha256": digest,
        "source_global_step": checkpoint.get("global_step"), "source_epoch": checkpoint.get("epoch"),
        "retained_tensor_count": len(mapping), "retained_mapping": mapping,
        "curve_attention_migrated": include_attention,
        "retained_parameter_count": sum(p.numel() for name, p in model.named_parameters() if name in mapping),
        "new_parameter_count": sum(p.numel() for name, p in model.named_parameters() if name not in mapping),
        "new_keys": new_keys, "ignored_source_keys": ignored,
        "deterministic_new_buffers": sorted(key for key in new_keys if key in dict(model.named_buffers())),
        "retained_tensors_bitwise_equal": True,
    }


def _check_complete_configuration(cfg: Mapping, checkpoint: Mapping) -> None:
    source_cfg = _checkpoint_config(checkpoint)
    if source_cfg.get("name") != "AutoEncoder_uv_intersection":
        raise ValueError("Complete load requires an AutoEncoder_uv_intersection checkpoint.")
    if _encoder_options(cfg) != _encoder_options(source_cfg) or asdict(_decoder_options(cfg)) != asdict(_decoder_options(source_cfg)):
        raise ValueError("Complete UV model configuration differs from the checkpoint.")


def _load_complete(model: BaseVAE, cfg: Mapping, path: str) -> dict:
    checkpoint, weights, digest = _read_checkpoint(path)
    _check_complete_configuration(cfg, checkpoint)
    model.load_state_dict(weights, strict=True)
    return {"complete_checkpoint": str(Path(path).resolve()), "source_sha256": digest, "strict_load": True}


def _training_targets(batch: Mapping) -> dict[str, Tensor]:
    pairs, curves, offset = [], [], 0
    for count, boundary in zip(batch["face_counts"], batch["boundary_targets"]):
        pairs.append(boundary["face_pairs"] + offset)
        curves.append(boundary["curves"])
        offset = offset + count
    return {
        "face_local": batch["face_norm"], "face_center_scale": batch["face_bbox"],
        "face_xyz": batch["face_points"][..., :3],
        "edge_pairs": torch.cat(pairs), "edge_xyz": torch.cat(curves),
    }


def _numpy(values: Mapping[str, Tensor]) -> dict:
    return {name: value.detach().cpu().numpy() for name, value in values.items()}


class AutoEncoder_uv_intersection(BaseVAE):
    """同源 encoder 与 UV decoder；freeze_encoder 明确选择全量训练或冻结对照。"""

    def __init__(self, cfg: Mapping):
        super().__init__()
        self._configuration = OmegaConf.to_container(OmegaConf.create(cfg), resolve=True)
        self.freeze_encoder = bool(cfg.get("freeze_encoder", True))
        self.runtime_mode = cfg.get("mode", "train")
        self.edge_threshold = float(cfg.get("edge_threshold", 0.5))
        source_path, complete_path = cfg.get("init_uv_decoder_checkpoint"), cfg.get("checkpoint")
        restoring = bool(cfg.get("restoring_checkpoint", False))
        if sum(bool(value) for value in (source_path, complete_path, restoring)) > 1:
            raise ValueError("UV component initialization, complete checkpoint and resume are mutually exclusive.")
        if (self.freeze_encoder or self.runtime_mode == "frozen_inference") and not any((source_path, complete_path, restoring)):
            raise ValueError("Refusing to freeze a random encoder: initialize from the trained 64-D checkpoint.")
        if self.runtime_mode not in {"train", "frozen_inference"}:
            raise ValueError(f"Unknown runtime mode {self.runtime_mode!r}.")
        options = _encoder_options(cfg)
        if options["face_encoder_kind"] != "cnn" or options["encoder_edge_value_enabled"]:
            raise ValueError("UV training adapter retains the CNN/GAT encoder, without optional encoder replacements.")
        # The donor is temporary: only encoder/posterior children become model state.
        donor = AutoEncoder_light({**cfg, "checkpoint": None, "mode": "train"})
        self.face_feature_dim = donor.face_feature_dim
        self.encoder = HolisticEncoder(donor)
        self.posterior = FacePosterior(donor)
        del donor
        self.decoder = UVIntersectionDecoder(_decoder_options(cfg))
        weights = cfg.get("loss_weights") or {}
        self.criterion = UVDecoderLoss(
            classification_weight=float(weights.get("classification_weight", 0.1)),
            incidence_weight=float(weights.get("incidence_weight", 0.1)),
        )
        self.initialization_report = {}
        if source_path:
            self.initialization_report = _initialize_retained(self, cfg, source_path)
        elif complete_path:
            self.initialization_report = _load_complete(self, cfg, complete_path)
        if self.freeze_encoder:
            self.encoder.requires_grad_(False)
            self.posterior.requires_grad_(False)
        if self.runtime_mode == "frozen_inference":
            self.requires_grad_(False)
        self.train(self.runtime_mode == "train")

    def validate_complete_checkpoint(self, checkpoint: Mapping) -> None:
        """Called by Lightning resume before accepting same-shape but different semantics."""
        _check_complete_configuration(self._configuration, checkpoint)

    def train(self, mode: bool = True):
        super().train(mode and self.runtime_mode != "frozen_inference")
        if self.freeze_encoder:
            self.encoder.eval()
            self.posterior.eval()
        return self

    def encode(self, batch: Mapping, v_test: bool = False) -> dict[str, Tensor]:
        with torch.no_grad() if self.freeze_encoder else nullcontext():
            return self.encoder(batch)

    def sample(self, face_features: Tensor, v_is_test: bool = False) -> tuple[Tensor, Tensor, Tensor]:
        with torch.no_grad() if self.freeze_encoder else nullcontext():
            return self.posterior(face_features, use_mean=v_is_test)

    def decode(self, latents: Tensor, face_counts: Tensor | None = None) -> dict[str, Tensor]:
        if face_counts is None:
            face_counts = torch.tensor([len(latents)], dtype=torch.long, device=latents.device)
        return self.decoder(latents, face_counts)

    def forward(self, batch: Mapping, v_test: bool = False, v_use_mean: bool = False) -> tuple[dict, dict]:
        features = self.encode(batch)["face_features"]
        latents, kl_loss, statistics = self.sample(features, v_is_test=v_test or v_use_mean or not self.training)
        prediction = self.decode(latents, batch["face_counts"])
        losses = self.criterion(prediction, _training_targets(batch))
        losses["kl_loss"] = kl_loss.detach() if self.freeze_encoder else kl_loss
        losses["weighted_kl_loss"] = kl_loss * 0 if self.freeze_encoder else kl_loss
        losses["total_loss"] = losses.pop("total") + losses["weighted_kl_loss"]
        data = self.export_training_items(prediction, latents, statistics, batch) if v_test else {}
        return losses, data

    def export_items(self, prediction: Mapping, latents: Tensor, face_counts: Tensor) -> list[dict]:
        items, offset = [], 0
        all_pairs = prediction["edge_pairs"]
        for count_value in face_counts:
            count = int(count_value)
            mask = (all_pairs[:, 0] >= offset) & (all_pairs[:, 0] < offset + count)
            local_pairs = all_pairs[mask] - offset
            probability = prediction["edge_logits"][mask].float().sigmoid()
            active = probability > self.edge_threshold
            pairs = local_pairs[active]
            adjacency = torch.zeros((count, count), device=latents.device, dtype=torch.bool)
            adjacency[pairs[:, 0], pairs[:, 1]] = True
            adjacency[pairs[:, 1], pairs[:, 0]] = True
            item = _numpy({
                "face_latents": latents[offset:offset + count].float(),
                "pred_face": prediction["face_xyz"][offset:offset + count].float(),
                "pred_surface_poles": prediction["surface_poles"][offset:offset + count].float(),
                "pred_edge_owner_pairs": pairs, "pred_edge_confidence": probability[active],
                "pred_edge_uv": prediction["edge_uv"][mask][active].float(),
                "pred_edge_xyz_by_face": prediction["edge_xyz"][mask][active].float(),
                "pred_face_adj": adjacency, "all_candidate_pairs": local_pairs,
                "all_candidate_prob": probability,
                "surface_knots_u": self.decoder.surface.basis.knots,
                "surface_knots_v": self.decoder.surface.basis.knots,
            })
            item.update(representation=np.asarray("dual_uv_surface_v1"),
                        surface_degree=np.asarray(self.decoder.surface.basis.degree),
                        surface_periodic=np.asarray([False, False]),
                        surface_grid_axes=np.asarray("v,u"), uv_coordinates=np.asarray("u,v"))
            items.append(item)
            offset += count
        return items

    def export_training_items(self, prediction, latents, statistics, batch) -> dict:
        items = self.export_items(prediction, latents, batch["face_counts"])
        offset = 0
        for item, count_value, boundary in zip(items, batch["face_counts"], batch["boundary_targets"]):
            count = int(count_value)
            item.update(_numpy({
                "cached_latent_stats": statistics[offset:offset + count].float(),
                "gt_face": batch["face_points"][offset:offset + count, ..., :3].float(),
                "gt_edge": boundary["curves"].float(), "gt_edge_owner_pairs": boundary["face_pairs"],
            }))
            offset += count
        return {"items": items}

    def decode_latents(self, face_latents: Tensor) -> dict:
        counts = torch.tensor([len(face_latents)], device=face_latents.device, dtype=torch.long)
        return self.export_items(self.decode(face_latents, counts), face_latents, counts)[0]

    def decode_state(self, latents: Tensor, face_counts: Tensor | None = None) -> dict[str, Tensor]:
        return self.decode(latents, face_counts)
