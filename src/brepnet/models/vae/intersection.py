"""Holistic VAE with pair intersections, refinement and optional UV supervision.

Read this class for model assembly and forward. Components, migration policies
and losses live in their corresponding modules within this package.
"""
from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext

import numpy as np
import torch
from torch import Tensor

from .base import BaseVAE
from .legacy import AutoEncoder_light
from .encoders import HolisticEncoder, FacePosterior, UVFaceEncoder
from .decoders.intersection import IntersectionDecoder
from .state import RefinementOutput
from .losses.intersection import IntersectionLoss
from .checkpoints import IntersectionCheckpointLoader


class AutoEncoder_intersection(BaseVAE):
    """Legacy geometry heads with optional refinement and final curve feedback."""

    def __init__(self, cfg: dict):
        super().__init__()
        self.freeze_encoder = bool(cfg["freeze_encoder"])
        self.runtime_mode = cfg["mode"]
        self.edge_threshold = float(cfg["edge_threshold"])
        if self.runtime_mode not in {"train", "frozen_inference"}:
            raise ValueError(f"Unknown mode {self.runtime_mode!r}")

        legacy_path, full_path = cfg["init_legacy_checkpoint"], cfg["checkpoint"]
        refinement_path = cfg.get("init_refinement_checkpoint")
        curve_feedback_path = cfg.get("init_curve_feedback_checkpoint")
        edge_value_path = cfg.get("init_encoder_edge_value_checkpoint")
        face_encoder_path = cfg.get("init_face_encoder_checkpoint")
        uv_boundary_path = cfg.get("init_uv_boundary_checkpoint")
        uv_boundary_cfg = dict(cfg.get("uv_boundary") or {})
        uv_boundary_enabled = bool(uv_boundary_cfg.get("enabled", False))
        face_encoder_cfg = dict(cfg.get("face_encoder") or {})
        face_encoder_kind = face_encoder_cfg.pop("kind", "cnn")
        if face_encoder_kind not in {"cnn", "uv_attention"}:
            raise ValueError(f"Unknown face_encoder.kind {face_encoder_kind!r}")
        uv_encoder_enabled = face_encoder_kind == "uv_attention"
        edge_value_cfg = cfg.get("encoder_edge_value") or {}
        restoring = bool(cfg.get("restoring_checkpoint", False))
        sources = (legacy_path, refinement_path, curve_feedback_path, edge_value_path,
                   face_encoder_path, uv_boundary_path, full_path)
        if sum(bool(path) for path in sources) > 1:
            raise ValueError(
                "Choose only one of init_legacy_checkpoint, init_refinement_checkpoint, "
                "init_curve_feedback_checkpoint, init_encoder_edge_value_checkpoint, "
                "init_face_encoder_checkpoint, init_uv_boundary_checkpoint or checkpoint"
            )
        if restoring and any(sources):
            raise ValueError("Resume must restore the new checkpoint without a second initialization")
        if (self.freeze_encoder or self.runtime_mode == "frozen_inference") and not (any(sources) or restoring):
            raise ValueError(
                "Refusing to freeze random weights: provide init_legacy_checkpoint, "
                "init_refinement_checkpoint, init_curve_feedback_checkpoint, "
                "init_encoder_edge_value_checkpoint, init_face_encoder_checkpoint, "
                "init_uv_boundary_checkpoint or checkpoint"
            )
        intersection_cfg = cfg.get("intersection")
        curve_conditioning_cfg = cfg.get("curve_conditioning")
        curve_point_feedback_cfg = cfg.get("curve_point_feedback")
        conditioning_enabled = bool((curve_conditioning_cfg or {}).get("enabled", False))

        # Reuse the original heads and parameter names; only register their final owners.
        legacy_cfg = {**cfg, "checkpoint": legacy_path, "mode": "train"}
        legacy = AutoEncoder_light(legacy_cfg)
        legacy_face_state = None
        if uv_encoder_enabled:
            if face_encoder_path:
                legacy_face_state = legacy.face_coords.state_dict()
            # Do not shift initialization of the retained decoder/refiner.
            with torch.random.fork_rng(devices=[]):
                legacy.face_coords = UVFaceEncoder(
                    in_channels=legacy.in_channels,
                    out_channels=legacy.face_feature_dim,
                    **face_encoder_cfg,
                )
        self.face_feature_dim = legacy.face_feature_dim
        self.encoder = HolisticEncoder(legacy, edge_value_cfg)
        self.posterior = FacePosterior(legacy)
        self.decoder = IntersectionDecoder(
            legacy, cfg["refinement"], intersection_cfg,
            curve_conditioning_cfg=curve_conditioning_cfg,
            curve_point_feedback_cfg=curve_point_feedback_cfg,
            uv_boundary_cfg=uv_boundary_cfg,
        )
        self.criterion = IntersectionLoss(cfg["loss_weights"], uv_boundary_enabled=uv_boundary_enabled)
        self.initialization_report: dict = {}
        initializer = IntersectionCheckpointLoader(self)
        if refinement_path:
            if conditioning_enabled:
                initializer.init_curve_conditioning(refinement_path)
            else:
                initializer.init_from_refinement(refinement_path, legacy.inter.state_dict())
        if curve_feedback_path:
            initializer.init_curve_feedback(curve_feedback_path)
        if edge_value_path:
            initializer.init_encoder_edge_value(edge_value_path)
        if face_encoder_path:
            initializer.init_face_encoder(face_encoder_path, legacy_face_state)
        if uv_boundary_path:
            initializer.init_uv_boundary(uv_boundary_path)
        if full_path:
            checkpoint = torch.load(str(full_path), map_location="cpu", weights_only=False)
            state = checkpoint.get("state_dict", checkpoint)
            weights = {key.removeprefix("model."): value for key, value in state.items()}
            self.load_state_dict(weights, strict=True)
        if self.freeze_encoder:
            self.encoder.requires_grad_(False)
            self.posterior.requires_grad_(False)
        if self.runtime_mode == "frozen_inference":
            self.requires_grad_(False)
        self.train(self.runtime_mode == "train")


    def train(self, mode: bool = True):
        super().train(mode and self.runtime_mode != "frozen_inference")
        if self.freeze_encoder:
            self.encoder.eval()
            self.posterior.eval()
        return self

    def encode(self, batch: Mapping[str, Tensor], v_test: bool = False) -> dict[str, Tensor]:
        # Keep the public encode/sample contract consumed by LatentProvider.
        with torch.no_grad() if self.freeze_encoder else nullcontext():
            return self.encoder(batch)

    def sample(self, face_features: Tensor, v_is_test: bool = False) -> tuple[Tensor, Tensor, Tensor]:
        with torch.no_grad() if self.freeze_encoder else nullcontext():
            return self.posterior(face_features, use_mean=v_is_test)

    def decode(self, latents: Tensor, face_counts: Tensor | None = None) -> RefinementOutput:
        if face_counts is None:
            face_counts = torch.tensor([len(latents)], dtype=torch.long, device=latents.device)
        return self.decoder(latents, face_counts)

    def forward(self, batch: Mapping[str, Tensor], v_test: bool = False) -> tuple[dict, dict]:
        # Default widths: F packed faces, Egt edge samples, P candidate face pairs.
        encoding = self.encode(batch, v_test)  # face [F,32], edge teacher [Egt,64]
        # z [F,32], scalar KL, cached mean/std [F,64].
        latents, kl_loss, statistics = self.sample(encoding["face_features"], v_is_test=v_test)
        prediction = self.decode(latents, batch["face_counts"])  # geometry + [P] logits
        # Training-only edge reconstruction; decoder predictions never read GT edges.
        teacher_geometry = self.decoder.edge_geometry(encoding["edge_features"])
        losses = self.criterion(
            prediction, batch, encoding["edge_features"], teacher_geometry,
            kl_loss, self.freeze_encoder,
        )
        data = self.export_training_items(prediction, latents, statistics, batch) if v_test else {}
        return losses, data

    def export_items(self, prediction: RefinementOutput, latents: Tensor) -> list[dict]:
        stage, pairs = prediction.refined, prediction.pairs
        items, face_offset = [], 0
        for object_index, count_tensor in enumerate(pairs.face_counts):
            count = int(count_tensor)
            object_mask = pairs.object_ids == object_index
            local_pairs = pairs.indices[object_mask] - face_offset
            probability = stage.edge_logits[object_mask].float().sigmoid()
            active = probability > self.edge_threshold
            chosen_pairs = local_pairs[active]
            edge_ids = torch.arange(len(chosen_pairs), device=local_pairs.device)[:, None]
            adjacency = torch.zeros((count, count), dtype=torch.bool, device=latents.device)
            adjacency_probability = torch.zeros((count, count), device=latents.device)
            adjacency[chosen_pairs[:, 0], chosen_pairs[:, 1]] = True
            adjacency_probability[local_pairs[:, 0], local_pairs[:, 1]] = probability
            face_slice = slice(face_offset, face_offset + count)
            arrays = {
                "face_latents": latents[face_slice].float(),
                "pred_face": stage.face_geometry.points[face_slice].float(),
                "pred_edge": stage.edge_geometry.points[object_mask][active].float(),
                "pred_edge_face_connectivity": torch.cat((edge_ids, chosen_pairs), dim=-1),
                "pred_edge_confidence": probability[active],
                "pred_face_adj": adjacency,
                "pred_face_adj_prob": adjacency_probability,
                "all_candidate_pairs": local_pairs,
                "all_candidate_prob": probability,
                "coarse_candidate_prob": prediction.coarse.edge_logits[object_mask].float().sigmoid(),
            }
            if prediction.edge_uv is not None:
                arrays["pred_edge_uv"] = prediction.edge_uv[object_mask][active].float()
            item = {key: value.detach().cpu().numpy() for key, value in arrays.items()}
            item["representation"] = np.asarray("legacy_half_edges")
            items.append(item)
            face_offset += count
        return items

    def export_training_items(self, prediction, latents, statistics, batch) -> dict:
        items = self.export_items(prediction, latents)
        offset = 0
        connectivity = batch["edge_face_connectivity"]
        for item, count_tensor in zip(items, batch["face_counts"]):
            count = int(count_tensor)
            row_mask = (connectivity[:, 1] >= offset) & (connectivity[:, 1] < offset + count)
            rows = connectivity[row_mask]
            edge_ids = rows[:, 0]
            local_pairs = rows[:, 1:] - offset
            local_ids = torch.arange(len(rows), device=rows.device)[:, None]
            adjacency = torch.zeros((count, count), device=rows.device, dtype=torch.bool)
            adjacency[local_pairs[:, 0], local_pairs[:, 1]] = True
            arrays = {
                "cached_latent_stats": statistics[offset:offset + count].float(),
                "gt_face": batch["face_points"][offset:offset + count, ..., :3].float(),
                "gt_edge": batch["edge_points"][edge_ids, ..., :3].float(),
                "gt_edge_face_connectivity": torch.cat((local_ids, local_pairs), dim=-1),
                "gt_face_adj": adjacency,
            }
            if prediction.edge_uv is not None and "edge_uv_targets" in batch:
                arrays["gt_edge_uv"] = batch["edge_uv_targets"]["uv"][row_mask].float()
                arrays["gt_edge_uv_valid"] = batch["edge_uv_targets"]["valid"][row_mask]
            item.update({key: value.detach().cpu().numpy() for key, value in arrays.items()})
            offset += count
        return {"items": items}

    def decode_latents(self, face_latents: Tensor) -> dict:
        """Decode a supplied face set; no GT topology or additional face pruning."""
        return self.export_items(self.decode_state(face_latents), face_latents)[0]


    def decode_state(self, latents: Tensor, face_counts: Tensor | None = None) -> RefinementOutput:
        """Compatibility name for differentiable, prediction-only decoding."""
        return self.decode(latents, face_counts)
