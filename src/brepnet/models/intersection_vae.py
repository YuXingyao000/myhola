"""HoLa VAE with one geometry-informed intersection refinement pass.

The assembly is deliberately explicit: encoder -> posterior -> decoder -> loss.
The decoder receives no GT edges; supervision is confined to the loss module.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext

import numpy as np
import torch
from torch import Tensor, nn

from src.brepnet.models.holistic_encoder import FacePosterior, HolisticEncoder
from src.brepnet.models.intersection_decoder import IntersectionDecoder
from src.brepnet.models.intersection_losses import IntersectionLoss
from src.brepnet.models.intersection_state import RefinementOutput
from src.brepnet.models.vae import AutoEncoder_light


def validate_encoder_batch(batch: Mapping[str, Tensor]) -> None:
    counts = batch["face_counts"]
    faces, edges = batch["face_points"], batch["edge_points"]
    if counts.dtype != torch.long or counts.ndim != 1 or not len(counts) or (counts <= 0).any():
        raise ValueError("face_counts must contain positive integer object sizes")
    if int(counts.sum()) != len(faces):
        raise ValueError("face_counts does not partition face_points")
    if faces.shape[1:] != (16, 16, 6) or edges.shape[1:] != (16, 6):
        raise ValueError("Expected face_points [F,16,16,6] and edge_points [E,16,6]")
    if not torch.isfinite(faces).all() or not torch.isfinite(edges).all():
        raise ValueError("Encoder geometry contains nonfinite values")
    conn = batch["edge_face_connectivity"]
    if conn.dtype != torch.long or conn.ndim != 2 or conn.shape[1] != 3:
        raise ValueError("edge_face_connectivity must be integer [E,3]")
    if len(conn) != len(edges) or len(conn[:, 0].unique()) != len(edges):
        raise ValueError("GT edge samples require one-to-one directed connectivity")
    if len(conn) and ((conn < 0).any() or (conn[:, 0] >= len(edges)).any() or (conn[:, 1:] >= len(faces)).any()):
        raise ValueError("GT connectivity index outside geometry arrays")
    object_ids = torch.arange(len(counts), device=counts.device).repeat_interleave(counts)
    expected_mask = object_ids[:, None] != object_ids[None, :]
    if batch["attn_mask"].dtype != torch.bool or not torch.equal(batch["attn_mask"], expected_mask):
        raise ValueError("attn_mask must isolate exactly the objects described by face_counts")
    if len(conn) and ((conn[:, 1] == conn[:, 2]).any() or (object_ids[conn[:, 1]] != object_ids[conn[:, 2]]).any()):
        raise ValueError("Seam or cross-object GT connectivity is unsupported")


class AutoEncoder_intersection(nn.Module):
    """Strictly migrated original VAE components plus a new residual refiner."""

    def __init__(self, cfg: dict):
        super().__init__()
        self.freeze_encoder = bool(cfg["freeze_encoder"])
        self.runtime_mode = cfg["mode"]
        self.edge_threshold = float(cfg["edge_threshold"])
        if not 0 < self.edge_threshold < 1:
            raise ValueError("edge_threshold must be between zero and one")
        if self.runtime_mode not in {"train", "frozen_inference"}:
            raise ValueError(f"Unknown mode {self.runtime_mode!r}")
        if cfg["trainable_scope"] != "all":
            raise ValueError("Use freeze_encoder; intersection VAE requires trainable_scope=all")
        if cfg["in_channels"] != 6 or cfg["loss"] != "l1":
            raise ValueError("Intersection refinement currently requires six-channel L1 geometry")
        if float(cfg.get("intersection_noise_std", 0.0)) != 0:
            raise ValueError("This refinement experiment requires intersection_noise_std=0")

        legacy_path, full_path = cfg["init_legacy_checkpoint"], cfg["checkpoint"]
        restoring = bool(cfg.get("restoring_checkpoint", False))
        if legacy_path and full_path:
            raise ValueError("Choose init_legacy_checkpoint or checkpoint, not both")
        if restoring and (legacy_path or full_path):
            raise ValueError("Resume must restore the new checkpoint without a second initialization")
        if (self.freeze_encoder or self.runtime_mode == "frozen_inference") and not (legacy_path or full_path or restoring):
            raise ValueError("Refusing to freeze random weights: provide init_legacy_checkpoint or checkpoint")

        # Construct the original modules once, strictly load their full state,
        # then transfer them to explicit owners. No duplicate legacy model is
        # registered in this model, and all original geometry heads are reused.
        legacy_cfg = {**cfg, "checkpoint": legacy_path, "mode": "train"}
        legacy = AutoEncoder_light(legacy_cfg)
        self.face_feature_dim = legacy.face_feature_dim
        self.encoder = HolisticEncoder(legacy)
        self.posterior = FacePosterior(legacy)
        self.decoder = IntersectionDecoder(legacy, cfg["refinement"])
        self.criterion = IntersectionLoss(cfg["loss_weights"])
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
        validate_encoder_batch(batch)
        with torch.no_grad() if self.freeze_encoder else nullcontext():
            return self.encoder(batch)

    def sample(self, face_features: Tensor, v_is_test: bool = False) -> tuple[Tensor, Tensor, Tensor]:
        with torch.no_grad() if self.freeze_encoder else nullcontext():
            return self.posterior(face_features, use_mean=v_is_test)

    def decode_state(self, latents: Tensor, face_counts: Tensor | None = None) -> RefinementOutput:
        if face_counts is None:
            face_counts = torch.tensor([len(latents)], dtype=torch.long, device=latents.device)
        return self.decoder(latents, face_counts)

    def forward(self, batch: Mapping[str, Tensor], v_test: bool = False) -> tuple[dict, dict]:
        encoding = self.encode(batch, v_test)
        latents, kl_loss, statistics = self.sample(encoding["face_features"], v_is_test=v_test)
        prediction = self.decode_state(latents, batch["face_counts"])
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
            rows = connectivity[(connectivity[:, 1] >= offset) & (connectivity[:, 1] < offset + count)]
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
            item.update({key: value.detach().cpu().numpy() for key, value in arrays.items()})
            offset += count
        return {"items": items}

    def decode_latents(self, face_latents: Tensor) -> dict:
        """Decode a supplied face set; no GT topology or additional face pruning."""
        return self.export_items(self.decode_state(face_latents), face_latents)[0]

    def inference(self, face_latents: Tensor) -> dict:
        return self.decode_latents(face_latents)
