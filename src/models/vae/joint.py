"""Shared-boundary VAE: inherited holistic encoder and a joint decoder."""
from __future__ import annotations

from collections.abc import Mapping
import numpy as np
import torch
from torch import nn, Tensor

from .legacy import AutoEncoder, AutoEncoder_light
from .decoders.joint import JointBoundaryDecoder
from .losses.joint import joint_boundary_loss
from .checkpoints import JointCheckpointLoader


class AutoEncoder_joint(AutoEncoder):
    ENCODER_MODULES = (
        "face_coords", "edge_coords", "graph_face_edge", "face_attn_proj_in",
        "face_attn_proj_out", "face_attn", "global_feature1", "global_feature2", "gaussian_proj",
    )
    DEFAULTS = {
        **AutoEncoder_light.DEFAULTS,
        "freeze_encoder": True,
        "init_encoder_checkpoint": None,
        "joint_hidden_dim": 256,
        "joint_num_queries": 128,
        "joint_num_rounds": 3,
        "joint_num_heads": 8,
        "joint_dropout": 0.0,
        "joint_edge_to_face": True,
        "joint_existence_threshold": 0.5,
        "joint_loss_weights": {"face": 1.0, "curve": 1.0, "existence": 0.1, "ownership": 0.1, "consistency": 0.1},
        "joint_match_weights": {"curve": 1.0, "ownership": 0.25},
        "joint_no_boundary_weight": 0.1,
    }

    def __init__(self, v_conf):
        # Build only the inherited encoder, not a temporary enormous old decoder.
        nn.Module.__init__(self)
        conf = self._with_defaults(v_conf, self.DEFAULTS)
        self.hidden_channels = int(conf["hidden_channels"])
        self.latent_channels = int(conf["latent_channels"])
        self.runtime_mode = conf["mode"]
        self.norm = conf["norm"]
        self.in_channels = int(conf["in_channels"])
        self.face_feature_dim = self.latent_channels * 4
        self.edge_feature_dim = self.face_feature_dim * 2
        self.num_gat_layers = int(conf["num_gat_layers"])
        self.bottleneck_dim = int(conf["bottleneck_dim"])
        self.num_encoder_layers = int(conf["num_encoder_layers"])
        self.nhead = int(conf["nhead"])
        self.sigmoid = bool(conf["sigmoid"])
        self.gaussian_weights = float(conf["gaussian_weights"])
        self.kl_reduction = conf["kl_reduction"]
        self.freeze_encoder = bool(conf["freeze_encoder"])
        primitive_blocks = int(conf["primitive_blocks_per_stage"])
        self.face_coords = self._make_face_encoder(primitive_blocks)
        self.edge_coords = self._make_edge_encoder(primitive_blocks)
        self.graph_face_edge = self._make_graph_layers()
        self.face_attn_proj_in = nn.Linear(self.face_feature_dim, self.bottleneck_dim)
        self.face_attn_proj_out = nn.Linear(self.bottleneck_dim, self.face_feature_dim)
        layer = nn.TransformerEncoderLayer(
            self.bottleneck_dim, self.nhead, dim_feedforward=2048, dropout=0.1,
            batch_first=True, norm_first=True,
        )
        self.face_attn = nn.TransformerEncoder(layer, self.num_encoder_layers, nn.LayerNorm(self.bottleneck_dim))
        self.global_feature1 = nn.Sequential(
            nn.Linear(self.face_feature_dim, self.face_feature_dim), nn.LeakyReLU(),
            nn.Linear(self.face_feature_dim, self.face_feature_dim),
        )
        self.global_feature2 = nn.Sequential(
            nn.Linear(self.face_feature_dim * 2, self.face_feature_dim), nn.LeakyReLU(),
            nn.Linear(self.face_feature_dim, self.face_feature_dim),
        )
        self.gaussian_proj = self._make_gaussian_proj()
        self.decoder_joint = JointBoundaryDecoder(
            self.face_feature_dim,
            hidden_dim=int(conf["joint_hidden_dim"]),
            num_queries=int(conf["joint_num_queries"]),
            num_rounds=int(conf["joint_num_rounds"]),
            num_heads=int(conf["joint_num_heads"]),
            dropout=float(conf["joint_dropout"]),
            edge_to_face=bool(conf["joint_edge_to_face"]),
        )
        self.existence_threshold = float(conf["joint_existence_threshold"])
        self.loss_weights = dict(conf["joint_loss_weights"])
        self.match_weights = dict(conf["joint_match_weights"])
        self.no_boundary_weight = float(conf["joint_no_boundary_weight"])
        checkpoint, encoder_checkpoint = conf["checkpoint"], conf["init_encoder_checkpoint"]
        if checkpoint is not None and encoder_checkpoint is not None:
            raise ValueError("checkpoint (strict new model) and init_encoder_checkpoint (legacy migration) are mutually exclusive")
        if encoder_checkpoint is not None:
            self.init_encoder_checkpoint(encoder_checkpoint)
        if checkpoint is not None:
            self.load_state_dict(self._checkpoint_weights(checkpoint), strict=True)
        if self.freeze_encoder:
            for name in self.ENCODER_MODULES:
                for parameter in getattr(self, name).parameters():
                    parameter.requires_grad_(False)
        self._apply_runtime_mode()
        self.train(self.training)


    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_encoder:
            for name in self.ENCODER_MODULES:
                getattr(self, name).eval()
        return self

    def decode_state(self, face_latents: Tensor, face_counts: Tensor | None = None) -> dict[str, Tensor]:
        """Differentiable, GT-free decoding; counts partition input, never prune queries."""
        if face_counts is None:
            face_counts = torch.tensor([len(face_latents)], dtype=torch.long, device=face_latents.device)
        return self.decoder_joint(face_latents, face_counts)

    def decode(self, v_encoding_result, v_data=None, v_deduplicated=False):
        # Keep a recognizable VAE entry point, but do not consume connectivity.
        if v_deduplicated:
            raise ValueError("Joint decoder does not silently deduplicate face entities; pass a unique face set")
        counts = v_encoding_result.get("face_counts")
        if counts is None and v_data is not None:
            counts = v_data["face_counts"]
        return self.decode_state(v_encoding_result["face_latents"], counts)

    def export_items(self, state: Mapping[str, Tensor], face_latents: Tensor) -> list[dict]:
        """Export prediction-only incidence, retaining raw duplicates and same-face assignments."""
        result = []
        offset = 0
        for batch_index, count_value in enumerate(state["face_counts"]):
            count = int(count_value)
            probability = state["existence_logits"][batch_index].sigmoid()
            active = probability >= self.existence_threshold
            query_ids = active.nonzero(as_tuple=True)[0]
            owners = state["ownership_logits"][batch_index, active, :, :count].argmax(-1)
            edge_ids = torch.arange(len(query_ids), device=owners.device, dtype=torch.long)
            connectivity = torch.cat((edge_ids[:, None], owners), dim=1)
            adjacency = torch.zeros((count, count), device=owners.device, dtype=torch.bool)
            adjacency[owners[:, 0], owners[:, 1]] = True
            adjacency[owners[:, 1], owners[:, 0]] = True
            array = lambda value: value.detach().cpu().numpy()
            result.append({
                "representation": np.asarray("shared_boundary_v1"),
                "face_latents": array(face_latents[offset:offset + count].float()),
                "pred_face": array(state["face_points"][batch_index, :count].float()),
                "pred_edge": array(state["curves"][batch_index, active].float()),
                "pred_edge_face_connectivity": array(connectivity),
                "pred_edge_confidence": array(probability[active].float()),
                "pred_face_adj": array(adjacency),
                "edge_query_ids": array(query_ids),
                "all_edge_existence_prob": array(probability.float()),
                "all_edge_face_assignment_prob": array(state["ownership_logits"][batch_index, :, :, :count].softmax(-1).float()),
            })
            offset += count
        return result

    def decode_latents(self, face_latents: Tensor) -> dict:
        state = self.decode_state(face_latents)
        return self.export_items(state, face_latents)[0]

    def forward(self, v_data, v_test=False):
        encoding = self.encode(v_data, v_test)
        latents, kl_loss, stats = self.sample(encoding["face_features"], v_is_test=v_test)
        state = self.decode_state(latents, v_data["face_counts"])
        loss = joint_boundary_loss(
            state, v_data["boundary_targets"], v_data["face_points"][..., :3],
            self.loss_weights, self.match_weights, self.no_boundary_weight,
            self.existence_threshold, v_data.get("v_prefix"),
        )
        loss["kl_loss"] = kl_loss
        loss["total_loss"] = loss["total_loss"] + kl_loss
        data = {}
        if v_test:
            items = self.export_items(state, latents)
            offset = 0
            for item, count_value, target in zip(items, v_data["face_counts"], v_data["boundary_targets"]):
                count = int(count_value)
                item["cached_latent_stats"] = stats[offset:offset + count].detach().float().cpu().numpy()
                item["gt_face"] = v_data["face_points"][offset:offset + count].detach().float().cpu().numpy()
                item["gt_edge"] = target["curves"].detach().float().cpu().numpy()
                pairs = target["face_pairs"].detach().cpu().numpy()
                item["gt_edge_face_connectivity"] = np.concatenate((np.arange(len(pairs))[:, None], pairs), axis=1)
                offset += count
            data = {"items": items}
        return loss, data


    _checkpoint_weights = staticmethod(JointCheckpointLoader.read_weights)

    def init_encoder_checkpoint(self, path) -> None:
        """Public component-migration entry; the policy is owned by the loader."""
        JointCheckpointLoader(self).init_encoder(path)
