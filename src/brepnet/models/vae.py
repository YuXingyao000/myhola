"""Variational AutoEncoder models for BRep generation.

This module contains all VAE architectures used in the HoLa-BRep pipeline.
Each model follows the same high-level structure:

    encode()     -- Produce per-face and per-edge feature vectors via:
                    CNN (surface/curve grids) -> Transformer self-attention
                    -> Graph message-passing (GATv2Conv) -> global aggregation.

    sample()     -- Reparameterization trick: z = mean + std * eps, returning
                    the sampled latent, KL divergence loss, and (mean, std).

    decode()     -- Reconstruct geometry from latent codes:
                    Transformer self-attention -> CNN transpose-conv decoders
                    for face/edge local coordinates + bounding-box decoders,
                    plus intersection prediction for edge-face connectivity.

    inference()  -- Decode from a latent vector *without* ground-truth graph
                    structure: predicts face adjacency, deduplicates faces,
                    and reconstructs all edge/face points and connectivity.

Exported classes:
    - AutoEncoder                        (main/full VAE)
    - AutoEncoder_light                  (lighter variant, fewer layers)
    - AutoEncoder_light_exp              (light experimental variant)
    - AutoEncoder_intersection           (retained intersection + refinement)
    - AutoEncoder_joint                  (retained shared-boundary experiment)
    - AutoEncoder_intersection           (one-round geometric refinement)
    - AutoEncoder_joint                  (shared-boundary decoder)
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
import copy
import logging
import math
import time

import numpy as np
from scipy.optimize import linear_sum_assignment

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from einops import rearrange
from einops.layers.torch import Rearrange

from torch_geometric.nn import GATv2Conv
from torch_scatter import scatter_mean

from src.brepnet.dataset import denormalize_coord1112
from src.brepnet.models.blocks import (
    ResBlockXD,
    AttnIntersection,
    EdgeValueGATv2Conv,
    UVFaceEncoder,
    DecoderStage,
    IntersectionDecoder,
    JointBoundaryDecoder,
    PrimitiveGeometry,
    RefinementOutput,
    closest_polyline_points,
    closest_surface_points,
)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def add_timer(time_statics, v_attr, timer):
    if v_attr not in time_statics:
        time_statics[v_attr] = 0.
    time_statics[v_attr] += time.time() - timer
    return time.time()


def profile_time(time_dict, key, v_timer):
    torch.cuda.synchronize()
    cur = time.time()
    time_dict[key] += cur - v_timer
    return cur


# ---------------------------------------------------------------------------
# AutoEncoder - Main / Full VAE
# ---------------------------------------------------------------------------


class AutoEncoder(nn.Module):
    DEFAULTS = {
        "hidden_channels": 768,
        "latent_channels": 8,
        "with_intersection": True,
        "sigmoid": False,
        "gaussian_weights": 0.0,
        "loss": "l1",
        "num_gat_layers": 10,
        "bottleneck_dim": 768,
        "num_encoder_layers": 12,
        "num_decoder_layers": 12,
        "nhead": 16,
        "intersection_dim": 768,
        "intersection_layers": 12,
        "intersection_noise_std": 0.0,
        "use_intersection_noise": True,
        "primitive_blocks_per_stage": 2,
        "point_decoder_blocks_per_stage": 2,
        "face_bbox_decoder_blocks": 7,
        "edge_bbox_decoder_blocks": 7,
        "kl_reduction": "sum",
        "checkpoint": None,
        "mode": "train",
    }

    @staticmethod
    def _with_defaults(v_conf, defaults):
        conf = copy.deepcopy(defaults)
        conf.update(dict(v_conf))
        return conf

    def __init__(self, v_conf):
        super().__init__()
        v_conf = self._with_defaults(v_conf, self.DEFAULTS)

        self.hidden_channels = int(v_conf["hidden_channels"])
        self.latent_channels = int(v_conf["latent_channels"])
        self.runtime_mode = v_conf["mode"]
        self.norm = v_conf["norm"]
        self.in_channels = v_conf["in_channels"]
        self.with_intersection = v_conf["with_intersection"]
        self.face_feature_dim = self.latent_channels * 2 * 2
        self.edge_feature_dim = self.face_feature_dim * 2

        self.num_gat_layers = int(v_conf["num_gat_layers"])
        self.bottleneck_dim = int(v_conf["bottleneck_dim"])
        self.num_encoder_layers = int(v_conf["num_encoder_layers"])
        self.num_decoder_layers = int(v_conf["num_decoder_layers"])
        self.nhead = int(v_conf["nhead"])
        self.intersection_dim = int(v_conf["intersection_dim"])
        self.intersection_layers = int(v_conf["intersection_layers"])
        self.intersection_noise_std = float(v_conf["intersection_noise_std"])
        self.use_intersection_noise = bool(v_conf["use_intersection_noise"])
        self.kl_reduction = v_conf["kl_reduction"]

        primitive_blocks = int(v_conf["primitive_blocks_per_stage"])
        point_decoder_blocks = int(v_conf["point_decoder_blocks_per_stage"])
        self.face_coords = self._make_face_encoder(primitive_blocks)
        self.edge_coords = self._make_edge_encoder(primitive_blocks)
        self.graph_face_edge = self._make_graph_layers()
        self._make_face_fusion_layers()

        self.global_feature1 = nn.Sequential(
            nn.Linear(self.face_feature_dim, self.face_feature_dim),
            nn.LeakyReLU(),
            nn.Linear(self.face_feature_dim, self.face_feature_dim),
        )
        self.global_feature2 = nn.Sequential(
            nn.Linear(self.face_feature_dim * 2, self.face_feature_dim),
            nn.LeakyReLU(),
            nn.Linear(self.face_feature_dim, self.face_feature_dim),
        )

        self.inter = AttnIntersection(
            self.face_feature_dim, self.intersection_dim, self.intersection_layers
        )
        self.classifier = nn.Linear(self.edge_feature_dim, 1)

        self.face_points_decoder = self._make_face_points_decoder(point_decoder_blocks)
        self.edge_points_decoder = self._make_edge_points_decoder(point_decoder_blocks)
        self.face_center_scale_decoder = self._make_bbox_decoder(
            self.face_feature_dim, int(v_conf["face_bbox_decoder_blocks"])
        )
        self.edge_center_scale_decoder = self._make_bbox_decoder(
            self.edge_feature_dim, int(v_conf["edge_bbox_decoder_blocks"])
        )

        self.sigmoid = v_conf["sigmoid"]
        self.gaussian_weights = v_conf["gaussian_weights"]
        self.gaussian_proj = self._make_gaussian_proj()

        self.times = {
            "Encoder": 0,
            "Fuser": 0,
            "Sample": 0,
            "global": 0,
            "Decoder": 0,
            "Intersection": 0,
            "Loss": 0,
        }

        self.loss_fn = nn.L1Loss() if v_conf["loss"] == "l1" else nn.MSELoss()
        self._load_checkpoint(v_conf["checkpoint"])
        self._apply_runtime_mode()

    def _load_checkpoint(self, checkpoint_path):
        if checkpoint_path is None:
            return
        checkpoint = torch.load(str(checkpoint_path), weights_only=False, map_location="cpu")
        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        weights = {
            key[len("model."):] if key.startswith("model.") else key: value
            for key, value in state_dict.items()
        }
        self.load_state_dict(weights)

    def _apply_runtime_mode(self):
        if self.runtime_mode == "train":
            return
        if self.runtime_mode != "frozen_inference":
            raise ValueError(
                f"Unknown VAE mode '{self.runtime_mode}'. Expected 'train' or 'frozen_inference'."
            )
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.runtime_mode == "frozen_inference":
            super().train(False)
        return self

    def _append_encoder_stage(
        self, layers, dim, in_channels, out_channels, spatial_shape, block_count
    ):
        cur_channels = in_channels
        for _ in range(max(block_count - 1, 0)):
            layers.append(
                ResBlockXD(
                    dim, cur_channels, cur_channels, 3, 1, 1,
                    v_norm=self.norm,
                    v_norm_shape=(cur_channels, *spatial_shape),
                )
            )
        layers.append(
            ResBlockXD(
                dim, cur_channels, out_channels, 3, 1, 1,
                v_norm=self.norm,
                v_norm_shape=(out_channels, *spatial_shape),
            )
        )

    def _append_decoder_stage(
        self, layers, dim, in_channels, out_channels, spatial_shape, block_count
    ):
        layers.append(
            ResBlockXD(
                dim, in_channels, out_channels, 3, 1, 1,
                v_norm=self.norm,
                v_norm_shape=(out_channels, *spatial_shape),
            )
        )
        for _ in range(max(block_count - 1, 0)):
            layers.append(
                ResBlockXD(
                    dim, out_channels, out_channels, 3, 1, 1,
                    v_norm=self.norm,
                    v_norm_shape=(out_channels, *spatial_shape),
                )
            )

    def _make_face_encoder(self, block_count):
        hidden_channels = self.hidden_channels
        latent_channels = self.latent_channels
        layers = [
            nn.Conv2d(
                self.in_channels,
                hidden_channels // 8,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.LayerNorm((hidden_channels // 8, 16, 16)),
            nn.LeakyReLU(),
        ]
        stages = [
            (hidden_channels // 8, hidden_channels // 4, (16, 16)),
            (hidden_channels // 4, hidden_channels // 2, (8, 8)),
            (hidden_channels // 2, hidden_channels, (4, 4)),
            (hidden_channels, hidden_channels, (2, 2)),
        ]
        for idx, (in_channels, out_channels, spatial_shape) in enumerate(stages):
            self._append_encoder_stage(
                layers, 2, in_channels, out_channels, spatial_shape, block_count
            )
            if idx < len(stages) - 1:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        layers.extend([
            nn.Conv2d(hidden_channels, latent_channels, kernel_size=1, stride=1, padding=0),
            Rearrange("b n h w -> b (n h w)"),
        ])
        return nn.Sequential(*layers)

    def _make_edge_encoder(self, block_count):
        hidden_channels = self.hidden_channels
        layers = [
            nn.Conv1d(
                self.in_channels,
                hidden_channels // 8,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.LayerNorm((hidden_channels // 8, 16)),
            nn.LeakyReLU(),
        ]
        stages = [
            (hidden_channels // 8, hidden_channels // 4, (16,)),
            (hidden_channels // 4, hidden_channels // 2, (8,)),
            (hidden_channels // 2, hidden_channels, (4,)),
            (hidden_channels, hidden_channels, (2,)),
        ]
        for idx, (in_channels, out_channels, spatial_shape) in enumerate(stages):
            self._append_encoder_stage(
                layers, 1, in_channels, out_channels, spatial_shape, block_count
            )
            if idx < len(stages) - 1:
                layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
        layers.extend([
            nn.Conv1d(
                hidden_channels,
                self.face_feature_dim,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            Rearrange("b n w -> b (n w)"),
        ])
        return nn.Sequential(*layers)

    def _make_graph_layers(self):
        layers = nn.ModuleList()
        for _ in range(self.num_gat_layers):
            layers.append(
                GATv2Conv(
                    self.face_feature_dim,
                    self.face_feature_dim,
                    heads=1,
                    edge_dim=self.edge_feature_dim,
                )
            )
            layers.append(nn.LeakyReLU())
        return layers

    def _make_face_fusion_layers(self):
        bd = self.bottleneck_dim
        self.face_attn_proj_in = nn.Linear(self.face_feature_dim, bd)
        self.face_attn_proj_out = nn.Linear(bd, self.face_feature_dim)
        layer = nn.TransformerEncoderLayer(
            bd, self.nhead, dim_feedforward=2048, dropout=0.1,
            batch_first=True, norm_first=True,
        )
        self.face_attn = nn.TransformerEncoder(
            layer, self.num_encoder_layers, nn.LayerNorm(bd)
        )

        self.face_attn_proj_in2 = nn.Linear(self.face_feature_dim, bd)
        self.face_attn_proj_out2 = nn.Linear(bd, self.face_feature_dim)
        layer2 = nn.TransformerEncoderLayer(
            bd, self.nhead, dim_feedforward=2048, dropout=0.1,
            batch_first=True, norm_first=True,
        )
        self.face_attn2 = nn.TransformerEncoder(layer2, self.num_decoder_layers)

    def _make_face_points_decoder(self, block_count):
        hidden_channels = self.hidden_channels
        layers = [Rearrange("b (n h w) -> b n h w", h=2, w=2)]
        self._append_decoder_stage(
            layers, 2, self.latent_channels, hidden_channels, (2, 2), block_count
        )
        layers.append(
            nn.ConvTranspose2d(hidden_channels, hidden_channels // 2, kernel_size=2, stride=2)
        )
        self._append_decoder_stage(
            layers, 2, hidden_channels // 2, hidden_channels // 2, (4, 4), block_count
        )
        layers.append(
            nn.ConvTranspose2d(
                hidden_channels // 2, hidden_channels // 4, kernel_size=2, stride=2
            )
        )
        self._append_decoder_stage(
            layers, 2, hidden_channels // 4, hidden_channels // 4, (8, 8), block_count
        )
        layers.append(
            nn.ConvTranspose2d(
                hidden_channels // 4, hidden_channels // 8, kernel_size=2, stride=2
            )
        )
        self._append_decoder_stage(
            layers, 2, hidden_channels // 8, hidden_channels // 8, (16, 16), block_count
        )
        layers.extend([
            nn.Conv2d(
                hidden_channels // 8,
                self.in_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            Rearrange("... c w h -> ... w h c", c=self.in_channels),
        ])
        return nn.Sequential(*layers)

    def _make_edge_points_decoder(self, block_count):
        hidden_channels = self.hidden_channels
        layers = [Rearrange("b (n w)-> b n w", n=self.face_feature_dim, w=2)]
        self._append_decoder_stage(
            layers, 1, self.face_feature_dim, hidden_channels, (2,), block_count
        )
        layers.append(
            nn.ConvTranspose1d(hidden_channels, hidden_channels // 2, kernel_size=2, stride=2)
        )
        self._append_decoder_stage(
            layers, 1, hidden_channels // 2, hidden_channels // 2, (4,), block_count
        )
        layers.append(
            nn.ConvTranspose1d(
                hidden_channels // 2, hidden_channels // 4, kernel_size=2, stride=2
            )
        )
        self._append_decoder_stage(
            layers, 1, hidden_channels // 4, hidden_channels // 4, (8,), block_count
        )
        layers.append(
            nn.ConvTranspose1d(
                hidden_channels // 4, hidden_channels // 8, kernel_size=2, stride=2
            )
        )
        self._append_decoder_stage(
            layers, 1, hidden_channels // 8, hidden_channels // 8, (16,), block_count
        )
        layers.extend([
            nn.Conv1d(
                hidden_channels // 8,
                self.in_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            Rearrange("... c w -> ... w c", c=self.in_channels),
        ])
        return nn.Sequential(*layers)

    def _make_bbox_decoder(self, in_dim, block_count):
        hidden_channels = self.hidden_channels
        layers = [
            ResBlockXD(
                0,
                in_dim,
                hidden_channels,
                v_norm=self.norm,
                v_norm_shape=(hidden_channels,),
            ),
        ]
        for _ in range(max(block_count - 1, 0)):
            layers.append(
                ResBlockXD(
                    0,
                    hidden_channels,
                    hidden_channels,
                    v_norm=self.norm,
                    v_norm_shape=(hidden_channels,),
                )
            )
        layers.append(nn.Linear(hidden_channels, 4))
        return nn.Sequential(*layers)

    def _make_gaussian_proj(self):
        if self.gaussian_weights > 0:
            return nn.Sequential(
                nn.Linear(self.face_feature_dim, self.edge_feature_dim),
                nn.LeakyReLU(),
                nn.Linear(self.edge_feature_dim, self.edge_feature_dim),
            )
        return nn.Sequential(
            nn.Linear(self.face_feature_dim, self.face_feature_dim),
            nn.LeakyReLU(),
            nn.Linear(self.face_feature_dim, self.face_feature_dim),
            nn.Identity() if not self.sigmoid else nn.Sigmoid(),
        )

    def _add_intersection_training_noise(self, feature_pair: Tensor) -> Tensor:
        if (
            self.use_intersection_noise
            and self.training
            and self.intersection_noise_std > 0
        ):
            return feature_pair + torch.randn_like(feature_pair) * self.intersection_noise_std
        return feature_pair

    def sample(self, v_fused_face_features, v_is_test=False):
        if self.gaussian_weights <= 0:
            fused_face_features = self.gaussian_proj(v_fused_face_features)
            kl_loss = torch.zeros_like(v_fused_face_features[0, 0])
            return fused_face_features, kl_loss, fused_face_features

        fused_face_features_gau = self.gaussian_proj(v_fused_face_features)
        fused_face_features_gau = fused_face_features_gau.reshape(
            -1, self.face_feature_dim, 2
        )
        mean = fused_face_features_gau[:, :, 0]
        logvar = fused_face_features_gau[:, :, 1]

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        fused_face_features = eps.mul(std).add_(mean)
        kl_terms = 1 + logvar - mean.pow(2) - logvar.exp()
        if self.kl_reduction == "mean":
            kl_loss = -0.5 * torch.mean(kl_terms)
        else:
            kl_loss = -0.5 * torch.sum(kl_terms)
        kl_loss = kl_loss * self.gaussian_weights
        if v_is_test:
            fused_face_features = mean
        return fused_face_features, kl_loss, torch.cat([mean, std], dim=1)

    def profile_time(self, timer, key):
        torch.cuda.synchronize()
        self.times[key] += time.time() - timer
        timer = time.time()
        return timer

    def encode(self, v_data, v_test):
        face_points = rearrange(v_data["face_points"][..., :self.in_channels], 'b h w n -> b n h w').contiguous()
        edge_points = rearrange(v_data["edge_points"][..., :self.in_channels], 'b h n -> b n h').contiguous()
        face_features = self.face_coords(face_points)
        edge_features = self.edge_coords(edge_points)

        # Face attn
        attn_x = self.face_attn_proj_in(face_features)
        attn_x = self.face_attn(attn_x, v_data["attn_mask"])
        attn_x = self.face_attn_proj_out(attn_x)
        fused_face_features = face_features + attn_x

        # # Face graph
        edge_face_connectivity = v_data["edge_face_connectivity"]
        x = fused_face_features
        edge_index = edge_face_connectivity[:, 1:].permute(1, 0)
        edge_attr = edge_features[edge_face_connectivity[:, 0]]
        for layer in self.graph_face_edge:
            if isinstance(layer, GATv2Conv):
                x = layer(x, edge_index, edge_attr) + x
            else:
                x = layer(x)
        fused_face_features = x + fused_face_features

        # Global
        bs = v_data["face_counts"].shape[0]
        index = torch.arange(bs, device=fused_face_features.device).repeat_interleave(v_data["face_counts"])
        face_latents = fused_face_features
        gf = scatter_mean(fused_face_features, index, dim=0)
        gf = self.global_feature1(gf)
        gf = gf.repeat_interleave(v_data["face_counts"], dim=0)
        face_latents = torch.cat((fused_face_features, gf), dim=1)
        face_latents = self.global_feature2(face_latents) + fused_face_features

        return {
            "face_features": face_latents,
            "edge_features": edge_features,
        }

    def decode(self, v_encoding_result, v_data=None, v_deduplicated=False):
        face_latents = v_encoding_result["face_latents"]
        face_feature = self.face_attn_proj_in2(face_latents)
        if v_data is None:
            num_faces = face_latents.shape[0]
            attn_mask = torch.zeros((num_faces, num_faces), dtype=bool, device=face_latents.device)
        else:
            attn_mask = v_data["attn_mask"]
        face_feature = self.face_attn2(face_feature, attn_mask)
        face_latents = self.face_attn_proj_out2(face_feature)

        decoding_results = {}
        decoding_results["face_points_local"] = self.face_points_decoder(face_latents)
        decoding_results["face_center_scale"] = self.face_center_scale_decoder(face_latents)

        if v_deduplicated: # Deduplicate
            face_points_local = decoding_results["face_points_local"]
            face_center_scale = decoding_results["face_center_scale"]
            pred_face_points = denormalize_coord1112(face_points_local, face_center_scale)
            num_faces = face_latents.shape[0]

            deduplicate_face_id = []
            for i in range(num_faces):
                is_duplicate = False
                for j in deduplicate_face_id:
                    if torch.sqrt(((pred_face_points[i]-pred_face_points[j])**2).sum(dim=-1)).mean() < 1e-3:
                        is_duplicate=True
                        break
                if not is_duplicate:
                    deduplicate_face_id.append(i)
            face_latents = face_latents[deduplicate_face_id]

        if v_data is None:
            num_faces = face_latents.shape[0]
            device = face_latents.device
            indexes = torch.stack(torch.meshgrid(torch.arange(num_faces), torch.arange(num_faces), indexing="ij"), dim=2)

            indexes = indexes.reshape(-1,2).to(device)
            feature_pair = face_latents[indexes]

            feature_pair = self.inter(feature_pair)
            pred = self.classifier(feature_pair)[...,0]
            pred_labels = torch.sigmoid(pred) > 0.5

            intersected_edge_feature = feature_pair[pred_labels]
            decoding_results["pred_face_adj"] = pred_labels.reshape(-1)
            decoding_results["pred_edge_face_connectivity"] = torch.cat((torch.arange(intersected_edge_feature.shape[0], device=device)[:,None], indexes[pred_labels]), dim=1)
        else:
            edge_face_connectivity = v_data["edge_face_connectivity"]
            v_zero_positions = v_data["zero_positions"]

            true_intersection_embedding = face_latents[edge_face_connectivity[:, 1:]]
            false_intersection_embedding = face_latents[v_zero_positions]
            id_false_start = true_intersection_embedding.shape[0]
            feature_pair = torch.cat((true_intersection_embedding, false_intersection_embedding), dim=0)

            feature_pair = self._add_intersection_training_noise(feature_pair)
            feature_pair = self.inter(feature_pair)
            pred = self.classifier(feature_pair)

            gt_labels = torch.ones_like(pred)
            gt_labels[id_false_start:] = 0
            loss_edge = F.binary_cross_entropy_with_logits(pred, gt_labels)

            intersected_edge_feature = feature_pair[:id_false_start]

            decoding_results["loss_edge_feature"] = self.loss_fn(
                intersected_edge_feature,
                v_encoding_result["edge_features"][edge_face_connectivity[:, 0]].detach(),
            )

            decoding_results["loss_edge"] = loss_edge

        decoding_results["edge_points_local"] = self.edge_points_decoder(intersected_edge_feature)
        decoding_results["edge_center_scale"] = self.edge_center_scale_decoder(intersected_edge_feature)
        if "edge_features" in v_encoding_result:
            decoding_results["edge_points_local1"] = self.edge_points_decoder(v_encoding_result["edge_features"])
            decoding_results["edge_center_scale1"] = self.edge_center_scale_decoder(v_encoding_result["edge_features"])

        decoding_results["face_latents"] = v_encoding_result["face_latents"]
        return decoding_results

    def loss(self, v_decoding_result, v_data):
        # Loss
        loss={}
        loss["face_norm"] = self.loss_fn(
            v_decoding_result["face_points_local"],
            v_data["face_norm"]
        )
        loss["face_bbox"] = self.loss_fn(
            v_decoding_result["face_center_scale"],
            v_data["face_bbox"]
        )

        loss["edge_norm1"] = self.loss_fn(
            v_decoding_result["edge_points_local1"],
            v_data["edge_norm"]
        )
        loss["edge_bbox1"] = self.loss_fn(
            v_decoding_result["edge_center_scale1"],
            v_data["edge_bbox"]
        )
        loss["edge_feature"] = v_decoding_result["loss_edge_feature"]
        loss["edge_classification"] = v_decoding_result["loss_edge"] * 0.1
        edge_face_connectivity = v_data["edge_face_connectivity"]
        loss["edge_norm"] = self.loss_fn(
            v_decoding_result["edge_points_local"],
            v_data["edge_norm"][edge_face_connectivity[:, 0]]
        )
        loss["edge_bbox"] = self.loss_fn(
            v_decoding_result["edge_center_scale"],
            v_data["edge_bbox"][edge_face_connectivity[:, 0]]
        )
        if self.gaussian_weights > 0:
            loss["kl_loss"] = v_decoding_result["kl_loss"]
        return loss

    def forward(self, v_data, v_test=False):
        encoding_result = self.encode(v_data, v_test)
        face_latents, kl_loss, cached_latent_stats = self.sample(encoding_result["face_features"], v_is_test=v_test)
        encoding_result["face_latents"] = face_latents
        decoding_result = self.decode(encoding_result, v_data)
        decoding_result["kl_loss"] = kl_loss
        loss = self.loss(decoding_result, v_data)
        loss["total_loss"] = sum(loss.values())
        data = {}
        if v_test:
            pred_data = self.decode(encoding_result)

            num_faces = v_data["face_points"].shape[0]
            face_adj = torch.zeros((num_faces, num_faces), dtype=bool, device=loss["total_loss"].device)
            conn = v_data["edge_face_connectivity"]
            face_adj[conn[:, 1], conn[:, 2]] = True
            data["cached_latent_stats"] = cached_latent_stats.cpu().numpy()
            data["gt_face_adj"] = face_adj.reshape(-1)
            data["pred_face_adj"] = pred_data["pred_face_adj"].reshape(-1)
            data["gt_edge"] = v_data["edge_points"].detach().cpu().numpy()
            data["gt_edge_face_connectivity"] = v_data["edge_face_connectivity"].detach().cpu().numpy()
            pred_edge = denormalize_coord1112(pred_data["edge_points_local"], pred_data["edge_center_scale"])
            data["pred_edge"] = pred_edge.detach().cpu().numpy()
            data["pred_edge_face_connectivity"] = pred_data["pred_edge_face_connectivity"].detach().cpu().numpy()
            loss["edge_coords"] = nn.functional.l1_loss(
                denormalize_coord1112(decoding_result["edge_points_local"], decoding_result["edge_center_scale"])[..., :3],
                v_data["edge_points"][v_data["edge_face_connectivity"][:, 0]][..., :3]
            )

            loss["edge_coords1"] = nn.functional.l1_loss(
                denormalize_coord1112(decoding_result["edge_points_local1"], decoding_result["edge_center_scale1"])[..., :3],
                v_data["edge_points"][..., :3]
            )
            data["gt_face"] = v_data["face_points"].detach().cpu().numpy()
            pred_face = denormalize_coord1112(pred_data["face_points_local"], pred_data["face_center_scale"])
            data["pred_face"] = pred_face.detach().cpu().numpy()
            loss["face_coords"] = nn.functional.l1_loss(
                pred_face[..., :3],
                v_data["face_points"][..., :3]
            )

        return loss, data

    def inference(self, v_face_features):
        return self.decode_latents(v_face_features)

    def decode_latents(self, face_latents):
        pred_data = self.decode({"face_latents": face_latents}, v_deduplicated=True)
        return {
            "face_latents": face_latents.to(torch.float32).cpu().numpy(),
            "pred_face_adj": pred_data["pred_face_adj"],
            "pred_face_adj_prob": pred_data["pred_edge_face_connectivity"].reshape(-1).to(torch.float32).cpu().numpy(),
            "pred_edge_face_connectivity": pred_data["pred_edge_face_connectivity"].to(torch.float32).cpu().numpy(),
            "pred_face": denormalize_coord1112(pred_data["face_points_local"], pred_data["face_center_scale"])[...,:3].to(torch.float32).cpu().numpy(),
            "pred_edge": denormalize_coord1112(pred_data["edge_points_local"], pred_data["edge_center_scale"])[...,:3].to(torch.float32).cpu().numpy(),
        }

# ---------------------------------------------------------------------------
# AutoEncoder_light - Lighter variant (fewer layers)
# ---------------------------------------------------------------------------


class AutoEncoder_light(AutoEncoder):
    DEFAULTS = {
        **AutoEncoder.DEFAULTS,
        "num_gat_layers": 5,
        "num_encoder_layers": 8,
        "num_decoder_layers": 8,
        "intersection_dim": 512,
        "intersection_layers": 8,
        "primitive_blocks_per_stage": 1,
        "point_decoder_blocks_per_stage": 1,
    }


# ---------------------------------------------------------------------------
# AutoEncoder_light_exp - Light experimental variant
# ---------------------------------------------------------------------------


class AutoEncoder_light_exp(AutoEncoder_light):
    DEFAULTS = {
        **AutoEncoder_light.DEFAULTS,
        "face_bbox_decoder_blocks": 8,
        "kl_reduction": "mean",
        "use_intersection_noise": False,
    }



# ---------------------------------------------------------------------------
# Shared holistic encoder / posterior (legacy weights retained)
# ---------------------------------------------------------------------------


class HolisticEncoder(nn.Module):
    def __init__(self, legacy: nn.Module, edge_value_cfg: Mapping | None = None):
        super().__init__()
        self.in_channels = legacy.in_channels
        self.face_coords = legacy.face_coords
        self.edge_coords = legacy.edge_coords
        self.face_attn_proj_in = legacy.face_attn_proj_in
        self.face_attn = legacy.face_attn
        self.face_attn_proj_out = legacy.face_attn_proj_out
        self.graph_face_edge = legacy.graph_face_edge
        self.global_feature1 = legacy.global_feature1
        self.global_feature2 = legacy.global_feature2
        # Added 2026-10-15: retain every GAT score/value parameter; only the
        # direct edge-content message is new. No change when disabled.
        if (edge_value_cfg or {}).get("enabled", False):
            hidden_dim = int(edge_value_cfg.get("hidden_dim", 64))
            if hidden_dim <= 0:
                raise ValueError("encoder_edge_value.hidden_dim must be positive")
            # Model construction must not shift subsequent dropout/sampling RNG.
            with torch.random.fork_rng(devices=[]):
                for index, layer in enumerate(self.graph_face_edge):
                    if not isinstance(layer, GATv2Conv):
                        continue
                    replacement = EdgeValueGATv2Conv(
                        layer.in_channels, layer.out_channels, heads=layer.heads,
                        edge_dim=layer.edge_dim, value_hidden_dim=hidden_dim,
                        concat=layer.concat, negative_slope=layer.negative_slope,
                        dropout=layer.dropout, add_self_loops=layer.add_self_loops,
                        fill_value=layer.fill_value, bias=layer.bias is not None,
                        share_weights=layer.share_weights,
                        residual=layer.res is not None,
                    )
                    fresh = {key: value for key, value in replacement.state_dict().items()
                             if key.startswith("value_net.")}
                    replacement.load_state_dict({**layer.state_dict(), **fresh}, strict=True)
                    self.graph_face_edge[index] = replacement

    def fuse_graph(self, faces: Tensor, edges: Tensor, connectivity: Tensor) -> Tensor:
        edge_index = connectivity[:, 1:].T
        edge_attributes = edges[connectivity[:, 0]]
        hidden = faces
        for layer in self.graph_face_edge:
            if isinstance(layer, GATv2Conv):
                hidden = hidden + layer(hidden, edge_index, edge_attributes)
            else:
                hidden = layer(hidden)
        return faces + hidden

    def fuse_global(self, faces: Tensor, face_counts: Tensor) -> Tensor:
        object_ids = torch.arange(len(face_counts), device=faces.device)
        object_ids = object_ids.repeat_interleave(face_counts)
        pooled = scatter_mean(faces, object_ids, dim=0)
        context = self.global_feature1(pooled).repeat_interleave(face_counts, dim=0)
        return faces + self.global_feature2(torch.cat((faces, context), dim=-1))

    def forward(self, batch: Mapping[str, Tensor]) -> dict[str, Tensor]:
        face_points = batch["face_points"][..., :self.in_channels].permute(0, 3, 1, 2).contiguous()
        edge_points = batch["edge_points"][..., :self.in_channels].permute(0, 2, 1).contiguous()
        faces = self.face_coords(face_points)
        # Empty edges are legitimate for a single-face input. Avoid asking CNN
        # backends to infer a zero-length batch's reshape dimensions.
        edges = self.edge_coords(edge_points) if len(edge_points) else faces.new_empty((0, faces.shape[-1] * 2))

        hidden = self.face_attn_proj_in(faces)
        hidden = self.face_attn(hidden, batch["attn_mask"])
        faces = faces + self.face_attn_proj_out(hidden)
        faces = self.fuse_graph(faces, edges, batch["edge_face_connectivity"])
        faces = self.fuse_global(faces, batch["face_counts"])
        return {"face_features": faces, "edge_features": edges}


class FacePosterior(nn.Module):
    def __init__(self, legacy: nn.Module):
        super().__init__()
        self.projection = legacy.gaussian_proj
        self.latent_dim = legacy.face_feature_dim
        self.kl_weight = legacy.gaussian_weights
        self.kl_reduction = legacy.kl_reduction

    def forward(self, face_features: Tensor, use_mean: bool = False) -> tuple[Tensor, Tensor, Tensor]:
        projected = self.projection(face_features)
        if self.kl_weight <= 0:
            return projected, projected.new_zeros(()), projected
        statistics = projected.reshape(-1, self.latent_dim, 2)
        mean, log_variance = statistics[..., 0], statistics[..., 1]
        std = (0.5 * log_variance).exp()
        latents = mean if use_mean else mean + std * torch.randn_like(std)
        terms = 1 + log_variance - mean.square() - log_variance.exp()
        kl = -0.5 * (terms.mean() if self.kl_reduction == "mean" else terms.sum())
        return latents, kl * self.kl_weight, torch.cat((mean, std), dim=-1)


# ---------------------------------------------------------------------------
# Intersection refinement supervision
# ---------------------------------------------------------------------------


@dataclass
class DirectedTargets:
    """GT rows mapped onto the independently constructed candidate set."""

    candidate_rows: Tensor
    edge_ids: Tensor
    reciprocal_rows: Tensor
    reciprocal_edge_ids: Tensor
    labels: Tensor


def _require_integer(value: Tensor, name: str) -> None:
    if value.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"{name} must be an integer tensor")


def _require_shape(value: Tensor, expected: tuple[int, ...], name: str) -> None:
    if tuple(value.shape) != tuple(expected):
        raise ValueError(f"{name} must have shape {tuple(expected)}, got {tuple(value.shape)}")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains nonfinite values")


def _mean_l1(prediction: Tensor, target: Tensor) -> Tensor:
    if prediction.shape != target.shape:
        raise ValueError(f"Reconstruction shape mismatch: {prediction.shape} vs {target.shape}")
    if not prediction.numel():
        return prediction.sum() * 0.0
    return F.l1_loss(prediction.float(), target.float())


def balanced_binary_cross_entropy(logits: Tensor, labels: Tensor) -> Tensor:
    """Give each present class equal weight, including empty-candidate batches."""
    if logits.shape != labels.shape:
        raise ValueError("Classification logits and labels must have identical shapes")
    if not logits.numel():
        return logits.sum() * 0.0
    losses = F.binary_cross_entropy_with_logits(logits.float(), labels.float(), reduction="none")
    positive = labels.bool()
    class_losses = []
    if positive.any():
        class_losses.append(losses[positive].mean())
    if (~positive).any():
        class_losses.append(losses[~positive].mean())
    return torch.stack(class_losses).mean()


def _map_directed_targets(output: RefinementOutput, batch: Mapping) -> DirectedTargets:
    pairs = output.pairs
    device = pairs.indices.device
    counts = batch["face_counts"].to(device=device)
    _require_integer(counts, "face_counts")
    if counts.ndim != 1 or not len(counts) or (counts <= 0).any():
        raise ValueError("face_counts must contain positive counts for a nonempty batch")
    if not torch.equal(counts, pairs.face_counts.to(device=device)):
        raise ValueError("Decoder and target face_counts disagree")
    face_count = int(counts.sum())
    edge_count = len(batch["edge_points"])
    candidate_count = len(pairs.indices)
    _require_integer(pairs.indices, "candidate indices")
    _require_integer(pairs.reverse, "candidate reverse")
    _require_integer(pairs.object_ids, "candidate object_ids")
    _require_shape(pairs.indices, (candidate_count, 2), "candidate indices")
    _require_shape(pairs.reverse, (candidate_count,), "candidate reverse")
    _require_shape(pairs.object_ids, (candidate_count,), "candidate object_ids")
    if candidate_count != int((counts * (counts - 1)).sum()):
        raise ValueError("Decoder candidates must include all ordered within-object face pairs")
    face_objects = torch.repeat_interleave(torch.arange(len(counts), device=device), counts)
    if candidate_count:
        if (pairs.indices < 0).any() or (pairs.indices >= face_count).any():
            raise ValueError("Candidate face index is outside the packed face array")
        if (pairs.indices[:, 0] == pairs.indices[:, 1]).any():
            raise ValueError("Candidate seam/self pairs are unsupported")
        owners = face_objects[pairs.indices]
        if (owners[:, 0] != owners[:, 1]).any() or not torch.equal(owners[:, 0], pairs.object_ids):
            raise ValueError("Candidate pairs cross object boundaries or have wrong object_ids")
        if (pairs.reverse < 0).any() or (pairs.reverse >= candidate_count).any():
            raise ValueError("Candidate reciprocal index is outside the candidate array")
        if not torch.equal(pairs.indices[pairs.reverse], pairs.indices.flip(-1)):
            raise ValueError("Candidate reciprocal mapping is incorrect")

    candidate_keys = pairs.indices[:, 0] * face_count + pairs.indices[:, 1]
    sorted_keys, sorted_rows = candidate_keys.sort()
    if candidate_count > 1 and (sorted_keys[1:] == sorted_keys[:-1]).any():
        raise ValueError("Duplicate directed candidate face pair")

    connectivity = batch["edge_face_connectivity"].to(device=device)
    _require_integer(connectivity, "edge_face_connectivity")
    if connectivity.ndim != 2 or connectivity.shape[1] != 3:
        raise ValueError("edge_face_connectivity must have shape [E,3]")
    edge_ids = connectivity[:, 0].long()
    owner_pairs = connectivity[:, 1:].long()
    if len(connectivity) != edge_count:
        raise ValueError("GT connectivity must cover every edge sample exactly once")
    if len(connectivity):
        if (edge_ids < 0).any() or (edge_ids >= edge_count).any():
            raise ValueError("GT edge index is outside the edge sample array")
        if len(edge_ids.unique()) != edge_count:
            raise ValueError("Duplicate GT edge index or missing edge coverage")
        if (owner_pairs < 0).any() or (owner_pairs >= face_count).any():
            raise ValueError("GT face index is outside the packed face array")
        if (owner_pairs[:, 0] == owner_pairs[:, 1]).any():
            raise ValueError("GT seam/self pairs are unsupported")
        if (face_objects[owner_pairs[:, 0]] != face_objects[owner_pairs[:, 1]]).any():
            raise ValueError("GT connectivity crosses object boundaries")
        target_keys = owner_pairs[:, 0] * face_count + owner_pairs[:, 1]
        if len(target_keys.unique()) != len(target_keys):
            raise ValueError("Duplicate GT directed face pair")
        locations = torch.searchsorted(sorted_keys, target_keys)
        if not candidate_count or (locations >= candidate_count).any():
            raise ValueError("GT directed pair is missing from decoder candidates")
        if not torch.equal(sorted_keys[locations], target_keys):
            raise ValueError("GT directed pair is missing from decoder candidates")
        candidate_rows = sorted_rows[locations]
    else:
        candidate_rows = torch.empty(0, device=device, dtype=torch.long)

    pair_to_edge = torch.full((candidate_count,), -1, device=device, dtype=torch.long)
    pair_to_edge[candidate_rows] = edge_ids
    reciprocal_rows = pairs.reverse[candidate_rows]
    reciprocal_edge_ids = pair_to_edge[reciprocal_rows]
    if (reciprocal_edge_ids < 0).any():
        raise ValueError("Every GT directed edge must have a reciprocal edge")
    labels = output.refined.edge_logits.new_zeros(candidate_count)
    labels[candidate_rows] = 1.0
    return DirectedTargets(candidate_rows, edge_ids, reciprocal_rows, reciprocal_edge_ids, labels)


class IntersectionLoss(nn.Module):
    """Explicit raw losses, weighted training terms, and detached diagnostics."""

    def __init__(self, cfg: dict, uv_boundary_enabled: bool = False):
        super().__init__()
        self.uv_boundary_enabled = uv_boundary_enabled
        self.uv_weight = float(cfg.get("uv_weight", 0.01))
        self.uv_anchor_weight = float(cfg.get("uv_anchor_weight", 0.1))
        self.coarse_weight          = float(cfg.get("coarse_weight", 0.25))
        self.classification_weight  = float(cfg.get("classification_weight", 0.1))
        self.feature_weight         = float(cfg.get("feature_weight", 1.0))
        self.encoder_edge_weight    = float(cfg.get("encoder_edge_weight", 1.0))
        self.xyz_weight             = float(cfg.get("xyz_weight", 1.0))
        self.incidence_weight       = float(cfg.get("incidence_weight", 0.1))
        self.twin_weight            = float(cfg.get("twin_weight", 0.1))
        self.geometry_tolerance     = float(cfg.get("geometry_tolerance", 1e-5))
        self.geometry_chunk_size    = int(cfg.get("geometry_chunk_size", 32))
        for name in (
            "coarse_weight", "classification_weight", "feature_weight", "encoder_edge_weight",
            "xyz_weight", "incidence_weight", "twin_weight", "geometry_tolerance",
            "uv_weight", "uv_anchor_weight",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.geometry_chunk_size <= 0:
            raise ValueError("geometry_chunk_size must be positive")

    def _validate_geometry(
        self,
        output: RefinementOutput,
        batch: Mapping,
        teacher_edge_features: Tensor,
        teacher_geometry: PrimitiveGeometry,
    ) -> None:
        face_count = int(output.pairs.face_counts.sum())
        edge_count = len(batch["edge_points"])
        candidate_count = len(output.pairs.indices)
        if output.coarse.face_features.ndim != 2 or output.coarse.edge_features.ndim != 2:
            raise ValueError("Decoder features must be rank-two tensors")
        face_feature_dim = output.coarse.face_features.shape[-1]
        edge_feature_dim = output.coarse.edge_features.shape[-1]
        face_shape, edge_shape = batch["face_norm"].shape, batch["edge_norm"].shape
        if len(face_shape) != 4 or face_shape[0] != face_count or face_shape[-1] != 6:
            raise ValueError("face_norm must have shape [sum_faces,U,V,6]")
        if len(edge_shape) != 3 or edge_shape[0] != edge_count or edge_shape[-1] != 6:
            raise ValueError("edge_norm must have shape [GT_edges,L,6]")
        if min(face_shape[1:3]) < 2 or edge_shape[1] < 2:
            raise ValueError("Geometry must contain at least 2x2 face samples and 2 edge samples")
        expected = {
            "face_norm": face_shape, "edge_norm": edge_shape,
            "face_bbox": (face_count, 4), "edge_bbox": (edge_count, 4),
            "face_points": face_shape, "edge_points": edge_shape,
        }
        for name, shape in expected.items():
            _require_shape(batch[name], shape, name)
        for name, stage in (("coarse", output.coarse), ("refined", output.refined)):
            _require_shape(stage.face_geometry.local_points, face_shape, f"{name} face local")
            _require_shape(stage.face_geometry.center_scale, (face_count, 4), f"{name} face bbox")
            _require_shape(stage.face_geometry.points, (*face_shape[:-1], 3), f"{name} face XYZ")
            _require_shape(stage.edge_geometry.local_points, (candidate_count, *edge_shape[1:]), f"{name} edge local")
            _require_shape(stage.edge_geometry.center_scale, (candidate_count, 4), f"{name} edge bbox")
            _require_shape(stage.edge_geometry.points, (candidate_count, edge_shape[1], 3), f"{name} edge XYZ")
            _require_shape(stage.face_features, (face_count, face_feature_dim), f"{name} face features")
            _require_shape(stage.edge_features, (candidate_count, edge_feature_dim), f"{name} edge features")
            _require_shape(stage.edge_logits, (candidate_count,), f"{name} edge logits")
        _require_shape(teacher_edge_features, (edge_count, edge_feature_dim), "encoder edge features")
        _require_shape(teacher_geometry.local_points, edge_shape, "encoder edge local")
        _require_shape(teacher_geometry.center_scale, (edge_count, 4), "encoder edge bbox")
        _require_shape(teacher_geometry.points, (*edge_shape[:-1], 3), "encoder edge XYZ")
        _require_shape(output.delta_face, (face_count, face_feature_dim), "delta face")
        _require_shape(output.delta_edge, (candidate_count, edge_feature_dim), "delta edge")

    def _reconstruction(self, stage: DecoderStage, batch: Mapping, targets: DirectedTargets) -> dict[str, Tensor]:
        return {
            "face_local": _mean_l1(stage.face_geometry.local_points, batch["face_norm"]),
            "face_bbox": _mean_l1(stage.face_geometry.center_scale, batch["face_bbox"]),
            "edge_local": _mean_l1(stage.edge_geometry.local_points[targets.candidate_rows], batch["edge_norm"][targets.edge_ids]),
            "edge_bbox": _mean_l1(stage.edge_geometry.center_scale[targets.candidate_rows], batch["edge_bbox"][targets.edge_ids]),
            "classification": balanced_binary_cross_entropy(stage.edge_logits, targets.labels),
        }

    def _incidence(self, output: RefinementOutput, batch: Mapping, targets: DirectedTargets) -> Tensor:
        geometry = output.refined
        if not len(targets.candidate_rows):
            return (geometry.edge_geometry.points.sum() + geometry.face_geometry.points.sum()) * 0.0
        curves = geometry.edge_geometry.points[targets.candidate_rows].float()
        owners = output.pairs.indices[targets.candidate_rows]
        gt_curves = batch["edge_points"][targets.edge_ids, :, :3].float()
        side_losses = []
        for side in range(2):
            _, distances = closest_surface_points(
                curves, geometry.face_geometry.points[owners[:, side]].float(), self.geometry_chunk_size,
            )
            with torch.no_grad():
                _, baseline = closest_surface_points(
                    gt_curves, batch["face_points"][owners[:, side], ..., :3].float(), self.geometry_chunk_size,
                )
            side_losses.append(F.relu(distances - baseline.detach() - self.geometry_tolerance).mean())
        return torch.stack(side_losses).mean()

    def _twin(self, output: RefinementOutput, batch: Mapping, targets: DirectedTargets) -> Tensor:
        curves = output.refined.edge_geometry.points.float()
        if not len(targets.candidate_rows):
            return curves.sum() * 0.0
        gt_curves = batch["edge_points"][targets.edge_ids, :, :3].float()
        gt_twins = batch["edge_points"][targets.reciprocal_edge_ids, :, :3].float()
        with torch.no_grad():
            closest, segments, fractions = closest_polyline_points(gt_curves, gt_twins)
            baseline = (gt_curves - closest).abs()
        twins = curves[targets.reciprocal_rows]
        gather_indices = segments[..., None].expand(-1, -1, 3)
        starts = twins.gather(1, gather_indices)
        ends = twins.gather(1, gather_indices + 1)
        mapped_twins = starts + fractions[..., None] * (ends - starts)
        residuals = (curves[targets.candidate_rows] - mapped_twins).abs()
        # Every directed GT row participates, hence both correspondence directions.
        return F.relu(residuals - baseline.detach() - self.geometry_tolerance).mean()

    def _uv_boundary(
        self, output: RefinementOutput, batch: Mapping, targets: DirectedTargets,
    ) -> dict[str, Tensor]:
        """Anchor geometry at fixed GT UV; predicted UV cannot slide the anchor.

        Sidecars follow connectivity ROW order, while edge_points use edge IDs.
        Bilinear interpolation reads [v,u] grids with normalized (u,v) inputs.
        The GT residual removes the exact same grid interpolation discrepancy.
        """
        labels = batch.get("edge_uv_targets")
        if not isinstance(labels, Mapping) or any(
            key not in labels for key in ("uv", "valid", "residual", "method")
        ):
            raise ValueError("Enabled UV boundary supervision requires edge_uv_targets")
        rows = len(batch["edge_face_connectivity"])
        shape = (rows, 2, 16)
        uv, valid = labels["uv"], labels["valid"]
        if uv.shape != (*shape, 2) or valid.shape != shape or valid.dtype != torch.bool:
            raise ValueError("UV targets must be connectivity-row [R,2,16,2] with boolean [R,2,16] valid")
        if labels["residual"].shape != shape or labels["method"].shape != shape:
            raise ValueError("UV residual and method must have shape [R,2,16]")
        if labels["method"].dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            raise ValueError("UV method must be an integer tensor")
        if not torch.isfinite(uv[valid]).all() or not torch.isfinite(labels["residual"][valid]).all():
            raise ValueError("Valid UV supervision must be finite")
        if (labels["residual"][valid] < 0).any() or ((labels["method"][valid] < 1) | (labels["method"][valid] > 2)).any():
            raise ValueError("Valid UV supervision requires nonnegative residual and method 1 or 2")
        if output.edge_uv is None:
            raise ValueError("Enabled UV supervision requires the auxiliary edge_uv prediction")
        _require_shape(output.edge_uv, (len(output.pairs.indices), 2, 16, 2), "edge_uv")
        prediction = output.edge_uv[targets.candidate_rows].float()
        geometry = output.refined
        curves = geometry.edge_geometry.points[targets.candidate_rows].float()
        surfaces = geometry.face_geometry.points.float()
        coverage = valid.float().mean() if valid.numel() else prediction.detach().sum() * 0
        if not valid.any():
            zero = prediction.sum() * 0 + curves.sum() * 0 + surfaces.sum() * 0
            return {"uv": zero, "uv_anchor": zero, "uv_valid_fraction": coverage.detach()}
        # Invalid positions are sanitized before grid_sample, then excluded.
        coordinates = torch.where(valid[..., None], uv.float(), 0.0).detach()
        owners = output.pairs.indices[targets.candidate_rows]

        def sample(grid: Tensor) -> Tensor:
            values = grid[owners].reshape(rows * 2, *grid.shape[1:]).permute(0, 3, 1, 2)
            locations = (coordinates * 2 - 1).reshape(rows * 2, 1, 16, 2)
            sampled = F.grid_sample(values, locations, mode="bilinear", padding_mode="border", align_corners=True)
            return sampled.squeeze(2).transpose(1, 2).reshape(rows, 2, 16, 3)

        with torch.autocast(device_type=curves.device.type, enabled=False):
            predicted_surface = sample(surfaces)
            with torch.no_grad():
                gt_curves = batch["edge_points"][targets.edge_ids, :, :3].float()
                residual = gt_curves[:, None] - sample(batch["face_points"][..., :3].float())
            discrepancy = curves[:, None] - predicted_surface - residual
            return {
                "uv": (prediction - coordinates).abs()[valid].mean(),
                "uv_anchor": discrepancy.abs()[valid].mean(),
                "uv_valid_fraction": coverage.detach(),
            }

    @staticmethod
    def _diagnostics(output: RefinementOutput, targets: DirectedTargets) -> dict[str, Tensor]:
        result = {}
        with torch.no_grad():
            positives = targets.labels.bool()
            for name, stage in (("coarse", output.coarse), ("refined", output.refined)):
                probabilities = stage.edge_logits.float().sigmoid()
                selected = probabilities > 0.5
                true_positive = (selected & positives).sum().float()
                result[f"{name}_precision"] = true_positive / selected.sum().clamp_min(1)
                result[f"{name}_recall"] = true_positive / positives.sum().clamp_min(1)
                result[f"{name}_positive_probability"] = (
                    probabilities[positives].mean() if positives.any() else probabilities.sum() * 0.0
                )
            result["delta_face_mean"] = output.delta_face.float().abs().mean()
            result["delta_edge_mean"] = (
                output.delta_edge.float().abs().mean() if output.delta_edge.numel() else output.delta_edge.sum() * 0.0
            )
        return result

    def forward(
        self,
        output: RefinementOutput,
        batch: Mapping,
        teacher_edge_features: Tensor,
        teacher_geometry: PrimitiveGeometry,
        kl_loss: Tensor,
        encoder_frozen: bool,
    ) -> dict[str, Tensor]:
        targets = _map_directed_targets(output, batch)
        self._validate_geometry(output, batch, teacher_edge_features, teacher_geometry)
        _require_shape(kl_loss, (), "kl_loss")
        losses, weighted = {}, {}
        for name, stage, stage_weight in (
            ("coarse", output.coarse, self.coarse_weight),
            ("refined", output.refined, 1.0),
        ):
            for term, value in self._reconstruction(stage, batch, targets).items():
                key = f"{name}_{term}"
                weight = stage_weight * (self.classification_weight if term == "classification" else 1.0)
                losses[key] = value
                weighted[f"weighted_{key}"] = weight * value

        losses.update({
            "edge_feature": _mean_l1(output.coarse.edge_features[targets.candidate_rows], teacher_edge_features[targets.edge_ids].detach()),
            "encoder_edge_local": _mean_l1(teacher_geometry.local_points, batch["edge_norm"]),
            "encoder_edge_bbox": _mean_l1(teacher_geometry.center_scale, batch["edge_bbox"]),
            "face_xyz": _mean_l1(output.refined.face_geometry.points, batch["face_points"][..., :3]),
            "edge_xyz": _mean_l1(output.refined.edge_geometry.points[targets.candidate_rows], batch["edge_points"][targets.edge_ids, :, :3]),
            "incidence": self._incidence(output, batch, targets),
            "twin": self._twin(output, batch, targets),
            "kl_loss": kl_loss.detach() if encoder_frozen else kl_loss,
        })
        weights = {
            "edge_feature": self.feature_weight,
            "encoder_edge_local": self.encoder_edge_weight,
            "encoder_edge_bbox": self.encoder_edge_weight,
            "face_xyz": self.xyz_weight,
            "edge_xyz": self.xyz_weight,
            "incidence": self.incidence_weight,
            "twin": self.twin_weight,
            "kl_loss": 0.0 if encoder_frozen else 1.0,
        }
        if self.uv_boundary_enabled:
            losses.update(self._uv_boundary(output, batch, targets))
            weights.update({"uv": self.uv_weight, "uv_anchor": self.uv_anchor_weight})
        weighted.update({f"weighted_{name}": weight * losses[name] for name, weight in weights.items()})
        losses.update(weighted)
        losses["total_loss"] = torch.stack(tuple(weighted.values())).sum()
        losses.update(self._diagnostics(output, targets))
        return losses


# ---------------------------------------------------------------------------
# AutoEncoder_intersection - Original inter with geometric refinement
# ---------------------------------------------------------------------------


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
    """Legacy geometry heads with optional refinement and final curve feedback."""

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
        allowed_face_options = {
            "hidden_dim", "embedding_dim", "num_layers", "nhead",
            "dim_feedforward", "dropout",
        }
        if face_encoder_cfg.keys() - allowed_face_options:
            raise ValueError(f"Unknown face_encoder options: {sorted(face_encoder_cfg.keys() - allowed_face_options)}")
        uv_encoder_enabled = face_encoder_kind == "uv_attention"
        edge_value_cfg = cfg.get("encoder_edge_value") or {}
        edge_value_enabled = bool(edge_value_cfg.get("enabled", False))
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
        curve_feedback_enabled = bool((curve_point_feedback_cfg or {}).get("enabled", False))
        intersection_kind = (intersection_cfg or {}).get("kind", "legacy")
        if uv_boundary_path and not uv_boundary_enabled:
            raise ValueError("init_uv_boundary_checkpoint requires uv_boundary.enabled")
        if uv_boundary_enabled and any((legacy_path, refinement_path, curve_feedback_path,
                                        edge_value_path, face_encoder_path)):
            raise ValueError("UV boundary requires init_uv_boundary_checkpoint, checkpoint or resume")
        if uv_boundary_enabled and (
            uv_encoder_enabled or edge_value_enabled or curve_feedback_enabled
            or intersection_kind != "legacy" or not cfg["refinement"]["enabled"]
            or not conditioning_enabled
        ):
            raise ValueError("UV boundary retains conditioned CNN legacy refinement without other experiment branches")
        # Added 2026-10-17: replace only the local face encoder. Keep the accepted
        # conditioned legacy-inter/refinement decoder and all original losses.
        if face_encoder_path and not uv_encoder_enabled:
            raise ValueError("init_face_encoder_checkpoint requires face_encoder.kind=uv_attention")
        if uv_encoder_enabled and (legacy_path or refinement_path or curve_feedback_path or edge_value_path):
            raise ValueError("UV face encoder requires init_face_encoder_checkpoint, checkpoint or resume")
        if uv_encoder_enabled and (
            intersection_kind != "legacy" or not cfg["refinement"]["enabled"]
            or not conditioning_enabled or curve_feedback_enabled or edge_value_enabled
        ):
            raise ValueError(
                "UV face encoder retains conditioned legacy refinement without point feedback or edge values"
            )
        if uv_encoder_enabled and self.freeze_encoder and self.runtime_mode == "train":
            raise ValueError("UV face encoder training requires freeze_encoder=false")
        if edge_value_path and not edge_value_enabled:
            raise ValueError("init_encoder_edge_value_checkpoint requires encoder_edge_value.enabled")
        if edge_value_enabled and (legacy_path or refinement_path or curve_feedback_path):
            raise ValueError(
                "Encoder edge values require init_encoder_edge_value_checkpoint, checkpoint or resume"
            )
        if edge_value_enabled and (
            intersection_kind != "legacy" or not cfg["refinement"]["enabled"]
            or not conditioning_enabled or curve_feedback_enabled
        ):
            raise ValueError(
                "Encoder edge-value experiment retains conditioned legacy refinement without point feedback"
            )
        if edge_value_enabled and self.freeze_encoder and self.runtime_mode == "train":
            raise ValueError("Encoder edge-value training requires freeze_encoder=false")
        if curve_feedback_enabled and (
            intersection_kind != "legacy" or not cfg["refinement"]["enabled"] or not conditioning_enabled
        ):
            raise ValueError(
                "Curve point feedback requires legacy intersection with refinement "
                "and curve_conditioning enabled"
            )
        if curve_feedback_path and not curve_feedback_enabled:
            raise ValueError("init_curve_feedback_checkpoint requires enabled curve_point_feedback")
        if curve_feedback_enabled and (legacy_path or refinement_path):
            raise ValueError(
                "Curve point feedback must retain trained conditioning: use "
                "init_curve_feedback_checkpoint, checkpoint or resume"
            )
        if refinement_path and not (
            intersection_kind == "local_attention" or conditioning_enabled
        ):
            raise ValueError(
                "init_refinement_checkpoint requires local_attention or enabled curve_conditioning"
            )

        # Construct the original modules once, strictly load their full state,
        # then transfer them to explicit owners. No duplicate legacy model is
        # registered in this model, and all original geometry heads are reused.
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
        if refinement_path:
            if conditioning_enabled:
                self._init_curve_conditioning(refinement_path)
            else:
                self._init_from_refinement(refinement_path, legacy.inter.state_dict())
        if curve_feedback_path:
            self._init_curve_feedback(curve_feedback_path)
        if edge_value_path:
            self._init_encoder_edge_value(edge_value_path)
        if face_encoder_path:
            self._init_face_encoder(face_encoder_path, legacy_face_state)
        if uv_boundary_path:
            self._init_uv_boundary(uv_boundary_path)
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

    def _init_uv_boundary(self, path: str) -> None:
        """Add only an auxiliary UV head to the complete adopted conditioned VAE."""
        checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, Mapping):
            raise ValueError("UV boundary source must contain a state_dict mapping")
        raw_state = checkpoint.get("state_dict", checkpoint)
        if not isinstance(raw_state, Mapping):
            raise ValueError("UV boundary source state_dict must be a mapping")
        source = {}
        for key, value in raw_state.items():
            if not isinstance(key, str):
                raise ValueError("UV boundary source state keys must be strings")
            normalized = key.removeprefix("model.")
            if normalized in source:
                raise ValueError(f"Ambiguous checkpoint key after removing model prefix: {normalized}")
            source[normalized] = value
        prefix = "decoder.uv_boundary."
        current = self.state_dict()
        fresh = {key: value for key, value in current.items() if key.startswith(prefix)}
        if not fresh:
            raise ValueError("UV boundary migration requires enabled UV head parameters")
        expected = set(current) - set(fresh)
        missing = sorted(expected - source.keys())
        unexpected = sorted(source.keys() - expected)
        mismatched = sorted(key for key in expected & source.keys()
                            if not isinstance(source[key], Tensor) or source[key].shape != current[key].shape)
        if missing or unexpected or mismatched:
            raise ValueError(
                "UV boundary migration requires the complete conditioned CNN refinement model; "
                f"missing={missing}, unexpected={unexpected}, shape_mismatch={mismatched}"
            )
        self.load_state_dict({**source, **fresh}, strict=True)
        self.initialization_report = {
            "source": str(path), "source_epoch": checkpoint.get("epoch"),
            "source_global_step": checkpoint.get("global_step"),
            "retained_state_keys": len(source), "initialized_uv_boundary_keys": sorted(fresh),
            "initialized_uv_boundary_parameters": sum(value.numel() for value in fresh.values()),
        }
        logger.info("UV boundary initialization: %s", self.initialization_report)

    def _init_face_encoder(self, path: str, legacy_face_state: Mapping[str, Tensor]) -> None:
        """Replace exactly the CNN; strictly retain the complete conditioned VAE.

        Validate the discarded CNN too: a partial, already-UV, local-inter,
        point-feedback or edge-value checkpoint is not a migration source.
        Complete UV models must use checkpoint / resume instead.
        """
        checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, Mapping):
            raise ValueError("Face encoder checkpoint must be a mapping")
        raw_state = checkpoint.get("state_dict", checkpoint)
        if not isinstance(raw_state, Mapping):
            raise ValueError("Face encoder state_dict must be a mapping")
        source = {}
        for key, value in raw_state.items():
            if not isinstance(key, str):
                raise ValueError("Face encoder source state keys must be strings")
            normalized = key.removeprefix("model.")
            if normalized in source:
                raise ValueError(f"Ambiguous checkpoint key after removing model prefix: {normalized}")
            source[normalized] = value

        prefix = "encoder.face_coords."
        expected_old = {prefix + key: value for key, value in legacy_face_state.items()}
        old = {key: value for key, value in source.items() if key.startswith(prefix)}
        if old.keys() != expected_old.keys() or any(
            not isinstance(value, Tensor) or value.shape != expected_old[key].shape
            for key, value in old.items()
        ):
            raise ValueError("Face encoder source must contain the complete original face CNN state")
        current = self.state_dict()
        retained = {key: value for key, value in source.items() if not key.startswith(prefix)}
        expected = {key for key in current if not key.startswith(prefix)}
        missing = sorted(expected - retained.keys())
        unexpected = sorted(retained.keys() - expected)
        mismatched = sorted(
            key for key in expected & retained.keys()
            if not isinstance(retained[key], Tensor) or retained[key].shape != current[key].shape
        )
        if missing or unexpected or mismatched:
            raise ValueError(
                "Face encoder migration may replace only encoder.face_coords in the complete conditioned model; "
                f"missing={missing}, unexpected={unexpected}, shape_mismatch={mismatched}"
            )
        fresh = {key: value for key, value in current.items() if key.startswith(prefix)}
        self.load_state_dict({**retained, **fresh}, strict=True)
        self.initialization_report = {
            "source": str(path),
            "source_epoch": checkpoint.get("epoch"),
            "source_global_step": checkpoint.get("global_step"),
            "retained_state_keys": len(retained),
            "discarded_face_encoder_state_keys": len(old),
            "discarded_face_encoder_parameters": sum(value.numel() for value in old.values()),
            "initialized_face_encoder_state_keys": sorted(fresh),
            "initialized_face_encoder_parameters": sum(value.numel() for value in fresh.values()),
        }
        logger.info("UV face encoder initialization: %s", self.initialization_report)

    def _init_from_refinement(self, path: str, legacy_inter_state: Mapping[str, Tensor]) -> None:
        """Replace exactly the legacy inter, strictly preserving every other weight.

        This is initialization of a new architecture/run, never optimizer resume.
        Validate even the discarded keys so a partial or already-local checkpoint
        cannot be silently accepted as a legacy refinement source.
        """
        checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
        source = checkpoint.get("state_dict", checkpoint)
        source = {key.removeprefix("model."): value for key, value in source.items()}
        prefix = "decoder.inter."
        expected_old = {prefix + key: value for key, value in legacy_inter_state.items()}
        old = {key: value for key, value in source.items() if key.startswith(prefix)}
        if old.keys() != expected_old.keys():
            raise ValueError("Refinement source must contain the complete legacy decoder.inter state")
        for key, value in old.items():
            if value.shape != expected_old[key].shape:
                raise ValueError(f"Legacy intersection shape mismatch for {key}")

        current = self.state_dict()
        retained = {key: value for key, value in source.items() if not key.startswith(prefix)}
        expected = {key for key in current if not key.startswith(prefix)}
        if retained.keys() != expected:
            missing = sorted(expected - retained.keys())
            unexpected = sorted(retained.keys() - expected)
            raise ValueError(
                f"Refinement migration may replace only decoder.inter; "
                f"missing={missing}, unexpected={unexpected}"
            )
        # strict=True also rejects any shape mismatch in the retained modules.
        fresh_inter = {key: value for key, value in current.items() if key.startswith(prefix)}
        self.load_state_dict({**retained, **fresh_inter}, strict=True)
        self.initialization_report = {
            "source": str(path),
            "source_epoch": checkpoint.get("epoch"),
            "source_global_step": checkpoint.get("global_step"),
            "retained_state_keys": len(retained),
            "discarded_legacy_inter_keys": len(old),
            "initialized_local_inter_keys": len(fresh_inter),
        }
        logger.info("Local intersection initialization: %s", self.initialization_report)

    def _init_curve_conditioning(self, path: str) -> None:
        """Add only zero-initialized curve modulation to a complete refinement model.

        This migration retains the legacy inter and every pre-existing parameter.
        Local-inter and already-conditioned checkpoints are not valid sources;
        loading a complete conditioned model uses checkpoint / resume instead.
        """
        checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, Mapping):
            raise ValueError("Refinement checkpoint must contain a state_dict mapping")
        raw_state = checkpoint.get("state_dict", checkpoint)
        if not isinstance(raw_state, Mapping):
            raise ValueError("Refinement checkpoint state_dict must be a mapping")
        source = {}
        for key, value in raw_state.items():
            if not isinstance(key, str):
                raise ValueError("Refinement state keys must be strings")
            normalized = key.removeprefix("model.")
            if normalized in source:
                raise ValueError(f"Ambiguous checkpoint key after removing model prefix: {normalized}")
            source[normalized] = value

        prefix = "decoder.edge_geometry.conditioning."
        current = self.state_dict()
        fresh = {key: value for key, value in current.items() if key.startswith(prefix)}
        if not fresh:
            raise ValueError("Curve conditioning migration requires enabled conditioning parameters")
        expected = {key for key in current if not key.startswith(prefix)}
        missing = sorted(expected - source.keys())
        unexpected = sorted(source.keys() - expected)
        mismatched = sorted(
            key for key in expected & source.keys()
            if not isinstance(source[key], Tensor) or source[key].shape != current[key].shape
        )
        if missing or unexpected or mismatched:
            raise ValueError(
                "Curve conditioning migration requires the complete original refinement model; "
                f"missing={missing}, unexpected={unexpected}, shape_mismatch={mismatched}"
            )
        self.load_state_dict({**source, **fresh}, strict=True)
        self.initialization_report = {
            "source": str(path),
            "source_epoch": checkpoint.get("epoch"),
            "source_global_step": checkpoint.get("global_step"),
            "retained_state_keys": len(source),
            "initialized_curve_conditioning_keys": sorted(fresh),
            "initialized_curve_conditioning_parameters": sum(value.numel() for value in fresh.values()),
        }
        logger.info("Curve conditioning initialization: %s", self.initialization_report)

    def _init_curve_feedback(self, path: str) -> None:
        """Add only point feedback to a complete, already-conditioned refinement model.

        All source tensors, including learned conditioning, must be present with
        exactly their existing shapes. Only decoder.curve_point_feedback is new.
        Full feedback checkpoints use strict checkpoint / resume loading instead.
        """
        checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, Mapping):
            raise ValueError("Curve feedback source must contain a state_dict mapping")
        raw_state = checkpoint.get("state_dict", checkpoint)
        if not isinstance(raw_state, Mapping):
            raise ValueError("Curve feedback source state_dict must be a mapping")
        source = {}
        for key, value in raw_state.items():
            if not isinstance(key, str):
                raise ValueError("Curve feedback source state keys must be strings")
            normalized = key.removeprefix("model.")
            if normalized in source:
                raise ValueError(f"Ambiguous checkpoint key after removing model prefix: {normalized}")
            source[normalized] = value

        prefix = "decoder.curve_point_feedback."
        current = self.state_dict()
        fresh = {key: value for key, value in current.items() if key.startswith(prefix)}
        if not fresh:
            raise ValueError("Curve feedback migration requires enabled curve_point_feedback parameters")
        expected = {key for key in current if not key.startswith(prefix)}
        missing = sorted(expected - source.keys())
        unexpected = sorted(source.keys() - expected)
        mismatched = sorted(
            key for key in expected & source.keys()
            if not isinstance(source[key], Tensor) or source[key].shape != current[key].shape
        )
        if missing or unexpected or mismatched:
            raise ValueError(
                "Curve feedback migration requires the complete conditioned legacy-inter refinement model "
                "without existing point feedback; "
                f"missing={missing}, unexpected={unexpected}, shape_mismatch={mismatched}"
            )
        self.load_state_dict({**source, **fresh}, strict=True)
        self.initialization_report = {
            "source": str(path),
            "source_epoch": checkpoint.get("epoch"),
            "source_global_step": checkpoint.get("global_step"),
            "retained_state_keys": len(source),
            "initialized_curve_point_feedback_keys": sorted(fresh),
            "initialized_curve_point_feedback_parameters": sum(value.numel() for value in fresh.values()),
        }
        logger.info("Curve point feedback initialization: %s", self.initialization_report)

    def _init_encoder_edge_value(self, path: str) -> None:
        """Initialize only new GAT messages; preserve the full conditioned model.

        This starts a new optimizer/run. Existing edge-value checkpoints must
        use strict full checkpoint loading or trainer resume instead.
        """
        checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, Mapping):
            raise ValueError("Encoder edge-value checkpoint must be a mapping")
        raw_state = checkpoint.get("state_dict", checkpoint)
        if not isinstance(raw_state, Mapping):
            raise ValueError("Encoder edge-value state_dict must be a mapping")
        source = {}
        for key, value in raw_state.items():
            if not isinstance(key, str):
                raise ValueError("Encoder edge-value state keys must be strings")
            normalized = key.removeprefix("model.")
            if normalized in source:
                raise ValueError(f"Ambiguous checkpoint key after removing model prefix: {normalized}")
            source[normalized] = value
        current = self.state_dict()
        fresh = {
            f"encoder.graph_face_edge.{index}.{key}": value
            for index, layer in enumerate(self.encoder.graph_face_edge)
            if isinstance(layer, EdgeValueGATv2Conv)
            for key, value in layer.state_dict().items()
            if key.startswith("value_net.")
        }
        if not fresh:
            raise ValueError("Encoder edge-value migration requires enabled value_net parameters")
        expected = current.keys() - fresh.keys()
        missing = sorted(expected - source.keys())
        unexpected = sorted(source.keys() - expected)
        mismatched = sorted(
            key for key in expected & source.keys()
            if not isinstance(source[key], Tensor) or source[key].shape != current[key].shape
        )
        if missing or unexpected or mismatched:
            raise ValueError(
                "Encoder edge-value migration requires the complete conditioned model without value_net; "
                f"missing={missing}, unexpected={unexpected}, shape_mismatch={mismatched}"
            )
        self.load_state_dict({**source, **fresh}, strict=True)
        self.initialization_report = {
            "source": str(path),
            "source_epoch": checkpoint.get("epoch"),
            "source_global_step": checkpoint.get("global_step"),
            "retained_state_keys": len(source),
            "initialized_encoder_edge_value_keys": sorted(fresh),
            "initialized_encoder_edge_value_parameters": sum(value.numel() for value in fresh.values()),
        }
        logger.info("Encoder edge-value initialization: %s", self.initialization_report)

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

    def inference(self, face_latents: Tensor) -> dict:
        return self.decode_latents(face_latents)


# ---------------------------------------------------------------------------
# Shared-boundary matching and supervision
# ---------------------------------------------------------------------------


def boundary_costs(curves: Tensor, ownership_logits: Tensor, target: Mapping) -> tuple[Tensor, Tensor]:
    """Return [queries, targets] reversal/side-swap invariant L1 and NLL costs."""
    curves = curves.float()
    truth = target["curves"].to(device=curves.device, dtype=torch.float32)
    pairs = target["face_pairs"].to(device=curves.device, dtype=torch.long)
    direct = (curves[:, None] - truth[None]).abs().mean(dim=(-1, -2))
    reverse = (curves[:, None] - truth.flip(1)[None]).abs().mean(dim=(-1, -2))
    geometry = torch.minimum(direct, reverse)
    closed = target.get("is_closed")
    if closed is not None:
        # Closed targets include the repeated endpoint. Their sample origin is
        # arbitrary, so search both directions and every discrete cyclic origin.
        closed = closed.to(device=curves.device, dtype=torch.bool)
        closed_indices = closed.nonzero(as_tuple=True)[0]
        if len(closed_indices):
            unique = truth[closed_indices, :-1]
            positions = torch.arange(unique.shape[1], device=unique.device)
            forward_indices = (positions[None, :] - positions[:, None]) % unique.shape[1]
            # Preserve the original order: forward shifts 0..L-2, then reverse
            # shifts 0..L-2. The order matters for min() gradients at exact ties.
            indices = torch.cat((forward_indices, unique.shape[1] - 1 - forward_indices), dim=0)
            indices = torch.cat((indices, indices[:, :1]), dim=1)
            variants = unique[:, indices]
            closed_cost = (curves[:, None, None] - variants[None]).abs().mean(dim=(-1, -2)).min(dim=-1).values
            geometry = geometry.clone()
            geometry[:, closed_indices] = closed_cost
    log_prob = ownership_logits.float().log_softmax(-1)
    direct_owner = -(log_prob[:, 0, pairs[:, 0]] + log_prob[:, 1, pairs[:, 1]]) / 2
    swap_owner = -(log_prob[:, 0, pairs[:, 1]] + log_prob[:, 1, pairs[:, 0]]) / 2
    return geometry, torch.minimum(direct_owner, swap_owner)


def validate_boundary_target(target: Mapping, faces: int, capacity: int, resolution: int, name: str) -> None:
    curves, pairs = target["curves"], target["face_pairs"]
    if curves.ndim != 3 or curves.shape[1:] != (resolution, 3):
        raise ValueError(f"{name}: boundary curves must have shape [K, {resolution}, 3]")
    if pairs.shape != (len(curves), 2) or pairs.dtype != torch.long:
        raise ValueError(f"{name}: face_pairs must be int64 [K, 2]")
    if len(curves) > capacity:
        raise ValueError(f"{name}: {len(curves)} target boundaries exceed query capacity {capacity}")
    if not torch.isfinite(curves).all():
        raise ValueError(f"{name}: nonfinite target boundary geometry")
    if "is_closed" in target:
        closed = target["is_closed"]
        if closed.shape != (len(curves),) or closed.dtype != torch.bool:
            raise ValueError(f"{name}: is_closed must be bool [K]")
    if pairs.numel() and (pairs.min() < 0 or pairs.max() >= faces):
        raise ValueError(f"{name}: boundary face indices outside [0, {faces})")
    if pairs.numel() and (pairs[:, 0] == pairs[:, 1]).any():
        raise ValueError(f"{name}: seam targets are unsupported by shared_boundary_v1")
    if len(pairs) and len(torch.unique(pairs.sort(dim=-1).values, dim=0)) != len(pairs):
        raise ValueError(f"{name}: multiple target boundaries for one face pair are unsupported")


def _match_costs(
    geometry: Tensor, ownership: Tensor, curve_weight: float, ownership_weight: float
) -> tuple[Tensor, Tensor]:
    """Solve detached matching while retaining the original differentiable costs."""
    with torch.no_grad():
        cost = curve_weight * geometry + ownership_weight * ownership
        if not torch.isfinite(cost).all():
            raise ValueError("Nonfinite boundary matching cost")
        rows, columns = linear_sum_assignment(cost.detach().float().cpu().numpy())
    return (
        torch.as_tensor(rows, device=geometry.device, dtype=torch.long),
        torch.as_tensor(columns, device=geometry.device, dtype=torch.long),
    )


def match_boundaries(
    curves: Tensor, ownership_logits: Tensor, target: Mapping, curve_weight: float, ownership_weight: float
) -> tuple[Tensor, Tensor]:
    if len(target["curves"]) > len(curves):
        raise ValueError("Target boundary count exceeds query capacity")
    with torch.no_grad():
        geometry, ownership = boundary_costs(curves, ownership_logits, target)
        return _match_costs(geometry, ownership, curve_weight, ownership_weight)


def surface_boundary_consistency(surfaces: Tensor, curves: Tensor, ownership_logits: Tensor) -> Tensor:
    """Expected squared distance to each side's sampled surface (no UV claim).

    Distances are evaluated per face BEFORE weighting by ownership. Measuring
    distance to an average of unrelated surfaces would allow ghost boundaries.
    """
    if not len(curves):
        return curves.sum() * 0
    face_count = len(surfaces)
    resolution = curves.shape[1]
    distances = torch.cdist(curves.float().flatten(0, 1), surfaces.float().reshape(-1, 3)).square()
    distances = distances.reshape(len(curves), resolution, face_count, -1).min(dim=-1).values.mean(dim=1)
    return (ownership_logits.float().softmax(-1) * distances[:, None, :]).sum(-1).mean()


def joint_boundary_loss(
    state: Mapping[str, Tensor],
    targets: Sequence[Mapping],
    target_faces: Tensor,
    weights: Mapping[str, float],
    match_weights: Mapping[str, float],
    no_boundary_weight: float = 0.1,
    existence_threshold: float = 0.5,
    sample_names: Sequence[str] | None = None,
) -> dict[str, Tensor]:
    batch = len(state["face_counts"])
    if len(targets) != batch:
        raise ValueError("boundary_targets must contain one dictionary per object")
    if target_faces.shape != state["face_points"][state["face_valid"]].shape:
        raise ValueError("Target faces must be flat [sum_faces, 16, 16, 3] in unchanged face order")
    if not torch.isfinite(target_faces).all():
        raise ValueError("Target face geometry contains nonfinite values")
    face_loss = F.l1_loss(state["face_points"][state["face_valid"]].float(), target_faces.float())
    zero = state["curves"].sum() * 0
    curve_losses, owner_losses, existence_losses, consistency_losses = [], [], [], []
    count_errors, prediction_counts, ownership_accuracy, duplicate_counts, invalid_counts = [], [], [], [], []
    for item, target in enumerate(targets):
        count = int(state["face_counts"][item])
        curves = state["curves"][item]
        ownership = state["ownership_logits"][item, :, :, :count]
        name = str(sample_names[item]) if sample_names is not None else f"batch item {item}"
        validate_boundary_target(target, count, len(curves), curves.shape[1], name)
        geometry, owner = boundary_costs(curves, ownership, target)
        rows, columns = _match_costs(geometry, owner, **{
            "curve_weight": float(match_weights["curve"]),
            "ownership_weight": float(match_weights["ownership"]),
        })
        exists = torch.zeros_like(state["existence_logits"][item])
        exists[rows] = 1
        existence_weight = torch.where(exists.bool(), 1.0, no_boundary_weight)
        existence_losses.append((F.binary_cross_entropy_with_logits(
            state["existence_logits"][item], exists, reduction="none"
        ) * existence_weight).mean())
        if len(rows):
            curve_losses.append(geometry[rows, columns].mean())
            owner_losses.append(owner[rows, columns].mean())
            consistency_losses.append(surface_boundary_consistency(
                state["face_points"][item, :count], curves[rows], ownership[rows]
            ))
            predicted_pair = ownership[rows].argmax(-1).sort(-1).values
            target_pair = target["face_pairs"].to(curves.device)[columns].sort(-1).values
            ownership_accuracy.append((predicted_pair == target_pair).all(-1).float().mean())
        else:
            curve_losses.append(zero)
            owner_losses.append(zero)
            consistency_losses.append(zero)
            ownership_accuracy.append(zero.detach())
        with torch.no_grad():
            active = state["existence_logits"][item].sigmoid() >= existence_threshold
            predicted = ownership[active].argmax(-1).sort(-1).values
            invalid = (predicted[:, 0] == predicted[:, 1])
            valid_pairs = predicted[~invalid]
            duplicate_counts.append(curves.new_tensor(len(valid_pairs) - len(torch.unique(valid_pairs, dim=0))))
            invalid_counts.append(invalid.sum().to(curves.dtype))
            prediction_counts.append(active.sum().to(curves.dtype))
            count_errors.append((active.sum() - len(target["curves"])).abs().to(curves.dtype))
    loss = {
        "face": face_loss,
        "curve": torch.stack(curve_losses).mean(),
        "ownership": torch.stack(owner_losses).mean(),
        "existence": torch.stack(existence_losses).mean(),
        "consistency": torch.stack(consistency_losses).mean(),
    }
    loss["total_loss"] = sum(float(weights[key]) * value for key, value in loss.items())
    loss.update({
        "prediction_count": torch.stack(prediction_counts).mean(),
        "boundary_count_error": torch.stack(count_errors).mean(),
        "matched_ownership_accuracy": torch.stack(ownership_accuracy).mean().detach(),
        "raw_duplicate_count": torch.stack(duplicate_counts).mean(),
        "raw_invalid_incidence_count": torch.stack(invalid_counts).mean(),
    })
    return loss


# ---------------------------------------------------------------------------
# AutoEncoder_joint - Shared-boundary variant
# ---------------------------------------------------------------------------


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
        if not 0 <= self.existence_threshold <= 1:
            raise ValueError("joint_existence_threshold must be in [0, 1]")
        if set(self.loss_weights) != {"face", "curve", "existence", "ownership", "consistency"}:
            raise ValueError("joint_loss_weights must explicitly contain face/curve/existence/ownership/consistency")
        if set(self.match_weights) != {"curve", "ownership"}:
            raise ValueError("joint_match_weights must explicitly contain curve/ownership")
        if any(not np.isfinite(value) or value < 0 for value in [*self.loss_weights.values(), *self.match_weights.values()]):
            raise ValueError("Joint loss and matching weights must be finite and nonnegative")
        if not 0 < self.no_boundary_weight <= 1:
            raise ValueError("joint_no_boundary_weight must be in (0, 1]")
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

    @staticmethod
    def _checkpoint_weights(path) -> dict:
        checkpoint = torch.load(str(path), weights_only=False, map_location="cpu", mmap=True)
        state = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        if not isinstance(state, Mapping):
            raise ValueError("Checkpoint must contain a state_dict mapping")
        weights = {}
        for key, value in state.items():
            normalized = key[len("model."):] if key.startswith("model.") else key
            if normalized in weights:
                raise ValueError(f"Ambiguous checkpoint key after removing model prefix: {normalized}")
            weights[normalized] = value
        return weights

    def init_encoder_checkpoint(self, path) -> None:
        """Validated component migration; old decoder keys are explicitly ignored."""
        weights = self._checkpoint_weights(path)
        migrated = 0
        validated = []
        for name in self.ENCODER_MODULES:
            module = getattr(self, name)
            expected = module.state_dict()
            prefix = name + "."
            incoming = {key[len(prefix):]: value for key, value in weights.items() if key.startswith(prefix)}
            missing, unexpected = set(expected) - set(incoming), set(incoming) - set(expected)
            mismatched = [
                key for key in set(expected) & set(incoming)
                if not isinstance(incoming[key], Tensor) or incoming[key].shape != expected[key].shape
            ]
            if missing or unexpected or mismatched:
                raise ValueError(
                    f"Encoder migration failed for {name}: missing={sorted(missing)}, "
                    f"unexpected={sorted(unexpected)}, shape_mismatch={sorted(mismatched)}. "
                    "Use the exact legacy encoder architecture and posterior configuration."
                )
            validated.append((module, incoming))
            migrated += len(incoming)
        # Validate every component before modifying any parameter.
        for module, incoming in validated:
            module.load_state_dict(incoming, strict=True)
        logger.info("Migrated %d validated encoder/posterior tensors from %s; legacy decoder was not loaded", migrated, path)

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
        counts = v_data["face_counts"]
        if counts.ndim != 1 or not len(counts) or (counts <= 0).any() or int(counts.sum()) != len(v_data["face_points"]):
            raise ValueError("face_counts must partition encoder inputs into nonempty objects")
        if not torch.isfinite(v_data["face_points"]).all() or not torch.isfinite(v_data["edge_points"]).all():
            raise ValueError("Encoder geometry contains nonfinite values")
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


def build_autoencoder(v_conf):
    model_classes = {
        "AutoEncoder": AutoEncoder,
        "AutoEncoder_light": AutoEncoder_light,
        "AutoEncoder_light_exp": AutoEncoder_light_exp,
        "AutoEncoder_intersection": AutoEncoder_intersection,
        "AutoEncoder_joint": AutoEncoder_joint,
    }
    return model_classes[v_conf["name"]](v_conf)


__all__ = [
    "AutoEncoder",
    "AutoEncoder_light",
    "AutoEncoder_light_exp",
    "AutoEncoder_intersection",
    "AutoEncoder_joint",
    "build_autoencoder",
]
