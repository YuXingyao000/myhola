"""Variational AutoEncoder models for BRep generation.

This module contains all VAE architectures used in the HoLa-BRep pipeline.
Each model follows the same high-level structure:

    encode()     -- Produce per-face and per-edge feature vectors via:
                    CNN (surface/curve grids) -> Graph message-passing (GATv2Conv)
                    -> Transformer self-attention -> global feature aggregation.

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
"""

import copy
import time

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
)


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


def build_autoencoder(v_conf):
    model_classes = {
        "AutoEncoder": AutoEncoder,
        "AutoEncoder_light": AutoEncoder_light,
        "AutoEncoder_light_exp": AutoEncoder_light_exp,
    }
    return model_classes[v_conf["name"]](v_conf)


__all__ = [
    "AutoEncoder",
    "AutoEncoder_light",
    "AutoEncoder_light_exp",
    "build_autoencoder",
]
