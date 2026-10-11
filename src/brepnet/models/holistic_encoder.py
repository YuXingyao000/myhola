"""Explicit encoder/posterior modules retaining the original HoLa contract.

Legacy modules are transferred at construction, so parameter shapes and weights
remain identical. No decoder or loss computation lives in this file.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn
from torch_geometric.nn import GATv2Conv
from torch_scatter import scatter_mean


class HolisticEncoder(nn.Module):
    def __init__(self, legacy: nn.Module):
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
