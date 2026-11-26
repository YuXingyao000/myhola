"""Add native trimming-curve information after the original HoLa Encoder GAT."""
from __future__ import annotations

import torch
from torch import nn
from einops import rearrange
from torch_geometric.nn import GATv2Conv
from torch_scatter import scatter_mean

from .legacy import AutoEncoder_light


class PcurveFeatureBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, 4 * width), nn.GELU(),
            nn.Linear(4 * width, width),
        )

    def forward(self, curves):
        return curves + self.ffn(curves)


class PcurveCrossAttention(nn.Module):
    def __init__(self, width: int, heads: int, dropout: float):
        super().__init__()
        self.query_norm = nn.LayerNorm(width)
        self.memory_norm = nn.LayerNorm(width)
        self.ffn_norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(width, 4 * width), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(4 * width, width), nn.Dropout(dropout),
        )

    def forward(self, query, memory, padding_mask):
        memory = self.memory_norm(memory)
        message, _ = self.attention(
            self.query_norm(query), memory, memory,
            key_padding_mask=padding_mask, need_weights=False,
        )
        query = query + self.dropout(message)
        return query + self.ffn(self.ffn_norm(query))


class PcurveFaceInput(nn.Module):
    def __init__(self, face_dim: int, *, width: int, curve_depth: int,
                 layers: int, heads: int, dropout: float):
        super().__init__()
        self.curve_encoder = nn.Sequential(
            nn.Linear(16 * 2, width),
            *(PcurveFeatureBlock(width) for _ in range(curve_depth)),
        )
        self.face_projection = nn.Linear(face_dim, width)
        self.layers = nn.ModuleList(
            PcurveCrossAttention(width, heads, dropout) for _ in range(layers)
        )
        self.output_projection = nn.Linear(width, face_dim)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, face, uv, curve_indices, padding_mask):
        curves = self.curve_encoder(uv.flatten(1))  # [H, 16*2] -> [H, D]
        memory = curves[curve_indices.clamp_min(0)]  # [N, max_curves_per_face, D]
        query = self.face_projection(face).unsqueeze(1)  # [N, 1, D]
        for layer in self.layers:
            query = layer(query, memory, padding_mask)
        return face + self.output_projection(query.squeeze(1))


class AutoEncoder_pcurve_input(AutoEncoder_light):
    """Original Decoder/losses; one insertion between Encoder GAT and global pool."""

    def __init__(self, config):
        source = config.get("init_hola_checkpoint")
        complete = config.get("checkpoint")
        if source and complete:
            raise ValueError("Choose published HoLa initialization or a complete pcurve-input checkpoint")
        # Strictly load all original tensors before registering the new branch.
        super().__init__({**dict(config), "checkpoint": source})
        self.pcurve_input = PcurveFaceInput(self.face_feature_dim, **dict(config["pcurve_input"]))
        self._load_checkpoint(complete)
        self._apply_runtime_mode()

    def encode(self, v_data, v_test):
        face_points = rearrange(v_data["face_points"][..., :self.in_channels], 'b h w n -> b n h w').contiguous()
        edge_points = rearrange(v_data["edge_points"][..., :self.in_channels], 'b h n -> b n h').contiguous()
        face_features = self.face_coords(face_points)
        edge_features = self.edge_coords(edge_points)

        # Original face attention and GAT.
        attn_x = self.face_attn_proj_in(face_features)
        attn_x = self.face_attn(attn_x, v_data["attn_mask"])
        attn_x = self.face_attn_proj_out(attn_x)
        fused_face_features = face_features + attn_x
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

        # New: each face reads its own native UV curves, including seam uses.
        fused_face_features = self.pcurve_input(
            fused_face_features, v_data["pcurve_uv"],
            v_data["pcurve_indices"], v_data["pcurve_padding_mask"],
        )

        # Original global aggregation and posterior input.
        bs = v_data["face_counts"].shape[0]
        index = torch.arange(bs, device=fused_face_features.device).repeat_interleave(v_data["face_counts"])
        face_latents = fused_face_features
        gf = scatter_mean(fused_face_features, index, dim=0)
        gf = self.global_feature1(gf)
        gf = gf.repeat_interleave(v_data["face_counts"], dim=0)
        face_latents = torch.cat((fused_face_features, gf), dim=1)
        face_latents = self.global_feature2(face_latents) + fused_face_features
        return {"face_features": face_latents, "edge_features": edge_features}
