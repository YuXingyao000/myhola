from __future__ import annotations

import torch
from torch import Tensor, nn

from src.brepnet.models.blocks import sincos_embedding


class NoConditionFuser(nn.Module):
    def forward(self, latent_hidden: Tensor, condition: Tensor | None) -> Tensor:
        return latent_hidden


class CrossAttentionConditionFuser(nn.Module):
    def __init__(self, hidden_dim: int, condition_dim: int, cross_attention_dim: int, num_layers: int):
        super().__init__()
        self.query_projection = nn.Linear(hidden_dim, cross_attention_dim)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=cross_attention_dim,
            nhead=cross_attention_dim // 64,
            norm_first=True,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers,
            nn.LayerNorm(cross_attention_dim),
        )
        self.condition_projection = nn.Identity() if condition_dim == cross_attention_dim else nn.Linear(condition_dim, cross_attention_dim)
        self.output_projection = nn.Linear(cross_attention_dim, hidden_dim)

    def forward(self, latent_hidden: Tensor, condition: Tensor) -> Tensor:
        if condition.dim() == 4:
            condition = condition[:, 0]
        elif condition.dim() == 2:
            condition = condition[:, None]
        condition = self.condition_projection(condition)
        latent_hidden = self.query_projection(latent_hidden)
        latent_hidden = self.decoder(tgt=latent_hidden, memory=condition)
        return self.output_projection(latent_hidden)


def build_condition_fuser(fuser_cfg: dict, condition_cfg: dict, hidden_dim: int) -> nn.Module:
    if condition_cfg["type"] == "none":
        return NoConditionFuser()
    if fuser_cfg["type"] == "cross_attention":
        return CrossAttentionConditionFuser(
            hidden_dim=hidden_dim,
            condition_dim=fuser_cfg["condition_dim"],
            cross_attention_dim=fuser_cfg["hidden_dim"],
            num_layers=fuser_cfg["num_layers"],
        )
    raise ValueError(f"Unknown condition fuser type '{fuser_cfg['type']}'.")


class TopologyBias(nn.Module):
    def __init__(self, cfg: dict, num_train_timesteps: int):
        super().__init__()
        self.enabled = cfg["enabled"]
        self.scale = cfg["scale"]
        self.num_train_timesteps = num_train_timesteps

    def forward(self, face_adjacency: Tensor | None, timesteps: Tensor) -> Tensor | None:
        if not self.enabled:
            return None
        if face_adjacency is None:
            return None
        timestep_weight = timesteps.float() / float(self.num_train_timesteps)
        return face_adjacency.to(timesteps.device) * timestep_weight[:, None, None] * self.scale


class BRepDenoiser(nn.Module):
    def __init__(self, latent_dim: int, cfg: dict, condition_fuser: nn.Module):
        super().__init__()
        self.hidden_dim = cfg["hidden_dim"]
        self.condition_fuser = condition_fuser
        self.input_projection = nn.Sequential(
            nn.Linear(latent_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.hidden_dim // cfg["nhead_divisor"],
            norm_first=True,
            dim_feedforward=cfg["feedforward_dim"],
            dropout=cfg["dropout"],
            batch_first=True,
        )
        self.backbone = nn.TransformerEncoder(layer, cfg["num_layers"], nn.LayerNorm(self.hidden_dim))
        self.output_projection = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, latent_dim),
        )

    def forward(
        self,
        noisy_latent_sequence: Tensor,
        timesteps: Tensor,
        condition: Tensor | None,
        attention_bias: Tensor | None = None,
    ) -> Tensor:
        hidden = self.input_projection(noisy_latent_sequence)
        hidden = self.condition_fuser(hidden, condition)
        hidden = hidden + self.time_embedding(sincos_embedding(timesteps, self.hidden_dim)).unsqueeze(1)
        hidden = self.run_backbone(hidden, attention_bias)
        return self.output_projection(hidden)

    def run_backbone(self, hidden: Tensor, attention_bias: Tensor | None) -> Tensor:
        if attention_bias is None:
            return self.backbone(hidden)

        batch_size, seq_len, _ = attention_bias.shape
        nhead = self.backbone.layers[0].self_attn.num_heads
        src_mask = attention_bias.unsqueeze(1).expand(-1, nhead, -1, -1)
        src_mask = src_mask.reshape(batch_size * nhead, seq_len, seq_len)

        output = hidden
        for layer in self.backbone.layers:
            output = layer(output, src_mask=src_mask)
        if self.backbone.norm is not None:
            output = self.backbone.norm(output)
        return output
