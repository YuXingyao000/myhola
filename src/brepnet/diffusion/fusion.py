"""图像条件融合；保留原有子模块名称以兼容 checkpoint。"""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor, nn

from .losses import clip_style_symmetric_infonce


class NoConditionFuser(nn.Module):
    def forward(
        self,
        latent_hidden: Tensor,
        condition: Tensor | None,
        alignment_hidden: Tensor | None,
    ) -> tuple[Tensor, Tensor | None]:
        return latent_hidden, None


class CrossAttentionConditionFuser(nn.Module):
    """CAD slots 查询图像 memory；默认 [B,F,768] → [B,F,1024] → [B,F,768]。"""

    def __init__(
        self,
        hidden_dim: int,
        condition_dim: int,
        cross_attention_dim: int,
        num_layers: int,
        alignment_cfg: dict,
    ):
        super().__init__()
        self.query_projection = nn.Linear(hidden_dim, cross_attention_dim)
        # 每个 decoder block 自带 self-attention、cross-attention 和 FFN。
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
        self.alignment_enabled = alignment_cfg["enabled"]
        self.alignment_weight = alignment_cfg["weight"]
        self.alignment_temperature = alignment_cfg["temperature"]
        self.alignment_projection_dim = alignment_cfg["projection_dim"]
        self.cad_alignment_projection = nn.Linear(cross_attention_dim, self.alignment_projection_dim)
        self.condition_alignment_projection = nn.Linear(cross_attention_dim, self.alignment_projection_dim)
        self.cad_alignment_head = nn.Linear(self.alignment_projection_dim, self.alignment_projection_dim)
        self.condition_alignment_head = nn.Linear(self.alignment_projection_dim, self.alignment_projection_dim)

    def prepare_condition(self, condition: Tensor) -> Tensor:
        """单图 [B,1,T,D] → [B,T,D]；全局条件 [B,D] → [B,1,D]。"""
        if condition.dim() == 4:
            condition = condition[:, 0]
        elif condition.dim() == 2:
            condition = condition[:, None]
        return self.condition_projection(condition)

    def forward(
        self,
        latent_hidden: Tensor,
        condition: Tensor,
        alignment_hidden: Tensor | None,
    ) -> tuple[Tensor, Tensor | None]:
        condition_memory = self.prepare_condition(condition)
        cad_tgt = self.query_projection(latent_hidden)

        align_loss = None
        if self.alignment_enabled:
            # 全局对象级对比目标：clean CAD 和图像分别池化，不监督 face/pixel 对应。
            assert alignment_hidden is not None
            cad_align_src = self.query_projection(alignment_hidden)
            cad_align_tokens = self.cad_alignment_projection(cad_align_src)
            condition_align_tokens = self.condition_alignment_projection(condition_memory)

            cad_global = cad_align_tokens.mean(dim=1)
            condition_global = condition_align_tokens.mean(dim=1)
            cad_embedding = F.normalize(self.cad_alignment_head(cad_global), dim=-1)
            condition_embedding = F.normalize(self.condition_alignment_head(condition_global), dim=-1)
            align_loss = clip_style_symmetric_infonce(
                cad_embedding,
                condition_embedding,
                self.alignment_temperature,
            ) * self.alignment_weight

        latent_hidden = self.decoder(tgt=cad_tgt, memory=condition_memory)
        latent_hidden = self.output_projection(latent_hidden)
        return latent_hidden, align_loss


def build_condition_fuser(fuser_cfg: dict, condition_cfg: dict, hidden_dim: int) -> nn.Module:
    if condition_cfg["type"] == "none":
        return NoConditionFuser()
    if fuser_cfg["type"] == "cross_attention":
        return CrossAttentionConditionFuser(
            hidden_dim=hidden_dim,
            condition_dim=fuser_cfg["condition_dim"],
            cross_attention_dim=fuser_cfg["hidden_dim"],
            num_layers=fuser_cfg["num_layers"],
            alignment_cfg=fuser_cfg["alignment"],
        )
    raise ValueError(f"Unknown condition fuser type '{fuser_cfg['type']}'.")
