"""面 latent 去噪器：输入投影、条件融合、时间编码、自注意力主干、输出投影。"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from src.brepnet.models.blocks import sincos_embedding


class BRepDenoiser(nn.Module):
    """[B, F, C] noisy latent → 同形状预测；默认 C=32、F=30、hidden=768。"""

    def __init__(self, latent_dim: int, cfg: dict, condition_fuser: nn.Module):
        super().__init__()
        self.hidden_dim = cfg["hidden_dim"]
        self.condition_fuser = condition_fuser
        # 输入/时间/输出投影均为两层 MLP，保持旧参数注册名称。
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
        # 每层：pre-LN self-attention + residual，再 pre-LN FFN + residual。
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
        clean_latent_sequence: Tensor | None = None,
        self_attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        # clean latent 仅供 alignment 支路；去噪主路读取 noisy latent。
        time_embedding = self.time_embedding(sincos_embedding(timesteps, self.hidden_dim)).unsqueeze(1)
        hidden = self.input_projection(noisy_latent_sequence)
        alignment_hidden = self.input_projection(clean_latent_sequence) if clean_latent_sequence is not None else None
        hidden, align_loss = self.condition_fuser(
            hidden,
            condition,
            alignment_hidden,
        )
        hidden = hidden + time_embedding  # [B, 1, H] 广播；时间在 fuser 之后注入。
        hidden = self.run_backbone(hidden, self_attention_mask)
        return self.output_projection(hidden), align_loss

    def run_backbone(self, hidden: Tensor, self_attention_mask: Tensor | None) -> Tensor:
        """只在主干 self-attention 使用可选 topology mask/bias。"""
        if self_attention_mask is None:
            return self.backbone(hidden)

        batch_size, seq_len, _ = self_attention_mask.shape
        nhead = self.backbone.layers[0].self_attn.num_heads
        src_mask = self_attention_mask.unsqueeze(1).expand(-1, nhead, -1, -1)
        src_mask = src_mask.reshape(batch_size * nhead, seq_len, seq_len)

        output = hidden
        for layer in self.backbone.layers:
            output = layer(output, src_mask=src_mask)
        if self.backbone.norm is not None:
            output = self.backbone.norm(output)
        return output
