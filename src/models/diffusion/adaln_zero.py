"""原 concat 去噪器的 AdaLN-Zero blocks；零门控，统一无 dropout 前向。"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .preference import LegacySVRDenoiser


class AdaLNZeroBlock(nn.Module):
    """保留预训练 attention / FFN；条件控制 LN 与初始为零的残差门。"""

    def __init__(self, layer: nn.TransformerEncoderLayer, condition_width: int, hidden_width: int):
        super().__init__()
        width = layer.self_attn.embed_dim
        self.self_attn = layer.self_attn
        self.self_attn.dropout = 0.0
        self.linear1, self.linear2 = layer.linear1, layer.linear2
        self.activation = layer.activation
        self.norm1 = nn.LayerNorm(width, eps=layer.norm1.eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(width, eps=layer.norm2.eps, elementwise_affine=False)
        self.modulation = nn.Sequential(
            nn.Linear(condition_width, hidden_width), nn.SiLU(), nn.Linear(hidden_width, 6 * width),
        )
        nn.init.zeros_(self.modulation[2].weight)
        nn.init.zeros_(self.modulation[2].bias)

    def forward(self, hidden: Tensor, condition: Tensor) -> Tensor:
        # [B,1280] -> six [B,1,1024], shared by all face slots.
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = self.modulation(condition)[:, None].chunk(6, -1)
        normalized = (1 + scale_a) * self.norm1(hidden) + shift_a
        attention = self.self_attn(normalized, normalized, normalized, need_weights=False)[0]
        hidden = hidden + gate_a * attention
        normalized = (1 + scale_f) * self.norm2(hidden) + shift_f
        feedforward = self.linear2(self.activation(self.linear1(normalized)))
        return hidden + gate_f * feedforward


class AdaLNZeroDenoiser(LegacySVRDenoiser):
    """CLS-concat + time -> 24 zero-gated blocks -> original epsilon head."""

    def __init__(self, baseline: LegacySVRDenoiser, hidden_width: int = 256):
        nn.Module.__init__(self)
        self.width = baseline.width
        for name in ("p_embed", "time_embed", "net1", "fc_out", "img_fc"):
            setattr(self, name, getattr(baseline, name))
        condition_width = self.img_fc[-1].out_features + self.width
        self.net1.layers = nn.ModuleList([
            AdaLNZeroBlock(layer, condition_width, hidden_width) for layer in self.net1.layers
        ])

    def forward(self, noisy_latent: Tensor, timestep: Tensor, condition: Tensor) -> Tensor:
        time = self.time_condition(timestep)                         # [B,1,1024]
        latent = self.p_embed(noisy_latent)                           # [B,30,768]
        hidden = torch.cat((latent, condition.expand(-1, latent.shape[1], -1)), -1) + time
        context = torch.cat((condition[:, 0], time[:, 0]), -1)       # [B,1280]
        for layer in self.net1.layers:
            hidden = layer(hidden, context)
        if self.net1.norm is not None:
            hidden = self.net1.norm(hidden)
        return self.fc_out(hidden)                                  # epsilon [B,30,32]
