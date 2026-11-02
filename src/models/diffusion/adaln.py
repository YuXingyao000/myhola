"""在原 concat denoiser 上增加图像／时间调制；零增量保留旧网络。"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .preference import LegacySVRDenoiser


class DeltaModulation(nn.Module):
    """全局条件 → attention / FFN 各自的 scale、shift、gate 增量。"""

    def __init__(self, condition_width: int, width: int, hidden_width: int):
        super().__init__()
        self.input = nn.Linear(condition_width, hidden_width)
        self.activation = nn.SiLU()
        self.output = nn.Linear(hidden_width, 6 * width)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, condition: Tensor) -> tuple[Tensor, ...]:
        # [B,1280] → six [B,1,1024], shared by every face slot.
        delta = self.output(self.activation(self.input(condition)))
        return delta[:, None].chunk(6, dim=-1)


def modulated_encoder_layer(layer: nn.TransformerEncoderLayer, hidden: Tensor,
                            delta: tuple[Tensor, ...]) -> Tensor:
    """复用旧 affine LN / attention / ReLU / dropout；残差门为 1 + Δ。"""
    scale_a, shift_a, gate_a, scale_f, shift_f, gate_f = delta
    normalized = (1 + scale_a) * layer.norm1(hidden) + shift_a
    attention = layer.self_attn(normalized, normalized, normalized, need_weights=False)[0]
    hidden = hidden + (1 + gate_a) * layer.dropout1(attention)

    normalized = (1 + scale_f) * layer.norm2(hidden) + shift_f
    feedforward = layer.linear2(layer.dropout(layer.activation(layer.linear1(normalized))))
    return hidden + (1 + gate_f) * layer.dropout2(feedforward)


class AdaLNSVRDenoiser(LegacySVRDenoiser):
    """保留原五个模块及 state keys，仅新增每层独立的 modulation MLP。"""

    def __init__(self, baseline: LegacySVRDenoiser, hidden_width: int = 256):
        nn.Module.__init__(self)
        self.width = baseline.width
        for name in ("p_embed", "time_embed", "net1", "fc_out", "img_fc"):
            setattr(self, name, getattr(baseline, name))
        condition_width = self.img_fc[-1].out_features + self.width
        self.modulations = nn.ModuleList([
            DeltaModulation(condition_width, self.width, hidden_width)
            for _ in self.net1.layers
        ])

    def forward(self, noisy_latent: Tensor, timestep: Tensor, condition: Tensor) -> Tensor:
        time = self.time_condition(timestep)                         # [B,1,1024]
        latent = self.p_embed(noisy_latent)                           # [B,30,768]
        tokens = torch.cat((latent, condition.repeat(1, noisy_latent.shape[1], 1)), -1)
        hidden = tokens + time                                      # [B,30,1024]
        context = torch.cat((condition[:, 0], time[:, 0]), dim=-1)    # [B,1280]
        for layer, modulation in zip(self.net1.layers, self.modulations):
            hidden = modulated_encoder_layer(layer, hidden, modulation(context))
        if self.net1.norm is not None:
            hidden = self.net1.norm(hidden)
        return self.fc_out(hidden)                                  # epsilon [B,30,32]
