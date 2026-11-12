"""AdaLN 去噪主干与连续的局部图像控制支路。"""

from __future__ import annotations

from copy import deepcopy

import torch
from torch import Tensor, nn

from .adaln_zero import AdaLNZeroBlock, AdaLNZeroDenoiser


class PatchControlBlock(nn.Module):
    """复制已学 AdaLN block，读取图像 patches，输出同槽位的零增量。"""

    def __init__(self, backbone_block: AdaLNZeroBlock, patch_width: int):
        super().__init__()
        width = backbone_block.self_attn.embed_dim
        heads = backbone_block.self_attn.num_heads
        self.block = deepcopy(backbone_block).requires_grad_(True)
        self.query_norm = nn.LayerNorm(width)
        self.patch_norm = nn.LayerNorm(patch_width)
        self.cross_attention = nn.MultiheadAttention(
            width, heads, kdim=patch_width, vdim=patch_width, dropout=0.0, batch_first=True,
        )
        self.zero_out = nn.Linear(width, width)
        nn.init.zeros_(self.zero_out.weight)
        nn.init.zeros_(self.zero_out.bias)

    def forward(self, hidden: Tensor, context: Tensor, patches: Tensor) -> tuple[Tensor, Tensor]:
        hidden = self.block(hidden, context)                    # [B,30,1024]
        query = self.query_norm(hidden)
        memory = self.patch_norm(patches)                      # [B,256,1024]
        hidden = hidden + self.cross_attention(query, memory, memory, need_weights=False)[0]
        return hidden, self.zero_out(hidden)


class AdaLNControlNetDenoiser(nn.Module):
    """完整冻结已学主干；对应层控制副本连续递推、独立训练。"""

    def __init__(self, backbone: AdaLNZeroDenoiser, patch_width: int):
        super().__init__()
        self.backbone = backbone.requires_grad_(False).eval()
        self.control_blocks = nn.ModuleList([
            PatchControlBlock(block, patch_width) for block in backbone.net1.layers
        ])

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, noisy_latent: Tensor, timestep: Tensor,
                condition: Tensor, patches: Tensor) -> Tensor:
        time = self.backbone.time_condition(timestep)           # [B,1,1024]
        latent = self.backbone.p_embed(noisy_latent)             # [B,30,768]
        hidden = torch.cat((latent, condition.expand(-1, latent.shape[1], -1)), -1) + time
        context = torch.cat((condition[:, 0], time[:, 0]), -1)  # [B,1280]
        control = hidden
        for backbone_block, control_block in zip(self.backbone.net1.layers, self.control_blocks):
            # 冻结参数仍需对输入反传，将梯度传回前面各层的控制增量。
            hidden = backbone_block(hidden, context)
            control, residual = control_block(control, context, patches)
            hidden = hidden + residual
        if self.backbone.net1.norm is not None:
            hidden = self.backbone.net1.norm(hidden)
        return self.backbone.fc_out(hidden)                     # epsilon [B,30,32]
