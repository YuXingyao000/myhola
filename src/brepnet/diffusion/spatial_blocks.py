"""空间条件与面去噪共用的 attention；面 slots 不添加位置编号。"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint


def time_encoding(timesteps: Tensor, width: int) -> Tensor:
    frequency = torch.exp(-math.log(10000) * torch.arange(
        width // 2, device=timesteps.device, dtype=torch.float32) / (width // 2))
    phase = timesteps[:, None].float() * frequency[None]
    return torch.cat((phase.cos(), phase.sin()), dim=-1)


class AttentionRead(nn.Module):
    """归一化 Q/K/V；返回消息，residual 由调用者执行。"""

    def __init__(self, width: int, memory_width: int, heads: int, dropout: float):
        super().__init__()
        self.query_norm = nn.LayerNorm(width)
        self.memory_norm = nn.LayerNorm(memory_width)
        self.attention = nn.MultiheadAttention(
            width, heads, dropout=dropout, kdim=memory_width, vdim=memory_width,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: Tensor, memory: Tensor) -> Tensor:
        memory = self.memory_norm(memory)
        update, _ = self.attention(self.query_norm(query), memory, memory, need_weights=False)
        return self.dropout(update)


def feedforward(width: int, hidden: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(nn.Linear(width, hidden), nn.GELU(), nn.Dropout(dropout),
                         nn.Linear(hidden, width), nn.Dropout(dropout))


class SpatialQueryBlock(nn.Module):
    """512 个空间 query：self-attention → image cross-attention → FFN。"""

    def __init__(self, width: int, heads: int, ff_width: int, dropout: float):
        super().__init__()
        self.self_attention = AttentionRead(width, width, heads, dropout)
        self.image_attention = AttentionRead(width, width, heads, dropout)
        self.ff_norm = nn.LayerNorm(width)
        self.ff = feedforward(width, ff_width, dropout)

    def forward(self, spatial: Tensor, image: Tensor) -> Tensor:
        spatial = spatial + self.self_attention(spatial, spatial)
        spatial = spatial + self.image_attention(spatial, image)
        return spatial + self.ff(self.ff_norm(spatial))


class SpatialDenoisingBlock(nn.Module):
    """面 self-attention → 读取空间/RGB memory → FFN；时间调制每层。"""

    def __init__(self, width: int, memory_width: int, heads: int, ff_width: int, dropout: float):
        super().__init__()
        self.self_norm = nn.LayerNorm(width, elementwise_affine=False)
        self.self_attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.self_dropout = nn.Dropout(dropout)
        self.cross_attention = AttentionRead(width, memory_width, heads, dropout)
        self.ff_norm = nn.LayerNorm(width, elementwise_affine=False)
        self.ff = feedforward(width, ff_width, dropout)
        # shift/scale 初始为 0，gate 初始为 1，使条件分支从第一步就有梯度。
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, 6 * width))
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)
        with torch.no_grad():
            self.modulation[-1].bias[2 * width:3 * width].fill_(1)
            self.modulation[-1].bias[5 * width:6 * width].fill_(1)

    def forward(self, faces: Tensor, time: Tensor, memory: Tensor) -> Tensor:
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = self.modulation(time)[:, None].chunk(6, -1)
        query = self.self_norm(faces) * (1 + scale_a) + shift_a
        update, _ = self.self_attention(query, query, query, need_weights=False)
        faces = faces + gate_a * self.self_dropout(update)
        faces = faces + self.cross_attention(faces, memory)
        return faces + gate_f * self.ff(self.ff_norm(faces) * (1 + scale_f) + shift_f)


class SpatialMemoryDenoiser(nn.Module):
    """[B,30,32] + memory[B,T,512] → epsilon[B,30,32]；T 不改变面数。"""

    def __init__(self, cfg: dict, memory_width: int):
        super().__init__()
        self.width = cfg["width"]
        self.memory_width = memory_width
        if self.width < 2 or self.width % 2 or cfg["layers"] < 1:
            raise ValueError("Use an even positive denoiser width and at least one block")
        self.activation_checkpointing = cfg["activation_checkpointing"]
        self.input_projection = nn.Sequential(nn.Linear(32, self.width), nn.LayerNorm(self.width),
                                              nn.SiLU(), nn.Linear(self.width, self.width))
        self.time_embedding = nn.Sequential(nn.Linear(self.width, self.width), nn.SiLU(),
                                            nn.Linear(self.width, self.width))
        self.blocks = nn.ModuleList([
            SpatialDenoisingBlock(self.width, memory_width, cfg["heads"], cfg["ff_width"], cfg["dropout"])
            for _ in range(cfg["layers"])
        ])
        self.output_projection = nn.Sequential(nn.LayerNorm(self.width), nn.Linear(self.width, 32))

    def forward(self, latent: Tensor, timesteps: Tensor, memory: Tensor) -> Tensor:
        if latent.ndim != 3 or latent.shape[1:] != (30, 32) or timesteps.shape != (len(latent),):
            raise ValueError("Expected latent [B,30,32] and timesteps [B]")
        if memory.ndim != 3 or memory.shape[0] != len(latent) or memory.shape[1] < 1 or memory.shape[2] != self.memory_width:
            raise ValueError("Invalid spatial memory batch/token/channel shape")
        time = self.time_embedding(time_encoding(timesteps, self.width))
        hidden = self.input_projection(latent) + time[:, None]
        for block in self.blocks:
            if self.activation_checkpointing and self.training and torch.is_grad_enabled():
                hidden = checkpoint(block, hidden, time, memory, use_reentrant=False)
            else:
                hidden = block(hidden, time, memory)
        return self.output_projection(hidden)
