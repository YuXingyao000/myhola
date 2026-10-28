"""图像 patch → 带 XYZ 地址的空间 memory；无相机、GT 几何或面对应输入。"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .spatial_blocks import SpatialQueryBlock


class SpatialImageMemory(nn.Module):
    """默认 8³ 个格心 query，512 通道；可附加 256 个 RGB patch tokens。"""

    def __init__(self, cfg: dict, image_width: int = 1024):
        super().__init__()
        side, bands, width = cfg["grid_side"], cfg["fourier_bands"], cfg["width"]
        if side < 2 or bands < 1 or cfg["layers"] < 1 or cfg["extent"] <= 0:
            raise ValueError("Spatial grid, Fourier bands, extent and layer count must be positive")
        self.image_width = image_width
        self.retain_rgb = cfg["retain_rgb"]
        # 固定 canonical latent 坐标系的格心；不是从 GT 表面采样的位置。
        axis = ((torch.arange(side, dtype=torch.float32) + .5) / side * 2 - 1) * cfg["extent"]
        coordinates = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), -1).reshape(-1, 3)
        self.register_buffer("coordinates", coordinates)
        self.register_buffer("frequencies", math.pi * 2 ** torch.arange(bands, dtype=torch.float32))
        self.coordinate_projection = nn.Sequential(nn.Linear(3 + 6 * bands, width), nn.SiLU(),
                                                   nn.Linear(width, width))
        self.image_projection = nn.Sequential(nn.LayerNorm(image_width), nn.Linear(image_width, width))
        self.blocks = nn.ModuleList([
            SpatialQueryBlock(width, cfg["heads"], cfg["ff_width"], cfg["dropout"])
            for _ in range(cfg["layers"])
        ])
        self.output_norm = nn.LayerNorm(width)
        self.spatial_type = nn.Parameter(torch.zeros(1, 1, width))
        if self.retain_rgb:
            self.rgb_type = nn.Parameter(torch.zeros(1, 1, width))

    def forward(self, image_tokens: Tensor) -> Tensor:
        """DINO CLS+patch [B,257,1024] → [B,512(+256),512]。"""
        if image_tokens.ndim != 3 or image_tokens.shape[1:] != (257, self.image_width):
            raise ValueError("Expected DINO CLS + 16x16 patches at the configured channel width")
        phase = self.coordinates[..., None] * self.frequencies
        position = torch.cat((self.coordinates, phase.sin().flatten(1), phase.cos().flatten(1)), -1)
        spatial = self.coordinate_projection(position)[None].expand(len(image_tokens), -1, -1)
        image = self.image_projection(image_tokens)
        for block in self.blocks:
            spatial = block(spatial, image)
        spatial = self.output_norm(spatial) + self.spatial_type
        if self.retain_rgb:
            return torch.cat((spatial, image[:, 1:] + self.rgb_type), dim=1)
        return spatial
