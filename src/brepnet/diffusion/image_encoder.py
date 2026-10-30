"""单视图图像编码：冻结 ViT 特征与可训练的 token 投影。"""

from __future__ import annotations

import torch
from torch import nn


class DINOv2ImageEncoder(nn.Module):
    """Frozen DINOv2 ViT-L backbone (HuggingFace `facebook/dinov2-large`) with a learnable projection head.

    For 224 x 224 images the backbone produces [B, 257, 1024] tokens
    (1 CLS + 16x16 patches). The learnable projection maps these to
    [B, 257, projection_dim].
    """

    def __init__(self, projection_dim: int = 1024):
        super().__init__()
        self.projection_dim = projection_dim

        from transformers import Dinov2Model
        self.img_model = Dinov2Model.from_pretrained('facebook/dinov2-large')

        # 冻结 backbone，梯度只更新后续 projection。
        for param in self.img_model.parameters():
            param.requires_grad = False

        # 逐 token 投影：[B, 257, 1024] → [B, 257, projection_dim]。
        self.projection = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.LayerNorm(1024),
            nn.SiLU(),
            nn.Linear(1024, projection_dim),
        )

    def train(self, mode: bool = True):
        """Keep the frozen backbone in eval mode regardless of parent state."""
        super().train(mode)
        self.img_model.eval()
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """``images: [B, 3, 224, 224]`` (DINOv2-normalized) → ``[B, 257, projection_dim]``."""
        with torch.no_grad():
            img_feature = self.img_model(images).last_hidden_state
        return self.projection(img_feature)
