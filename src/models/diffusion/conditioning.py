"""条件装配：单视图图像交给 DINOv2 编码器；无条件时不构建编码器。"""

from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from .image_encoder import DINOv2ImageEncoder


class ConditionExtractor(nn.Module):
    """为 denoiser 提取单视图图像条件。

    图像输入 ``[B, 3, 224, 224]`` 返回 ``[B, 1, 257, D]``，其中 ``D=projection_dim``。
    当前图像分支始终消费原始 ``imgs``，没有特征缓存读取或多视图融合。
    """

    def __init__(self, projection_dim: int = 1024):
        super().__init__()
        self.projection_dim = projection_dim
        self.image_encoder = DINOv2ImageEncoder(projection_dim=projection_dim)

    def train(self, mode: bool = True):
        """Keep the frozen backbone in eval mode."""
        super().train(mode)
        self.image_encoder.train(mode)  # delegates to its own override
        return self

    def forward(self, data: Dict) -> torch.Tensor:
        """读取 ``data['conditions']['imgs']: [B, 3, 224, 224]``，返回 ``[B, 1, 257, D]``。"""
        img_feature = self.image_encoder(data["conditions"]["imgs"])
        # 保留 CLS 和所有 patch tokens；长度为 1 的轴沿用原接口。
        return img_feature[:, None]


def build_condition_encoder(condition_cfg: dict) -> nn.Module | None:
    """按配置选择单图或无条件；其他条件类型已移除。"""
    condition_type = condition_cfg["type"]
    if condition_type == "none":
        return None
    if condition_type != "single_img":
        raise NotImplementedError(f"{condition_type} conditioning was removed; only single_img and none remain.")
    return ConditionExtractor(projection_dim=condition_cfg["output_dim"])
