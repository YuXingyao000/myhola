"""条件装配与路由：将图像或点云交给对应的编码器。"""

from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn

from .image_encoder import DINOv2ImageEncoder
from .legacy_encoders import CameraEmbedding
from .point_encoder import PointNetEncoder


class ConditionExtractor(nn.Module):
    """为 denoiser 提取一个条件分支；图像优先于点云。

    图像输入 ``[B, 3, 224, 224]`` 返回 ``[B, 1, 257, D]``；点云输入
    ``[B, 1, N, 6]`` 返回 ``[B, 1, D]``，其中 ``D=projection_dim``。
    当前图像分支始终消费原始 ``imgs``，没有特征缓存读取或多视图融合。

    ``config`` 由 :func:`build_condition_encoder` 从 Hydra 条件配置转换。
    直接构造时仍需提供对应分支使用的配置键，不在这里静默补全配置。
    """

    def __init__(self, config: dict, projection_dim: int = 1024):
        super().__init__()
        self.projection_dim = projection_dim
        self.with_img = False
        self.with_pc = False

        condition = config["condition"]
        if "multi_img" in condition:
            raise NotImplementedError("multi_img is legacy HoLa-BRep multi-view conditioning and is disabled.")
        if "txt" in condition:
            raise NotImplementedError("txt conditioning is disabled until the new dataset format is defined.")

        # 图像编码；保留构造顺序以兼容旧权重及随机初始化。
        if ("single_img" in condition or "sketch" in condition):
            self.with_img = True
            backbone = config["backbone"]
            da_ckpt = config["depth_anything_v2_ckpt"]
            self.image_encoder = DINOv2ImageEncoder(
                projection_dim=projection_dim,
                backbone=backbone,
                depth_anything_v2_ckpt=da_ckpt,
            )
            # 单视图 forward 不使用它；保留参数名称以加载旧 checkpoint。
            self.camera_embedding = CameraEmbedding(
                num_views=8,
                output_dim=projection_dim,
            )

        # 点云为独立条件分支，不与图像同时融合。
        if "pc" in condition:
            self.with_pc = True
            self.point_encoder = PointNetEncoder(config, output_dim=projection_dim)

    def train(self, mode: bool = True):
        """Keep all frozen sub-encoders in eval mode."""
        super().train(mode)
        if self.with_img:
            self.image_encoder.train(mode)  # delegates to its own override
        return self

    def forward(self, data: Dict) -> Optional[torch.Tensor]:
        """读取 ``data['conditions']``，返回条件特征或 ``None``。

        ``imgs: [B, 3, 224, 224] → [B, 1, 257, D]``；
        ``points: [B, 1, N, 6] → [B, 1, D]``。
        点云可附带 ``data['rotation_id']: [B]``；图像不读取 ``img_id``。
        """
        condition = None

        conditions = data["conditions"]
        if self.with_img:
            imgs = conditions["imgs"]
            img_feature = self.image_encoder(imgs)

            # 保留 CLS 和所有 patch tokens；长度为 1 的轴沿用原接口。
            condition = img_feature[:, None]
        elif self.with_pc:
            pc = conditions["points"]
            rotation_id = data.get("rotation_id", None)
            feat = self.point_encoder(pc[:, 0, :, :], rotation_id)
            condition = feat[:, None]

        return condition


def build_condition_encoder(condition_cfg: dict) -> nn.Module | None:
    """按配置选择单图、草图、点云或无条件；禁用路线显式报错。"""
    condition_type = condition_cfg["type"]
    if condition_type == "none":
        return None
    if condition_type in {"multi_img", "text"}:
        raise NotImplementedError(f"{condition_type} conditioning is disabled in the single-view runtime.")

    modality_by_type = {
        "single_img": ["single_img"],
        "sketch": ["sketch"],
        "point_cloud": ["pc"],
    }
    modalities = modality_by_type[condition_type]
    _cfg = {
        "condition": modalities,
        "backbone": condition_cfg["image"]["backbone"],
        "depth_anything_v2_ckpt": condition_cfg["image"]["depth_anything_v2_ckpt"],
        "is_aug": condition_cfg["point_cloud"]["augment_probability"] > 0,
        "aug_points_prob": condition_cfg["point_cloud"]["augment_probability"],
        "point_encoder": condition_cfg["point_cloud"]["encoder"],
    }
    return ConditionExtractor(
        config=_cfg,
        projection_dim=condition_cfg["output_dim"],
    )
