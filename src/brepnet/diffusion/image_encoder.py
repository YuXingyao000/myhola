"""单视图图像编码：冻结 ViT 特征与可训练的 token 投影。"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class DINOv2ImageEncoder(nn.Module):
    """Frozen DINOv2 ViT-L backbone with a learnable projection head.

    Supports two backbone variants controlled by `backbone`:
      - "dinov2" (default): HuggingFace `facebook/dinov2-large`
      - "depth_anything_v2": Depth Anything V2 ViT-L encoder (fine-tuned DINOv2)

    For 224 x 224 images, both produce [B, 257, 1024] tokens
    (1 CLS + 16x16 patches). The learnable projection maps these to
    [B, 257, projection_dim]. Depth Anything uses only its ViT features;
    this branch does not produce a depth map.

    Parameters
    ----------
    projection_dim : int
        Output feature dimension (default 1024).
    backbone : str
        Which backbone to load: "dinov2" or "depth_anything_v2".
    depth_anything_v2_ckpt : str or None
        Path to Depth Anything V2 checkpoint (required if backbone="depth_anything_v2").
    """

    def __init__(
        self,
        projection_dim: int = 1024,
        backbone: str = "dinov2",
        depth_anything_v2_ckpt: Optional[str] = None,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.projection_dim = projection_dim

        # 按配置加载 backbone；第三方依赖只在使用时导入。
        if backbone == "depth_anything_v2":
            from thirdparty.Depth_Anything_V2.depth_anything_v2.dpt import DepthAnythingV2

            model_configs = {
                'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
                'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
                'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
                'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]},
            }
            encoder = 'vitl'
            _da_v2 = DepthAnythingV2(**model_configs[encoder])
            if depth_anything_v2_ckpt is None:
                depth_anything_v2_ckpt = (
                    f'thirdparty/Depth_Anything_V2/checkpoints/depth_anything_v2_{encoder}.pth'
                )
            _da_v2.load_state_dict(
                torch.load(depth_anything_v2_ckpt, map_location='cpu', weights_only=True)
            )
            self.img_model = _da_v2.pretrained
            del _da_v2
        elif backbone == "dinov2":
            from transformers import Dinov2Model
            self.img_model = Dinov2Model.from_pretrained('facebook/dinov2-large')
        else:
            raise ValueError(
                f"Unknown backbone '{backbone}'. Choose 'dinov2' or 'depth_anything_v2'."
            )

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
        """Extract image features and project them.

        Parameters
        ----------
        images : Tensor of shape [B, 3, 224, 224]
            Input images (normalized for DINOv2).

        Returns
        -------
        Tensor of shape [B, 257, projection_dim]
            Projected image features (1 CLS + 256 patch tokens).
        """
        with torch.no_grad():
            if self.backbone_name == "depth_anything_v2":
                backbone_out = self.img_model.forward_features(images)
                img_feature = torch.cat([
                    backbone_out["x_norm_clstoken"].unsqueeze(1),
                    backbone_out["x_norm_patchtokens"],
                ], dim=1)
            else:
                # HuggingFace Dinov2Model
                img_feature = self.img_model(images).last_hidden_state

        # img_feature: [B, 257, 1024]
        return self.projection(img_feature)
