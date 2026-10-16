"""兼容旧导入；条件编码实现已迁至 :mod:`src.brepnet.diffusion`。"""

from src.brepnet.diffusion.conditioning import ConditionExtractor
from src.brepnet.diffusion.image_encoder import DINOv2ImageEncoder
from src.brepnet.diffusion.legacy_encoders import CameraEmbedding, TextEncoder
from src.brepnet.diffusion.point_encoder import PointNetEncoder

__all__ = [
    "CameraEmbedding",
    "ConditionExtractor",
    "DINOv2ImageEncoder",
    "PointNetEncoder",
    "TextEncoder",
]
