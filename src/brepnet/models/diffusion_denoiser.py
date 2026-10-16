"""兼容旧导入路径；实现已迁移到 src.brepnet.diffusion。"""

from src.brepnet.diffusion.denoiser import BRepDenoiser
from src.brepnet.diffusion.fusion import ImageCrossAttention
from src.brepnet.diffusion.topology import (
    LearnedTopologyPredictor,
    LearnedTopologySelfAttentionMask,
    OracleTopologySelfAttentionMask,
    build_soft_topology_attention_bias,
    build_topology_self_attention,
)

__all__ = [
    "BRepDenoiser",
    "ImageCrossAttention",
    "LearnedTopologyPredictor",
    "LearnedTopologySelfAttentionMask",
    "OracleTopologySelfAttentionMask",
    "build_soft_topology_attention_bias",
    "build_topology_self_attention",
]
