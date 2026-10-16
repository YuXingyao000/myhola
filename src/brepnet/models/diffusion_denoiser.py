"""兼容旧导入路径；实现已迁移到 src.brepnet.diffusion。"""

from src.brepnet.diffusion.denoiser import BRepDenoiser
from src.brepnet.diffusion.fusion import (
    CrossAttentionConditionFuser,
    NoConditionFuser,
    build_condition_fuser,
)
from src.brepnet.diffusion.losses import clip_style_symmetric_infonce
from src.brepnet.diffusion.topology import (
    LearnedTopologyPredictor,
    LearnedTopologySelfAttentionMask,
    OracleTopologySelfAttentionMask,
    build_soft_topology_attention_bias,
    build_topology_self_attention,
)

__all__ = [
    "BRepDenoiser",
    "CrossAttentionConditionFuser",
    "NoConditionFuser",
    "build_condition_fuser",
    "clip_style_symmetric_infonce",
    "LearnedTopologyPredictor",
    "LearnedTopologySelfAttentionMask",
    "OracleTopologySelfAttentionMask",
    "build_soft_topology_attention_bias",
    "build_topology_self_attention",
]
