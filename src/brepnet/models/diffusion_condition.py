"""兼容旧导入；条件工厂已迁至 :mod:`src.brepnet.diffusion.conditioning`。"""

from src.brepnet.diffusion.conditioning import ConditionExtractor, build_condition_encoder

__all__ = ["ConditionExtractor", "build_condition_encoder"]
