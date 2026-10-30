"""
Img2Brep Model Registry & Factory Functions.

Provides centralized model construction through a small name-based registry.
"""

from __future__ import annotations

import importlib
import logging
from typing import Any, Dict

from omegaconf import OmegaConf

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy import helpers (avoid pulling heavy deps at module import time)
# ---------------------------------------------------------------------------

def _lazy_import(module_path: str, class_name: str):
    """Return a callable that lazily imports and returns the class."""

    def _load():
        mod = importlib.import_module(module_path)
        return getattr(mod, class_name)

    return _load


# ---------------------------------------------------------------------------
# Model Registry
# ---------------------------------------------------------------------------

_MODEL_LOADERS: Dict[str, Any] = {
    "AutoEncoder_uv_intersection": _lazy_import(
        "src.models.vae", "AutoEncoder_uv_intersection"
    ),
    "AutoEncoder_intersection": _lazy_import(
        "src.models.vae", "AutoEncoder_intersection"
    ),
    "AutoEncoder_joint": _lazy_import(
        "src.models.vae", "AutoEncoder_joint"
    ),
    "AutoEncoder": _lazy_import(
        "src.models.vae", "AutoEncoder"
    ),
    "AutoEncoder_light": _lazy_import(
        "src.models.vae", "AutoEncoder_light"
    ),
    "AutoEncoder_light_exp": _lazy_import(
        "src.models.vae", "AutoEncoder_light_exp"
    ),
    "Diffusion": _lazy_import(
        "src.models.diffusion.model", "Diffusion"
    ),
    "PreferenceDiffusion": _lazy_import(
        "src.models.diffusion.preference", "PreferenceDiffusion"
    ),
    "PhotoBaselineDiffusion": _lazy_import(
        "src.models.diffusion.photo_baseline", "PhotoBaselineDiffusion"
    ),
    "SpatialPhotoDiffusion": _lazy_import(
        "src.models.diffusion.spatial_model", "SpatialPhotoDiffusion"
    ),
}

# ---------------------------------------------------------------------------
# Factory: build_model
# ---------------------------------------------------------------------------

def build_model(model_cfg, condition_cfg=None) -> Any:
    model_cfg = OmegaConf.to_container(model_cfg, resolve=True)
    condition_cfg = OmegaConf.to_container(condition_cfg, resolve=True) if condition_cfg is not None else None
    name = model_cfg["name"]

    if name not in _MODEL_LOADERS:
        available = sorted(_MODEL_LOADERS.keys())
        raise KeyError(
            f"Unknown model '{name}'. Available: {available}"
        )

    model_cls = _MODEL_LOADERS[name]()
    logger.info("Building model '%s'", name)
    if name == "Diffusion":
        return model_cls(model_cfg, condition_cfg)
    return model_cls(model_cfg)

# ---------------------------------------------------------------------------
# Convenience: list available models
# ---------------------------------------------------------------------------

def available_models() -> list[str]:
    """Return sorted list of registered model names."""
    return sorted(_MODEL_LOADERS.keys())


def __getattr__(name: str) -> Any:
    """Lazy class exports, e.g. ``from src.models import AutoEncoder``."""
    if name in _MODEL_LOADERS:
        model_cls = _MODEL_LOADERS[name]()
        globals()[name] = model_cls
        return model_cls
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "build_model",
    "available_models",
    *_MODEL_LOADERS.keys(),
]
