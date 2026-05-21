"""
HoLa-BRep Model Registry & Factory Functions.

Provides centralized model construction via:
  - build_model(cfg)              : Instantiate encoder/decoder models
  - build_strategy(cfg)           : Instantiate training strategies
  - build_condition_extractor(cfg): Instantiate condition extraction modules

Supports both Hydra `_target_` instantiation and legacy name-based lookup
through MODEL_REGISTRY / STRATEGY_REGISTRY dictionaries.
"""

from __future__ import annotations

import importlib
import logging
from typing import Any, Dict, Optional

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
    "AutoEncoder_1119": _lazy_import(
        "src.brepnet.models.vae", "AutoEncoder_1119"
    ),
    "AutoEncoder_1119_light": _lazy_import(
        "src.brepnet.models.vae", "AutoEncoder_1119_light"
    ),
    "AutoEncoder_1119_TokenVAE": _lazy_import(
        "src.brepnet.models.vae", "AutoEncoder_1119_TokenVAE"
    ),
    "AutoEncoder_1119_TokenEncConvDecVAE": _lazy_import(
        "src.brepnet.models.vae", "AutoEncoder_1119_TokenEncConvDecVAE"
    ),
    "AutoEncoder_1225": _lazy_import(
        "src.brepnet.models.vae", "AutoEncoder_1225"
    ),
    "DiffusionCrossAttn": _lazy_import(
        "src.brepnet.models.diffusion", "DiffusionCrossAttn"
    ),
    "DiffusionConcat": _lazy_import(
        "src.brepnet.models.diffusion", "DiffusionConcat"
    ),
}

# Backward-compatible aliases
_MODEL_ALIASES: Dict[str, str] = {
    "AutoEncoder_0925": "AutoEncoder_1119",
    "AutoEncoder_featuredv2": "AutoEncoder_1119",
    "AutoEncoder_1119_Light": "AutoEncoder_1119_light",
    "Diffusion_condition": "DiffusionCrossAttn",
    "Diffusion_condition_mm": "DiffusionConcat",
}


def _resolve_model_name(name: str) -> str:
    """Resolve aliases to canonical model names."""
    canonical = _MODEL_ALIASES.get(name, name)
    if canonical != name:
        logger.debug("Resolved model alias '%s' -> '%s'", name, canonical)
    return canonical


# Public registry (populated on first access to keep import lightweight)
MODEL_REGISTRY: Dict[str, Any] = {}


def _ensure_model_registry():
    """Populate MODEL_REGISTRY if not already done."""
    if MODEL_REGISTRY:
        return
    for name, loader in _MODEL_LOADERS.items():
        MODEL_REGISTRY[name] = loader
    # Register aliases pointing to the same loader
    for alias, canonical in _MODEL_ALIASES.items():
        MODEL_REGISTRY[alias] = _MODEL_LOADERS[canonical]


# ---------------------------------------------------------------------------
# Strategy Registry
# ---------------------------------------------------------------------------

_STRATEGY_LOADERS: Dict[str, Any] = {
    "VAEStrategy": _lazy_import(
        "brepnet.models.strategies", "VAEStrategy"
    ),
    "DiffusionStrategy": _lazy_import(
        "brepnet.models.strategies", "DiffusionStrategy"
    ),
    "GuidedDiffusionStrategy": _lazy_import(
        "brepnet.models.strategies", "GuidedDiffusionStrategy"
    ),
    "KnowledgeDistillation": _lazy_import(
        "src.brepnet.models.strategies", "KnowledgeDistillation"
    ),
    "FeatureDomainMapper": _lazy_import(
        "src.brepnet.models.strategies", "FeatureDomainMapper"
    ),
    "TwoStagePipeline": _lazy_import(
        "src.brepnet.models.strategies", "TwoStagePipeline"
    ),
}

STRATEGY_REGISTRY: Dict[str, Any] = {}


def _ensure_strategy_registry():
    """Populate STRATEGY_REGISTRY if not already done."""
    if STRATEGY_REGISTRY:
        return
    for name, loader in _STRATEGY_LOADERS.items():
        STRATEGY_REGISTRY[name] = loader


# ---------------------------------------------------------------------------
# Factory: build_model
# ---------------------------------------------------------------------------

def build_model(cfg) -> Any:
    """
    Build a model from a configuration object.

    Supports two modes:
      1. Hydra-style: cfg contains a `_target_` key for full class path resolution.
      2. Legacy name-based: cfg contains a `name` key looked up in MODEL_REGISTRY.

    Parameters
    ----------
    cfg : OmegaConf DictConfig or dict-like
        Model configuration. Must have either `_target_` or `name`.

    Returns
    -------
    nn.Module
        The instantiated model.
    """
    # Hydra _target_ instantiation
    target = getattr(cfg, "_target_", None) or (
        cfg.get("_target_") if isinstance(cfg, dict) else None
    )
    if target is not None:
        logger.info("Instantiating model via Hydra _target_: %s", target)
        from hydra.utils import instantiate
        return instantiate(cfg)

    # Legacy name-based lookup
    name = getattr(cfg, "name", None) or (
        cfg.get("name") if isinstance(cfg, dict) else None
    )
    if name is None:
        raise ValueError(
            "Model config must specify either '_target_' (Hydra) or 'name' (legacy). "
            f"Got keys: {list(cfg.keys()) if isinstance(cfg, dict) else dir(cfg)}"
        )

    canonical_name = _resolve_model_name(name)
    _ensure_model_registry()

    if canonical_name not in MODEL_REGISTRY:
        available = sorted(set(list(MODEL_REGISTRY.keys())))
        raise KeyError(
            f"Unknown model '{name}' (resolved: '{canonical_name}'). "
            f"Available: {available}"
        )

    model_cls = MODEL_REGISTRY[canonical_name]
    # If it's a lazy loader, call it to get the actual class
    if callable(model_cls) and not isinstance(model_cls, type):
        model_cls = model_cls()
        MODEL_REGISTRY[canonical_name] = model_cls

    # Extract constructor kwargs (exclude meta keys)
    meta_keys = {"name", "_target_", "_recursive_", "_convert_"}
    if isinstance(cfg, dict):
        kwargs = {k: v for k, v in cfg.items() if k not in meta_keys}
    else:
        from omegaconf import OmegaConf
        kwargs = {
            k: v
            for k, v in OmegaConf.to_container(cfg, resolve=True).items()
            if k not in meta_keys
        }

    logger.info("Building model '%s' with %d config keys", canonical_name, len(kwargs))
    import inspect

    init_params = list(inspect.signature(model_cls.__init__).parameters.values())[1:]
    if (
        len(init_params) == 1
        and init_params[0].kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
        and init_params[0].name in {"cfg", "conf", "v_conf"}
    ):
        return model_cls(kwargs)
    return model_cls(**kwargs)


# ---------------------------------------------------------------------------
# Factory: build_strategy
# ---------------------------------------------------------------------------

def build_strategy(cfg) -> Optional[Any]:
    """
    Build a training strategy from configuration, or return None if not specified.

    Parameters
    ----------
    cfg : OmegaConf DictConfig, dict, or None
        Strategy configuration. If None or empty, returns None.

    Returns
    -------
    Strategy instance or None
    """
    if cfg is None:
        return None

    # Hydra _target_ instantiation
    target = getattr(cfg, "_target_", None) or (
        cfg.get("_target_") if isinstance(cfg, dict) else None
    )
    if target is not None:
        logger.info("Instantiating strategy via Hydra _target_: %s", target)
        from hydra.utils import instantiate
        return instantiate(cfg)

    # Legacy name-based lookup
    name = getattr(cfg, "name", None) or (
        cfg.get("name") if isinstance(cfg, dict) else None
    )
    if name is None:
        return None

    _ensure_strategy_registry()

    if name not in STRATEGY_REGISTRY:
        available = sorted(STRATEGY_REGISTRY.keys())
        raise KeyError(
            f"Unknown strategy '{name}'. Available: {available}"
        )

    strategy_cls = STRATEGY_REGISTRY[name]
    if callable(strategy_cls) and not isinstance(strategy_cls, type):
        strategy_cls = strategy_cls()
        STRATEGY_REGISTRY[name] = strategy_cls

    meta_keys = {"name", "_target_", "_recursive_", "_convert_"}
    if isinstance(cfg, dict):
        kwargs = {k: v for k, v in cfg.items() if k not in meta_keys}
    else:
        from omegaconf import OmegaConf
        kwargs = {
            k: v
            for k, v in OmegaConf.to_container(cfg, resolve=True).items()
            if k not in meta_keys
        }

    logger.info("Building strategy '%s'", name)
    return strategy_cls(**kwargs)


# ---------------------------------------------------------------------------
# Factory: build_condition_extractor
# ---------------------------------------------------------------------------

def build_condition_extractor(cfg) -> Optional[Any]:
    """
    Build a condition extraction module from configuration.

    Condition extractors transform raw conditioning inputs (text, images, point
    clouds, etc.) into latent representations consumed by the diffusion model.

    Parameters
    ----------
    cfg : OmegaConf DictConfig, dict, or None
        Condition extractor configuration. If None, returns None.

    Returns
    -------
    nn.Module or None
    """
    if cfg is None:
        return None

    # Hydra _target_ instantiation (preferred)
    target = getattr(cfg, "_target_", None) or (
        cfg.get("_target_") if isinstance(cfg, dict) else None
    )
    if target is not None:
        logger.info(
            "Instantiating condition extractor via Hydra _target_: %s", target
        )
        from hydra.utils import instantiate
        return instantiate(cfg)

    # Legacy name-based: import from condition_extractors submodule
    name = getattr(cfg, "name", None) or (
        cfg.get("name") if isinstance(cfg, dict) else None
    )
    if name is None:
        return None

    logger.info("Building condition extractor '%s'", name)
    try:
        mod = importlib.import_module("brepnet.models.condition_extractors")
        extractor_cls = getattr(mod, name)
    except (ImportError, AttributeError) as e:
        raise ImportError(
            f"Could not load condition extractor '{name}' from "
            f"brepnet.models.condition_extractors: {e}"
        ) from e

    meta_keys = {"name", "_target_", "_recursive_", "_convert_"}
    if isinstance(cfg, dict):
        kwargs = {k: v for k, v in cfg.items() if k not in meta_keys}
    else:
        from omegaconf import OmegaConf
        kwargs = {
            k: v
            for k, v in OmegaConf.to_container(cfg, resolve=True).items()
            if k not in meta_keys
        }

    return extractor_cls(**kwargs)


# ---------------------------------------------------------------------------
# Convenience: list available models/strategies
# ---------------------------------------------------------------------------

def available_models() -> list[str]:
    """Return sorted list of all registered model names (including aliases)."""
    _ensure_model_registry()
    return sorted(MODEL_REGISTRY.keys())


def available_strategies() -> list[str]:
    """Return sorted list of all registered strategy names."""
    _ensure_strategy_registry()
    return sorted(STRATEGY_REGISTRY.keys())


def __getattr__(name: str) -> Any:
    """Lazy class exports, e.g. ``from src.brepnet.models import AutoEncoder_1119``."""
    canonical_name = _resolve_model_name(name)
    if canonical_name in _MODEL_LOADERS:
        model_cls = _MODEL_LOADERS[canonical_name]()
        globals()[name] = model_cls
        if canonical_name == name:
            MODEL_REGISTRY[canonical_name] = model_cls
        return model_cls
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "build_model",
    "build_strategy",
    "build_condition_extractor",
    "available_models",
    "available_strategies",
    *_MODEL_LOADERS.keys(),
    *_MODEL_ALIASES.keys(),
]
