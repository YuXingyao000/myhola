"""Prompt construction for FLUX.1-Kontext generation.

Two modes (selected via configs/datagen/prompt/*.yaml):
  fixed   — Single prompt for all models, pre_encode_text=True (fast)
  dynamic — Per-model random material/finish/process description, pre_encode_text=False (diverse)

Usage:
    from omegaconf import DictConfig
    prompt_text = build_prompt(cfg.prompt, model_id="00001234", seed=42)
    neg_text = get_negative(cfg.prompt)
    pre_enc = should_pre_encode(cfg.prompt)
"""
from __future__ import annotations

import random
import json

from omegaconf import DictConfig, OmegaConf


def decode_prompt_config(serialized: str | None) -> DictConfig | None:
    """Decode the complete prompt policy forwarded by the Hydra launcher."""
    if serialized is None:
        return None
    cfg = OmegaConf.create(json.loads(serialized))
    if not isinstance(cfg, DictConfig) or cfg.get("mode") not in {"fixed", "dynamic"}:
        raise ValueError("Prompt config mode must be fixed or dynamic.")
    if cfg.mode == "dynamic" and should_pre_encode(cfg):
        raise ValueError("Dynamic prompts require pre_encode_text=false.")
    # Validate required fields before loading weights.
    build_prompt(cfg, model_id="00000000")
    get_negative(cfg)
    return cfg


def build_prompt(prompt_cfg: DictConfig, model_id: str, seed: int = 42) -> str:
    """Build a positive prompt for a given model.

    In fixed mode: returns the static prompt (same for all models).
    In dynamic mode: composes material+finish+process+condition per model_id.
    """
    if prompt_cfg.mode == "fixed":
        return prompt_cfg.positive

    # Dynamic mode: deterministic random selection per model
    rng = random.Random(f"{seed}:{model_id}")
    description = ", ".join([
        rng.choice(list(prompt_cfg.materials)),
        rng.choice(list(prompt_cfg.finishes)),
        rng.choice(list(prompt_cfg.processes)),
        rng.choice(list(prompt_cfg.conditions)),
    ])
    return prompt_cfg.base_positive.replace("{material}", description)


def get_negative(prompt_cfg: DictConfig) -> str:
    """Return the negative prompt."""
    return prompt_cfg.negative


def should_pre_encode(prompt_cfg: DictConfig) -> bool:
    """Whether to pre-encode text embeddings (True for fixed, False for dynamic)."""
    return prompt_cfg.pre_encode_text
