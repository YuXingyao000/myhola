"""End-to-end data generation pipeline: Blender → FLUX → Pack.

Usage:
    python -m src.brepnet.data.datagen.run_pipeline
    python -m src.brepnet.data.datagen.run_pipeline mode=single-view
    python -m src.brepnet.data.datagen.run_pipeline +skip_blender=true +skip_flux=true
"""
from __future__ import annotations

import subprocess
import sys

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base="1.3", config_path="../../../../configs", config_name="datagen")
def main(cfg: DictConfig) -> None:
    skip_blender = cfg.get("skip_blender", False)
    skip_flux = cfg.get("skip_flux", False)
    skip_pack = cfg.get("skip_pack", False)

    # Pass the full resolved config as CLI overrides to sub-commands
    # This ensures all path/param overrides propagate
    overrides = _build_overrides(cfg)

    if not skip_blender:
        print("=== Stage 1: Blender Render ===")
        _run_module("src.brepnet.data.datagen.run_blender", overrides)

    if not skip_flux:
        print("=== Stage 2: FLUX Generation ===")
        _run_module("src.brepnet.data.datagen.run_flux", overrides)

    if not skip_pack:
        print("=== Stage 3: Pack NPZ ===")
        _run_module("src.brepnet.data.datagen.run_pack", overrides)


def _build_overrides(cfg: DictConfig) -> list[str]:
    """Extract key overrides to pass to sub-commands."""
    overrides = []
    # Pass mode
    overrides.append(f"mode={cfg.mode}")
    # Pass path overrides if they differ from defaults
    for key in ("model_root", "transformer_path", "base_model_path", "cond_target"):
        val = OmegaConf.select(cfg, f"paths.{key}")
        if val is not None:
            overrides.append(f"paths.{key}={val}")
    return overrides


def _run_module(module: str, overrides: list[str]) -> None:
    cmd = [sys.executable, "-m", module] + overrides
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
