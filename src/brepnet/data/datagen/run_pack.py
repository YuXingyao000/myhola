"""Pack Blender/FLUX images into training-ready npz files.

Usage:
    python -m src.brepnet.data.datagen.run_pack
    python -m src.brepnet.data.datagen.run_pack mode=single-view
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

from .config import DATAGEN_ROOT, PROJECT_ROOT, resolve_path


@hydra.main(version_base="1.3", config_path="../../../../configs", config_name="datagen")
def main(cfg: DictConfig) -> None:
    mode = cfg.mode

    if mode == "cube24":
        blender_root = resolve_path(cfg.paths.blender_cube24_out)
        flux_root = resolve_path(cfg.paths.flux_cube24_out)
    else:
        blender_root = resolve_path(cfg.paths.blender_single_view_out)
        flux_root = resolve_path(cfg.paths.flux_single_view_out)

    target_root = Path(cfg.paths.cond_target)
    model_list = PROJECT_ROOT / cfg.paths.train_list

    if mode == "cube24":
        cmd = [
            sys.executable,
            "-m",
            "src.brepnet.data.datagen.tools.pack_natural_npz",
            "--render-root",
            str(blender_root),
            "--flux-root",
            str(flux_root),
            "--target-root",
            str(target_root),
            "--model-list",
            str(model_list),
            "--num-workers",
            str(cfg.flux.pack_workers),
            "--cube24",
        ]
    else:
        cmd = [
            sys.executable,
            "-m",
            "src.brepnet.data.datagen.tools.pack_single_view_npz",
            "--blender-root",
            str(blender_root),
            "--flux-root",
            str(flux_root),
            "--target-root",
            str(target_root),
            "--model-list",
            str(model_list),
            "--num-workers",
            str(cfg.flux.pack_workers),
        ]

    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
