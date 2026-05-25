"""Blender render orchestrator with Hydra config.

Usage:
    python -m src.brepnet.data.datagen.run_blender
    python -m src.brepnet.data.datagen.run_blender mode=single-view
    python -m src.brepnet.data.datagen.run_blender blender.num_gpus=4 paths.model_root=/new/path
"""
from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig

from .config import DATAGEN_ROOT, PROJECT_ROOT, resolve_path
from .launcher import rank_env, run_parallel_ranked
from .utils import load_model_ids, split_for_gpus


@hydra.main(version_base="1.3", config_path="../../../../configs", config_name="datagen")
def main(cfg: DictConfig) -> None:
    mode = cfg.mode  # "cube24" or "single-view"
    num_gpus = cfg.blender.num_gpus

    # Resolve paths
    model_list_path = PROJECT_ROOT / cfg.paths.train_list
    model_root = Path(cfg.paths.model_root)
    materials_root = DATAGEN_ROOT / "materials"
    logs_root = resolve_path(cfg.paths.logs_root)

    if mode == "cube24":
        script_name = "render_cube24.py"
        output_root = resolve_path(cfg.paths.blender_cube24_out)
    else:
        script_name = "render_single_view.py"
        output_root = resolve_path(cfg.paths.blender_single_view_out)

    # Material
    material_subdir = cfg.blender.material_subdir

    # Load and split model list
    model_ids = load_model_ids(model_list_path)
    split_dir = DATAGEN_ROOT / "runtime_lists" / "blender" / mode
    local_rank_lists = split_for_gpus(model_ids, num_gpus, split_dir)

    # Build per-rank commands
    commands = []
    for rank in range(num_gpus):
        cmd = [
            str(cfg.blender.blender_bin),
            "--background",
            "--factory-startup",
            "--addons",
            "cycles",
            "--python",
            str(DATAGEN_ROOT / "blender_scripts" / script_name),
            "--",
            "--model-root",
            str(model_root),
            "--mesh-name",
            cfg.paths.mesh_name,
            "--materials-root",
            str(materials_root),
            "--output-root",
            str(output_root),
            "--device",
            cfg.blender.device,
            "--samples",
            str(cfg.blender.samples),
            "--png-compression",
            str(cfg.blender.png_compression),
            "--model-list",
            str(local_rank_lists[rank]),
        ]
        if cfg.blender.skip_existing:
            cmd.append("--skip-existing")
        if cfg.blender.require_gpu:
            cmd.append("--require-gpu")
        if material_subdir:
            cmd.extend(["--material-subdir", material_subdir])

        log_path = logs_root / "blender" / mode / f"rank_{rank}.log"
        commands.append((rank, cmd, rank_env(rank), log_path))

    output_root.mkdir(parents=True, exist_ok=True)
    run_parallel_ranked(commands)


if __name__ == "__main__":
    main()
