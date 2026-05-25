"""FLUX generation orchestrator with Hydra config.

Usage:
    python -m src.brepnet.data.datagen.run_flux
    python -m src.brepnet.data.datagen.run_flux mode=single-view
    python -m src.brepnet.data.datagen.run_flux datagen/prompt=fixed flux.num_steps=28
"""
from __future__ import annotations

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

from .config import DATAGEN_ROOT, PROJECT_ROOT, resolve_path
from .launcher import rank_env, run_parallel_ranked
from .prompts import build_prompt, get_negative, should_pre_encode
from .utils import load_model_ids, split_for_gpus


@hydra.main(version_base="1.3", config_path="../../../../configs", config_name="datagen")
def main(cfg: DictConfig) -> None:
    mode = cfg.mode  # "cube24" or "single-view"
    num_gpus = cfg.flux.num_gpus

    # Resolve paths
    model_list_path = PROJECT_ROOT / cfg.paths.train_list
    transformer_path = Path(cfg.paths.transformer_path)
    base_model_path = Path(cfg.paths.base_model_path)
    logs_root = resolve_path(cfg.paths.logs_root)

    if mode == "cube24":
        module_name = "src.brepnet.data.datagen.flux_scripts.generate_cube24_dynamic"
        render_root = resolve_path(cfg.paths.blender_cube24_out)
        output_root = resolve_path(cfg.paths.flux_cube24_out)
    else:
        module_name = "src.brepnet.data.datagen.flux_scripts.generate_single_view"
        render_root = resolve_path(cfg.paths.blender_single_view_out)
        output_root = resolve_path(cfg.paths.flux_single_view_out)

    # Load and split model list
    model_ids = load_model_ids(model_list_path)
    split_dir = DATAGEN_ROOT / "runtime_lists" / "flux" / mode
    local_rank_lists = split_for_gpus(model_ids, num_gpus, split_dir)

    # Prompt config
    prompt_cfg = cfg.prompt
    neg_prompt = get_negative(prompt_cfg)
    pre_encode = should_pre_encode(prompt_cfg)

    # For fixed mode: single prompt for all models
    # For dynamic mode: the worker script handles per-model prompt building
    base_prompt = prompt_cfg.positive if prompt_cfg.mode == "fixed" else prompt_cfg.base_positive

    # Build per-rank commands
    python_exe = sys.executable
    commands = []
    for rank in range(num_gpus):
        cmd = [
            python_exe,
            "-m",
            module_name,
            "--render-root",
            str(render_root),
            "--output-root",
            str(output_root),
            "--transformer-path",
            str(transformer_path),
            "--base-model-path",
            str(base_model_path),
            "--num-steps",
            str(cfg.flux.num_steps),
            "--guidance-scale",
            str(cfg.flux.guidance_scale),
            "--true-cfg-scale",
            str(cfg.flux.true_cfg_scale),
            "--seed",
            str(cfg.flux.seed),
            "--model-list",
            str(local_rank_lists[rank]),
            "--prompt-base",
            base_prompt,
            "--neg-prompt",
            neg_prompt,
        ]
        if cfg.flux.skip_existing:
            cmd.append("--skip-existing")

        log_path = logs_root / "flux" / mode / f"rank_{rank}.log"
        commands.append((rank, cmd, rank_env(rank), log_path))

    output_root.mkdir(parents=True, exist_ok=True)
    run_parallel_ranked(commands)


if __name__ == "__main__":
    main()
