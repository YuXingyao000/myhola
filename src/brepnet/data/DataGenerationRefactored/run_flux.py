from __future__ import annotations

import argparse
import re
from pathlib import Path

from .config import PATHS, RUNTIME
from .launcher import rank_env, run_parallel_ranked


MODEL_ID_PATTERN = re.compile(r"^\d{8}$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FLUX generation jobs without bash.")
    parser.add_argument("mode", choices=("single-view", "cube24-dynamic"))
    parser.add_argument("--num-gpus", type=int, default=RUNTIME.num_gpus)
    parser.add_argument("--render-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--max-models", type=int, default=None)
    parser.add_argument(
        "--model-list",
        type=Path,
        default=None,
        help="Generate this exact model list on the current machine. It is split into per-GPU local rank files.",
    )
    parser.add_argument(
        "--machine-list-dir",
        type=Path,
        default=None,
        help="Directory containing machine_{index}.txt files from split_model_lists.py.",
    )
    parser.add_argument("--machine-index", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    module_name = (
        "src.brepnet.data.DataGenerationRefactored.flux_scripts.generate_cube24_dynamic"
        if args.mode == "cube24-dynamic"
        else "src.brepnet.data.DataGenerationRefactored.flux_scripts.generate_single_view"
    )
    render_root = args.render_root or default_render_root(args.mode)
    output_root = args.output_root or default_output_root(args.mode)
    model_list = resolve_model_list(args)
    local_rank_lists = split_for_local_gpus(model_list, args.num_gpus, args.mode) if model_list else None

    commands = []
    for rank in range(args.num_gpus):
        cmd = [
            RUNTIME.python,
            "-m",
            module_name,
            "--render-root",
            str(render_root),
            "--output-root",
            str(output_root),
            "--transformer-path",
            str(PATHS.transformer_path),
            "--base-model-path",
            str(PATHS.base_model_path),
            "--num-steps",
            str(RUNTIME.num_steps),
            "--guidance-scale",
            str(RUNTIME.guidance_scale),
            "--true-cfg-scale",
            str(RUNTIME.true_cfg_scale),
            "--seed",
            str(RUNTIME.flux_seed),
        ]
        if local_rank_lists is not None:
            cmd.extend(["--model-list", str(local_rank_lists[rank])])
        else:
            cmd.extend(["--model-list-dir", str(PATHS.render_list_dir), "--rank", str(rank)])
        if RUNTIME.skip_existing:
            cmd.append("--skip-existing")
        if args.max_models is not None:
            cmd.extend(["--max-models", str(args.max_models)])

        log_path = PATHS.logs_root / "flux" / args.mode / f"rank_{rank}.log"
        commands.append((rank, cmd, rank_env(rank), log_path))

    output_root.mkdir(parents=True, exist_ok=True)
    run_parallel_ranked(commands)


def resolve_model_list(args: argparse.Namespace) -> Path | None:
    if args.model_list is not None and args.machine_list_dir is not None:
        raise ValueError("Use either --model-list or --machine-list-dir, not both.")
    if args.model_list is not None:
        return args.model_list.resolve()
    if args.machine_list_dir is None:
        return None
    if args.machine_index is None:
        raise ValueError("--machine-list-dir requires --machine-index.")
    if args.machine_index < 0:
        raise ValueError("--machine-index must be non-negative.")
    return (args.machine_list_dir / f"machine_{args.machine_index}.txt").resolve()


def split_for_local_gpus(model_list: Path, num_gpus: int, mode: str) -> list[Path]:
    if num_gpus <= 0:
        raise ValueError("--num-gpus must be positive.")

    model_ids = read_model_ids(model_list)
    output_dir = PATHS.project_root / "runtime_lists" / "local_gpu_flux" / mode / model_list.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    rank_lists: list[list[str]] = [[] for _ in range(num_gpus)]
    for idx, model_id in enumerate(model_ids):
        rank_lists[idx % num_gpus].append(model_id)

    output_paths = []
    for rank, ids in enumerate(rank_lists):
        path = output_dir / f"rank_{rank}.txt"
        path.write_text("\n".join(ids) + ("\n" if ids else ""), encoding="utf-8")
        output_paths.append(path)
    return output_paths


def read_model_ids(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Model list does not exist: {path}")

    ids: list[str] = []
    seen = set()
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        model_id = raw.strip()
        if not model_id or model_id.startswith("#"):
            continue
        if not MODEL_ID_PATTERN.match(model_id):
            raise ValueError(f"Invalid model id {model_id!r} in {path}:{line_number}; expected 8 digits.")
        if model_id in seen:
            continue
        seen.add(model_id)
        ids.append(model_id)
    return ids


def default_render_root(mode: str) -> Path:
    if mode == "cube24-dynamic":
        return PATHS.blender_cube24_out
    return PATHS.blender_single_view_out


def default_output_root(mode: str) -> Path:
    if mode == "cube24-dynamic":
        return PATHS.flux_cube24_out
    return PATHS.flux_single_view_out


if __name__ == "__main__":
    main()
