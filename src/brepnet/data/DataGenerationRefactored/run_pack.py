from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from .config import PATHS, RUNTIME


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pack Blender/FLUX images into natural.npz.")
    parser.add_argument("layout", choices=("single-view", "cube24"))
    parser.add_argument("--blender-root", type=Path, default=None)
    parser.add_argument("--flux-root", type=Path, default=None)
    parser.add_argument("--target-root", type=Path, default=None)
    parser.add_argument("--model-list", type=Path, default=None)
    parser.add_argument("--num-workers", type=int, default=RUNTIME.pack_workers)
    parser.add_argument("--overwrite", action="store_true", default=RUNTIME.pack_overwrite)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    blender_root = args.blender_root or default_blender_root(args.layout)
    flux_root = args.flux_root or default_flux_root(args.layout)
    target_root = args.target_root or default_target_root(args.layout)
    model_list = args.model_list or default_model_list()

    cmd = [
        RUNTIME.python,
        "-m",
        "src.brepnet.data.DataGenerationRefactored.tools.pack_natural_npz",
        "--render-root",
        str(blender_root),
        "--flux-root",
        str(flux_root),
        "--target-root",
        str(target_root),
        "--model-list",
        str(model_list),
        "--num-workers",
        str(args.num_workers),
    ]
    if args.layout == "cube24":
        cmd.append("--cube24")
    if args.overwrite:
        cmd.append("--overwrite")

    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def default_blender_root(layout: str) -> Path:
    if layout == "cube24":
        return PATHS.blender_cube24_out
    return PATHS.blender_single_view_out


def default_flux_root(layout: str) -> Path:
    if layout == "cube24":
        return PATHS.flux_cube24_out
    return PATHS.flux_single_view_out


def default_target_root(layout: str) -> Path:
    if layout == "cube24":
        return PATHS.cond_cube24_target
    return PATHS.cond_single_view_target


def default_model_list() -> Path:
    preferred = PATHS.test_list_dir / "model_list.txt"
    if preferred.is_file():
        return preferred
    return PATHS.render_list_dir / "all.txt"


if __name__ == "__main__":
    main()
