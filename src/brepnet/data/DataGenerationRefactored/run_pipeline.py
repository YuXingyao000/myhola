from __future__ import annotations

import argparse
import subprocess

from .config import PATHS, RUNTIME


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run common end-to-end generation flows.")
    parser.add_argument("flow", choices=("cube24", "single-view"))
    parser.add_argument("--num-gpus", type=int, default=RUNTIME.num_gpus)
    parser.add_argument("--skip-lists", action="store_true")
    parser.add_argument("--skip-blender", action="store_true")
    parser.add_argument("--skip-flux", action="store_true")
    parser.add_argument("--skip-pack", action="store_true")
    parser.add_argument("--max-models", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.skip_lists:
        run([RUNTIME.python, "-m", "src.brepnet.data.DataGenerationRefactored.run_lists", "--num-gpus", str(args.num_gpus)])

    if args.flow == "cube24":
        blender_mode = "cube24"
        flux_mode = "cube24-dynamic"
        pack_layout = "cube24"
    else:
        blender_mode = "single-view"
        flux_mode = "single-view"
        pack_layout = "single-view"

    if not args.skip_blender:
        cmd = [RUNTIME.python, "-m", "src.brepnet.data.DataGenerationRefactored.run_blender", blender_mode, "--num-gpus", str(args.num_gpus)]
        if args.max_models is not None:
            cmd.extend(["--max-models", str(args.max_models)])
        run(cmd)

    if not args.skip_flux:
        cmd = [RUNTIME.python, "-m", "src.brepnet.data.DataGenerationRefactored.run_flux", flux_mode, "--num-gpus", str(args.num_gpus)]
        if args.max_models is not None:
            cmd.extend(["--max-models", str(args.max_models)])
        run(cmd)

    if not args.skip_pack:
        run([RUNTIME.python, "-m", "src.brepnet.data.DataGenerationRefactored.run_pack", pack_layout])


def run(cmd: list[str]) -> None:
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
