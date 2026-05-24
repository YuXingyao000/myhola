from __future__ import annotations

import argparse
import subprocess

from .config import LIST_SAMPLING, PATHS, RUNTIME


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample model ids and split them into rank lists.")
    parser.add_argument("--skip-sample", action="store_true", help="Only rebuild render rank lists.")
    parser.add_argument("--num-gpus", type=int, default=RUNTIME.num_gpus)
    parser.add_argument("--seed", type=int, default=RUNTIME.list_seed)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.skip_sample:
        subprocess.run(
            [
                RUNTIME.python,
                "-m",
                "src.brepnet.data.DataGenerationRefactored.tools.sample_lists",
                "--source_list_dir",
                str(PATHS.list_dir),
                "--output_dir",
                str(PATHS.test_list_dir),
                "--seed",
                str(args.seed),
                "--num_train",
                str(LIST_SAMPLING.num_train),
                "--num_val",
                str(LIST_SAMPLING.num_val),
                "--num_test",
                str(LIST_SAMPLING.num_test),
            ],
            check=True,
        )

    subprocess.run(
        [
            RUNTIME.python,
            "-m",
            "src.brepnet.data.DataGenerationRefactored.tools.prepare_render_lists",
            "--list-dir",
            str(PATHS.test_list_dir),
            "--output-dir",
            str(PATHS.render_list_dir),
            "--num-gpus",
            str(args.num_gpus),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
