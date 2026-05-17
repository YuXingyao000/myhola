import argparse
import subprocess
import sys
from pathlib import Path

from src.brepnet.experiments.condition_debug.common import prepare_output_dir, safe_mkdir, write_run_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run post-processing and conditional B-Rep eval.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--post-root", required=True)
    parser.add_argument("--gt-root", required=True)
    parser.add_argument("--list", default="")
    parser.add_argument("--num-cpus", type=int, default=32)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--use-cuda", action="store_true")
    parser.add_argument("--skip-construct", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def run_command(command: list[str], log_path: Path) -> None:
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write("+ " + " ".join(command) + "\n")
        log_file.flush()
        subprocess.run(command, check=True, stdout=log_file, stderr=subprocess.STDOUT)


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    post_root = Path(args.post_root)
    if not data_root.exists():
        raise FileNotFoundError(f"Missing data root: {data_root}")
    if not Path(args.gt_root).exists():
        raise FileNotFoundError(f"Missing GT root: {args.gt_root}")
    if args.list and not Path(args.list).exists():
        raise FileNotFoundError(f"Missing list file: {args.list}")

    if not args.skip_construct:
        prepare_output_dir(post_root, force=args.force)
    else:
        safe_mkdir(post_root)
    metadata_root = safe_mkdir(post_root / "_metadata")
    write_run_metadata(metadata_root / "post_eval.json", args)
    log_path = metadata_root / "post_eval.log"

    if not args.skip_construct:
        command = [
            sys.executable,
            "-m",
            "src.brepnet.post.construct_brep",
            "--data_root",
            str(data_root),
            "--out_root",
            str(post_root),
            "--use_ray",
            "--num_cpus",
            str(args.num_cpus),
            "--from_scratch",
            "--timeout",
            str(args.timeout),
        ]
        if args.list:
            command.extend(["--list", args.list])
        if args.use_cuda:
            command.append("--use_cuda")
        run_command(command, log_path)

    if not args.skip_eval:
        command = [
            sys.executable,
            "-m",
            "src.brepnet.eval.run",
            "--pred-root",
            str(post_root),
            "--gt-root",
            args.gt_root,
            "--metrics",
            "condition",
            "--use-ray",
            "--num-cpus",
            str(args.num_cpus),
            "--from-scratch",
        ]
        if args.list:
            command.extend(["--split-list", args.list])
        run_command(command, log_path)


if __name__ == "__main__":
    main()
