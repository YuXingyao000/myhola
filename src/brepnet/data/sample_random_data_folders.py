import argparse
import random
from pathlib import Path


DEFAULT_INPUT = Path("/mnt/d/python/src/brepnet/data/list/deduplicated_deepcad_testing_7_30.txt")
DEFAULT_OUTPUT = Path("/mnt/d/python/src/brepnet/data/list/deduplicated_deepcad_testing_7_30_sample_100.txt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Randomly sample data folder names from a text file."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Path to the source txt file. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Path to save the sampled folder names. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=100,
        help="Number of folder names to sample. Default: 100",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible sampling.",
    )
    return parser.parse_args()


def load_lines(file_path: Path) -> list[str]:
    with file_path.open("r", encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip()]


def main() -> None:
    args = parse_args()

    if args.count <= 0:
        raise ValueError("--count must be a positive integer.")
    if not args.input.exists():
        raise FileNotFoundError(f"Input file does not exist: {args.input}")

    lines = load_lines(args.input)
    if args.count > len(lines):
        raise ValueError(
            f"Requested {args.count} folders, but only found {len(lines)} entries in {args.input}."
        )

    rng = random.Random(args.seed)
    sampled_lines = rng.sample(lines, args.count)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        file.write("\n".join(sampled_lines))
        file.write("\n")

    print(f"Saved {len(sampled_lines)} sampled folder names to: {args.output}")


if __name__ == "__main__":
    main()
