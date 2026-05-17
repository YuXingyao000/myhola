"""Compatibility entry for dataset duplicate checks."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.brepnet.eval.io import write_json
from src.brepnet.eval.metrics.uniqueness import *  # noqa: F401,F403
from src.brepnet.eval.metrics.uniqueness import evaluate_unique


def main() -> None:
    parser = argparse.ArgumentParser(description="Check duplicates in one dataset")
    parser.add_argument("--train_root", type=str, required=True)
    parser.add_argument("--n_bit", type=int, default=4)
    parser.add_argument("--atol", type=float, default=None)
    parser.add_argument("--use_ray", action="store_true")
    parser.add_argument("--compute_batch_size", type=int, default=10000)
    parser.add_argument("--min_face", type=int, default=None)
    parser.add_argument("--txt", type=str, default=None)
    parser.add_argument("--load_batch_size", type=int, default=100)
    parser.add_argument("--num_cpus", type=int, default=32)
    args = parser.parse_args()
    del args.txt, args.load_batch_size, args.num_cpus
    summary = evaluate_unique(
        args.train_root,
        None,
        n_bit=args.n_bit,
        atol=args.atol,
        use_ray=args.use_ray,
        compute_batch_size=args.compute_batch_size,
        min_face=args.min_face,
    )
    write_json(Path(args.train_root).with_name(Path(args.train_root).name + "_deduplicate_summary.json"), summary)
    print(summary)


if __name__ == "__main__":
    main()
