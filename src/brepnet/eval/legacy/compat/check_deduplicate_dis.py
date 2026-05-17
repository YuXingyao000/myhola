"""Compatibility entry for generated duplicate graph checks."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.brepnet.eval.io import write_json
from src.brepnet.eval.metrics.uniqueness import *  # noqa: F401,F403
from src.brepnet.eval.metrics.uniqueness import evaluate_novel_with_nearest, evaluate_unique


def main() -> None:
    parser = argparse.ArgumentParser(description="Check generated duplicates")
    parser.add_argument("--fake", "--fake_root", dest="fake_root", type=str, required=True)
    parser.add_argument("--fake_post", type=str, default="")
    parser.add_argument("--train_root", type=str, default="")
    parser.add_argument("--n_bit", type=int, default=4)
    parser.add_argument("--atol", type=float, default=None)
    parser.add_argument("--use_ray", action="store_true")
    parser.add_argument("--compute_batch_size", type=int, default=200000)
    parser.add_argument("--only_unique", action="store_true")
    args = parser.parse_args()
    fake_post = args.fake_post or None
    summaries = {
        "unique": evaluate_unique(
            args.fake_root,
            fake_post,
            n_bit=args.n_bit,
            atol=args.atol,
            use_ray=args.use_ray,
            compute_batch_size=args.compute_batch_size,
        )
    }
    if args.train_root and fake_post and not args.only_unique:
        summaries["novel"] = evaluate_novel_with_nearest(
            args.fake_root,
            fake_post,
            args.train_root,
            n_bit=args.n_bit,
            atol=args.atol,
            use_ray=args.use_ray,
        )
    write_json(Path(args.fake_root) / "eval_deduplicate_summary.json", summaries)
    print(summaries)


if __name__ == "__main__":
    main()
