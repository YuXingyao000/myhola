"""Compatibility entry for unique / novel metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.brepnet.eval.io import write_json
from src.brepnet.eval.metrics.uniqueness import *  # noqa: F401,F403
from src.brepnet.eval.metrics.uniqueness import evaluate_novel_with_nearest, evaluate_unique


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate unique and novel ratios")
    parser.add_argument("--fake_root", type=str, required=True)
    parser.add_argument("--fake_post", type=str, required=True)
    parser.add_argument("--train_root", type=str, default="")
    parser.add_argument("--n_bit", type=int, default=4)
    parser.add_argument("--atol", type=float, default=None)
    parser.add_argument("--use_ray", action="store_true")
    parser.add_argument("--min_face", type=int, default=None)
    parser.add_argument("--only_unique", action="store_true")
    args = parser.parse_args()

    unique_summary = evaluate_unique(
        args.fake_root,
        args.fake_post,
        n_bit=args.n_bit,
        atol=args.atol,
        use_ray=args.use_ray,
        min_face=args.min_face,
    )
    summaries = {"unique": unique_summary}
    if not args.only_unique:
        if not args.train_root:
            raise ValueError("--train_root is required unless --only_unique is set")
        summaries["novel"] = evaluate_novel_with_nearest(
            args.fake_root,
            args.fake_post,
            args.train_root,
            n_bit=args.n_bit,
            atol=args.atol,
            use_ray=args.use_ray,
        )
    write_json(Path(args.fake_root) / "eval_unique_novel_summary.json", summaries)
    print(summaries)


if __name__ == "__main__":
    main()
