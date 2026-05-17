"""Compatibility entry for point-cloud set metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.brepnet.eval.io import write_json
from src.brepnet.eval.metrics.point_cloud_set import *  # noqa: F401,F403
from src.brepnet.eval.metrics.point_cloud_set import evaluate_point_cloud_sets


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate point-cloud set metrics")
    parser.add_argument("--fake", type=str, required=True)
    parser.add_argument("--real", type=str, required=True)
    parser.add_argument("--n_test", type=int, default=1000)
    parser.add_argument("--multi", type=float, default=3)
    parser.add_argument("--times", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()
    result = evaluate_point_cloud_sets(
        args.fake,
        args.real,
        n_test=args.n_test,
        multi=args.multi,
        times=args.times,
        batch_size=args.batch_size,
    )
    write_json(Path(args.fake + "_results.json"), result)
    print(result)


if __name__ == "__main__":
    main()
