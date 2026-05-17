"""Compatibility entry for LFD distribution reports."""

from __future__ import annotations

import argparse

from src.brepnet.eval.io import write_json
from src.brepnet.eval.metrics.lfd import summarize_lfd_pickle


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize and summarize LFD pickle")
    parser.add_argument("pkl", type=str)
    parser.add_argument("output_png", type=str)
    parser.add_argument("src_step_root", nargs="?", default=None)
    args = parser.parse_args()
    summary = summarize_lfd_pickle(args.pkl, output_png=args.output_png, src_step_root=args.src_step_root)
    write_json(args.output_png + ".json", summary)
    print(
        f"{summary['mean_lfd_resampled']} "
        f"{summary['median_lfd_resampled']} "
        f"{summary['p75_lfd_resampled']}"
    )


if __name__ == "__main__":
    main()
