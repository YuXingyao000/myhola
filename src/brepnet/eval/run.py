"""Unified evaluation entry point."""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from src.brepnet.eval.io import flatten_scalar_result, save_npz_result, write_csv, write_json, write_text
from src.brepnet.eval.parallel import run_tasks
from src.brepnet.eval.protocol import EvalSample, condition_result_path, error_path, iter_eval_samples


def parse_metrics(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"num_eval": len(rows)}
    keys = sorted({key for row in rows for key in row.keys() if key != "prefix"})
    for key in keys:
        values = [row[key] for row in rows if isinstance(row.get(key), (int, float, bool, np.generic))]
        if values:
            summary[f"{key}_mean"] = float(np.mean(values))
            summary[f"{key}_median"] = float(np.median(values))
    return summary


def run_condition_metric(args, samples: list[EvalSample]) -> dict[str, Any]:
    from src.brepnet.eval.metrics import condition

    jobs = [sample for sample in samples if args.from_scratch or condition.load_condition_result(sample.pred_dir) is None]

    def worker(sample: EvalSample):
        try:
            return condition.eval_one(
                args.pred_root,
                args.gt_root,
                sample.name,
                is_point2cad=args.is_point2cad,
                is_complexgen=args.is_complexgen,
                is_nvdnet=args.is_nvdnet,
                write_legacy_eval=args.write_legacy_eval,
            )
        except Exception:
            write_text(error_path(sample), traceback.format_exc())
            return None

    run_tasks(jobs, worker, use_ray=args.use_ray, num_cpus=args.num_cpus, timeout=args.timeout, desc="Condition")
    summary = condition.compute_statistics(args.pred_root, args.only_valid, args.split_list)
    rows = []
    for sample in samples:
        result = condition.load_condition_result(sample.pred_dir)
        if result is not None:
            rows.append(flatten_scalar_result(sample.name, result))
    write_csv(Path(args.pred_root) / "eval_condition.csv", rows)
    write_json(Path(args.pred_root) / "eval_condition_summary.json", summary)
    return {"condition": summary}


def run_validity_metric(args, samples: list[EvalSample]) -> dict[str, Any]:
    from src.brepnet.eval.metrics import validity

    results = run_tasks(samples, validity.evaluate_sample, use_ray=args.use_ray, num_cpus=args.num_cpus, desc="Validity")
    rows = []
    for sample, result in zip(samples, results):
        if result is None:
            continue
        save_npz_result(sample.pred_dir / "eval_validity.npz", result)
        rows.append(flatten_scalar_result(sample.name, result))
    summary = summarize_rows(rows)
    write_csv(Path(args.pred_root) / "eval_validity.csv", rows)
    write_json(Path(args.pred_root) / "eval_validity_summary.json", summary)
    return {"validity": summary}


def run_complexity_metric(args, samples: list[EvalSample]) -> dict[str, Any]:
    from src.brepnet.eval.metrics import complexity

    results = run_tasks(samples, complexity.evaluate_sample, use_ray=args.use_ray, num_cpus=args.num_cpus, desc="Complexity")
    rows = []
    for sample, result in zip(samples, results):
        if result is None:
            continue
        save_npz_result(sample.pred_dir / "eval_complexity.npz", result)
        rows.append(flatten_scalar_result(sample.name, result))
    summary = summarize_rows(rows)
    write_csv(Path(args.pred_root) / "eval_complexity.csv", rows)
    write_json(Path(args.pred_root) / "eval_complexity_summary.json", summary)
    return {"complexity": summary}


def run_unique_metric(args) -> dict[str, Any]:
    from src.brepnet.eval.metrics import uniqueness

    summary = uniqueness.evaluate_unique(
        args.pred_root,
        args.pred_post_root or args.pred_root,
        n_bit=args.n_bit,
        atol=args.atol,
        use_ray=args.use_ray,
        min_face=args.min_face,
    )
    write_json(Path(args.pred_root) / "eval_unique_summary.json", summary)
    return {"unique": summary}


def run_lfd_metric(args) -> dict[str, Any]:
    from src.brepnet.eval.metrics import lfd

    if not args.lfd_pkl:
        raise ValueError("--lfd-pkl is required when metrics includes lfd")
    report_dir = Path(args.pred_root) / "reports"
    summary = lfd.summarize_lfd_pickle(
        args.lfd_pkl,
        output_png=report_dir / "lfd_distribution.png",
        src_step_root=args.pred_root if args.lfd_valid_only else None,
    )
    write_json(Path(args.pred_root) / "eval_lfd_summary.json", summary)
    return {"lfd": summary}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate BREP generation results")
    parser.add_argument("--pred-root", "--eval_root", dest="pred_root", required=True)
    parser.add_argument("--gt-root", "--gt_root", dest="gt_root", default="")
    parser.add_argument("--split-list", "--list", dest="split_list", default="")
    parser.add_argument("--metrics", default="condition", help="Comma list: condition,validity,complexity,unique,lfd")
    parser.add_argument("--sample", "--prefix", dest="sample", default="")
    parser.add_argument("--from-scratch", action="store_true")
    parser.add_argument("--use-ray", "--use_ray", dest="use_ray", action="store_true")
    parser.add_argument("--num-cpus", "--num_cpus", dest="num_cpus", type=int, default=16)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--only-valid", "--only_valid", dest="only_valid", action="store_true")

    parser.add_argument("--write-legacy-eval", action="store_true")

    parser.add_argument("--is-point2cad", "--is_point2cad", dest="is_point2cad", action="store_true")
    parser.add_argument("--is-complexgen", "--is_complexgen", dest="is_complexgen", action="store_true")
    parser.add_argument("--is-nvdnet", "--is_nvdnet", dest="is_nvdnet", action="store_true")

    parser.add_argument("--pred-post-root", default="")
    parser.add_argument("--n-bit", "--n_bit", dest="n_bit", type=int, default=4)
    parser.add_argument("--atol", type=float, default=None)
    parser.add_argument("--min-face", "--min_face", dest="min_face", type=int, default=None)

    parser.add_argument("--lfd-pkl", default="")
    parser.add_argument("--lfd-valid-only", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    metric_names = parse_metrics(args.metrics)
    samples = iter_eval_samples(args.pred_root, args.gt_root or None, split_list=args.split_list, sample=args.sample or None)
    if not samples and any(metric in metric_names for metric in ("condition", "validity", "complexity")):
        raise ValueError(f"No samples found under {args.pred_root}")

    summaries: dict[str, Any] = {}
    for metric in metric_names:
        if metric == "condition":
            if not args.gt_root:
                raise ValueError("--gt-root is required for condition metric")
            summaries.update(run_condition_metric(args, samples))
        elif metric == "validity":
            summaries.update(run_validity_metric(args, samples))
        elif metric == "complexity":
            summaries.update(run_complexity_metric(args, samples))
        elif metric == "unique":
            summaries.update(run_unique_metric(args))
        elif metric == "lfd":
            summaries.update(run_lfd_metric(args))
        else:
            raise ValueError(f"Unknown metric: {metric}")
    write_json(Path(args.pred_root) / "eval_summary.json", summaries)


if __name__ == "__main__":
    main()
