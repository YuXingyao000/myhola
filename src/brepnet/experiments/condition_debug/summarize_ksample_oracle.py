import argparse
from pathlib import Path
from typing import Any

import numpy as np

from src.brepnet.experiments.condition_debug.common import (
    finite_values,
    load_eval_result,
    mean_or_none,
    median_or_none,
    read_prefixes,
    write_csv,
    write_json,
    write_run_metadata,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize K-sampling oracle results.")
    parser.add_argument("--baseline-post-root", required=True)
    parser.add_argument("--ksample-post-root", required=True)
    parser.add_argument("--list", required=True)
    parser.add_argument("--num-samples", type=int, required=True)
    parser.add_argument("--layout", choices=["flat", "sample-subdirs"], default="sample-subdirs")
    parser.add_argument("--suffix-template", default="__s{sample:02d}")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    return parser.parse_args()


def best_by_metric(candidates: list[dict[str, Any]], metric: str, require_success: bool = False) -> dict[str, Any] | None:
    valid = []
    for candidate in candidates:
        if candidate.get("eval_missing"):
            continue
        if require_success and not candidate.get("is_success"):
            continue
        value = candidate.get(metric)
        try:
            f_value = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(f_value):
            valid.append((f_value, candidate))
    if not valid:
        return None
    valid.sort(key=lambda item: item[0])
    return valid[0][1]


def main() -> None:
    args = parse_args()
    prefixes = read_prefixes(args.list)
    rows = []
    for prefix in prefixes:
        baseline = load_eval_result(args.baseline_post_root, prefix)
        baseline_missing = baseline is None
        baseline = baseline or {"prefix": prefix, "is_success": False, "eval_missing": True}

        candidates = []
        for sample_idx in range(args.num_samples):
            if args.layout == "sample-subdirs":
                candidate_root = Path(args.ksample_post_root) / f"s{sample_idx:02d}"
                candidate_prefix = prefix
            else:
                candidate_root = Path(args.ksample_post_root)
                suffix = args.suffix_template.format(sample=sample_idx)
                candidate_prefix = f"{prefix}{suffix}"
            candidate = load_eval_result(candidate_root, candidate_prefix)
            if candidate is None:
                candidate = {
                    "prefix": candidate_prefix,
                    "candidate_root": str(candidate_root),
                    "candidate_sample": sample_idx,
                    "is_success": False,
                    "eval_missing": True,
                }
            else:
                candidate["candidate_root"] = str(candidate_root)
                candidate["candidate_sample"] = sample_idx
                candidate["eval_missing"] = False
            candidates.append(candidate)

        best = best_by_metric(candidates, "face_cd", require_success=False)
        best_valid = best_by_metric(candidates, "face_cd", require_success=True)
        num_valid = sum(bool(item.get("is_success")) for item in candidates)
        num_eval_missing = sum(bool(item.get("eval_missing")) for item in candidates)

        row = {
            "prefix": prefix,
            "baseline_eval_missing": baseline_missing,
            "baseline_is_success": bool(baseline.get("is_success")),
            "baseline_face_cd": baseline.get("face_cd"),
            "baseline_edge_cd": baseline.get("edge_cd"),
            "baseline_num_recon_face": baseline.get("num_recon_face"),
            "baseline_num_gt_face": baseline.get("num_gt_face"),
            "num_candidates": args.num_samples,
            "num_valid_candidates": num_valid,
            "num_eval_missing_candidates": num_eval_missing,
            "any_valid": num_valid > 0,
            "best_prefix": best.get("prefix") if best else None,
            "best_sample": best.get("candidate_sample") if best else None,
            "best_face_cd": best.get("face_cd") if best else None,
            "best_edge_cd": best.get("edge_cd") if best else None,
            "best_is_success": bool(best.get("is_success")) if best else False,
            "best_valid_prefix": best_valid.get("prefix") if best_valid else None,
            "best_valid_sample": best_valid.get("candidate_sample") if best_valid else None,
            "best_valid_face_cd": best_valid.get("face_cd") if best_valid else None,
        }
        if row["baseline_face_cd"] is not None and row["best_face_cd"] is not None:
            row["oracle_face_cd_gain"] = float(row["baseline_face_cd"]) - float(row["best_face_cd"])
        rows.append(row)

    summary = {
        "num_prefixes": len(rows),
        "baseline_valid_rate": mean_or_none([1.0 if row["baseline_is_success"] else 0.0 for row in rows]),
        "oracle_any_valid_rate": mean_or_none([1.0 if row["any_valid"] else 0.0 for row in rows]),
        "baseline_face_cd_mean": mean_or_none(finite_values(rows, "baseline_face_cd")),
        "baseline_face_cd_median": median_or_none(finite_values(rows, "baseline_face_cd")),
        "best_face_cd_mean": mean_or_none(finite_values(rows, "best_face_cd")),
        "best_face_cd_median": median_or_none(finite_values(rows, "best_face_cd")),
        "best_valid_face_cd_mean": mean_or_none(finite_values(rows, "best_valid_face_cd")),
        "best_valid_face_cd_median": median_or_none(finite_values(rows, "best_valid_face_cd")),
        "oracle_face_cd_gain_mean": mean_or_none(finite_values(rows, "oracle_face_cd_gain")),
        "oracle_face_cd_gain_median": median_or_none(finite_values(rows, "oracle_face_cd_gain")),
    }
    write_json(args.output_json, {"summary": summary, "rows": rows})
    write_csv(args.output_csv, rows)
    write_run_metadata(Path(args.output_json).with_suffix(".metadata.json"), args)
    print(summary)


if __name__ == "__main__":
    main()
