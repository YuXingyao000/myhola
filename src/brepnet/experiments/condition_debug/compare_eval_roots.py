import argparse
from pathlib import Path
from typing import Any

import numpy as np

from src.brepnet.experiments.condition_debug.common import (
    finite_values,
    load_eval_result,
    mean_or_none,
    read_prefixes,
    write_csv,
    write_json,
    write_run_metadata,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two eval_condition post roots.")
    parser.add_argument("--baseline-post-root", required=True)
    parser.add_argument("--candidate-post-root", required=True)
    parser.add_argument("--baseline-name", default="baseline")
    parser.add_argument("--candidate-name", default="candidate")
    parser.add_argument("--list", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    return parser.parse_args()


def get_metric(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value):
        return None
    return value


def add_face_count_fields(result: dict[str, Any] | None) -> dict[str, Any] | None:
    if result is None:
        return None
    if "num_recon_face" in result and "num_gt_face" in result:
        result["face_count_error"] = result["num_recon_face"] - result["num_gt_face"]
        result["abs_face_count_error"] = abs(result["face_count_error"])
    return result


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    both_eval = [row for row in rows if row["baseline_eval_exists"] and row["candidate_eval_exists"]]
    summary = {
        "num_rows": len(rows),
        "both_eval_count": len(both_eval),
        "baseline_eval_missing_count": sum(not row["baseline_eval_exists"] for row in rows),
        "candidate_eval_missing_count": sum(not row["candidate_eval_exists"] for row in rows),
        "baseline_valid_rate": sum(row["baseline_is_success"] for row in rows) / len(rows) if rows else None,
        "candidate_valid_rate": sum(row["candidate_is_success"] for row in rows) / len(rows) if rows else None,
        "invalid_to_valid_count": sum((not row["baseline_is_success"]) and row["candidate_is_success"] for row in rows),
        "valid_to_invalid_count": sum(row["baseline_is_success"] and (not row["candidate_is_success"]) for row in rows),
        "both_valid_count": sum(row["baseline_is_success"] and row["candidate_is_success"] for row in rows),
        "both_invalid_count": sum((not row["baseline_is_success"]) and (not row["candidate_is_success"]) for row in rows),
    }
    if summary["baseline_valid_rate"] is not None and summary["candidate_valid_rate"] is not None:
        summary["valid_rate_delta"] = summary["candidate_valid_rate"] - summary["baseline_valid_rate"]

    for key in ["face_cd", "edge_cd", "vertex_cd", "face_fscore", "edge_fscore", "vertex_fscore", "fe_fscore", "ev_fscore", "abs_face_count_error"]:
        b_values = finite_values(rows, f"baseline_{key}")
        c_values = finite_values(rows, f"candidate_{key}")
        deltas = finite_values(rows, f"delta_{key}")
        improvements = finite_values(rows, f"improvement_{key}") if key.endswith("_cd") or key == "abs_face_count_error" else []
        summary[f"baseline_{key}_mean"] = mean_or_none(b_values)
        summary[f"candidate_{key}_mean"] = mean_or_none(c_values)
        summary[f"delta_{key}_mean"] = mean_or_none(deltas)
        if improvements:
            summary[f"improvement_{key}_mean"] = mean_or_none(improvements)
            summary[f"improved_{key}_count"] = sum(value > 0 for value in improvements)
            summary[f"worse_{key}_count"] = sum(value < 0 for value in improvements)
    return summary


def main() -> None:
    args = parse_args()
    prefixes = read_prefixes(args.list)
    rows = []
    for prefix in prefixes:
        baseline = add_face_count_fields(load_eval_result(args.baseline_post_root, prefix))
        candidate = add_face_count_fields(load_eval_result(args.candidate_post_root, prefix))
        row: dict[str, Any] = {
            "prefix": prefix,
            "baseline_eval_exists": baseline is not None,
            "candidate_eval_exists": candidate is not None,
            "baseline_is_success": bool(baseline and baseline.get("is_success")),
            "candidate_is_success": bool(candidate and candidate.get("is_success")),
        }
        metric_keys = [
            "face_cd",
            "edge_cd",
            "vertex_cd",
            "face_fscore",
            "edge_fscore",
            "vertex_fscore",
            "fe_fscore",
            "ev_fscore",
            "num_recon_face",
            "num_gt_face",
            "abs_face_count_error",
        ]
        for key in metric_keys:
            b_value = get_metric(baseline or {}, key)
            c_value = get_metric(candidate or {}, key)
            row[f"baseline_{key}"] = b_value
            row[f"candidate_{key}"] = c_value
            if b_value is not None and c_value is not None:
                row[f"delta_{key}"] = c_value - b_value
                if key.endswith("_cd") or key == "abs_face_count_error":
                    row[f"improvement_{key}"] = b_value - c_value
        rows.append(row)

    summary = summarize(rows)
    write_json(
        args.output_json,
        {
            "summary": summary,
            "rows": rows,
            "baseline_name": args.baseline_name,
            "candidate_name": args.candidate_name,
        },
    )
    write_csv(args.output_csv, rows)
    write_run_metadata(Path(args.output_json).with_suffix(".metadata.json"), args)
    print(summary)


if __name__ == "__main__":
    main()
