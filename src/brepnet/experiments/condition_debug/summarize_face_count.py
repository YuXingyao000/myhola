import argparse
from collections import defaultdict
from pathlib import Path

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
    parser = argparse.ArgumentParser(description="Analyze face-count error against validity and metrics.")
    parser.add_argument("--post-root", required=True)
    parser.add_argument("--list", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    return parser.parse_args()


def face_count_bin(abs_error: int | float | None) -> str:
    if abs_error is None:
        return "missing"
    if abs_error == 0:
        return "0"
    if abs_error == 1:
        return "1"
    if abs_error == 2:
        return "2"
    if abs_error <= 5:
        return "3-5"
    return ">5"


def summarize_group(rows):
    return {
        "count": len(rows),
        "valid_rate": mean_or_none([1.0 if row.get("is_success") else 0.0 for row in rows]),
        "face_cd_mean": mean_or_none(finite_values(rows, "face_cd")),
        "face_cd_median": median_or_none(finite_values(rows, "face_cd")),
        "edge_cd_mean": mean_or_none(finite_values(rows, "edge_cd")),
        "fe_fscore_mean": mean_or_none(finite_values(rows, "fe_fscore")),
        "ev_fscore_mean": mean_or_none(finite_values(rows, "ev_fscore")),
        "abs_face_count_error_mean": mean_or_none(finite_values(rows, "abs_face_count_error")),
    }


def main() -> None:
    args = parse_args()
    rows = []
    missing = []
    for prefix in read_prefixes(args.list):
        result = load_eval_result(args.post_root, prefix)
        if result is None:
            missing.append(prefix)
            rows.append({"prefix": prefix, "eval_missing": True, "is_success": False})
            continue
        result["eval_missing"] = False
        result["face_count_error"] = result["num_recon_face"] - result["num_gt_face"]
        result["abs_face_count_error"] = abs(result["face_count_error"])
        if result["face_count_error"] > 0:
            result["face_count_direction"] = "over"
        elif result["face_count_error"] < 0:
            result["face_count_direction"] = "under"
        else:
            result["face_count_direction"] = "exact"
        result["abs_face_count_bin"] = face_count_bin(result["abs_face_count_error"])
        rows.append(result)

    by_bin = defaultdict(list)
    by_direction = defaultdict(list)
    by_validity = defaultdict(list)
    for row in rows:
        by_bin[row.get("abs_face_count_bin", "missing")].append(row)
        by_direction[row.get("face_count_direction", "missing")].append(row)
        by_validity["valid" if row.get("is_success") else "invalid"].append(row)

    summary = {
        "num_rows": len(rows),
        "missing_eval_count": len(missing),
        "overall": summarize_group(rows),
        "by_abs_face_count_bin": {key: summarize_group(value) for key, value in sorted(by_bin.items())},
        "by_face_count_direction": {key: summarize_group(value) for key, value in sorted(by_direction.items())},
        "by_validity": {key: summarize_group(value) for key, value in sorted(by_validity.items())},
    }

    write_json(args.output_json, {"summary": summary, "rows": rows})
    write_csv(args.output_csv, rows)
    write_run_metadata(Path(args.output_json).with_suffix(".metadata.json"), args)
    print(summary)


if __name__ == "__main__":
    main()

