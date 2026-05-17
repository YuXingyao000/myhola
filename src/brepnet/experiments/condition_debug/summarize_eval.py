import argparse
from pathlib import Path

from src.brepnet.experiments.condition_debug.common import (
    load_eval_result,
    read_prefixes,
    summarize_rows,
    write_csv,
    write_json,
    write_run_metadata,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize eval_condition eval.npz files.")
    parser.add_argument("--post-root", required=True)
    parser.add_argument("--list", default="")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    post_root = Path(args.post_root)
    if not post_root.exists():
        raise FileNotFoundError(f"Missing post root: {post_root}")
    if args.list:
        folders = read_prefixes(args.list)
    else:
        folders = sorted(item.name for item in post_root.iterdir() if item.is_dir() and not item.name.startswith("_"))

    rows = []
    missing_eval = []
    for folder in folders:
        result = load_eval_result(post_root, folder)
        if result is None:
            missing_eval.append(folder)
            rows.append({"prefix": folder, "eval_missing": True, "is_success": False})
            continue
        result["eval_missing"] = False
        if "num_recon_face" in result and "num_gt_face" in result:
            result["face_count_error"] = result["num_recon_face"] - result["num_gt_face"]
            result["abs_face_count_error"] = abs(result["face_count_error"])
        rows.append(result)

    summary = summarize_rows(rows)
    summary["post_root"] = str(post_root)
    summary["missing_eval_count"] = len(missing_eval)
    summary["missing_eval_prefixes"] = missing_eval

    write_json(args.output_json, {"summary": summary, "rows": rows})
    write_csv(args.output_csv, rows)
    write_run_metadata(Path(args.output_json).with_suffix(".metadata.json"), args)
    print(summary)


if __name__ == "__main__":
    main()

