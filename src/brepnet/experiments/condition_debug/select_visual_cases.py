import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


POST_ARTIFACTS = (
    "recon_brep.step",
    "recon_brep.stl",
    "separate_faces.ply",
    "eval_condition.npz",
    "eval.npz",
    "success.txt",
)

GT_ARTIFACTS = (
    "normalized_shape.step",
    "post_processed_shape.step",
    "mesh.ply",
    "pc.ply",
)


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--candidate-post-root", required=True)
    parser.add_argument("--candidate-raw-root", default=None)
    parser.add_argument("--baseline-post-root", required=True)
    parser.add_argument("--baseline-raw-root", default=None)
    parser.add_argument("--gt-root", default=None)
    parser.add_argument("--cond-root", default=None)
    parser.add_argument("--list", dest="list_path", default=None)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--metric", default="face_cd")
    parser.add_argument("--num-success", type=int, default=5)
    parser.add_argument("--num-fail", type=int, default=5)
    parser.add_argument("--fail-sort", choices=("asc", "desc"), default="asc")
    parser.add_argument("--force", action="store_true")


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def scalar_to_python(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return scalar_to_python(value.item())
        return scalar_to_python(value.tolist())
    if isinstance(value, dict):
        return {key: scalar_to_python(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [scalar_to_python(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def safe_mkdir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def prepare_output_dir(path: str | Path, force: bool = False) -> Path:
    path = Path(path)
    if path.exists():
        if not force:
            raise FileExistsError(f"Output already exists: {path}. Pass --force to overwrite it.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_prefixes(list_path: str | Path) -> list[str]:
    with Path(list_path).open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    safe_mkdir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(scalar_to_python(data), handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    safe_mkdir(path.parent)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def git_status_short() -> str:
    try:
        result = subprocess.run(
            ["git", "status", "--short"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return result.stdout
    except Exception as exc:
        return f"failed to collect git status: {exc}"


def write_run_metadata(path: str | Path, args: argparse.Namespace, extra: dict[str, Any] | None = None) -> None:
    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "argv": sys.argv,
        "args": vars(args),
        "cwd": os.getcwd(),
        "git_status_short": git_status_short(),
    }
    if extra:
        metadata.update(extra)
    write_json(path, metadata)


def load_eval_result(post_root: str | Path, folder: str) -> dict[str, Any] | None:
    item_root = Path(post_root) / folder
    eval_path = None
    for filename in ("eval_condition.npz", "eval.npz"):
        candidate = item_root / filename
        if candidate.exists():
            eval_path = candidate
            break
    if eval_path is None:
        return None
    try:
        item = np.load(eval_path, allow_pickle=True)["results"].item()
    except Exception:
        return None
    result = {key: scalar_to_python(value) for key, value in item.items()}
    result["prefix"] = folder
    result["is_success"] = (item_root / "success.txt").exists()
    result["eval_path"] = str(eval_path)
    return result


def discover_prefixes(post_root: Path, list_path: str | None) -> list[str]:
    if list_path:
        return read_prefixes(list_path)
    prefixes = []
    for path in sorted(post_root.iterdir()):
        if path.is_dir() and not path.name.startswith("_"):
            prefixes.append(path.name)
    return prefixes


def has_visual_artifact(post_root: Path, prefix: str) -> bool:
    item_root = post_root / prefix
    return any((item_root / name).exists() for name in ("recon_brep.step", "recon_brep.stl", "separate_faces.ply"))


def copy_file(src: Path, dst: Path) -> str | None:
    if not src.exists() or not src.is_file():
        return None
    safe_mkdir(dst.parent)
    shutil.copy2(src, dst)
    return str(dst)


def copy_post_artifacts(src_root: Path | None, prefix: str, dst_root: Path) -> list[str]:
    if src_root is None:
        return []
    item_root = src_root / prefix
    copied = []
    for filename in POST_ARTIFACTS:
        copied_path = copy_file(item_root / filename, dst_root / filename)
        if copied_path:
            copied.append(copied_path)
    return copied


def copy_raw_artifacts(src_root: Path | None, prefix: str, dst_root: Path) -> list[str]:
    if src_root is None:
        return []
    item_root = src_root / prefix
    copied = []
    for pattern in ("*.png", "*_edge.obj", "data.npz"):
        for src in sorted(item_root.glob(pattern)):
            copied_path = copy_file(src, dst_root / src.name)
            if copied_path:
                copied.append(copied_path)
    return copied


def copy_gt_artifacts(gt_root: Path | None, prefix: str, dst_root: Path) -> list[str]:
    if gt_root is None:
        return []
    item_root = gt_root / prefix
    copied = []
    for filename in GT_ARTIFACTS:
        copied_path = copy_file(item_root / filename, dst_root / filename)
        if copied_path:
            copied.append(copied_path)
    return copied


def save_np_image(array: np.ndarray, dst: Path) -> str:
    safe_mkdir(dst.parent)
    image = np.asarray(array)
    if image.dtype != np.uint8:
        if image.max(initial=0) <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
    Image.fromarray(image).save(dst)
    return str(dst)


def copy_condition_artifacts(cond_root: Path | None, prefix: str, dst_root: Path) -> list[str]:
    if cond_root is None:
        return []
    item_root = cond_root / prefix
    copied = []

    single_view_path = item_root / "single_view.npz"
    if single_view_path.exists():
        data = np.load(single_view_path)
        for key in ("flux", "flux_masked", "blender"):
            if key in data.files:
                copied.append(save_np_image(data[key], dst_root / f"single_view_{key}.png"))

    combined_path = item_root / "combined_imgs.npz"
    if combined_path.exists():
        data = np.load(combined_path)
        for key in ("natural_img", "sketch_img"):
            if key in data.files:
                copied.append(save_np_image(data[key], dst_root / f"combined_{key}.png"))

    return copied


def metric_row(prefix: str, candidate: dict[str, Any], baseline: dict[str, Any] | None, metric: str) -> dict[str, Any]:
    row = {
        "prefix": prefix,
        "candidate_success": bool(candidate.get("is_success")),
        f"candidate_{metric}": candidate.get(metric),
        "candidate_face_cd": candidate.get("face_cd"),
        "candidate_edge_cd": candidate.get("edge_cd"),
        "candidate_vertex_cd": candidate.get("vertex_cd"),
        "candidate_face_fscore": candidate.get("face_fscore"),
        "candidate_edge_fscore": candidate.get("edge_fscore"),
        "candidate_vertex_fscore": candidate.get("vertex_fscore"),
        "candidate_fe_fscore": candidate.get("fe_fscore"),
        "candidate_ev_fscore": candidate.get("ev_fscore"),
        "candidate_num_recon_face": candidate.get("num_recon_face"),
        "candidate_num_gt_face": candidate.get("num_gt_face"),
        "candidate_abs_face_count_error": candidate.get("abs_face_count_error"),
    }
    if baseline is None:
        row["baseline_missing"] = True
        return row

    row.update(
        {
            "baseline_missing": False,
            "baseline_success": bool(baseline.get("is_success")),
            f"baseline_{metric}": baseline.get(metric),
            "baseline_face_cd": baseline.get("face_cd"),
            "baseline_edge_cd": baseline.get("edge_cd"),
            "baseline_vertex_cd": baseline.get("vertex_cd"),
            "baseline_face_fscore": baseline.get("face_fscore"),
            "baseline_edge_fscore": baseline.get("edge_fscore"),
            "baseline_vertex_fscore": baseline.get("vertex_fscore"),
            "baseline_fe_fscore": baseline.get("fe_fscore"),
            "baseline_ev_fscore": baseline.get("ev_fscore"),
            "baseline_num_recon_face": baseline.get("num_recon_face"),
            "baseline_num_gt_face": baseline.get("num_gt_face"),
            "baseline_abs_face_count_error": baseline.get("abs_face_count_error"),
        }
    )
    return row


def collect_cases(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidate_post_root = Path(args.candidate_post_root)
    baseline_post_root = Path(args.baseline_post_root)
    prefixes = discover_prefixes(candidate_post_root, args.list_path)

    cases = []
    skipped = {
        "missing_candidate_eval": 0,
        "missing_metric": 0,
        "missing_visual_artifact": 0,
    }
    for prefix in prefixes:
        candidate = load_eval_result(candidate_post_root, prefix)
        if candidate is None:
            skipped["missing_candidate_eval"] += 1
            continue
        metric_value = as_float(candidate.get(args.metric))
        if metric_value is None:
            skipped["missing_metric"] += 1
            continue
        if not has_visual_artifact(candidate_post_root, prefix):
            skipped["missing_visual_artifact"] += 1
            continue
        baseline = load_eval_result(baseline_post_root, prefix)
        cases.append(
            {
                "prefix": prefix,
                "metric_value": metric_value,
                "candidate": candidate,
                "baseline": baseline,
            }
        )

    metadata = {
        "num_prefixes": len(prefixes),
        "num_eligible": len(cases),
        "skipped": skipped,
    }
    return cases, metadata


def select_cases(cases: list[dict[str, Any]], args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    success_cases = [case for case in cases if case["candidate"].get("is_success")]
    fail_cases = [case for case in cases if not case["candidate"].get("is_success")]
    success_cases = sorted(success_cases, key=lambda case: case["metric_value"])
    fail_cases = sorted(fail_cases, key=lambda case: case["metric_value"], reverse=args.fail_sort == "desc")
    return success_cases[: args.num_success], fail_cases[: args.num_fail]


def write_case_package(
    output_root: Path,
    group_name: str,
    selected_cases: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    candidate_post_root = Path(args.candidate_post_root)
    baseline_post_root = Path(args.baseline_post_root)
    candidate_raw_root = Path(args.candidate_raw_root) if args.candidate_raw_root else None
    baseline_raw_root = Path(args.baseline_raw_root) if args.baseline_raw_root else None
    gt_root = Path(args.gt_root) if args.gt_root else None
    cond_root = Path(args.cond_root) if args.cond_root else None

    rows = []
    for rank, case in enumerate(selected_cases, start=1):
        prefix = case["prefix"]
        case_root = safe_mkdir(output_root / group_name / f"{rank:02d}_{prefix}")
        copied = {
            "candidate_post": copy_post_artifacts(candidate_post_root, prefix, case_root / "candidate" / "post"),
            "candidate_raw": copy_raw_artifacts(candidate_raw_root, prefix, case_root / "candidate" / "raw"),
            "baseline_post": copy_post_artifacts(baseline_post_root, prefix, case_root / "baseline" / "post"),
            "baseline_raw": copy_raw_artifacts(baseline_raw_root, prefix, case_root / "baseline" / "raw"),
            "gt": copy_gt_artifacts(gt_root, prefix, case_root / "gt"),
            "condition": copy_condition_artifacts(cond_root, prefix, case_root / "condition"),
        }
        row = {
            "group": group_name,
            "rank": rank,
            "case_dir": str(case_root),
            **metric_row(prefix, case["candidate"], case["baseline"], args.metric),
            "copied_candidate_post_count": len(copied["candidate_post"]),
            "copied_baseline_post_count": len(copied["baseline_post"]),
            "copied_gt_count": len(copied["gt"]),
            "copied_condition_count": len(copied["condition"]),
        }
        rows.append(row)
        write_json(case_root / "case.json", {**row, "copied": copied})
    return rows


def write_readme(output_root: Path, args: argparse.Namespace, rows: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    num_success = sum(row["group"] == "success" for row in rows)
    num_fail = sum(row["group"] == "fail" for row in rows)
    text = f"""# Visual Case Selection

Candidate cases are selected from:
`{args.candidate_post_root}`

Baseline artifacts are copied for the same prefixes from:
`{args.baseline_post_root}`

When `--cond-root` is provided, source condition images are exported from `single_view.npz`, including `single_view_flux.png` and `single_view_flux_masked.png`.

Selection metric: `{args.metric}`.

- `success/`: candidate successful cases sorted by smallest `{args.metric}`.
- `fail/`: candidate failed cases with finite `{args.metric}` and post artifacts, sorted by `{args.fail_sort}`.

Selected {num_success} success cases and {num_fail} fail cases from {metadata["num_eligible"]} eligible cases.
See `selection.csv` and `selection.json` for metrics and paths.
"""
    (output_root / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_args(parser)
    args = parser.parse_args()

    output_root = prepare_output_dir(args.output_root, args.force)
    cases, metadata = collect_cases(args)
    success_cases, fail_cases = select_cases(cases, args)

    rows = []
    rows.extend(write_case_package(output_root, "success", success_cases, args))
    rows.extend(write_case_package(output_root, "fail", fail_cases, args))

    metadata.update(
        {
            "num_candidate_success": int(sum(case["candidate"].get("is_success") for case in cases)),
            "num_candidate_fail": int(sum(not case["candidate"].get("is_success") for case in cases)),
            "num_selected_success": len(success_cases),
            "num_selected_fail": len(fail_cases),
        }
    )
    write_csv(output_root / "selection.csv", rows)
    write_json(output_root / "selection.json", {"metadata": metadata, "cases": rows})
    write_run_metadata(output_root / "_metadata" / "select_visual_cases.json", args, metadata)
    write_readme(output_root, args, rows, metadata)

    print(f"Wrote {len(rows)} visual cases to {output_root}")
    print(f"Selected success={len(success_cases)} fail={len(fail_cases)}")


if __name__ == "__main__":
    main()
