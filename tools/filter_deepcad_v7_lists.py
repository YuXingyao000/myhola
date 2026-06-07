#!/usr/bin/env python3
"""Filter DeepCAD model lists to entries that exist in the v7 training assets.

Default behavior is non-destructive: filtered lists are written to
src/brepnet/data/list/filtered_v7/. Use --in-place to replace the input lists;
that mode writes a .bak copy next to each original before overwriting it.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path


DEFAULT_DATA_ROOT = Path("/mnt/d/data/deepcad_v7")
DEFAULT_COND_ROOT = Path("/mnt/d/data/deepcad_v7_cond")
DEFAULT_LATENT_ROOT = Path("/mnt/d/data/ae_cache/1119_deepcad_aug1_11k_24")
DEFAULT_OUTPUT_DIR = Path("src/brepnet/data/list/filtered_v7")
DEFAULT_REPORT_DIR = Path("experiments/2026-06-07/list_filter_report")

DEFAULT_LISTS = (
    Path("src/brepnet/data/list/deduplicated_deepcad_testing_7_30_sample_100.txt"),
    Path("src/brepnet/data/list/deduplicated_deepcad_testing_7_30.txt"),
    Path("src/brepnet/data/list/deduplicated_deepcad_training_7_30.txt"),
    Path("src/brepnet/data/list/deduplicated_deepcad_validation_7_30.txt"),
)

NUM_VIEWS = 24


@dataclass(frozen=True)
class RemovedEntry:
    list_path: str
    model_id: str
    reason: str
    detail: str


def load_model_ids(path: Path) -> list[str]:
    ids: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        model_id = raw_line.strip()
        if model_id and not model_id.startswith("#"):
            ids.append(model_id)
    return ids


def missing_reason(model_id: str, args: argparse.Namespace) -> tuple[str | None, str]:
    raw_data = args.data_root / model_id / "data.npz"
    if not raw_data.is_file():
        return "missing_raw_data_npz", str(raw_data)

    cond_dir = args.condition_root / model_id
    if not cond_dir.is_dir():
        return "missing_condition_dir", str(cond_dir)

    imgs_npz = cond_dir / "imgs.npz"
    if args.require_imgs and not imgs_npz.is_file():
        return "missing_imgs_npz", str(imgs_npz)

    real_photo_npz = cond_dir / "real_photo.npz"
    if args.require_real_photo and not real_photo_npz.is_file():
        return "missing_real_photo_npz", str(real_photo_npz)

    feature_path = cond_dir / "img_feature_dinov2.npy"
    if args.require_cached_condition and not feature_path.is_file():
        return "missing_cached_condition_feature", str(feature_path)

    view_ids = range(NUM_VIEWS) if args.latent_views == "all" else (0,)
    missing_latent_views = []
    for view_id in view_ids:
        latent_feature = args.latent_root / f"{model_id}_{view_id}" / "features.npy"
        if not latent_feature.is_file():
            missing_latent_views.append(view_id)

    if missing_latent_views:
        return (
            "missing_latent_features",
            f"missing rotations={missing_latent_views}; root={args.latent_root}",
        )

    return None, ""


def unique_preserve_order(items: list[str]) -> tuple[list[str], int]:
    seen: set[str] = set()
    out: list[str] = []
    duplicate_count = 0
    for item in items:
        if item in seen:
            duplicate_count += 1
            continue
        seen.add(item)
        out.append(item)
    return out, duplicate_count


def output_path_for(list_path: Path, args: argparse.Namespace) -> Path:
    if args.in_place:
        return list_path
    return args.output_dir / list_path.name


def write_list(path: Path, model_ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(model_ids) + ("\n" if model_ids else ""), encoding="utf-8")


def filter_one_list(list_path: Path, args: argparse.Namespace) -> dict[str, object]:
    input_ids = load_model_ids(list_path)
    if args.dedupe:
        candidate_ids, duplicate_count = unique_preserve_order(input_ids)
    else:
        candidate_ids = input_ids
        duplicate_count = 0

    kept: list[str] = []
    removed: list[RemovedEntry] = []
    for model_id in candidate_ids:
        reason, detail = missing_reason(model_id, args)
        if reason is None:
            kept.append(model_id)
        else:
            removed.append(
                RemovedEntry(
                    list_path=str(list_path),
                    model_id=model_id,
                    reason=reason,
                    detail=detail,
                )
            )

    out_path = output_path_for(list_path, args)
    if not args.dry_run:
        if args.in_place:
            backup_path = list_path.with_suffix(list_path.suffix + args.backup_suffix)
            if not backup_path.exists() or args.overwrite_backup:
                backup_path.write_text(list_path.read_text(encoding="utf-8"), encoding="utf-8")
            else:
                raise FileExistsError(
                    f"Backup already exists: {backup_path}. Use --overwrite-backup to replace it."
                )
        write_list(out_path, kept)

    reason_counts = Counter(item.reason for item in removed)
    return {
        "input_path": str(list_path),
        "output_path": str(out_path),
        "original_count": len(input_ids),
        "duplicate_count": duplicate_count,
        "checked_count": len(candidate_ids),
        "kept_count": len(kept),
        "removed_count": len(removed),
        "removed_by_reason": dict(reason_counts.most_common()),
        "removed": [asdict(item) for item in removed],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", dest="lists", type=Path, action="append", help="List file to filter. Repeatable. Defaults to the four common DeepCAD lists.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--condition-root", type=Path, default=DEFAULT_COND_ROOT)
    parser.add_argument("--latent-root", type=Path, default=DEFAULT_LATENT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--latent-views", choices=("all", "identity"), default="all")
    parser.add_argument("--no-require-imgs", dest="require_imgs", action="store_false")
    parser.add_argument("--no-require-real-photo", dest="require_real_photo", action="store_false")
    parser.add_argument("--require-cached-condition", action="store_true")
    parser.add_argument("--no-dedupe", dest="dedupe", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--in-place", action="store_true", help="Replace each input list after writing a .bak backup.")
    parser.add_argument("--backup-suffix", default=".bak")
    parser.add_argument("--overwrite-backup", action="store_true")
    parser.set_defaults(require_imgs=True, require_real_photo=True, dedupe=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    list_paths = args.lists if args.lists else list(DEFAULT_LISTS)

    summaries = []
    all_removed = []
    for list_path in list_paths:
        if not list_path.is_file():
            raise FileNotFoundError(f"List file not found: {list_path}")
        summary = filter_one_list(list_path, args)
        summaries.append({k: v for k, v in summary.items() if k != "removed"})
        all_removed.extend(summary["removed"])

    if not args.dry_run:
        args.report_dir.mkdir(parents=True, exist_ok=True)
        (args.report_dir / "summary.json").write_text(
            json.dumps(
                {
                    "data_root": str(args.data_root),
                    "condition_root": str(args.condition_root),
                    "latent_root": str(args.latent_root),
                    "latent_views": args.latent_views,
                    "require_imgs": args.require_imgs,
                    "require_real_photo": args.require_real_photo,
                    "require_cached_condition": args.require_cached_condition,
                    "in_place": args.in_place,
                    "lists": summaries,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        removed_lines = ["list_path\tmodel_id\treason\tdetail"]
        removed_lines.extend(
            f"{item['list_path']}\t{item['model_id']}\t{item['reason']}\t{item['detail']}"
            for item in all_removed
        )
        (args.report_dir / "removed.tsv").write_text("\n".join(removed_lines) + "\n", encoding="utf-8")

    for summary in summaries:
        print(
            f"{summary['input_path']} -> {summary['output_path']} | "
            f"kept {summary['kept_count']}/{summary['checked_count']} "
            f"removed {summary['removed_count']} duplicates {summary['duplicate_count']} "
            f"reasons={summary['removed_by_reason']}"
        )
    if args.dry_run:
        print("dry-run: no files written")
    else:
        print(f"report: {args.report_dir}")


if __name__ == "__main__":
    main()
