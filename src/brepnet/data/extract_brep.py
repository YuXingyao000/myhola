"""Explicit, non-destructive CLI for the existing STEP half-edge extractor.

This prepares geometry only; it does not create splits, images or latent caches.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from src.brepnet.data.datagen.utils import load_model_ids


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True, help="Root containing CAD ID/STEP files.")
    parser.add_argument("--output-root", type=Path, required=True, help="New or empty output directory.")
    parser.add_argument("--model-list", type=Path, required=True, help="Explicit source CAD ID list; no new split is created.")
    parser.add_argument("--exclude-list", type=Path, action="append", default=[], help="Optional explicit exclusion list; repeatable.")
    parser.add_argument("--num-workers", type=int, default=1, help="CPU workers; 1 avoids starting Ray.")
    parser.add_argument("--dry-run", action="store_true", help="Validate paths and lists without loading OCC or extracting geometry.")
    args = parser.parse_args(argv)
    if args.num_workers < 1:
        parser.error("--num-workers must be positive")
    return args


def prepare_plan(args: argparse.Namespace) -> tuple[list[str], dict]:
    args.data_root = args.data_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.model_list = args.model_list.expanduser().resolve()
    args.exclude_list = [p.expanduser().resolve() for p in args.exclude_list]
    if not args.data_root.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {args.data_root}")
    if args.output_root.is_relative_to(args.data_root) or args.data_root.is_relative_to(args.output_root):
        raise ValueError("Input and output directories must not overlap.")
    if args.output_root.exists() and (not args.output_root.is_dir() or any(args.output_root.iterdir())):
        raise FileExistsError(f"Output must be new or empty; existing files are preserved: {args.output_root}")
    source_ids = load_model_ids(args.model_list)
    excluded = {model_id for path in args.exclude_list for model_id in load_model_ids(path)}
    model_ids = [model_id for model_id in source_ids if model_id not in excluded]
    if not model_ids:
        raise ValueError("No CAD IDs remain in the supplied list after explicit exclusions.")
    manifest = {
        "data_root": str(args.data_root),
        "output_root": str(args.output_root),
        "source_lists": [
            {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in [args.model_list, *args.exclude_list]
        ],
        "source_count": len(source_ids),
        "excluded_count": len(source_ids) - len(model_ids),
        "selected_count": len(model_ids),
        "num_workers": args.num_workers,
        "status": "planned",
        "scope": "geometry only; original split retained; no latent or condition generation",
    }
    return model_ids, manifest


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    model_ids, manifest = prepare_plan(args)
    print(json.dumps(manifest, indent=2))
    if args.dry_run:
        return

    # Keep --help and --dry-run usable before the new OCC environment is installed.
    from src.brepnet.data.extract_brep_from_step_half_structure import get_brep, get_brep_ray

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root / "extraction_manifest.json"
    manifest["status"] = "running"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    succeeded = 0
    with (args.output_root / "extraction_results.jsonl").open("w", encoding="utf-8") as report:
        def record(model_id: str, ok: bool) -> None:
            nonlocal succeeded
            succeeded += int(ok)
            report.write(json.dumps({"model_id": model_id, "success": bool(ok)}) + "\n")
            report.flush()

        if args.num_workers == 1:
            for model_id in model_ids:
                record(model_id, get_brep(args.data_root, args.output_root, model_id))
        else:
            import ray

            ray.init(num_cpus=args.num_workers, include_dashboard=False)
            try:
                remaining = iter(model_ids)
                pending = {}
                while True:
                    while len(pending) < 2 * args.num_workers:
                        model_id = next(remaining, None)
                        if model_id is None:
                            break
                        pending[get_brep_ray.remote(args.data_root, args.output_root, model_id)] = model_id
                    if not pending:
                        break
                    ready, _ = ray.wait(list(pending), num_returns=1)
                    for task in ready:
                        record(pending.pop(task), ray.get(task))
            finally:
                ray.shutdown()
    manifest.update(status="completed", succeeded=succeeded, failed=len(model_ids) - succeeded)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Processed {succeeded}/{len(model_ids)} files; failures remain in extraction_results.jsonl")
    if succeeded != len(model_ids):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
