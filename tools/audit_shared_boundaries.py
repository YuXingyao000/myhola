"""Audit the complete intended split before shared-boundary VAE training.

This checks the existing merged face-pair representation, not native B-Rep
physical-edge coverage. Every requested ID receives a row, including failures.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np

from src.brepnet.data.shared_boundaries import build_shared_boundary_targets


def write_accepted_list(rows: list[dict], output_dir: Path) -> dict:
    """Materialize an explicitly audited subset, preserving all rejection reasons."""
    accepted = [row["model_id"] for row in rows if row["status"] == "ok"]
    rejected = [row for row in rows if row["status"] != "ok"]
    text = "".join(model_id + "\n" for model_id in accepted)
    (output_dir / "accepted.txt").write_text(text, encoding="utf-8")
    with (output_dir / "rejected.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rejected)
    return {"accepted_list": "accepted.txt", "rejection_manifest": "rejected.csv",
            "accepted_sha256": hashlib.sha256(text.encode()).hexdigest()}


def audit_sample(model_id: str, *, data_root: Path, reciprocal_tolerance: float,
                 closed_tolerance: float, num_queries: int) -> dict:
    row = {"model_id": model_id, "status": "failed", "num_faces": None,
           "num_boundaries": None, "num_closed": None, "closed_phase_used": None,
           "max_reciprocal_residual": None, "error": ""}
    try:
        with np.load(data_root / model_id / "data.npz", allow_pickle=False) as data:
            faces = data["sample_points_faces"]
            targets = build_shared_boundary_targets(
                faces, data["sample_points_lines"], data["edge_face_connectivity"],
                sample_id=model_id, reciprocal_tolerance=reciprocal_tolerance,
                closed_tolerance=closed_tolerance, allow_closed_phase=True,
            )
        count = len(targets["curves"])
        row.update(num_faces=len(faces), num_boundaries=count,
                   num_closed=int(targets["is_closed"].sum()),
                   closed_phase_used=int(targets["closed_phase_used"].sum()),
                   max_reciprocal_residual=float(
                       targets["reciprocal_residual"].max(initial=0)))
        if count > num_queries:
            raise ValueError(f"{count} boundaries exceed query capacity {num_queries}")
        row["status"] = "ok"
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-list", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reciprocal-tolerance", type=float, default=0.01)
    parser.add_argument("--closed-tolerance", type=float, default=1e-5)
    parser.add_argument("--num-queries", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0,
                        help="Diagnostic prefix only; 0 audits the entire split.")
    parser.add_argument("--write-accepted-list", action="store_true",
                        help="Write accepted.txt and rejected.csv after a complete audit. "
                             "Exit status still reports whether any IDs were rejected.")
    args = parser.parse_args()
    if args.workers < 1 or args.num_queries < 1 or args.limit < 0:
        parser.error("workers/num-queries must be positive and limit nonnegative")
    if args.limit and args.write_accepted_list:
        parser.error("accepted lists require the entire split; do not combine with --limit")
    if any(not math.isfinite(x) or x <= 0 for x in
           [args.reciprocal_tolerance, args.closed_tolerance]):
        parser.error("tolerances must be finite and positive")
    split_bytes = args.split_list.read_bytes()
    ids = [line.strip() for line in split_bytes.decode().splitlines() if line.strip()]
    if not ids or len(ids) != len(set(ids)):
        parser.error("split must be nonempty and contain unique IDs")
    if any(Path(item).name != item or item in {".", ".."} for item in ids):
        parser.error("split entries must be individual model IDs, not paths")
    intended = len(ids)
    if args.limit:
        ids = ids[:args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "samples.csv"
    summary_path = args.output_dir / "summary.json"
    if csv_path.exists() or summary_path.exists():
        parser.error("output already contains an audit; choose a new directory")
    audit = partial(audit_sample, data_root=args.data_root,
                    reciprocal_tolerance=args.reciprocal_tolerance,
                    closed_tolerance=args.closed_tolerance, num_queries=args.num_queries)
    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for idx, row in enumerate(pool.map(audit, ids), start=1):
            rows.append(row)
            if idx % 2000 == 0:
                print(f"Audited {idx}/{len(ids)}", flush=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    valid = [row for row in rows if row["status"] == "ok"]
    summary = {
        "representation": "shared_boundary_v1",
        "data_root": str(args.data_root.resolve()),
        "split_list": str(args.split_list.resolve()),
        "split_sha256": hashlib.sha256(split_bytes).hexdigest(),
        "intended_count": intended, "audited_count": len(rows),
        "complete_split": len(rows) == intended,
        "passed": len(valid), "failed": len(rows) - len(valid),
        "num_queries": args.num_queries,
        "reciprocal_tolerance": args.reciprocal_tolerance,
        "closed_tolerance": args.closed_tolerance,
        "max_boundaries_among_passed": max((r["num_boundaries"] for r in valid), default=0),
        "closed_phase_used_among_passed": sum(r["closed_phase_used"] for r in valid),
        "max_reciprocal_residual_among_passed": max(
            (r["max_reciprocal_residual"] for r in valid), default=0),
        "failure_examples": [r for r in rows if r["status"] != "ok"][:10],
    }
    if args.write_accepted_list:
        summary.update(write_accepted_list(rows, args.output_dir))
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return int(summary["failed"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
