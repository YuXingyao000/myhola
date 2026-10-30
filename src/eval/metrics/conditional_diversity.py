"""Raw geometry repetition and variation among samples of the same input image.

Every intended candidate must appear in the manifest, including missing outputs.
Coordinates are used as stored: no centering, rescaling, alignment, or GT selection.
Raw surface grids are untrimmed; these metrics do not establish B-Rep equivalence
or that different completions are valid and consistent with the input image.

The within-image distance is the mean over unordered candidate pairs, using the
sum of the two directed mean squared nearest-neighbor distances. It is inspired
by the separate diversity/fidelity measurements of Multimodal Shape Completion:
https://github.com/rundiwu/Multimodal-Shape-Completion
Their released TMD sums pairs with factor 2/(K-1), which is K times our pair mean;
the values here must not be presented as a reproduction of that TMD protocol.
HoLa uses nearest-training-set LFD for novelty, a different question:
https://arxiv.org/html/2504.14257v1#S5.SS1

JSON/JSONL rows require method, image_id, candidate_id, raw_path, and validity.
Relative raw_path values in the CLI are relative to the manifest directory.
Example: python -m src.eval.metrics.conditional_diversity \
    --manifest candidates.jsonl --output-dir diversity_report --workers 8
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
from itertools import combinations
import json
from pathlib import Path
from typing import Any
from zipfile import BadZipFile

import numpy as np
from scipy.spatial import cKDTree

from src.eval.io import write_csv, write_json


QUANTIZATION_STEPS = (1e-4, 1e-3, 1e-2)
PRIMITIVES = ("face", "edge")
Record = dict[str, Any]


def _xyz_points(value: np.ndarray, name: str) -> np.ndarray:
    points = np.asarray(value, dtype=np.float64)
    if points.ndim < 2 or points.shape[-1] < 3 or points.size == 0:
        raise ValueError(f"{name} must contain nonempty XYZ points")
    points = points[..., :3].reshape(-1, 3)
    if not np.isfinite(points).all():
        raise ValueError(f"{name} contains nonfinite coordinates")
    return points


def geometry_fingerprints(
    raw_path: str | Path,
    quantization_steps: tuple[float, ...] = QUANTIZATION_STEPS,
) -> dict[str, str]:
    """Hash the separately quantized surface and curve point sets at each step.

    Point order, face order, curve direction, and repeated identical points do
    not affect the fingerprint. Equal fingerprints are only equal occupied
    quantization cells, not exact solids; bin boundaries can split near copies.
    """
    with np.load(raw_path, allow_pickle=False) as raw:
        points = {name: _xyz_points(raw[f"pred_{name}"], name) for name in PRIMITIVES}
    result = {}
    for step in quantization_steps:
        if not np.isfinite(step) or step <= 0:
            raise ValueError("Quantization steps must be finite and positive")
        digest = hashlib.sha256()
        digest.update(f"raw_geometry_cells_v1:{step:.17g}".encode("ascii"))
        for name in PRIMITIVES:
            cells = np.unique(np.floor(points[name] / step + 0.5).astype("<i8"), axis=0)
            digest.update(name.encode("ascii"))
            digest.update(np.asarray(cells.shape, dtype="<i8").tobytes())
            digest.update(cells.tobytes())
        result[format(step, ".12g")] = digest.hexdigest()
    return result


def _record_key(record: Record) -> tuple[str, str, str]:
    return record["method"], record["image_id"], record["candidate_id"]


def _check_records(records: list[Record]) -> None:
    keys = [_record_key(record) for record in records]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate method/image_id/candidate_id in manifest")


def _fingerprint_subset(records: list[Record], steps: list[str]) -> dict[str, Any]:
    output: dict[str, Any] = {"intended_count": len(records), "quantization": {}}
    for step in steps:
        groups: dict[str, list[Record]] = defaultdict(list)
        missing = []
        for record in records:
            identity = {key: record[key] for key in ("image_id", "candidate_id")}
            fingerprint = record.get("raw_fingerprints", {}).get(step)
            if record.get("raw_status") == "ok" and fingerprint:
                groups[fingerprint].append(identity)
            else:
                missing.append({**identity, "raw_status": record.get("raw_status", "not_recorded")})
        measured = sum(map(len, groups.values()))
        duplicate_groups = [members for members in groups.values() if len(members) > 1]
        repeated_count = sum(map(len, duplicate_groups))
        duplicate_pairs = sum(len(members) * (len(members) - 1) // 2 for members in duplicate_groups)
        possible_pairs = measured * (measured - 1) // 2
        output["quantization"][step] = {
            "measured_count": measured,
            "missing_count": len(missing),
            "unique_fingerprint_count": len(groups),
            "unique_fingerprint_ratio": len(groups) / measured if measured else None,
            "repeated_member_count": repeated_count,
            "repeated_member_ratio": repeated_count / measured if measured else None,
            "duplicate_pair_count": duplicate_pairs,
            "measurable_pair_count": possible_pairs,
            "duplicate_pair_ratio": duplicate_pairs / possible_pairs if possible_pairs else None,
            "duplicate_groups": duplicate_groups,
            "unmeasured": missing,
        }
    return output


def summarize_fingerprints(records: list[Record]) -> dict[str, Any]:
    """Summarize all/valid subsets per method, retaining intended denominators.

    These are cross-image raw geometry diagnostics, not conditional diversity.
    Ratios use the explicitly reported measured subset; missing outputs are
    listed and are never assumed to have a unique fingerprint.
    """
    _check_records(records)
    steps = sorted({step for row in records for step in row.get("raw_fingerprints", {})}, key=float)
    if not steps:
        steps = [format(step, ".12g") for step in QUANTIZATION_STEPS]
    methods = {}
    for method in sorted({row["method"] for row in records}):
        rows = sorted((row for row in records if row["method"] == method), key=_record_key)
        methods[method] = {
            "intended_count": len(rows),
            "validity_counts": dict(Counter(row.get("validity", "unknown") for row in rows)),
            "raw_status_counts": dict(Counter(row.get("raw_status", "not_recorded") for row in rows)),
            "all": _fingerprint_subset(rows, steps),
            "valid": _fingerprint_subset([row for row in rows if row.get("validity") == "valid"], steps),
        }
    return {
        "schema_version": "raw_geometry_fingerprints_v1",
        "coordinate_policy": "stored_coordinates_no_alignment_or_renormalization",
        "scope": "cross_image_raw_point_set_repetition_not_conditional_diversity",
        "methods": methods,
    }


def _load_candidate(record: Record) -> tuple[Record, dict[str, np.ndarray]]:
    row = {key: record[key] for key in ("method", "image_id", "candidate_id")}
    row.update(raw_path=str(Path(record["raw_path"]).resolve()), validity=record.get("validity", "unknown"))
    points = {}
    try:
        with np.load(row["raw_path"], allow_pickle=False) as raw:
            for name in PRIMITIVES:
                try:
                    points[name] = _xyz_points(raw[f"pred_{name}"], name)
                    row[f"{name}_status"] = "ok"
                    row[f"{name}_point_count"] = len(points[name])
                except (KeyError, ValueError) as error:
                    row[f"{name}_status"] = "geometry_error"
                    row[f"{name}_error"] = str(error)
    except (OSError, ValueError, EOFError, BadZipFile) as error:
        status = "missing_file" if isinstance(error, FileNotFoundError) else "read_error"
        for name in PRIMITIVES:
            row[f"{name}_status"] = status
            row[f"{name}_error"] = str(error)
    row["raw_status"] = "ok" if len(points) == 2 else "partial" if points else "unavailable"
    return row, points


def _distance_summary(pair_rows: list[Record], name: str) -> dict[str, Any]:
    values = [row[f"{name}_cd_sq"] for row in pair_rows if row[f"{name}_cd_sq"] is not None]
    status = "complete" if values and len(values) == len(pair_rows) else "partial" if values else "not_available"
    return {
        "status": status,
        "intended_pair_count": len(pair_rows),
        "measured_pair_count": len(values),
        "missing_pair_count": len(pair_rows) - len(values),
        "mean_pairwise_cd_sq": float(np.mean(values)) if values else None,
    }


def _evaluate_group(records: list[Record]) -> tuple[Record, list[Record], list[Record]]:
    loaded = [_load_candidate(record) for record in records]
    trees = [{name: cKDTree(points) for name, points in item[1].items()} for item in loaded]
    pair_rows = []
    for left, right in combinations(range(len(loaded)), 2):
        first, first_points = loaded[left]
        second, second_points = loaded[right]
        pair: Record = {
            "method": first["method"], "image_id": first["image_id"],
            "candidate_a": first["candidate_id"], "candidate_b": second["candidate_id"],
            "raw_path_a": first["raw_path"], "raw_path_b": second["raw_path"],
            "validity_a": first["validity"], "validity_b": second["validity"],
            "both_valid": first["validity"] == second["validity"] == "valid",
        }
        for name in PRIMITIVES:
            pair[f"{name}_cd_sq"] = None
            if name not in first_points or name not in second_points:
                pair[f"{name}_status"] = "missing_geometry"
                continue
            forward = trees[right][name].query(first_points[name], workers=1)[0]
            backward = trees[left][name].query(second_points[name], workers=1)[0]
            pair[f"{name}_cd_sq"] = float(np.mean(forward**2) + np.mean(backward**2))
            pair[f"{name}_status"] = "ok"
        pair_rows.append(pair)
    candidates = [item[0] for item in loaded]
    group: Record = {
        "method": records[0]["method"], "image_id": records[0]["image_id"],
        "intended_candidate_count": len(records),
        "valid_candidate_count": sum(row["validity"] == "valid" for row in candidates),
        "candidate_ids": [row["candidate_id"] for row in candidates],
        "status": "not_available" if len(records) < 2 else "complete",
        "reason": "fewer_than_two_intended_candidates" if len(records) < 2 else None,
    }
    for subset, pairs in (("all", pair_rows), ("valid", [row for row in pair_rows if row["both_valid"]])):
        group[subset] = {name: _distance_summary(pairs, name) for name in PRIMITIVES}
    if len(records) >= 2 and any(group["all"][name]["status"] != "complete" for name in PRIMITIVES):
        group["status"] = "partial" if any(group["all"][name]["measured_pair_count"] for name in PRIMITIVES) else "not_available"
        group["reason"] = "missing_or_invalid_raw_geometry"
    return group, candidates, pair_rows


def _summarize_groups(groups: list[Record]) -> dict[str, Any]:
    output: dict[str, Any] = {"intended_image_count": len(groups)}
    for subset in ("all", "valid"):
        output[subset] = {}
        for name in PRIMITIVES:
            items = [group[subset][name] for group in groups]
            values = [item["mean_pairwise_cd_sq"] for item in items if item["measured_pair_count"]]
            complete = [item["mean_pairwise_cd_sq"] for item in items if item["status"] == "complete"]
            output[subset][name] = {
                "status": "ok" if values else "not_available",
                "measured_image_count": len(values),
                "complete_image_count": len(complete),
                "partial_image_count": sum(item["status"] == "partial" for item in items),
                "intended_pair_count": sum(item["intended_pair_count"] for item in items),
                "measured_pair_count": sum(item["measured_pair_count"] for item in items),
                "mean_pairwise_cd_sq": float(np.mean(values)) if values else None,
                "complete_images_mean_pairwise_cd_sq": float(np.mean(complete)) if complete else None,
            }
    return output


def evaluate_conditional_diversity(records: list[Record], *, workers: int = 4) -> dict[str, Any]:
    """Evaluate all within-image pairs without generating or selecting samples.

    Each manifest row is one intended candidate. Missing outputs must remain in
    the manifest. Each image has equal weight in method means, including a
    separately reported mean restricted to complete groups. A K=1 group has
    null means, never zero diversity. Valid-pair metrics condition on validity;
    their coverage is reported separately from the all-candidate calculation.
    """
    _check_records(records)
    if workers < 1:
        raise ValueError("workers must be positive")
    grouped: dict[tuple[str, str], list[Record]] = defaultdict(list)
    for record in sorted(records, key=_record_key):
        grouped[(record["method"], record["image_id"])].append(record)
    jobs = list(grouped.values())
    if workers == 1:
        evaluated = [_evaluate_group(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            evaluated = list(pool.map(_evaluate_group, jobs))
    groups = [item[0] for item in evaluated]
    return {
        "schema_version": "conditional_raw_diversity_v1",
        "distance_definition": "mean(A_to_B_squared_nearest) + mean(B_to_A_squared_nearest)",
        "coordinate_policy": "stored_coordinates_no_alignment_or_renormalization",
        "sampling": "all_stored_raw_surface_and_curve_points_no_resampling",
        "image_weighting": "equal_weight_for_images_with_a_measured_pair; complete_groups_also_reported",
        "methods": {
            method: _summarize_groups([group for group in groups if group["method"] == method])
            for method in sorted({group["method"] for group in groups})
        },
        "group_rows": groups,
        "candidate_rows": [row for item in evaluated for row in item[1]],
        "pair_rows": [row for item in evaluated for row in item[2]],
    }


def _flatten_group(group: Record) -> Record:
    row = {key: value for key, value in group.items() if key not in ("all", "valid", "candidate_ids")}
    row["candidate_ids"] = ",".join(group["candidate_ids"])
    for subset in ("all", "valid"):
        for name in PRIMITIVES:
            row.update({f"{subset}_{name}_{key}": value for key, value in group[subset][name].items()})
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True, help="JSON list / {records: [...]} or JSONL of all intended candidates")
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory for JSON and per-candidate/pair/image CSV")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--quantization-steps", type=float, nargs="+", default=QUANTIZATION_STEPS)
    args = parser.parse_args()
    source = args.manifest.resolve()
    content = source.read_text(encoding="utf-8")
    if source.suffix == ".jsonl":
        records = [json.loads(line) for line in content.splitlines() if line.strip()]
    else:
        payload = json.loads(content)
        records = payload["records"] if isinstance(payload, dict) else payload
    for record in records:
        path = Path(record["raw_path"])
        record["raw_path"] = str((source.parent / path).resolve() if not path.is_absolute() else path)
    _check_records(records)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    result = evaluate_conditional_diversity(records, workers=args.workers)
    for row in result["candidate_rows"]:
        row["raw_fingerprints"] = (
            geometry_fingerprints(row["raw_path"], tuple(args.quantization_steps)) if row["raw_status"] == "ok" else {}
        )
    write_json(args.output_dir / "manifest.json", records)
    write_json(args.output_dir / "diversity.json", result)
    write_json(args.output_dir / "fingerprints.json", summarize_fingerprints(result["candidate_rows"]))
    write_csv(args.output_dir / "candidates.csv", [
        {key: value for key, value in row.items() if key != "raw_fingerprints"} for row in result["candidate_rows"]
    ])
    write_csv(args.output_dir / "pairs.csv", result["pair_rows"])
    write_csv(args.output_dir / "images.csv", [_flatten_group(group) for group in result["group_rows"]])


if __name__ == "__main__":
    main()
