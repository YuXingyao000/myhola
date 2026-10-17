"""Fixed-denominator validation metrics and isolated OCC post-processing.

Run from the repository root. This wrapper reuses the existing geometry and
post-processing algorithms; GT is read only by metric computation.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

# Direct script execution must resolve the repository's tools package.
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np
from scipy.optimize import linear_sum_assignment

from tools.vae_compare_metrics import squared_chamfer
from tools import vae_compare_post as post


ROOT = Path(__file__).resolve().parent
SPLIT_SHA256 = "6c952c3556860c0a2782a0bedbf139b48bc2769b385788baf0fcb82fac1be587"
RAW_METRICS = ("face_cd_sq", "edge_cd_sq", "face_f1", "face_count_abs_error")
POST_METRICS = ("face_cd_sq", "edge_cd_sq", "post_seconds", "wall_seconds")


def frozen_ids(split: Path) -> list[str]:
    data = split.read_bytes()
    ids = data.decode().splitlines()
    if (hashlib.sha256(data).hexdigest() != SPLIT_SHA256
            or len(ids) != 100 or len(set(ids)) != 100):
        raise ValueError("This experiment requires the frozen validation100.txt")
    return ids


def read_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def prepare_post_input(root: Path, method: str, sample_id: str) -> Path:
    """Normalize exact integer-valued legacy indices; never change geometry.

    Old AE inference exported connectivity as float32. Preserve that native
    export and keep a separately hashed, dtype-only adapter for the post API.
    Nonintegral/nonfinite indices remain invalid instead of being rounded.
    """
    source = root / "raw_predictions" / method / sample_id / "data.npz"
    target = root / "post_inputs" / method / sample_id / "data.npz"
    if not source.is_file():
        return target
    manifest = root / "metadata" / "adapters" / method / f"{sample_id}.json"
    source_sha = post.sha256_file(source)
    adapter_sha = post.sha256_file(Path(__file__))
    previous = post.read_json(manifest)
    if previous:
        if (previous["native_raw_sha256"] != source_sha or not target.is_file()
                or previous["post_input_sha256"] != post.sha256_file(target)
                or previous.get("adapter_code_sha256") != adapter_sha):
            raise ValueError(f"Changed native prediction or adapted export: {method}/{sample_id}")
        return target
    if target.exists():
        raise ValueError(f"Untracked adapted export: {target}")
    arrays = read_arrays(source)
    conn = arrays.get("pred_edge_face_connectivity")
    old_dtype = str(conn.dtype) if conn is not None else None
    converted = False
    if (conn is not None and np.issubdtype(conn.dtype, np.floating)
            and np.isfinite(conn).all() and np.equal(conn, np.trunc(conn)).all()
            and np.all(conn >= -(2 ** 63)) and np.all(conn < 2 ** 63)):
        integer_conn = conn.astype(np.int64)
        if not np.array_equal(integer_conn.astype(conn.dtype), conn):
            raise ValueError("Connectivity conversion must be lossless")
        arrays["pred_edge_face_connectivity"] = integer_conn
        converted = True
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, **arrays)
    post.atomic_json(manifest, {
        "native_raw_sha256": source_sha, "post_input_sha256": post.sha256_file(target),
        "adapter_version": "integer_connectivity_dtype_v1", "adapter_code_sha256": adapter_sha,
        "connectivity_source_dtype": old_dtype, "connectivity_converted_to_int64": converted,
        "geometry_changed": False, "index_values_changed": False,
    })
    return target


def valid_points(value: np.ndarray, kind: str) -> np.ndarray:
    """Validate independently, so bad edges cannot erase valid face metrics."""
    points = np.asarray(value)
    rank = 4 if kind == "face" else 3
    if points.ndim != rank or points.shape[-1] not in (3, 6):
        raise ValueError(f"Invalid {kind} shape: {points.shape}")
    if not len(points) or any(size < 2 for size in points.shape[1:-1]):
        raise ValueError(f"Empty or undersampled {kind} prediction")
    xyz = np.asarray(points[..., :3], dtype=np.float64)
    if not np.isfinite(xyz).all():
        raise ValueError(f"Nonfinite {kind} coordinates")
    return xyz


def face_detection(predicted: np.ndarray, truth: np.ndarray) -> dict[str, float | int]:
    """Primitive F1 uses Hungarian assignment and symmetric squared CD < .1."""
    costs = np.empty((len(predicted), len(truth)), dtype=np.float64)
    for i, face in enumerate(predicted):
        for j, target in enumerate(truth):
            costs[i, j] = squared_chamfer(face.reshape(-1, 3), target.reshape(-1, 3))[2]
    pred_ids, gt_ids = linear_sum_assignment(costs)
    tp = int((costs[pred_ids, gt_ids] < 0.1).sum())
    return {"face_f1": 2.0 * tp / (len(predicted) + len(truth)),
            "face_precision": tp / len(predicted), "face_recall": tp / len(truth),
            "face_true_positive": tp}


def score_raw(sample_id: str, method: str, root: Path, data_root: Path) -> dict[str, Any]:
    row: dict[str, Any] = {
        "sample_id": sample_id, "method": method, "raw_exists": False,
        "raw_complete": False, "post_input_valid": False,
        "native_export_integer_dtype": None, "status": "missing_raw", "face_status": "not_run",
        "edge_status": "not_run", "gt_status": "not_run", "error": "",
    }
    path = root / "raw_predictions" / method / sample_id / "data.npz"
    if not path.is_file():
        return row
    row.update(raw_exists=True, raw_sha256=post.sha256_file(path))
    try:
        prediction = read_arrays(path)
    except Exception as exc:
        row.update(status="unreadable_raw", error=f"{type(exc).__name__}: {exc}")
        return row
    conn = prediction.get("pred_edge_face_connectivity")
    row["native_export_integer_dtype"] = bool(conn is not None and np.issubdtype(conn.dtype, np.integer))
    # raw_complete means usable after the recorded lossless dtype adapter.
    # The native NPZ remains untouched and supplies every geometry metric below.
    try:
        adapted = prepare_post_input(root, method, sample_id)
        row.update(post.validate_raw_prediction(adapted), raw_complete=True, post_input_valid=True,
                   post_input_sha256=post.sha256_file(adapted))
        row["status"] = "ok"
    except Exception as exc:
        row.update(status="invalid_raw", error=f"{type(exc).__name__}: {exc}")
    try:
        gt_path = data_root / sample_id / "data.npz"
        truth = read_arrays(gt_path)
        row.update(gt_status="ok", gt_sha256=post.sha256_file(gt_path))
    except Exception as exc:
        row.update(gt_status="unreadable_gt", gt_error=f"{type(exc).__name__}: {exc}")
        return row
    for kind, gt_key in (("face", "sample_points_faces"), ("edge", "sample_points_lines")):
        try:
            predicted = valid_points(prediction[f"pred_{kind}"], kind)
            target = valid_points(truth[gt_key], kind)
            forward, backward, total = squared_chamfer(predicted.reshape(-1, 3), target.reshape(-1, 3))
            row.update({f"{kind}_status": "ok", f"{kind}_cd_sq": total,
                        f"{kind}_pred_to_gt_sq": forward, f"{kind}_gt_to_pred_sq": backward,
                        f"pred_{kind}_count": len(predicted), f"gt_{kind}_count": len(target)})
            if kind == "face":
                row.update(face_detection(predicted, target))
                row["face_count_abs_error"] = abs(len(predicted) - len(target))
        except Exception as exc:
            row.update({f"{kind}_status": "error", f"{kind}_error": f"{type(exc).__name__}: {exc}"})
    return row


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted(set().union(*(row.keys() for row in rows)))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def raw_phase(args: argparse.Namespace, ids: list[str]) -> None:
    for method in args.methods:
        with ThreadPoolExecutor(max_workers=16) as executor:
            rows = list(executor.map(lambda sample: score_raw(sample, method, args.root, args.data_root), ids))
        destination = args.root / "metadata" / "raw_metrics" / f"{method}.json"
        post.atomic_json(destination, {"split_sha256": SPLIT_SHA256, "intended_count": len(ids),
                                       "evaluator_sha256": post.sha256_file(Path(__file__)), "rows": rows})
        print(json.dumps({"method": method, "raw_complete": sum(r["raw_complete"] for r in rows),
                          "intended": len(ids)}), flush=True)


def post_protocol(args: argparse.Namespace) -> dict:
    return {
        "version": "svr_comparison_20261017_v1", "source": "svr_conditioned_ddpm",
        "root": str(args.root), "data_root": str(args.data_root), "split_sha256": SPLIT_SHA256,
        "intended_count": 100, "mode": "optimized", "post_policy": "legacy",
        "drop_num": 0, "max_optimize_iter": 200, "adaptive_max_optimize_iter": 600,
        "use_cuda": False, "threads_per_process": 1, "workers": 16,
        "timeout_seconds": 180.0, "timeout_scope": "whole subprocess including geometry",
        "sampling": post.SAMPLING, "native_validity": "native read, TopAbs_SOLID and BRepCheck",
        "post_algorithm_files": {str(p.relative_to(REPO_ROOT)): post.sha256_file(p)
                                 for p in sorted((REPO_ROOT / "src/brepnet/post").glob("*.py"))},
        "runner_sha256": post.sha256_file(Path(post.__file__)),
        "adapter": "lossless integer-valued float connectivity to int64; native raw retained",
        "evaluator_sha256": post.sha256_file(Path(__file__)),
    }


def post_phase(args: argparse.Namespace, ids: list[str]) -> None:
    """Only new jobs execute; algorithm failures and interrupted jobs never retry."""
    import fcntl

    metadata = args.root / "metadata" / "post"
    metadata.mkdir(parents=True, exist_ok=True)
    protocol = post_protocol(args)
    digest = post.fingerprint(protocol)
    with (metadata / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        old_protocol = post.read_json(metadata / "protocol.json")
        if old_protocol and old_protocol != protocol:
            raise ValueError("Post protocol/code changed: use a fresh experiment root")
        post.atomic_json(metadata / "protocol.json", protocol)
        jobs = []
        for method in args.methods:
            for sample_id in ids:
                raw_root = args.root / "post_inputs" / method
                raw = prepare_post_input(args.root, method, sample_id)
                raw_sha = post.sha256_file(raw) if raw.is_file() else None
                status_path = metadata / "samples" / method / f"{sample_id}.json"
                previous = post.read_json(status_path)
                if previous:
                    if previous.get("protocol_fingerprint") != digest or previous.get("raw_sha256") != raw_sha:
                        raise ValueError(f"Changed raw/protocol for {method}/{sample_id}; refusing result mixing")
                    if previous.get("status") not in post.TERMINAL:
                        previous.update(status="interrupted", error="Previous process ended without terminal state; no retry")
                        post.atomic_json(status_path, previous)
                    continue
                jobs.append({"sample_id": sample_id, "method": method, "mode": "optimized",
                             "post_policy": "legacy", "timeout_seconds": 180.0,
                             "raw_root": str(raw_root), "raw_sha256": raw_sha,
                             "post_root": str(args.root / "post" / method),
                             "data_root": str(args.data_root),
                             "cache_root": str(args.root / "metadata" / "gt_step_samples"),
                             "max_optimize_iter": 200, "protocol_fingerprint": digest})
        print(f"Post: {len(jobs)} pending jobs, 16 workers, 180s whole-object budget", flush=True)
        if jobs:
            post.preload_cpu_dependencies()
            post.run_isolated_jobs(jobs, metadata, workers=16, timeout=180.0)


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.number)) and not isinstance(value, bool) and math.isfinite(float(value))


def numeric_stats(values: list[float]) -> dict:
    values = [float(v) for v in values if is_number(v)]
    if not values:
        return {"n": 0, "mean": None, "median": None, "p95": None}
    return {"n": len(values), "mean": float(np.mean(values)), "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95))}


def metric_comparisons(rows: dict[str, list[dict]], metrics: tuple[str, ...]) -> dict:
    """Each metric has its own denominator, plus common and pairwise ID sets."""
    result = {}
    methods = list(rows)
    for metric in metrics:
        values = {method: {r["sample_id"]: float(r[metric]) for r in items if is_number(r.get(metric))}
                  for method, items in rows.items()}
        common = sorted(set.intersection(*(set(v) for v in values.values())))
        pairs = {}
        for index, left in enumerate(methods):
            for right in methods[index + 1:]:
                paired = sorted(set(values[left]) & set(values[right]))
                pairs[f"{left}__{right}"] = {
                    "ids": paired, "n": len(paired), "left": left, "right": right,
                    "left_stats": numeric_stats([values[left][i] for i in paired]),
                    "right_stats": numeric_stats([values[right][i] for i in paired]),
                    "right_minus_left": numeric_stats([values[right][i] - values[left][i] for i in paired]),
                }
        result[metric] = {"own": {m: numeric_stats(list(v.values())) for m, v in values.items()},
                          "common_ids": common, "common_n": len(common),
                          "common": {m: numeric_stats([v[i] for i in common]) for m, v in values.items()},
                          "paired": pairs}
    return result


def summarize(ids: list[str], raw: dict[str, list[dict]], processed: dict[str, list[dict]]) -> dict:
    """The intended IDs, never the existing directories, define all coverage rates."""
    summary: dict[str, Any] = {"intended_count": len(ids), "methods": {}}
    for method in raw:
        r, p = raw[method], processed[method]
        for stage, records in (("raw", r), ("post", p)):
            if len(records) != len(ids) or {v["sample_id"] for v in records} != set(ids):
                raise ValueError(f"{method}/{stage} does not cover exactly the intended IDs")
        counts = {"raw_files": sum(bool(x.get("raw_exists")) for x in r),
                  "raw_complete": sum(bool(x.get("raw_complete")) for x in r),
                  "step_outputs": sum(bool(x.get("step_exists")) for x in p),
                  "readable_steps": sum(bool(x.get("step_readable")) for x in p),
                  "native_valid_solids": sum(bool(x.get("native_valid_solid")) for x in p),
                  "compat_valid_solids": sum(bool(x.get("compat_valid_solid")) for x in p),
                  "post_timeouts": sum(x.get("status") == "timeout" for x in p)}
        summary["methods"][method] = {
            "counts": counts, "rates": {key: value / len(ids) for key, value in counts.items()},
            "raw_status_counts": dict(Counter(x.get("status", "not_run") for x in r)),
            "post_status_counts": dict(Counter(x.get("status", "not_run") for x in p)),
        }
    summary["raw_metrics"] = metric_comparisons(raw, RAW_METRICS)
    summary["post_metrics"] = metric_comparisons(processed, POST_METRICS)
    return summary


def report_phase(args: argparse.Namespace, ids: list[str]) -> None:
    raw, processed, inference = {}, {}, {}
    for method in args.methods:
        payload = post.read_json(args.root / "metadata" / "raw_metrics" / f"{method}.json")
        if payload.get("split_sha256") != SPLIT_SHA256:
            raise ValueError(f"Missing or incompatible raw metrics for {method}")
        raw[method] = payload["rows"]
        raw_by_id = {r["sample_id"]: r for r in raw[method]}
        processed[method] = []
        inference[method] = []
        for sample_id in ids:
            inference_row = post.read_json(
                args.root / "metadata" / "inference" / method / f"{sample_id}.json")
            inference_row.update(sample_id=sample_id, method=method)
            inference_row.setdefault("status", "not_run")
            inference[method].append(inference_row)
            raw_path = args.root / "raw_predictions" / method / sample_id / "data.npz"
            raw_sha = post.sha256_file(raw_path) if raw_path.is_file() else None
            if raw_by_id[sample_id].get("raw_sha256") != raw_sha:
                raise ValueError(f"Raw changed since scoring: {method}/{sample_id}")
            if (inference_row.get("status") == "ok" and inference_row.get("output_sha256")
                    and inference_row["output_sha256"] != raw_sha):
                raise ValueError(f"Raw does not match inference record: {method}/{sample_id}")
            row = post.read_json(args.root / "metadata" / "post" / "samples" / method / f"{sample_id}.json")
            if not row:
                row = {"sample_id": sample_id, "method": method, "status": "not_run"}
            elif row.get("raw_sha256") != (post.sha256_file(args.root / "post_inputs" / method / sample_id / "data.npz")
                                           if raw_path.is_file() else None):
                raise ValueError(f"Raw does not match post record: {method}/{sample_id}")
            row = dict(row)
            row["native_raw_sha256"] = raw_sha
            row["step_exists"] = (args.root / "post" / method / sample_id / "recon_brep.step").is_file()
            for old, new in (("face_cd", "face_cd_sq"), ("edge_cd", "edge_cd_sq"),
                             ("post_elapsed_seconds", "post_seconds")):
                if old in row:
                    row[new] = row[old]
            processed[method].append(row)
    summary = summarize(ids, raw, processed)
    summary["inference_metrics"] = metric_comparisons(inference, ("inference_seconds",))
    summary["all_post_terminal"] = all(r.get("status") in post.TERMINAL
                                        for rows in processed.values() for r in rows)
    summary["all_inference_terminal"] = all(r.get("status") in {"ok", "error"}
                                             for rows in inference.values() for r in rows)
    for method, rows in inference.items():
        counts = dict(Counter(r["status"] for r in rows))
        summary["methods"][method]["inference_status_counts"] = counts
        summary["methods"][method]["inference_success_count"] = counts.get("ok", 0)
        summary["methods"][method]["inference_success_rate"] = counts.get("ok", 0) / len(ids)
    summary.update(split_sha256=SPLIT_SHA256, raw_f1_threshold_cd_sq=0.1,
                   post_protocol=post.read_json(args.root / "metadata" / "post" / "protocol.json"),
                   evaluator_sha256=post.sha256_file(Path(__file__)),
                   limitations=["Validation100, K=1: checkpoint/system comparison, not equal-budget architecture ablation.",
                                "raw_complete/post_input_valid allow the recorded lossless integer-valued float connectivity cast; native raw remains unchanged.",
                                "Raw grid CD and trimmed native STEP CD use different sampling and are separate metrics.",
                                "Missing/failed geometry remains in coverage denominators; numerical means use explicit computable sets.",
                                "Native solid validity uses stored tolerances; it is not an engineering precision guarantee."])
    destination = args.root / "report"
    post.atomic_json(destination / "summary.json", summary)
    write_csv(destination / "raw_per_object.csv", [r for rows in raw.values() for r in rows])
    write_csv(destination / "post_per_object.csv", [r for rows in processed.values() for r in rows])
    write_csv(destination / "inference_per_object.csv", [r for rows in inference.values() for r in rows])
    print(json.dumps({"report": str(destination), "methods": summary["methods"]}, indent=2), flush=True)


def self_check() -> None:
    """Known distances, unmatched primitives, invalid geometry and denominator tests."""
    plane = np.zeros((1, 2, 2, 3), dtype=np.float64)
    translated = plane.copy()
    translated[..., 0] = 0.2
    assert abs(squared_chamfer(plane.reshape(-1, 3), translated.reshape(-1, 3))[2] - 0.08) < 1e-12
    assert face_detection(plane, plane)["face_f1"] == 1.0
    far = plane.copy()
    far[..., 0] = 1.0
    assert face_detection(far, plane)["face_f1"] == 0.0
    assert face_detection(np.concatenate([plane, far]), plane)["face_f1"] == 2.0 / 3.0
    ids = ["a", "b", "c"]
    raw = {"left": [{"sample_id": i, "raw_exists": i != "c", "raw_complete": i != "c",
                     **({"face_cd_sq": n} if i != "c" else {})} for i, n in zip(ids, (1.0, 3.0, 0.0))],
           "right": [{"sample_id": i, "raw_exists": True, "raw_complete": True,
                      **({"face_cd_sq": n} if i != "b" else {})} for i, n in zip(ids, (2.0, 0.0, 4.0))]}
    processed = {m: [{"sample_id": i, "native_valid_solid": i == "a",
                      "status": "timeout" if i == "c" else "ok"} for i in ids] for m in raw}
    result = summarize(ids, raw, processed)
    assert result["methods"]["left"]["rates"]["raw_complete"] == 2 / 3
    assert result["methods"]["left"]["rates"]["native_valid_solids"] == 1 / 3
    assert result["raw_metrics"]["face_cd_sq"]["common_ids"] == ["a"]
    assert result["raw_metrics"]["face_cd_sq"]["common"]["left"]["mean"] == 1.0
    assert result["raw_metrics"]["face_cd_sq"]["paired"]["left__right"]["right_minus_left"]["mean"] == 1.0
    for points in (np.empty((0, 2, 2, 3)), np.full((1, 2, 2, 3), np.nan)):
        try:
            valid_points(points, "face")
        except ValueError:
            pass
        else:
            raise AssertionError("Bad geometry was accepted")
    # Exercise real NPZ loading and failure preservation without OCC or GPU work.
    import tempfile
    with tempfile.TemporaryDirectory(prefix="svr-comparison-check-") as directory:
        root = Path(directory)
        edge = np.zeros((1, 2, 3))

        def write_sample(sample_id, conn, predicted_edge=edge):
            raw_dir = root / "raw_predictions" / "method" / sample_id
            truth_dir = root / "truth" / sample_id
            raw_dir.mkdir(parents=True)
            truth_dir.mkdir(parents=True)
            np.savez(truth_dir / "data.npz", sample_points_faces=plane, sample_points_lines=edge)
            np.savez(raw_dir / "data.npz", pred_face=plane, pred_edge=predicted_edge,
                     pred_edge_face_connectivity=conn, pred_face_adj_prob=np.asarray([np.nan]))
            return raw_dir / "data.npz"

        write_sample("example", np.asarray([[0, 0, 0]], dtype=np.int64))
        scored = score_raw("example", "method", root, root / "truth")
        assert scored["raw_complete"] and scored["face_cd_sq"] == 0.0 and scored["face_f1"] == 1.0
        assert scored["native_export_integer_dtype"] and scored["post_input_valid"]
        assert score_raw("missing", "method", root, root / "truth")["status"] == "missing_raw"
        write_sample("bad_edge", np.asarray([[0, 0, 0]], dtype=np.int64), np.full_like(edge, np.nan))
        scored = score_raw("bad_edge", "method", root, root / "truth")
        assert not scored["raw_complete"] and scored["face_cd_sq"] == 0.0
        assert scored["edge_status"] == "error" and "edge_cd_sq" not in scored

        original = write_sample("integer_float", np.asarray([[0, 0, 0]], dtype=np.float32))
        before = original.read_bytes()
        scored = score_raw("integer_float", "method", root, root / "truth")
        assert scored["post_input_valid"] and not scored["native_export_integer_dtype"]
        adapted = prepare_post_input(root, "method", "integer_float")
        assert original.read_bytes() == before
        native, converted = read_arrays(original), read_arrays(adapted)
        assert converted["pred_edge_face_connectivity"].dtype == np.int64
        for key in native:
            assert np.array_equal(native[key], converted[key], equal_nan=True)
            if key != "pred_edge_face_connectivity":
                assert native[key].dtype == converted[key].dtype
        for name, value in (("fractional", 0.5), ("nonfinite", np.nan),
                            ("infinite", np.inf), ("out_of_int64", float(2 ** 63))):
            write_sample(name, np.asarray([[value, 0, 0]], dtype=np.float64))
            scored = score_raw(name, "method", root, root / "truth")
            assert not scored["post_input_valid"] and scored["status"] == "invalid_raw"
            assert scored["face_cd_sq"] == 0.0
            assert read_arrays(root / "post_inputs" / "method" / name / "data.npz")["pred_edge_face_connectivity"].dtype == np.float64
    print("Self-check passed: CD/F1, failures/denominators, lossless legacy indices, nonintegral/nonfinite/range rejection.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("raw", "post", "report", "all", "self-check"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--data-root", type=Path, default=Path("/mnt/d/data/deepcad_v7"))
    parser.add_argument("--methods", nargs="+", default=[])
    args = parser.parse_args()
    if args.phase == "self-check":
        self_check()
        return
    if not args.methods or len(set(args.methods)) != len(args.methods):
        parser.error("Provide unique --methods labels")
    if any(Path(m).name != m or m in {".", ".."} for m in args.methods):
        parser.error("Method labels must be individual directory names")
    args.root, args.data_root = args.root.resolve(), args.data_root.resolve()
    ids = frozen_ids(args.root / "validation100.txt")
    for phase, fn in (("raw", raw_phase), ("post", post_phase), ("report", report_phase)):
        if args.phase in (phase, "all"):
            fn(args, ids)


if __name__ == "__main__":
    main()
