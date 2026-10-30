"""Raw and native STEP metrics for offline diffusion preference candidates.

OCC runs in an isolated process. A single deadline covers raw metrics,
``recover_v1``, native validity, and repeated STEP geometry measurements.
"""

from __future__ import annotations

from functools import lru_cache
import hashlib
import multiprocessing as mp
import os
from pathlib import Path
import random
import tempfile
import time
import traceback

from tools import vae_compare_post as post


METRIC_VERSION = "svr_preference_metrics_v1"
FACE_MATCH_THRESHOLD = 0.1


def metric_seed(seed: int, image_id: str, stage: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{image_id}:{stage}".encode()).digest()[:4], "little")


def _read_arrays(path: Path) -> dict:
    import numpy as np
    with np.load(path, allow_pickle=False) as arrays:
        return {name: arrays[name] for name in arrays.files}


def _points(array, rank: int):
    import numpy as np
    value = np.asarray(array)
    if value.ndim != rank or value.shape[-1] not in (3, 6) or not value.size:
        raise ValueError(f"Expected nonempty rank-{rank} XYZ or XYZ+normal array")
    value = np.asarray(value[..., :3], dtype=np.float64)
    if not np.isfinite(value).all():
        raise ValueError("Geometry contains NaN/Inf")
    return value


def face_detection(predicted, target) -> dict:
    """One-to-one face matches use the existing symmetric squared CD < 0.1."""
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    costs = np.array([
        [post.squared_chamfer(face.reshape(-1, 3), truth.reshape(-1, 3))["cd"]
         for truth in target] for face in predicted
    ])
    row, column = linear_sum_assignment(costs)
    true_positive = int((costs[row, column] < FACE_MATCH_THRESHOLD).sum())
    return {"face_f1": 2 * true_positive / (len(predicted) + len(target)),
            "face_precision": true_positive / len(predicted),
            "face_recall": true_positive / len(target),
            "face_true_positive": true_positive}


def score_geometry(pred_faces, pred_edges, gt_faces, gt_edges) -> dict:
    result = face_detection(pred_faces, gt_faces)
    for name, predicted, truth in (("face", pred_faces, gt_faces), ("edge", pred_edges, gt_edges)):
        values = post.squared_chamfer(predicted.reshape(-1, 3), truth.reshape(-1, 3))
        result.update({f"{name}_cd_sq": values["cd"],
                       f"{name}_pred_to_gt_sq": values["accuracy"],
                       f"{name}_gt_to_pred_sq": values["completeness"],
                       f"pred_{name}_count": len(predicted), f"gt_{name}_count": len(truth)})
    return result


def score_raw(raw_path: Path, gt_npz: Path) -> dict:
    predicted, truth = _read_arrays(raw_path), _read_arrays(gt_npz)
    return score_geometry(
        _points(predicted["pred_face"], 4), _points(predicted["pred_edge"], 3),
        _points(truth["sample_points_faces"], 4), _points(truth["sample_points_lines"], 3))


def _post_input(raw_path: Path, destination: Path) -> None:
    """Keep native raw intact; adapt only exact integer-valued legacy indices."""
    import numpy as np
    arrays = _read_arrays(raw_path)
    connectivity = arrays["pred_edge_face_connectivity"]
    destination.parent.mkdir(parents=True)
    if np.issubdtype(connectivity.dtype, np.floating):
        if (not np.isfinite(connectivity).all() or
                not np.equal(connectivity, np.trunc(connectivity)).all() or
                not ((connectivity >= -(2 ** 63)) & (connectivity < 2 ** 63)).all()):
            raise ValueError("Legacy connectivity is not exactly representable by int64")
        arrays["pred_edge_face_connectivity"] = connectivity.astype(np.int64)
        np.savez_compressed(destination, **arrays)
    else:
        destination.symlink_to(raw_path.resolve())
    post.validate_raw_prediction(destination)


def _gt_samples(path: Path, cache_root: Path, image_id: str, seed: int) -> dict:
    """Concurrent candidates share the exact GT point sets for each repeat."""
    import numpy as np
    key = post.fingerprint({"step_sha256": post.sha256_file(path),
                            "sampling": post.SAMPLING, "seed": seed})
    cache = cache_root / f"{image_id}-{key}.npz"
    if cache.exists():
        return _read_arrays(cache)
    samples = post.sample_native_geometry(post.read_step_at_precision(path), seed)
    cache_root.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=cache_root, suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        np.savez_compressed(handle, **samples)
    os.replace(temporary, cache)
    return samples


def score_post(shape, gt_step: Path, cache_root: Path, image_id: str,
               candidate_id: str, repeat: int, seed: int) -> dict:
    gt_seed = metric_seed(seed, image_id, f"gt:{repeat}")
    pred_seed = metric_seed(seed, image_id, f"pred:{candidate_id}:{repeat}")
    truth = _gt_samples(gt_step, cache_root, image_id, gt_seed)
    predicted = post.sample_native_geometry(shape, pred_seed)
    result = score_geometry(
        predicted["face_points"].reshape(int(predicted["num_faces"]), -1, 3),
        predicted["edge_points"].reshape(int(predicted["num_edges"]), -1, 3),
        truth["face_points"].reshape(int(truth["num_faces"]), -1, 3),
        truth["edge_points"].reshape(int(truth["num_edges"]), -1, 3))
    return {"repeat": repeat, "pred_seed": pred_seed, "gt_seed": gt_seed, **result}


def _confirmed_no_solid(diagnostics: dict) -> bool:
    """Missing STEP is a geometry label only after all policy paths completed."""
    paths = diagnostics.get("paths", [])
    rejected = {"native_rejected", "incomplete_faces", "no_accepted_solid"}
    return (diagnostics.get("status") == "no_accepted_solid" and bool(paths) and
            all(path.get("status") in rejected for path in paths))


def _execute_candidate(job: dict, status_path: Path) -> dict:
    import numpy as np
    started = time.monotonic()
    output = Path(job["output_dir"])
    candidate_id = job["candidate_id"]
    row = {"image_id": job["image_id"], "candidate_id": candidate_id,
           "protocol_fingerprint": job["protocol_fingerprint"],
           "status": "running", "stage": "raw", "validity": "unknown",
           "raw": {}, "post": [], "error": ""}

    def record(**changes):
        row.update(changes)
        row["elapsed_seconds"] = time.monotonic() - started
        post.atomic_json(status_path, row)

    record()
    try:
        raw_path = Path(job["raw_path"])
        record(raw=score_raw(raw_path, Path(job["gt_npz"])))
        _post_input(raw_path, output / "input" / candidate_id / "data.npz")
    except Exception as exc:
        record(status="input_error", error=f"{type(exc).__name__}: {exc}")
        return row
    try:
        from src.postprocess.construct_brep import construct_brep_for_sample
        post_seed = metric_seed(job["seed"], job["image_id"], f"post:{candidate_id}")
        random.seed(post_seed)
        np.random.seed(post_seed)
        record(stage="post", post_seed=post_seed)
        construct_brep_for_sample(
            output / "input", output / "post", candidate_id,
            v_drop_num=0, is_ray=False, is_log=False, is_optimize_geom=True,
            v_max_optimize_iter=job["max_optimize_iter"], isdebug=False, use_cuda=False,
            from_scratch=True, is_save_data=False, post_policy="recover_v1",
            deadline_monotonic=started + job["timeout_seconds"])
        directory = output / "post" / candidate_id
        diagnostics = post.read_json(directory / "post_diagnostics.json")
        record(post_diagnostics=diagnostics, stage="native_validity")
        if diagnostics.get("status") == "budget_exhausted":
            record(status="timeout")
            return row
        step = directory / "recon_brep.step"
        if not step.is_file():
            record(status="no_step",
                   validity="invalid" if _confirmed_no_solid(diagnostics) else "unknown")
            return row
        shape = post.read_step_at_precision(step)
        native = post.native_validity(shape)
        record(native=native, step_path=str(step), step_sha256=post.sha256_file(step),
               validity="valid" if native["native_valid_solid"] else "invalid")
        # Invalid candidates need only deterministic raw metrics for pairing.
        if row["validity"] == "invalid":
            record(status="ok", stage="done")
            return row
        record(stage="post_metrics")
        for repeat in range(job["repeats"]):
            metrics = score_post(shape, Path(job["gt_step"]), Path(job["gt_cache_root"]),
                                 job["image_id"], candidate_id, repeat, job["seed"])
            record(post=[*row["post"], metrics])
        record(status="ok", stage="done")
    except Exception as exc:
        traceback.print_exc()
        record(status="evaluation_error", error=f"{type(exc).__name__}: {exc}")
    return row


def _candidate_process(job: dict, status_path: Path, log_path: Path) -> None:
    os.setsid()
    post.configure_cpu()
    with log_path.open("ab", buffering=0) as handle:
        os.dup2(handle.fileno(), 1)
        os.dup2(handle.fileno(), 2)
        _execute_candidate(job, status_path)


@lru_cache(maxsize=1)
def _code_hashes() -> dict:
    post_root = Path(__file__).resolve().parents[1] / "postprocess"
    paths = [Path(__file__), Path(post.__file__), *post_root.glob("*.py")]
    return {str(path): post.sha256_file(path) for path in sorted(paths)}


def evaluate_candidate(raw_path: str | Path, gt_npz: str | Path, gt_step: str | Path,
                       output_dir: str | Path, image_id: str, *, candidate_id: str = "c00",
                       timeout_seconds: float = 180, repeats: int = 3, seed: int = 42,
                       max_optimize_iter: int = 200, gt_cache_root: str | Path | None = None,
                       start_method: str = "spawn") -> dict:
    """Resume a matching terminal result, otherwise run one bounded CPU child.

    ``fork`` is intended only for an explicitly CPU-only parent that has called
    ``tools.vae_compare_post.preload_cpu_dependencies``. GPU callers use spawn.
    """
    if repeats < 2 or timeout_seconds <= 0:
        raise ValueError("Require repeated geometry measurements and a positive time budget")
    if Path(candidate_id).name != candidate_id or candidate_id in ("", ".", ".."):
        raise ValueError("candidate_id must be a single directory name")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    job = {
        "version": METRIC_VERSION, "raw_path": str(Path(raw_path).resolve()),
        "gt_npz": str(Path(gt_npz).resolve()), "gt_step": str(Path(gt_step).resolve()),
        "output_dir": str(output), "image_id": image_id, "candidate_id": candidate_id,
        "timeout_seconds": timeout_seconds, "repeats": repeats, "seed": seed,
        "max_optimize_iter": max_optimize_iter, "sampling": post.SAMPLING,
        "face_match_threshold": FACE_MATCH_THRESHOLD, "code_sha256": _code_hashes(),
        "gt_cache_root": str(Path(gt_cache_root or output / "gt_cache").resolve()),
    }
    job["input_sha256"] = {key: post.sha256_file(Path(job[key])) if Path(job[key]).is_file() else None
                           for key in ("raw_path", "gt_npz", "gt_step")}
    job["protocol_fingerprint"] = post.fingerprint(job)
    status_path = output / "metrics.json"
    previous = post.read_json(status_path)
    if previous:
        if previous.get("protocol_fingerprint") != job["protocol_fingerprint"]:
            raise ValueError(f"Candidate evaluation protocol or inputs changed: {output}")
        if previous.get("status") == "running":
            previous.update(status="interrupted", validity="unknown")
            post.atomic_json(status_path, previous)
        return previous
    post.atomic_json(output / "protocol.json", job)
    process = mp.get_context(start_method).Process(
        target=_candidate_process, args=(job, status_path, output / "worker.log"))
    started = time.monotonic()
    process.start()
    try:
        process.join(timeout=timeout_seconds)
        expired = process.is_alive()
        if expired:
            post.terminate_process(process)
        row = post.read_json(status_path) or {
            "image_id": image_id, "candidate_id": candidate_id,
            "protocol_fingerprint": job["protocol_fingerprint"], "raw": {}, "post": []}
        if expired:
            row.update(status="timeout", validity="unknown",
                       error=f"Whole-candidate time budget exceeded ({timeout_seconds:g}s)")
        elif process.exitcode != 0 or row.get("status") in (None, "running"):
            row.update(status="crash", validity="unknown", error=f"Worker exited {process.exitcode}")
        row.update(worker_exit_code=process.exitcode, wall_seconds=time.monotonic() - started)
        post.atomic_json(status_path, row)
        return row
    finally:
        if process.is_alive():
            post.terminate_process(process)
