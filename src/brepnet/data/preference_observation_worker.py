"""CPU-only observation measurements over immutable saved DPO candidates.

The coordinator isolates each CAD in a bounded process. This worker verifies
its saved inputs before reusing any result and checkpoints after each candidate;
it never resamples diffusion, reencodes a latent or runs B-Rep repair.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import time

import numpy as np
from PIL import Image

from src.brepnet.diffusion.image_loading import load_image_view
from src.brepnet.eval.metrics import observation as metrics
from tools.vae_compare_post import atomic_json, fingerprint, read_json, sha256_file


class SourceIntegrityError(ValueError):
    """A recorded immutable source changed or its provenance is inconsistent."""


def _json(path: Path, *, required: bool = False) -> dict:
    try:
        result = read_json(path)
    except (ValueError, OSError) as error:
        raise SourceIntegrityError(f"Cannot read source record {path}: {error}") from error
    if not isinstance(result, dict) or (required and not result):
        raise SourceIntegrityError(f"Missing or malformed source record: {path}")
    return result


def _record_file(path: Path, hashes: dict, expected: str | None = None) -> str | None:
    """Hash each physical input once per CAD, including optional absent files."""
    name = str(path)
    if name not in hashes:
        try:
            hashes[name] = sha256_file(path) if path.is_file() else None
        except OSError as error:
            raise SourceIntegrityError(f"Cannot hash source {path}: {error}") from error
    value = hashes[name]
    if expected is not None and value != expected:
        raise SourceIntegrityError(f"Recorded source SHA256 mismatch: {path}")
    return value


def _identity(record: dict, image_id: str, candidate_id: str, path: Path) -> None:
    if record and (record.get("image_id") != image_id or
                   record.get("candidate_id") != candidate_id):
        raise SourceIntegrityError(f"Source candidate identity mismatch: {path}")


def _candidate(
    directory: Path, image_id: str, candidate_id: str, common_hashes: dict,
    gt_npz: Path, gt_step: Path,
) -> tuple[dict, dict]:
    hashes = dict(common_hashes)
    inference_path = directory / "inference.json"
    metric_path = directory / "evaluation/metrics.json"
    eval_path = directory / "evaluation/protocol.json"
    inference, measured, evaluation = (_json(path) for path in
                                       (inference_path, metric_path, eval_path))
    for path in (inference_path, metric_path, eval_path):
        _record_file(path, hashes)
    for record, path in ((inference, inference_path), (measured, metric_path),
                         (evaluation, eval_path)):
        _identity(record, image_id, candidate_id, path)
    output_hashes = inference.get("output_sha256", {})
    if not isinstance(output_hashes, dict):
        raise SourceIntegrityError(f"Malformed inference hashes: {inference_path}")
    if inference.get("status") == "ok" and not all(
        output_hashes.get(name) for name in ("latent.npy", "data.npz", "mappings.npz")
    ):
        raise SourceIntegrityError(f"Successful inference lacks output hashes: {inference_path}")
    for name, expected in output_hashes.items():
        if Path(name).name != name or name in ("", ".", ".."):
            raise SourceIntegrityError(f"Unsafe inference output name: {inference_path}")
        _record_file(directory / name, hashes, expected)
    for name in ("latent.npy", "data.npz", "mappings.npz"):
        _record_file(directory / name, hashes)
    if evaluation:
        saved = evaluation.get("protocol_fingerprint")
        body = {key: value for key, value in evaluation.items() if key != "protocol_fingerprint"}
        expected_inputs = {
            "raw_path": hashes[str(directory / "data.npz")],
            "gt_npz": hashes[str(gt_npz)], "gt_step": hashes[str(gt_step)],
        }
        if (saved != fingerprint(body) or evaluation.get("input_sha256") != expected_inputs or
                (measured and measured.get("protocol_fingerprint") != saved)):
            raise SourceIntegrityError(f"Evaluation protocol or inputs mismatch: {eval_path}")
        for name, path in (("raw_path", directory / "data.npz"),
                           ("gt_npz", gt_npz), ("gt_step", gt_step)):
            if Path(evaluation.get(name, "")).resolve() != path:
                raise SourceIntegrityError(f"Evaluation source path mismatch: {eval_path}: {name}")
    elif measured and measured.get("protocol_fingerprint"):
        raise SourceIntegrityError(f"Missing recorded evaluation protocol: {eval_path}")

    row = deepcopy(measured or inference)
    row.update(image_id=image_id, candidate_id=candidate_id)
    row.setdefault("status", "source_missing")
    row.setdefault("validity", "unknown")
    row.setdefault("raw", {})
    row["source_metrics_path"] = str(metric_path)
    row["source_inference_status"] = inference.get("status", "missing")
    latent_path = directory / "latent.npy"
    row["latent_status"] = "missing"
    row["latent_path"] = None
    if hashes[str(latent_path)] is not None and inference.get("status") == "ok":
        try:
            latent = np.load(latent_path, allow_pickle=False)
            valid = latent.dtype == np.dtype("float32") and latent.shape == (30, 32)
            valid = valid and bool(np.isfinite(latent).all())
        except (ValueError, OSError, EOFError) as error:
            valid = False
            row["latent_error"] = str(error)
        row["latent_status"] = "ok" if valid else "invalid"
        if valid:
            row["latent_path"] = str(latent_path)
    step_name = measured.get("step_path")
    if step_name:
        step = Path(step_name).resolve()
        if not step.is_relative_to(directory / "evaluation"):
            raise SourceIntegrityError(f"STEP is outside candidate evaluation: {step}")
        if not measured.get("step_sha256"):
            raise SourceIntegrityError(f"Recorded STEP lacks SHA256: {metric_path}")
        _record_file(step, hashes, measured["step_sha256"])
        row["step_path"] = str(step)
    else:
        row["step_path"] = None
    row["source_fingerprint"] = fingerprint(hashes)
    return row, hashes


def _verify_sources(source_run: Path, split: str, image_id: str, protocol: dict) -> tuple:
    source_root = source_run / "candidates" / split / image_id
    gt_root = Path(protocol["raw_data_root"]).resolve() / image_id
    gt_npz, gt_step = gt_root / "data.npz", gt_root / "normalized_shape.step"
    image_path = Path(protocol["condition_data_root"]).resolve() / image_id / "imgs.npz"
    source_path = source_root / "sources.json"
    sources = _json(source_path, required=True)
    hashes = {}
    _record_file(source_path, hashes)
    for path in (gt_npz, gt_step, image_path):
        if not isinstance(sources.get(str(path)), str):
            raise SourceIntegrityError(f"Missing source SHA256: {source_path}: {path}")
        _record_file(path, hashes, sources[str(path)])
    cache = Path(protocol.get("gt_cache_root", source_run / "gt_cache")).resolve() / f"{image_id}_0"
    cache_path = cache / "source.json"
    cached = _json(cache_path, required=True)
    _record_file(cache_path, hashes)
    if (cached.get("gt_npz_sha256") != hashes[str(gt_npz)] or
            cached.get("precision") != "float32" or cached.get("layout") != "mean32,std32"):
        raise SourceIntegrityError(f"GT cache provenance mismatch: {cache_path}")
    expected = cached.get("output_sha256", {}).get("features.npy")
    if not isinstance(expected, str):
        raise SourceIntegrityError(f"Missing GT cache SHA256: {cache_path}")
    _record_file(cache / "features.npy", hashes, expected)
    try:
        features = np.load(cache / "features.npy", allow_pickle=False)
        valid = (features.dtype == np.dtype("float32") and features.ndim == 2 and
                 0 < features.shape[0] <= 30 and features.shape[1] == 64 and
                 np.isfinite(features).all() and (features[:, 32:] >= 0).all())
    except (ValueError, OSError, EOFError) as error:
        raise SourceIntegrityError(f"Unreadable GT cache: {cache}: {error}") from error
    if not valid:
        raise SourceIntegrityError(f"Invalid GT mean/std cache: {cache}")
    rows, all_hashes = [], dict(hashes)
    for index in range(int(protocol["candidates"])):
        candidate_id = f"c{index:02d}"
        row, candidate_hashes = _candidate(
            source_root / candidate_id, image_id, candidate_id, hashes, gt_npz, gt_step)
        rows.append(row)
        all_hashes.update(candidate_hashes)
    return rows, all_hashes, gt_step, image_path


def evaluate_image(source_run: Path, output_dir: Path, split: str,
                   image_id: str, protocol: dict) -> dict:
    """Verify all saved sources, then measure/reuse every candidate for one CAD.

    ``protocol`` carries raw/condition roots, candidate count, fingerprint and
    explicit observation settings. All source paths are read-only. Integrity
    errors propagate; rendering/data failures become unavailable observations.
    """
    if split not in ("train", "validation") or Path(image_id).name != image_id or image_id in ("", ".", ".."):
        raise ValueError(f"Invalid split/image identity: {split}/{image_id}")
    started = time.monotonic()
    source_run, output_dir = Path(source_run).resolve(), Path(output_dir).resolve()
    rows, sources, gt_step, image_path = _verify_sources(source_run, split, image_id, protocol)
    dataset_fingerprint = protocol["fingerprint"]
    image_path_out = output_dir / "cases" / split / image_id / "metrics.json"
    previous_image = _json(image_path_out)
    source_fingerprint = fingerprint(sources)
    if previous_image and (previous_image.get("protocol_fingerprint") != dataset_fingerprint or
                           previous_image.get("source_fingerprint") != source_fingerprint):
        raise SourceIntegrityError(f"Cached image protocol or source changed: {image_path_out}")
    result = {
        "image_id": image_id, "split": split, "status": "running",
        "attempt_id": protocol.get("attempt_id"),
        "protocol_fingerprint": dataset_fingerprint,
        "source_fingerprint": source_fingerprint, "source_sha256": sources,
        "candidates": rows,
    }
    paths = []
    for row in rows:
        path = output_dir / "candidates" / split / image_id / row["candidate_id"] / "observation.json"
        paths.append(path)
        previous = _json(path)
        if previous:
            if (previous.get("dataset_protocol_fingerprint") != dataset_fingerprint or
                    previous.get("source_fingerprint") != row["source_fingerprint"]):
                raise SourceIntegrityError(f"Cached candidate protocol or source changed: {path}")
            if previous.get("observation", {}).get("status") in ("ok", "error", "timeout", "crash"):
                row["observation"] = previous["observation"]
        row["dataset_protocol_fingerprint"] = dataset_fingerprint
        row.setdefault("observation", {"status": "pending"})
    atomic_json(image_path_out, result)

    settings = protocol["observation"]
    camera = metrics.legacy_svr_camera(settings["resolution"], settings["view_id"])
    context_error = None
    gt_depth = input_mask = None
    pending = [row for row in rows if row["observation"]["status"] == "pending" and
               row["latent_status"] == "ok" and row.get("step_path")]
    if pending:
        try:
            image = load_image_view(image_path, settings["image_key"], settings["view_id"], read_single_view=True)
            input_mask = np.min(image, axis=-1) < settings["mask_threshold"]
            if input_mask.shape != (camera.resolution, camera.resolution):
                input_mask = np.asarray(Image.fromarray(input_mask).resize(
                    (camera.resolution, camera.resolution), Image.Resampling.NEAREST))
            if not input_mask.any():
                raise ValueError("Input image has no foreground")
            gt_depth = metrics.render_step(gt_step, camera, settings["mesh_deflection"])
        except Exception as error:
            context_error = f"{type(error).__name__}: {error}"
    for row, path in zip(rows, paths):
        if row["observation"]["status"] != "pending":
            continue
        candidate_started = time.monotonic()
        if row["latent_status"] != "ok":
            row["observation"] = {"status": "error", "stage": "latent",
                                  "error": f"Saved latent is {row['latent_status']}"}
        elif not row.get("step_path"):
            row["observation"] = {"status": "error", "stage": "step", "error": "No recorded STEP"}
        elif context_error:
            row["observation"] = {"status": "error", "stage": "gt_or_input", "error": context_error}
        else:
            row["observation"] = {"status": "running", "stage": "render"}
            atomic_json(image_path_out, result)
            try:
                predicted = metrics.render_step(Path(row["step_path"]), camera, settings["mesh_deflection"])
                row["observation"] = metrics.compare_depth_maps(predicted, gt_depth, input_mask, camera)
            except Exception as error:
                row["observation"] = {"status": "error", "stage": "candidate_render_or_metrics",
                                      "error": f"{type(error).__name__}: {error}"}
        row["observation"]["elapsed_seconds"] = time.monotonic() - candidate_started
        atomic_json(path, row)
        atomic_json(image_path_out, result)
    result["status"] = "complete"
    result["elapsed_seconds"] = time.monotonic() - started
    atomic_json(image_path_out, result)
    return result
