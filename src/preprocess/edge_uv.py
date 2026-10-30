"""NumPy-only contract for versioned, connectivity-row-aligned UV sidecars."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

SCHEMA_VERSION = 1
SAMPLE_COUNT = 16
UV_DOMAIN_TOLERANCE = 1e-5


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_source_geometry(path: str | Path) -> dict[str, np.ndarray]:
    """Read the existing geometry contract; corrupt source geometry is fatal."""
    try:
        with np.load(path, allow_pickle=False) as source:
            arrays = {key: source[key].copy() for key in (
                "sample_points_faces", "sample_points_lines", "edge_face_connectivity"
            )}
    except Exception as error:
        raise ValueError(f"{path}: unreadable source geometry: {type(error).__name__}: {error}") from error
    faces = arrays["sample_points_faces"]
    edges = arrays["sample_points_lines"]
    rows = arrays["edge_face_connectivity"]
    if faces.ndim != 4 or faces.shape[1:3] != (16, 16) or faces.shape[-1] < 3 or len(faces) == 0:
        raise ValueError(f"{path}: expected nonempty faces [F,16,16,C>=3]")
    if edges.ndim != 3 or edges.shape[1] != 16 or edges.shape[-1] < 3:
        raise ValueError(f"{path}: expected edges [E,16,C>=3]")
    if faces.dtype.kind != "f" or edges.dtype.kind != "f":
        raise ValueError(f"{path}: source geometry coordinates must use floating dtypes")
    if not np.isfinite(faces).all() or not np.isfinite(edges).all():
        raise ValueError(f"{path}: nonfinite source geometry")
    validate_connectivity(rows, len(faces), len(edges))
    return arrays


def validate_connectivity(rows: np.ndarray, num_faces: int, num_edges: int | None = None) -> None:
    if rows.ndim != 2 or rows.shape[1] != 3 or rows.dtype.kind not in "iu":
        raise ValueError("edge_face_connectivity must be integer [R,3]")
    if (rows < 0).any() or (rows[:, 1:] >= num_faces).any():
        raise ValueError("edge_face_connectivity has invalid indices")
    if num_edges is not None and (rows[:, 0] >= num_edges).any():
        raise ValueError("edge_face_connectivity has invalid edge indices")


def empty_edge_uv_sidecar(
    connectivity: np.ndarray,
    num_faces: int,
    *,
    source_sha256: str,
    step_sha256: str = "",
    parameters: dict[str, Any] | None = None,
    reason: str = "not_processed",
) -> dict[str, np.ndarray]:
    """A legal all-invalid label preserves all original geometry supervision."""
    shape = (len(connectivity), 2, SAMPLE_COUNT)
    return {
        "schema_version": np.array(SCHEMA_VERSION, dtype=np.int64),
        "source_sha256": np.array(source_sha256),
        "step_sha256": np.array(step_sha256),
        "parameters_json": np.array(json.dumps(parameters or {}, sort_keys=True)),
        "edge_face_connectivity": np.asarray(connectivity, dtype=np.int64).copy(),
        "edge_uv": np.zeros((*shape, 2), dtype=np.float32),
        "edge_uv_valid": np.zeros(shape, dtype=np.bool_),
        "edge_uv_residual": np.full(shape, -1.0, dtype=np.float32),
        "edge_uv_method": np.zeros(shape, dtype=np.uint8),
        "face_uv_bounds": np.full((num_faces, 4), np.nan, dtype=np.float64),
        "face_step_indices": np.full(num_faces, -1, dtype=np.int64),
        "face_match_error": np.full(num_faces, -1.0, dtype=np.float64),
        "face_periods": np.full((num_faces, 2), np.nan, dtype=np.float64),
        "side_reason": np.full((len(connectivity), 2), reason, dtype="U256"),
        "status": np.array("invalid"),
        "reason": np.array(reason),
    }


def _scalar_string(arrays: dict[str, np.ndarray], name: str) -> str:
    value = arrays[name]
    if value.shape != () or value.dtype.kind not in "US":
        raise ValueError(f"{name} must be a scalar string")
    return str(value.item())


def validate_edge_uv_sidecar(
    arrays: dict[str, np.ndarray], *, connectivity: np.ndarray, num_faces: int,
    source_sha256: str,
) -> dict[str, np.ndarray]:
    """Validate labels without importing OCC, torch, or the legacy extractor."""
    required = set(empty_edge_uv_sidecar(connectivity, num_faces, source_sha256=source_sha256))
    if missing := required - arrays.keys():
        raise ValueError(f"UV sidecar missing fields: {sorted(missing)}")
    version = arrays["schema_version"]
    if version.shape != () or version.dtype.kind not in "iu" or int(version) != SCHEMA_VERSION:
        raise ValueError(f"unsupported UV schema_version: {version}")
    digest = _scalar_string(arrays, "source_sha256")
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest) or digest != source_sha256:
        raise ValueError("UV source_sha256 mismatch")
    validate_connectivity(connectivity, num_faces)
    saved_rows = arrays["edge_face_connectivity"]
    validate_connectivity(saved_rows, num_faces)
    if not np.array_equal(saved_rows, connectivity):
        raise ValueError("UV connectivity differs from original source row order")
    shape = (len(connectivity), 2, SAMPLE_COUNT)
    contracts = {
        "edge_uv": ((*shape, 2), np.dtype("float32")),
        "edge_uv_valid": (shape, np.dtype("bool")),
        "edge_uv_residual": (shape, np.dtype("float32")),
        "edge_uv_method": (shape, np.dtype("uint8")),
    }
    for key, (expected_shape, expected_dtype) in contracts.items():
        if arrays[key].shape != expected_shape or arrays[key].dtype != expected_dtype:
            raise ValueError(f"{key}: expected {expected_shape} {expected_dtype}")
        if not np.isfinite(arrays[key]).all():
            raise ValueError(f"{key}: nonfinite training field")
    uv, valid, residual, method = (arrays[key] for key in contracts)
    if (method > 2).any() or not np.array_equal(method > 0, valid):
        raise ValueError("method must be 0 exactly for invalid points, 1=native or 2=projection otherwise")
    if ((residual < 0) & (residual != -1)).any() or (residual[valid] < 0).any():
        raise ValueError("residual must be nonnegative, or -1 for unknown invalid points")
    if ((uv[valid] < -UV_DOMAIN_TOLERANCE) | (uv[valid] > 1 + UV_DOMAIN_TOLERANCE)).any():
        raise ValueError("valid UV lies outside the normalized source chart")
    parameters = json.loads(_scalar_string(arrays, "parameters_json"))
    if not isinstance(parameters, dict):
        raise ValueError("parameters_json must contain an object")
    tolerance = float(parameters.get("projection_tolerance", 1e-3))
    if not np.isfinite(tolerance) or tolerance <= 0 or (residual[valid] > tolerance * (1 + 1e-6)).any():
        raise ValueError("valid UV exceeds the recorded residual tolerance")
    bounds = arrays["face_uv_bounds"]
    face_indices = arrays["face_step_indices"]
    errors = arrays["face_match_error"]
    periods = arrays["face_periods"]
    if bounds.shape != (num_faces, 4) or periods.shape != (num_faces, 2):
        raise ValueError("invalid face chart metadata shape")
    if face_indices.shape != (num_faces,) or face_indices.dtype.kind not in "iu" or errors.shape != (num_faces,):
        raise ValueError("invalid face matching metadata")
    matched = face_indices >= 0
    if (face_indices < -1).any() or (face_indices[matched] >= num_faces).any() or len(np.unique(face_indices[matched])) != int(matched.sum()):
        raise ValueError("face_step_indices must be a partial bijection in the source face range")
    match_tolerance = float(parameters.get("match_tolerance", 1e-5))
    if not np.isfinite(match_tolerance) or match_tolerance <= 0:
        raise ValueError("invalid recorded face matching tolerance")
    owners = connectivity[:, 1:][valid.any(axis=2)]
    if owners.size:
        b = bounds[owners]
        if not np.isfinite(b).all() or (b[:, 1] <= b[:, 0]).any() or (b[:, 3] <= b[:, 2]).any():
            raise ValueError("valid UV owner has invalid face bounds")
        if (face_indices[owners] < 0).any() or not np.isfinite(errors[owners]).all() or (errors[owners] < 0).any() or (errors[owners] > match_tolerance).any():
            raise ValueError("valid UV owner lacks verified face matching")
        if not np.isfinite(periods[owners]).all() or (periods[owners] < 0).any():
            raise ValueError("valid UV owner has invalid periodic metadata")
    if arrays["side_reason"].shape != (len(connectivity), 2) or arrays["side_reason"].dtype.kind not in "US":
        raise ValueError("invalid side_reason metadata")
    status = _scalar_string(arrays, "status")
    expected_status = "ok" if valid.size and valid.all() else ("partial" if valid.any() else "invalid")
    if status != expected_status:
        raise ValueError(f"UV status {status!r} disagrees with valid mask ({expected_status})")
    _scalar_string(arrays, "reason")
    _scalar_string(arrays, "step_sha256")
    return arrays


def load_edge_uv_sidecar(
    path: str | Path, *, source_path: str | Path, connectivity: np.ndarray, num_faces: int,
) -> dict[str, np.ndarray]:
    """Load and strictly validate against the current source NPZ and its rows."""
    with np.load(path, allow_pickle=False) as source:
        arrays = {key: source[key].copy() for key in source.files}
    try:
        return validate_edge_uv_sidecar(
            arrays, connectivity=connectivity, num_faces=num_faces, source_sha256=sha256_file(source_path)
        )
    except (ValueError, KeyError, TypeError) as error:
        raise ValueError(f"{path}: {error}") from error


def atomic_save_npz(path: str | Path, arrays: dict[str, np.ndarray]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
