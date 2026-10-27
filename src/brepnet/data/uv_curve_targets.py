"""Current-chart UV targets: strict NumPy cache IO and batched offline projection.

The chart is the *current* fixed cubic interpolation of GT 16x16 XYZ, not an
OCC/native chart. Projection failures mask only UV supervision. No CAD or edge
is removed, and the original representative curve point order is preserved.
Torch is imported only when the offline projector is instantiated.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

from .edge_uv import read_source_geometry, sha256_file
from .shared_boundaries import build_shared_boundary_targets


SCHEMA_VERSION = 1
REPRESENTATION = "fixed_bspline_uv_targets_v1"
SURFACE_SIGNATURE = "grid16x16_degree3_open_uniform_interpolation_vu_uv01_nonperiodic_v1"
REASON_FLAGS = {"residual": 1, "degenerate_jacobian": 2, "ambiguous_branch": 4,
                "parameter_jump": 8, "degenerate_curve": 16}


@dataclass(frozen=True)
class UVProjectionConfig:
    coarse_grid_size: int = 17
    starts: int = 6
    seed_separation: float = 0.12
    iterations: int = 20
    max_step: float = 0.15
    damping: float = 1e-7
    residual_tolerance: float = 1e-3
    continuity_step: float = 0.35
    continuity_weight: float = 1.0
    ambiguity_uv_separation: float = 0.10
    ambiguity_cost_margin: float = 0.05
    jacobian_relative_min: float = 1e-6
    jacobian_sine_min: float = 1e-4
    curve_extent_min: float = 1e-8
    reciprocal_tolerance: float = 0.01
    closed_tolerance: float = 1e-5
    allow_closed_phase: bool = True

    def validate(self) -> None:
        for name in ("coarse_grid_size", "starts", "iterations"):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an integer")
        if type(self.allow_closed_phase) is not bool:
            raise ValueError("allow_closed_phase must be bool")
        if self.coarse_grid_size < 3 or not 2 <= self.starts <= 16 or self.iterations < 1:
            raise ValueError("Invalid projection grid/start/iteration count")
        for key, value in asdict(self).items():
            if key in {"coarse_grid_size", "starts", "iterations", "allow_closed_phase"}:
                continue
            if type(value) not in {int, float} or not np.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be finite and positive")


def parameters_json(config: UVProjectionConfig) -> str:
    config.validate()
    return json.dumps(asdict(config), sort_keys=True, separators=(",", ":"))


def _string(arrays: dict, key: str) -> str:
    value = np.asarray(arrays[key])
    if value.ndim != 0 or value.dtype.kind not in "US":
        raise ValueError(f"{key} must be a scalar string")
    return str(value.item())


def validate_uv_curve_targets(
    arrays: dict[str, np.ndarray], *, source_sha256: str, face_pairs: np.ndarray,
    source_edge_ids: np.ndarray, expected_parameters_sha256: str | None = None,
) -> dict[str, np.ndarray]:
    required = {"schema_version", "representation", "surface_signature", "source_sha256",
                "parameters_json", "parameters_sha256", "face_pairs", "source_edge_ids",
                "uv", "valid", "residual", "reason_flags", "is_closed", "projection_metadata_json"}
    if missing := required - arrays.keys():
        raise ValueError(f"UV cache missing fields: {sorted(missing)}")
    schema = np.asarray(arrays["schema_version"])
    if schema.ndim or schema.dtype.kind not in "iu" or schema.item() != SCHEMA_VERSION:
        raise ValueError("UV cache schema_version mismatch")
    for key, expected in (("representation", REPRESENTATION), ("surface_signature", SURFACE_SIGNATURE),
                          ("source_sha256", source_sha256)):
        if _string(arrays, key) != expected:
            raise ValueError(f"UV cache {key} mismatch")
    params = _string(arrays, "parameters_json")
    UVProjectionConfig(**json.loads(params)).validate()
    digest = hashlib.sha256(params.encode()).hexdigest()
    if _string(arrays, "parameters_sha256") != digest:
        raise ValueError("UV cache parameters digest mismatch")
    if expected_parameters_sha256 is not None and digest != expected_parameters_sha256:
        raise ValueError("UV cache projection protocol mismatch")
    for key, expected in (("face_pairs", face_pairs), ("source_edge_ids", source_edge_ids)):
        value = np.asarray(arrays[key])
        if value.dtype.kind not in "iu" or not np.array_equal(value, expected):
            raise ValueError(f"UV cache {key} or representative order mismatch")
    count = len(face_pairs)
    shapes = {"uv": (count, 2, 16, 2), "valid": (count, 2, 16),
              "residual": (count, 2, 16), "reason_flags": (count, 2, 16), "is_closed": (count,)}
    for key, shape in shapes.items():
        if np.asarray(arrays[key]).shape != shape:
            raise ValueError(f"UV cache {key}: expected {shape}, got {arrays[key].shape}")
    if arrays["valid"].dtype != np.bool_ or arrays["is_closed"].dtype != np.bool_:
        raise ValueError("UV cache valid/is_closed must be bool")
    for key in ("uv", "residual"):
        if arrays[key].dtype != np.float32 or not np.isfinite(arrays[key]).all():
            raise ValueError(f"UV cache {key} must be finite float32")
    if (arrays["uv"] < 0).any() or (arrays["uv"] > 1).any() or (arrays["residual"] < 0).any():
        raise ValueError("UV cache coordinates/residual out of range")
    if (arrays["reason_flags"].dtype.kind not in "iu"
            or (arrays["reason_flags"] < 0).any() or (arrays["reason_flags"] > 31).any()):
        raise ValueError("UV cache reason_flags invalid")
    if not np.array_equal(arrays["valid"], arrays["reason_flags"] == 0):
        raise ValueError("UV cache mask and reasons disagree")
    if np.any(arrays["valid"] & (arrays["residual"] > json.loads(params)["residual_tolerance"] + 1e-8)):
        raise ValueError("UV cache marks an excessive residual valid")
    json.loads(_string(arrays, "projection_metadata_json"))
    return arrays


def load_uv_curve_targets(path: str | Path, **validation: Any) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key].copy() for key in data.files}
    return validate_uv_curve_targets(arrays, **validation)


def write_uv_curve_targets(path: str | Path, arrays: dict[str, np.ndarray]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    descriptor, temporary = tempfile.mkstemp(prefix=".uv-target-", suffix=".npz", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_projection_source(path: str | Path, config: UVProjectionConfig, sample_id: str) -> dict:
    source = read_source_geometry(path)
    boundary = build_shared_boundary_targets(
        source["sample_points_faces"], source["sample_points_lines"], source["edge_face_connectivity"],
        sample_id=sample_id, reciprocal_tolerance=config.reciprocal_tolerance,
        closed_tolerance=config.closed_tolerance, allow_closed_phase=config.allow_closed_phase,
    )
    return {"sample_id": sample_id, "source_sha256": sha256_file(path),
            "faces": source["sample_points_faces"][..., :3], "boundary": boundary}


class FixedSurfaceEvaluator:
    """Exact fixed chart semantics; analytic first derivatives for offline GN."""

    def __init__(self, device: str):
        import torch
        self.torch = torch
        self.device = torch.device(device)
        self.dtype = torch.float64
        breaks = torch.linspace(0, 1, 14, dtype=self.dtype, device=self.device)
        self.knots = torch.cat((breaks[:1].repeat(3), breaks, breaks[-1:].repeat(3)))
        nodes = torch.linspace(0, 1, 16, dtype=self.dtype, device=self.device)
        B = self.basis(nodes)[0]
        self.inverse = torch.linalg.solve(B, torch.eye(16, dtype=self.dtype, device=self.device))

    def tensor(self, value):
        return self.torch.as_tensor(value, dtype=self.dtype, device=self.device)

    def basis(self, parameter):
        t = self.torch
        x, knots = parameter.unsqueeze(-1), self.knots
        left, right = knots[:-1], knots[1:]
        last_span = (x == 1) & (right == 1) & (left < right)
        basis = (((x >= left) & (x < right)) | last_span).to(self.dtype)
        for degree in range(1, 4):
            width = knots.numel() - degree - 1
            dl = knots[degree:degree + width] - knots[:width]
            dr = knots[degree + 1:] - knots[1:width + 1]
            il = (dl > 0) / t.where(dl > 0, dl, t.ones_like(dl))
            ir = (dr > 0) / t.where(dr > 0, dr, t.ones_like(dr))
            if degree == 3:
                derivative = degree * (basis[..., :-1] * il - basis[..., 1:] * ir)
            basis = (x - knots[:width]) * il * basis[..., :-1] + (knots[degree + 1:] - x) * ir * basis[..., 1:]
        return basis, derivative

    def fit(self, samples):
        return self.torch.einsum("bi,fijc,aj->fbac", self.inverse, samples, self.inverse)

    def evaluate(self, poles, uv, derivatives=False):
        bu, du = self.basis(uv[..., 0]); bv, dv = self.basis(uv[..., 1])
        def product(a, b):
            return self.torch.einsum("s...a,sbac,s...b->s...c", a, poles, b)
        points = product(bu, bv)
        return (points, product(du, bv), product(bu, dv)) if derivatives else points


def _initial_seeds(surface: FixedSurfaceEvaluator, poles, targets, config: UVProjectionConfig):
    t = surface.torch
    axis = t.linspace(0, 1, config.coarse_grid_size, dtype=surface.dtype, device=surface.device)
    v, u = t.meshgrid(axis, axis, indexing="ij")
    grid_uv = t.stack((u, v), -1).reshape(-1, 2)
    grid_xyz = surface.evaluate(poles, grid_uv[None].expand(len(poles), -1, -1))
    distance = (targets[:, :, None] - grid_xyz[:, None]).square().sum(-1)
    seeds = []
    for _ in range(config.starts):
        index = distance.argmin(-1)
        selected = grid_uv[index]
        seeds.append(selected)
        nearby = (grid_uv[None, None] - selected[:, :, None]).square().sum(-1) < config.seed_separation**2
        distance = distance.masked_fill(nearby, float("inf"))
    return t.stack(seeds, dim=2)


def _refine_seeds(surface: FixedSurfaceEvaluator, poles, targets, uv, config: UVProjectionConfig):
    t = surface.torch
    scales = surface.tensor([1., .5, .25, .125])
    for _ in range(config.iterations):
        xyz, ju, jv = surface.evaluate(poles, uv, derivatives=True)
        residual = xyz - targets[:, :, None]
        a, b, c = ju.square().sum(-1), (ju*jv).sum(-1), jv.square().sum(-1)
        damp = config.damping * (a+c).clamp_min(1e-12)
        aa, cc = a+damp, c+damp
        det = (aa*cc-b*b).clamp_min(1e-30)
        ru, rv = (ju*residual).sum(-1), (jv*residual).sum(-1)
        step = t.stack(((cc*ru-b*rv)/det, (aa*rv-b*ru)/det), -1)
        norm = step.norm(dim=-1, keepdim=True).clamp_min(1e-30)
        step = step * (config.max_step/norm).clamp_max(1)
        trial = (uv[..., None, :] - step[..., None, :] * scales[:, None]).clamp(0, 1)
        trial_xyz = surface.evaluate(poles, trial)
        trial_error = (trial_xyz - targets[:, :, None, None]).square().sum(-1)
        best_error, best = trial_error.min(-1)
        best_uv = trial.gather(-2, best[..., None, None].expand(*best.shape, 1, 2)).squeeze(-2)
        uv = t.where((best_error < residual.square().sum(-1))[..., None], best_uv, uv)
    xyz, ju, jv = surface.evaluate(poles, uv, derivatives=True)
    return [value.detach().cpu().numpy() for value in (uv, (xyz-targets[:, :, None]).norm(dim=-1), ju, jv)]


def _select_paths(uv, residual, ju, jv, face_scale, curves, config: UVProjectionConfig) -> dict:
    """Open parameter paths; no periodic wrap or forced UV closure is invented."""
    paths, length, starts = residual.shape
    unary = np.square(residual / config.residual_tolerance)
    transition = np.square(uv[:, 1:, :, None] - uv[:, :-1, None]).sum(-1)
    transition *= config.continuity_weight / config.continuity_step**2
    forward = np.empty_like(unary); previous = np.zeros_like(unary, dtype=np.int64)
    forward[:, 0] = unary[:, 0]
    for point in range(1, length):
        costs = transition[:, point-1] + forward[:, point-1, None]
        previous[:, point] = costs.argmin(-1)
        forward[:, point] = unary[:, point] + costs.min(-1)
    backward = np.zeros_like(unary)
    for point in range(length-2, -1, -1):
        costs = transition[:, point] + (unary[:, point+1]+backward[:, point+1])[:, :, None]
        backward[:, point] = costs.min(1)
    index = np.empty((paths, length), dtype=np.int64)
    index[:, -1] = forward[:, -1].argmin(-1)
    rows = np.arange(paths)
    for point in range(length-2, -1, -1):
        index[:, point] = previous[rows, point+1, index[:, point+1]]
    gather = (rows[:, None], np.arange(length)[None], index)
    selected, errors = uv[gather], residual[gather]
    u, v = ju[gather], jv[gather]
    nu, nv = np.linalg.norm(u, axis=-1), np.linalg.norm(v, axis=-1)
    sine = np.linalg.norm(np.cross(u, v), axis=-1) / np.maximum(nu*nv, 1e-30)
    degenerate = (np.minimum(nu, nv) < face_scale[:, None]*config.jacobian_relative_min) | (sine < config.jacobian_sine_min)
    marginal = forward + backward
    competitive = marginal <= forward[:, -1].min(-1)[:, None, None] + config.ambiguity_cost_margin
    separated = np.linalg.norm(uv-selected[:, :, None], axis=-1) > config.ambiguity_uv_separation
    ambiguous = (competitive & separated & (residual <= config.residual_tolerance)).any(-1)
    jumps = np.linalg.norm(np.diff(selected, axis=1), axis=-1) > config.continuity_step
    jump_mask = np.zeros((paths, length), bool)
    jump_mask[:, :-1] |= jumps; jump_mask[:, 1:] |= jumps
    collapsed = np.linalg.norm(curves-curves.mean(1, keepdims=True), axis=-1).max(1) <= config.curve_extent_min
    flags = ((errors > config.residual_tolerance).astype(np.uint8)
             | (degenerate.astype(np.uint8)*2) | (ambiguous.astype(np.uint8)*4)
             | (jump_mask.astype(np.uint8)*8) | (collapsed[:, None].astype(np.uint8)*16))
    return {"uv": selected.astype(np.float32), "valid": flags == 0,
            "residual": errors.astype(np.float32), "reason_flags": flags}


class FixedSurfaceProjector:
    """Batched GT-only multistart projection; no model checkpoint is consulted."""

    def __init__(self, device="cpu", config: UVProjectionConfig | None = None):
        self.config = config or UVProjectionConfig()
        self.config.validate()
        self.surface = FixedSurfaceEvaluator(device)

    def project_batch(self, sources: list[dict]) -> list[dict[str, np.ndarray]]:
        if not sources:
            return []
        cfg, surface = self.config, self.surface
        t = surface.torch
        counts = [len(s["boundary"]["curves"]) for s in sources]
        owners, faces, curves = [], [], []
        offset = 0
        for source in sources:
            faces.append(source["faces"])
            owners.append(source["boundary"]["face_pairs"] + offset)
            curves.append(source["boundary"]["curves"])
            offset += len(source["faces"])
        all_faces = np.concatenate(faces)
        all_owners = np.concatenate(owners).reshape(-1)
        all_curves = np.repeat(np.concatenate(curves), 2, axis=0)
        if len(all_curves):
            with t.inference_mode():
                poles = surface.fit(surface.tensor(all_faces))[t.as_tensor(all_owners, device=surface.device)]
                target = surface.tensor(all_curves)
                initial = _initial_seeds(surface, poles, target, cfg)
                candidates = _refine_seeds(surface, poles, target, initial, cfg)
            scale = np.linalg.norm(all_faces-all_faces.mean((1,2), keepdims=True), axis=-1).max((1,2))
            result = _select_paths(*candidates, np.maximum(scale[all_owners], 1e-12), all_curves, cfg)
            # The cache stores FP32 UV. Certify those actual values, not only
            # the FP64 iterate that existed before serialization.
            with t.inference_mode():
                stored_xyz = surface.evaluate(poles, surface.tensor(result["uv"]))
                stored_residual = (stored_xyz-target).norm(dim=-1).cpu().numpy()
            result["residual"] = stored_residual.astype(np.float32)
            result["reason_flags"] &= np.uint8(254)
            result["reason_flags"] |= (stored_residual > cfg.residual_tolerance).astype(np.uint8)
            result["valid"] = result["reason_flags"] == 0
        else:
            result = {"uv": np.empty((0,16,2), np.float32), "valid": np.empty((0,16), bool),
                      "residual": np.empty((0,16), np.float32), "reason_flags": np.empty((0,16), np.uint8)}
        outputs, cursor = [], 0
        for source, count in zip(sources, counts):
            boundary = source["boundary"]
            params = parameters_json(cfg)
            output = {"schema_version": np.array(SCHEMA_VERSION, np.int64), "representation": np.array(REPRESENTATION),
                      "surface_signature": np.array(SURFACE_SIGNATURE), "source_sha256": np.array(source["source_sha256"]),
                      "parameters_json": np.array(params), "parameters_sha256": np.array(hashlib.sha256(params.encode()).hexdigest()),
                      "face_pairs": boundary["face_pairs"].astype(np.int64), "source_edge_ids": boundary["source_edge_ids"].astype(np.int64),
                      "is_closed": boundary["is_closed"].astype(bool),
                      "projection_metadata_json": np.array(json.dumps({"method":"multistart_bounded_gauss_newton_path_dp", "dtype":"float64", "stored_uv_residual_recomputed":True, "reason_flags":REASON_FLAGS, "closure":"3D closure retained; no UV periodic wrap or forced closure"}, sort_keys=True))}
            for key, value in result.items():
                output[key] = value[cursor:cursor+2*count].reshape(count,2,*value.shape[1:])
            validate_uv_curve_targets(output, source_sha256=source["source_sha256"], face_pairs=boundary["face_pairs"], source_edge_ids=boundary["source_edge_ids"])
            outputs.append(output); cursor += 2*count
        return outputs
