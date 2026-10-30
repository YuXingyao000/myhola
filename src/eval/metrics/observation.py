"""Single-view observation metrics in the original HoLa SVR camera.

Both STEP shapes are rendered independently, without alignment or healing.
Visible 3-D distances use paired GT geometry; silhouette distances use the
input image itself. Coordinates retain the dataset's existing normalization.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Iterable

import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion, distance_transform_edt
from scipy.spatial import cKDTree

from src.models.diffusion.image_loading import load_image_view


PROTOCOL_VERSION = "legacy_svr_view0_perspective_v1"


@lru_cache(maxsize=1)
def renderer_dependencies() -> dict[str, str]:
    """Fingerprint the loaded worker's renderer and OCC environment once."""
    import OCC

    source = Path(__file__)
    triangulation_source = source.parents[2] / "preprocess" / "extract_visibility.py"
    return {
        "observation_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "triangulation_sha256": hashlib.sha256(triangulation_source.read_bytes()).hexdigest(),
        "pythonocc_version": OCC.VERSION,
    }


@dataclass(frozen=True)
class Camera:
    """Camera axes: x right, y up, z forward; image rows point down."""

    intrinsic: np.ndarray
    world_to_cam: np.ndarray
    resolution: int


def legacy_svr_camera(resolution: int = 224, view_id: int = 0) -> Camera:
    """Reproduce extract_imgs.py's identity view, without fitting to the GT."""
    if view_id != 0:
        raise ValueError("Observation protocol currently supports only SVR view_id=0")
    if resolution < 2:
        raise ValueError("Camera resolution must be at least 2")
    eye = np.array([2., 2., 2.])
    forward = -eye / np.linalg.norm(eye)
    right = np.cross(forward, np.array([2., 2., 3.]))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    transform = np.eye(4)
    transform[:3, :3] = np.stack([right, up, forward])
    transform[:3, 3] = -transform[:3, :3] @ eye
    focal = .5 * resolution / np.tan(np.deg2rad(45.) / 2.)
    center = (resolution - 1.) / 2.
    intrinsic = np.array([[focal, 0., center], [0., focal, center], [0., 0., 1.]])
    return Camera(intrinsic, transform, resolution)


def project_points(points: np.ndarray, camera: Camera) -> tuple[np.ndarray, np.ndarray]:
    cam = np.asarray(points, dtype=np.float64) @ camera.world_to_cam[:3, :3].T
    cam += camera.world_to_cam[:3, 3]
    if np.any(cam[:, 2] <= 0.):
        raise ValueError("Geometry intersects or lies behind the camera plane")
    xy = cam[:, :2] / cam[:, 2, None]
    xy *= np.array([camera.intrinsic[0, 0], -camera.intrinsic[1, 1]])
    xy += camera.intrinsic[:2, 2]
    return xy, cam[:, 2]


def unproject_points(xy: np.ndarray, depth: np.ndarray, camera: Camera) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float64)
    depth = np.asarray(depth, dtype=np.float64)
    cam = np.column_stack([
        (xy[:, 0] - camera.intrinsic[0, 2]) * depth / camera.intrinsic[0, 0],
        -(xy[:, 1] - camera.intrinsic[1, 2]) * depth / camera.intrinsic[1, 1],
        depth,
    ])
    # The orthogonal camera frame has y up: do not apply OpenCV's y-down rule.
    return (cam - camera.world_to_cam[:3, 3]) @ camera.world_to_cam[:3, :3]


def visible_points(depth: np.ndarray, camera: Camera) -> np.ndarray:
    rows, columns = np.nonzero(np.isfinite(depth))
    return unproject_points(np.column_stack([columns, rows]), depth[rows, columns], camera)


def rasterize_meshes(
    meshes: Iterable[tuple[np.ndarray, np.ndarray]], camera: Camera,
) -> np.ndarray:
    """Two-sided perspective z-buffer at pixel centers; NaN is background."""
    size = camera.resolution
    depth = np.full((size, size), np.inf, dtype=np.float64)
    for vertices, triangles in meshes:
        if not np.all(np.isfinite(vertices)):
            raise ValueError("Mesh contains non-finite vertices")
        projected, vertex_depth = project_points(vertices, camera)
        for triangle in triangles:
            xy, z = projected[triangle], vertex_depth[triangle]
            low = np.maximum(np.floor(xy.min(axis=0)).astype(int), 0)
            high = np.minimum(np.ceil(xy.max(axis=0)).astype(int), size - 1)
            if np.any(low > high):
                continue
            x0, y0 = xy[0]
            x1, y1 = xy[1]
            x2, y2 = xy[2]
            denominator = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
            if abs(denominator) < 1e-12:
                continue
            x, y = np.meshgrid(np.arange(low[0], high[0] + 1), np.arange(low[1], high[1] + 1))
            w0 = ((y1 - y2) * (x - x2) + (x2 - x1) * (y - y2)) / denominator
            w1 = ((y2 - y0) * (x - x2) + (x0 - x2) * (y - y2)) / denominator
            w2 = 1. - w0 - w1
            inside = (w0 >= -1e-9) & (w1 >= -1e-9) & (w2 >= -1e-9)
            rows, columns = y[inside], x[inside]
            # Camera-space z is interpolated reciprocally under perspective.
            values = 1. / (w0[inside] / z[0] + w1[inside] / z[1] + w2[inside] / z[2])
            depth[rows, columns] = np.minimum(depth[rows, columns], values)
    depth[~np.isfinite(depth)] = np.nan
    if not np.any(np.isfinite(depth)):
        raise ValueError("STEP has no visible foreground in the specified camera")
    return depth


def render_step(step_path: Path, camera: Camera, mesh_deflection: float) -> np.ndarray:
    """Triangulate trimmed OCC faces, including readable non-solid shapes."""
    from src.preprocess.extract_visibility import load_shape_and_faces, triangulate_faces

    if mesh_deflection <= 0.:
        raise ValueError("mesh_deflection must be positive")
    try:
        shape, faces = load_shape_and_faces(Path(step_path))
        meshes = triangulate_faces(shape, faces, mesh_deflection)
        return rasterize_meshes(((mesh.vertices, mesh.triangles) for mesh in meshes), camera)
    except (ValueError, RuntimeError) as error:
        raise ValueError(f"Cannot render {step_path}: {error}") from error


def silhouette_metrics(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    """IoU and the sum of directed mean boundary distances / image diagonal."""
    left, right = np.asarray(left, dtype=bool), np.asarray(right, dtype=bool)
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("Silhouette masks must have the same 2-D shape")
    if not left.any() or not right.any():
        raise ValueError("Silhouette metrics require two nonempty foreground masks")
    left_boundary = left & ~binary_erosion(left, border_value=0)
    right_boundary = right & ~binary_erosion(right, border_value=0)
    distance = distance_transform_edt(~right_boundary)[left_boundary].mean()
    distance += distance_transform_edt(~left_boundary)[right_boundary].mean()
    iou = np.count_nonzero(left & right) / np.count_nonzero(left | right)
    return float(iou), float(distance / np.hypot(*left.shape))


def compare_depth_maps(
    pred_depth: np.ndarray, gt_depth: np.ndarray, input_mask: np.ndarray, camera: Camera,
) -> dict[str, object]:
    """Measure visible geometry and input agreement; absence of overlap is explicit."""
    expected_shape = (camera.resolution, camera.resolution)
    if any(value.shape != expected_shape for value in (pred_depth, gt_depth, input_mask)):
        raise ValueError("Depth and mask dimensions must match the camera resolution")
    pred_mask, gt_mask = np.isfinite(pred_depth), np.isfinite(gt_depth)
    iou, boundary = silhouette_metrics(pred_mask, input_mask)
    gt_iou, gt_boundary = silhouette_metrics(gt_mask, input_mask)
    pred_points, gt_points = visible_points(pred_depth, camera), visible_points(gt_depth, camera)
    pred_to_gt = float(np.square(cKDTree(gt_points).query(pred_points)[0]).mean())
    gt_to_pred = float(np.square(cKDTree(pred_points).query(gt_points)[0]).mean())
    overlap = pred_mask & gt_mask
    depth_mae = float(np.abs(pred_depth[overlap] - gt_depth[overlap]).mean()) if overlap.any() else None
    return {
        "status": "ok",
        "visible_face_cd_sq": pred_to_gt + gt_to_pred,
        "visible_face_pred_to_gt_cd_sq": pred_to_gt,
        "visible_face_gt_to_pred_cd_sq": gt_to_pred,
        "silhouette_iou": iou,
        "silhouette_boundary_distance": boundary,
        "gt_input_silhouette_iou": gt_iou,
        "gt_input_silhouette_boundary_distance": gt_boundary,
        "depth_mae": depth_mae,
        "overlap_coverage": float(overlap.sum() / gt_mask.sum()),
        "pred_foreground_pixels": int(pred_mask.sum()),
        "gt_foreground_pixels": int(gt_mask.sum()),
        "input_foreground_pixels": int(input_mask.sum()),
        "depth_overlap_pixels": int(overlap.sum()),
    }


def _cached_gt_depth(
    gt_step: Path, camera: Camera, mesh_deflection: float, cache_root: Path | None,
) -> np.ndarray:
    if cache_root is None:
        return render_step(gt_step, camera, mesh_deflection)
    digest = hashlib.sha256()
    with gt_step.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    protocol = {
        "version": PROTOCOL_VERSION, "gt_sha256": digest.hexdigest(),
        "renderer_dependencies": renderer_dependencies(),
        "resolution": camera.resolution, "mesh_deflection": mesh_deflection,
        "intrinsic": camera.intrinsic.tolist(), "world_to_cam": camera.world_to_cam.tolist(),
    }
    encoded = json.dumps(protocol, sort_keys=True)
    key = hashlib.sha256(encoded.encode()).hexdigest()
    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_path = cache_root / f"{key}.npz"
    if cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as saved:
            return saved["depth"]
    depth = render_step(gt_step, camera, mesh_deflection)
    # Different method workers can reach the same GT concurrently.
    with tempfile.NamedTemporaryFile(dir=cache_root, suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
        np.savez_compressed(handle, depth=depth, protocol=np.array(encoded))
    os.replace(temporary, cache_path)
    return depth


def evaluate_observation(
    pred_step: Path, gt_step: Path, image_path: Path, *, image_key: str = "svr_imgs",
    view_id: int = 0, resolution: int = 224, mesh_deflection: float = .002,
    mask_threshold: int = 245, cache_root: Path | None = None,
) -> dict[str, object]:
    """Evaluate the fixed SVR camera; failures raise and must remain in the denominator.

    The GT cache is optional and external to the original prediction directory.
    ``overlap_coverage`` is shared foreground / GT foreground. Depth MAE is
    unavailable (None) when there is no shared foreground. All 3-D distances
    remain in normalized dataset coordinates; boundary distance is dimensionless.
    """
    camera = legacy_svr_camera(resolution, view_id)
    if not 0 < mask_threshold <= 255:
        raise ValueError("mask_threshold must lie in (0, 255]")
    image = load_image_view(image_path, image_key, view_id, read_single_view=True)
    mask = np.min(image, axis=-1) < mask_threshold
    if mask.shape != (resolution, resolution):
        mask = np.asarray(Image.fromarray(mask).resize((resolution, resolution), Image.Resampling.NEAREST))
    pred_depth = render_step(Path(pred_step), camera, mesh_deflection)
    gt_depth = _cached_gt_depth(Path(gt_step), camera, mesh_deflection, cache_root)
    result = compare_depth_maps(pred_depth, gt_depth, mask, camera)
    result["protocol"] = {
        "version": PROTOCOL_VERSION, "view_id": view_id, "image_key": image_key,
        "renderer_dependencies": renderer_dependencies(),
        "resolution": resolution, "mesh_deflection": mesh_deflection, "mask_threshold": mask_threshold,
        "intrinsic": camera.intrinsic.tolist(), "world_to_cam": camera.world_to_cam.tolist(),
        "boundary_definition": "sum of directed mean boundary pixel distances / image diagonal",
        "overlap_coverage_definition": "shared foreground pixels / GT foreground pixels",
    }
    return result
