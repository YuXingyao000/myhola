"""Extract per-face visibility buffers for DeepCAD v7.

The output keeps the original ``data.npz`` face order.  This script only accepts
models where reloading ``normalized_shape.step`` preserves the direct face order
well enough to match ``sample_points_faces``.  It does not attempt to remap face
ids; uncertain models are skipped and recorded in the report.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.GeomAbs import (
    GeomAbs_BSplineSurface,
    GeomAbs_Cone,
    GeomAbs_Cylinder,
    GeomAbs_Plane,
    GeomAbs_Sphere,
    GeomAbs_Torus,
)
from OCC.Core.GeomLProp import GeomLProp_SLProps
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Extend.DataExchange import read_step_file

from src.common.occ_utils import get_primitives
from src.preprocess.rotations import NUM_CUBE24_VIEWS


DEFAULT_RAW_ROOT = Path("/mnt/d/data/deepcad_v7")
DEFAULT_CONDITION_ROOT = Path("/mnt/d/data/deepcad_v7_cond")
DEFAULT_OUTPUT_ROOT = Path("/mnt/d/data/deepcad_v7_visibility")
DEFAULT_MODEL_LIST = Path("src/preprocess/splits/deduplicated_deepcad_testing_7_30.txt")
IDENTITY24_TO_EULER64 = (
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    24,
    25,
    26,
    27,
)

SUPPORTED_SURFACES = {
    GeomAbs_Plane,
    GeomAbs_Cylinder,
    GeomAbs_Cone,
    GeomAbs_Sphere,
    GeomAbs_Torus,
    GeomAbs_BSplineSurface,
}


@dataclass(frozen=True)
class Camera:
    eye: np.ndarray
    center: np.ndarray
    up: np.ndarray
    fovy_degrees: float
    image_size: int
    rotation_id: int
    rotation_basis: str = "identity_first_cube24"

    @property
    def world_to_cam(self) -> np.ndarray:
        forward = normalize(self.center - self.eye)
        right = normalize(np.cross(forward, self.up))
        true_up = normalize(np.cross(right, forward))

        rotation = np.stack([right, true_up, forward], axis=0)
        translation = -rotation @ self.eye

        matrix = np.eye(4, dtype=np.float32)
        matrix[:3, :3] = rotation.astype(np.float32)
        matrix[:3, 3] = translation.astype(np.float32)
        return matrix

    @property
    def cam_to_world(self) -> np.ndarray:
        return np.linalg.inv(self.world_to_cam).astype(np.float32)

    @property
    def intrinsic(self) -> np.ndarray:
        focal = 0.5 * float(self.image_size) / math.tan(math.radians(self.fovy_degrees) * 0.5)
        center = (float(self.image_size) - 1.0) * 0.5
        return np.array(
            [
                [focal, 0.0, center],
                [0.0, focal, center],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )


@dataclass
class FaceOrderCheck:
    ok: bool
    status: str
    direct_costs: np.ndarray
    reason: str = ""


@dataclass
class FaceMesh:
    face_id: int
    vertices: np.ndarray
    triangles: np.ndarray


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        raise ValueError("Cannot normalize zero-length vector")
    return vector / norm


def read_model_ids(path: Path, limit: int | None = None) -> list[str]:
    ids: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        model_id = line.strip()
        if model_id and not model_id.startswith("#"):
            ids.append(model_id)
        if limit is not None and len(ids) >= limit:
            break
    return ids


def parse_views(value: str) -> list[int]:
    if value == "all":
        return list(range(NUM_CUBE24_VIEWS))

    views: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start_s, end_s = item.split("-", 1)
            views.extend(range(int(start_s), int(end_s) + 1))
        else:
            views.append(int(item))

    unique_views = list(dict.fromkeys(views))
    for view_id in unique_views:
        if view_id < 0 or view_id >= NUM_CUBE24_VIEWS:
            raise ValueError(f"view id must be in [0, 23], got {view_id}")
    return unique_views


def disable_occ_log() -> None:
    from OCC.Core.Message import Message_Alarm, message

    printers = message.DefaultMessenger().Printers()
    for idx in range(printers.Length()):
        printers.Value(idx + 1).SetTraceLevel(Message_Alarm)


def build_camera(rotation_id: int, image_size: int, fovy_degrees: float, up_mode: str) -> Camera:
    euler64_id = IDENTITY24_TO_EULER64[rotation_id]
    angles = np.array([euler64_id % 4, euler64_id // 4 % 4, euler64_id // 16], dtype=np.float64)
    camera_rotation = Rotation.from_euler("xyz", angles * np.pi / 2.0).as_matrix().T

    base_eye = np.array([2.0, 2.0, 2.0], dtype=np.float64)
    if up_mode == "legacy_point":
        # Matches extract_imgs.py, which passed gp_Dir(matrix @ [2, 2, 3]).
        base_up = np.array([2.0, 2.0, 3.0], dtype=np.float64)
    elif up_mode == "z":
        base_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        raise ValueError(f"Unknown up mode: {up_mode}")

    eye = camera_rotation @ base_eye
    up = camera_rotation @ base_up
    return Camera(
        eye=eye.astype(np.float32),
        center=np.zeros(3, dtype=np.float32),
        up=up.astype(np.float32),
        fovy_degrees=fovy_degrees,
        image_size=image_size,
        rotation_id=rotation_id,
    )


def sample_face(face, resolution: int) -> np.ndarray:
    surface = BRepAdaptor_Surface(face)
    if surface.GetType() not in SUPPORTED_SURFACES:
        raise ValueError(f"Unsupported surface type: {surface.GetType()}")

    u_values = np.linspace(surface.FirstUParameter(), surface.LastUParameter(), num=resolution)
    v_values = np.linspace(surface.FirstVParameter(), surface.LastVParameter(), num=resolution)

    samples: list[np.ndarray] = []
    for v in v_values:
        for u in u_values:
            pnt = surface.Value(float(u), float(v))
            props = GeomLProp_SLProps(surface.Surface().Surface(), float(u), float(v), 1, 0.01)
            normal = props.Normal()
            samples.append(
                np.array(
                    [
                        pnt.X(),
                        pnt.Y(),
                        pnt.Z(),
                        normal.X(),
                        normal.Y(),
                        normal.Z(),
                    ],
                    dtype=np.float32,
                )
            )
    return np.stack(samples, axis=0).reshape(resolution, resolution, 6)


def face_signature(samples: np.ndarray) -> np.ndarray:
    points = samples[..., :3].reshape(-1, 3).astype(np.float64)
    normals = samples[..., 3:].reshape(-1, 3).astype(np.float64)

    centroid = points.mean(axis=0)
    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    distances = np.linalg.norm(points - centroid[None, :], axis=1)
    quantiles = np.quantile(distances, [0.1, 0.5, 0.9])

    normal_mean = normals.mean(axis=0)
    normal_norm = np.linalg.norm(normal_mean)
    if normal_norm > 1e-8:
        normal_mean = normal_mean / normal_norm
    else:
        normal_mean = np.zeros(3, dtype=np.float64)

    return np.concatenate([centroid, bbox_min, bbox_max, quantiles, normal_mean]).astype(np.float64)


def signature_cost(a: np.ndarray, b: np.ndarray) -> float:
    pos = float(np.linalg.norm(a[:12] - b[:12]))
    normal_a = a[12:15]
    normal_b = b[12:15]
    normal_cost = min(
        float(np.linalg.norm(normal_a - normal_b)),
        float(np.linalg.norm(normal_a + normal_b)),
    )
    return pos + 0.02 * normal_cost


def check_direct_face_order(faces: list, data_face_samples: np.ndarray, args: argparse.Namespace) -> FaceOrderCheck:
    if len(faces) != int(data_face_samples.shape[0]):
        return FaceOrderCheck(
            ok=False,
            status="face_count_mismatch",
            direct_costs=np.array([], dtype=np.float32),
            reason=f"occ_faces={len(faces)} data_faces={data_face_samples.shape[0]}",
        )

    data_signatures = [face_signature(data_face_samples[idx]) for idx in range(len(faces))]
    occ_samples = [sample_face(face, args.sample_resolution) for face in faces]
    occ_signatures = [face_signature(samples) for samples in occ_samples]

    direct_costs = np.array(
        [
            signature_cost(occ_signatures[idx], data_signatures[idx])
            for idx in range(len(faces))
        ],
        dtype=np.float32,
    )

    cost_matrix = np.zeros((len(faces), len(faces)), dtype=np.float32)
    for occ_idx, occ_sig in enumerate(occ_signatures):
        for data_idx, data_sig in enumerate(data_signatures):
            cost_matrix[occ_idx, data_idx] = signature_cost(occ_sig, data_sig)

    nearest = np.argmin(cost_matrix, axis=1)
    direct_is_nearest = bool(np.all(nearest == np.arange(len(faces))))
    max_direct_cost = float(direct_costs.max(initial=0.0))

    if not direct_is_nearest:
        bad = np.where(nearest != np.arange(len(faces)))[0][:10].tolist()
        return FaceOrderCheck(
            ok=False,
            status="face_order_mismatch",
            direct_costs=direct_costs,
            reason=f"direct index is not nearest for occ faces {bad}",
        )

    if max_direct_cost > args.max_direct_cost:
        return FaceOrderCheck(
            ok=False,
            status="face_order_cost_too_high",
            direct_costs=direct_costs,
            reason=f"max_direct_cost={max_direct_cost:.6g} > {args.max_direct_cost:.6g}",
        )

    return FaceOrderCheck(
        ok=True,
        status="direct",
        direct_costs=direct_costs,
        reason=f"max_direct_cost={max_direct_cost:.6g}",
    )


def load_shape_and_faces(step_path: Path) -> tuple[object, list]:
    shape = read_step_file(str(step_path), verbosity=False)
    faces = list(get_primitives(shape, TopAbs_FACE))
    return shape, faces


def triangulate_faces(shape, faces: list, mesh_deflection: float) -> list[FaceMesh]:
    BRepMesh_IncrementalMesh(shape, mesh_deflection)

    meshes: list[FaceMesh] = []
    for face_id, face in enumerate(faces):
        loc = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation(face, loc)
        if triangulation is None:
            raise ValueError(f"Face {face_id} has no triangulation")

        transform = loc.Transformation()
        vertices = np.zeros((triangulation.NbNodes(), 3), dtype=np.float32)
        for node_idx in range(1, triangulation.NbNodes() + 1):
            if hasattr(triangulation, "Node"):
                pnt = triangulation.Node(node_idx).Transformed(transform)
            else:
                pnt = triangulation.Nodes().Value(node_idx).Transformed(transform)
            vertices[node_idx - 1] = (pnt.X(), pnt.Y(), pnt.Z())

        triangles = np.zeros((triangulation.NbTriangles(), 3), dtype=np.int32)
        for tri_idx in range(1, triangulation.NbTriangles() + 1):
            if hasattr(triangulation, "Triangle"):
                tri = triangulation.Triangle(tri_idx)
            else:
                tri = triangulation.Triangles().Value(tri_idx)
            n1, n2, n3 = tri.Get()
            triangles[tri_idx - 1] = (n1 - 1, n2 - 1, n3 - 1)

        meshes.append(FaceMesh(face_id=face_id, vertices=vertices, triangles=triangles))

    return meshes


def project_points(points: np.ndarray, camera: Camera) -> tuple[np.ndarray, np.ndarray]:
    ones = np.ones((points.shape[0], 1), dtype=np.float32)
    hom = np.concatenate([points.astype(np.float32), ones], axis=1)
    cam = (camera.world_to_cam @ hom.T).T[:, :3]

    z = cam[:, 2]
    focal = camera.intrinsic[0, 0]
    center = camera.intrinsic[0, 2]
    projected = np.empty((points.shape[0], 2), dtype=np.float32)
    projected[:, 0] = center + focal * cam[:, 0] / np.maximum(z, 1e-8)
    projected[:, 1] = center - focal * cam[:, 1] / np.maximum(z, 1e-8)
    return projected, z.astype(np.float32)


def triangle_pixels(
    xy: np.ndarray,
    z: np.ndarray,
    image_size: int,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if np.any(z <= eps):
        return None

    min_x = max(0, int(math.floor(float(np.min(xy[:, 0])))))
    max_x = min(image_size - 1, int(math.ceil(float(np.max(xy[:, 0])))))
    min_y = max(0, int(math.floor(float(np.min(xy[:, 1])))))
    max_y = min(image_size - 1, int(math.ceil(float(np.max(xy[:, 1])))))
    if min_x > max_x or min_y > max_y:
        return None

    xs = np.arange(min_x, max_x + 1, dtype=np.float32)
    ys = np.arange(min_y, max_y + 1, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)

    x0, y0 = xy[0]
    x1, y1 = xy[1]
    x2, y2 = xy[2]
    denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    if abs(float(denom)) < eps:
        return None

    w0 = ((y1 - y2) * (grid_x - x2) + (x2 - x1) * (grid_y - y2)) / denom
    w1 = ((y2 - y0) * (grid_x - x2) + (x0 - x2) * (grid_y - y2)) / denom
    w2 = 1.0 - w0 - w1
    mask = (w0 >= -eps) & (w1 >= -eps) & (w2 >= -eps)
    if not np.any(mask):
        return None

    pixel_x = grid_x[mask].astype(np.int32)
    pixel_y = grid_y[mask].astype(np.int32)
    pixel_z = (w0[mask] * z[0] + w1[mask] * z[1] + w2[mask] * z[2]).astype(np.float32)
    return pixel_y, pixel_x, pixel_z


def render_visibility(meshes: list[FaceMesh], camera: Camera) -> dict[str, np.ndarray]:
    image_size = camera.image_size
    num_faces = len(meshes)
    depth_map = np.full((image_size, image_size), np.inf, dtype=np.float32)
    face_id_map = np.full((image_size, image_size), -1, dtype=np.int16)
    projected_pixels = np.zeros(num_faces, dtype=np.int64)

    for mesh in meshes:
        projected_mask = np.zeros((image_size, image_size), dtype=bool)
        xy_all, z_all = project_points(mesh.vertices, camera)

        for tri in mesh.triangles:
            xy = xy_all[tri]
            z = z_all[tri]
            pixels = triangle_pixels(xy, z, image_size)
            if pixels is None:
                continue
            ys, xs, zs = pixels
            projected_mask[ys, xs] = True

            current = depth_map[ys, xs]
            update = zs < current
            if np.any(update):
                update_y = ys[update]
                update_x = xs[update]
                depth_map[update_y, update_x] = zs[update]
                face_id_map[update_y, update_x] = mesh.face_id

        projected_pixels[mesh.face_id] = int(projected_mask.sum())

    valid_mask = face_id_map >= 0
    visible_pixels = np.zeros(num_faces, dtype=np.int64)
    if np.any(valid_mask):
        visible_pixels = np.bincount(face_id_map[valid_mask].astype(np.int64), minlength=num_faces)

    visible_fraction = np.zeros(num_faces, dtype=np.float32)
    nonzero = projected_pixels > 0
    visible_fraction[nonzero] = (
        visible_pixels[nonzero].astype(np.float32) / projected_pixels[nonzero].astype(np.float32)
    )
    screen_coverage = visible_pixels.astype(np.float32) / float(image_size * image_size)

    depth_out = depth_map.copy()
    depth_out[~valid_mask] = np.nan

    return {
        "face_id_map": face_id_map,
        "depth_map": depth_out,
        "valid_mask": valid_mask,
        "visible_pixels": visible_pixels,
        "projected_pixels": projected_pixels,
        "visible_fraction": visible_fraction,
        "screen_coverage": screen_coverage,
    }


def color_for_face_ids(face_id_map: np.ndarray) -> np.ndarray:
    h, w = face_id_map.shape
    image = np.zeros((h, w, 3), dtype=np.uint8)
    valid = face_id_map >= 0
    ids = face_id_map.astype(np.int64)
    image[..., 0] = ((ids * 37 + 53) % 255).astype(np.uint8)
    image[..., 1] = ((ids * 67 + 97) % 255).astype(np.uint8)
    image[..., 2] = ((ids * 101 + 193) % 255).astype(np.uint8)
    image[~valid] = 0
    return image


def visibility_heatmap(face_id_map: np.ndarray, visible_fraction: np.ndarray) -> np.ndarray:
    h, w = face_id_map.shape
    image = np.zeros((h, w, 3), dtype=np.uint8)
    valid = face_id_map >= 0
    values = np.zeros((h, w), dtype=np.float32)
    values[valid] = visible_fraction[face_id_map[valid].astype(np.int64)]
    image[..., 0] = np.clip(values * 255.0, 0, 255).astype(np.uint8)
    image[..., 1] = np.clip((1.0 - np.abs(values - 0.5) * 2.0) * 255.0, 0, 255).astype(np.uint8)
    image[..., 2] = np.clip((1.0 - values) * 255.0, 0, 255).astype(np.uint8)
    image[~valid] = 0
    return image


def depth_visualization(depth_map: np.ndarray) -> np.ndarray:
    valid = np.isfinite(depth_map)
    image = np.zeros(depth_map.shape, dtype=np.uint8)
    if not np.any(valid):
        return image
    values = depth_map[valid]
    lo = float(np.percentile(values, 1.0))
    hi = float(np.percentile(values, 99.0))
    if hi <= lo:
        hi = lo + 1e-6
    normalized = np.clip((depth_map - lo) / (hi - lo), 0.0, 1.0)
    image[valid] = ((1.0 - normalized[valid]) * 255.0).astype(np.uint8)
    return image


def silhouette_overlay(svr_image: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    base = Image.fromarray(svr_image.astype(np.uint8)).resize(
        (valid_mask.shape[1], valid_mask.shape[0]),
        resample=Image.BILINEAR,
    )
    image = np.asarray(base).copy()
    edges = mask_edges(valid_mask)
    image[edges] = np.array([255, 0, 0], dtype=np.uint8)
    return image


def mask_edges(mask: np.ndarray) -> np.ndarray:
    edges = np.zeros_like(mask, dtype=bool)
    edges[1:, :] |= mask[1:, :] != mask[:-1, :]
    edges[:-1, :] |= mask[1:, :] != mask[:-1, :]
    edges[:, 1:] |= mask[:, 1:] != mask[:, :-1]
    edges[:, :-1] |= mask[:, 1:] != mask[:, :-1]
    return edges


def write_debug_images(
    debug_dir: Path,
    model_id: str,
    view_id: int,
    render: dict[str, np.ndarray],
    visible_fraction: np.ndarray,
    condition_root: Path,
) -> None:
    view_dir = debug_dir / model_id / f"view_{view_id:03d}"
    view_dir.mkdir(parents=True, exist_ok=True)

    Image.fromarray(color_for_face_ids(render["face_id_map"])).save(view_dir / "face_id_color.png")
    Image.fromarray(visibility_heatmap(render["face_id_map"], visible_fraction)).save(
        view_dir / "visibility_heatmap.png"
    )
    Image.fromarray(depth_visualization(render["depth_map"])).save(view_dir / "depth.png")
    Image.fromarray((render["valid_mask"].astype(np.uint8) * 255)).save(view_dir / "silhouette.png")

    imgs_path = condition_root / model_id / "imgs.npz"
    if imgs_path.is_file():
        with np.load(imgs_path) as imgs:
            if "svr_imgs" in imgs and imgs["svr_imgs"].shape[0] > view_id:
                svr = imgs["svr_imgs"][view_id]
                Image.fromarray(svr.astype(np.uint8)).save(view_dir / "svr.png")
                Image.fromarray(silhouette_overlay(svr, render["valid_mask"])).save(
                    view_dir / "overlay_svr_silhouette.png"
                )

    summary = {
        "model_id": model_id,
        "view_id": view_id,
        "visible_fraction": {
            str(idx): float(value) for idx, value in enumerate(visible_fraction.tolist())
        },
    }
    (view_dir / "visibility.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def save_visibility_npz(
    output_path: Path,
    render: dict[str, np.ndarray],
    camera: Camera,
    order_check: FaceOrderCheck,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    num_faces = int(render["visible_fraction"].shape[0])
    identity = np.arange(num_faces, dtype=np.int16)

    np.savez_compressed(
        output_path,
        face_id_map=render["face_id_map"].astype(np.int16),
        depth_map=render["depth_map"].astype(np.float32),
        valid_mask=render["valid_mask"],
        visible_pixels=render["visible_pixels"].astype(np.int64),
        projected_pixels=render["projected_pixels"].astype(np.int64),
        visible_fraction=render["visible_fraction"].astype(np.float32),
        screen_coverage=render["screen_coverage"].astype(np.float32),
        occ_to_data_face_id=identity,
        data_to_occ_face_id=identity,
        matching_cost=order_check.direct_costs.astype(np.float32),
        order_status=np.array(order_check.status),
        camera_K=camera.intrinsic,
        camera_T_world_to_cam=camera.world_to_cam,
        camera_T_cam_to_world=camera.cam_to_world,
        image_size=np.array([camera.image_size, camera.image_size], dtype=np.int32),
        fovy=np.array(camera.fovy_degrees, dtype=np.float32),
        eye=camera.eye.astype(np.float32),
        center=camera.center.astype(np.float32),
        up=camera.up.astype(np.float32),
        view_id=np.array(camera.rotation_id, dtype=np.int32),
        rotation_basis=np.array(camera.rotation_basis),
    )


def summarize_results(results: list[dict[str, object]]) -> dict[str, object]:
    summary: dict[str, object] = {
        "total": len(results),
        "accepted": 0,
        "skipped": 0,
        "failed": 0,
        "by_status": {},
    }
    by_status: dict[str, int] = {}
    visible_means: list[float] = []
    coverage_means: list[float] = []

    for result in results:
        status = str(result.get("status", "unknown"))
        by_status[status] = by_status.get(status, 0) + 1
        if status in {"generated", "audited", "debugged", "exists"}:
            summary["accepted"] = int(summary["accepted"]) + 1
        elif status.startswith("skip"):
            summary["skipped"] = int(summary["skipped"]) + 1
        elif status == "failed":
            summary["failed"] = int(summary["failed"]) + 1

        if "mean_visible_fraction" in result:
            visible_means.append(float(result["mean_visible_fraction"]))
        if "mean_screen_coverage" in result:
            coverage_means.append(float(result["mean_screen_coverage"]))

    summary["by_status"] = by_status
    if visible_means:
        summary["mean_visible_fraction"] = float(np.mean(visible_means))
    if coverage_means:
        summary["mean_screen_coverage"] = float(np.mean(coverage_means))
    return summary


def process_model(model_id: str, args: argparse.Namespace, views: list[int], debug_index: int) -> dict[str, object]:
    raw_dir = args.raw_root / model_id
    step_path = raw_dir / "normalized_shape.step"
    data_path = raw_dir / "data.npz"
    if not step_path.is_file():
        return {"model_id": model_id, "status": "skip_missing_step", "reason": str(step_path)}
    if not data_path.is_file():
        return {"model_id": model_id, "status": "skip_missing_data", "reason": str(data_path)}

    try:
        with np.load(data_path) as data:
            if "sample_points_faces" not in data:
                return {"model_id": model_id, "status": "skip_missing_sample_points_faces"}
            data_face_samples = data["sample_points_faces"].astype(np.float32)

        shape, faces = load_shape_and_faces(step_path)
        order_check = check_direct_face_order(faces, data_face_samples, args)
        result: dict[str, object] = {
            "model_id": model_id,
            "face_count": int(data_face_samples.shape[0]),
            "order_status": order_check.status,
            "order_reason": order_check.reason,
            "max_direct_cost": float(order_check.direct_costs.max(initial=0.0)),
            "mean_direct_cost": float(order_check.direct_costs.mean()) if order_check.direct_costs.size else 0.0,
        }
        if not order_check.ok:
            result["status"] = f"skip_{order_check.status}"
            return result

        if args.mode == "audit":
            result["status"] = "audited"
            return result

        meshes = triangulate_faces(shape, faces, args.mesh_deflection)
        visible_means = []
        coverage_means = []
        generated_views = []

        save_debug_for_model = args.save_debug and debug_index < args.debug_model_limit
        debug_views = set(views[: args.debug_view_limit]) if save_debug_for_model else set()

        for view_id in views:
            output_path = args.output_root / model_id / f"view_{view_id:03d}" / "visibility.npz"
            if output_path.is_file() and not args.overwrite:
                generated_views.append(view_id)
                continue

            camera = build_camera(view_id, args.image_size, args.fovy, args.up_mode)
            render = render_visibility(meshes, camera)

            sanity = sanity_check_render(render)
            if sanity is not None:
                result["status"] = "failed"
                result["reason"] = f"view {view_id}: {sanity}"
                return result

            save_visibility_npz(output_path, render, camera, order_check)

            if view_id in debug_views:
                write_debug_images(
                    args.output_root / "debug",
                    model_id,
                    view_id,
                    render,
                    render["visible_fraction"],
                    args.condition_root,
                )

            visible_means.append(float(np.mean(render["visible_fraction"])))
            coverage_means.append(float(np.mean(render["screen_coverage"])))
            generated_views.append(view_id)

        result["status"] = "debugged" if save_debug_for_model else "generated"
        result["views"] = generated_views
        if visible_means:
            result["mean_visible_fraction"] = float(np.mean(visible_means))
        if coverage_means:
            result["mean_screen_coverage"] = float(np.mean(coverage_means))
        return result

    except Exception as exc:  # noqa: BLE001 - batch scripts must report and continue.
        return {
            "model_id": model_id,
            "status": "failed",
            "reason": str(exc),
            "traceback": traceback.format_exc(),
        }


def sanity_check_render(render: dict[str, np.ndarray]) -> str | None:
    face_id_map = render["face_id_map"]
    depth_map = render["depth_map"]
    valid_mask = render["valid_mask"]
    visible_pixels = render["visible_pixels"]
    projected_pixels = render["projected_pixels"]
    visible_fraction = render["visible_fraction"]

    if not np.all((face_id_map >= -1) & (face_id_map < len(visible_pixels))):
        return "face_id_map contains invalid ids"
    if np.any(visible_pixels < 0) or np.any(projected_pixels < 0):
        return "negative pixel counts"
    if np.any(visible_pixels > projected_pixels):
        return "visible_pixels exceeds projected_pixels"
    if np.any((visible_fraction < -1e-6) | (visible_fraction > 1.0 + 1e-6)):
        return "visible_fraction outside [0, 1]"
    if int(visible_pixels.sum()) != int(valid_mask.sum()):
        return "sum(visible_pixels) != foreground pixels"
    if np.any(~np.isfinite(depth_map[valid_mask])):
        return "non-finite depth on valid pixels"
    if int(valid_mask.sum()) == 0:
        return "empty render"
    return None


def write_reports(args: argparse.Namespace, results: list[dict[str, object]]) -> tuple[Path, Path]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = args.output_root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = report_dir / f"{timestamp}_{args.mode}.jsonl"
    summary_path = report_dir / f"{timestamp}_{args.mode}_summary.json"

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")

    summary = summarize_results(results)
    summary["mode"] = args.mode
    summary["model_list"] = str(args.model_list)
    summary["views"] = args.views
    summary["image_size"] = args.image_size
    summary["output_root"] = str(args.output_root)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return jsonl_path, summary_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["audit", "debug", "generate"], default="audit")
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--condition-root", type=Path, default=DEFAULT_CONDITION_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model-list", type=Path, default=DEFAULT_MODEL_LIST)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--views", type=str, default="all", help="'all', a comma list, or a range like 0-23")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--fovy", type=float, default=45.0)
    parser.add_argument("--up-mode", choices=["legacy_point", "z"], default="legacy_point")
    parser.add_argument("--sample-resolution", type=int, default=16)
    parser.add_argument("--mesh-deflection", type=float, default=0.001)
    parser.add_argument("--max-direct-cost", type=float, default=1e-3)
    parser.add_argument("--save-debug", action="store_true")
    parser.add_argument("--debug-model-limit", type=int, default=5)
    parser.add_argument("--debug-view-limit", type=int, default=24)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    args.raw_root = args.raw_root.resolve()
    args.condition_root = args.condition_root.resolve()
    args.output_root = args.output_root.resolve()
    args.model_list = args.model_list.resolve()

    disable_occ_log()

    model_ids = read_model_ids(args.model_list, args.limit)
    views = parse_views(args.views)
    args.output_root.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, object]] = []
    for debug_index, model_id in enumerate(tqdm(model_ids, desc=f"{args.mode} visibility")):
        results.append(process_model(model_id, args, views, debug_index))

    jsonl_path, summary_path = write_reports(args, results)
    print(f"Wrote report: {jsonl_path}")
    print(f"Wrote summary: {summary_path}")
    print(json.dumps(summarize_results(results), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
