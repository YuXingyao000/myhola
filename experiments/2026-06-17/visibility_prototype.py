#!/usr/bin/env python3
"""Prototype per-face visibility from DeepCAD v7 sampled face points.

This is a deliberately lightweight first pass for 2026-06-17:
it estimates visibility from data.npz `sample_points_faces` with a software
z-buffer and point splats. It is not a final OCC/mesh face-id renderer.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.brepnet.data.rotations import cube24_rotation_matrix


DEFAULT_RAW_ROOT = Path("/mnt/d/data/deepcad_v7")
DEFAULT_COND_ROOT = Path("/mnt/d/data/deepcad_v7_cond")
DEFAULT_MODEL_LIST = Path("src/brepnet/data/list/deduplicated_deepcad_testing_7_30.txt")
DEFAULT_OUTPUT_DIR = Path("experiments/2026-06-17/visibility_prototype")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--condition-root", type=Path, default=DEFAULT_COND_ROOT)
    parser.add_argument("--model-list", type=Path, default=DEFAULT_MODEL_LIST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-id", action="append", default=None)
    parser.add_argument("--num-models", type=int, default=3)
    parser.add_argument("--view-id", type=int, default=0)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--splat-radius", type=int, default=3)
    parser.add_argument("--camera-distance", type=float, default=2.0)
    parser.add_argument("--camera-fov", type=float, default=45.0)
    return parser.parse_args()


def read_model_ids(args: argparse.Namespace) -> list[str]:
    if args.model_id:
        return args.model_id
    model_ids = [
        line.strip()
        for line in args.model_list.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    return model_ids[: args.num_models]


def camera_basis(distance: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Match the Blender cube24 renderer convention: fixed camera at (d,d,d)
    # looking at the origin, while the model is rotated per view.
    eye = np.array([distance, distance, distance], dtype=np.float32)
    target = np.zeros(3, dtype=np.float32)
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    up_guess = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    right = np.cross(forward, up_guess)
    if np.linalg.norm(right) < 1e-6:
        up_guess = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        right = np.cross(forward, up_guess)
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)
    return eye, forward, right, up


def project_points(
    points: np.ndarray,
    view_id: int,
    resolution: int,
    camera_distance: float,
    camera_fov: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rotation = cube24_rotation_matrix(view_id).astype(np.float32)
    rotated = (rotation @ points.T).T
    eye, forward, right, up = camera_basis(camera_distance)
    rel = rotated - eye[None, :]
    cam_x = rel @ right
    cam_y = rel @ up
    cam_z = rel @ forward
    fov_scale = math.tan(math.radians(camera_fov) * 0.5)
    ndc_x = cam_x / np.maximum(cam_z * fov_scale, 1e-6)
    ndc_y = cam_y / np.maximum(cam_z * fov_scale, 1e-6)
    pixel_x = (ndc_x + 1.0) * 0.5 * (resolution - 1)
    pixel_y = (1.0 - ndc_y) * 0.5 * (resolution - 1)
    in_frame = (
        (cam_z > 1e-6)
        & (pixel_x >= 0)
        & (pixel_x < resolution)
        & (pixel_y >= 0)
        & (pixel_y < resolution)
    )
    pixels = np.stack([pixel_x, pixel_y], axis=1)
    return pixels, cam_z, in_frame


def face_palette(num_faces: int) -> np.ndarray:
    cmap = plt.get_cmap("tab20", max(num_faces, 1))
    colors = (cmap(np.arange(num_faces))[:, :3] * 255).astype(np.uint8)
    return colors


def estimate_visibility(
    face_points: np.ndarray,
    view_id: int,
    resolution: int,
    splat_radius: int,
    camera_distance: float,
    camera_fov: float,
) -> tuple[list[dict], np.ndarray, np.ndarray]:
    num_faces = int(face_points.shape[0])
    points = face_points[..., :3].reshape(num_faces, -1, 3).astype(np.float32)
    flat_points = points.reshape(-1, 3)
    face_ids = np.repeat(np.arange(num_faces), points.shape[1])
    pixels, depths, in_frame = project_points(
        flat_points,
        view_id=view_id,
        resolution=resolution,
        camera_distance=camera_distance,
        camera_fov=camera_fov,
    )

    z_buffer = np.full((resolution, resolution), np.inf, dtype=np.float32)
    face_buffer = np.full((resolution, resolution), -1, dtype=np.int32)
    projected_masks = [set() for _ in range(num_faces)]

    valid_indices = np.where(in_frame)[0]
    # Front-to-back sorting makes overlapping equal-depth splats deterministic.
    valid_indices = valid_indices[np.argsort(depths[valid_indices])]
    radius = max(0, int(splat_radius))
    for idx in valid_indices:
        x = int(round(float(pixels[idx, 0])))
        y = int(round(float(pixels[idx, 1])))
        z = float(depths[idx])
        face_id = int(face_ids[idx])
        x0 = max(0, x - radius)
        x1 = min(resolution - 1, x + radius)
        y0 = max(0, y - radius)
        y1 = min(resolution - 1, y + radius)
        for yy in range(y0, y1 + 1):
            for xx in range(x0, x1 + 1):
                projected_masks[face_id].add((yy, xx))
                if z < z_buffer[yy, xx]:
                    z_buffer[yy, xx] = z
                    face_buffer[yy, xx] = face_id

    records = []
    for face_id in range(num_faces):
        projected_pixels = len(projected_masks[face_id])
        visible_pixels = int((face_buffer == face_id).sum())
        visible_fraction = visible_pixels / projected_pixels if projected_pixels else 0.0
        records.append(
            {
                "face_id": face_id,
                "projected_pixels": projected_pixels,
                "visible_pixels": visible_pixels,
                "visible_fraction": visible_fraction,
            }
        )
    return records, face_buffer, z_buffer


def load_condition_images(cond_dir: Path, view_id: int) -> tuple[np.ndarray | None, np.ndarray | None]:
    svr = None
    real = None
    imgs_path = cond_dir / "imgs.npz"
    if imgs_path.is_file():
        with np.load(imgs_path) as data:
            imgs = data["svr_imgs"]
            svr = imgs[int(view_id) % imgs.shape[0]]
    real_path = cond_dir / "real_photo.npz"
    if real_path.is_file():
        with np.load(real_path) as data:
            real = data["flux"]
            if real.ndim == 4:
                real = real[int(view_id) % real.shape[0]]
    return svr, real


def save_image(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array.astype(np.uint8)).save(path)


def render_face_id(face_buffer: np.ndarray, num_faces: int) -> np.ndarray:
    image = np.full((*face_buffer.shape, 3), 255, dtype=np.uint8)
    colors = face_palette(num_faces)
    mask = face_buffer >= 0
    image[mask] = colors[face_buffer[mask]]
    return image


def render_visibility_heatmap(face_buffer: np.ndarray, records: list[dict]) -> np.ndarray:
    fractions = np.array([item["visible_fraction"] for item in records], dtype=np.float32)
    heat = np.zeros(face_buffer.shape, dtype=np.float32)
    mask = face_buffer >= 0
    heat[mask] = fractions[face_buffer[mask]]
    rgba = plt.get_cmap("viridis")(heat)
    image = (rgba[:, :, :3] * 255).astype(np.uint8)
    image[~mask] = 255
    return image


def save_panel(path: Path, title: str, images: list[tuple[str, np.ndarray | None]]) -> None:
    cols = len(images)
    fig, axes = plt.subplots(1, cols, figsize=(4 * cols, 4))
    if cols == 1:
        axes = [axes]
    fig.suptitle(title)
    for ax, (label, image) in zip(axes, images):
        ax.set_title(label)
        ax.axis("off")
        if image is not None:
            ax.imshow(image)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def process_model(model_id: str, args: argparse.Namespace) -> dict:
    raw_dir = args.raw_root / model_id
    cond_dir = args.condition_root / model_id
    out_dir = args.output_dir / model_id
    out_dir.mkdir(parents=True, exist_ok=True)
    data_path = raw_dir / "data.npz"
    if not data_path.is_file():
        raise FileNotFoundError(f"Missing data.npz for {model_id}: {data_path}")
    with np.load(data_path) as data:
        face_points = data["sample_points_faces"]
        face_adj = data["face_adj"]
    records, face_buffer, _ = estimate_visibility(
        face_points,
        view_id=args.view_id,
        resolution=args.resolution,
        splat_radius=args.splat_radius,
        camera_distance=args.camera_distance,
        camera_fov=args.camera_fov,
    )
    num_faces = int(face_points.shape[0])
    visible_faces = [item["face_id"] for item in records if item["visible_fraction"] > 0.05]
    strongly_visible_faces = [item["face_id"] for item in records if item["visible_fraction"] > 0.5]
    payload = {
        "model_id": model_id,
        "view_id": args.view_id,
        "method": "sample_points_faces point-splat z-buffer prototype",
        "resolution": args.resolution,
        "splat_radius": args.splat_radius,
        "camera": {
            "location": [args.camera_distance, args.camera_distance, args.camera_distance],
            "target": [0.0, 0.0, 0.0],
            "fov_degrees": args.camera_fov,
        },
        "num_faces": num_faces,
        "num_edges": int(face_adj.sum() // 2),
        "visible_face_count_gt_0p05": len(visible_faces),
        "visible_face_count_gt_0p5": len(strongly_visible_faces),
        "faces": records,
    }
    (out_dir / "visibility.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    face_id_image = render_face_id(face_buffer, num_faces)
    heatmap = render_visibility_heatmap(face_buffer, records)
    save_image(out_dir / "face_id_render.png", face_id_image)
    save_image(out_dir / "visibility_heatmap.png", heatmap)
    svr, real = load_condition_images(cond_dir, args.view_id)
    if svr is not None:
        save_image(out_dir / "svr_view0.png", svr)
    if real is not None:
        save_image(out_dir / "real_photo_flux.png", real)
    save_panel(
        out_dir / "debug_panel.png",
        f"{model_id} view {args.view_id}",
        [
            ("svr", svr),
            ("real_flux", real),
            ("face_id", face_id_image),
            ("visibility", heatmap),
        ],
    )
    return payload


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_ids = read_model_ids(args)
    summaries = []
    failures = []
    for model_id in model_ids:
        try:
            summaries.append(process_model(model_id, args))
            print(f"processed {model_id}", flush=True)
        except Exception as exc:  # keep prototype batch debuggable
            failures.append({"model_id": model_id, "error": repr(exc)})
            print(f"failed {model_id}: {exc}", flush=True)
    summary = {
        "model_ids": model_ids,
        "num_requested": len(model_ids),
        "num_success": len(summaries),
        "num_failed": len(failures),
        "failures": failures,
        "outputs": [
            {
                "model_id": item["model_id"],
                "num_faces": item["num_faces"],
                "visible_face_count_gt_0p05": item["visible_face_count_gt_0p05"],
                "visible_face_count_gt_0p5": item["visible_face_count_gt_0p5"],
                "mean_visible_fraction": float(np.mean([face["visible_fraction"] for face in item["faces"]])),
                "max_visible_fraction": float(np.max([face["visible_fraction"] for face in item["faces"]])),
            }
            for item in summaries
        ],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
