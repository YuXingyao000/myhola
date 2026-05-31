"""Debug identity-first 24-rotation migration from legacy euler64 assets.

The debug output is organized by identity24 id, where:
    identity24 id 0 = legacy euler64 id 0 = raw Blender cube24 id 18

This is the basis requested for migration: the current single-view FLUX image is
written as real_photo.npz["flux"] and treated as rotation id 0.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation


OLD_COND = Path("/mnt/d/data/deepcad_v6_cond")
NEW_COND = Path("/mnt/d/data/deepcad_v7_cond")
DATA_ROOT = Path("/mnt/d/data/deepcad_v6")
AE_CACHE_OLD = Path("/mnt/d/data/ae_cache/1119_deepcad_aug1_11k")
AE_CACHE_NEW = Path("/mnt/d/data/ae_cache/1119_deepcad_aug1_11k_24")
OUTPUT_ROOT = Path("debug_output")
DEFAULT_BLENDER_SAMPLE_ROOT = Path("blender_cube24_sample")

NUM_IDENTITY24_VIEWS = 24
NUM_EULER64_VIEWS = 64
SINGLE_VIEW_IDENTITY24_ID = 0

AXIS_DIRECTIONS = (
    np.array([1.0, 0.0, 0.0]),
    np.array([-1.0, 0.0, 0.0]),
    np.array([0.0, 1.0, 0.0]),
    np.array([0.0, -1.0, 0.0]),
    np.array([0.0, 0.0, 1.0]),
    np.array([0.0, 0.0, -1.0]),
)


def matrix_key(matrix: np.ndarray) -> tuple[int, ...]:
    return tuple(int(v) for v in np.rint(matrix).astype(np.int8).reshape(-1))


def build_blender24_matrices() -> list[np.ndarray]:
    matrices: list[np.ndarray] = []
    seen: set[tuple[int, ...]] = set()
    for z_axis in AXIS_DIRECTIONS:
        for y_axis in AXIS_DIRECTIONS:
            if abs(float(np.dot(z_axis, y_axis))) > 1e-6:
                continue
            x_axis = np.cross(y_axis, z_axis)
            matrix = np.stack([x_axis, y_axis, z_axis], axis=1)
            key = matrix_key(matrix)
            if key in seen:
                continue
            seen.add(key)
            matrices.append(matrix)
    if len(matrices) != NUM_IDENTITY24_VIEWS:
        raise RuntimeError(f"Expected 24 Blender rotations, got {len(matrices)}")
    return matrices


def build_euler64_matrices() -> list[np.ndarray]:
    matrices = []
    for idx in range(NUM_EULER64_VIEWS):
        angles = np.array([idx % 4, idx // 4 % 4, idx // 16], dtype=np.float32)
        matrices.append(Rotation.from_euler("xyz", angles * np.pi / 2).as_matrix())
    return matrices


def build_identity24_tables() -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], dict[int, list[int]]]:
    blender_mats = build_blender24_matrices()
    euler_mats = build_euler64_matrices()

    identity_to_euler: list[int] = []
    key_to_identity: dict[tuple[int, ...], int] = {}
    for euler_id, euler_matrix in enumerate(euler_mats):
        key = matrix_key(euler_matrix)
        if key in key_to_identity:
            continue
        key_to_identity[key] = len(identity_to_euler)
        identity_to_euler.append(euler_id)

    if len(identity_to_euler) != NUM_IDENTITY24_VIEWS:
        raise RuntimeError(f"Expected 24 unique euler64 rotations, got {len(identity_to_euler)}")
    if identity_to_euler[0] != 0:
        raise RuntimeError("identity24 id 0 must map to euler64 id 0")

    identity_to_blender: list[int] = []
    for euler_id in identity_to_euler:
        euler_matrix = euler_mats[euler_id]
        for blender_id, blender_matrix in enumerate(blender_mats):
            if np.allclose(euler_matrix, blender_matrix, atol=1e-6):
                identity_to_blender.append(blender_id)
                break
        else:
            raise RuntimeError(f"euler64 id {euler_id} has no raw Blender24 counterpart")

    blender_to_identity = [-1] * NUM_IDENTITY24_VIEWS
    for identity_id, blender_id in enumerate(identity_to_blender):
        blender_to_identity[blender_id] = identity_id
    if any(value < 0 for value in blender_to_identity):
        raise RuntimeError("Failed to invert raw Blender24 -> identity24 mapping")
    if identity_to_blender[0] != 18:
        raise RuntimeError("identity24 id 0 should be raw Blender24 id 18")

    identity_to_all_euler = {identity_id: [] for identity_id in range(NUM_IDENTITY24_VIEWS)}
    for euler_id, euler_matrix in enumerate(euler_mats):
        identity_id = key_to_identity[matrix_key(euler_matrix)]
        identity_to_all_euler[identity_id].append(euler_id)

    return tuple(identity_to_euler), tuple(identity_to_blender), tuple(blender_to_identity), identity_to_all_euler


IDENTITY24_TO_EULER64, IDENTITY24_TO_BLENDER24, BLENDER24_TO_IDENTITY24, IDENTITY24_TO_ALL_EULER64 = (
    build_identity24_tables()
)
EULER64_IDS = np.array(IDENTITY24_TO_EULER64, dtype=np.int64)


def build_rotation_meta() -> dict[str, object]:
    return {
        "rotation_basis": "identity_first_cube24",
        "num_views": NUM_IDENTITY24_VIEWS,
        "identity24_to_euler64": list(IDENTITY24_TO_EULER64),
        "identity24_to_blender24": list(IDENTITY24_TO_BLENDER24),
        "blender24_to_identity24": list(BLENDER24_TO_IDENTITY24),
        "real_photo_identity24_id": SINGLE_VIEW_IDENTITY24_ID,
        "real_photo_blender24_id": IDENTITY24_TO_BLENDER24[SINGLE_VIEW_IDENTITY24_ID],
        "real_photo_euler64_id": IDENTITY24_TO_EULER64[SINGLE_VIEW_IDENTITY24_ID],
    }


def save_png(array: np.ndarray, path: Path, size: tuple[int, int] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(array.astype(np.uint8)).convert("RGB")
    if size is not None and image.size != size:
        image = image.resize(size, Image.BILINEAR)
    image.save(path)


def load_rgb(path: Path, size: tuple[int, int] | None = None) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("RGB")
        if size is not None and image.size != size:
            image = image.resize(size, Image.BILINEAR)
        return np.asarray(image, dtype=np.uint8)


def resize_array(array: np.ndarray, size: tuple[int, int] = (224, 224)) -> np.ndarray:
    return np.asarray(Image.fromarray(array.astype(np.uint8)).convert("RGB").resize(size, Image.BILINEAR), dtype=np.uint8)


def foreground_mask(image: np.ndarray) -> np.ndarray:
    return (image[..., :3] < 245).any(axis=-1)


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return float(intersection / union) if union else 0.0


def write_mapping_table(output: Path) -> None:
    lines = [
        "identity24_id | old_blender24_id | old_euler64_id | all_euler64_duplicates | role",
        "-" * 92,
    ]
    for identity_id in range(NUM_IDENTITY24_VIEWS):
        role = "real_photo_flux_zero" if identity_id == SINGLE_VIEW_IDENTITY24_ID else ""
        lines.append(
            f"{identity_id:13d} | {IDENTITY24_TO_BLENDER24[identity_id]:16d} | "
            f"{IDENTITY24_TO_EULER64[identity_id]:14d} | "
            f"{IDENTITY24_TO_ALL_EULER64[identity_id]} | {role}"
        )
    (output / "mapping_table.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def compare_blender_sample(model_id: str, blender_sample_root: Path, identity_images: np.ndarray, output: Path) -> None:
    sample_dir = blender_sample_root / model_id
    if not sample_dir.is_dir():
        print(f"- blender sample not found: {sample_dir}")
        return

    missing = [idx for idx in range(NUM_IDENTITY24_VIEWS) if not (sample_dir / f"{idx:02d}.png").is_file()]
    if missing:
        print(f"- blender sample incomplete: missing {missing}")
        return

    reordered_dir = output / "blender_reordered_identity24"
    side_by_side_dir = output / "side_by_side_identity24"
    reordered_dir.mkdir(exist_ok=True)
    side_by_side_dir.mkdir(exist_ok=True)

    for identity_id in range(NUM_IDENTITY24_VIEWS):
        old_blender_id = IDENTITY24_TO_BLENDER24[identity_id]
        blender_img = load_rgb(sample_dir / f"{old_blender_id:02d}.png", size=(224, 224))
        occ_img = identity_images[identity_id]
        save_png(blender_img, reordered_dir / f"{identity_id:02d}_from_blender24_{old_blender_id:02d}.png")
        separator = np.ones((224, 4, 3), dtype=np.uint8) * 128
        combined = np.concatenate([blender_img, separator, occ_img], axis=1)
        save_png(
            combined,
            side_by_side_dir
            / f"identity24_{identity_id:02d}_blender24_{old_blender_id:02d}_euler64_{IDENTITY24_TO_EULER64[identity_id]:02d}.png",
        )
    print("✓ blender_reordered_identity24/ (00.png is old Blender 18)")
    print("✓ side_by_side_identity24/ (left: reordered Blender, right: migrated SVR/OCC)")

    blender_masks = [foreground_mask(load_rgb(sample_dir / f"{idx:02d}.png", size=(224, 224))) for idx in range(24)]
    occ_masks = [foreground_mask(identity_images[idx]) for idx in range(24)]
    scores = np.zeros((24, 24), dtype=np.float64)
    for identity_id in range(24):
        for blender_id in range(24):
            scores[identity_id, blender_id] = mask_iou(occ_masks[identity_id], blender_masks[blender_id])

    row_ind, col_ind = linear_sum_assignment(-scores)
    assignment = [None] * 24
    for row, col in zip(row_ind, col_ind):
        assignment[int(row)] = int(col)

    lines = [
        f"model_id: {model_id}",
        f"blender_sample_root: {blender_sample_root}",
        "rows are new identity24 ids; columns are raw Blender24 file ids.",
        f"expected_identity24_to_blender24: {list(IDENTITY24_TO_BLENDER24)}",
        f"one_to_one_assignment_identity24_to_blender24: {assignment}",
        "note: symmetric models can make one-model matching ambiguous; inspect side_by_side_identity24.",
        "",
        "identity24_id | expected_blender24 | assigned_blender24 | assigned_iou | top5_blender24_iou",
        "-" * 100,
    ]
    for identity_id in range(24):
        top_ids = np.argsort(scores[identity_id])[::-1][:5]
        top_text = [(int(idx), round(float(scores[identity_id, idx]), 4)) for idx in top_ids]
        assigned = assignment[identity_id]
        lines.append(
            f"{identity_id:13d} | {IDENTITY24_TO_BLENDER24[identity_id]:18d} | "
            f"{assigned:18d} | {scores[identity_id, assigned]:12.4f} | {top_text}"
        )
    (output / "identity24_to_blender24_match.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"✓ identity24_to_blender24_match.txt assignment={assignment}")


def compare_real_photo(model_id: str, blender_sample_root: Path, identity_images: np.ndarray, output: Path) -> None:
    single_view_path = OLD_COND / model_id / "single_view.npz"
    sample_dir = blender_sample_root / model_id
    if not single_view_path.is_file():
        return

    with np.load(single_view_path) as single_data:
        if "blender" in single_data:
            single_img = resize_array(single_data["blender"])
            single_key = "blender"
        elif "flux" in single_data:
            single_img = resize_array(single_data["flux"])
            single_key = "flux"
        else:
            return

        result_dir = output / "result_sample"
        save_png(single_img, result_dir / f"real_photo_{single_key}_identity24_00.png")
        for key in ("flux", "flux_masked"):
            if key in single_data:
                save_png(resize_array(single_data[key]), result_dir / f"real_photo_{key}_identity24_00.png")

    if sample_dir.is_dir() and all((sample_dir / f"{idx:02d}.png").is_file() for idx in range(24)):
        reference_images = [
            load_rgb(sample_dir / f"{IDENTITY24_TO_BLENDER24[identity_id]:02d}.png", size=(224, 224))
            for identity_id in range(24)
        ]
        reference_label = "reordered Blender sample"
    else:
        reference_images = [identity_images[identity_id] for identity_id in range(24)]
        reference_label = "migrated SVR/OCC fallback"

    single_mask = foreground_mask(single_img)
    scores = [mask_iou(single_mask, foreground_mask(reference_images[identity_id])) for identity_id in range(24)]
    order = np.argsort(scores)[::-1]

    lines = [
        f"model_id: {model_id}",
        f"real_photo_key_used_for_matching: {single_key}",
        f"reference: {reference_label}",
        "expected_identity24_id: 0",
        f"identity24_0_old_blender24_id: {IDENTITY24_TO_BLENDER24[0]}",
        f"identity24_0_old_euler64_id: {IDENTITY24_TO_EULER64[0]}",
        "top_matches:",
    ]
    for identity_id in order[:8]:
        lines.append(f"  identity24 {int(identity_id):02d}: iou={scores[int(identity_id)]:.4f}")

    if sample_dir.is_dir() and (sample_dir / f"{IDENTITY24_TO_BLENDER24[0]:02d}.png").is_file():
        blender_zero = load_rgb(sample_dir / f"{IDENTITY24_TO_BLENDER24[0]:02d}.png", size=(224, 224))
        separator = np.ones((224, 4, 3), dtype=np.uint8) * 128
        combined = np.concatenate([single_img, separator, blender_zero, separator, identity_images[0]], axis=1)
        save_png(combined, output / "real_photo_vs_identity24_00.png")
        lines.append("side_by_side: real_photo_vs_identity24_00.png = real_photo | Blender old 18 | SVR euler64 0")

    (output / "real_photo_to_identity24_match.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"✓ real_photo_to_identity24_match.txt best={int(order[0])}")

def compare_all_blender_samples(blender_sample_root: Path, output: Path) -> None:
    if not blender_sample_root.is_dir():
        return

    sample_ids = sorted(path.name for path in blender_sample_root.iterdir() if path.is_dir())
    if not sample_ids:
        return

    aggregate_scores = np.zeros((24, 24), dtype=np.float64)
    used_model_ids = []
    per_model_assignments = {}

    for sample_id in sample_ids:
        sample_dir = blender_sample_root / sample_id
        imgs_path = OLD_COND / sample_id / "imgs.npz"
        if not imgs_path.is_file():
            continue
        if any(not (sample_dir / f"{idx:02d}.png").is_file() for idx in range(24)):
            continue

        with np.load(imgs_path) as data:
            identity_images = data["svr_imgs"][EULER64_IDS]

        blender_masks = [foreground_mask(load_rgb(sample_dir / f"{idx:02d}.png", size=(224, 224))) for idx in range(24)]
        occ_masks = [foreground_mask(identity_images[idx]) for idx in range(24)]
        scores = np.zeros((24, 24), dtype=np.float64)
        for identity_id in range(24):
            for blender_id in range(24):
                scores[identity_id, blender_id] = mask_iou(occ_masks[identity_id], blender_masks[blender_id])

        row_ind, col_ind = linear_sum_assignment(-scores)
        assignment = [None] * 24
        for row, col in zip(row_ind, col_ind):
            assignment[int(row)] = int(col)
        per_model_assignments[sample_id] = assignment
        aggregate_scores += scores
        used_model_ids.append(sample_id)

    if not used_model_ids:
        return

    aggregate_scores /= len(used_model_ids)
    row_ind, col_ind = linear_sum_assignment(-aggregate_scores)
    aggregate_assignment = [None] * 24
    for row, col in zip(row_ind, col_ind):
        aggregate_assignment[int(row)] = int(col)

    lines = [
        f"blender_sample_root: {blender_sample_root}",
        f"used_model_ids: {used_model_ids}",
        f"expected_identity24_to_blender24: {list(IDENTITY24_TO_BLENDER24)}",
        f"aggregate_assignment_identity24_to_blender24: {aggregate_assignment}",
        "",
        "per_model_assignments:",
    ]
    for sample_id in used_model_ids:
        lines.append(f"  {sample_id}: {per_model_assignments[sample_id]}")
    lines.extend([
        "",
        "identity24_id | expected_blender24 | aggregate_blender24 | aggregate_iou | top5_blender24_iou",
        "-" * 100,
    ])
    for identity_id in range(24):
        top_ids = np.argsort(aggregate_scores[identity_id])[::-1][:5]
        top_text = [(int(idx), round(float(aggregate_scores[identity_id, idx]), 4)) for idx in top_ids]
        assigned = aggregate_assignment[identity_id]
        lines.append(
            f"{identity_id:13d} | {IDENTITY24_TO_BLENDER24[identity_id]:18d} | "
            f"{assigned:18d} | {aggregate_scores[identity_id, assigned]:13.4f} | {top_text}"
        )
    (output / "identity24_to_blender24_match_all_samples.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(f"✓ identity24_to_blender24_match_all_samples.txt assignment={aggregate_assignment}")


def write_dino_feature_debug(model_id: str, output: Path) -> None:
    feat_path = OLD_COND / model_id / "img_feature_dinov2.npy"
    if not feat_path.is_file():
        return

    features = np.load(feat_path)
    expected_rows = 64 + 64 + 8 * 64
    if features.ndim != 2 or features.shape[0] != expected_rows:
        return

    dim = features.shape[1]
    svr_feats = features[:64]
    sketch_feats = features[64:128]
    new_features = np.concatenate(
        [svr_feats[EULER64_IDS], sketch_feats[EULER64_IDS]],
        axis=0,
    )
    np.save(output / "result_sample" / "img_feature_dinov2.npy", new_features)
    print(f"✓ result_sample/img_feature_dinov2.npy {new_features.shape}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--blender-sample-root", type=Path, default=DEFAULT_BLENDER_SAMPLE_ROOT)
    parser.add_argument("--skip-blender-compare", action="store_true")
    args = parser.parse_args()

    model_id = args.model_id
    output = args.output or OUTPUT_ROOT / model_id
    output.mkdir(parents=True, exist_ok=True)

    print(f"Model: {model_id}")
    print(f"Output: {output}")
    print("Basis: identity_first_cube24")
    print(f"identity24 -> euler64: {list(IDENTITY24_TO_EULER64)}")
    print(f"identity24 -> raw Blender24: {list(IDENTITY24_TO_BLENDER24)}")
    print("real_photo.npz['flux'] / single-view FLUX id: identity24 0")
    print()

    mesh_copied = False
    for mesh_name in ("mesh.stl", "mesh.ply"):
        mesh_src = DATA_ROOT / model_id / mesh_name
        if mesh_src.is_file():
            shutil.copy2(mesh_src, output / mesh_name)
            mesh_copied = True
            print(f"✓ {mesh_name}")
    if not mesh_copied:
        print(f"- mesh not found under {DATA_ROOT / model_id}")

    imgs_path = OLD_COND / model_id / "imgs.npz"
    if not imgs_path.is_file():
        print(f"✗ imgs.npz not found: {imgs_path}")
        return

    with np.load(imgs_path) as data:
        svr_imgs = data["svr_imgs"]
        identity_images = svr_imgs[EULER64_IDS]
        result_data = {"svr_imgs": identity_images}
        if "sketch_imgs" in data:
            result_data["sketch_imgs"] = data["sketch_imgs"][EULER64_IDS]
    print(f"✓ imgs.npz loaded: svr_imgs {svr_imgs.shape}")

    (output / "rotation_meta.json").write_text(
        json.dumps(build_rotation_meta(), indent=2) + "\n",
        encoding="utf-8",
    )
    print("✓ rotation_meta.json")

    image_dir = output / "identity24_svr_images"
    image_dir.mkdir(exist_ok=True)
    for identity_id, euler64_id in enumerate(IDENTITY24_TO_EULER64):
        save_png(
            identity_images[identity_id],
            image_dir
            / f"identity24_{identity_id:02d}_euler64_{euler64_id:02d}_blender24_{IDENTITY24_TO_BLENDER24[identity_id]:02d}.png",
        )
    print("✓ identity24_svr_images/")

    duplicate_dir = output / "euler64_duplicates_by_identity24"
    duplicate_dir.mkdir(exist_ok=True)
    for identity_id in range(24):
        view_dir = duplicate_dir / f"identity24_{identity_id:02d}"
        view_dir.mkdir(exist_ok=True)
        for euler64_id in IDENTITY24_TO_ALL_EULER64[identity_id]:
            save_png(svr_imgs[euler64_id], view_dir / f"euler64_{euler64_id:02d}.png")
    print("✓ euler64_duplicates_by_identity24/")

    result_dir = output / "result_sample"
    result_dir.mkdir(exist_ok=True)
    for legacy_name in ("single_view.npz", "natural.npz", "sketch_and_natural.npz"):
        legacy_path = result_dir / legacy_name
        if legacy_path.exists():
            legacy_path.unlink()
    np.savez_compressed(result_dir / "imgs.npz", **result_data)
    for identity_id in (0, 1, 12, 23):
        save_png(
            identity_images[identity_id],
            result_dir
            / f"svr_identity24_{identity_id:02d}_euler64_{IDENTITY24_TO_EULER64[identity_id]:02d}.png",
        )
    print("✓ result_sample/imgs.npz + selected SVR pngs")

    write_dino_feature_debug(model_id, output)

    single_view_path = OLD_COND / model_id / "single_view.npz"
    if single_view_path.is_file():
        shutil.copy2(single_view_path, result_dir / "real_photo.npz")
        print("✓ result_sample/real_photo.npz")

    latent_dir = result_dir / "latent"
    latent_dir.mkdir(exist_ok=True)
    for identity_id in (0, 1, 12, 23):
        euler64_id = IDENTITY24_TO_EULER64[identity_id]
        src = AE_CACHE_OLD / f"{model_id}_{euler64_id}" / "features.npy"
        if src.is_file():
            feature = np.load(src)
            np.savetxt(
                latent_dir / f"identity24_{identity_id:02d}_from_euler64_{euler64_id:02d}.txt",
                feature[:3],
                fmt="%.6f",
                header=f"shape={feature.shape}",
            )
    print("✓ result_sample/latent/")

    write_mapping_table(output)
    print("✓ mapping_table.txt")

    if not args.skip_blender_compare:
        compare_blender_sample(model_id, args.blender_sample_root, identity_images, output)
        compare_real_photo(model_id, args.blender_sample_root, identity_images, output)
        compare_all_blender_samples(args.blender_sample_root, output)

    migrated_dir = NEW_COND / model_id
    if migrated_dir.is_dir():
        print(f"✓ migrated output exists: {migrated_dir}")
    else:
        print(f"- migrated output not found yet: {migrated_dir}")

    print(f"\nDone. Check: {output}")


if __name__ == "__main__":
    main()
