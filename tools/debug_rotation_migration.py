"""Debug Blender cube24 based migration from legacy euler64 assets.

For one model_id this script writes:
1. OCC/SVR images re-indexed into Blender cube24 order.
2. The legacy euler64 duplicates for each Blender cube24 id.
3. A small migrated npz sample and selected latent previews.
4. If a Blender cube24 sample directory exists, a 24x24 visual matching report.

Rotation contract:
    Blender cube24 00.png..23.png is the basis.
    Blender24 -> OCC24 is identity for the checked samples.
    OCC/SVR legacy arrays are euler64, so Blender24 -> euler64 is used here.
    single_view.npz is identity view, i.e. Blender/OCC cube24 id 18.

Usage:
    python tools/debug_rotation_migration.py --model-id 00000797
    python tools/debug_rotation_migration.py --model-id 00000797 --blender-sample-root blender_cube24_sample
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

NUM_BLENDER24_VIEWS = 24
BLENDER24_TO_OCC24 = tuple(range(NUM_BLENDER24_VIEWS))
SINGLE_VIEW_BLENDER_ID = 18


def build_blender24_mappings() -> tuple[tuple[int, ...], tuple[int | None, ...], dict[int, list[int]]]:
    """Return Blender24->euler64, euler64->Blender24, and duplicate groups."""

    dirs = (
        np.array([1.0, 0.0, 0.0]),
        np.array([-1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, -1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, 0.0, -1.0]),
    )

    blender_mats = []
    seen = set()
    for z_axis in dirs:
        for y_axis in dirs:
            if abs(float(np.dot(z_axis, y_axis))) > 1e-6:
                continue
            x_axis = np.cross(y_axis, z_axis)
            matrix = np.stack([x_axis, y_axis, z_axis], axis=1)
            key = tuple(int(round(v)) for v in matrix.flatten())
            if key in seen:
                continue
            seen.add(key)
            blender_mats.append(matrix)
    if len(blender_mats) != NUM_BLENDER24_VIEWS:
        raise RuntimeError(f"Expected 24 Blender rotations, got {len(blender_mats)}")

    euler_mats = []
    for idx in range(64):
        angles = np.array([idx % 4, idx // 4 % 4, idx // 16], dtype=np.float32)
        euler_mats.append(Rotation.from_euler("xyz", angles * np.pi / 2).as_matrix())

    blender_to_euler = []
    for blender_matrix in blender_mats:
        for euler_id, euler_matrix in enumerate(euler_mats):
            if np.allclose(blender_matrix, euler_matrix, atol=1e-6):
                blender_to_euler.append(euler_id)
                break
        else:
            raise RuntimeError("Blender cube24 rotation has no euler64 counterpart")

    euler_to_blender: list[int | None] = [None] * 64
    for euler_id, euler_matrix in enumerate(euler_mats):
        for blender_id, blender_matrix in enumerate(blender_mats):
            if np.allclose(euler_matrix, blender_matrix, atol=1e-6):
                euler_to_blender[euler_id] = blender_id
                break

    blender_to_all_euler = {blender_id: [] for blender_id in range(NUM_BLENDER24_VIEWS)}
    for euler_id, blender_id in enumerate(euler_to_blender):
        if blender_id is not None:
            blender_to_all_euler[blender_id].append(euler_id)

    return tuple(blender_to_euler), tuple(euler_to_blender), blender_to_all_euler


BLENDER24_TO_EULER64, EULER64_TO_BLENDER24, BLENDER24_TO_ALL_EULER64 = build_blender24_mappings()
EULER64_IDS = np.array(BLENDER24_TO_EULER64, dtype=np.int64)


def build_rotation_meta() -> dict[str, object]:
    return {
        "rotation_basis": "blender_cube24",
        "num_views": NUM_BLENDER24_VIEWS,
        "blender24_to_occ24": list(BLENDER24_TO_OCC24),
        "blender24_to_euler64": list(BLENDER24_TO_EULER64),
        "single_view_blender_id": SINGLE_VIEW_BLENDER_ID,
        "single_view_occ24_id": BLENDER24_TO_OCC24[SINGLE_VIEW_BLENDER_ID],
        "single_view_euler64_id": BLENDER24_TO_EULER64[SINGLE_VIEW_BLENDER_ID],
    }


def save_png(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def load_rgb(path: Path, size: tuple[int, int] | None = None) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("RGB")
        if size is not None and image.size != size:
            image = image.resize(size, Image.BILINEAR)
        return np.asarray(image, dtype=np.uint8)


def foreground_mask(image: np.ndarray) -> np.ndarray:
    return (image[..., :3] < 245).any(axis=-1)


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return float(intersection / union) if union else 0.0


def compare_blender_sample(model_id: str, blender_sample_root: Path, occ24_images: np.ndarray, output: Path) -> None:
    sample_dir = blender_sample_root / model_id
    if not sample_dir.is_dir():
        print(f"- blender sample not found: {sample_dir}")
        return

    missing = [idx for idx in range(NUM_BLENDER24_VIEWS) if not (sample_dir / f"{idx:02d}.png").is_file()]
    if missing:
        print(f"- blender sample incomplete: missing {missing}")
        return

    occ_masks = [foreground_mask(occ24_images[idx]) for idx in range(NUM_BLENDER24_VIEWS)]
    scores = np.zeros((NUM_BLENDER24_VIEWS, NUM_BLENDER24_VIEWS), dtype=np.float64)
    for blender_id in range(NUM_BLENDER24_VIEWS):
        blender_img = load_rgb(sample_dir / f"{blender_id:02d}.png", size=(224, 224))
        blender_mask = foreground_mask(blender_img)
        for occ_id in range(NUM_BLENDER24_VIEWS):
            scores[blender_id, occ_id] = mask_iou(blender_mask, occ_masks[occ_id])

    row_ind, col_ind = linear_sum_assignment(-scores)
    assignment = [None] * NUM_BLENDER24_VIEWS
    for row, col in zip(row_ind, col_ind):
        assignment[int(row)] = int(col)

    lines = [
        f"model_id: {model_id}",
        f"blender_sample_root: {blender_sample_root}",
        "note: single-model silhouette matching may be ambiguous for symmetric shapes.",
        f"one_to_one_assignment_blender24_to_occ24: {assignment}",
        "expected_identity: " + str(list(BLENDER24_TO_OCC24)),
        "",
        "blender_id | assigned_occ_id | assigned_iou | top5_occ_id_iou_euler64",
        "-" * 86,
    ]
    for blender_id in range(NUM_BLENDER24_VIEWS):
        top_ids = np.argsort(scores[blender_id])[::-1][:5]
        top_text = [
            (int(occ_id), round(float(scores[blender_id, occ_id]), 4), BLENDER24_TO_EULER64[int(occ_id)])
            for occ_id in top_ids
        ]
        assigned = assignment[blender_id]
        lines.append(
            f"{blender_id:10d} | {assigned:15d} | "
            f"{scores[blender_id, assigned]:12.4f} | {top_text}"
        )

    (output / "blender24_to_occ24_match.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"✓ blender24_to_occ24_match.txt assignment={assignment}")


def compare_single_view(model_id: str, blender_sample_root: Path, output: Path) -> None:
    sample_dir = blender_sample_root / model_id
    single_view_path = OLD_COND / model_id / "single_view.npz"
    if not sample_dir.is_dir() or not single_view_path.is_file():
        return

    single_data = np.load(single_view_path)
    if "blender" not in single_data:
        return

    single_blender = np.asarray(
        Image.fromarray(single_data["blender"]).resize((224, 224), Image.BILINEAR),
        dtype=np.uint8,
    )
    single_mask = foreground_mask(single_blender)
    scores = []
    for blender_id in range(NUM_BLENDER24_VIEWS):
        sample_path = sample_dir / f"{blender_id:02d}.png"
        if not sample_path.is_file():
            return
        sample_mask = foreground_mask(load_rgb(sample_path, size=(224, 224)))
        scores.append(mask_iou(single_mask, sample_mask))

    order = np.argsort(scores)[::-1]
    lines = [
        f"model_id: {model_id}",
        f"single_view_expected_blender_id: {SINGLE_VIEW_BLENDER_ID}",
        "top_matches:",
    ]
    for blender_id in order[:8]:
        lines.append(f"  blender24 {int(blender_id):02d}: iou={scores[int(blender_id)]:.4f}")
    (output / "single_view_to_blender24_match.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"✓ single_view_to_blender24_match.txt best={int(order[0])}")


def compare_all_blender_samples(blender_sample_root: Path, output: Path) -> None:
    """Aggregate Blender24->OCC24 matching over every sample dir under root."""

    if not blender_sample_root.is_dir():
        return

    sample_ids = sorted(path.name for path in blender_sample_root.iterdir() if path.is_dir())
    if not sample_ids:
        return

    aggregate_scores = np.zeros((NUM_BLENDER24_VIEWS, NUM_BLENDER24_VIEWS), dtype=np.float64)
    used_model_ids = []
    per_model_assignments = {}

    for sample_id in sample_ids:
        sample_dir = blender_sample_root / sample_id
        imgs_path = OLD_COND / sample_id / "imgs.npz"
        if not imgs_path.is_file():
            continue
        if any(not (sample_dir / f"{idx:02d}.png").is_file() for idx in range(NUM_BLENDER24_VIEWS)):
            continue

        svr_imgs = np.load(imgs_path)["svr_imgs"]
        occ24_images = svr_imgs[EULER64_IDS]
        occ_masks = [foreground_mask(occ24_images[idx]) for idx in range(NUM_BLENDER24_VIEWS)]
        scores = np.zeros((NUM_BLENDER24_VIEWS, NUM_BLENDER24_VIEWS), dtype=np.float64)

        for blender_id in range(NUM_BLENDER24_VIEWS):
            blender_img = load_rgb(sample_dir / f"{blender_id:02d}.png", size=(224, 224))
            blender_mask = foreground_mask(blender_img)
            for occ_id in range(NUM_BLENDER24_VIEWS):
                scores[blender_id, occ_id] = mask_iou(blender_mask, occ_masks[occ_id])

        row_ind, col_ind = linear_sum_assignment(-scores)
        assignment = [None] * NUM_BLENDER24_VIEWS
        for row, col in zip(row_ind, col_ind):
            assignment[int(row)] = int(col)
        per_model_assignments[sample_id] = assignment
        aggregate_scores += scores
        used_model_ids.append(sample_id)

    if not used_model_ids:
        return

    aggregate_scores /= len(used_model_ids)
    row_ind, col_ind = linear_sum_assignment(-aggregate_scores)
    aggregate_assignment = [None] * NUM_BLENDER24_VIEWS
    for row, col in zip(row_ind, col_ind):
        aggregate_assignment[int(row)] = int(col)

    lines = [
        f"blender_sample_root: {blender_sample_root}",
        f"used_model_ids: {used_model_ids}",
        f"aggregate_assignment_blender24_to_occ24: {aggregate_assignment}",
        "expected_identity: " + str(list(BLENDER24_TO_OCC24)),
        "",
        "per_model_assignments:",
    ]
    for sample_id in used_model_ids:
        lines.append(f"  {sample_id}: {per_model_assignments[sample_id]}")
    lines.extend([
        "",
        "blender_id | aggregate_occ_id | aggregate_iou | top5_occ_id_iou_euler64",
        "-" * 90,
    ])
    for blender_id in range(NUM_BLENDER24_VIEWS):
        top_ids = np.argsort(aggregate_scores[blender_id])[::-1][:5]
        top_text = [
            (int(occ_id), round(float(aggregate_scores[blender_id, occ_id]), 4), BLENDER24_TO_EULER64[int(occ_id)])
            for occ_id in top_ids
        ]
        assigned = aggregate_assignment[blender_id]
        lines.append(
            f"{blender_id:10d} | {assigned:16d} | "
            f"{aggregate_scores[blender_id, assigned]:13.4f} | {top_text}"
        )

    (output / "blender24_to_occ24_match_all_samples.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(f"✓ blender24_to_occ24_match_all_samples.txt assignment={aggregate_assignment}")


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
    print("Basis: Blender cube24")
    print(f"Blender24 -> OCC24: {list(BLENDER24_TO_OCC24)}")
    print(f"Blender24 -> euler64: {list(BLENDER24_TO_EULER64)}")
    print(f"single_view blender/flux id: {SINGLE_VIEW_BLENDER_ID}")
    print()

    mesh_copied = False
    for mesh_name in ("mesh.stl", "mesh.ply"):
        mesh_src = DATA_ROOT / model_id / mesh_name
        if mesh_src.is_file():
            shutil.copy2(mesh_src, output / mesh_name)
            mesh_copied = True
            print(f"✓ {mesh_name}")
    if not mesh_copied:
        print(f"✗ mesh not found under {DATA_ROOT / model_id}")

    imgs_path = OLD_COND / model_id / "imgs.npz"
    if not imgs_path.is_file():
        print(f"✗ imgs.npz not found: {imgs_path}")
        return

    data = np.load(imgs_path)
    svr_imgs = data["svr_imgs"]
    print(f"✓ imgs.npz loaded: svr_imgs {svr_imgs.shape}")

    occ24_images = svr_imgs[EULER64_IDS]

    rotation_meta = build_rotation_meta()
    (output / "rotation_meta.json").write_text(
        json.dumps(rotation_meta, indent=2) + "\n",
        encoding="utf-8",
    )
    print("✓ rotation_meta.json")

    occ_dir = output / "blender24_occ_images"
    occ_dir.mkdir(exist_ok=True)
    for blender_id, euler64_id in enumerate(BLENDER24_TO_EULER64):
        save_png(
            occ24_images[blender_id],
            occ_dir / f"blender24_{blender_id:02d}_occ24_{blender_id:02d}_euler64_{euler64_id:02d}.png",
        )
    print("✓ blender24_occ_images/ (24 OCC/SVR images in Blender basis)")

    mapping_dir = output / "euler64_duplicates_by_blender24"
    mapping_dir.mkdir(exist_ok=True)
    for blender_id in range(NUM_BLENDER24_VIEWS):
        duplicate_dir = mapping_dir / f"blender24_{blender_id:02d}"
        duplicate_dir.mkdir(exist_ok=True)
        for euler64_id in BLENDER24_TO_ALL_EULER64[blender_id]:
            save_png(svr_imgs[euler64_id], duplicate_dir / f"euler64_{euler64_id:02d}.png")
    print("✓ euler64_duplicates_by_blender24/")

    result_dir = output / "result_sample"
    result_dir.mkdir(exist_ok=True)
    np.savez_compressed(result_dir / "svr.npz", images=occ24_images)
    for blender_id in [0, 1, 12, SINGLE_VIEW_BLENDER_ID, 23]:
        save_png(occ24_images[blender_id], result_dir / f"svr_blender24_{blender_id:02d}.png")
    print("✓ result_sample/svr.npz + selected pngs")

    single_view_path = OLD_COND / model_id / "single_view.npz"
    if single_view_path.is_file():
        single_data = np.load(single_view_path)
        shutil.copy2(single_view_path, result_dir / "real_photo.npz")
        key_to_name = {
            "blender": "single_view_blender_blender24_18.png",
            "flux": "single_view_flux_blender24_18.png",
            "flux_masked": "single_view_flux_masked_blender24_18.png",
        }
        for key, filename in key_to_name.items():
            if key in single_data:
                save_png(single_data[key], result_dir / filename)
        print("✓ result_sample/real_photo.npz + single_view_*_blender24_18.png")

    text_path = OLD_COND / model_id / "text.txt"
    text_feat_path = OLD_COND / model_id / "text_feat.npy"
    if text_path.is_file():
        description = text_path.read_text(encoding="utf-8").strip()
        feature = np.load(text_feat_path) if text_feat_path.is_file() else np.zeros((4, 1024), dtype=np.float32)
        np.savez_compressed(result_dir / "text.npz", description=np.array(description), feature=feature)
        (result_dir / "text_content.txt").write_text(description, encoding="utf-8")
        print("✓ result_sample/text.npz")

    latent_dir = result_dir / "latent"
    latent_dir.mkdir(exist_ok=True)
    for blender_id in [0, 1, 12, SINGLE_VIEW_BLENDER_ID, 23]:
        euler64_id = BLENDER24_TO_EULER64[blender_id]
        src = AE_CACHE_OLD / f"{model_id}_{euler64_id}" / "features.npy"
        if src.is_file():
            feature = np.load(src)
            np.savetxt(
                latent_dir / f"blender24_{blender_id:02d}_euler64_{euler64_id:02d}.txt",
                feature[:3],
                fmt="%.6f",
                header=f"shape={feature.shape}",
            )
    print("✓ result_sample/latent/")

    table_lines = [
        "blender24_id | occ24_id | euler64_canonical | all_euler64_duplicates | role",
        "-" * 86,
    ]
    for blender_id in range(NUM_BLENDER24_VIEWS):
        role = "single_view_identity" if blender_id == SINGLE_VIEW_BLENDER_ID else ""
        table_lines.append(
            f"{blender_id:12d} | {BLENDER24_TO_OCC24[blender_id]:8d} | "
            f"{BLENDER24_TO_EULER64[blender_id]:17d} | "
            f"{BLENDER24_TO_ALL_EULER64[blender_id]} | {role}"
        )
    (output / "mapping_table.txt").write_text("\n".join(table_lines) + "\n", encoding="utf-8")
    print("✓ mapping_table.txt")

    if not args.skip_blender_compare:
        compare_blender_sample(model_id, args.blender_sample_root, occ24_images, output)
        compare_single_view(model_id, args.blender_sample_root, output)
        compare_all_blender_samples(args.blender_sample_root, output)

    migrated_dir = NEW_COND / model_id
    if migrated_dir.is_dir():
        print(f"✓ migrated output exists: {migrated_dir}")
    else:
        print(f"- migrated output not found yet: {migrated_dir}")

    print(f"\nDone. Check: {output}")


if __name__ == "__main__":
    main()
