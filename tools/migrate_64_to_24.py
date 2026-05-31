"""Migrate legacy euler64 assets to an identity-first 24-rotation basis.

The target basis is NOT the raw Blender cube24 file order. It is a reordered
cube24 basis where id 0 is the identity view, matching the current single-view
FLUX image that is written as real_photo.npz["flux"].

Key contract:
    identity24 id 0 = legacy euler64 id 0 = old Blender cube24 id 18

Inputs:
    /mnt/d/data/deepcad_v6_cond/{model_id}/imgs.npz
        svr_imgs[64], sketch_imgs[64], mvr_imgs[8 * 64]
    /mnt/d/data/deepcad_v6_cond/{model_id}/single_view.npz
        optional single-view data; this is identity24 id 0
    /mnt/d/data/deepcad_v6_cond/{model_id}/img_feature_dinov2.npy
        optional cached image features, old layout: 64 + 64 + 8 * 64
    /mnt/d/data/ae_cache/1119_deepcad_aug1_11k/{model_id}_{euler64_id}/

Outputs:
    /mnt/d/data/deepcad_v7_cond/{model_id}/imgs.npz
        svr_imgs[24], sketch_imgs[24], identity24 order
    /mnt/d/data/deepcad_v7_cond/{model_id}/real_photo.npz
        optional FLUX data; current shape [H,W,3], future shape [24,H,W,3]
    /mnt/d/data/deepcad_v7_cond/{model_id}/img_feature_dinov2.npy
        optional cached features, new layout: 24 + 24
    /mnt/d/data/ae_cache/1119_deepcad_aug1_11k_24/{model_id}_{identity24_id}/
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


OLD_COND = Path("/mnt/d/data/deepcad_v6_cond")
NEW_COND = Path("/mnt/d/data/deepcad_v7_cond")
AE_CACHE_OLD = Path("/mnt/d/data/ae_cache/1119_deepcad_aug1_11k")
AE_CACHE_NEW = Path("/mnt/d/data/ae_cache/1119_deepcad_aug1_11k_24")

TRAIN_MODEL_LIST = Path("src/brepnet/data/list/deduplicated_deepcad_training_7_30.txt")
VAL_MODEL_LIST = Path("src/brepnet/data/list/deduplicated_deepcad_validation_7_30.txt")
TEST_MODEL_LIST = Path("src/brepnet/data/list/deduplicated_deepcad_testing_7_30.txt")
DEFAULT_MODEL_LISTS = (TRAIN_MODEL_LIST, VAL_MODEL_LIST, TEST_MODEL_LIST)

NUM_IDENTITY24_VIEWS = 24
NUM_EULER64_VIEWS = 64
NUM_MVR_CAMERAS = 8
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
    """Raw Blender render_cube24.py order: 00.png..23.png."""

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


def build_identity24_tables() -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Build identity24 -> euler64 / raw Blender24 / inverse Blender24 mapping.

    identity24 order is the first occurrence of each unique cube rotation while
    scanning legacy euler64 ids from 0 to 63. This makes id 0 the identity view.
    """

    blender_mats = build_blender24_matrices()
    euler_mats = build_euler64_matrices()

    identity_to_euler: list[int] = []
    seen: set[tuple[int, ...]] = set()
    for euler_id, euler_matrix in enumerate(euler_mats):
        key = matrix_key(euler_matrix)
        if key in seen:
            continue
        seen.add(key)
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

    return tuple(identity_to_euler), tuple(identity_to_blender), tuple(blender_to_identity)


IDENTITY24_TO_EULER64, IDENTITY24_TO_BLENDER24, BLENDER24_TO_IDENTITY24 = build_identity24_tables()
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
        "notes": (
            "identity24 id 0 is the identity view and matches real_photo.npz['flux']. "
            "It corresponds to legacy euler64 id 0 and raw Blender cube24 id 18."
        ),
    }


ROTATION_META = build_rotation_meta()


def load_ids_from_list(path: Path) -> list[str]:
    ids = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        model_id = raw_line.strip()
        if model_id and not model_id.startswith("#"):
            ids.append(model_id)
    return ids


def load_model_ids(model_list: Path | None, train_only: bool) -> list[str]:
    if model_list is not None:
        return list(dict.fromkeys(load_ids_from_list(model_list)))

    lists = (TRAIN_MODEL_LIST,) if train_only else DEFAULT_MODEL_LISTS
    ids: list[str] = []
    for path in lists:
        ids.extend(load_ids_from_list(path))
    return list(dict.fromkeys(ids))


def has_complete_legacy_cache(model_id: str) -> bool:
    for euler64_id in IDENTITY24_TO_EULER64:
        feat_path = AE_CACHE_OLD / f"{model_id}_{euler64_id}" / "features.npy"
        if not feat_path.is_file():
            return False
    return True


def write_rotation_meta(output_dir: Path) -> None:
    (output_dir / "rotation_meta.json").write_text(
        json.dumps(ROTATION_META, indent=2) + "\n",
        encoding="utf-8",
    )


def copy_cache_dir(src_dir: Path, dst_dir: Path, overwrite_cache: bool) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for src_file in src_dir.iterdir():
        if not src_file.is_file():
            continue
        dst_file = dst_dir / src_file.name
        if overwrite_cache or not dst_file.is_file():
            shutil.copy2(src_file, dst_file)


def migrate_imgs_npz(old_dir: Path, new_dir: Path) -> None:
    with np.load(old_dir / "imgs.npz") as old:
        new_data = {}
        if "svr_imgs" in old:
            new_data["svr_imgs"] = old["svr_imgs"][EULER64_IDS]
        if "sketch_imgs" in old:
            new_data["sketch_imgs"] = old["sketch_imgs"][EULER64_IDS]

    if new_data:
        np.savez_compressed(new_dir / "imgs.npz", **new_data)


def migrate_dino_features(old_dir: Path, new_dir: Path) -> None:
    feat_path = old_dir / "img_feature_dinov2.npy"
    if not feat_path.is_file():
        return

    features = np.load(feat_path)
    expected_rows = NUM_EULER64_VIEWS + NUM_EULER64_VIEWS + NUM_MVR_CAMERAS * NUM_EULER64_VIEWS
    if features.ndim != 2 or features.shape[0] != expected_rows:
        return

    dim = features.shape[1]
    svr_feats = features[:NUM_EULER64_VIEWS]
    sketch_feats = features[NUM_EULER64_VIEWS : 2 * NUM_EULER64_VIEWS]

    new_features = np.concatenate(
        [
            svr_feats[EULER64_IDS],
            sketch_feats[EULER64_IDS],
        ],
        axis=0,
    )
    np.save(new_dir / "img_feature_dinov2.npy", new_features)


def migrate_one_model(model_id: str, dry_run: bool = False, overwrite_cache: bool = True) -> str:
    old_dir = OLD_COND / model_id
    new_dir = NEW_COND / model_id

    if not old_dir.is_dir():
        return "skip_no_dir"

    if not (old_dir / "imgs.npz").is_file():
        return "skip_missing_imgs"

    if not has_complete_legacy_cache(model_id):
        return "skip_incomplete_cache"

    if dry_run:
        return "ok"

    new_dir.mkdir(parents=True, exist_ok=True)
    for legacy_name in ("single_view.npz", "natural.npz", "sketch_and_natural.npz"):
        legacy_path = new_dir / legacy_name
        if legacy_path.exists():
            legacy_path.unlink()

    migrate_imgs_npz(old_dir, new_dir)
    migrate_dino_features(old_dir, new_dir)

    single_view_path = old_dir / "single_view.npz"
    if single_view_path.is_file():
        shutil.copy2(single_view_path, new_dir / "real_photo.npz")

    pc_path = old_dir / "pc.ply"
    if pc_path.is_file():
        shutil.copy2(pc_path, new_dir / "pc.ply")

    text_path = old_dir / "text.txt"
    text_feat_path = old_dir / "text_feat.npy"
    if text_path.is_file():
        description = text_path.read_text(encoding="utf-8").strip()
        feature = (
            np.load(text_feat_path)
            if text_feat_path.is_file()
            else np.zeros((4, 1024), dtype=np.float32)
        )
        np.savez_compressed(new_dir / "text.npz", description=np.array(description), feature=feature)

    for identity_id, euler64_id in enumerate(IDENTITY24_TO_EULER64):
        src_dir = AE_CACHE_OLD / f"{model_id}_{euler64_id}"
        dst_dir = AE_CACHE_NEW / f"{model_id}_{identity_id}"
        copy_cache_dir(src_dir, dst_dir, overwrite_cache=overwrite_cache)

    write_rotation_meta(new_dir)
    return "ok"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-models", type=int, default=None)
    parser.add_argument("--num-cpus", type=int, default=32, help="Ray parallel CPU count")
    parser.add_argument("--no-ray", action="store_true", help="Run serially")
    parser.add_argument("--model-list", type=Path, default=None, help="Explicit model id list")
    parser.add_argument("--train-only", action="store_true", help="Use only the legacy training list")
    parser.add_argument(
        "--skip-existing-cache",
        action="store_true",
        help="Do not overwrite files in existing migrated AE cache folders",
    )
    parser.add_argument(
        "--overwrite-cache",
        action="store_true",
        help="Compatibility option; cache overwriting is already the default",
    )
    args = parser.parse_args()

    model_ids = load_model_ids(args.model_list, train_only=args.train_only)
    if args.max_models is not None:
        model_ids = model_ids[: args.max_models]

    overwrite_cache = args.overwrite_cache or not args.skip_existing_cache

    list_label = args.model_list if args.model_list else ("train only" if args.train_only else "train+val+test")
    print(f"Models: {len(model_ids)}")
    print(f"Dry run: {args.dry_run}")
    print(f"Model list: {list_label}")
    print(f"Output cond: {NEW_COND}")
    print(f"Output cache: {AE_CACHE_NEW}")
    print("Basis: identity_first_cube24")
    print(f"identity24 -> euler64: {list(IDENTITY24_TO_EULER64)}")
    print(f"identity24 -> raw Blender24: {list(IDENTITY24_TO_BLENDER24)}")
    print("real_photo.npz['flux'] / single-view FLUX id: identity24 0")
    print(f"Overwrite cache: {overwrite_cache}")
    print()

    stats = {
        "ok": 0,
        "skip_no_dir": 0,
        "skip_missing_imgs": 0,
        "skip_incomplete_cache": 0,
    }
    incomplete_models = []

    if args.no_ray or args.dry_run:
        for index, model_id in enumerate(model_ids, start=1):
            result = migrate_one_model(model_id, dry_run=args.dry_run, overwrite_cache=overwrite_cache)
            stats[result] = stats.get(result, 0) + 1
            if result == "skip_incomplete_cache":
                incomplete_models.append(model_id)
            if index % 1000 == 0:
                print(f"  Progress: {index}/{len(model_ids)} | {stats}")
    else:
        import ray

        ray.init(num_cpus=args.num_cpus)

        @ray.remote
        def migrate_batch(batch_ids: list[str]) -> list[tuple[str, str]]:
            results = []
            for batch_model_id in batch_ids:
                results.append(
                    (
                        batch_model_id,
                        migrate_one_model(
                            batch_model_id,
                            dry_run=False,
                            overwrite_cache=overwrite_cache,
                        ),
                    )
                )
            return results

        batch_size = 50
        futures = []
        for start in range(0, len(model_ids), batch_size):
            futures.append(migrate_batch.remote(model_ids[start : start + batch_size]))

        for index, future in enumerate(futures, start=1):
            for model_id, result in ray.get(future):
                stats[result] = stats.get(result, 0) + 1
                if result == "skip_incomplete_cache":
                    incomplete_models.append(model_id)
            if index % 20 == 0:
                done = min(index * batch_size, len(model_ids))
                print(f"  Progress: ~{done}/{len(model_ids)} | {stats}")

        ray.shutdown()

    print(f"\nDone: {stats}")
    if incomplete_models:
        print(f"\nModels with incomplete ae_cache ({len(incomplete_models)}):")
        for model_id in incomplete_models[:20]:
            print(f"  {model_id}")
        if len(incomplete_models) > 20:
            print(f"  ... and {len(incomplete_models) - 20} more")


if __name__ == "__main__":
    main()
