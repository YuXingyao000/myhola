"""Migrate legacy euler64 condition/cache data to the Blender cube24 basis.

Input:
    /mnt/d/data/deepcad_v6_cond/{model_id}/imgs.npz
        keys: svr_imgs[64], sketch_imgs[64], mvr_imgs[512]
    /mnt/d/data/deepcad_v6_cond/{model_id}/single_view.npz
        keys: blender, flux, flux_masked
    /mnt/d/data/ae_cache/1119_deepcad_aug1_11k/{model_id}_{euler64_id}/features.npy

Output:
    /mnt/d/data/deepcad_v7_cond/{model_id}/svr.npz
        key: images[24, 224, 224, 3], Blender cube24 order
    /mnt/d/data/deepcad_v7_cond/{model_id}/sketch.npz
        key: images[24, 224, 224, 3], Blender cube24 order
    /mnt/d/data/deepcad_v7_cond/{model_id}/mvr.npz
        key: images[192, 224, 224, 3], [8 multi-view cameras, 24 Blender rotations]
    /mnt/d/data/deepcad_v7_cond/{model_id}/real_photo.npz
        copied from single_view.npz
    /mnt/d/data/deepcad_v7_cond/{model_id}/rotation_meta.json
    /mnt/d/data/ae_cache/1119_deepcad_aug1_11k_24/{model_id}_{blender24_id}/features.npy

Rotation contract:
    - Blender render_cube24.py 00.png..23.png is the canonical basis.
    - The two checked blender_cube24_sample models match OCC cube24 with identity mapping.
    - Legacy OCC/SVR and AE cache use euler64 ids, so this script indexes them
      with BLENDER24_TO_EULER64.
    - single_view.npz is the identity view: Blender/OCC cube24 id 18, not id 0.

Usage:
    python tools/migrate_64_to_24.py --dry-run
    python tools/migrate_64_to_24.py --no-ray
    python tools/migrate_64_to_24.py --max-models 10
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

NUM_BLENDER24_VIEWS = 24
BLENDER24_TO_OCC24 = tuple(range(NUM_BLENDER24_VIEWS))
SINGLE_VIEW_BLENDER_ID = 18


def build_blender24_to_euler64() -> tuple[int, ...]:
    """Map Blender cube24 ids to the first equivalent legacy euler64 id."""

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

    mapping = []
    for blender_matrix in blender_mats:
        for euler_id, euler_matrix in enumerate(euler_mats):
            if np.allclose(blender_matrix, euler_matrix, atol=1e-6):
                mapping.append(euler_id)
                break
        else:
            raise RuntimeError("Blender cube24 rotation has no euler64 counterpart")
    return tuple(mapping)


BLENDER24_TO_EULER64 = build_blender24_to_euler64()
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
        "notes": (
            "Blender cube24 00.png..23.png is canonical. single_view.npz is the "
            "identity view, which is Blender/OCC cube24 id 18."
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
    ids = []
    for path in lists:
        ids.extend(load_ids_from_list(path))
    return list(dict.fromkeys(ids))


def has_complete_legacy_cache(model_id: str) -> bool:
    for blender_id, euler64_id in enumerate(BLENDER24_TO_EULER64):
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


def migrate_one_model(model_id: str, dry_run: bool = False, overwrite_cache: bool = False) -> str:
    old_dir = OLD_COND / model_id
    new_dir = NEW_COND / model_id

    if not old_dir.is_dir():
        return "skip_no_dir"

    imgs_path = old_dir / "imgs.npz"
    if not imgs_path.is_file():
        return "skip_missing_imgs"

    if not has_complete_legacy_cache(model_id):
        return "skip_incomplete_cache"

    if dry_run:
        return "ok"

    new_dir.mkdir(parents=True, exist_ok=True)

    old = np.load(imgs_path)
    if "svr_imgs" in old:
        np.savez_compressed(new_dir / "svr.npz", images=old["svr_imgs"][EULER64_IDS])
    if "sketch_imgs" in old:
        np.savez_compressed(new_dir / "sketch.npz", images=old["sketch_imgs"][EULER64_IDS])
    if "mvr_imgs" in old:
        mvr = old["mvr_imgs"].reshape(8, 64, 224, 224, 3)
        mvr = mvr[:, EULER64_IDS].reshape(8 * NUM_BLENDER24_VIEWS, 224, 224, 3)
        np.savez_compressed(new_dir / "mvr.npz", images=mvr)

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

    for blender_id, euler64_id in enumerate(BLENDER24_TO_EULER64):
        src_dir = AE_CACHE_OLD / f"{model_id}_{euler64_id}"
        dst_dir = AE_CACHE_NEW / f"{model_id}_{blender_id}"
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
        "--overwrite-cache",
        action="store_true",
        help="Overwrite files in existing migrated AE cache folders",
    )
    args = parser.parse_args()

    model_ids = load_model_ids(args.model_list, train_only=args.train_only)
    if args.max_models is not None:
        model_ids = model_ids[: args.max_models]

    list_label = args.model_list if args.model_list else ("train only" if args.train_only else "train+val+test")
    print(f"Models: {len(model_ids)}")
    print(f"Dry run: {args.dry_run}")
    print(f"Model list: {list_label}")
    print(f"Output cond: {NEW_COND}")
    print(f"Output cache: {AE_CACHE_NEW}")
    print("Basis: Blender cube24")
    print("Blender24 -> OCC24: identity")
    print(f"Blender24 -> euler64: {list(BLENDER24_TO_EULER64)}")
    print(
        "single_view blender/flux id: "
        f"{SINGLE_VIEW_BLENDER_ID} (euler64={BLENDER24_TO_EULER64[SINGLE_VIEW_BLENDER_ID]})"
    )
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
            result = migrate_one_model(model_id, dry_run=args.dry_run, overwrite_cache=args.overwrite_cache)
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
                            overwrite_cache=args.overwrite_cache,
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
