"""一次性脚本：将旧数据迁移到 deepcad_cond_v2 (24-rotation, 清晰命名)

输入:
    /mnt/d/data/deepcad_v6_cond/{model_id}/imgs.npz          → svr_imgs[64], sketch_imgs[64], mvr_imgs[512]
    /mnt/d/data/deepcad_v6_cond/{model_id}/single_view.npz   → blender, flux, flux_masked
    /mnt/d/data/deepcad_v6_cond/{model_id}/pc.ply
    /mnt/d/data/deepcad_v6_cond/{model_id}/text.txt
    /mnt/d/data/deepcad_v6_cond/{model_id}/text_feat.npy
    /mnt/d/data/ae_cache/1119_deepcad_aug1_11k/{model_id}_{euler64_id}/features.npy

输出:
    /mnt/d/data/deepcad_cond_v2/{model_id}/svr.npz           → images[24, 224, 224, 3]
    /mnt/d/data/deepcad_cond_v2/{model_id}/sketch.npz        → images[24, 224, 224, 3]
    /mnt/d/data/deepcad_cond_v2/{model_id}/mvr.npz           → images[192, 224, 224, 3]
    /mnt/d/data/deepcad_cond_v2/{model_id}/real_photo.npz    → flux, blender, flux_masked
    /mnt/d/data/deepcad_cond_v2/{model_id}/pc.ply
    /mnt/d/data/deepcad_cond_v2/{model_id}/text.npz          → description(str), feature(4,1024)
    /mnt/d/data/ae_cache/1119_deepcad_aug1_11k_24/{model_id}_{cube24_id}/features.npy

用法:
    python tools/migrate_64_to_24.py --dry-run
    python tools/migrate_64_to_24.py
    python tools/migrate_64_to_24.py --max-models 10
"""
import os
import shutil
import argparse
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation

# ═══════════════════════════════════════════════════════════════
# 路径 (写死，一次性脚本)
# ═══════════════════════════════════════════════════════════════
OLD_COND = Path("/mnt/d/data/deepcad_v6_cond")
NEW_COND = Path("/mnt/d/data/deepcad_cond_v2")
AE_CACHE_OLD = Path("/mnt/d/data/ae_cache/1119_deepcad_aug1_11k")
AE_CACHE_NEW = Path("/mnt/d/data/ae_cache/1119_deepcad_aug1_11k_24")
MODEL_LIST = Path("src/brepnet/data/list/deduplicated_deepcad_training_7_30.txt")


# ═══════════════════════════════════════════════════════════════
# 构建 cube24 → euler64 映射
# ═══════════════════════════════════════════════════════════════

def build_cube24_to_euler64():
    dirs = [
        np.array([1., 0., 0.]), np.array([-1., 0., 0.]),
        np.array([0., 1., 0.]), np.array([0., -1., 0.]),
        np.array([0., 0., 1.]), np.array([0., 0., -1.]),
    ]
    cube_mats = []
    seen = set()
    for z in dirs:
        for y in dirs:
            if abs(float(np.dot(z, y))) > 1e-6:
                continue
            x = np.cross(y, z)
            mat = np.stack([x, y, z], axis=1)
            key = tuple(int(round(v)) for v in mat.flatten())
            if key in seen:
                continue
            seen.add(key)
            cube_mats.append(mat)
    assert len(cube_mats) == 24

    euler_mats = []
    for idx in range(64):
        angles = np.array([idx % 4, idx // 4 % 4, idx // 16]) * np.pi / 2
        euler_mats.append(Rotation.from_euler('xyz', angles).as_matrix())

    mapping = []
    for cm in cube_mats:
        for j, em in enumerate(euler_mats):
            if np.allclose(cm, em, atol=1e-6):
                mapping.append(j)
                break
        else:
            raise RuntimeError("No match")
    return mapping


CUBE24_TO_EULER64 = build_cube24_to_euler64()
EULER64_IDS = np.array(CUBE24_TO_EULER64)


# ═══════════════════════════════════════════════════════════════
# 迁移逻辑
# ═══════════════════════════════════════════════════════════════

def load_model_ids():
    ids = []
    for line in MODEL_LIST.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            ids.append(line)
    return ids


def migrate_one_model(model_id, dry_run=False):
    old_dir = OLD_COND / model_id
    new_dir = NEW_COND / model_id

    if not old_dir.is_dir():
        return "skip_no_dir"

    if not dry_run:
        new_dir.mkdir(parents=True, exist_ok=True)

    # --- svr.npz + sketch.npz + mvr.npz (从 imgs.npz 拆出, 64→24) ---
    imgs_path = old_dir / "imgs.npz"
    if imgs_path.is_file():
        if not dry_run:
            old = np.load(imgs_path)
            if "svr_imgs" in old:
                np.savez_compressed(str(new_dir / "svr.npz"),
                                    images=old["svr_imgs"][EULER64_IDS])
            if "sketch_imgs" in old:
                np.savez_compressed(str(new_dir / "sketch.npz"),
                                    images=old["sketch_imgs"][EULER64_IDS])
            if "mvr_imgs" in old:
                mvr = old["mvr_imgs"].reshape(8, 64, 224, 224, 3)
                mvr = mvr[:, EULER64_IDS].reshape(192, 224, 224, 3)
                np.savez_compressed(str(new_dir / "mvr.npz"), images=mvr)

    # --- real_photo.npz (从 single_view.npz 复制改名) ---
    sv_path = old_dir / "single_view.npz"
    if sv_path.is_file():
        if not dry_run:
            shutil.copy2(sv_path, new_dir / "real_photo.npz")

    # --- pc.ply (点云条件, 直接复制) ---
    pc_path = old_dir / "pc.ply"
    if pc_path.is_file():
        if not dry_run:
            shutil.copy2(pc_path, new_dir / "pc.ply")

    # --- text.npz (合并 text.txt + text_feat.npy) ---
    txt_path = old_dir / "text.txt"
    feat_path = old_dir / "text_feat.npy"
    if txt_path.is_file():
        if not dry_run:
            desc = txt_path.read_text(encoding="utf-8").strip()
            feat = np.load(feat_path) if feat_path.is_file() else np.zeros((4, 1024), dtype=np.float32)
            np.savez_compressed(str(new_dir / "text.npz"),
                                description=np.array(desc),
                                feature=feat)

    # --- ae_cache: 先检查 24 个旋转是否都有 features.npy ---
    missing_cache = False
    for cube24_id in range(24):
        euler64_id = CUBE24_TO_EULER64[cube24_id]
        feat_path = AE_CACHE_OLD / f"{model_id}_{euler64_id}" / "features.npy"
        if not feat_path.is_file():
            missing_cache = True
            break

    if missing_cache:
        # 这个模型 ae_cache 不完整，整个跳过（不迁移 cond 也不迁移 cache）
        # 清理已创建的 new_dir
        if not dry_run and new_dir.is_dir():
            shutil.rmtree(new_dir)
        return "skip_incomplete_cache"

    # ae_cache 完整，复制整个文件夹 + 重命名
    for cube24_id in range(24):
        euler64_id = CUBE24_TO_EULER64[cube24_id]
        src_dir = AE_CACHE_OLD / f"{model_id}_{euler64_id}"
        dst_dir = AE_CACHE_NEW / f"{model_id}_{cube24_id}"
        if not dst_dir.is_dir():
            if not dry_run:
                dst_dir.mkdir(parents=True, exist_ok=True)
                for f in src_dir.iterdir():
                    shutil.copy2(f, dst_dir / f.name)

    return "ok"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-models", type=int, default=None)
    parser.add_argument("--num-cpus", type=int, default=32, help="Ray 并行 CPU 数")
    parser.add_argument("--no-ray", action="store_true", help="禁用 Ray, 串行执行")
    args = parser.parse_args()

    model_ids = load_model_ids()
    if args.max_models:
        model_ids = model_ids[:args.max_models]

    print(f"Models: {len(model_ids)}")
    print(f"Dry run: {args.dry_run}")
    print(f"Output cond: {NEW_COND}")
    print(f"Output cache: {AE_CACHE_NEW}")
    print(f"Mapping: cube24_id=0 → euler64_id={CUBE24_TO_EULER64[0]}")
    print()

    if args.no_ray or args.dry_run:
        # 串行执行
        stats = {"ok": 0, "skip_no_dir": 0, "skip_incomplete_cache": 0}
        incomplete_models = []
        for i, model_id in enumerate(model_ids):
            result = migrate_one_model(model_id, args.dry_run)
            stats[result] = stats.get(result, 0) + 1
            if result == "skip_incomplete_cache":
                incomplete_models.append(model_id)
            if (i + 1) % 1000 == 0:
                print(f"  Progress: {i+1}/{len(model_ids)} | {stats}")
    else:
        # Ray 并行
        import ray
        ray.init(num_cpus=args.num_cpus)

        @ray.remote
        def migrate_batch(batch_ids):
            results = []
            for model_id in batch_ids:
                results.append((model_id, migrate_one_model(model_id, dry_run=False)))
            return results

        batch_size = 50
        futures = []
        for i in range(0, len(model_ids), batch_size):
            batch = model_ids[i:i + batch_size]
            futures.append(migrate_batch.remote(batch))

        stats = {"ok": 0, "skip_no_dir": 0, "skip_incomplete_cache": 0}
        incomplete_models = []
        for i, future in enumerate(futures):
            batch_results = ray.get(future)
            for model_id, result in batch_results:
                stats[result] = stats.get(result, 0) + 1
                if result == "skip_incomplete_cache":
                    incomplete_models.append(model_id)
            if (i + 1) % 20 == 0:
                done = (i + 1) * batch_size
                print(f"  Progress: ~{done}/{len(model_ids)} | {stats}")

        ray.shutdown()

    print(f"\nDone: {stats}")
    if incomplete_models:
        print(f"\nModels with incomplete ae_cache ({len(incomplete_models)}):")
        for mid in incomplete_models[:20]:
            print(f"  {mid}")
        if len(incomplete_models) > 20:
            print(f"  ... and {len(incomplete_models) - 20} more")

    print(f"\nDone: {stats}")
    if incomplete_models:
        print(f"\nModels with incomplete ae_cache ({len(incomplete_models)}):")
        for mid in incomplete_models[:20]:
            print(f"  {mid}")
        if len(incomplete_models) > 20:
            print(f"  ... and {len(incomplete_models) - 20} more")


if __name__ == "__main__":
    main()
