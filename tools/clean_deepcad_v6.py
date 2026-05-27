"""复制 deepcad_v6 中有用的 GT 数据到新目录 (不修改原数据)。

从 /mnt/d/data/deepcad_v6/{model_id}/ 中只复制:
  - normalized_shape.step  (GT CAD 模型)
  - mesh.ply               (三角化网格)
  - data.npz               (BRep 控制点+拓扑, 但去掉里面的 imgs key)

不复制:
  - pc.ply                 (点云, 可从 mesh.ply 按需生成)
  - post_processed_shape.step (验证中间产物)
  - img_feature_dinov2.npy    (旧缓存, 已迁移到 deepcad_v7_cond)

输出到: /mnt/d/data/deepcad_v7/{model_id}/

用法:
    python tools/clean_deepcad_v6.py --dry-run
    python tools/clean_deepcad_v6.py
    python tools/clean_deepcad_v6.py --max-models 10
"""
import argparse
import shutil
import numpy as np
from pathlib import Path

OLD_DATA = Path("/mnt/d/data/deepcad_v6")
NEW_DATA = Path("/mnt/d/data/deepcad_v7")

# data.npz 中要保留的 key (去掉 imgs)
KEEP_NPZ_KEYS = [
    "sample_points_lines",
    "sample_points_faces",
    "edge_face_connectivity",
    "face_adj",
    "zero_positions",
]


def load_all_model_ids():
    lists = [
        Path("src/brepnet/data/list/deduplicated_deepcad_training_7_30.txt"),
        Path("src/brepnet/data/list/deduplicated_deepcad_validation_7_30.txt"),
        Path("src/brepnet/data/list/deduplicated_deepcad_testing_7_30.txt"),
    ]
    ids = set()
    for lst in lists:
        if lst.is_file():
            for line in lst.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    ids.add(line)
    return sorted(ids)


def migrate_one_model(model_id, dry_run=False):
    old_dir = OLD_DATA / model_id
    new_dir = NEW_DATA / model_id

    if not old_dir.is_dir():
        return "skip_no_dir"
    if new_dir.is_dir():
        return "skip_exists"

    if not dry_run:
        new_dir.mkdir(parents=True, exist_ok=True)

    # 复制 normalized_shape.step
    step_path = old_dir / "normalized_shape.step"
    if step_path.is_file() and not dry_run:
        shutil.copy2(step_path, new_dir / "normalized_shape.step")

    # 复制 mesh.ply
    mesh_path = old_dir / "mesh.ply"
    if mesh_path.is_file() and not dry_run:
        shutil.copy2(mesh_path, new_dir / "mesh.ply")

    # 复制 data.npz (只保留有用的 keys, 去掉 imgs)
    npz_path = old_dir / "data.npz"
    if npz_path.is_file() and not dry_run:
        old_data = np.load(npz_path, allow_pickle=True)
        new_data = {k: old_data[k] for k in KEEP_NPZ_KEYS if k in old_data}
        np.savez_compressed(str(new_dir / "data.npz"), **new_data)

    return "ok"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-models", type=int, default=None)
    parser.add_argument("--num-cpus", type=int, default=32, help="Ray 并行 CPU 数")
    parser.add_argument("--no-ray", action="store_true", help="禁用 Ray, 串行执行")
    args = parser.parse_args()

    model_ids = load_all_model_ids()
    if args.max_models:
        model_ids = model_ids[:args.max_models]

    print(f"Models: {len(model_ids)}")
    print(f"Dry run: {args.dry_run}")
    print(f"Input:  {OLD_DATA}")
    print(f"Output: {NEW_DATA}")
    print(f"data.npz keeps: {KEEP_NPZ_KEYS}")
    print()

    if args.no_ray or args.dry_run:
        stats = {"ok": 0, "skip_no_dir": 0, "skip_exists": 0}
        for i, model_id in enumerate(model_ids):
            result = migrate_one_model(model_id, args.dry_run)
            stats[result] = stats.get(result, 0) + 1
            if (i + 1) % 1000 == 0:
                print(f"  Progress: {i+1}/{len(model_ids)} | {stats}")
    else:
        import ray
        ray.init(num_cpus=args.num_cpus)

        @ray.remote
        def clean_batch(batch_ids):
            results = []
            for model_id in batch_ids:
                results.append(migrate_one_model(model_id, dry_run=False))
            return results

        batch_size = 50
        futures = []
        for i in range(0, len(model_ids), batch_size):
            futures.append(clean_batch.remote(model_ids[i:i + batch_size]))

        stats = {"ok": 0, "skip_no_dir": 0, "skip_exists": 0}
        for i, future in enumerate(futures):
            for result in ray.get(future):
                stats[result] = stats.get(result, 0) + 1
            if (i + 1) % 20 == 0:
                print(f"  Progress: ~{(i+1)*batch_size}/{len(model_ids)} | {stats}")

        ray.shutdown()

    print(f"\nDone: {stats}")


if __name__ == "__main__":
    main()
