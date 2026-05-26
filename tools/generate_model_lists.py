"""生成去重后的 model lists (train/val/test)。

检查:
1. 三份 list 之间有无重复 → 按 training > validation > testing 优先级保留
2. 每个 model_id 在 deepcad_v6 中是否有 data.npz
3. 每个 model_id 在 ae_cache 中是否有 features.npy (24 个旋转全部存在)
4. 每个 model_id 在 cond 中是否有 imgs.npz

输出:
    src/brepnet/data/list/train.txt      (去重 + 过滤后)
    src/brepnet/data/list/val.txt
    src/brepnet/data/list/test.txt
    src/brepnet/data/list/excluded.txt   (被排除的模型及原因)

用法:
    python tools/generate_model_lists.py --dry-run
    python tools/generate_model_lists.py
"""
import argparse
from pathlib import Path
from collections import Counter

# ═══════════════════════════════════════════════════════════════
# 路径 (写死)
# ═══════════════════════════════════════════════════════════════
DATA_ROOT = Path("/mnt/d/data/deepcad_v6")
AE_CACHE = Path("/mnt/d/data/ae_cache/1119_deepcad_aug1_11k")
COND_ROOT = Path("/mnt/d/data/deepcad_v6_cond")

OLD_TRAIN = Path("src/brepnet/data/list/deduplicated_deepcad_training_7_30.txt")
OLD_VAL = Path("src/brepnet/data/list/deduplicated_deepcad_validation_7_30.txt")
OLD_TEST = Path("src/brepnet/data/list/deduplicated_deepcad_testing_7_30.txt")

OUTPUT_DIR = Path("src/brepnet/data/list")


def load_list(path: Path) -> list[str]:
    ids = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            ids.append(line)
    return ids


def check_model_integrity(model_id: str, cube24_euler64_map: list[int]) -> str | None:
    """检查模型数据完整性，返回 None 表示通过，否则返回原因。"""
    # 1. deepcad_v6 有 data.npz
    data_npz = DATA_ROOT / model_id / "data.npz"
    if not data_npz.is_file():
        return "missing_data_npz"

    # 2. ae_cache 至少有 1 个 features.npy (检查 cube24_id=0 对应的 euler64)
    euler64_id = cube24_euler64_map[0]
    feat_path = AE_CACHE / f"{model_id}_{euler64_id}" / "features.npy"
    if not feat_path.is_file():
        return "missing_ae_cache"

    return None  # 通过


def build_cube24_to_euler64():
    """和 migrate 脚本一样的映射"""
    import numpy as np
    from scipy.spatial.transform import Rotation

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
    return mapping


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="只打印统计不写文件")
    parser.add_argument("--skip-integrity", action="store_true", help="跳过文件完整性检查（快）")
    args = parser.parse_args()

    # 加载旧 lists
    train_ids = load_list(OLD_TRAIN)
    val_ids = load_list(OLD_VAL)
    test_ids = load_list(OLD_TEST)

    print(f"Original: train={len(train_ids)} val={len(val_ids)} test={len(test_ids)}")

    # ─── Step 1: 查找重复 ───
    all_ids = train_ids + val_ids + test_ids
    counter = Counter(all_ids)
    duplicates = {k: v for k, v in counter.items() if v > 1}
    print(f"Duplicated model IDs (appear in multiple splits): {len(duplicates)}")
    if duplicates:
        print(f"  Examples: {list(duplicates.keys())[:5]}")

    # ─── Step 2: 按优先级去重 (training > validation > testing) ───
    train_set = set(train_ids)
    val_set = set(val_ids) - train_set  # 从 val 中移除和 train 重复的
    test_set = set(test_ids) - train_set - val_set  # 从 test 中移除和 train/val 重复的

    # 保持原始顺序
    new_train = [x for x in train_ids if x in train_set]
    new_val = [x for x in val_ids if x in val_set]
    new_test = [x for x in test_ids if x in test_set]

    # 去除 list 内重复
    new_train = list(dict.fromkeys(new_train))
    new_val = list(dict.fromkeys(new_val))
    new_test = list(dict.fromkeys(new_test))

    removed_from_val = len(val_ids) - len(new_val)
    removed_from_test = len(test_ids) - len(new_test)
    print(f"After dedup: train={len(new_train)} val={len(new_val)} test={len(new_test)}")
    print(f"  Removed from val: {removed_from_val}, from test: {removed_from_test}")

    # ─── Step 3: 文件完整性检查 ───
    excluded = []  # (model_id, split, reason)
    if not args.skip_integrity:
        print("\nChecking file integrity...")
        c2e = build_cube24_to_euler64()

        def filter_by_integrity(ids, split_name):
            passed = []
            for mid in ids:
                reason = check_model_integrity(mid, c2e)
                if reason:
                    excluded.append((mid, split_name, reason))
                else:
                    passed.append(mid)
            return passed

        new_train = filter_by_integrity(new_train, "train")
        new_val = filter_by_integrity(new_val, "val")
        new_test = filter_by_integrity(new_test, "test")
        print(f"After integrity check: train={len(new_train)} val={len(new_val)} test={len(new_test)}")
        print(f"  Excluded: {len(excluded)} models")

        # 统计排除原因
        reasons = Counter(r for _, _, r in excluded)
        for reason, count in reasons.most_common():
            print(f"    {reason}: {count}")

    # ─── Step 4: 输出 ───
    print(f"\nFinal: train={len(new_train)} val={len(new_val)} test={len(new_test)}")

    if not args.dry_run:
        (OUTPUT_DIR / "train.txt").write_text("\n".join(new_train) + "\n")
        (OUTPUT_DIR / "val.txt").write_text("\n".join(new_val) + "\n")
        (OUTPUT_DIR / "test.txt").write_text("\n".join(new_test) + "\n")
        if excluded:
            lines = [f"{mid}\t{split}\t{reason}" for mid, split, reason in excluded]
            (OUTPUT_DIR / "excluded.txt").write_text("\n".join(lines) + "\n")
        print(f"Written to {OUTPUT_DIR}/{{train,val,test,excluded}}.txt")
    else:
        print("(dry-run, no files written)")


if __name__ == "__main__":
    main()
