"""Debug 脚本：可视化验证 64→24 数据迁移的正确性。

对一个 model_id:
1. 输出迁移后的 24 张 svr 图片 (cube24_images/)
2. 输出每个 cube24 对应的所有 euler64 重复图 (euler64_mapping/)
3. 输出迁移后的数据样例 (result_sample/) — 验证最终 npz 内容正确

用法:
    python tools/debug_rotation_migration.py --model-id 00261287
"""
import argparse
import shutil
import numpy as np
from pathlib import Path
from PIL import Image
from scipy.spatial.transform import Rotation

# ═══════════════════════════════════════════════════════════════
# 路径 (写死)
# ═══════════════════════════════════════════════════════════════
OLD_COND = Path("/mnt/d/data/deepcad_v6_cond")
NEW_COND = Path("/mnt/d/data/deepcad_cond_v2")
DATA_ROOT = Path("/mnt/d/data/deepcad_v6")
AE_CACHE_OLD = Path("/mnt/d/data/ae_cache/1119_deepcad_aug1_11k")
AE_CACHE_NEW = Path("/mnt/d/data/ae_cache/1119_deepcad_aug1_11k_24")
OUTPUT_ROOT = Path("debug_output")


# ═══════════════════════════════════════════════════════════════
# 映射 (和 migrate 一样)
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

    # cube24 → euler64 (one-to-one, 24 entries)
    c2e = []
    for cm in cube_mats:
        for j, em in enumerate(euler_mats):
            if np.allclose(cm, em, atol=1e-6):
                c2e.append(j)
                break

    # euler64 → cube24 (many-to-one, 64 entries, None for non-unique)
    e2c = [None] * 64
    for eid in range(64):
        for cid, cm in enumerate(cube_mats):
            if np.allclose(euler_mats[eid], cm, atol=1e-6):
                e2c[eid] = cid
                break

    # cube24 → all euler64 ids (for showing duplicates)
    c2e_all = {i: [] for i in range(24)}
    for eid in range(64):
        cid = e2c[eid]
        if cid is not None:
            c2e_all[cid].append(eid)

    return c2e, e2c, c2e_all


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    model_id = args.model_id
    output = args.output or OUTPUT_ROOT / model_id
    output.mkdir(parents=True, exist_ok=True)

    c2e, e2c, c2e_all = build_cube24_to_euler64()

    print(f"Model: {model_id}")
    print(f"Output: {output}")
    print()

    # --- 1. 复制 mesh ---
    mesh_src = DATA_ROOT / model_id / "mesh.stl"
    if mesh_src.is_file():
        shutil.copy2(mesh_src, output / "mesh.stl")
        print("✓ mesh.stl")
    else:
        print(f"✗ mesh.stl not found: {mesh_src}")

    # --- 2. 加载旧 imgs.npz ---
    imgs_path = OLD_COND / model_id / "imgs.npz"
    if not imgs_path.is_file():
        print(f"✗ imgs.npz not found: {imgs_path}")
        return
    data = np.load(imgs_path)
    svr_imgs = data["svr_imgs"]  # [64, 224, 224, 3]
    print(f"✓ imgs.npz loaded: svr_imgs {svr_imgs.shape}")

    # --- 3. cube24_images: 迁移后的 24 张图 ---
    cube24_dir = output / "cube24_images"
    cube24_dir.mkdir(exist_ok=True)
    for cid in range(24):
        eid = c2e[cid]
        Image.fromarray(svr_imgs[eid]).save(cube24_dir / f"{cid:02d}_from_euler{eid:02d}.png")
    print(f"✓ cube24_images/ (24 images)")

    # --- 4. euler64_mapping: 每个 cube24 对应哪些 euler64 重复图 ---
    mapping_dir = output / "euler64_mapping"
    mapping_dir.mkdir(exist_ok=True)
    for cid in range(24):
        cdir = mapping_dir / f"cube24_{cid:02d}"
        cdir.mkdir(exist_ok=True)
        for eid in c2e_all[cid]:
            Image.fromarray(svr_imgs[eid]).save(cdir / f"euler64_{eid:02d}.png")
    print(f"✓ euler64_mapping/ (duplicates per cube24)")

    # --- 5. result_sample: 迁移后的实际 npz 内容 ---
    result_dir = output / "result_sample"
    result_dir.mkdir(exist_ok=True)

    # svr.npz
    new_svr = svr_imgs[np.array(c2e)]  # [24, 224, 224, 3]
    np.savez_compressed(str(result_dir / "svr.npz"), images=new_svr)
    # 也存几张 png 方便直接看
    for i in [0, 1, 12, 23]:
        Image.fromarray(new_svr[i]).save(result_dir / f"svr_cube24_{i:02d}.png")

    # real_photo.npz
    sv_path = OLD_COND / model_id / "single_view.npz"
    if sv_path.is_file():
        sv_data = np.load(sv_path)
        shutil.copy2(sv_path, result_dir / "real_photo.npz")
        if "flux" in sv_data:
            Image.fromarray(sv_data["flux"]).save(result_dir / "flux.png")
        if "blender" in sv_data:
            Image.fromarray(sv_data["blender"]).save(result_dir / "blender.png")
        if "flux_masked" in sv_data:
            Image.fromarray(sv_data["flux_masked"]).save(result_dir / "flux_masked.png")
        print(f"✓ result_sample/real_photo.npz + pngs")

    # text.npz
    txt_path = OLD_COND / model_id / "text.txt"
    feat_path = OLD_COND / model_id / "text_feat.npy"
    if txt_path.is_file():
        desc = txt_path.read_text(encoding="utf-8").strip()
        feat = np.load(feat_path) if feat_path.is_file() else np.zeros((4, 1024))
        np.savez_compressed(str(result_dir / "text.npz"),
                            description=np.array(desc), feature=feat)
        (result_dir / "text_content.txt").write_text(desc)
        print(f"✓ result_sample/text.npz")

    # latent
    latent_dir = result_dir / "latent"
    latent_dir.mkdir(exist_ok=True)
    for cid in [0, 1, 12, 23]:
        eid = c2e[cid]
        src = AE_CACHE_OLD / f"{model_id}_{eid}" / "features.npy"
        if src.is_file():
            feat = np.load(src)
            np.savetxt(str(latent_dir / f"cube24_{cid:02d}_euler64_{eid:02d}.txt"),
                       feat[:3], fmt="%.6f", header=f"shape={feat.shape}")
    print(f"✓ result_sample/latent/ (first 3 rows of features)")

    # --- 6. mapping_table.txt ---
    table = "cube24_id | euler64_canonical | all_euler64_duplicates\n"
    table += "-" * 60 + "\n"
    for cid in range(24):
        table += f"  {cid:2d}      |       {c2e[cid]:2d}         | {c2e_all[cid]}\n"
    (output / "mapping_table.txt").write_text(table)
    print(f"✓ mapping_table.txt")

    print(f"\nDone! Check: {output}")


if __name__ == "__main__":
    main()
