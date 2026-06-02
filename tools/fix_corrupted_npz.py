"""Scan deepcad_v7_cond for corrupted npz files and re-migrate only those.

Usage:
    # 先扫描，列出损坏文件
    cd /mnt/d/python && python tools/fix_corrupted_npz.py --scan-only

    # 修复（从 v6_cond 重新迁移损坏的 imgs.npz）
    cd /mnt/d/python && python tools/fix_corrupted_npz.py --fix
"""

from __future__ import annotations

import argparse
import zipfile
import zlib
from pathlib import Path

import numpy as np

NEW_COND = Path("/mnt/d/data/deepcad_v7_cond")
OLD_COND = Path("/mnt/d/data/deepcad_v6_cond")

# 需要检查的 npz 文件列表
CHECK_FILES = ["imgs.npz", "svr.npz", "real_photo.npz"]


def is_npz_valid(path: Path) -> bool:
    """Try to open and read all arrays in a npz file."""
    try:
        with np.load(path) as data:
            for key in data.files:
                _ = data[key]  # force decompression
        return True
    except (OSError, EOFError, KeyError, zipfile.BadZipFile, zlib.error, ValueError):
        return False


def scan(cond_root: Path) -> list[tuple[str, str]]:
    """Return list of (model_id, filename) for corrupted files."""
    corrupted = []
    model_dirs = sorted(p for p in cond_root.iterdir() if p.is_dir())
    total = len(model_dirs)
    for i, model_dir in enumerate(model_dirs):
        if (i + 1) % 5000 == 0:
            print(f"  Scanned {i+1}/{total}, found {len(corrupted)} corrupted so far")
        for filename in CHECK_FILES:
            fpath = model_dir / filename
            if fpath.is_file() and not is_npz_valid(fpath):
                corrupted.append((model_dir.name, filename))
    return corrupted


def fix_imgs_npz(model_id: str):
    """Re-migrate imgs.npz for one model from v6_cond."""
    from tools.migrate_64_to_24 import EULER64_IDS, NUM_IDENTITY24_VIEWS

    old_path = OLD_COND / model_id / "imgs.npz"
    new_dir = NEW_COND / model_id
    if not old_path.is_file():
        print(f"  {model_id}: source imgs.npz not found, skip")
        return False

    old = np.load(old_path)
    new_data = {}
    if "svr_imgs" in old:
        new_data["svr_imgs"] = old["svr_imgs"][EULER64_IDS]
    if "sketch_imgs" in old:
        new_data["sketch_imgs"] = old["sketch_imgs"][EULER64_IDS]
    if new_data:
        new_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(new_dir / "imgs.npz", **new_data)
        return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan-only", action="store_true", help="Only scan, don't fix")
    parser.add_argument("--fix", action="store_true", help="Scan and re-migrate corrupted files")
    parser.add_argument("--cond-root", type=Path, default=NEW_COND)
    args = parser.parse_args()

    if not args.scan_only and not args.fix:
        args.scan_only = True

    print(f"Scanning: {args.cond_root}")
    corrupted = scan(args.cond_root)
    print(f"\nFound {len(corrupted)} corrupted files:")
    for model_id, filename in corrupted[:50]:
        print(f"  {model_id}/{filename}")
    if len(corrupted) > 50:
        print(f"  ... and {len(corrupted) - 50} more")

    if args.scan_only:
        # Save list for reference
        out = Path("experiments/2026-06-01/corrupted_files.txt")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(f"{mid}/{fn}" for mid, fn in corrupted) + "\n")
        print(f"Saved list to {out}")
        return

    # Fix
    fixed = 0
    for model_id, filename in corrupted:
        if filename in ("imgs.npz", "svr.npz"):
            if fix_imgs_npz(model_id):
                fixed += 1
                print(f"  Fixed: {model_id}/{filename}")
        else:
            print(f"  Skip: {model_id}/{filename} (no fix logic for this file type)")

    print(f"\nDone: fixed {fixed}/{len(corrupted)}")


if __name__ == "__main__":
    main()
