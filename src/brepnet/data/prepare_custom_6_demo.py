"""Prepare a minimal qualitative-inference dataset from 6 real photos.

For each input image this script:
    1. Center-crops to the shortest side and resizes to 512x512 RGB.
    2. Writes a preview PNG for sanity check.
    3. Creates the folder layout expected by `AutoEncoder_dataset3` +
       `prepare_condition` in `src/brepnet/dataset.py`, namely:

        <out_root>/data_root/<prefix>/data.npz       (placeholder copy)
        <out_root>/cond_root/<prefix>/imgs.npz       (key: svr_imgs)
        <out_root>/cond_root/<prefix>/natural.npz    (key: natural_imgs_compress)

       Only index 0 of each image array is actually used at inference time
       because `dataset.is_aug=0` pins `v_id_aug` to 0, so the arrays are
       stored with shape (1, 512, 512, 3) uint8.

    4. Writes `<out_root>/test_list.txt` with one prefix per line.

The placeholder `data.npz` only feeds the optional test-loss computation
inside `TrainDiffusion.test_step`; it does not enter the sampling path.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


def center_crop_square(img: Image.Image) -> Image.Image:
    w, h = img.size
    s = min(w, h)
    left = (w - s) // 2
    top = (h - s) // 2
    return img.crop((left, top, left + s, top + s))


def load_and_prepare(image_path: Path, out_size: int) -> np.ndarray:
    with Image.open(image_path) as im:
        im = im.convert("RGB")
        im = center_crop_square(im)
        im = im.resize((out_size, out_size), Image.BICUBIC)
        arr = np.asarray(im, dtype=np.uint8)
    if arr.shape != (out_size, out_size, 3):
        raise RuntimeError(f"Unexpected shape {arr.shape} for {image_path}")
    return arr


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input_dir",
        type=Path,
        default=Path("/mnt/d/data/gt_photo"),
        help="Folder with the 6 photos (png/jpg).",
    )
    p.add_argument(
        "--out_root",
        type=Path,
        default=Path("/mnt/d/data/my_6_demo"),
        help="Where to create data_root/ and cond_root/.",
    )
    p.add_argument(
        "--placeholder_data_npz",
        type=Path,
        default=Path("/mnt/d/data/deepcad_v6/00786107/data.npz"),
        help="Any valid data.npz from the original DeepCAD dataset. "
             "Used purely as a placeholder to satisfy dataset loading.",
    )
    p.add_argument("--size", type=int, default=512, help="Output image size.")
    p.add_argument(
        "--extensions",
        nargs="+",
        default=[".png", ".jpg", ".jpeg", ".webp", ".bmp"],
        help="Accepted image extensions (case-insensitive).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input_dir.is_dir():
        raise FileNotFoundError(f"Input folder not found: {args.input_dir}")
    if not args.placeholder_data_npz.is_file():
        raise FileNotFoundError(
            f"Placeholder data.npz not found: {args.placeholder_data_npz}"
        )

    allowed = {e.lower() for e in args.extensions}
    image_paths = sorted(
        p for p in args.input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in allowed
    )
    if not image_paths:
        raise RuntimeError(f"No images found in {args.input_dir}")

    print(f"Found {len(image_paths)} images in {args.input_dir}")

    data_root = args.out_root / "data_root"
    cond_root = args.out_root / "cond_root"
    preview_root = args.out_root / "preview"
    list_path = args.out_root / "test_list.txt"

    data_root.mkdir(parents=True, exist_ok=True)
    cond_root.mkdir(parents=True, exist_ok=True)
    preview_root.mkdir(parents=True, exist_ok=True)

    prefixes: list[str] = []
    for image_path in image_paths:
        prefix = image_path.stem
        prefixes.append(prefix)

        arr = load_and_prepare(image_path, args.size)
        stacked = arr[None, ...]

        sample_data_dir = data_root / prefix
        sample_data_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.placeholder_data_npz, sample_data_dir / "data.npz")

        sample_cond_dir = cond_root / prefix
        sample_cond_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(sample_cond_dir / "imgs.npz", svr_imgs=stacked)
        np.savez_compressed(
            sample_cond_dir / "natural.npz",
            natural_imgs_compress=stacked,
        )

        Image.fromarray(arr).save(preview_root / f"{prefix}.png")

        print(f"  [{prefix}] {image_path.name} -> {arr.shape}")

    list_path.write_text("\n".join(prefixes) + "\n")

    print()
    print(f"Done. Layout under {args.out_root}:")
    print(f"  data_root     : {data_root}")
    print(f"  cond_root     : {cond_root}")
    print(f"  preview       : {preview_root}")
    print(f"  test_list.txt : {list_path}")
    print()
    print("Use these in your inference command:")
    print(f"  dataset.data_root={data_root}")
    print(f"  dataset.cond_root={cond_root}")
    print(f"  dataset.test_dataset={list_path}")


if __name__ == "__main__":
    main()
