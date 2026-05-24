r"""
Pack rendered natural images into natural.npz files.

Single input (--render-root only):
    Saves natural_imgs / natural_imgs_compress from that tree (Blender or FLUX).

Dual input (--render-root + --flux-root):
    --render-root must be Blender output; --flux-root must be FLUX output.
    Saves:
      natural_imgs / natural_imgs_compress  -> from FLUX (same keys as flux-only packs)
      blender_natural_imgs / blender_natural_imgs_compress -> from Blender

Output layout:
    target-root/{model_id}/natural.npz

Usage:
    python pack_natural_npz.py \
        --render-root /path/to/output_single_view \
        --target-root /mnt/d/data/deepcad_v6_cond

    Single-material single-view Blender output ({model_id}/0/0.png only) -> 8-slot npz:

    python pack_natural_npz.py \
        --render-root /path/to/output_single_view_metal010 \
        --target-root /mnt/d/data/deepcad_v6_cond \
        --material8-from-single-slot

    python pack_natural_npz.py \
        --render-root /path/to/output_single_view \
        --flux-root /path/to/output_flux_single_view \
        --target-root /mnt/d/data/deepcad_v6_cond \
        --model-list /path/to/all.txt

Cube-24 dual pack (Blender + FLUX, flat 00.png..23.png under each model id):

    python pack_natural_npz.py --cube24 \
        --render-root /path/to/output_cube24_single_material \
        --flux-root /path/to/output_flux_cube24_dynamic \
        --target-root /path/to/deepcad_v6_cond_cube24 \
        --model-list /path/to/model_list.txt
"""

from __future__ import annotations

import argparse
import concurrent.futures
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image


MODEL_DIR_PATTERN = re.compile(r"^\d{8}$")
MATERIAL_COUNT = 8
VIEW_COUNT_CUBE24 = 24
FULL_SIZE = (1024, 1024)
COMPRESS_SIZE = (224, 224)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pack rendered PNGs into natural.npz (8 material views or 24 cube views)."
    )
    parser.add_argument(
        "--render-root",
        type=Path,
        required=True,
        help="Blender output root, or single-source root when --flux-root is omitted.",
    )
    parser.add_argument(
        "--flux-root",
        type=Path,
        default=None,
        help="Optional FLUX output root; when set, --render-root is treated as Blender.",
    )
    parser.add_argument(
        "--target-root",
        type=Path,
        required=True,
        help="Root directory like deepcad_v6_cond/{model_id}/natural.npz",
    )
    parser.add_argument(
        "--model-list",
        type=Path,
        default=None,
        help="Optional txt file listing model ids to pack.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of worker threads used to pack model folders.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing natural.npz files.",
    )
    parser.add_argument(
        "--cube24",
        action="store_true",
        help=(
            "Use 24-view cube layout: images at {model_id}/{view:02d}.png (00..23) "
            "instead of {model_id}/0/{0..7}.png."
        ),
    )
    parser.add_argument(
        "--material8-from-single-slot",
        action="store_true",
        help=(
            "For material8 layout only: expect only {model_id}/0/0.png and replicate it to 8 slots "
            "in natural.npz (matches single --material-subdir Blender renders)."
        ),
    )
    return parser.parse_args()


def load_model_ids_from_list(list_path: Path) -> list[str]:
    if not list_path.is_file():
        raise FileNotFoundError(f"Model list does not exist: {list_path}")

    model_ids = []
    seen_ids = set()
    for line_number, raw_line in enumerate(list_path.read_text(encoding="utf-8").splitlines(), start=1):
        model_id = raw_line.strip()
        if not model_id or model_id.startswith("#"):
            continue
        if not MODEL_DIR_PATTERN.match(model_id):
            raise ValueError(
                f"Invalid model id {model_id!r} in {list_path} line {line_number}; expected 8 digits."
            )
        if model_id in seen_ids:
            continue
        seen_ids.add(model_id)
        model_ids.append(model_id)
    return model_ids


def find_model_dirs(render_root: Path, model_list: Path | None = None) -> list[Path]:
    if not render_root.exists():
        raise FileNotFoundError(f"Render root does not exist: {render_root}")

    if model_list is not None:
        model_ids = load_model_ids_from_list(model_list)
        model_dirs = [render_root / model_id for model_id in model_ids]
        missing_dirs = [model_dir for model_dir in model_dirs if not model_dir.is_dir()]
        if missing_dirs:
            preview = ", ".join(str(path) for path in missing_dirs[:5])
            raise FileNotFoundError(
                f"Some model output folders listed in {model_list} do not exist under {render_root}: {preview}"
            )
        return model_dirs

    model_dirs = [
        path
        for path in sorted(render_root.iterdir())
        if path.is_dir() and MODEL_DIR_PATTERN.match(path.name)
    ]
    return model_dirs


def load_image_pair(image_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        if img.size != FULL_SIZE:
            raise ValueError(
                f"Expected image size {FULL_SIZE} for {image_path}, got {img.size}."
            )

        full_img = np.array(img, dtype=np.uint8)
        compress_img = np.array(
            img.resize(COMPRESS_SIZE, Image.BICUBIC),
            dtype=np.uint8,
        )

    if full_img.shape != (FULL_SIZE[1], FULL_SIZE[0], 3):
        raise ValueError(f"Unexpected full image shape for {image_path}: {full_img.shape}")
    if compress_img.shape != (COMPRESS_SIZE[1], COMPRESS_SIZE[0], 3):
        raise ValueError(
            f"Unexpected compressed image shape for {image_path}: {compress_img.shape}"
        )

    return full_img, compress_img


def load_material_stack(image_root: Path) -> tuple[np.ndarray, np.ndarray]:
    if not image_root.is_dir():
        raise FileNotFoundError(f"Missing view folder: {image_root}")
    full_images = []
    compress_images = []
    for material_idx in range(MATERIAL_COUNT):
        image_path = image_root / f"{material_idx}.png"
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing material image: {image_path}")
        full_img, compress_img = load_image_pair(image_path)
        full_images.append(full_img)
        compress_images.append(compress_img)

    natural_imgs = np.stack(full_images, axis=0)
    natural_imgs_compress = np.stack(compress_images, axis=0)

    if natural_imgs.shape != (MATERIAL_COUNT, FULL_SIZE[1], FULL_SIZE[0], 3):
        raise ValueError(f"Unexpected natural_imgs shape: {natural_imgs.shape}")
    if natural_imgs_compress.shape != (MATERIAL_COUNT, COMPRESS_SIZE[1], COMPRESS_SIZE[0], 3):
        raise ValueError(f"Unexpected natural_imgs_compress shape: {natural_imgs_compress.shape}")

    return natural_imgs, natural_imgs_compress


def load_zero_padded_view_stack(image_root: Path, view_count: int) -> tuple[np.ndarray, np.ndarray]:
    """Load view_count PNGs named 00.png .. (view_count-1) as two-digit zero-padded names."""
    if not image_root.is_dir():
        raise FileNotFoundError(f"Missing image folder: {image_root}")
    full_images = []
    compress_images = []
    for view_idx in range(view_count):
        image_path = image_root / f"{view_idx:02d}.png"
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing view image: {image_path}")
        full_img, compress_img = load_image_pair(image_path)
        full_images.append(full_img)
        compress_images.append(compress_img)

    natural_imgs = np.stack(full_images, axis=0)
    natural_imgs_compress = np.stack(compress_images, axis=0)

    if natural_imgs.shape != (view_count, FULL_SIZE[1], FULL_SIZE[0], 3):
        raise ValueError(f"Unexpected natural_imgs shape: {natural_imgs.shape}")
    if natural_imgs_compress.shape != (view_count, COMPRESS_SIZE[1], COMPRESS_SIZE[0], 3):
        raise ValueError(f"Unexpected natural_imgs_compress shape: {natural_imgs_compress.shape}")

    return natural_imgs, natural_imgs_compress


def load_material_stack_replicate_single(
    blender_view: Path, slot_count: int = MATERIAL_COUNT
) -> tuple[np.ndarray, np.ndarray]:
    """Load model_id/0/0.png only and stack the same image slot_count times (material8 npz layout)."""
    image_path = blender_view / "0.png"
    if not image_path.is_file():
        raise FileNotFoundError(
            f"Single-slot material render expected at {image_path} (use render_dataset_single_view.py "
            "with --material-subdir)."
        )
    full_img, compress_img = load_image_pair(image_path)
    full_stack = np.stack([full_img] * slot_count, axis=0)
    compress_stack = np.stack([compress_img] * slot_count, axis=0)
    if full_stack.shape != (slot_count, FULL_SIZE[1], FULL_SIZE[0], 3):
        raise ValueError(f"Unexpected natural_imgs shape: {full_stack.shape}")
    if compress_stack.shape != (slot_count, COMPRESS_SIZE[1], COMPRESS_SIZE[0], 3):
        raise ValueError(f"Unexpected natural_imgs_compress shape: {compress_stack.shape}")
    return full_stack, compress_stack


def process_model_dir(
    model_dir: Path,
    target_root: Path,
    overwrite: bool,
    flux_root: Path | None,
    cube24: bool,
    material8_from_single_slot: bool,
) -> str:
    model_id = model_dir.name
    output_dir = target_root / model_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "natural.npz"

    if output_path.exists() and not overwrite:
        return f"skip {model_id} (exists: {output_path})"

    if cube24:
        if flux_root is None:
            natural_imgs, natural_imgs_compress = load_zero_padded_view_stack(
                model_dir, VIEW_COUNT_CUBE24
            )
            np.savez_compressed(
                output_path,
                natural_imgs=natural_imgs,
                natural_imgs_compress=natural_imgs_compress,
            )
        else:
            flux_dir = flux_root / model_id
            blender_imgs, blender_compress = load_zero_padded_view_stack(
                model_dir, VIEW_COUNT_CUBE24
            )
            flux_imgs, flux_compress = load_zero_padded_view_stack(flux_dir, VIEW_COUNT_CUBE24)
            np.savez_compressed(
                output_path,
                natural_imgs=flux_imgs,
                natural_imgs_compress=flux_compress,
                blender_natural_imgs=blender_imgs,
                blender_natural_imgs_compress=blender_compress,
            )
    else:
        blender_view = model_dir / "0"
        if material8_from_single_slot:
            load_mat = load_material_stack_replicate_single
        else:
            load_mat = load_material_stack

        if flux_root is None:
            natural_imgs, natural_imgs_compress = load_mat(blender_view)
            np.savez_compressed(
                output_path,
                natural_imgs=natural_imgs,
                natural_imgs_compress=natural_imgs_compress,
            )
        else:
            flux_view = flux_root / model_id / "0"
            blender_imgs, blender_compress = load_mat(blender_view)
            flux_imgs, flux_compress = load_mat(flux_view)
            np.savez_compressed(
                output_path,
                natural_imgs=flux_imgs,
                natural_imgs_compress=flux_compress,
                blender_natural_imgs=blender_imgs,
                blender_natural_imgs_compress=blender_compress,
            )

    return f"saved {output_path}"


def main() -> None:
    args = parse_args()
    render_root = args.render_root.resolve()
    target_root = args.target_root.resolve()
    flux_root = args.flux_root.resolve() if args.flux_root is not None else None
    model_list = args.model_list.resolve() if args.model_list is not None else None

    if args.num_workers <= 0:
        raise ValueError("--num-workers must be a positive integer.")

    if flux_root is not None and not flux_root.is_dir():
        raise FileNotFoundError(f"--flux-root does not exist or is not a directory: {flux_root}")

    model_dirs = find_model_dirs(render_root, model_list=model_list)
    print(f"render_root={render_root}")
    if args.cube24 and args.material8_from_single_slot:
        raise ValueError("Use only one of --cube24 or --material8-from-single-slot.")
    if args.cube24:
        print(f"layout=cube24 ({VIEW_COUNT_CUBE24} views: 00.png..23.png under each model id)")
    elif args.material8_from_single_slot:
        print("layout=material8-from-single (only model_id/0/0.png, replicated x8 in npz)")
    else:
        print("layout=material8 (8 images under model_id/0/)")
    if flux_root is not None:
        print(f"flux_root={flux_root} (dual pack: natural_* from FLUX, blender_* from render-root)")
    print(f"target_root={target_root}")
    if model_list is not None:
        print(f"model_list={model_list}")
    print(f"model_count={len(model_dirs)}")
    print(f"num_workers={args.num_workers}")

    if args.cube24:
        for md in model_dirs:
            for label, root in (("Blender", md), ("FLUX", flux_root / md.name if flux_root else None)):
                if root is None:
                    continue
                for i in range(VIEW_COUNT_CUBE24):
                    p = root / f"{i:02d}.png"
                    if not p.is_file():
                        raise FileNotFoundError(f"Pre-check {label}: missing {p}")
    elif args.material8_from_single_slot:
        for md in model_dirs:
            for label, root in (("Blender", md / "0"), ("FLUX", flux_root / md.name / "0" if flux_root else None)):
                if root is None:
                    continue
                p = root / "0.png"
                if not p.is_file():
                    raise FileNotFoundError(f"Pre-check {label}: missing {p}")

    saved = 0
    skipped = 0
    failed = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        future_to_model = {
            executor.submit(
                process_model_dir,
                model_dir,
                target_root,
                args.overwrite,
                flux_root,
                args.cube24,
                args.material8_from_single_slot,
            ): model_dir.name
            for model_dir in model_dirs
        }
        for future in concurrent.futures.as_completed(future_to_model):
            model_id = future_to_model[future]
            try:
                message = future.result()
                print(message)
                if message.startswith("saved "):
                    saved += 1
                else:
                    skipped += 1
            except Exception as exc:
                failed += 1
                print(f"error {model_id}: {exc}")

    print(f"done saved={saved} skipped={skipped} failed={failed}")
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
