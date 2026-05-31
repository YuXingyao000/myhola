"""Prepare the real_photo folder for BRepNet single-image testing.

The current dataset loader expects, per sample prefix:

    data_root/<prefix>/data.npz
    cond_root/<prefix>/imgs.npz         key: svr_imgs
    cond_root/<prefix>/real_photo.npz   key: flux

This script keeps the raw photos untouched and writes a small test dataset
under ``prepared/`` by default. Each input photo becomes one test sample, which
is the most direct way to run all 16 real camera angles with
``dataset.condition=[single_img]`` and ``dataset.is_aug=0``.
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps


HERE = Path(__file__).resolve().parent
DEFAULT_OUT_ROOT = HERE / "prepared"
DEFAULT_PLACEHOLDER_DATA_NPZ = Path("/mnt/d/data/deepcad_v6/00786107/data.npz")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
DEFAULT_IMAGE_SIZE = 512
DEFAULT_NUM_SVR_VIEWS = 1
DEFAULT_OBJECT_FILL = 0.62
GENERATED_DIRS = ("data_root", "cond_root", "processed")
GENERATED_FILES = ("test_list.txt", "manifest.csv", "contact_sheet.png")


def natural_sort_key(path: Path) -> list[object]:
    parts = re.split(r"(\d+)", path.name)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def discover_images(input_dir: Path, extensions: tuple[str, ...]) -> list[Path]:
    allowed = {ext.lower() for ext in extensions}
    images = [
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in allowed
    ]
    return sorted(images, key=natural_sort_key)


def center_crop_square(image: Image.Image) -> Image.Image:
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    return image.crop((left, top, left + side, top + side))


def pad_square(image: Image.Image, size: int, fill: tuple[int, int, int]) -> Image.Image:
    width, height = image.size
    scale = min(size / width, size / height)
    resized = image.resize((round(width * scale), round(height * scale)), Image.BICUBIC)
    canvas = Image.new("RGB", (size, size), fill)
    left = (size - resized.width) // 2
    top = (size - resized.height) // 2
    canvas.paste(resized, (left, top))
    return canvas


def object_bbox_from_cool_pixels(image: Image.Image) -> tuple[int, int, int, int] | None:
    """Find the blue-gray printed part against the warm wood background."""

    arr = np.asarray(image, dtype=np.int16)
    red = arr[..., 0]
    green = arr[..., 1]
    blue = arr[..., 2]
    brightness = arr.mean(axis=-1)

    mask = (
        (blue - red > 18)
        & (green - red > 4)
        & (blue > 70)
        & (brightness > 45)
    )

    ys, xs = np.nonzero(mask)
    if xs.size < 100:
        return None

    x0 = int(xs.min())
    y0 = int(ys.min())
    x1 = int(xs.max()) + 1
    y1 = int(ys.max()) + 1
    return x0, y0, x1, y1


def crop_around_bbox(
    image: Image.Image,
    bbox: tuple[int, int, int, int],
    object_fill: float,
) -> Image.Image:
    width, height = image.size
    x0, y0, x1, y1 = bbox
    box_w = max(1, x1 - x0)
    box_h = max(1, y1 - y0)
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0

    crop_side = int(np.ceil(max(box_w, box_h) / object_fill))
    crop_side = max(crop_side, max(box_w, box_h))
    crop_side = min(crop_side, width, height)

    left = int(round(cx - crop_side / 2.0))
    top = int(round(cy - crop_side / 2.0))
    left = min(max(left, 0), width - crop_side)
    top = min(max(top, 0), height - crop_side)
    return image.crop((left, top, left + crop_side, top + crop_side))


def load_prepared_image(
    image_path: Path,
    size: int,
    crop_mode: str,
    pad_color: tuple[int, int, int],
    object_fill: float,
) -> tuple[np.ndarray, tuple[int, int], tuple[int, int, int, int] | None]:
    bbox = None
    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        original_size = image.size
        if crop_mode == "center-crop":
            image = center_crop_square(image).resize((size, size), Image.BICUBIC)
        elif crop_mode == "pad":
            image = pad_square(image, size, pad_color)
        elif crop_mode == "object-crop":
            bbox = object_bbox_from_cool_pixels(image)
            if bbox is None:
                image = center_crop_square(image)
            else:
                image = crop_around_bbox(image, bbox, object_fill)
            image = image.resize((size, size), Image.BICUBIC)
        else:
            raise ValueError(f"Unsupported crop mode: {crop_mode}")
        array = np.asarray(image, dtype=np.uint8)

    expected_shape = (size, size, 3)
    if array.shape != expected_shape:
        raise RuntimeError(f"{image_path} produced {array.shape}, expected {expected_shape}")
    return array, original_size, bbox


def parse_rgb(text: str) -> tuple[int, int, int]:
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected R,G,B")
    try:
        rgb = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected integer R,G,B") from exc
    if any(value < 0 or value > 255 for value in rgb):
        raise argparse.ArgumentTypeError("RGB values must be in [0, 255]")
    return rgb  # type: ignore[return-value]


def make_contact_sheet(images: list[np.ndarray], labels: list[str], tile: int = 160) -> Image.Image:
    cols = 4
    rows = int(np.ceil(len(images) / cols))
    label_h = 24
    sheet = Image.new("RGB", (cols * tile, rows * (tile + label_h)), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    for idx, (array, label) in enumerate(zip(images, labels)):
        row, col = divmod(idx, cols)
        tile_img = Image.fromarray(array).resize((tile, tile), Image.BICUBIC)
        x = col * tile
        y = row * (tile + label_h)
        sheet.paste(tile_img, (x, y))
        draw.text((x + 6, y + tile + 5), label, fill=(0, 0, 0))
    return sheet


def write_sample(
    prefix: str,
    image_array: np.ndarray,
    source_path: Path,
    data_root: Path,
    cond_root: Path,
    processed_root: Path,
    placeholder_data_npz: Path,
    num_svr_views: int,
) -> None:
    sample_data_dir = data_root / prefix
    sample_cond_dir = cond_root / prefix
    sample_data_dir.mkdir(parents=True, exist_ok=True)
    sample_cond_dir.mkdir(parents=True, exist_ok=True)

    shutil.copyfile(placeholder_data_npz, sample_data_dir / "data.npz")

    svr_imgs = np.repeat(image_array[None, ...], num_svr_views, axis=0)
    np.savez_compressed(sample_cond_dir / "imgs.npz", svr_imgs=svr_imgs)

    # Current real-photo captures are single identity images. Future datasets
    # can store flux as [24, H, W, 3] in the same file.
    np.savez_compressed(sample_cond_dir / "real_photo.npz", flux=image_array)

    Image.fromarray(image_array).save(processed_root / f"{prefix}.png")
    (sample_cond_dir / "source.txt").write_text(str(source_path) + "\n", encoding="utf-8")


def prepare_output_root(out_root: Path, overwrite: bool) -> None:
    if not out_root.exists():
        out_root.mkdir(parents=True)
        return
    if not overwrite:
        raise FileExistsError(f"{out_root} already exists; pass --overwrite to replace generated files")

    for dirname in GENERATED_DIRS:
        path = out_root / dirname
        if path.exists():
            shutil.rmtree(path)
    for filename in GENERATED_FILES:
        path = out_root / filename
        if path.exists():
            path.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=HERE)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--prefix", default="real_photo")
    parser.add_argument("--size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument(
        "--crop-mode",
        choices=("object-crop", "center-crop", "pad"),
        default="object-crop",
        help=(
            "object-crop zooms around the blue-gray printed part; center-crop "
            "matches the earlier demo script; pad keeps the full photo."
        ),
    )
    parser.add_argument("--pad-color", type=parse_rgb, default=(255, 255, 255))
    parser.add_argument(
        "--object-fill",
        type=float,
        default=DEFAULT_OBJECT_FILL,
        help=(
            "For object-crop, target fraction of the final square occupied by "
            "the detected object bbox. Larger values zoom in more."
        ),
    )
    parser.add_argument(
        "--num-svr-views",
        type=int,
        default=DEFAULT_NUM_SVR_VIEWS,
        help=(
            "Number of views stored in imgs.npz/svr_imgs. The provided testing "
            "command uses dataset.is_aug=0, so one view is enough. Use 24 if a "
            "caller will index cube24 views."
        ),
    )
    parser.add_argument(
        "--placeholder-data-npz",
        type=Path,
        default=DEFAULT_PLACEHOLDER_DATA_NPZ,
        help="Any valid data.npz used to satisfy AutoEncoder_dataset3 during testing.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    out_root = args.out_root.resolve()
    placeholder_data_npz = args.placeholder_data_npz.resolve()

    if not input_dir.is_dir():
        raise FileNotFoundError(f"input directory does not exist: {input_dir}")
    if not placeholder_data_npz.is_file():
        raise FileNotFoundError(
            f"placeholder data.npz does not exist: {placeholder_data_npz}"
        )
    if args.size <= 0:
        raise ValueError("--size must be positive")
    if args.num_svr_views <= 0:
        raise ValueError("--num-svr-views must be positive")
    if not 0.2 <= args.object_fill <= 0.95:
        raise ValueError("--object-fill must be in [0.2, 0.95]")
    prepare_output_root(out_root, args.overwrite)

    image_paths = discover_images(input_dir, IMAGE_EXTENSIONS)
    if not image_paths:
        raise RuntimeError(f"no images found in {input_dir}")

    data_root = out_root / "data_root"
    cond_root = out_root / "cond_root"
    processed_root = out_root / "processed"
    data_root.mkdir(parents=True)
    cond_root.mkdir(parents=True)
    processed_root.mkdir(parents=True)

    prefixes: list[str] = []
    manifest_rows: list[dict[str, str]] = []
    processed_arrays: list[np.ndarray] = []
    labels: list[str] = []

    for idx, image_path in enumerate(image_paths):
        prefix = f"{args.prefix}_{idx:02d}"
        array, original_size, bbox = load_prepared_image(
            image_path,
            args.size,
            args.crop_mode,
            args.pad_color,
            args.object_fill,
        )
        write_sample(
            prefix=prefix,
            image_array=array,
            source_path=image_path,
            data_root=data_root,
            cond_root=cond_root,
            processed_root=processed_root,
            placeholder_data_npz=placeholder_data_npz,
            num_svr_views=args.num_svr_views,
        )
        prefixes.append(prefix)
        processed_arrays.append(array)
        labels.append(prefix)
        manifest_rows.append(
            {
                "prefix": prefix,
                "source": str(image_path),
                "original_width": str(original_size[0]),
                "original_height": str(original_size[1]),
                "output_width": str(args.size),
                "output_height": str(args.size),
                "crop_mode": args.crop_mode,
                "object_fill": str(args.object_fill),
                "detected_bbox": "" if bbox is None else ",".join(str(v) for v in bbox),
                "num_svr_views": str(args.num_svr_views),
            }
        )
        print(f"[{idx:02d}] {image_path.name} -> {prefix} {array.shape}")

    list_path = out_root / "test_list.txt"
    list_path.write_text("\n".join(prefixes) + "\n", encoding="utf-8")

    manifest_path = out_root / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)

    contact_sheet = make_contact_sheet(processed_arrays, labels)
    contact_sheet.save(out_root / "contact_sheet.png")

    print()
    print(f"Wrote {len(prefixes)} samples to {out_root}")
    print(f"  data_root    = {data_root}")
    print(f"  cond_root    = {cond_root}")
    print(f"  test_dataset = {list_path}")
    print(f"  manifest     = {manifest_path}")
    print()
    print("Use these overrides in your test command:")
    print(f"  dataset.data_root={data_root}")
    print(f"  dataset.cond_root={cond_root}")
    print(f"  dataset.train_dataset={list_path}")
    print(f"  dataset.val_dataset={list_path}")
    print(f"  dataset.test_dataset={list_path}")


if __name__ == "__main__":
    main()
