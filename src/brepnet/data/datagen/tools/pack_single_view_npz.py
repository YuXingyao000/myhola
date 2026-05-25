r"""Pack single-view Blender / FLUX / FLUX-masked PNGs into single_view.npz.

For each model id, reads the single-view renders produced by
``run_blender.py single-view``, ``run_flux.py single-view`` and
``mask_flux_with_blender.py``::

    {blender_root}/{model_id}/0/0.png       -> key ``blender``
    {flux_root}/{model_id}/0/0.png          -> key ``flux``
    {flux_masked_root}/{model_id}/0/0.png   -> key ``flux_masked``

Each PNG is resized to ``IMAGE_SIZE`` (default 512x512) RGB uint8 and stored as
shape ``(IMAGE_SIZE, IMAGE_SIZE, 3)`` per key. The resulting npz is written to::

    {target_root}/{model_id}/single_view.npz

Models that are missing any of the three required PNGs are skipped (with a
log message) unless ``--require=`` is relaxed (see ``--require``).

Usage (defaults pulled from ``config.PATHS``)::

    python -m src.brepnet.data.DataGenerationRefactored.tools.pack_single_view_npz

Override roots / target / list / size::

    python -m src.brepnet.data.DataGenerationRefactored.tools.pack_single_view_npz \
        --blender-root .../outputs/blender_single_view \
        --flux-root    .../outputs/flux_single_view \
        --flux-masked-root .../outputs/flux_single_view_masked \
        --target-root  /mnt/d/data/deepcad_v6_cond \
        --model-list   .../test_list/model_list.txt \
        --num-workers  64
"""

from __future__ import annotations

import argparse
import concurrent.futures
import re
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

from ..config import PATHS, RUNTIME


MODEL_DIR_PATTERN = re.compile(r"^\d{8}$")
IMAGE_SIZE = 512
OUTPUT_NAME = "single_view.npz"

ALL_KEYS: tuple[str, ...] = ("blender", "flux", "flux_masked")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--blender-root",
        type=Path,
        default=PATHS.blender_single_view_out,
        help="Root holding {model_id}/0/0.png Blender renders.",
    )
    p.add_argument(
        "--flux-root",
        type=Path,
        default=PATHS.flux_single_view_out,
        help="Root holding {model_id}/0/0.png FLUX renders.",
    )
    p.add_argument(
        "--flux-masked-root",
        type=Path,
        default=PATHS.project_root / "outputs" / "flux_single_view_masked",
        help="Root holding {model_id}/0/0.png FLUX-masked renders.",
    )
    p.add_argument(
        "--target-root",
        type=Path,
        default=PATHS.cond_single_view_target,
        help="Per-model output root (writes {target_root}/{model_id}/single_view.npz).",
    )
    p.add_argument(
        "--model-list",
        type=Path,
        default=None,
        help="Optional txt file restricting which model ids to pack (one 8-digit id per line).",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=RUNTIME.pack_workers,
        help="Number of worker threads.",
    )
    p.add_argument(
        "--image-size",
        type=int,
        default=IMAGE_SIZE,
        help="Output image side length (kept square).",
    )
    p.add_argument(
        "--require",
        nargs="+",
        choices=ALL_KEYS,
        default=list(ALL_KEYS),
        help="Keys that MUST exist for a model to be packed; default requires all three.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        default=RUNTIME.pack_overwrite,
        help="Overwrite existing single_view.npz files.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report what would be packed without writing files.",
    )
    return p.parse_args()


def load_model_ids_from_list(list_path: Path) -> list[str]:
    if not list_path.is_file():
        raise FileNotFoundError(f"Model list does not exist: {list_path}")
    seen: set[str] = set()
    out: list[str] = []
    for line_no, raw in enumerate(list_path.read_text(encoding="utf-8").splitlines(), start=1):
        mid = raw.strip()
        if not mid or mid.startswith("#"):
            continue
        if not MODEL_DIR_PATTERN.match(mid):
            raise ValueError(f"Invalid model id {mid!r} in {list_path} line {line_no}; expected 8 digits.")
        if mid in seen:
            continue
        seen.add(mid)
        out.append(mid)
    return out


def discover_model_ids(root: Path) -> set[str]:
    if not root.exists():
        return set()
    return {
        p.name
        for p in root.iterdir()
        if p.is_dir() and MODEL_DIR_PATTERN.match(p.name)
    }


def png_path_for(root: Path, model_id: str) -> Path:
    return root / model_id / "0" / "0.png"


def load_resized_png(image_path: Path, size: int) -> np.ndarray:
    with Image.open(image_path) as im:
        im = im.convert("RGB")
        if im.size != (size, size):
            im = im.resize((size, size), Image.BICUBIC)
        arr = np.asarray(im, dtype=np.uint8)
    if arr.shape != (size, size, 3):
        raise ValueError(f"Unexpected shape {arr.shape} for {image_path}")
    return arr


def select_model_ids(
    blender_root: Path,
    flux_root: Path,
    flux_masked_root: Path,
    model_list: Path | None,
    required: Iterable[str],
) -> tuple[list[str], dict[str, list[str]]]:
    """Return (selected_ids, missing_breakdown).

    ``missing_breakdown[key]`` lists ids that fail the existence check for that key.
    Ids missing any *required* key are excluded from the selection.
    """
    discovered: dict[str, set[str]] = {
        "blender": discover_model_ids(blender_root),
        "flux": discover_model_ids(flux_root),
        "flux_masked": discover_model_ids(flux_masked_root),
    }

    if model_list is not None:
        candidate_ids = load_model_ids_from_list(model_list)
    else:
        candidate_ids = sorted(set().union(*discovered.values()))

    required_set = set(required)
    selected: list[str] = []
    missing: dict[str, list[str]] = {k: [] for k in ALL_KEYS}
    for mid in candidate_ids:
        # Confirm the actual PNG exists, not just the directory.
        png_ok = {
            "blender": png_path_for(blender_root, mid).is_file(),
            "flux": png_path_for(flux_root, mid).is_file(),
            "flux_masked": png_path_for(flux_masked_root, mid).is_file(),
        }
        for k, ok in png_ok.items():
            if not ok:
                missing[k].append(mid)
        if all(png_ok[k] for k in required_set):
            selected.append(mid)
    return selected, missing


def pack_one(
    model_id: str,
    blender_root: Path,
    flux_root: Path,
    flux_masked_root: Path,
    target_root: Path,
    image_size: int,
    required: set[str],
    overwrite: bool,
) -> str:
    out_dir = target_root / model_id
    out_path = out_dir / OUTPUT_NAME
    if out_path.is_file() and not overwrite:
        return f"skip {model_id} (exists)"

    sources = {
        "blender": png_path_for(blender_root, model_id),
        "flux": png_path_for(flux_root, model_id),
        "flux_masked": png_path_for(flux_masked_root, model_id),
    }

    arrays: dict[str, np.ndarray] = {}
    for key, png in sources.items():
        if not png.is_file():
            if key in required:
                raise FileNotFoundError(f"Missing required {key} png for {model_id}: {png}")
            continue
        arrays[key] = load_resized_png(png, image_size)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **arrays)
    return f"saved {model_id} ({', '.join(sorted(arrays))})"


def main() -> None:
    args = parse_args()
    blender_root = args.blender_root.resolve()
    flux_root = args.flux_root.resolve()
    flux_masked_root = args.flux_masked_root.resolve()
    target_root = args.target_root.resolve()
    model_list = args.model_list.resolve() if args.model_list is not None else None

    if args.num_workers <= 0:
        raise ValueError("--num-workers must be a positive integer.")
    if args.image_size <= 0:
        raise ValueError("--image-size must be a positive integer.")

    print(f"blender_root      = {blender_root}")
    print(f"flux_root         = {flux_root}")
    print(f"flux_masked_root  = {flux_masked_root}")
    print(f"target_root       = {target_root}")
    print(f"model_list        = {model_list}")
    print(f"image_size        = {args.image_size}x{args.image_size}")
    print(f"required keys     = {sorted(args.require)}")
    print(f"num_workers       = {args.num_workers}")
    print(f"overwrite         = {args.overwrite}")

    selected, missing = select_model_ids(
        blender_root,
        flux_root,
        flux_masked_root,
        model_list,
        args.require,
    )
    for key in ALL_KEYS:
        if missing[key]:
            print(f"missing {key:<12s}: {len(missing[key])}")
    print(f"selected models   = {len(selected)}")

    if args.dry_run:
        for mid in selected[:10]:
            print(f"  would pack {mid}")
        if len(selected) > 10:
            print(f"  ... and {len(selected) - 10} more")
        return

    if not selected:
        print("Nothing to pack.")
        return

    target_root.mkdir(parents=True, exist_ok=True)

    saved = 0
    skipped = 0
    failed = 0
    required_set = set(args.require)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as pool:
        future_to_mid = {
            pool.submit(
                pack_one,
                mid,
                blender_root,
                flux_root,
                flux_masked_root,
                target_root,
                args.image_size,
                required_set,
                args.overwrite,
            ): mid
            for mid in selected
        }
        for fut in concurrent.futures.as_completed(future_to_mid):
            mid = future_to_mid[fut]
            try:
                msg = fut.result()
            except Exception as exc:
                failed += 1
                print(f"error {mid}: {exc}")
                continue
            if msg.startswith("saved"):
                saved += 1
            else:
                skipped += 1
            if (saved + skipped) % 500 == 0:
                print(f"  progress: saved={saved} skipped={skipped} failed={failed}")

    print(f"done saved={saved} skipped={skipped} failed={failed}")
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
