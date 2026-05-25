"""Use blender white-background renders as masks to white out the FLUX image's
background. Walks ``outputs/flux_single_view`` recursively and writes
white-background versions to ``outputs/flux_single_view_masked``.

Run from a python env that has ``ray`` installed (e.g. ``conda activate torch``)::

    python -m src.brepnet.data.DataGenerationRefactored.mask_flux_with_blender
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import ray
from PIL import Image
from tqdm import tqdm

HERE = Path(__file__).parent
BLENDER_ROOT = HERE / "outputs/blender_single_view"
FLUX_ROOT = HERE / "outputs/flux_single_view"
OUT_ROOT = HERE / "outputs/flux_single_view_masked"

# Pixels in the blender render whose luminance (PIL ``L``, ITU-R 601) is
# below this are treated as foreground (i.e. the model). Anything brighter
# is background.
WHITE_THRESHOLD = 252


def mask_one(blender_path: Path, flux_path: Path, out_path: Path) -> None:
    blender_gray = np.array(Image.open(blender_path).convert("L"))
    flux = np.array(Image.open(flux_path).convert("RGB"))
    mask = blender_gray < WHITE_THRESHOLD
    out = np.where(mask[..., None], flux, 255).astype(np.uint8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out).save(out_path)


@ray.remote
def _mask_remote(blender_path: str, flux_path: str, out_path: str) -> str | None:
    try:
        mask_one(Path(blender_path), Path(flux_path), Path(out_path))
        return None
    except Exception as exc:
        return f"{flux_path}: {exc!r}"


def main() -> None:
    tasks = []
    for flux_path in FLUX_ROOT.rglob("*.png"):
        rel = flux_path.relative_to(FLUX_ROOT)
        blender_path = BLENDER_ROOT / rel
        out_path = OUT_ROOT / rel
        if not blender_path.is_file() or out_path.is_file():
            continue
        tasks.append((blender_path, flux_path, out_path))

    print(f"To process: {len(tasks)} (out -> {OUT_ROOT})")
    if not tasks:
        return

    ray.init(ignore_reinit_error=True, log_to_driver=False)
    futures = [_mask_remote.remote(str(b), str(f), str(o)) for b, f, o in tasks]
    errors = []
    with tqdm(total=len(futures)) as pbar:
        while futures:
            done, futures = ray.wait(futures, num_returns=min(64, len(futures)))
            for err in ray.get(done):
                if err:
                    errors.append(err)
            pbar.update(len(done))
    ray.shutdown()

    if errors:
        print(f"errors: {len(errors)} (showing first 5):")
        for msg in errors[:5]:
            print(f"  {msg}")


if __name__ == "__main__":
    main()
