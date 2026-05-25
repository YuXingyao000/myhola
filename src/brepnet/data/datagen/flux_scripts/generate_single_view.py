r"""
Generate FLUX.1-Kontext images from Blender-rendered single-view material images.

Delegates to the local `external/generation` package.

Input layout:
    render-root/{model_id}/0/{material_idx}.png

Output layout:
    output-root/{model_id}/0/{material_idx}.png

Typical direct usage:
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 RANK=0 \
    python3 ./generate_flux_kontext_from_blender.py \
        --render-root /path/to/output_single_view \
        --output-root /path/to/output_flux_single_view \
        --model-list-dir /path/to/render_lists
"""

from __future__ import annotations

import sys
from pathlib import Path

_EXTERNAL_ROOT = Path(__file__).resolve().parents[1] / "external"
if not (_EXTERNAL_ROOT / "generation").is_dir():
    raise SystemExit(f"Expected generation package at {_EXTERNAL_ROOT / 'generation'}")
sys.path.insert(0, str(_EXTERNAL_ROOT))

from generation.generate import main  # noqa: E402

if __name__ == "__main__":
    main()
