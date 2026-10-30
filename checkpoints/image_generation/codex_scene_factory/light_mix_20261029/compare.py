"""Arrange existing light-mix PNGs without changing their pixels or framing."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


FONT = "/root/.cache/cad_scene_figures/SourceHanSansSC-Regular.otf"


def make_board(spec, output):
    """The JSON explicitly names every source image and its column label."""
    columns, rows = spec["columns"], spec["rows"]
    width, row_height = 512 * len(columns), 552
    page = Image.new("RGB", (width, 132 + row_height * len(rows)), "#f6f7f9")
    draw = ImageDraw.Draw(page)
    title_font, label_font, small_font = [ImageFont.truetype(FONT, n) for n in (30, 23, 19)]
    draw.text((16, 10), spec["title"], font=title_font, fill="#172a3a")
    draw.text((16, 55), spec["subtitle"], font=small_font, fill="#526575")
    for col, label in enumerate(columns):
        draw.text((col * 512 + 16, 95), label, font=label_font, fill="#172a3a")
    records = []
    for row, item in enumerate(rows):
        y = 132 + row * row_height
        draw.text((16, y), item["label"], font=label_font, fill="#172a3a")
        assert len(item["images"]) == len(columns)
        for col, path in enumerate(item["images"]):
            im = Image.open(path).convert("RGB")
            assert im.size == (512, 512), path
            page.paste(im, (col * 512, y + 40))
            pixels = np.asarray(im)
            records.append({"label": item["label"], "column": columns[col], "image": path,
                            "channel_zero_fraction": float((pixels == 0).mean()),
                            "channel_255_fraction": float((pixels == 255).mean())})
    output.parent.mkdir(parents=True, exist_ok=True)
    page.save(output)
    output.with_suffix(".json").write_text(json.dumps({"spec": spec, "diagnostics": records},
                                                     ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    make_board(json.loads(args.spec.read_text()), args.output)
