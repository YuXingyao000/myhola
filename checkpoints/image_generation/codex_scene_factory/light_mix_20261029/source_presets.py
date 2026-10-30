"""Create a small, brightness-matched per-lamp comparison from linear EXRs.

Run with Blender --background --python source_presets.py -- --input_dir ...
Each candidate retains one ceiling lamp and reduces the others to 15%.
All groups then receive the same scale to match the baseline's mean linear
luminance. This control separates lighting direction from overall brightness;
it is recorded explicitly and is not an automatic realism criterion.
"""

import argparse
import json
from pathlib import Path
import sys

import bpy
import numpy as np


def mean_luminance(path):
    image = bpy.data.images.load(str(path), check_existing=False)
    values = np.empty(len(image.pixels), np.float32)
    image.pixels.foreach_get(values)
    rgb = values.reshape(-1, 4)[:, :3]
    mean = float((rgb.astype(np.float64) @ np.array([0.2126, 0.7152, 0.0722])).mean())
    bpy.data.images.remove(image)
    return mean


def make_presets(input_dir, ceiling_groups, output):
    metadata = json.loads((input_dir / "metadata.json").read_text())
    groups = metadata["light_groups"]["groups"]
    assert ceiling_groups and set(ceiling_groups) <= set(groups)
    luminance = {g: mean_luminance(input_dir / ("light_" + g + ".exr")) for g in groups}
    total = sum(luminance.values())
    assert total > 0
    presets, skipped = {}, []
    for key in ceiling_groups:
        raw = {g: (0.15 if g in ceiling_groups and g != key else 1.0) for g in groups}
        mean = sum(luminance[g] * raw[g] for g in groups)
        scale = total / mean
        if scale > 2:
            skipped.append({"key": key, "reason": "would require more than 2x intensity scale", "scale": scale})
            continue
        presets["key_" + key] = {
            "description": f"保留{key}，其他顶灯降至15%；全组乘{scale:.6f}以匹配原始全画面平均线性亮度。",
            "weights": {g: raw[g] * scale for g in groups},
        }
    result = {"presets": presets, "luminance": luminance, "skipped": skipped,
              "input_dir": str(input_dir), "ceiling_groups": ceiling_groups,
              "protocol": "Preserve one ceiling lamp; others 0.15. Match mean scene-linear luminance; scale<=2. No exposure/look changes."}
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--ceiling_groups", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    make_presets(args.input_dir, args.ceiling_groups, args.output)
