"""Describe final retouch outputs and make fixed enhanced/official/projected panels."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import binary_erosion

from run import encode, linearize, sha256, write_json


def brightness(pixels: np.ndarray) -> np.ndarray:
    """Display-domain weighted RGB, not measured scene luminance."""
    return pixels.astype(np.float32) @ np.array([0.2126, 0.7152, 0.0722])


def statistics(pixels: np.ndarray) -> dict:
    return {
        "mean_rgb_255": pixels.mean(axis=0).tolist(),
        "brightness_p01_p50_p99_255": np.percentile(brightness(pixels), [1, 50, 99]).tolist(),
        "any_channel_zero_fraction": float((pixels == 0).any(axis=1).mean()),
        "all_channels_zero_fraction": float((pixels == 0).all(axis=1).mean()),
        "any_channel_white_fraction": float((pixels == 255).any(axis=1).mean()),
        "all_channels_white_fraction": float((pixels == 255).all(axis=1).mean()),
        "all_channels_near_black_fraction": float((pixels <= 3).all(axis=1).mean()),
        "all_channels_near_white_fraction": float((pixels >= 252).all(axis=1).mean()),
    }


def gradient(pixels: np.ndarray) -> np.ndarray:
    vertical, horizontal = np.gradient(brightness(pixels))
    return np.hypot(vertical, horizontal)


def save_comparison(paths: dict, box: tuple, destination: Path, sample_id: str) -> None:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    labels = ["Enhanced input", "Official DeepLPF", "Smooth local gain"]
    full = Image.new("RGB", (1536, 552), "#fafafa")
    crop = Image.new("RGB", (1152, 424), "#fafafa")
    for col, (path, label) in enumerate(zip(paths.values(), labels)):
        image = Image.open(path).convert("RGB")
        ImageDraw.Draw(full).text((col * 512 + 12, 10), f"{sample_id} | {label}", font=font, fill="#222222")
        full.paste(image, (col * 512, 40))
        ImageDraw.Draw(crop).text((col * 384 + 8, 10), label, font=font, fill="#222222")
        crop.paste(image.crop(box).resize((384, 384)), (col * 384, 40))
    full.save(destination / "refinement_comparison.jpg", quality=95)
    crop.save(destination / "target_comparison.jpg", quality=95)


def evaluate(output: Path) -> None:
    request = json.loads((output / "request.json").read_text())
    rows = []
    for sample in request["samples"]:
        sample_id = sample["model_id"]
        destination = output / sample_id
        paths = {"enhanced": Path(sample["rgb_file"]),
                 "official": destination / "deeplpf.png",
                 "projected": destination / "local_gain.png"}
        pixels = {name: np.asarray(Image.open(path).convert("RGB")) for name, path in paths.items()}
        assert sha256(paths["enhanced"]) == sample["rgb_sha256"]
        assert sha256(Path(sample["mask_file"])) == sample["mask_sha256"]
        assert all(image.shape == (512, 512, 3) for image in pixels.values())
        mask = np.asarray(Image.open(sample["mask_file"]).convert("L")) > 0
        interior = binary_erosion(mask, iterations=2)
        source_gradient = gradient(pixels["enhanced"])
        valid = source_gradient[interior]
        threshold = max(1.0, float(np.percentile(valid, 90))) if len(valid) else 1.0
        edges = interior & (source_gradient >= threshold)
        row = {"model_id": sample_id, "input_internal_edge_pixels": int(edges.sum()), "variants": {}}
        for name, image in pixels.items():
            delta = np.abs(image.astype(np.float32) - pixels["enhanced"])
            row["variants"][name] = {
                "path": str(paths[name]), "sha256": sha256(paths[name]),
                "whole_image": statistics(image.reshape(-1, 3)), "cad": statistics(image[mask]),
                "cad_rgb_mae_255": float(delta[mask].mean()),
                "cad_changed_pixel_fraction": float((delta[mask] > 0).any(axis=1).mean()),
                "internal_edge_gradient_ratio": float(gradient(image)[edges].mean() / source_gradient[edges].mean()) if edges.any() else None,
            }
        gain = np.load(destination / "gain.npy")
        recalculated = (encode(linearize(pixels["enhanced"].astype(np.float32) / 255) * gain) * 255).astype(np.uint8)
        row["projection_recompute_max_abs_error_255"] = int(np.abs(recalculated.astype(int) - pixels["projected"].astype(int)).max())
        assert row["projection_recompute_max_abs_error_255"] <= 1
        yy, xx = np.where(mask)
        side = int(max(xx.max() - xx.min(), yy.max() - yy.min()) + 48)
        left = int((xx.min() + xx.max() - side) / 2)
        top = int((yy.min() + yy.max() - side) / 2)
        left, top = min(max(left, 0), 512 - side), min(max(top, 0), 512 - side)
        box = (left, top, left + side, top + side)
        row["target_crop_box"] = box
        save_comparison(paths, box, destination, sample_id)
        rows.append(row)
    write_json(output / "refinement_metrics.json", {
        "samples": rows, "coverage": {"planned": 12, "complete": len(rows) * 2},
        "note": "Descriptive statistics, not quality/geometry acceptance. RGB averages do not estimate white-balance accuracy.",
    })
    write_json(output / "refinement_manifest.json", [
        {"model_id": row["model_id"], **{key: value["path"] for key, value in row["variants"].items()},
         "comparison": str(output / row["model_id"] / "refinement_comparison.jpg"),
         "target_comparison": str(output / row["model_id"] / "target_comparison.jpg")}
        for row in rows
    ])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, required=True)
    evaluate(parser.parse_args().output_dir)
