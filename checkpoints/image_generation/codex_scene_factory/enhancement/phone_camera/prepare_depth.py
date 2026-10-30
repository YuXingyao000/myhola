"""Convert Step 3 camera-Z EXRs to near-white/far-black depth controls on CPU.

OpenEXR can be loaded from an isolated directory with --openexr_path, without
installing anything into the training or generation Python environments.
"""

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def inverse_depth_control(depth: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Normalize finite positive inverse depths independently for one image."""
    valid = np.isfinite(depth) & (depth > 0)
    if not valid.any():
        raise ValueError("No finite positive depth pixels")
    inverse = 1.0 / depth[valid].astype(np.float64)
    near, far = float(inverse.max()), float(inverse.min())
    if near == far:
        raise ValueError("Constant depth has no relative-depth range")
    gray = np.zeros(depth.shape, dtype=np.uint8)
    gray[valid] = np.rint((inverse - far) / (near - far) * 255).astype(np.uint8)
    rgb = np.repeat(gray[..., None], 3, axis=2)
    metrics = {
        "valid_pixels": int(valid.sum()), "invalid_pixels": int((~valid).sum()),
        "z_min_m": float(depth[valid].min()), "z_max_m": float(depth[valid].max()),
        "inverse_depth_min_per_m": far, "inverse_depth_max_per_m": near,
        "control_min": int(gray[valid].min()), "control_max": int(gray[valid].max()),
        "control_unique_values": int(np.unique(gray).size),
    }
    return rgb, valid, metrics


def make_preview(records: list[dict], output_dir: Path) -> None:
    """Show all six source/control pairs at full input resolution."""
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    for page_index in range(0, len(records), 3):
        page_records = records[page_index:page_index + 3]
        page = Image.new("RGB", (1024, 552 * len(page_records)), "#fafafa")
        draw = ImageDraw.Draw(page)
        for row, record in enumerate(page_records):
            for col, (path, label) in enumerate([
                (record["rgb_file"], f"{record['model_id']} · source RGB"),
                (record["depth_file"], "Inverse depth · near white, far black"),
            ]):
                draw.text((col * 512 + 12, row * 552 + 12), label, font=font, fill="#222222")
                page.paste(Image.open(path).convert("RGB"), (col * 512, row * 552 + 40))
        page.save(output_dir / f"preview_{page_index // 3:02d}.jpg", quality=95)


def run(args) -> None:
    if args.openexr_path:
        sys.path.insert(0, str(args.openexr_path))
    import OpenEXR

    samples = json.loads(args.samples.read_text())
    assert samples
    args.output_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, args.output_dir / "prepare_depth.py")
    request = {
        "samples_file": str(args.samples.resolve()), "samples_sha256": sha256(args.samples),
        "planned": len(samples), "channel": args.channel, "device": "cpu",
        "reader": "OpenEXR.File(separate_channels=True)", "openexr_version": OpenEXR.__version__,
        "openexr_path": str(args.openexr_path), "numpy_version": np.__version__,
        "source_convention": "camera-axis Z meters, as declared by Step 3 metadata",
        "normalization": "round(255*(1/z-min_valid(1/z))/(max_valid(1/z)-min_valid(1/z)))",
        "validity": "finite and z>0; invalid pixels black; constant/empty depth fails",
        "orientation": "reader's top-left row order; no rotation, flip, resize, or spatial filter",
        "limitation": "Metric inverse-depth normalization is not claimed equivalent to Depth Anything V2 training conditions.",
        "code_sha256": sha256(Path(__file__)), "command": sys.argv,
    }
    write_json(args.output_dir / "request.json", request)
    records = []
    started = time.perf_counter()
    for sample in samples:
        source = Path(sample["source"]["images"]["depth"])
        depth = OpenEXR.File(str(source), separate_channels=True).channels()[args.channel].pixels
        assert depth.shape == (512, 512)
        control, valid, metrics = inverse_depth_control(depth)
        sample_dir = args.output_dir / sample["model_id"]
        sample_dir.mkdir()
        control_path = sample_dir / "depth.png"
        valid_path = sample_dir / "valid_mask.png"
        Image.fromarray(control).save(control_path)
        Image.fromarray(valid.astype(np.uint8) * 255).save(valid_path)
        record = {
            "model_id": sample["model_id"], "source_depth_file": str(source),
            "source_depth_sha256": sha256(source), "source_rgb_sha256": sample["rgb_sha256"],
            "rgb_file": sample["rgb_file"], "depth_file": str(control_path.resolve()),
            "depth_sha256": sha256(control_path), "valid_mask_file": str(valid_path.resolve()),
            "valid_mask_sha256": sha256(valid_path), "source_channel": args.channel,
            "output_shape": list(control.shape), "metrics": metrics, "status": "success",
        }
        write_json(sample_dir / "metadata.json", record)
        records.append(record)
        write_json(args.output_dir / "samples.json", records)
        print(sample["model_id"], metrics, flush=True)
    make_preview(records, args.output_dir)
    write_json(args.output_dir / "summary.json", {
        "planned": len(samples), "success": len(records), "failed": 0, "missing": 0,
        "elapsed_seconds": time.perf_counter() - started,
        "visual_alignment": "pending", "depth_estimator_distribution_equivalence": "not_claimed",
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--openexr_path", type=Path)
    parser.add_argument("--channel", default="V", help="Step 3 BW EXRs contain the V channel")
    run(parser.parse_args())
