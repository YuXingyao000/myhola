"""Read fixed RGB inputs → predict LUT → retouch → mix → save comparisons."""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from model import AdaptiveLUT


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def make_comparisons(samples: list[dict], output_dir: Path) -> None:
    """One legible full-resolution page per input, including both mix levels."""
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = ImageFont.truetype(font_path, 20)
    for sample in samples:
        page = Image.new("RGB", (1536, 1104), "#fafafa")
        draw = ImageDraw.Draw(page)
        for row, variant in enumerate(("paired", "unpaired")):
            paths = [Path(sample["rgb"])] + [
                output_dir / sample["id"] / f"{variant}_mix_{mix:.1f}.png"
                for mix in (0.5, 1.0)
            ]
            labels = [f"Original · {sample['id']}", f"{variant} · mix 0.5", f"{variant} · mix 1.0"]
            for col, (path, label) in enumerate(zip(paths, labels)):
                draw.text((col * 512 + 12, row * 552 + 12), label, font=font, fill="#222222")
                page.paste(Image.open(path).convert("RGB"), (col * 512, row * 552 + 40))
        page.save(output_dir / sample["id"] / "comparison.jpg", quality=95)


def run_workflow(args) -> None:
    torch.set_num_threads(args.threads)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=False)
    samples = json.loads(args.inputs.read_text())
    assert samples
    source_dir = Path(__file__).resolve().parent
    shutil.copytree(source_dir, output_dir / "source", ignore=shutil.ignore_patterns("__pycache__"))
    request = {
        "samples": [{**sample, "rgb_sha256": sha256(Path(sample["rgb"]))} for sample in samples],
        "variants": ["paired", "unpaired"], "mixes": [0.5, 1.0],
        "device": "cpu", "threads": args.threads, "torch": torch.__version__,
        "model_dir": str(args.model_dir),
        "weights_source": json.loads((args.model_dir / "source.json").read_text()),
        "source_sha256": {path.name: sha256(path) for path in (output_dir / "source").glob("*.py")},
        "command": sys.argv,
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source_dir, text=True).strip(),
        "input_contract": "512x512 display-referred 8-bit sRGB; no extra tone mapping",
        "output_contract": "whole-image color mapping; no spatial warp; CAD colors may change",
        "geometry_limit": "No pixel relocation, but tonal clipping may erase visible geometric evidence.",
    }
    write_json(output_dir / "request.json", request)
    (output_dir / "git.diff").write_bytes(subprocess.check_output(["git", "diff", "HEAD"], cwd=source_dir))
    results = []
    start = time.perf_counter()
    for variant in request["variants"]:
        model = AdaptiveLUT(args.model_dir, variant)
        for sample in request["samples"]:
            path = Path(sample["rgb"])
            assert sha256(path) == sample["rgb_sha256"]
            source = np.asarray(Image.open(path).convert("RGB"))
            assert source.shape == (512, 512, 3)
            image = torch.from_numpy(source.copy()).permute(2, 0, 1).float().unsqueeze(0) / 255
            infer_start = time.perf_counter()
            retouched, weights = model(image)
            infer_seconds = time.perf_counter() - infer_start
            assert torch.isfinite(retouched).all()
            sample_dir = output_dir / sample["id"]
            sample_dir.mkdir(exist_ok=True)
            for mix in request["mixes"]:
                mixed = image.lerp(retouched, mix)
                encoded = mixed.squeeze().mul(255).add(0.5).clamp(0, 255).permute(1, 2, 0).to(torch.uint8).numpy()
                image_path = sample_dir / f"{variant}_mix_{mix:.1f}.png"
                Image.fromarray(encoded).save(image_path)
                item = {
                    "id": sample["id"], "variant": variant, "mix": mix,
                    "status": "success", "output": str(image_path), "sha256": sha256(image_path),
                    "inference_seconds": infer_seconds, "inference_reused_for_two_mixes": True,
                    "weights": weights, "rgb_mae_255": float(np.abs(encoded.astype(float) - source).mean()),
                    "output_channel_zero_fraction": float((encoded == 0).mean()),
                    "output_channel_255_fraction": float((encoded == 255).mean()),
                    "preclip_low_fraction": float((mixed < 0).float().mean()),
                    "preclip_high_fraction": float((mixed > 1).float().mean()),
                    "source_channel_zero_fraction": float((source == 0).mean()),
                    "source_channel_255_fraction": float((source == 255).mean()),
                }
                results.append(item)
                with (output_dir / "results.jsonl").open("a") as stream:
                    stream.write(json.dumps(item) + "\n")
            print(f"{variant}: {sample['id']} ({infer_seconds:.3f}s)", flush=True)
    make_comparisons(samples, output_dir)
    write_json(output_dir / "summary.json", {
        "planned": len(samples) * 4, "success": len(results), "failed": 0, "missing": 0,
        "elapsed_seconds": time.perf_counter() - start,
        "inference_seconds": sum(item["inference_seconds"] for item in results if item["mix"] == 1.0),
        "human_review": "pending", "realism_improvement": "not_established",
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--model_dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    run_workflow(parser.parse_args())
