"""CPU DeepLPF inference and smooth local-gain projection onto original pixels."""

import argparse
import ast
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
from types import ModuleType

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter
import torch


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def load_network(model_dir: Path):
    """Use official inference definitions; omit only an unused training-loss import."""
    sys.path.insert(0, str(model_dir))
    source = model_dir / "model.py"
    tree = ast.parse(source.read_text())
    tree.body = [node for node in tree.body
                 if not (isinstance(node, ast.ImportFrom) and node.module == "losses")]
    module = ModuleType("deeplpf_official_inference")
    exec(compile(tree, str(source), "exec"), module.__dict__)
    net = module.DeepLPFNet().eval()
    checkpoint = model_dir / "pretrained_models/adobe_dpe/deeplpf_psnr_24.180_ssim_0.917_model.pt"
    net.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    return net


def linearize(rgb: np.ndarray) -> np.ndarray:
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def encode(rgb: np.ndarray) -> np.ndarray:
    rgb = np.clip(rgb, 0, 1)
    return np.where(rgb <= 0.0031308, rgb * 12.92, 1.055 * rgb ** (1 / 2.4) - 0.055)


def project_local_gain(source: np.ndarray, reference: np.ndarray, sigma: float = 24):
    """Transfer only broad local RGB exposure trends, never reference pixel locations."""
    source_linear = linearize(source)
    source_low = gaussian_filter(source_linear, sigma=(sigma, sigma, 0), mode="reflect")
    reference_low = gaussian_filter(linearize(reference), sigma=(sigma, sigma, 0), mode="reflect")
    gain = np.clip((reference_low + 0.01) / (source_low + 0.01), 0.5, 2.0)
    return encode(source_linear * gain), gain


def save_rgb(path: Path, rgb: np.ndarray) -> None:
    # Match the official DeepLPF CLI's uint8 truncation, rather than adding rounding.
    Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)).save(path)


def compare(source_path: str, sample_dir: Path, model_id: str) -> None:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    page = Image.new("RGB", (1536, 552), "#fafafa")
    draw = ImageDraw.Draw(page)
    for col, (path, label) in enumerate([
        (source_path, f"{model_id} · Original"),
        (sample_dir / "deeplpf.png", "Official DeepLPF"),
        (sample_dir / "local_gain.png", "Smooth local gain from DeepLPF"),
    ]):
        draw.text((col * 512 + 12, 12), label, fill="#222222", font=font)
        page.paste(Image.open(path).convert("RGB"), (col * 512, 40))
    page.save(sample_dir / "comparison.jpg", quality=95)


def validate_projection() -> dict:
    rng = np.random.default_rng(8)
    source = rng.uniform(0.05, 0.5, (64, 64, 3)).astype(np.float32)
    same, gain = project_local_gain(source, source)
    assert np.max(np.abs(source - same)) < 2e-7
    assert np.all(gain == 1)
    illumination = np.linspace(0.7, 1.4, 64)[None, :, None]
    reference = encode(linearize(source) * illumination)
    changed, gain = project_local_gain(source, reference, sigma=8)
    assert np.all(gain >= 0.5) and np.all(gain <= 2)
    assert float(gain.std()) > 0.1
    expected = encode(linearize(source) * gain)
    assert np.array_equal(changed, expected)
    return {"identity_max_abs_error": float(np.max(np.abs(source - same))),
            "varying_reference_gain_std": float(gain.std()), "status": "pass"}


def run(args) -> None:
    torch.set_num_threads(4)
    samples = json.loads(args.samples.read_text())
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, output / "run.py")
    model_source = json.loads((args.model_dir / "source.json").read_text())
    for record in model_source["files"]:
        assert sha256(args.model_dir / record["path"]) == record["sha256"]
    request = {
        "samples": samples, "model_source": model_source, "device": "cpu", "threads": 4,
        "torch": torch.__version__, "numpy": np.__version__, "source_sha256": sha256(Path(__file__)),
        "model_dir": str(args.model_dir), "official_adaptation": "Omit unused from losses import DeepLPFLoss; no inference code or weights changed.",
        "projection": {"sigma_pixels": 24, "epsilon_linear_rgb": 0.01, "gain_bounds": [0.5, 2]},
        "outputs": ["official_deeplpf", "smooth_local_gain"], "command": sys.argv,
        "input": "8bit sRGB display-referred PNG; no extra tone map",
        "cad_contract": "Whole-image retouch; CAD colors may change. Smooth projection does not relocate pixels.",
    }
    write_json(output / "request.json", request)
    write_json(output / "projection_validation.json", validate_projection())
    net = load_network(args.model_dir)
    parameter_count = sum(value.numel() for value in net.parameters())
    records = []
    for sample in samples:
        sample_dir = output / sample["model_id"]
        sample_dir.mkdir()
        assert sha256(Path(sample["rgb_file"])) == sample["rgb_sha256"]
        source_u8 = np.asarray(Image.open(sample["rgb_file"]).convert("RGB"))
        assert source_u8.shape == (512, 512, 3)
        source = source_u8.astype(np.float32) / 255
        image = torch.from_numpy(source).permute(2, 0, 1).unsqueeze(0)
        started = time.perf_counter()
        with torch.inference_mode():
            enhanced_tensor = net(image).clamp(0, 1)
        inference_seconds = time.perf_counter() - started
        assert torch.isfinite(enhanced_tensor).all()
        enhanced = enhanced_tensor[0].permute(1, 2, 0).numpy()
        started = time.perf_counter()
        projected, gain = project_local_gain(source, enhanced)
        projection_seconds = time.perf_counter() - started
        save_rgb(sample_dir / "deeplpf.png", enhanced)
        save_rgb(sample_dir / "local_gain.png", projected)
        np.save(sample_dir / "gain.npy", gain)
        save_rgb(sample_dir / "gain_visualization.png", (np.log2(gain) + 1) / 2)
        mask = np.asarray(Image.open(sample["mask_file"]).convert("L")) > 0
        item = {
            "model_id": sample["model_id"], "status": "success", "inference_seconds": inference_seconds,
            "projection_seconds": projection_seconds, "gain_min": float(gain.min()), "gain_max": float(gain.max()),
            "gain_spatial_std_per_channel": gain.std(axis=(0, 1)).tolist(), "outputs": {},
        }
        for name in ("deeplpf", "local_gain"):
            path = sample_dir / f"{name}.png"
            pixels = np.asarray(Image.open(path))
            delta = np.abs(pixels.astype(float) - source_u8)
            item["outputs"][name] = {"path": str(path.resolve()), "sha256": sha256(path),
                "rgb_mae_255": float(delta.mean()), "target_rgb_mae_255": float(delta[mask].mean()),
                "zero_channel_fraction": float((pixels == 0).mean()), "white_channel_fraction": float((pixels == 255).mean())}
        write_json(sample_dir / "metadata.json", item)
        records.append(item)
        write_json(output / "results.json", records)
        compare(sample["rgb_file"], sample_dir, sample["model_id"])
        print(sample["model_id"], f"DeepLPF {inference_seconds:.3f}s; projection {projection_seconds:.3f}s", flush=True)
    write_json(output / "summary.json", {
        "planned_images": len(samples) * 2, "success_images": len(records) * 2, "failed": 0, "missing": 0,
        "parameter_count": parameter_count, "inference_seconds": sum(row["inference_seconds"] for row in records),
        "projection_seconds": sum(row["projection_seconds"] for row in records),
        "human_review": "pending", "realism_gain": "not_established",
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--model_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    run(parser.parse_args())
