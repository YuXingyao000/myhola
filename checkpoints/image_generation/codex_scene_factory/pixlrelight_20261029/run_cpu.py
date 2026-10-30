"""Isolated PIXLRelight Cycles-input experiment, CPU / 8 threads only.

Run with /mnt/d/natural/pixlrelight_cpu_env/bin/python. The input directory
contains source.png and the six official diffuse/glossy EXRs; optional Cycles
groups follow the official contract. Inputs must be exactly 512x512. No image
crop, inpaint, gamma conversion, diffusion model, or automatic asset download.
Synthetic source images are an unvalidated domain, not photo-enhancement GT.
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "8"
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import argparse
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import sys
import time
import types

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import psutil
import torch
from omegaconf import OmegaConf
from safetensors import safe_open


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def git_output(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def install_official_namespace(repo):
    # Upstream src is a namespace package. This workspace also has a regular
    # package named src, which otherwise wins even when the repo is sys.path[0].
    if "src" in sys.modules:
        raise RuntimeError("Run this wrapper in its own Python process before importing workspace src")
    package = types.ModuleType("src")
    package.__path__ = [str(repo / "src")]
    sys.modules["src"] = package
    sys.path.insert(0, str(repo))


def value_stats(array):
    return {"min": float(array.min()), "max": float(array.max()), "mean": float(array.mean())}


def difference(reference, candidate, mask=None):
    if mask is not None:
        reference, candidate = reference[mask], candidate[mask]
    error = np.abs(candidate.astype(np.float64) - reference.astype(np.float64))
    return {"mae_0_1": float(error.mean()), "max_abs_0_1": float(error.max()),
            "rmse_0_1": float(np.sqrt(np.square(error).mean()))}


def overview(output_dir):
    columns = [("source.png", "Source / synthetic AgX RGB"),
               ("target.png", "Target / Cycles lighting"),
               ("relit.png", "PIXLRelight / experimental")]
    columns = [(name, label) for name, label in columns if (output_dir / name).is_file()]
    image = Image.new("RGB", (512 * len(columns), 550), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("/root/.cache/cad_scene_figures/SourceHanSansSC-Regular.otf", 20)
    for index, (name, label) in enumerate(columns):
        image.paste(Image.open(output_dir / name).convert("RGB"), (index * 512, 38))
        draw.text((index * 512 + 10, 8), label, font=font, fill="black")
    image.save(output_dir / "comparison.png")


def prepare_inputs(input_dir):
    from src.utils import load_fn

    source_path = input_dir / "source.png"
    with Image.open(source_path) as image:
        if image.size != (512, 512) or image.mode != "RGB":
            raise ValueError(f"This pilot requires 512x512 RGB source.png, got {image.size}/{image.mode}")
    source, shapes = load_fn.load_and_preprocess_images([str(source_path)], "pad", 512, 16)
    native = load_fn.load_native_resolution_image(source_path)
    if not torch.equal(source, native):
        raise AssertionError("512x512 source preprocessing must preserve every decoded RGB value")
    arrays = load_fn._resolve_cycles_arrays(input_dir)
    for name, array in arrays.items():
        if array.shape != (512, 512, 3) or not np.isfinite(array).all():
            raise ValueError(f"Invalid Cycles pass: {name}, {array.shape}")
    shading = np.maximum(arrays["diffuse_direct"] + arrays["diffuse_indirect"], 0).astype(np.float32)
    residual = np.maximum(load_fn._compute_residual_hdr(arrays), 0)
    p98_shading, p98_residual = float(np.quantile(shading, .98)), float(np.quantile(residual, .98))
    cutoff = max(p98_shading, p98_residual, 1e-8)
    albedo, shade, resid = load_fn.load_cycles_as_intrinsics(
        [arrays], target_hw=(512, 512), inpaint=False, percentile=98.0)
    intrinsics = torch.cat([albedo, shade, resid], dim=1)
    if not torch.isfinite(intrinsics).all() or intrinsics.shape != (1, 9, 512, 512):
        raise ValueError("Invalid normalized intrinsics")
    report = {
        "source_shape": list(source.shape), "intrinsics_shape": list(intrinsics.shape),
        "source_decoding": "Official PIL RGB / 255, no gamma conversion; source AgX appearance retained",
        "preprocess_mode": "pad", "target_size": 512, "patch_size": 16,
        "source_same_size_resize_values_unchanged": True, "actual_crop": False, "actual_padding": False,
        "inpaint": False, "passes": {name: value_stats(array) for name, array in arrays.items()},
        "albedo_formula": "clip(diffuse_color, 0, 1); metallic zero diffuse is retained without filling",
        "shading_formula": "max(diffuse_direct + diffuse_indirect, 0)",
        "residual_formula": "max(glossy_color*(glossy_direct+glossy_indirect) + optional transmission/volume/emission/environment, 0)",
        "normalization": {"percentile": 98, "p98_shading": p98_shading, "p98_residual": p98_residual,
                          "shared_cutoff": cutoff, "formula": "clip(S or R,0,cutoff)/cutoff",
                          "quantile_domain": "all spatial pixels and RGB channels, separately for S and R",
                          "shading_fraction_above_cutoff": float(np.mean(shading > cutoff)),
                          "residual_fraction_above_cutoff": float(np.mean(residual > cutoff)),
                          "uniform_lighting_scale_removed": True},
    }
    return source, intrinsics, report, sorted(arrays)


def load_model(repo_dir, model_dir):
    from src.models.net import Net
    from src.models.head import DPTHead
    from src.models.pixl_relight import PIXLRelight

    start = time.monotonic()
    config_path, weight_path = model_dir / "config.yaml", model_dir / "model.safetensors"
    config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)["model"]
    net_config = {key: value for key, value in config["net"].items() if key != "__object__"}
    head_config = {key: value for key, value in config["head"].items() if key != "__object__"}
    model = PIXLRelight(Net(**net_config), DPTHead(**head_config), rgbx=torch.nn.Identity())
    expected = model.state_dict()
    with safe_open(weight_path, framework="pt", device="cpu") as weights:
        keys = set(weights.keys())
        excluded = sorted(key for key in keys if key.startswith("rgbx."))
        retained = keys - set(excluded)
        missing, unexpected = sorted(set(expected) - retained), sorted(retained - set(expected))
        shapes = {key: {"expected": list(expected[key].shape), "checkpoint": weights.get_slice(key).get_shape()}
                  for key in retained & set(expected)
                  if list(expected[key].shape) != weights.get_slice(key).get_shape()}
        if missing or unexpected or shapes:
            raise RuntimeError(json.dumps({"missing": missing, "unexpected": unexpected, "shape_mismatch": shapes}))
        state = {key: weights.get_tensor(key) for key in sorted(retained)}
        model.load_state_dict(state, strict=True)
    del state, expected
    gc.collect()
    model.eval().requires_grad_(False)
    if any(parameter.device.type != "cpu" or parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise AssertionError("All model parameters must remain CPU float32")
    if any(name.startswith("src.rgbx") for name in sys.modules):
        raise AssertionError("The Marigold module must not be imported")
    report = {
        "checkpoint_sha256": sha256(weight_path), "checkpoint_bytes": weight_path.stat().st_size,
        "config_sha256": sha256(config_path), "strict_load": True,
        "checkpoint_tensor_count": len(keys), "loaded_tensor_count": len(retained),
        "excluded_prefix": "rgbx.", "excluded_keys": excluded,
        "missing_keys": missing, "unexpected_keys": unexpected, "shape_mismatches": shapes,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "rgbx_replacement": "torch.nn.Identity; never called by the Cycles branch",
        "marigold_imported": False, "network_downloads": False,
        "intrinsics_patch_embed_actual_class": type(model.net.intrinsics_patch_embed).__name__,
        "intrinsics_patch_embed_config": net_config["intrinsics_patch_embed"],
        "rope_dtype_from_official_config": net_config["pos_embed_rope_dtype"],
        "model_parameter_dtype": "float32", "load_seconds": time.monotonic() - start,
    }
    return model, report


def run(args):
    start = time.monotonic()
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(8)
    torch.manual_seed(0)
    install_official_namespace(args.repo_dir)
    source, intrinsics, preprocessing, pass_names = prepare_inputs(args.input_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(args.input_dir / "source.png", args.output_dir / "source.png")
    for name in ("target.png", "mask.png"):
        if (args.input_dir / name).is_file():
            with Image.open(args.input_dir / name) as image:
                if image.size != (512, 512):
                    raise ValueError(f"Unexpected {name} dimensions")
            shutil.copy2(args.input_dir / name, args.output_dir / name)
    inputs = ["source.png"] + [name + ".exr" for name in pass_names]
    inputs += [name for name in ("target.png", "mask.png", "metadata.json") if (args.input_dir / name).is_file()]
    report = {
        "status": "inputs_validated", "input_dir": str(args.input_dir), "output_dir": str(args.output_dir),
        "experiment": "Synthetic-source OOD relighting exploration; not a validated CG-to-photo enhancer",
        "identity_expectation": "Same-light input should approximately preserve source appearance; any drift is measured, not assumed harmless",
        "source_files_sha256": {name: sha256(args.input_dir / name) for name in inputs},
        "script_sha256": sha256(__file__), "repo_dir": str(args.repo_dir),
        "repo_commit": git_output(args.repo_dir, "rev-parse", "HEAD"),
        "repo_status_porcelain": git_output(args.repo_dir, "status", "--porcelain"),
        "licenses": {"repository_declared": "MIT", "model_weights": "CC-BY-NC-4.0",
                     "third_party_notice": "Some src/layers files retain Meta DINOv3 License Agreement headers; retained unchanged",
                     "repo_license_sha256": sha256(args.repo_dir / "LICENSE")},
        "runtime": {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
                    "device": "cpu", "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
                    "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
                    "opencv_threads": cv2.getNumThreads(), "seed": 0,
                    "versions": {name: importlib.metadata.version(name) for name in
                                 ("torch", "torchvision", "numpy", "omegaconf", "addict", "safetensors")}},
        "preprocessing": preprocessing,
    }
    write_json(args.output_dir / "report.json", report)
    if args.validate_only:
        overview(args.output_dir)
        print(json.dumps({"status": "inputs_validated_no_model_run", "report": str(args.output_dir / "report.json")}), flush=True)
        return
    model, loading = load_model(args.repo_dir, args.model_dir)
    report.update(status="weights_loaded", model=loading)
    write_json(args.output_dir / "report.json", report)
    print(json.dumps({"status": "weights_loaded", "model": loading}), flush=True)
    inference_start = time.monotonic()
    with torch.inference_mode():
        output = model(source, intrinsics)
    inference_seconds = time.monotonic() - inference_start
    if set(output) != {"rgb", "rgb_gain", "rgb_bias"}:
        raise RuntimeError(f"Unexpected model outputs: {list(output)}")
    gain, bias = output["rgb_gain"], output["rgb_bias"]
    raw_prediction = gain * source + bias
    prediction = raw_prediction.clamp(0, 1)
    if prediction.shape != source.shape or not torch.isfinite(raw_prediction).all():
        raise RuntimeError("Model output must be finite and 512x512")
    if not torch.equal(prediction, output["rgb"]):
        raise AssertionError("Native-resolution gain/bias application differs from official head")
    pred = prediction[0].numpy().transpose(1, 2, 0)
    src = source[0].numpy().transpose(1, 2, 0)
    Image.fromarray((pred * 255).astype(np.uint8)).save(args.output_dir / "relit.png")
    np.save(args.output_dir / "prediction_float32.npy", pred)
    np.savez_compressed(args.output_dir / "modulation.npz", gain=gain[0].numpy(), bias=bias[0].numpy())
    overview(args.output_dir)
    metrics = {"prediction_vs_source": difference(src, pred)}
    if (args.output_dir / "mask.png").is_file():
        mask = np.asarray(Image.open(args.output_dir / "mask.png").convert("L")) > 127
        if mask.any():
            metrics["prediction_vs_source_target_mask"] = difference(src, pred, mask)
    if (args.output_dir / "target.png").is_file():
        target = np.asarray(Image.open(args.output_dir / "target.png").convert("RGB"), dtype=np.float32) / 255
        metrics["prediction_vs_rendered_target"] = difference(target, pred)
    report.update(status="completed", inference_seconds=inference_seconds,
                  total_seconds=time.monotonic() - start,
                  peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                  final_rss_bytes=psutil.Process().memory_info().rss,
                  metrics=metrics, metrics_note="Pixel differences describe change, not photorealism or geometry correctness",
                  output={"shape": list(pred.shape), "quantization": "Official uint8 floor(pred*255)",
                          "gain": value_stats(gain.numpy()), "bias": value_stats(bias.numpy()),
                          "fraction_below_zero_before_clip": float((raw_prediction < 0).float().mean()),
                          "fraction_above_one_before_clip": float((raw_prediction > 1).float().mean()),
                          "crop": False, "inpaint": False, "spatial_warp": False,
                          "limitation": "Per-pixel gain/bias can alter or erase visible boundary evidence despite fixed pixel coordinates"})
    write_json(args.output_dir / "report.json", report)
    print(json.dumps({"status": "completed", "inference_seconds": inference_seconds,
                      "peak_rss_gb": report["peak_rss_bytes"] / 1e9, "metrics": metrics}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("input_dir", "output_dir", "repo_dir", "model_dir"):
        parser.add_argument("--" + argument, type=Path, required=True)
    parser.add_argument("--validate_only", action="store_true", help="Validate input preprocessing without constructing a model")
    parsed = parser.parse_args()
    for argument in ("input_dir", "output_dir", "repo_dir", "model_dir"):
        setattr(parsed, argument, getattr(parsed, argument).resolve())
    run(parsed)
