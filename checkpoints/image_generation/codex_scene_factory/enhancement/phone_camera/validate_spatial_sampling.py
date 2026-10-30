"""CPU 验证真实 FlowMatch 时间表与逐区域 mask；不加载模型权重。"""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler, FluxControlNetInpaintPipeline
from diffusers.pipelines.flux.pipeline_flux_controlnet_inpainting import calculate_shift
from PIL import Image

from spatial_sampling import build_spatial_plan, make_callback


def validate_sample(sample, profile):
    original = Image.open(sample["rgb_file"]).convert("RGB")
    target = Image.open(sample["mask_file"]).convert("L")
    plan = build_spatial_plan(original, target, profile["spatial_plan"])
    background = plan["background_cells"]
    interior = plan["interior_cells"]
    assert not (background & interior).any()
    assert (np.asarray(plan["initial_edit_mask"])[np.asarray(target) > 0] == 0).all()
    assert (np.asarray(plan["strict_blend_mask"])[np.asarray(target) > 0] == 0).all()
    assert not (np.asarray(plan["cad_interior_mask"]) > 0)[np.asarray(target) == 0].any()

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        profile["model_dir"], subfolder="scheduler", local_files_only=True,
    )
    steps = profile["num_inference_steps"]
    strength = profile["strengths"][0]
    size = profile["processing_size"]
    scheduler.set_timesteps(
        steps, device="cpu", sigmas=np.linspace(1, 1 / steps, steps),
        mu=calculate_shift((size // 16) ** 2),
    )
    first = steps - int(steps * strength)
    scheduler.set_begin_index(first)
    pipe = SimpleNamespace(
        scheduler=scheduler, num_timesteps=steps - first,
        _pack_latents=FluxControlNetInpaintPipeline._pack_latents,
    )
    callback, log = make_callback(pipe, plan)
    current_mask = pipe._pack_latents(
        torch.tensor(background)[None, None].repeat(1, 16, 1, 1).float(),
        1, 16, size // 8, size // 8,
    )
    latents = torch.zeros_like(current_mask)
    first_open = None
    for index, timestep in enumerate(scheduler.timesteps[first:]):
        expected_open = bool(log and log[-1]["cad_open_next_step"])
        grid = FluxControlNetInpaintPipeline._unpack_latents(current_mask, size, size, 8)
        grid = grid[0, 0].numpy().astype(bool)
        assert not (grid & ~(background | interior)).any()
        assert grid[interior].all() if expected_open else not grid[interior].any()
        if expected_open and interior.any() and first_open is None:
            first_open = index
        latents = scheduler.step(torch.zeros_like(latents), timestep, latents, return_dict=False)[0]
        current_mask = callback(pipe, index, timestep, {"mask": current_mask})["mask"]
        assert log[-1]["cad_open_this_step"] == expected_open

    assert log[0]["cad_open_this_step"] is False
    assert log[-1]["cad_open_next_step"] is False
    assert (first_open is not None) == bool(interior.any())
    return {
        "model_id": sample["model_id"], **plan["summary"],
        "initial_sigma": float(scheduler.sigmas[first]),
        "first_open_step": first_open,
        "open_steps": sum(step["cad_open_this_step"] for step in log),
        "schedule": log,
    }


def main():
    directory = Path(__file__).parent
    profile = json.loads((directory / "profiles.json").read_text())["depth_spatial"]
    samples = json.loads((directory / "samples.json").read_text())
    reports = [validate_sample(sample, profile) for sample in samples]

    thin = np.zeros((512, 512), dtype=np.uint8)
    thin[200:300, 250:252] = 255
    plan = build_spatial_plan(Image.new("RGB", (512, 512)), Image.fromarray(thin), profile["spatial_plan"])
    assert not plan["summary"]["material_edit_available"]
    try:
        build_spatial_plan(Image.new("RGB", (512, 512)), Image.new("L", (512, 512)), profile["spatial_plan"])
    except ValueError:
        pass
    else:
        raise AssertionError("Empty target mask was accepted")

    result = {
        "status": "pass", "device": "cpu", "model_weights_loaded": False,
        "source_sha256": hashlib.sha256((directory / "spatial_sampling.py").read_bytes()).hexdigest(),
        "profile": profile, "sample_count": len(reports), "samples": reports,
        "thin_case": "no_material_cells", "empty_case": "raises_ValueError",
        "limits": "Checks callback timing and mask packing only; no image-quality or geometry-validity conclusion.",
    }
    output = directory / "spatial_sampling_validation.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
