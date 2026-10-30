"""Offline scene-linear light-group mixing; run with Blender 5.2.2 on CPU.

Example (the output directory must not exist)::

    blender --background --factory-startup --threads 8 --python-exit-code 1 \
      --python compose_lights.py -- --input_dir /export/office/00301161 \
      --output_dir /comparison/office/00301161

The optional --presets JSON replaces the two exploratory presets, but always
retains the all_one reconstruction. Format::

    {"presets": {"daylight_priority": {
        "description": "Reduce artificial lights to one quarter",
        "weights": {"ceiling": 0.25, "task": 0.25, "other_lights": 0.25}
    }}}

Unspecified groups explicitly default to 1 and resolved weights are recorded.
All weights must be finite, nonnegative scalar multipliers. There is no LUT,
automatic exposure, normalization, per-group denoising, or geometric operation.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import sys
import time

import bpy
import numpy as np


DEFAULT_PRESETS = {
    "daylight_priority": {
        "description": "窗光、太阳及世界不变；顶灯、工作灯和其余灯降至 0.25；其他自发光保持 1。",
        "weights": {"ceiling": 0.25, "task": 0.25, "other_lights": 0.25},
    },
    "indoor_priority": {
        "description": "顶灯、工作灯、其余灯和自发光不变；太阳、窗光与世界降至 0.25。",
        "weights": {"sun": 0.25, "daylight": 0.25, "world": 0.25},
    },
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def pixels(image):
    values = np.empty(len(image.pixels), dtype=np.float32)
    image.pixels.foreach_get(values)
    return values.reshape(image.size[1], image.size[0], image.channels)


def load_image(path, data=False):
    image = bpy.data.images.load(str(path), check_existing=False)
    # Guide values are data. Radiance EXRs retain their detected linear space.
    if data:
        image.colorspace_settings.name = "Non-Color"
    values = pixels(image)
    if values.shape[-1] != 4 or not np.isfinite(values).all():
        raise ValueError(f"Expected finite RGBA image: {path}, shape={values.shape}")
    if not data and not image.colorspace_settings.is_data:
        if image.colorspace_settings.name != "Linear Rec.709":
            raise ValueError(f"Unexpected radiance colorspace: {image.colorspace_settings.name}: {path}")
    return image, values


def error_stats(reference, candidate, mask=None):
    ref = np.asarray(reference, dtype=np.float64)
    cand = np.asarray(candidate, dtype=np.float64)
    if ref.shape != cand.shape:
        raise ValueError(f"Comparison shape mismatch: {ref.shape} != {cand.shape}")
    if mask is not None:
        ref, cand = ref[mask], cand[mask]
    if not ref.size:
        return {"pixel_count": 0}
    error = np.abs(cand - ref)
    return {
        "mean_abs_error": float(error.mean()),
        "max_abs_error": float(error.max()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "relative_l1": float(error.sum() / max(np.abs(ref).sum(), 1e-12)),
        "changed_pixels": int(np.count_nonzero(np.any(error > 0, axis=-1))),
        "pixel_count": int(ref.size // ref.shape[-1]),
    }


def png_pixels(path):
    image = bpy.data.images.load(str(path), check_existing=False)
    # Eight-bit PNG pixels are returned in their encoded space, not scene-linear.
    rgb = np.rint(pixels(image)[..., :3] * 255).astype(np.int16)
    bpy.data.images.remove(image)
    return rgb


def color_settings(scene):
    view = scene.view_settings
    names = ("view_transform", "look", "exposure", "gamma", "use_curve_mapping",
             "use_white_balance", "white_balance_temperature", "white_balance_tint",
             "white_balance_whitepoint", "is_hdr", "support_emulation")
    result = {name: getattr(view, name) for name in names}
    result["white_balance_whitepoint"] = list(result["white_balance_whitepoint"])
    result["display_device"] = scene.display_settings.display_device
    result["dither_intensity"] = scene.render.dither_intensity
    if view.use_curve_mapping:
        result["curves"] = [[[point.location[0], point.location[1]] for point in curve.points]
                            for curve in view.curve_mapping.curves]
    return result


def resolved_presets(groups, path):
    custom = json.loads(path.read_text())["presets"] if path else DEFAULT_PRESETS
    all_one = {"description": "所有光源组权重为 1；线性重建对照。", "weights": {}}
    if not isinstance(custom, dict) or "all_one" in custom or "baseline_combined" in custom:
        raise ValueError("presets must be an object; all_one and baseline_combined are reserved")
    result = {}
    for name, spec in {"all_one": all_one, **custom}.items():
        if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
            raise ValueError(f"Unsafe preset name: {name}")
        weights = spec["weights"]
        if not isinstance(weights, dict) or set(weights) - set(groups):
            raise ValueError(f"Unknown groups in {name}: {weights}")
        values = {group: float(weights.get(group, 1.0)) for group in groups}
        if not all(math.isfinite(value) and value >= 0 for value in values.values()):
            raise ValueError(f"Weights must be finite and nonnegative: {name}")
        result[name] = {"description": spec.get("description", ""), "weights": values}
    return result


def float_image(name, rgba):
    image = bpy.data.images.new(name, width=rgba.shape[1], height=rgba.shape[0],
                                alpha=True, float_buffer=True)
    image.colorspace_settings.name = "Linear Rec.709"
    image.pixels.foreach_set(np.asarray(rgba, dtype=np.float32).ravel())
    image.update()
    return image


def image_node(tree, image):
    node = tree.nodes.new("CompositorNodeImage")
    node.image = image
    return node.outputs["Image"]


def file_output(tree, socket, folder, name, extension):
    output = tree.nodes.new("CompositorNodeOutputFile")
    output.directory = str(folder)
    output.file_name = name + "_"
    output.format.media_type = "IMAGE"
    output.format.file_format = "OPEN_EXR" if extension == "exr" else "PNG"
    output.format.color_mode = "RGBA" if extension == "exr" else "RGB"
    output.format.color_depth = "32" if extension == "exr" else "8"
    output.format.color_management = "FOLLOW_SCENE"
    if extension == "exr":
        output.format.exr_codec = "ZIP"
    output.save_as_render = extension == "png"
    output.file_output_items.new("RGBA", "Image")
    tree.links.new(socket, output.inputs[0])


def run(input_dir, output_dir, presets_file):
    start = time.monotonic()
    metadata_path = input_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    if metadata["status"] != "completed":
        raise ValueError("Light export must have completed before composition")
    groups = metadata["light_groups"]["groups"]
    if not groups or len(groups) != len(set(groups)):
        raise ValueError("Light group names must be nonempty and unique")
    presets = resolved_presets(groups, presets_file)
    source_scene = Path(metadata["source_scene"])
    if sha256(source_scene) != metadata["source_scene_sha256"]:
        raise ValueError("Source scene has changed since export")
    request_path = Path(metadata["source_request"])
    if sha256(request_path) != metadata["source_request_sha256"]:
        raise ValueError("Source request has changed since export")
    request = json.loads(request_path.read_text())
    original_rgb = (Path(request["output_dir"]) / "raw" / metadata["scene_id"] /
                    metadata["model_id"] / "rgb.png")
    if metadata["original_settings"]["compositor"] is not None:
        raise ValueError("This comparison requires an original scene without extra compositor effects")

    bpy.ops.wm.open_mainfile(filepath=str(source_scene))
    scene = bpy.context.scene
    original_color = color_settings(scene)
    for key, value in metadata["original_settings"]["color_management"].items():
        if original_color[key] != value:
            raise ValueError(f"Source color setting differs from export: {key}")
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.denoising_use_gpu = False
    scene.render.threads_mode = "FIXED"
    scene.render.threads = 8
    scene.render.compositor_device = "CPU"
    scene.render.compositor_denoise_device = "CPU"
    scene.render.compositor_precision = "FULL"
    scene.render.compositor_denoise_final_quality = "HIGH"
    scene.render.use_compositing = True
    scene.render.use_sequencer = False
    scene.render.use_border = scene.render.use_multiview = False
    scene.render.resolution_percentage = 100
    scene.render.pixel_aspect_x = scene.render.pixel_aspect_y = 1
    # With no Render Layers nodes, render.render evaluates only the compositor.
    # Clearing the camera also makes accidental 3D rendering fail explicitly.
    scene.camera = None

    images, arrays = {}, {}
    inputs = ["combined_noisy", "combined_denoised", "denoising_normal", "denoising_albedo"]
    inputs += ["light_" + group for group in groups]
    for name in inputs:
        images[name], arrays[name] = load_image(input_dir / (name + ".exr"),
                                               data=name.startswith("denoising_"))
    reference = arrays["combined_noisy"]
    if any(array.shape != reference.shape for array in arrays.values()):
        raise ValueError("EXR dimensions differ")
    scene.render.resolution_y, scene.render.resolution_x = reference.shape[:2]
    total = sum(arrays["light_" + group][..., :3].astype(np.float64) for group in groups)
    sum_check = error_stats(reference[..., :3], total)
    # Verify the exported partition before any nonlinear denoising/display step.
    if not np.allclose(total, reference[..., :3], rtol=2e-5, atol=2e-6):
        raise ValueError(f"Light groups do not reconstruct the noisy Combined: {sum_check}")
    mask_image = bpy.data.images.load(str(input_dir / "mask.png"), check_existing=False)
    target_mask = pixels(mask_image)[..., 0] > 0.5
    bpy.data.images.remove(mask_image)
    if target_mask.shape != reference.shape[:2] or not target_mask.any():
        raise ValueError("Target mask must match the EXRs and contain target pixels")
    luminance_coefficients = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)
    group_luminance = {}
    for group in groups:
        luminance = arrays["light_" + group][..., :3] @ luminance_coefficients
        group_luminance[group] = {"scene_mean": float(luminance.mean()),
                                  "target_mask_mean": float(luminance[target_mask].mean())}

    output_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(input_dir / "original.png", output_dir / "original_cycles.png")
    shutil.copy2(original_rgb, output_dir / "source_rgb.png")
    shutil.copy2(input_dir / "mask.png", output_dir / "mask.png")
    tree = bpy.data.node_groups.new("offline_light_mix", "CompositorNodeTree")
    scene.compositing_node_group = tree
    tree.interface.new_socket(name="Image", in_out="OUTPUT", socket_type="NodeSocketColor")
    final = tree.nodes.new("NodeGroupOutput")
    normal = image_node(tree, images["denoising_normal"])
    albedo = image_node(tree, images["denoising_albedo"])
    cases = {"baseline_combined": images["combined_noisy"]}
    for name, preset in presets.items():
        rgba = reference.copy()
        rgba[..., :3] = sum(arrays["light_" + group][..., :3].astype(np.float64) * weight
                           for group, weight in preset["weights"].items())
        if not np.isfinite(rgba).all():
            raise ValueError(f"Nonfinite weighted radiance: {name}")
        cases[name] = float_image(name, rgba)
    for name, image in cases.items():
        noisy = image_node(tree, image)
        denoise = tree.nodes.new("CompositorNodeDenoise")
        denoise.inputs["HDR"].default_value = True
        denoise.inputs["Prefilter"].default_value = "Accurate"
        denoise.inputs["Quality"].default_value = "High"
        tree.links.new(noisy, denoise.inputs["Image"])
        tree.links.new(albedo, denoise.inputs["Albedo"])
        tree.links.new(normal, denoise.inputs["Normal"])
        for suffix, socket in (("noisy", noisy), ("oidn", denoise.outputs["Image"])):
            for extension in ("exr", "png"):
                file_output(tree, socket, output_dir, name + "_" + suffix, extension)
        if name == "baseline_combined":
            tree.links.new(denoise.outputs["Image"], final.inputs["Image"])
    # A display-transformed view of Cycles' own output isolates export/color issues.
    file_output(tree, image_node(tree, images["combined_denoised"]),
                output_dir, "cycles_linear_display", "png")
    if any(node.bl_idname == "CompositorNodeRLayers" for node in tree.nodes):
        raise AssertionError("Offline compositor must never contain Render Layers")
    report = {
        "status": "compositing", "scene_id": metadata["scene_id"], "model_id": metadata["model_id"],
        "input_dir": str(input_dir), "source_request": str(request_path), "source_rgb": str(original_rgb),
        "export_metadata_sha256": sha256(metadata_path), "script_sha256": sha256(__file__),
        "source_scene_sha256": metadata["source_scene_sha256"], "blender": bpy.app.version_string,
        "input_sha256": {name: sha256(input_dir / (name + ".exr")) for name in inputs},
        "original_png_sha256": sha256(input_dir / "original.png"),
        "source_rgb_sha256": sha256(original_rgb), "mask_sha256": sha256(input_dir / "mask.png"),
        "device": "CPU", "threads": 8, "three_d_render": False,
        "equation": "I_linear = sum_g(a_g * L_g); preserve Combined alpha; no clipping or normalization",
        "source_transport_settings": {
            key: getattr(scene.cycles, key) for key in ("sample_clamp_direct", "sample_clamp_indirect",
                                                       "light_sampling_threshold", "blur_glossy")},
        "physical_equivalence_limit": "The all-one check validates the exported decomposition. "
                                      "Path clamping and sampling thresholds mean changed weights are "
                                      "not guaranteed equivalent to retracing changed light powers.",
        "color_management": original_color, "presets": presets,
        "denoising": {
            "engine": "OpenImageDenoise via CompositorNodeDenoise", "hdr": True,
            "prefilter": "Accurate", "quality": "High", "device": "CPU",
            "guides": ["denoising_normal.exr", "denoising_albedo.exr"],
            "applied_after_mixing": True, "all_cases_same_node_settings_and_guides": True,
            "specular_albedo_used": False,
            "cycles_equivalence": "Not assumed: Blender 5.2.2 compositor exposes no separate Specular Albedo "
                                  "input. Original Cycles denoising and compositor denoising may differ. "
                                  "baseline_combined and every mix use the same compositor path.",
        },
        "sum_check_scene_linear_rgb": sum_check,
        "group_radiance_mean_luminance": group_luminance,
        "luminance_definition": "Scene-linear Rec.709 Y = 0.2126 R + 0.7152 G + 0.0722 B; "
                                "arbitrary renderer units, before exposure/AgX; mask threshold > 0.5",
        "target_mask_pixels": int(target_mask.sum()),
        "sum_tolerance": {"rtol": 2e-5, "atol": 2e-6},
        "outputs": {name: {suffix + "_" + ext: name + "_" + suffix + "." + ext
                            for suffix in ("noisy", "oidn") for ext in ("png", "exr")}
                    for name in cases},
        "selection": "Exploratory physical source ratios; no automatic quality ranking or final selection",
    }
    if presets_file:
        report["presets_file"] = str(presets_file)
        report["presets_sha256"] = sha256(presets_file)
    write_json(output_dir / "report.json", report)
    bpy.ops.render.render(write_still=False)
    for name, files in report["outputs"].items():
        for filename in files.values():
            destination = output_dir / filename
            candidates = list(output_dir.glob(destination.stem + "_*" + destination.suffix))
            if len(candidates) != 1:
                raise RuntimeError(f"Expected one compositor output for {destination}: {candidates}")
            candidates[0].rename(destination)
    candidates = list(output_dir.glob("cycles_linear_display_*.png"))
    if len(candidates) != 1:
        raise RuntimeError("Missing Cycles display output")
    candidates[0].rename(output_dir / "cycles_linear_display.png")

    denoised = {}
    for name in cases:
        image, values = load_image(output_dir / (name + "_oidn.exr"))
        denoised[name] = values[..., :3]
        bpy.data.images.remove(image)
    original_png = png_pixels(output_dir / "original_cycles.png")
    baseline_png = png_pixels(output_dir / "baseline_combined_oidn.png")
    report["comparisons"] = {
        "linear_all_one_oidn_vs_baseline_oidn": error_stats(denoised["baseline_combined"], denoised["all_one"]),
        "linear_compositor_baseline_vs_cycles": error_stats(arrays["combined_denoised"][..., :3], denoised["baseline_combined"]),
        "png_8bit_compositor_baseline_vs_cycles": error_stats(original_png, baseline_png),
        "png_8bit_all_one_vs_baseline": error_stats(baseline_png, png_pixels(output_dir / "all_one_oidn.png")),
        "png_8bit_cycles_exr_display_vs_original": error_stats(original_png, png_pixels(output_dir / "cycles_linear_display.png")),
        "png_8bit_original_vs_historical_source": error_stats(png_pixels(output_dir / "source_rgb.png"), original_png),
    }
    if color_settings(scene) != original_color:
        raise AssertionError("Color management changed during mixing")
    report.update(status="completed", elapsed_seconds=time.monotonic() - start,
                  source_scene_unchanged=sha256(source_scene) == metadata["source_scene_sha256"])
    write_json(output_dir / "report.json", report)
    print(json.dumps({key: report[key] for key in ("status", "scene_id", "model_id", "elapsed_seconds",
                                                  "sum_check_scene_linear_rgb", "comparisons")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--presets", type=Path)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    run(args.input_dir.resolve(), args.output_dir.resolve(), args.presets.resolve() if args.presets else None)
