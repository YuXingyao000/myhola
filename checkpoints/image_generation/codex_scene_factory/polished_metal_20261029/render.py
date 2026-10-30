"""固定机位与姿态的材质消融；在Blender内运行，仅使用CPU。"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import bpy
from mathutils import Matrix
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from material import make_material


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def signature(part, camera):
    mesh = part.data
    arrays = {
        "vertices": [v.co[:] for v in mesh.vertices],
        "triangles": [p.vertices[:] for p in mesh.polygons],
        "corner_normals": [n.vector[:] for n in mesh.corner_normals],
        "smooth": [p.use_smooth for p in mesh.polygons],
        "sharp": [e.use_edge_sharp for e in mesh.edges],
        "model_matrix": part.matrix_world,
        "camera_matrix": camera.matrix_world,
    }
    return {name: hashlib.sha256(np.asarray(values).tobytes()).hexdigest()
            for name, values in arrays.items()}


def pixels(path):
    image = bpy.data.images.load(str(path), check_existing=False)
    data = np.asarray(image.pixels[:], dtype=np.float32).reshape(image.size[1], image.size[0], 4)
    bpy.data.images.remove(image)
    return data


def render_scene(request, scene_id, model_ids):
    source = json.loads(Path(request["source_request"]).read_text())
    pipeline = Path(source["source_package"])
    sys.path[:0] = [str(pipeline / "step3"), str(pipeline / "step2"), str(pipeline)]
    from part import load_part, remove_part
    from preview import configure_render, make_camera
    from capture import add_geometry_outputs, take_photo

    old_scene = Path(source["output_dir"]) / "raw" / scene_id
    for model_id in model_ids:
        prior = json.loads((old_scene / model_id / "metadata.json").read_text())
        bpy.ops.wm.open_mainfile(filepath=str(old_scene / "prepared_scene.blend"))
        config = dict(source["render_config"], samples=request["samples"], threads=8)
        configure_render(config)
        scene = bpy.context.scene
        scene.cycles.use_denoising = True
        scene.cycles.use_adaptive_sampling = False
        camera = make_camera({"camera_position_world": [0, 0, 1], "look_at_world": [0, 0, 0]}, config)
        camera.matrix_world = Matrix(prior["camera"]["camera_to_world_blender"])
        part = load_part(str(Path(source["stl_dir"]) / (model_id + ".stl")), 30)
        part.matrix_world = Matrix(prior["geometry"]["model_to_world"])
        outputs = add_geometry_outputs()
        part.pass_index = 1
        bpy.context.view_layer.update()
        fixed = signature(part, camera)
        reference = None

        for name, settings in request["profiles"].items():
            folder = Path(request["output_dir"]) / scene_id / model_id / name
            folder.mkdir(parents=True, exist_ok=False)
            profile = dict(settings, name=name)
            material = make_material(part, prior["material"], profile)
            part.data.materials.clear()
            part.data.materials.append(material)
            assert signature(part, camera) == fixed
            started = time.monotonic()
            images, mask = take_photo(folder, config["seed"], outputs)
            current = {key: pixels(images[key]) for key in ("mask", "depth", "rgb")}
            assert all(np.isfinite(value).all() for value in current.values())
            if reference is None:
                reference = current
            checks = {
                "signature_unchanged": signature(part, camera) == fixed,
                "mask_differing_pixels": int(np.count_nonzero(current["mask"][:, :, 0] != reference["mask"][:, :, 0])),
                "depth_max_abs_error_m": float(np.max(np.abs(current["depth"] - reference["depth"]))),
                "old_mask_differing_pixels": int(np.count_nonzero(current["mask"][:, :, 0] != pixels(old_scene / model_id / "mask.png")[:, :, 0])),
            }
            target = reference["mask"][:, :, 0] > .5
            passed = checks["signature_unchanged"] and checks["mask_differing_pixels"] == 0 and checks["depth_max_abs_error_m"] == 0
            record = {"scene": scene_id, "model": model_id, "profile": profile,
                      "material_base": prior["material"], "images": images, "mask": mask,
                      "checks": checks, "geometry_signature": fixed,
                      "target_mean_abs_saved_png_rgb_difference": float(np.abs(current["rgb"][:, :, :3] - reference["rgb"][:, :, :3])[target].mean()),
                      "render_seconds": time.monotonic() - started,
                      "samples": config["samples"], "seed": config["seed"], "device": "CPU",
                      "blender": bpy.app.version_string, "status": "completed" if passed else "failed"}
            write_json(folder / "metadata.json", record)
            assert passed, checks
            print("CASE_COMPLETE", scene_id, model_id, name, flush=True)
        remove_part(part)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--models", nargs="+", required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    render_scene(json.loads(args.request.read_text()), args.scene, args.models)
