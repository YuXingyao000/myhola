"""同一线性EXR的AgX look对照；没有三维渲染或图像空间滤波。"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import bpy
import numpy as np


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def png_pixels(path):
    image = bpy.data.images.load(str(path), check_existing=False)
    rgb = np.rint(np.asarray(image.pixels[:]).reshape(image.size[1], image.size[0], 4)[:, :, :3] * 255).astype(np.int16)
    bpy.data.images.remove(image)
    return rgb


def save_display_image(scene, image, folder, name):
    """保持原实验的Compositor文件输出路径，避免不同保存API的量化差异。"""
    scene.camera = None
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.render.threads_mode = "FIXED"
    scene.render.threads = 8
    scene.render.compositor_device = "CPU"
    scene.render.use_compositing = True
    scene.render.use_sequencer = False
    scene.render.resolution_x, scene.render.resolution_y = image.size
    scene.render.resolution_percentage = 100
    tree = bpy.data.node_groups.new("display_comparison", "CompositorNodeTree")
    scene.compositing_node_group = tree
    tree.interface.new_socket(name="Image", in_out="OUTPUT", socket_type="NodeSocketColor")
    node = tree.nodes.new("CompositorNodeImage")
    node.image = image
    output = tree.nodes.new("NodeGroupOutput")
    tree.links.new(node.outputs["Image"], output.inputs["Image"])
    file = tree.nodes.new("CompositorNodeOutputFile")
    file.directory, file.file_name = str(folder), name + "_"
    file.format.media_type = "IMAGE"
    file.format.file_format = "PNG"
    file.format.color_mode = "RGB"
    file.format.color_depth = "8"
    file.format.color_management = "FOLLOW_SCENE"
    file.save_as_render = True
    file.file_output_items.new("RGBA", "Image")
    tree.links.new(node.outputs["Image"], file.inputs[0])
    # 没有Render Layers且camera为空，这里只求值compositor，不渲染三维场景。
    bpy.ops.render.render()
    path, = folder.glob(name + "_*.png")
    path.rename(folder / (name + ".png"))


def run(spec):
    output = Path(spec["Outputs"]["root"])
    output.mkdir(parents=True, exist_ok=False)
    root = Path(spec["Inputs"]["root"])
    for scene_id in spec["Inputs"]["scenes"]:
        source = root / scene_id / spec["Inputs"]["model"]
        prior = json.loads((source / "metadata.json").read_text())
        bpy.ops.wm.open_mainfile(filepath=prior["source_scene"])
        scene = bpy.context.scene
        settings = scene.render.image_settings
        settings.file_format = "PNG"
        settings.color_mode = "RGB"
        settings.color_depth = "8"
        settings.color_management = "FOLLOW_SCENE"
        color = scene.view_settings
        assert color.view_transform == "AgX" and color.look == "AgX - Medium High Contrast"
        folder = output / scene_id
        folder.mkdir()
        source_image = source / spec["Inputs"]["image"]
        image = bpy.data.images.load(str(source_image), check_existing=False)
        assert image.colorspace_settings.name == "Linear Rec.709"
        assert np.isfinite(np.asarray(image.pixels[:])).all()
        config = {key: getattr(color, key) for key in ["view_transform", "look", "exposure", "gamma", "use_white_balance", "white_balance_temperature", "white_balance_tint"]}
        save_display_image(scene, image, folder, "original")
        color.look = "AgX - Base Contrast"
        save_display_image(scene, image, folder, "base_contrast")
        bpy.data.images.remove(image)
        old = png_pixels(source / "original.png")
        original = png_pixels(folder / "original.png")
        difference = np.abs(original - old)
        record = {"scene": scene_id, "source": str(source_image), "source_sha256": sha256(source_image),
                  "source_scene": prior["source_scene"], "source_scene_sha256": sha256(prior["source_scene"]),
                  "script_sha256": sha256(__file__), "original_color": config,
                  "candidate_look": "AgX - Base Contrast", "blender": bpy.app.version_string,
                  "reconstruction": {"mean_abs_rgb_255": float(difference.mean()), "max_abs_rgb_255": int(difference.max()),
                                     "different_pixels": int(np.any(difference != 0, axis=2).sum())}}
        (folder / "metadata.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(record["reconstruction"]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    run(json.loads(args.spec.read_text()))
