"""在 Blender 内执行模板脚本，将环境、相机和零件材质保存为 .blend。

此阶段不导入 CAD 零件。render.py 随后打开 .blend，负责批量拍照。
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import bpy

# Blender 直接执行本文件时，也能导入同级的相机和几何工具。
sys.path.insert(0, str(Path(__file__).resolve().parent))
from camera import configure_camera
from geometry import PerspectiveCamera


def build(
    template_file_path: Path,
    material_file_path: Path,
    output_file_path: Path,
) -> Path:
    """把两个 Python 模板构建成可供 render.py 打开的场景文件。"""
    template_file_path = template_file_path.resolve()
    material_file_path = material_file_path.resolve()
    output_file_path = output_file_path.resolve()
    template_dir = template_file_path.parent

    # 1. 核对素材文件，再从空场景执行两个构建函数。
    assets = _load_assets(template_dir)
    print(f"已核对 {len(assets)} 个素材文件", flush=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene_script = _load_script(template_file_path)
    material_script = _load_script(material_file_path)
    scene, support = scene_script.build_scene()
    material = material_script.build_material()

    # 材质此时还没赋给零件；fake user 确保它仍能随 .blend 保存和恢复。
    material.use_fake_user = True
    scene["part_material"] = material.name
    scene["support_object"] = support.name

    # 2. 添加外围程序负责的固定相机和输出设定。
    camera = bpy.data.objects.new("capture_camera", bpy.data.cameras.new("capture_camera"))
    scene.collection.objects.link(camera)
    configure_camera(scene, camera, PerspectiveCamera(), near_clip=0.01, far_clip=100.0)
    scene.render.engine = "CYCLES"
    scene.cycles.use_denoising = True
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"

    # 3. 记录来源，打包实际使用的资源；保存后 render.py 只需打开 .blend。
    for key, source in (("scene_script", template_file_path), ("material_script", material_file_path)):
        scene[key] = str(source)
        scene[key + "_sha256"] = _file_sha256(source)
    manifest_path = template_dir / "assets.json"
    manifest_hash = _file_sha256(manifest_path) if manifest_path.is_file() else ""
    scene["assets_manifest_sha256"] = manifest_hash
    scene["template_assets"] = json.dumps(assets, ensure_ascii=False)
    packed_resources = _pack_assets(template_dir, assets, output_file_path.parent)
    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output_file_path))

    # 构建记录与 .blend 同目录。素材来源等额外字段原样保留，不由 Blender 编造。
    report = {
        "blender_version": bpy.app.version_string,
        "blender_build_hash": bpy.app.build_hash.decode(),
        "scene_file": output_file_path.name,
        "template_sources": {
            key: scene[key] for key in (
                "scene_script", "scene_script_sha256", "material_script", "material_script_sha256",
            )
        },
        "assets_manifest_sha256": manifest_hash or None,
        "assets": assets,
        "packed_resources": packed_resources,
    }
    report_path = output_file_path.parent / "build_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"场景已保存：{output_file_path}", flush=True)
    print(f"构建记录：{report_path}", flush=True)
    return output_file_path


def _load_assets(template_dir: Path) -> list[dict]:
    """assets.json 是文件清单；纯程序布景没有外部素材时可省略。"""
    manifest_path = template_dir / "assets.json"
    assets = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else []
    for asset in assets:
        relative_path = Path(asset["path"])
        file_path = (template_dir / relative_path).resolve()
        if relative_path.is_absolute() or not file_path.is_relative_to(template_dir):
            raise ValueError(f"Asset path must stay inside the template: {asset['path']}")
        if _file_sha256(file_path) != asset["sha256"]:
            raise ValueError(f"Asset SHA-256 mismatch: {asset['path']}")
    return assets


def _pack_assets(template_dir: Path, assets: list[dict], blend_dir: Path) -> dict:
    """核对外部资源并内嵌；导入模型应使用 append，避免保留外部库依赖。"""
    linked_data = [item.name for item in bpy.data.user_map() if item.library]
    if linked_data:
        raise ValueError(f"Use append instead of linked libraries: {linked_data}")
    # append 会留下来源库记录；没有实际链接的数据后可移除，避免残留包外引用。
    for library in list(bpy.data.libraries):
        bpy.data.libraries.remove(library)
    images = [image for image in bpy.data.images if image.users]
    if any(image.source in {"SEQUENCE", "MOVIE", "TILED"} for image in images):
        raise ValueError("This template version supports single static images and HDRIs")
    images = [image for image in images if image.source == "FILE"]
    fonts = [font for font in bpy.data.fonts if font.users and font.filepath != "<builtin>"]
    sounds = [sound for sound in bpy.data.sounds if sound.users]
    resources = images + fonts + sounds
    declared_files = {(template_dir / asset["path"]).resolve() for asset in assets}
    source_paths = {}
    for resource in resources:
        # append 进来的模型可能已内嵌贴图；这些数据不再依赖原始文件路径。
        if resource.packed_file:
            continue
        file_path = Path(bpy.path.abspath(resource.filepath)).resolve()
        if file_path not in declared_files:
            raise ValueError(f"External resource is not declared in assets.json: {file_path}")
        source_paths[resource] = file_path
    bpy.ops.file.pack_all()
    for resource in resources:
        if not resource.packed_file:
            raise ValueError(f"Resource was not packed: {resource.name}")
        if resource in source_paths:
            resource.filepath = bpy.path.relpath(str(source_paths[resource]), start=str(blend_dir))
    return {
        "images": [image.name for image in images],
        "fonts": [font.name for font in fonts],
        "sounds": [sound.name for sound in sounds],
    }


def _file_sha256(file_path: Path) -> str:
    return hashlib.sha256(file_path.read_bytes()).hexdigest()


def _load_script(file_path: Path):
    """按文件路径加载模块；注册模块后再执行，兼容脚本内的类定义。"""
    spec = importlib.util.spec_from_file_location(file_path.stem, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene_script", type=Path, required=True)
    parser.add_argument("--material_script", type=Path, required=True)
    parser.add_argument("--output_blend", type=Path, required=True)
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    args = parser.parse_args(argv)
    build(args.scene_script, args.material_script, args.output_blend)


if __name__ == "__main__":
    main()
