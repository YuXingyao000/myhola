"""
Batch render DeepCAD meshes in a single fixed view.

Default: 8 materials (one PNG per material under {model_id}/0/{0..7}.png).

With --material-subdir: one PBR folder only (e.g. Metal010_1K-JPG) -> {model_id}/0/0.png.

Usage:
    blender --background --python render_dataset_single_view.py
    blender --background --python render_dataset_single_view.py -- ^
        --model-root D:\\Project\\HoLa-Brep\\data\\deepcad_v6 ^
        --model-list-dir .\\sampled_lists_valid ^
        --rank 0 ^
        --materials-root D:\\natural\\materials ^
        --output-root .\\output_single_view ^
        --device GPU
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, Iterable, List

import bpy
from mathutils import Matrix, Vector


DEFAULT_MODEL_ROOT = Path(r"D:\Project\HoLa-Brep\data\deepcad_v6")
DEFAULT_MATERIALS_ROOT = Path(r"D:\natural\materials")
DEFAULT_OUTPUT_ROOT = Path.cwd() / "output_single_view"

RENDER_RESOLUTION = 1024
CAMERA_LOCATION = (2.0, 2.0, 2.0)
CAMERA_TARGET = (0.0, 0.0, 0.0)
CAMERA_FOV_DEGREES = 45.0  # Match OCC renderer FOVy=45
BACKGROUND_LIGHT_COLOR = (0.72, 0.76, 0.82, 1.0)
BACKGROUND_LIGHT_STRENGTH = 0.24
GROUND_PLANE_MARGIN = 3.5
GROUND_PLANE_OFFSET = 0.03
TEXTURE_BOX_BLEND = 0.15
TEXTURE_SCALE = (2.5, 2.5, 2.5)
MATERIAL_COUNT = 8
MODEL_DIR_PATTERN = re.compile(r"^\d{8}$")

LOGGER = logging.getLogger("render_dataset_single_view")


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch render PLY models in Blender.")
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument(
        "--mesh-name",
        type=str,
        default="mesh.stl",
        help="Mesh filename inside each model folder. Falls back to mesh.ply when this file is missing.",
    )
    parser.add_argument(
        "--model-list-dir",
        type=Path,
        default=None,
        help="Directory containing rank_{rank}.txt model id lists.",
    )
    parser.add_argument(
        "--model-list",
        type=Path,
        default=None,
        help="Explicit text file containing model ids. Mutually exclusive with --model-list-dir.",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=None,
        help="Rank used to choose rank_{rank}.txt. Defaults to env RANK/LOCAL_RANK/SLURM_PROCID.",
    )
    parser.add_argument("--materials-root", type=Path, default=DEFAULT_MATERIALS_ROOT)
    parser.add_argument(
        "--material-subdir",
        type=str,
        default=None,
        help=(
            "If set, only use materials-root/<name> (must contain PBR maps). "
            "Renders a single image per model at {model_id}/0/0.png instead of 8 materials."
        ),
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--samples", type=int, default=64, help="Cycles samples per render.")
    parser.add_argument(
        "--png-compression",
        type=int,
        default=0,
        help="PNG compression level 0-100. Lower is faster and uses more disk.",
    )
    parser.add_argument(
        "--max-models",
        type=int,
        default=None,
        help="Render at most N models (for small-scale tests).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="AUTO",
        choices=("AUTO", "GPU", "CPU"),
        help="Cycles compute device preference.",
    )
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="Fail fast if GPU rendering is not available.",
    )
    parser.add_argument(
        "--disable-persistent-data",
        action="store_true",
        help="Disable Cycles persistent data cache.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a model if all expected material PNGs already exist under model_id/0/.",
    )

    if "--" in argv:
        script_args = argv[argv.index("--") + 1 :]
    else:
        script_args = []
    return parser.parse_args(script_args)


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def find_model_dirs(model_root: Path) -> List[Path]:
    if not model_root.exists():
        raise FileNotFoundError(f"Model root does not exist: {model_root}")

    model_dirs = [
        path
        for path in sorted(model_root.iterdir())
        if path.is_dir() and MODEL_DIR_PATTERN.match(path.name)
    ]
    return model_dirs


def resolve_rank(explicit_rank: int | None) -> int | None:
    if explicit_rank is not None:
        return explicit_rank

    for env_name in ("RANK", "LOCAL_RANK", "SLURM_PROCID"):
        env_value = os.environ.get(env_name)
        if env_value is None or env_value == "":
            continue
        try:
            return int(env_value)
        except ValueError as exc:
            raise ValueError(
                f"Environment variable {env_name} must be an integer, got {env_value!r}."
            ) from exc
    return None


def load_model_dirs_from_rank_list(model_root: Path, model_list_dir: Path, rank: int) -> List[Path]:
    if not model_root.exists():
        raise FileNotFoundError(f"Model root does not exist: {model_root}")
    if not model_list_dir.exists():
        raise FileNotFoundError(f"Model list dir does not exist: {model_list_dir}")
    if rank < 0:
        raise ValueError("--rank must be non-negative.")

    rank_list_path = model_list_dir / f"rank_{rank}.txt"
    if not rank_list_path.is_file():
        raise FileNotFoundError(f"Rank model list does not exist: {rank_list_path}")

    model_dirs: List[Path] = []
    seen_ids = set()
    for line_number, raw_line in enumerate(rank_list_path.read_text(encoding="utf-8").splitlines(), start=1):
        model_id = raw_line.strip()
        if not model_id or model_id.startswith("#"):
            continue
        if not MODEL_DIR_PATTERN.match(model_id):
            raise ValueError(
                f"Invalid model id {model_id!r} in {rank_list_path} line {line_number}; expected 8 digits."
            )
        if model_id in seen_ids:
            continue

        model_dir = model_root / model_id
        if not model_dir.is_dir():
            raise FileNotFoundError(f"Model dir from rank list not found: {model_dir}")

        seen_ids.add(model_id)
        model_dirs.append(model_dir)

    return model_dirs


def load_model_dirs_from_list(model_root: Path, model_list: Path) -> List[Path]:
    if not model_root.exists():
        raise FileNotFoundError(f"Model root does not exist: {model_root}")
    if not model_list.is_file():
        raise FileNotFoundError(f"Model list does not exist: {model_list}")

    model_dirs: List[Path] = []
    seen_ids = set()
    for line_number, raw_line in enumerate(model_list.read_text(encoding="utf-8").splitlines(), start=1):
        model_id = raw_line.strip()
        if not model_id or model_id.startswith("#"):
            continue
        if not MODEL_DIR_PATTERN.match(model_id):
            raise ValueError(
                f"Invalid model id {model_id!r} in {model_list} line {line_number}; expected 8 digits."
            )
        if model_id in seen_ids:
            continue

        model_dir = model_root / model_id
        if not model_dir.is_dir():
            raise FileNotFoundError(f"Model dir from model list not found: {model_dir}")

        seen_ids.add(model_id)
        model_dirs.append(model_dir)

    return model_dirs


def clear_scene_objects() -> None:
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for mesh in list(bpy.data.meshes):
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)


def get_available_render_engines(scene: bpy.types.Scene) -> List[str]:
    return [item.identifier for item in scene.render.bl_rna.properties["engine"].enum_items]


def ensure_cycles_engine(scene: bpy.types.Scene) -> bool:
    # Try direct switch first; this is the most reliable way to verify whether Cycles exists.
    try:
        scene.render.engine = "CYCLES"
        return True
    except Exception:
        pass

    try:
        bpy.ops.preferences.addon_enable(module="cycles")
    except Exception as exc:
        LOGGER.warning("stage=enable_cycles status=failed error=%s", exc)
    else:
        LOGGER.info("stage=enable_cycles status=requested")

    try:
        scene.render.engine = "CYCLES"
        LOGGER.info("stage=enable_cycles status=ok")
        return True
    except Exception as exc:
        LOGGER.warning("stage=enable_cycles status=no_effect error=%s", exc)

    return False


def _collect_cycles_devices(cycles_prefs: bpy.types.AddonPreferences) -> List:
    devices = []
    try:
        discovered = cycles_prefs.get_devices()
    except Exception:
        discovered = None

    if isinstance(discovered, (list, tuple)):
        for item in discovered:
            if isinstance(item, (list, tuple)):
                devices.extend([dev for dev in item if hasattr(dev, "type")])
            elif hasattr(item, "type"):
                devices.append(item)

    try:
        devices.extend([dev for dev in cycles_prefs.devices if hasattr(dev, "type")])
    except Exception:
        pass

    deduped = []
    seen = set()
    for dev in devices:
        key = (getattr(dev, "name", ""), getattr(dev, "type", ""))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(dev)
    return deduped


def configure_cycles_device(scene: bpy.types.Scene, requested_device: str) -> str:
    if scene.render.engine != "CYCLES":
        LOGGER.warning("stage=device engine=%s fallback=engine_default", scene.render.engine)
        return "ENGINE_DEFAULT"

    mode = requested_device.upper()
    if mode == "CPU":
        scene.cycles.device = "CPU"
        LOGGER.info("stage=device selected=CPU reason=requested_cpu")
        return "CPU"

    cycles_addon = bpy.context.preferences.addons.get("cycles")
    if cycles_addon is None:
        scene.cycles.device = "CPU"
        LOGGER.warning("stage=device selected=CPU reason=cycles_addon_missing")
        return "CPU"

    cycles_prefs = cycles_addon.preferences
    # CUDA is the most predictable backend for multi-process headless rendering.
    # OPTIX can enumerate successfully on some servers but still fail to create
    # visible GPU compute contexts for these short batch renders.
    backend_candidates = ("CUDA", "OPTIX", "HIP", "METAL", "ONEAPI")
    for backend in backend_candidates:
        try:
            cycles_prefs.compute_device_type = backend
            devices = _collect_cycles_devices(cycles_prefs)
        except Exception:
            continue

        gpu_devices = [dev for dev in devices if getattr(dev, "type", "CPU") != "CPU"]
        if not gpu_devices:
            continue

        for dev in devices:
            dev.use = getattr(dev, "type", "CPU") != "CPU"
        scene.cycles.device = "GPU"
        gpu_names = ",".join(getattr(dev, "name", "UnknownGPU") for dev in gpu_devices)
        LOGGER.info(
            "stage=device selected=GPU backend=%s cuda_visible_devices=%s gpus=%s",
            backend,
            os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            gpu_names,
        )
        return "GPU"

    scene.cycles.device = "CPU"
    if mode == "GPU":
        LOGGER.warning("stage=device requested=GPU selected=CPU reason=no_gpu_devices")
    else:
        LOGGER.info("stage=device requested=AUTO selected=CPU reason=no_gpu_devices")
    return "CPU"


def setup_render_settings(
    scene: bpy.types.Scene,
    samples: int,
    requested_device: str,
    png_compression: int,
    disable_persistent_data: bool,
) -> str:
    if ensure_cycles_engine(scene):
        scene.render.engine = "CYCLES"
        scene.cycles.samples = max(1, samples)
        LOGGER.info("stage=engine selected=CYCLES")
    else:
        engine_items = get_available_render_engines(scene)
        LOGGER.warning(
            "Cycles engine not available, using %s available=%s",
            scene.render.engine,
            ",".join(engine_items),
        )

    selected_device = configure_cycles_device(scene, requested_device=requested_device)
    if requested_device.upper() == "GPU" and selected_device != "GPU":
        LOGGER.warning("stage=device requested=GPU selected=%s", selected_device)

    scene.render.resolution_x = RENDER_RESOLUTION
    scene.render.resolution_y = RENDER_RESOLUTION
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.compression = min(100, max(0, png_compression))
    scene.render.use_file_extension = False
    scene.render.film_transparent = True
    if hasattr(scene.render, "use_persistent_data"):
        scene.render.use_persistent_data = not disable_persistent_data
        LOGGER.info(
            "stage=render_settings persistent_data=%s png_compression=%d color_mode=%s resolution=%dx%d film_transparent=%s",
            scene.render.use_persistent_data,
            scene.render.image_settings.compression,
            scene.render.image_settings.color_mode,
            scene.render.resolution_x,
            scene.render.resolution_y,
            scene.render.film_transparent,
        )
    else:
        LOGGER.info(
            "stage=render_settings persistent_data=unsupported png_compression=%d color_mode=%s resolution=%dx%d film_transparent=%s",
            scene.render.image_settings.compression,
            scene.render.image_settings.color_mode,
            scene.render.resolution_x,
            scene.render.resolution_y,
            scene.render.film_transparent,
        )
    return selected_device


def look_at(obj: bpy.types.Object, target: Vector) -> None:
    direction = target - obj.location
    if direction.length == 0:
        return
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def setup_camera(scene: bpy.types.Scene) -> bpy.types.Object:
    camera_data = bpy.data.cameras.new("DatasetCamera")
    camera_obj = bpy.data.objects.new("DatasetCamera", camera_data)
    scene.collection.objects.link(camera_obj)
    camera_obj.location = CAMERA_LOCATION
    look_at(camera_obj, Vector(CAMERA_TARGET))
    scene.camera = camera_obj
    # Match OCC renderer FOV (FOVy=45 degrees)
    import math
    camera_data.sensor_fit = 'VERTICAL'
    camera_data.lens_unit = 'MILLIMETERS'
    camera_data.sensor_height = 24.0
    camera_data.lens = 24.0 / (2.0 * math.tan(math.radians(CAMERA_FOV_DEGREES) / 2.0))
    return camera_obj


def setup_lighting(scene: bpy.types.Scene) -> None:
    key_light_data = bpy.data.lights.new("KeyLight", type="AREA")
    key_light_data.energy = 500.0
    key_light_data.size = 12.0
    key_light_obj = bpy.data.objects.new("KeyLight", key_light_data)
    key_light_obj.location = (4.5, -3.5, 5.2)
    scene.collection.objects.link(key_light_obj)
    look_at(key_light_obj, Vector(CAMERA_TARGET))

    fill_light_data = bpy.data.lights.new("FillLight", type="AREA")
    fill_light_data.energy = 42.0
    fill_light_data.size = 14.0
    fill_light_obj = bpy.data.objects.new("FillLight", fill_light_data)
    fill_light_obj.location = (-3.8, 3.5, 4.2)
    scene.collection.objects.link(fill_light_obj)
    look_at(fill_light_obj, Vector(CAMERA_TARGET))

    rim_light_data = bpy.data.lights.new("RimLight", type="AREA")
    rim_light_data.energy = 150.0
    rim_light_data.size = 3.0
    rim_light_obj = bpy.data.objects.new("RimLight", rim_light_data)
    rim_light_obj.location = (5.5, 2.0, 3.8)
    scene.collection.objects.link(rim_light_obj)
    look_at(rim_light_obj, Vector(CAMERA_TARGET))


def setup_world_background() -> None:
    scene = bpy.context.scene
    if scene.world is None:
        scene.world = bpy.data.worlds.new("DatasetWorld")

    world = scene.world
    world.use_nodes = True
    node_tree = world.node_tree
    nodes = node_tree.nodes
    links = node_tree.links
    nodes.clear()

    background = nodes.new(type="ShaderNodeBackground")
    background.inputs["Color"].default_value = BACKGROUND_LIGHT_COLOR
    background.inputs["Strength"].default_value = BACKGROUND_LIGHT_STRENGTH

    output = nodes.new(type="ShaderNodeOutputWorld")
    links.new(background.outputs["Background"], output.inputs["Surface"])


def setup_color_management(scene: bpy.types.Scene) -> None:
    try:
        scene.display_settings.display_device = "sRGB"
    except Exception:
        pass

    try:
        scene.view_settings.view_transform = "Standard"
    except Exception as exc:
        LOGGER.warning("stage=color_management set_view_transform_failed error=%s", exc)

    try:
        scene.view_settings.look = "None"
    except Exception:
        pass

    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    LOGGER.info(
        "stage=color_management display_device=%s view_transform=%s look=%s exposure=%.2f gamma=%.2f",
        scene.display_settings.display_device,
        scene.view_settings.view_transform,
        scene.view_settings.look,
        scene.view_settings.exposure,
        scene.view_settings.gamma,
    )


def setup_white_background_compositor(scene: bpy.types.Scene) -> None:
    scene.use_nodes = True
    if hasattr(scene.render, "use_compositing"):
        scene.render.use_compositing = True

    node_tree = scene.node_tree
    nodes = node_tree.nodes
    links = node_tree.links
    nodes.clear()

    render_layers = nodes.new(type="CompositorNodeRLayers")
    render_layers.location = (-250.0, 0.0)

    alpha_over = nodes.new(type="CompositorNodeAlphaOver")
    alpha_over.location = (60.0, 0.0)
    alpha_over.inputs[0].default_value = 1.0
    alpha_over.inputs[1].default_value = (1.0, 1.0, 1.0, 1.0)

    composite = nodes.new(type="CompositorNodeComposite")
    composite.location = (320.0, 0.0)

    links.new(render_layers.outputs["Image"], alpha_over.inputs[2])
    links.new(alpha_over.outputs["Image"], composite.inputs["Image"])


def ensure_ground_shadow_catcher(scene: bpy.types.Scene) -> bpy.types.Object:
    plane = bpy.data.objects.get("GroundShadowCatcher")
    if plane is None:
        bpy.ops.mesh.primitive_plane_add(size=2.0, location=(0.0, 0.0, -1.0))
        plane = bpy.context.object
        plane.name = "GroundShadowCatcher"
    if plane.name not in scene.collection.objects:
        scene.collection.objects.link(plane)

    plane.hide_render = False
    plane.hide_viewport = False
    if hasattr(plane, "is_shadow_catcher"):
        plane.is_shadow_catcher = True
    if hasattr(plane, "visible_shadow"):
        plane.visible_shadow = True
    return plane


def fit_ground_shadow_catcher(plane: bpy.types.Object, mesh_objects: List[bpy.types.Object]) -> None:
    min_corner, max_corner = _world_bounds(mesh_objects)
    center = (min_corner + max_corner) * 0.5
    span = max(max_corner.x - min_corner.x, max_corner.y - min_corner.y, 2.0)
    plane.location = (center.x, center.y, min_corner.z - GROUND_PLANE_OFFSET)
    plane.rotation_euler = (0.0, 0.0, 0.0)
    plane.scale = (span * GROUND_PLANE_MARGIN, span * GROUND_PLANE_MARGIN, 1.0)



def _find_texture(material_dir: Path, suffix_candidates: Iterable[str]) -> Path:
    exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".exr")
    for suffix in suffix_candidates:
        for ext in exts:
            matches = sorted(material_dir.glob(f"*_{suffix}{ext}"))
            if matches:
                return matches[0]
    raise FileNotFoundError(
        f"Missing texture in {material_dir} for suffixes: {list(suffix_candidates)}"
    )


def _ensure_pbr_textures(material_dir: Path) -> None:
    _find_texture(material_dir, ("Color", "BaseColor", "Albedo"))
    _find_texture(material_dir, ("Roughness",))
    _find_texture(material_dir, ("Metalness", "Metallic"))
    _find_texture(material_dir, ("NormalGL", "Normal", "NormalDX"))


def build_material_source_mapping(
    material_root: Path, material_subdir: str | None = None
) -> Dict[int, Path]:
    if not material_root.exists():
        raise FileNotFoundError(f"Materials root does not exist: {material_root}")

    if material_subdir is not None:
        material_dir = (material_root / material_subdir).resolve()
        if not material_dir.is_dir():
            raise FileNotFoundError(
                f"--material-subdir {material_subdir!r} is not a directory under {material_root}"
            )
        try:
            _ensure_pbr_textures(material_dir)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Material folder {material_dir} is missing required PBR textures: {exc}"
            ) from exc
        return {0: material_dir}

    candidates = []
    for path in material_root.iterdir():
        if not path.is_dir():
            continue
        try:
            _ensure_pbr_textures(path)
            candidates.append(path)
        except FileNotFoundError:
            continue

    candidates = sorted(candidates, key=lambda p: p.name)
    if len(candidates) != MATERIAL_COUNT:
        raise RuntimeError(
            f"Expected {MATERIAL_COUNT} material folders under {material_root}, found {len(candidates)}: "
            f"{[p.name for p in candidates]}"
        )

    return {idx: path for idx, path in enumerate(candidates)}


def _set_non_color(image: bpy.types.Image) -> None:
    try:
        image.colorspace_settings.name = "Non-Color"
    except Exception:
        image.colorspace_settings.is_data = True


def _image_node(
    node_tree: bpy.types.NodeTree, image_path: Path, location: tuple[float, float]
) -> bpy.types.ShaderNodeTexImage:
    image = bpy.data.images.load(str(image_path), check_existing=True)
    node = node_tree.nodes.new(type="ShaderNodeTexImage")
    node.image = image
    node.location = location
    node.projection = "BOX"
    node.projection_blend = TEXTURE_BOX_BLEND
    return node


def _build_pbr_material(name: str, material_dir: Path) -> bpy.types.Material:
    existing = bpy.data.materials.get(name)
    if existing is not None:
        bpy.data.materials.remove(existing, do_unlink=True)

    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    node_tree = material.node_tree
    nodes = node_tree.nodes
    links = node_tree.links
    nodes.clear()

    output = nodes.new(type="ShaderNodeOutputMaterial")
    output.location = (500.0, 0.0)

    principled = nodes.new(type="ShaderNodeBsdfPrincipled")
    principled.location = (200.0, 0.0)
    links.new(principled.outputs["BSDF"], output.inputs["Surface"])

    tex_coord = nodes.new(type="ShaderNodeTexCoord")
    tex_coord.location = (-1100.0, -80.0)

    mapping = nodes.new(type="ShaderNodeMapping")
    mapping.location = (-880.0, -80.0)
    mapping.inputs["Scale"].default_value = TEXTURE_SCALE
    links.new(tex_coord.outputs["Generated"], mapping.inputs["Vector"])

    color_tex = _image_node(
        node_tree,
        _find_texture(material_dir, ("Color", "BaseColor", "Albedo")),
        (-650.0, 260.0),
    )
    links.new(mapping.outputs["Vector"], color_tex.inputs["Vector"])
    links.new(color_tex.outputs["Color"], principled.inputs["Base Color"])

    rough_tex = _image_node(
        node_tree, _find_texture(material_dir, ("Roughness",)), (-650.0, 20.0)
    )
    _set_non_color(rough_tex.image)
    links.new(mapping.outputs["Vector"], rough_tex.inputs["Vector"])
    links.new(rough_tex.outputs["Color"], principled.inputs["Roughness"])

    metal_tex = _image_node(
        node_tree, _find_texture(material_dir, ("Metalness", "Metallic")), (-650.0, -200.0)
    )
    _set_non_color(metal_tex.image)
    links.new(mapping.outputs["Vector"], metal_tex.inputs["Vector"])
    links.new(metal_tex.outputs["Color"], principled.inputs["Metallic"])

    normal_tex = _image_node(
        node_tree,
        _find_texture(material_dir, ("NormalGL", "Normal", "NormalDX")),
        (-650.0, -430.0),
    )
    _set_non_color(normal_tex.image)
    links.new(mapping.outputs["Vector"], normal_tex.inputs["Vector"])

    normal_map = nodes.new(type="ShaderNodeNormalMap")
    normal_map.location = (-250.0, -430.0)
    links.new(normal_tex.outputs["Color"], normal_map.inputs["Color"])
    links.new(normal_map.outputs["Normal"], principled.inputs["Normal"])

    return material


def load_or_build_materials(
    material_root: Path, material_subdir: str | None = None
) -> Dict[int, bpy.types.Material]:
    source_mapping = build_material_source_mapping(material_root, material_subdir)
    materials: Dict[int, bpy.types.Material] = {}
    for material_idx, material_dir in source_mapping.items():
        LOGGER.info(
            "material=%s stage=load_material source=%s", material_idx, str(material_dir)
        )
        try:
            materials[material_idx] = _build_pbr_material(
                name=f"DatasetMaterial_{material_idx}",
                material_dir=material_dir,
            )
        except Exception as exc:
            raise RuntimeError(
                f"material={material_idx} stage=load_material error={exc} source={material_dir}"
            ) from exc
    return materials


def _import_mesh_file(filepath: Path) -> None:
    suffix = filepath.suffix.lower()
    if suffix == ".stl":
        if hasattr(bpy.ops.wm, "stl_import"):
            result = bpy.ops.wm.stl_import(filepath=str(filepath))
        else:
            result = bpy.ops.import_mesh.stl(filepath=str(filepath))
        if "FINISHED" not in result:
            raise RuntimeError(f"STL import failed with result: {result}")
        return

    if suffix != ".ply":
        raise ValueError(f"Unsupported mesh format for {filepath}; expected .stl or .ply")

    if hasattr(bpy.ops.wm, "ply_import"):
        result = bpy.ops.wm.ply_import(filepath=str(filepath))
        if "FINISHED" not in result:
            raise RuntimeError(f"PLY import failed with result: {result}")
        return

    if hasattr(bpy.ops.import_mesh, "ply"):
        result = bpy.ops.import_mesh.ply(filepath=str(filepath))
        if "FINISHED" not in result:
            raise RuntimeError(f"PLY import failed with result: {result}")
        return

    try:
        bpy.ops.preferences.addon_enable(module="io_mesh_ply")
    except Exception as exc:
        raise RuntimeError("No available PLY import operator in this Blender build.") from exc

    if not hasattr(bpy.ops.import_mesh, "ply"):
        raise RuntimeError("Failed to enable PLY importer addon 'io_mesh_ply'.")

    result = bpy.ops.import_mesh.ply(filepath=str(filepath))
    if "FINISHED" not in result:
        raise RuntimeError(f"PLY import failed with result: {result}")


def load_mesh(mesh_path: Path) -> List[bpy.types.Object]:
    scene = bpy.context.scene
    existing_names = set(scene.objects.keys())

    _import_mesh_file(mesh_path)

    new_objects = [
        scene.objects[name]
        for name in scene.objects.keys()
        if name not in existing_names and scene.objects[name].type == "MESH"
    ]

    if not new_objects:
        new_objects = [obj for obj in bpy.context.selected_objects if obj.type == "MESH"]

    if not new_objects:
        raise RuntimeError(f"No mesh objects imported from {mesh_path}")

    return new_objects


def resolve_mesh_path(model_dir: Path, mesh_name: str) -> Path:
    requested = model_dir / mesh_name
    if requested.is_file():
        return requested

    fallback = model_dir / "mesh.ply"
    if fallback.is_file():
        LOGGER.warning(
            "model=%s stage=check_input requested_missing=%s fallback=%s",
            model_dir.name,
            requested,
            fallback,
        )
        return fallback

    return requested


def _world_bounds(mesh_objects: List[bpy.types.Object]) -> tuple[Vector, Vector]:
    min_corner = Vector((float("inf"), float("inf"), float("inf")))
    max_corner = Vector((float("-inf"), float("-inf"), float("-inf")))

    for obj in mesh_objects:
        for corner in obj.bound_box:
            world_corner = obj.matrix_world @ Vector(corner)
            for axis in range(3):
                min_corner[axis] = min(min_corner[axis], world_corner[axis])
                max_corner[axis] = max(max_corner[axis], world_corner[axis])

    return min_corner, max_corner


def normalize_object_to_unit_box(mesh_objects: List[bpy.types.Object]) -> None:
    bpy.context.view_layer.update()
    min_corner, max_corner = _world_bounds(mesh_objects)

    center = (min_corner + max_corner) * 0.5
    size = max_corner - min_corner
    max_extent = max(size.x, size.y, size.z)
    if max_extent <= 0:
        raise RuntimeError("Invalid mesh bounds: max extent is zero.")

    scale_factor = 0.9 * 2.0 / max_extent
    transform = Matrix.Scale(scale_factor, 4) @ Matrix.Translation(-center)

    for obj in mesh_objects:
        obj.matrix_world = transform @ obj.matrix_world

    bpy.context.view_layer.update()
    min_corner, max_corner = _world_bounds(mesh_objects)
    epsilon = 1e-4
    if (
        min_corner.x < -1.0 - epsilon
        or min_corner.y < -1.0 - epsilon
        or min_corner.z < -1.0 - epsilon
        or max_corner.x > 1.0 + epsilon
        or max_corner.y > 1.0 + epsilon
        or max_corner.z > 1.0 + epsilon
    ):
        raise RuntimeError(
            f"Normalization failed: bounds out of range. min={tuple(min_corner)}, max={tuple(max_corner)}"
        )


def apply_material(mesh_objects: List[bpy.types.Object], material: bpy.types.Material) -> None:
    for obj in mesh_objects:
        if obj.type != "MESH":
            continue
        obj.data.materials.clear()
        obj.data.materials.append(material)


def render_to_file(output_path: Path, scene: bpy.types.Scene) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(output_path)
    bpy.context.view_layer.update()
    bpy.ops.render.render(write_still=True)


def cleanup_mesh_objects(mesh_objects: List[bpy.types.Object]) -> None:
    object_names = [obj.name for obj in mesh_objects]
    for name in object_names:
        obj = bpy.data.objects.get(name)
        if obj is not None:
            bpy.data.objects.remove(obj, do_unlink=True)

    for mesh in list(bpy.data.meshes):
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)


def main(args: argparse.Namespace) -> None:
    args.model_root = args.model_root.resolve()
    if args.model_list_dir is not None:
        args.model_list_dir = args.model_list_dir.resolve()
    args.materials_root = args.materials_root.resolve()
    args.output_root = args.output_root.resolve()
    args.rank = resolve_rank(args.rank)

    LOGGER.info("stage=start model_root=%s", args.model_root)
    if args.model_list_dir is not None:
        LOGGER.info("stage=start model_list_dir=%s", args.model_list_dir)
        LOGGER.info("stage=start rank=%s", args.rank)
    LOGGER.info("stage=start materials_root=%s", args.materials_root)
    if args.material_subdir:
        LOGGER.info("stage=start material_subdir=%s (single-material output)", args.material_subdir)
    LOGGER.info("stage=start output_root=%s", args.output_root)

    clear_scene_objects()
    scene = bpy.context.scene
    selected_device = setup_render_settings(
        scene,
        samples=args.samples,
        requested_device=args.device,
        png_compression=args.png_compression,
        disable_persistent_data=args.disable_persistent_data,
    )
    if args.require_gpu and selected_device != "GPU":
        raise RuntimeError(
            f"GPU was required but unavailable (selected_device={selected_device})."
        )
    setup_color_management(scene)
    setup_white_background_compositor(scene)
    setup_world_background()
    setup_camera(scene)
    setup_lighting(scene)
    ground_plane = ensure_ground_shadow_catcher(scene)

    total_start = time.perf_counter()
    materials_start = time.perf_counter()
    materials = load_or_build_materials(args.materials_root, args.material_subdir)
    material_indices = sorted(materials.keys())
    LOGGER.info(
        "stage=load_materials_done elapsed=%.2fs count=%d indices=%s",
        time.perf_counter() - materials_start,
        len(materials),
        material_indices,
    )
    if args.model_list is not None and args.model_list_dir is not None:
        raise ValueError("Use either --model-list or --model-list-dir, not both.")

    if args.model_list is not None:
        args.model_list = args.model_list.resolve()
        model_dirs = load_model_dirs_from_list(args.model_root, args.model_list)
        LOGGER.info(
            "stage=load_model_list count=%d source=%s",
            len(model_dirs),
            args.model_list,
        )
    elif args.model_list_dir is not None:
        if args.rank is None:
            raise ValueError(
                "--model-list-dir was provided, but no rank was found. "
                "Pass --rank or set RANK/LOCAL_RANK/SLURM_PROCID."
            )
        model_dirs = load_model_dirs_from_rank_list(args.model_root, args.model_list_dir, args.rank)
        LOGGER.info(
            "stage=load_rank_list rank=%d count=%d source=%s",
            args.rank,
            len(model_dirs),
            args.model_list_dir / f"rank_{args.rank}.txt",
        )
    else:
        model_dirs = find_model_dirs(args.model_root)
    if args.max_models is not None:
        if args.max_models <= 0:
            raise ValueError("--max-models must be a positive integer.")
        model_dirs = model_dirs[: args.max_models]
        LOGGER.info("stage=limit_models max_models=%d selected=%d", args.max_models, len(model_dirs))
    LOGGER.info("stage=scan_models count=%d", len(model_dirs))

    rendered_images = 0
    skipped_models = 0
    model_failures = 0

    for model_dir in model_dirs:
        model_start = time.perf_counter()
        model_id = model_dir.name

        if args.skip_existing:
            view_idx_check = 0
            all_exist = all(
                (args.output_root / model_id / str(view_idx_check) / f"{mi}.png").is_file()
                for mi in material_indices
            )
            if all_exist:
                skipped_models += 1
                LOGGER.info(
                    "model=%s stage=skip_existing (all %d PNGs present)", model_id, len(material_indices)
                )
                continue

        mesh_path = resolve_mesh_path(model_dir, args.mesh_name)
        if not mesh_path.is_file():
            LOGGER.error(
                "model=%s stage=check_input error=missing mesh at %s",
                model_id,
                mesh_path,
            )
            model_failures += 1
            continue

        LOGGER.info("model=%s stage=load_mesh path=%s", model_id, mesh_path)
        mesh_objects: List[bpy.types.Object] = []

        try:
            load_start = time.perf_counter()
            mesh_objects = load_mesh(mesh_path)
            load_elapsed = time.perf_counter() - load_start
            normalize_start = time.perf_counter()
            normalize_object_to_unit_box(mesh_objects)
            fit_ground_shadow_catcher(ground_plane, mesh_objects)
            normalize_elapsed = time.perf_counter() - normalize_start
            LOGGER.info(
                "model=%s stage=prepare_done load_elapsed=%.2fs normalize_elapsed=%.2fs object_count=%d",
                model_id,
                load_elapsed,
                normalize_elapsed,
                len(mesh_objects),
            )
        except Exception as exc:
            LOGGER.error(
                "model=%s stage=prepare error=%s traceback=%s",
                model_id,
                exc,
                traceback.format_exc().strip().replace("\n", " | "),
            )
            cleanup_mesh_objects(mesh_objects)
            model_failures += 1
            continue

        view_idx = 0
        for material_idx in material_indices:
            material = materials[material_idx]
            output_path = args.output_root / model_id / str(view_idx) / f"{material_idx}.png"
            LOGGER.info(
                "model=%s view=%d material=%d stage=render output=%s",
                model_id,
                view_idx,
                material_idx,
                output_path,
            )
            try:
                render_start = time.perf_counter()
                apply_material(mesh_objects, material)
                render_to_file(output_path, scene)
                render_elapsed = time.perf_counter() - render_start
                rendered_images += 1
                LOGGER.info(
                    "model=%s view=%d material=%d stage=render_done elapsed=%.2fs",
                    model_id,
                    view_idx,
                    material_idx,
                    render_elapsed,
                )
            except Exception as exc:
                LOGGER.error(
                    "model=%s view=%d material=%d stage=render error=%s traceback=%s",
                    model_id,
                    view_idx,
                    material_idx,
                    exc,
                    traceback.format_exc().strip().replace("\n", " | "),
                )
                continue

        cleanup_mesh_objects(mesh_objects)
        LOGGER.info(
            "model=%s stage=model_done elapsed=%.2fs",
            model_id,
            time.perf_counter() - model_start,
        )

    LOGGER.info(
        "stage=done models_total=%d skipped_models=%d model_failures=%d rendered_images=%d elapsed=%.2fs",
        len(model_dirs),
        skipped_models,
        model_failures,
        rendered_images,
        time.perf_counter() - total_start,
    )


if __name__ == "__main__":
    configure_logging()
    cli_args = parse_args(sys.argv)
    main(cli_args)
