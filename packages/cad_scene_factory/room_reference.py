"""Prepare a compact, traceable reference pack from the local The Shed archive.

This is ordinary Python: it reads existing audits and previews without starting
Blender. The room workflow copies the actual source blend and textures later
in adapt mode; prepare_quality_reference() exports only example images and notes.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil


REFERENCE_GUIDE = (
    Path(__file__).resolve().parents[2]
    / "checkpoints/image_generation/codex_scene_factory/room_workflow/REFERENCE.md"
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def prepare_reference(reference_dir: Path, output_dir: Path) -> dict:
    """Build reference.json, compact audit summaries, and copied existing PNGs.

    ``asset_directory`` is the directory to copy into a generated template's
    ``assets/<asset_subdir>``. The file manifest uses paths relative to that
    directory, so it can verify the copy without depending on its new location.
    The reference pack itself deliberately does not duplicate the large assets.
    """
    reference_dir = Path(reference_dir).resolve()
    output_dir = Path(output_dir).resolve()
    asset_directory = reference_dir / "source/the_shed"
    source_blend = asset_directory / "the_shed.blend"
    comparison_dir = reference_dir / "comparison"
    if output_dir.is_relative_to(asset_directory):
        raise ValueError("The reference output must not overwrite the source asset directory.")

    download = read_json(reference_dir / "download_manifest.json")
    groups = read_json(comparison_dir / "workbench_groups.json")
    shading = read_json(comparison_dir / "the_shed_shading.json")
    runtime = read_json(comparison_dir / "the_shed_runtime_flags.json")
    tabletop = read_json(comparison_dir / "tabletop_polygon.json")
    scene = next(item for item in shading["scenes"] if item["name"] == shading["active_scene"])

    source_hash = file_sha256(source_blend)
    downloaded_blend = next(
        item for item in download["blend_files"] if Path(item["path"]).name == source_blend.name
    )
    if source_hash != downloaded_blend["sha256"]:
        raise ValueError("The source blend no longer matches the recorded download.")

    # Hash once here. The host can verify both source preservation and a copied
    # template using the same manifest; large texture files stay out of prompts.
    asset_files = [
        {
            "path": path.relative_to(asset_directory).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in sorted(asset_directory.rglob("*"))
        if path.is_file()
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)
    manifest_path = output_dir / "source_assets.json"
    manifest = {
        "root": str(asset_directory),
        "file_count": len(asset_files),
        "total_bytes": sum(item["bytes"] for item in asset_files),
        "extensions": dict(Counter(Path(item["path"]).suffix.lower() for item in asset_files)),
        "files": asset_files,
    }
    write_json(manifest_path, manifest)

    images = []
    for image_id, description in (
        ("author_view", "原作者相机的本地低采样预览：整体空间、储物层次、日光与阴影。"),
        ("bench_context", "工作台中景：材质磨损、工具组合、桌面与墙面的空间关系。"),
    ):
        original = comparison_dir / f"{image_id}.png"
        target = images_dir / original.name
        shutil.copy2(original, target)
        images.append({
            "id": image_id,
            "path": str(target),
            "description": description,
            "sha256": file_sha256(target),
            "render_record": read_json(comparison_dir / f"{image_id}.json"),
        })

    # A semantic group can contain several objects. Composite assemblies also
    # include nearby contents which are separate groups in the original audit.
    assemblies = {
        "open_tool_chest_with_contents": [
            "open_tool_chest", "hand_drill", "wooden_hammer", "ratchet_wrench", "measuring_tape",
        ],
        "clamping_station": ["bench_vice", "clamped_desk_lamp"],
        "drilling_and_maintenance": [
            "electric_drill", "oil_can", "lubricant_spray", "flathead_screwdriver",
        ],
    }
    write_json(output_dir / "semantic_groups.json", {
        "source_audit": str(comparison_dir / "workbench_groups.json"),
        "coordinate_system": "Right-handed, Z up; world coordinates in metres.",
        "bounds_meaning": "World AABBs of semantic groups, not contact surfaces or collision solids.",
        "workbench_surface": tabletop,
        "groups": groups,
        "related_assemblies": assemblies,
    })

    # Retain the auditable light geometry and settings, not the full node dump.
    # Energy values do not measure image contribution or imply physical units
    # shared by SUN, AREA, and the world environment.
    light_keys = (
        "name", "type", "energy", "color", "hide_render", "world_matrix",
        "angle", "shape", "size", "size_y", "normalize", "exposure",
    )
    write_json(output_dir / "lighting_summary.json", {
        "source_audit": str(comparison_dir / "the_shed_shading.json"),
        "audit_blender_version": shading["blender_version"],
        "source_file_version": shading["source_file_version"],
        "lights": [{key: light[key] for key in light_keys if key in light} for light in scene["lights"]],
        "color_management": scene["color_management"],
        "source_camera": scene["camera"],
        "source_render": scene["render"],
        "source_cycles": scene["cycles"],
        "image_datablocks": shading["image_count"],
        "tiled_image_datablocks": len(runtime["tiled_images"]),
        "declared_udim_tiles": sum(len(item["tiles"]) for item in runtime["tiled_images"]),
        "atmosphere_objects": runtime["atmosphere"],
    })
    shutil.copy2(REFERENCE_GUIDE, output_dir / "overview.md")

    documents = []
    for name, description in (
        ("overview.md", "参考阅读顺序、材质与光照事实、改编要求及证据边界。"),
        ("semantic_groups.json", "13 个语义道具组、工具组合及米制工作台平面。"),
        ("lighting_summary.json", "实际光源变换、成像设置与 UDIM／气氛对象摘要。"),
    ):
        path = output_dir / name
        documents.append({"path": str(path), "description": description, "sha256": file_sha256(path)})

    reference = {
        "schema_version": 1,
        "id": "the_shed",
        "source_blend": str(source_blend),
        "asset_directory": str(asset_directory),
        "asset_subdir": "the_shed",
        "images": images,
        "documents": documents,
        "source": {
            "url": download["url"],
            "project_url": download["project"],
            "license": download["license"],
            "license_url": download["license_source"],
            "sha256": source_hash,
            "archive_sha256": download["archive_sha256"],
        },
        "asset_manifest": {
            "path": str(manifest_path),
            "file_count": manifest["file_count"],
            "total_bytes": manifest["total_bytes"],
            "sha256": file_sha256(manifest_path),
        },
        "audit_sources": [
            {"path": str(path), "sha256": file_sha256(path)}
            for path in (
                reference_dir / "download_manifest.json",
                comparison_dir / "workbench_groups.json",
                comparison_dir / "tabletop_polygon.json",
                comparison_dir / "the_shed_shading.json",
                comparison_dir / "the_shed_runtime_flags.json",
                comparison_dir / "layout_details.md",
                comparison_dir / "shading_notes.md",
            )
        ],
    }
    write_json(output_dir / "reference.json", reference)
    return reference


def prepare_quality_reference(reference_file: Path, output_dir: Path) -> dict:
    """只导出质量示例；不携带基础工程、原场景布局或构建审计的路径。"""
    original = read_json(Path(reference_file).resolve())
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"质量参考输出目录必须为空：{output_dir}")
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True)
    images = []
    for item in original["images"]:
        source = Path(item["path"])
        target = images_dir / source.name
        shutil.copy2(source, target)
        images.append({"id": item["id"], "path": str(target),
                       "description": item["description"], "sha256": file_sha256(target)})
    guide = Path(__file__).resolve().parent / "prompts" / "room" / "QUALITY_REFERENCE.md"
    notes = output_dir / "overview.md"
    shutil.copy2(guide, notes)
    source = original.get("source", {})
    reference = {
        "schema_version": 1, "id": original["id"] + "_quality",
        "images": images,
        "documents": [{"path": str(notes), "description": "Quality comparison only.",
                       "sha256": file_sha256(notes)}],
        "source": {key: source[key] for key in ("project_url", "license", "license_url") if key in source},
    }
    write_json(output_dir / "reference.json", reference)
    return reference


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--reference_dir", type=Path)
    inputs.add_argument("--quality_reference_file", type=Path)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    if args.quality_reference_file:
        reference = prepare_quality_reference(args.quality_reference_file, args.output_dir)
    else:
        reference = prepare_reference(args.reference_dir, args.output_dir)
    print(args.output_dir.resolve() / "reference.json")
    print(f"{len(reference['images'])} existing previews; no render.")


if __name__ == "__main__":
    main()
