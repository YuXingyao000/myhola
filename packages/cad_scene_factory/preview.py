"""普通 Python 入口：构建当前模板，再用固定零件组生成可审阅的预览。

quick 使用 3 件、64 samples；review 使用 12 件、256 samples 和三个空场景。
每次创建新目录，保存全体计划样本、失败、日志和图片，不调用 Codex。
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

from PIL import Image, ImageDraw


SCRIPT_DIR = Path(__file__).resolve().parent
EMPTY_HEIGHTS = [-0.901, -0.4505, 0.0]


def preview_template(
    template_dir: Path,
    *,
    blender: Path,
    stl_dir: Path,
    panel_file: Path,
    preset: str = "quick",
    output_dir: Path | None = None,
    device: str = "OPTIX",
    gpu_index: int = 0,
) -> dict:
    """返回 preview_report.json 的内容；technical 只代表本轮已实现的检查。

    panel_file 固定 12 个模型及 STL 摘要。quick_ids 必须从这 12 件中选择，
    两档使用完整列表的索引作 seed，因此同一个零件不会因为换档而换 seed。
    """
    template_dir = template_dir.resolve()
    stl_dir = stl_dir.resolve()
    panel_file = panel_file.resolve()
    panel = json.loads(panel_file.read_text(encoding="utf-8"))
    planned = _plan(panel, stl_dir, preset)
    samples = 64 if preset == "quick" else 256
    if output_dir is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        output_dir = template_dir / "previews" / f"{timestamp}_{preset}"
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    # 1. 固定本轮输入。原模板不写入 .blend、报告或预览图片。
    before = snapshot_inputs(template_dir, panel_file)
    for item in planned:
        if item["kind"] == "model":
            item["actual_stl_sha256"] = file_sha256(Path(item["model_file"]))
    request = {
        "template_dir": str(template_dir), "panel_file": str(panel_file),
        "panel_version": panel["version"], "preset": preset, "samples": samples,
        "device": device, "gpu_index": gpu_index, "planned": planned,
        "snapshot": before,
    }
    _write_json(output_dir / "request.json", request)
    _write_json(output_dir / "panel.json", panel)

    # 2. 先从当前脚本构建，再在另一个 Blender 进程中打开并批量拍照。
    scene_file = output_dir / "scene.blend"
    build_command = _blender_command(blender, "build.py") + [
        "--scene_script", str(template_dir / "build_scene.py"),
        "--material_script", str(template_dir / "build_material.py"),
        "--output_blend", str(scene_file),
    ]
    print(f"构建模板：{template_dir}", flush=True)
    build_result = _run(build_command, output_dir / "build.log")
    build_ok = build_result["returncode"] == 0 and scene_file.is_file()
    render_result = {"status": "not_run", "reason": "Build did not succeed"}
    if build_ok:
        models = [item for item in planned if item["kind"] == "model"]
        render_command = _blender_command(blender, "render.py") + [
            "--scene_file", str(scene_file), "--output_dir", str(output_dir / "renders"),
            "--samples", str(samples), "--device", device, "--gpu_index", str(gpu_index),
            "--continue_on_error", "--model_files", *[item["model_file"] for item in models],
            "--model_seeds", *[str(item["seed"]) for item in models],
        ]
        if preset == "review":
            render_command += ["--empty_heights", *[str(height) for height in EMPTY_HEIGHTS]]
        print(f"渲染 {len(planned)} 张预览：{preset}，{samples} samples", flush=True)
        render_result = _run(render_command, output_dir / "render.log")

    # 3. 按预定列表核对，文件缺失、异常退出都不能缩小统计分母。
    results = _collect_results(planned, output_dir / "renders")
    after = snapshot_inputs(template_dir, panel_file)
    model_results = [item for item in results if item["kind"] == "model"]
    models_unchanged = all(item["actual_stl_sha256"] == item["final_stl_sha256"] for item in model_results)
    models_match_panel = all(
        item["actual_stl_sha256"] == item["final_stl_sha256"] == item["expected_stl_sha256"]
        and item["actual_stl_sha256"] is not None
        for item in model_results
    )
    unchanged = before == after and not before["errors"] and models_unchanged
    build_report_exists = (output_dir / "build_report.json").is_file()
    technical_pass = (
        build_ok and build_report_exists and render_result.get("returncode") == 0
        and unchanged and models_match_panel and all(item["status"] == "success" for item in results)
    )
    report = {
        "preset": preset, "panel_version": panel["version"],
        "technical": "pass" if technical_pass else "fail", "visual": "pending",
        "counts": {
            "planned": len(results),
            **{status: sum(item["status"] == status for item in results)
               for status in ("success", "failed", "missing")},
        },
        "build": build_result, "build_report_exists": build_report_exists,
        "render": render_result, "inputs_unchanged": unchanged,
        "model_inputs_match_panel": models_match_panel,
        "snapshot_before": before, "snapshot_after": after, "results": results,
        "checks_not_run": [
            "template_relocation", "exact_support_surface_contact", "pixel_visibility",
            "visual_realism", "STEP_to_STL_fidelity", "entire_dataset_coverage",
        ],
        "geometry_check_scope": "CAD AABB framing, support AABB height/footprint, camera-to-CAD AABB blockers",
        "output_dir": str(output_dir),
    }
    _write_json(output_dir / "preview_report.json", report)
    _contact_sheet(results, output_dir / "renders", output_dir / "contact_sheet.png")
    print(f"技术检查：{report['technical']}；视觉状态：pending；报告：{output_dir}", flush=True)
    return report


def _plan(panel: dict, stl_dir: Path, preset: str) -> list[dict]:
    models = panel["models"]
    ids = [item["model_id"] for item in models]
    if len(ids) != 12 or len(set(ids)) != 12:
        raise ValueError("The fixed review panel must contain 12 distinct model IDs")
    if preset not in {"quick", "review"}:
        raise ValueError("preset must be quick or review")
    quick_ids = panel["quick_ids"]
    if len(set(quick_ids)) != 3 or len(quick_ids) != 3 or not set(quick_ids) <= set(ids):
        raise ValueError("quick_ids must identify three distinct models from the review panel")
    selected = models if preset == "review" else [item for item in models if item["model_id"] in quick_ids]
    planned = [
        {"kind": "model", "model_id": item["model_id"], "face_count": item["face_count"],
         "model_file": str(stl_dir / f"{item['model_id']}.stl"),
         "expected_stl_sha256": item["stl_sha256"], "seed": ids.index(item["model_id"]),
         "image_file": f"{index:06d}_{item['model_id']}.png"}
        for index, item in enumerate(selected)
    ]
    if preset == "review":
        planned.extend(
            {"kind": "empty", "environment_height": height, "seed": 0,
             "image_file": f"empty_{index:03d}.png"}
            for index, height in enumerate(EMPTY_HEIGHTS)
        )
    return planned


def snapshot_inputs(template_dir: Path, panel_file: Path) -> dict:
    """记录渲染输入，供预览与工作流共同判断结果是否仍对应当前模板。

    包含辅助 Python 模块和声明的素材；不包含新增的日志、审阅记录和图片。
    缺失/损坏也保留证据，不把写入普通工作记录误判为模板修改。
    """
    files = {name: file_sha256(template_dir / name) for name in ("build_scene.py", "build_material.py")}
    errors = [f"Missing required file: {name}" for name, digest in files.items() if digest is None]
    for path in sorted(template_dir.rglob("*.py")):
        files[str(path.relative_to(template_dir))] = file_sha256(path)
    if (template_dir / "scene_plan.md").is_file():
        files["scene_plan.md"] = file_sha256(template_dir / "scene_plan.md")
    manifest = template_dir / "assets.json"
    files["assets.json"] = file_sha256(manifest)
    if manifest.is_file():
        try:
            for asset in json.loads(manifest.read_text(encoding="utf-8")):
                path = (template_dir / asset["path"]).resolve()
                if not path.is_relative_to(template_dir):
                    raise ValueError(f"Asset path escapes template directory: {asset['path']}")
                files[asset["path"]] = file_sha256(path)
                if files[asset["path"]] is None or files[asset["path"]] != asset["sha256"]:
                    errors.append(f"Missing asset or SHA-256 mismatch: {asset['path']}")
        except (ValueError, KeyError, TypeError, OSError) as error:
            errors.append(f"Invalid assets.json: {error}")
    host_files = {name: file_sha256(SCRIPT_DIR / name) for name in (
        "preview.py", "build.py", "render.py", "camera.py", "geometry.py", "visibility.py",
        "call_llm.py", "workflow.py", "assets.py",
    )}
    # 提示词已独立为文件，也属于本轮固定的工作流输入。
    for path in sorted((SCRIPT_DIR / "prompts").rglob("*")):
        if path.is_file():
            host_files[str(path.relative_to(SCRIPT_DIR))] = file_sha256(path)
    return {
        "template_files": files, "panel_sha256": file_sha256(panel_file), "errors": errors,
        "host_files": host_files,
    }


def _collect_results(planned: list[dict], render_dir: Path) -> list[dict]:
    records = {}
    record_path = render_dir / "renders.jsonl"
    if record_path.is_file():
        for line in record_path.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
                records[Path(record["image_file"]).name] = record
            except (ValueError, KeyError, TypeError):
                # 中途终止可能留下半行；该样本仍以 missing 进入预定分母。
                continue
    results = []
    for item in planned:
        record = records.get(item["image_file"])
        result = dict(item, status="missing", geometry_checks="not_run")
        if item["kind"] == "model":
            result["final_stl_sha256"] = file_sha256(Path(item["model_file"]))
        if record is None:
            result["error"] = "No completed render record"
        else:
            result["status"] = "failed"
            result["geometry_checks"] = record.get("geometry_checks", "not_run")
            result["render_record"] = record
            try:
                if record["status"] != "success":
                    raise ValueError(record.get("error", "Rendering failed"))
                if record["seed"] != item["seed"]:
                    raise ValueError("Rendered seed differs from the fixed panel seed")
                if item["kind"] == "model":
                    if not (
                        item["actual_stl_sha256"] == result["final_stl_sha256"] == item["expected_stl_sha256"]
                        and item["actual_stl_sha256"] is not None
                    ):
                        raise ValueError("STL changed during preview or differs from the fixed panel")
                    if result["geometry_checks"] != "pass":
                        raise ValueError("Geometry checks did not pass")
                image_path = render_dir / item["image_file"]
                with Image.open(image_path) as image:
                    image.load()
                    if image.format != "PNG" or image.size != (512, 512) or image.mode != "RGB":
                        raise ValueError(f"Expected RGB PNG 512x512, got {image.format} {image.mode} {image.size}")
                result.update(status="success", image_sha256=file_sha256(image_path))
            except (ValueError, KeyError, OSError) as error:
                result["error"] = str(error)
        results.append(result)
    return results


def _contact_sheet(results: list[dict], render_dir: Path, destination: Path) -> None:
    cell_width, cell_height, columns = 276, 302, 3
    sheet = Image.new("RGB", (columns * cell_width, ((len(results) + columns - 1) // columns) * cell_height), "#20242b")
    draw = ImageDraw.Draw(sheet)
    for index, item in enumerate(results):
        x, y = index % columns * cell_width + 10, index // columns * cell_height + 10
        if item["status"] == "success":
            with Image.open(render_dir / item["image_file"]) as image:
                sheet.paste(image.resize((256, 256)), (x, y))
        else:
            draw.rectangle((x, y, x + 256, y + 256), fill="#4c292e")
            draw.text((x + 12, y + 110), item["status"].upper(), fill="white")
            draw.text((x + 12, y + 135), "See preview_report.json", fill="white")
        label = (f"{item['model_id']} | faces {item['face_count']}" if item["kind"] == "model"
                 else f"Empty environment | h={item['environment_height']}")
        draw.text((x, y + 262), label, fill="white")
        draw.text((x, y + 277), f"seed {item['seed']} | {item['status']}", fill="#bbbbbb")
    sheet.save(destination)


def _blender_command(blender: Path, script: str) -> list[str]:
    return [str(blender.resolve()), "--background", "--factory-startup", "--python-exit-code", "1",
            "--python", str(SCRIPT_DIR / script), "--"]


def _run(command: list[str], log_path: Path) -> dict:
    result = {"command": command, "log": str(log_path), "returncode": None}
    with log_path.open("w", encoding="utf-8") as log:
        try:
            process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
            result["returncode"] = process.returncode
        except OSError as error:
            result["error"] = str(error)
            log.write(str(error) + "\n")
    return result


def file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template_dir", type=Path, required=True)
    parser.add_argument("--blender", type=Path, required=True)
    parser.add_argument("--stl_dir", type=Path, required=True)
    parser.add_argument("--panel_file", type=Path, required=True)
    parser.add_argument("--preset", choices=("quick", "review"), default="quick")
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--device", choices=("OPTIX", "CUDA", "CPU"), default="OPTIX")
    parser.add_argument("--gpu_index", type=int, default=0)
    report = preview_template(**vars(parser.parse_args()))
    if report["technical"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
