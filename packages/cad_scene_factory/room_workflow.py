"""普通 Python API：参考资料 → Codex 制作 → Blender 验收 → 看图返修。

入口是 generate_room()。本流程只制作独立环境，不加载 CAD，也不调用图像生成模型。
Codex 的账号认证和事件记录复用 call_llm；场景操作交给 room_tools.py。
"""

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from PIL import Image, ImageDraw
from rich.console import Console

if __package__:
    from .call_llm import run_codex_turn
else:
    from call_llm import run_codex_turn


DIRECTORY = Path(__file__).resolve().parent
PROMPTS = DIRECTORY / "prompts" / "room"
CONSOLE = Console(stderr=True)
VIEW_IDS = {"room_overview", "room_reverse", "workbench_context",
            "workbench_detail", "surface_detail", "alternate_view"}


def generate_room(
    user_prompt: str,
    reference_file: Path,
    output_dir: Path,
    blender: Path,
    *,
    reference_mode: str = "adapt",
    asset_cache: Path | None = None,
    asset_catalog: Path | None = None,
    model: str = "gpt-6-astra",
    reasoning_effort: str = "ultra",
    sandbox: str = "workspace-write",
    device: str = "OPTIX",
    gpu_index: int = 0,
    samples: int = 128,
    seed: int = 8,
    max_repairs: int = 3,
    dry_run: bool = False,
) -> dict:
    """新建一个环境任务，返回实际完成状态；已有任务使用 resume_room()。

    adapt 复制参考工程供改编；quality_only 只读图片与说明，由 Agent 搭建新空间。
    reference_file 来自 room_reference.py。asset_cache 可指定共享下载缓存，
    真实依赖仍复制进 output_dir；ready_for_review 不等于用户已批准。
    """
    output_dir = Path(output_dir).resolve()
    reference_file = Path(reference_file).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"输出目录非空，请换目录或使用 --resume：{output_dir}")
    if samples <= 0 or max_repairs < 0:
        raise ValueError("samples 必须为正数，max_repairs 不能小于零")
    if reference_mode not in {"adapt", "quality_only"}:
        raise ValueError("reference_mode 必须为 adapt 或 quality_only")
    reference = _read_json(reference_file)
    if reference_mode == "quality_only" and any(
        key in reference for key in ("source_blend", "asset_directory", "asset_manifest")
    ):
        raise ValueError("quality_only 请使用仅含图片与说明的参考包，不传入基础工程")
    reference_paths = [reference_file]
    reference_paths += [Path(item["path"]) for item in reference["images"] + reference.get("documents", [])]
    runtime = {
        "python": sys.executable,
        "blender": str(Path(blender).resolve()),
        "room_tools": str(DIRECTORY / "room_tools.py"),
        "template_dir": str(output_dir),
        "reference_file": str(reference_file),
        "reference_bundle": reference,
        "reference_mode": reference_mode,
        "assets_script": str(DIRECTORY / "assets.py"),
        "asset_cache": str(Path(asset_cache).resolve() if asset_cache else output_dir / ".asset_cache"),
        "asset_catalog": str(Path(asset_catalog).resolve()) if asset_catalog else None,
        "device": device, "gpu_index": gpu_index, "seed": seed,
    }
    documents = {name: (PROMPTS / name).read_text(encoding="utf-8")
                 for name in ("CONTRACT.md", "GUIDE.md")}
    documents["ASSET_TOOLS.md"] = (PROMPTS.parent / "ASSET_TOOLS.md").read_text(encoding="utf-8")
    request = {
        "model": model, "reasoning_effort": reasoning_effort, "sandbox": sandbox,
        "output_dir": str(output_dir), "prompt": user_prompt, "runtime": runtime,
        "developer_instructions": "\n\n".join(documents.values())
        + "\n\n```json\n" + json.dumps(runtime, ensure_ascii=False, indent=2) + "\n```",
        "instruction_files": documents, "samples": samples, "max_repairs": max_repairs,
        "host_files": _host_snapshot(),
        "reference_files": {str(path): _sha256(path) for path in reference_paths},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "request.json", request)
    if dry_run:
        return {"status": "dry_run", "request_file": str(output_dir / "request.json")}

    result = {
        "status": "running", "stage": "prepare", "template_dir": str(output_dir),
        "session_id": None, "human_review": "pending", "repairs_used": 0,
        "rounds": [], "codex_turns": [], "remaining_issues": [],
    }
    return _execute(request, result, create=True)


def resume_room(output_dir: Path, *, feedback: str | None = None,
                max_repairs: int | None = None) -> dict:
    """续接明确的任务；可把独立审阅发现的问题反馈给原制作会话。

    已完成的记录不会覆盖。max_repairs 是整个任务累计返修上限。
    不会因为等待时间长就另开制作会话，也不使用 Codex 的最近会话。
    """
    output_dir = Path(output_dir).resolve()
    # 读状态和改预算也必须在锁内，不能先改写一个仍活跃任务的 request。
    with _task_lock(output_dir):
        request = _read_json(output_dir / "request.json")
        result = _read_json(output_dir / "workflow_result.json")
        if not result["session_id"]:
            create_result = output_dir / "workflow" / "create" / "result.json"
            turn = _read_json(create_result) if create_result.exists() else {}
            terminal = turn.get("status") in {"completed", "failed"} and turn.get("returncode") is not None
            if not terminal or not turn.get("session_id"):
                raise RuntimeError("创建阶段没有已终止且带明确会话 ID 的记录，不能另开会话")
            result["session_id"] = turn["session_id"]
            result["pending_creation"] = turn["status"] != "completed"
            result["active_turn_dir"] = str(create_result.parent)
            if not result["codex_turns"]:
                result["codex_turns"].append(_turn_summary(turn))
        if result.get("pending_creation"):
            turn = _read_json(Path(result["active_turn_dir"]) / "result.json")
            if turn.get("returncode") is None:
                raise RuntimeError("创建进程尚无退出记录，不能自动续接，请先核对进程")
        if max_repairs is not None:
            if max_repairs < result["repairs_used"]:
                raise ValueError("累计返修上限不能小于已经执行的返修次数")
            request["max_repairs"] = max_repairs
            _write_json(output_dir / "request.json", request)
        if result["status"] == "ready_for_review" and not feedback:
            return result
        if feedback:
            result["pending_feedback"] = {
                "decision": "revise", "summary": feedback,
                "remaining_issues": [feedback], "source": "external_review",
            }
        result.pop("error", None)
        result["status"] = "running"
        return _execute_locked(request, result, create=False)


def _execute(request: dict, result: dict, *, create: bool) -> dict:
    """同一模板同一时间只允许一个外围进程工作，避免误续接仍活跃的任务。"""
    with _task_lock(Path(request["output_dir"])):
        return _execute_locked(request, result, create=create)


@contextmanager
def _task_lock(output_dir: Path):
    with (output_dir / ".workflow.lock").open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("该模板的工作流进程仍在运行，请查看现有任务") from error
        handle.seek(0)
        handle.truncate()
        handle.write(str(os.getpid()))
        handle.flush()
        yield


def _execute_locked(request: dict, result: dict, *, create: bool) -> dict:
    """任务边界统一保存异常；内部按制作、预览、审阅、返修顺序执行。"""
    template_dir = Path(request["output_dir"])
    records = template_dir / "workflow"
    records.mkdir(exist_ok=True)
    _save_result(result)
    try:
        _check_host(request)
        if create:
            _stage_assets(request)
            result["stage"] = "create"
            result["pending_creation"] = True
            result["active_turn_dir"] = str(records / "create")
            _save_result(result)
            turn = run_codex_turn(
                request, _prompt("create") + request["prompt"], records / "create",
                images=_reference_images(request),
            )
            result["session_id"] = turn["session_id"]
            result["codex_turns"].append(_turn_summary(turn))
            result.pop("pending_creation")
            _save_result(result)
        elif result.get("pending_creation"):
            # 额度等外部失败可以保留已经读取的参考资料和工具结果，恢复原会话。
            number = len(list(records.glob("create_resume_*"))) + 1
            turn_dir = records / f"create_resume_{number:02d}"
            result.update(stage="create_resume", active_turn_dir=str(turn_dir))
            _save_result(result)
            turn = run_codex_turn(
                request, _prompt("resume_create"), turn_dir,
                session_id=result["session_id"], images=_reference_images(request),
            )
            result["codex_turns"].append(_turn_summary(turn))
            result.pop("pending_creation")
            _save_result(result)

        while True:
            _check_host(request)
            feedback = result.get("pending_feedback")
            if feedback:
                if result["repairs_used"] >= request["max_repairs"]:
                    return _finish(result, "incomplete", feedback["remaining_issues"])
                result["stage"] = "repair"
                result["repairs_used"] += 1
                _save_result(result)
                turn = run_codex_turn(
                    request, _prompt("repair") + json.dumps(feedback, ensure_ascii=False, indent=2),
                    records / f"repair_{result['repairs_used']:02d}",
                    session_id=result["session_id"],
                )
                result["codex_turns"].append(_turn_summary(turn))
                result.pop("pending_feedback")
                _save_result(result)
                _check_host(request)

            revision = len(result["rounds"])
            round_dir = records / f"round_{revision:02d}"
            round_dir.mkdir(exist_ok=False)
            record = {"revision": revision, "directory": str(round_dir), "status": "running"}
            result["rounds"].append(record)
            result["stage"] = "preview"
            _save_result(result)
            CONSOLE.rule(f"环境验收 · 第 {revision + 1} 版", style="cyan")
            report = preview_room(request, round_dir / "preview")
            record.update(status=report["status"], report=str(round_dir / "preview" / "preview_report.json"))
            _save_result(result)
            if report["status"] != "pass":
                review = {"decision": "revise", "summary": "Blender 构建或预览未通过",
                          "remaining_issues": report["errors"],
                          "technical_report": record["report"],
                          "counts": {key: report[key] for key in ("planned", "success", "failed")}}
            else:
                result["stage"] = "inspect"
                _save_result(result)
                review, turn = _inspect(request, result["session_id"], report, round_dir)
                result["codex_turns"].append(_turn_summary(turn))
            _write_json(round_dir / "review.json", review)
            record["decision"] = review["decision"]
            record["review"] = str(round_dir / "review.json")
            if review["decision"] == "ready":
                result["last_report"] = record["report"]
                result["room_blend"] = report["blend_file"]
                return _finish(result, "ready_for_review", [])
            if review["decision"] == "blocked":
                return _finish(result, "incomplete", review["remaining_issues"])
            result["pending_feedback"] = review
            _save_result(result)
    except (Exception, KeyboardInterrupt) as error:
        if result.get("pending_creation") and result.get("active_turn_dir"):
            turn_file = Path(result["active_turn_dir"]) / "result.json"
            if turn_file.is_file():
                turn = _read_json(turn_file)
                result["session_id"] = turn.get("session_id") or result["session_id"]
                if not any(item["turn_dir"] == turn["turn_dir"] for item in result["codex_turns"]):
                    result["codex_turns"].append(_turn_summary(turn))
        result["error"] = f"{type(error).__name__}: {error}"
        return _finish(result, "incomplete", [result["error"]])


def preview_room(request: dict, output_dir: Path) -> dict:
    """不调用模型，从模板脚本独立重建，再生成固定的六个环境视角。"""
    output_dir.mkdir(parents=True, exist_ok=False)
    template_dir = Path(request["output_dir"])
    report = {"status": "failed", "output_dir": str(output_dir), "errors": [], "images": [],
              "planned": len(VIEW_IDS), "success": 0, "failed": len(VIEW_IDS)}
    try:
        for name in ("build_scene.py", "views.json", "scene_plan.md", "assets.json"):
            if not (template_dir / name).is_file():
                raise FileNotFoundError(f"缺少交付文件：{name}")
        views = _read_json(template_dir / "views.json")
        ids = [item["id"] for item in views["views"]]
        if len(ids) != len(VIEW_IDS) or set(ids) != VIEW_IDS:
            raise ValueError(f"views.json 必须覆盖六个不同视角：{sorted(VIEW_IDS)}")
        report["asset_files_checked"] = _check_template_assets(template_dir)
        _write_json(output_dir / "views.json", views)
        _blender(request, ["build", "--template_dir", str(template_dir),
                          "--output_dir", str(output_dir)], output_dir / "build.log")
        build = _read_json(output_dir / "build_report.json")
        report["build"] = build
        if build["status"] != "pass":
            raise RuntimeError(f"构建检查失败：{build}")
        blend_file = output_dir / "room.blend"
        renders = output_dir / "renders"
        _blender(request, ["render", "--blend", str(blend_file),
                          "--views_file", str(output_dir / "views.json"),
                          "--output_dir", str(renders), "--samples", str(request["samples"]),
                          "--seed", str(request["runtime"]["seed"]),
                          "--device", request["runtime"]["device"],
                          "--gpu_index", str(request["runtime"]["gpu_index"])], output_dir / "render.log")
        rendered = _read_json(renders / "render_report.json")
        report["render"] = rendered
        if rendered["success"] != len(VIEW_IDS) or rendered["failed"]:
            raise RuntimeError("固定视角未全部成功，详见 render_report.json")
        images = [renders / item["image_file"] for item in rendered["results"]]
        for path, view in zip(images, views["views"]):
            with Image.open(path) as picture:
                picture.load()
                if list(picture.size) != view["resolution"]:
                    raise ValueError(f"画幅不匹配：{path}")
        _contact_sheet(images, output_dir / "contact_sheet.jpg")
        report.update(status="pass", blend_file=str(blend_file), images=[str(path) for path in images],
                      success=len(images), failed=0)
    except Exception as error:
        report["errors"].append(f"{type(error).__name__}: {error}")
        # Blender 失败也可能已经产出结构化报告及部分图片，保留真实分母和成功项。
        build_report = output_dir / "build_report.json"
        render_report = output_dir / "renders" / "render_report.json"
        if build_report.is_file():
            report["build"] = _read_json(build_report)
        if render_report.is_file():
            report["render"] = _read_json(render_report)
            rendered = report["render"]
            report.update(success=rendered["success"], failed=len(VIEW_IDS) - rendered["success"],
                          images=[item["image_file"] for item in rendered["results"] if item["status"] == "success"])
    _write_json(output_dir / "preview_report.json", report)
    return report


def _inspect(request: dict, session_id: str, report: dict, round_dir: Path) -> tuple[dict, dict]:
    images = [Path(path) for path in report["images"]]
    references = _reference_images(request)
    watched = [Path(request["output_dir"]) / name for name in ("build_scene.py", "views.json", "scene_plan.md", "assets.json")]
    watched += images + references + [Path(report["blend_file"]), round_dir / "preview" / "preview_report.json"]
    watched += sorted(Path(request["output_dir"]).rglob("*.py"))
    watched += sorted(path for path in (Path(request["output_dir"]) / "assets").rglob("*") if path.is_file())
    watched += [round_dir / "preview" / name for name in ("build_report.json", "scene_inventory.json", "views.json")]
    watched += [round_dir / "preview" / "renders" / "render_report.json"]
    before = {str(path): _sha256(path) for path in watched}
    _write_json(round_dir / "evidence.json", before)
    evidence = {"technical_report": str(round_dir / "preview" / "preview_report.json"),
                "current_images": [str(path) for path in images],
                "reference_images": [str(path) for path in references],
                "attachment_order": [str(path) for path in images + references]}
    turn = run_codex_turn(request, _prompt("inspect") + json.dumps(evidence, indent=2),
                          round_dir / "inspect", session_id=session_id, images=images + references,
                          output_schema=PROMPTS / "review.schema.json")
    if before != {str(path): _sha256(path) for path in watched}:
        raise RuntimeError("只读审阅改变了模板或证据，本轮判定不予接受")
    _check_host(request)
    review = json.loads(turn["reply"])
    for field, paths in (("inspected_images", images), ("reference_images", references)):
        expected = {path.name for path in paths}
        if len(review[field]) != len(expected) or set(review[field]) != expected:
            raise ValueError(f"审阅没有准确覆盖 {field}")
    if review["decision"] == "ready" and (review["issues"] or review["remaining_issues"]):
        raise ValueError("ready 不能仍有未解决问题")
    return review, turn


def _stage_assets(request: dict) -> None:
    reference = request["runtime"]["reference_bundle"]
    mode = request["runtime"].get("reference_mode", "adapt")
    if mode == "quality_only":
        (Path(request["output_dir"]) / "assets").mkdir()
        assets = {"reference_id": reference["id"], "reference_mode": mode, "files": []}
        _write_json(Path(request["output_dir"]) / "assets.json", assets)
        _write_json(Path(request["output_dir"]) / "workflow" / "source_assets.json", assets)
        return
    source = Path(reference["asset_directory"])
    target = Path(request["output_dir"]) / "assets" / reference["asset_subdir"]
    CONSOLE.print("复制参考工程与贴图：", str(target), markup=False)
    shutil.copytree(source, target)
    records = []
    manifest_path = Path(reference["asset_manifest"]["path"])
    if _sha256(manifest_path) != reference["asset_manifest"]["sha256"]:
        raise RuntimeError("参考资产清单摘要变化")
    for item in _read_json(manifest_path)["files"]:
        relative, digest = Path(item["path"]), item["sha256"]
        if _sha256(source / relative) != digest or _sha256(target / relative) != digest:
            raise RuntimeError(f"素材复制或源文件校验失败：{relative}")
        records.append({"path": str(Path("assets") / reference["asset_subdir"] / relative),
                        "bytes": item["bytes"], "sha256": digest})
    assets = {"reference_id": reference["id"], "source": reference["source"], "files": records}
    _write_json(Path(request["output_dir"]) / "assets.json", assets)
    _write_json(Path(request["output_dir"]) / "workflow" / "source_assets.json", assets)


def _check_host(request: dict) -> None:
    for filename, expected in request["host_files"].items():
        if _sha256(Path(filename)) != expected:
            raise RuntimeError(f"外围代码已变化，请记录新版本后另起任务：{filename}")
    for filename, expected in request.get("reference_files", {}).items():
        if _sha256(Path(filename)) != expected:
            raise RuntimeError(f"参考图片或说明已变化：{filename}")
    if request["runtime"].get("reference_mode", "adapt") == "quality_only":
        return
    reference = request["runtime"]["reference_bundle"]
    if _sha256(Path(reference["source_blend"])) != reference["source"]["sha256"]:
        raise RuntimeError("参考源工程摘要变化")
    manifest_path = Path(reference["asset_manifest"]["path"])
    if _sha256(manifest_path) != reference["asset_manifest"]["sha256"]:
        raise RuntimeError("参考资源清单摘要变化")
    copied = Path(request["output_dir"]) / "assets" / reference["asset_subdir"]
    if copied.exists():
        for item in _read_json(manifest_path)["files"]:
            if _sha256(copied / item["path"]) != item["sha256"]:
                raise RuntimeError(f"只读基础资产已变化：{item['path']}")
            if _sha256(Path(reference["asset_directory"]) / item["path"]) != item["sha256"]:
                raise RuntimeError(f"参考源素材已变化：{item['path']}")


def _check_template_assets(template_dir: Path) -> int:
    """构建前核对下载／新增文件；两种清单格式共用同一摘要检查。"""
    manifest = _read_json(template_dir / "assets.json")
    records = manifest["files"] if isinstance(manifest, dict) else manifest
    root = template_dir.resolve()
    for item in records:
        path = (root / item["path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"素材必须位于模板内：{item['path']}")
        if _sha256(path) != item["sha256"]:
            raise RuntimeError(f"素材摘要不符：{item['path']}")
    return len(records)


def _host_snapshot() -> dict:
    files = [DIRECTORY / name for name in ("room_workflow.py", "room_tools.py", "call_llm.py", "assets.py", "render.py", "geometry.py", "visibility.py")]
    files.append(PROMPTS.parent / "ASSET_TOOLS.md")
    files += sorted(path for path in PROMPTS.rglob("*") if path.is_file())
    return {str(path): _sha256(path) for path in files}


def _blender(request: dict, arguments: list[str], log_file: Path) -> None:
    runtime = request["runtime"]
    command = [runtime["blender"], "--background", "--threads", "8", "--python-exit-code", "1",
               "--python", runtime["room_tools"], "--", *arguments]
    _write_json(log_file.with_suffix(".command.json"), command)
    started = time.monotonic()
    with log_file.open("w", encoding="utf-8") as handle:
        process = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, text=True)
    if process.returncode:
        tail = log_file.read_text(encoding="utf-8", errors="replace")[-6000:]
        raise RuntimeError(f"Blender 退出码 {process.returncode}；{log_file}\n{tail}")
    CONSOLE.print(f"Blender {arguments[0]} 完成，{time.monotonic() - started:.1f} 秒")


def _contact_sheet(images: list[Path], destination: Path) -> None:
    sheet = Image.new("RGB", (1280, 3 * 510), "#181c21")
    draw = ImageDraw.Draw(sheet)
    for index, path in enumerate(images):
        with Image.open(path) as original:
            picture = original.convert("RGB")
            picture.thumbnail((640, 480))
        x, y = (index % 2) * 640, (index // 2) * 510
        sheet.paste(picture, (x, y))
        draw.text((x + 12, y + 486), path.stem, fill="white")
    sheet.save(destination, quality=92)


def _reference_images(request: dict) -> list[Path]:
    return [Path(item["path"]) for item in request["runtime"]["reference_bundle"]["images"]]


def _prompt(stage: str) -> str:
    return (PROMPTS / f"{stage}.md").read_text(encoding="utf-8") + "\n\n"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _turn_summary(turn: dict) -> dict:
    return {key: turn[key] for key in ("turn_dir", "session_id", "status", "usage")}


def _save_result(result: dict) -> None:
    _write_json(Path(result["template_dir"]) / "workflow_result.json", result)


def _finish(result: dict, status: str, issues: list[str]) -> dict:
    result.update(status=status, remaining_issues=issues)
    _save_result(result)
    CONSOLE.print("环境工作流：", status, markup=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--user_prompt_file", type=Path)
    parser.add_argument("--reference_file", type=Path)
    parser.add_argument("--reference_mode", choices=("adapt", "quality_only"), default="adapt")
    parser.add_argument("--asset_cache", type=Path)
    parser.add_argument("--asset_catalog", type=Path)
    parser.add_argument("--blender", type=Path)
    parser.add_argument("--model", default="gpt-6-astra")
    parser.add_argument("--reasoning_effort", default="ultra")
    parser.add_argument("--sandbox", default="workspace-write", choices=("workspace-write", "danger-full-access"))
    parser.add_argument("--device", default="OPTIX")
    parser.add_argument("--gpu_index", type=int, default=0)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--max_repairs", type=int)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--feedback_file", type=Path)
    args = parser.parse_args()
    if args.resume:
        feedback = args.feedback_file.read_text(encoding="utf-8") if args.feedback_file else None
        result = resume_room(args.output_dir, feedback=feedback, max_repairs=args.max_repairs)
    else:
        if not all((args.user_prompt_file, args.reference_file, args.blender)):
            parser.error("新任务需要 --user_prompt_file、--reference_file 和 --blender")
        result = generate_room(
            args.user_prompt_file.read_text(encoding="utf-8"), args.reference_file, args.output_dir, args.blender,
            reference_mode=args.reference_mode, asset_cache=args.asset_cache, asset_catalog=args.asset_catalog,
            model=args.model, reasoning_effort=args.reasoning_effort, sandbox=args.sandbox,
            device=args.device, gpu_index=args.gpu_index, samples=args.samples, seed=args.seed,
            max_repairs=3 if args.max_repairs is None else args.max_repairs, dry_run=args.dry_run,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] not in {"ready_for_review", "dry_run"}:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
