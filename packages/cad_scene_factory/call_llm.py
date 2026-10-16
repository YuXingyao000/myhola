import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from rich.console import Console

PROMPTS_DIR = Path(__file__).with_name("prompts")
DEVELOPER_INSTRUCTIONS = (PROMPTS_DIR / "DEVELOPER_INSTRUCTIONS.md").read_text(
    encoding="utf-8"
).strip()


def _workflow_context(output_dir: Path, blender, stl_dir, panel_file, asset_cache, asset_catalog, device, gpu_index):
    """显式组合本项目指南和工具位置，不依赖个人 skill 自动发现。"""
    directory = Path(__file__).resolve().parent
    documents = {}
    sections = []
    for name in ("SCENE_GUIDE.md", "ACCEPTANCE.md", "ASSET_TOOLS.md"):
        content = (PROMPTS_DIR / name).read_text(encoding="utf-8")
        documents[name] = {"sha256": hashlib.sha256(content.encode()).hexdigest(), "content": content}
        sections.append(f"# Project document: {name}\n\n{content}")

    runtime = {
        "python": sys.executable,
        "assets_script": str(directory / "assets.py"),
        "asset_cache": str(asset_cache),
        "asset_catalog": str(asset_catalog.resolve()) if asset_catalog else None,
        "preview_script": str(directory / "preview.py"),
        "blender": str(blender.resolve()) if blender else None,
        "stl_dir": str(stl_dir.resolve()) if stl_dir else None,
        "panel_file": str(panel_file.resolve()),
        "template_dir": str(output_dir),
        "device": device,
        "gpu_index": gpu_index,
    }
    preview_command = None
    if blender is not None and stl_dir is not None:
        preview_command = shlex.join([
            sys.executable, runtime["preview_script"],
            "--template_dir", str(output_dir), "--blender", runtime["blender"],
            "--stl_dir", runtime["stl_dir"], "--panel_file", runtime["panel_file"],
            "--preset", "quick",
            "--device", device, "--gpu_index", str(gpu_index),
        ])
    # 静态工作要求放在Markdown；Python只填入本次工具路径和命令。
    runtime_template = (PROMPTS_DIR / "RUNTIME.md").read_text(encoding="utf-8")
    documents["RUNTIME.md"] = {
        "sha256": hashlib.sha256(runtime_template.encode()).hexdigest(), "content": runtime_template,
    }
    sections.append(runtime_template.format(
        runtime_json=json.dumps(runtime, ensure_ascii=False, indent=2),
        quick_preview_command=preview_command or "(not configured)",
    ))
    return "\n\n".join(sections), documents, runtime


def _disabled_skills_config(workspace: Path) -> str:
    """仅扫描技能文件路径，生成本次调用禁用这些技能的 TOML 配置。"""
    codex_home = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))
    skill_roots = {
        codex_home / "skills",
        Path.home() / ".agents" / "skills",
        Path("/etc/codex/skills"),
    }
    for directory in (workspace, *workspace.parents):
        skill_roots.add(directory / ".agents" / "skills")
        skill_roots.add(directory / ".codex" / "skills")
    skill_files = sorted({
        path.resolve()
        for root in skill_roots
        for path in root.rglob("SKILL.md")
    })
    # Codex 0.160.1 实际按 SKILL.md 的完整路径匹配；目录路径不会生效。
    entries = [
        "{path=" + json.dumps(str(path), ensure_ascii=False) + ",enabled=false}"
        for path in skill_files
    ]
    return "[" + ",".join(entries) + "]"


def prepare_request(
    developer_instructions: str,
    user_prompt: str,
    output_dir: Path,
    image_file: Path | None = None,
    model: str = "gpt-6-astra",
    reasoning_effort: str = "ultra",
    sandbox: str = "workspace-write",
    *,
    blender: Path | None = None,
    stl_dir: Path | None = None,
    panel_file: Path = Path(__file__).with_name("review_models.json"),
    asset_cache: Path | None = None,
    asset_catalog: Path | None = None,
    device: str = "OPTIX",
    gpu_index: int = 0,
) -> dict:
    """组装本项目的完整输入；不创建目录，不启动 Codex，也不执行预览。"""
    output_dir = output_dir.resolve()
    if (blender is None) != (stl_dir is None):
        raise ValueError("blender 与 stl_dir 需要一起提供")
    asset_cache = (asset_cache or output_dir / ".asset_cache").resolve()
    workflow, documents, runtime = _workflow_context(
        output_dir, blender, stl_dir, panel_file, asset_cache, asset_catalog, device, gpu_index,
    )

    return {
        "model": model,
        "reasoning_effort": reasoning_effort,
        "sandbox": sandbox,
        "developer_instructions": developer_instructions + "\n\n" + workflow,
        "instruction_files": documents,
        "runtime": runtime,
        "review_panel": json.loads(panel_file.read_text(encoding="utf-8")),
        "prompt": user_prompt,
        "output_dir": str(output_dir),
        "reference_image": str(image_file.resolve()) if image_file is not None else None,
    }


def _codex_command(request, turn_dir, session_id, images, output_schema):
    """首次用 exec；后续显式 resume 指定会话，绝不选择最近的会话。"""
    output_dir = Path(request["output_dir"])
    asset_cache = Path(request["runtime"]["asset_cache"])
    # 这些参数属于 exec；resume 子命令本身没有 --cd/--sandbox/--add-dir。
    command = [
        "codex", "exec",
        "--cd", request["output_dir"],
        "--sandbox", request["sandbox"],
    ]
    if not asset_cache.is_relative_to(output_dir):
        command.extend(["--add-dir", str(asset_cache)])
    if session_id is not None:
        command.append("resume")

    # 首轮和续接轮都显式设置认证、模型和上下文来源，不依赖会话的隐含配置。
    command.extend([
        "--model", request["model"],
        "--skip-git-repo-check",
        "--ignore-user-config", "--ignore-rules",
        "-c", 'forced_login_method="chatgpt"',
        "-c", 'model_provider="openai"',
        "-c", "model_reasoning_effort=" + json.dumps(request["reasoning_effort"]),
        "-c", "project_doc_max_bytes=0",
        "-c", "developer_instructions=" + json.dumps(request["developer_instructions"], ensure_ascii=False),
        # 禁用与本任务无关的个人上下文来源；保留执行代码所需的工具。
        "-c", "features.memories=false",
        "-c", "memories.use_memories=false",
        "-c", "memories.generate_memories=false",
        "-c", "features.hooks=false",
        "-c", "features.apps=false",
        "-c", "features.plugins=false",
        "-c", "features.remote_plugin=false",
        "-c", "features.skill_search=false",
        "-c", "skills.config=" + _disabled_skills_config(output_dir),
        "--json", "-o", str(turn_dir / "reply.txt"),
    ])
    if output_schema is not None:
        command.extend(["--output-schema", str(output_schema.resolve())])
    for image in images or []:
        command.extend(["--image", str(image.resolve())])
    # 保留本次会话供后续阶段续接，不使用 --ephemeral。
    command.extend(["--", *([session_id] if session_id is not None else []), "-"])
    return command


def run_codex_turn(
    request: dict,
    prompt: str,
    turn_dir: Path,
    *,
    session_id: str | None = None,
    images: list[Path] | None = None,
    output_schema: Path | None = None,
) -> dict:
    """执行一个阶段，保存原始事件，再返回会话 ID、最终回复和用量。

    首轮省略 session_id；后续传入首轮事件给出的精确 ID。每轮使用独立的新目录。
    images 只包含这一轮要提供的图片，首次参考图也由调用方显式传入。
    输出 schema 限制最终回复格式；本函数不判断布景和审阅内容是否通过。
    启动、进程或事件校验失败时，先保留日志/result.json，再向外围抛出异常。

    强制 ChatGPT 登录并剥离模型 API key；额度与额外 credits 的扣除由账号侧决定。
    首轮不带入其他会话，续接只保留本次任务历史。关闭个人规则、记忆、skills、
    插件和 hooks 自动注入；Codex 内置指令、工具说明和本任务工具结果仍然存在。
    """
    output_dir = Path(request["output_dir"])
    turn_dir = turn_dir.resolve()
    turn_dir.mkdir(parents=True, exist_ok=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    Path(request["runtime"]["asset_cache"]).mkdir(parents=True, exist_ok=True)
    (turn_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
    command = _codex_command(request, turn_dir, session_id, images, output_schema)

    environment = os.environ.copy()
    environment.pop("CODEX_API_KEY", None)
    environment.pop("OPENAI_API_KEY", None)
    console = Console(stderr=True)
    console.rule("[bold cyan]Codex · " + turn_dir.name + "[/bold cyan]", style="cyan")
    console.print("模型：", request["model"], "；推理强度：", request["reasoning_effort"], markup=False)
    console.print("ChatGPT 登录；" + (f"续接会话 {session_id}" if session_id else "创建新会话"), markup=False)
    console.print("执行模式：", request["sandbox"], markup=False)
    console.print("模板目录：", str(output_dir), markup=False)

    result = {
        "session_id": session_id, "requested_session_id": session_id,
        "reply": "", "usage": None, "turn_dir": str(turn_dir),
        "status": "failed", "returncode": None,
    }
    try:
        # 运行中直接落盘，保留长任务的增量日志。
        with (turn_dir / "events.jsonl").open("w", encoding="utf-8") as events_file, \
                (turn_dir / "stderr.log").open("w", encoding="utf-8") as stderr_file, \
                console.status("[cyan]Codex 正在运行，请稍候…[/cyan]", spinner="dots"):
            process = subprocess.run(
                command, input=prompt, cwd=output_dir, env=environment,
                text=True, encoding="utf-8", stdout=events_file, stderr=stderr_file,
            )
        result["returncode"] = process.returncode
        thread_ids = set()
        completed = False
        failed = False
        turn_error = None
        for line in (turn_dir / "events.jsonl").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            if event["type"] == "thread.started":
                thread_ids.add(event["thread_id"])
                result["session_id"] = event["thread_id"]
            elif event["type"] == "turn.completed":
                completed = True
                result["usage"] = event.get("usage")
            elif event["type"] == "turn.failed":
                failed = True
                turn_error = event.get("error")
        reply_file = turn_dir / "reply.txt"
        if reply_file.is_file():
            result["reply"] = reply_file.read_text(encoding="utf-8").strip()
        process.check_returncode()
        if len(thread_ids) != 1:
            raise RuntimeError("Codex 事件中缺少唯一的 thread.started 会话 ID")
        if session_id is not None and result["session_id"] != session_id:
            raise RuntimeError(f"续接会话不匹配：请求 {session_id}，实际 {result['session_id']}")
        if failed or not completed:
            raise RuntimeError(f"Codex 阶段未完成：{turn_error or '缺少 turn.completed 事件'}")
        if not result["reply"]:
            raise RuntimeError("Codex 没有写出最终回复 reply.txt")
        result["status"] = "completed"
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
        console.print("本阶段失败；记录已保留：", str(turn_dir), style="bold red", markup=False)
        raise
    finally:
        (turn_dir / "result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return result


def call_llm(
    developer_instructions: str,
    user_prompt: str,
    output_dir: Path,
    image_file: Path | None = None,
    model: str = "gpt-6-astra",
    reasoning_effort: str = "ultra",
    sandbox: str = "workspace-write",
    *,
    blender: Path | None = None,
    stl_dir: Path | None = None,
    panel_file: Path = Path(__file__).with_name("review_models.json"),
    asset_cache: Path | None = None,
    asset_catalog: Path | None = None,
    device: str = "OPTIX",
    gpu_index: int = 0,
    dry_run: bool = False,
) -> str:
    """保留旧的一次生成接口；完整分阶段流程由 workflow.py 管理。"""
    request = prepare_request(
        developer_instructions, user_prompt, output_dir, image_file, model, reasoning_effort, sandbox,
        blender=blender, stl_dir=stl_dir, panel_file=panel_file, asset_cache=asset_cache,
        asset_catalog=asset_catalog, device=device, gpu_index=gpu_index,
    )
    output_dir = Path(request["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "request.json").write_text(
        json.dumps(request, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if dry_run:
        return f"输入已保存：{output_dir / 'request.json'}；未启动 Codex。"
    result = run_codex_turn(
        request, user_prompt, output_dir / "turns" / "generation",
        images=[image_file] if image_file is not None else None,
    )
    (output_dir / "reply.txt").write_text(result["reply"] + "\n", encoding="utf-8")
    # 正常退出只说明本轮对话完成；兼容入口仍需确认生成文件确实存在。
    missing_files = [
        filename
        for filename in ("build_scene.py", "build_material.py")
        if not (output_dir / filename).is_file()
    ]
    if missing_files:
        raise FileNotFoundError("缺少模板脚本：" + ", ".join(missing_files))
    return result["reply"]


def main():
    """原命令行入口保持可用，统一交给外围 workflow。"""
    from workflow import main as workflow_main
    workflow_main()


if __name__ == "__main__":
    main()
