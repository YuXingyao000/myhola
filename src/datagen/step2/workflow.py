"""普通 Python 主循环：选区 → 测量 → 提议 → 检查/预览 → 看图 → 有限返修。

这里不导入 bpy。Agent 返回数据，Blender 子进程执行场景操作。
每个场景只创建一个新会话；通过的候选退出循环，其余最多修改两次。
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from PIL import Image, ImageDraw
from rich.console import Console

DIRECTORY = Path(__file__).resolve().parent
sys.path.insert(0, str(DIRECTORY.parent))
from call_llm import run_codex_turn

PROMPTS = DIRECTORY / "prompts"
CONSOLE = Console()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def scene_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_tool(request, action, output_dir, **inputs):
    """串行 CPU 工具调用。执行失败直接抛出；日志和 incomplete 状态已经落盘。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    job = {"action": action, "base_scene": request["base_scene"], "config": request["config"],
           "output_dir": str(output_dir), **inputs}
    job_file = output_dir / "job.json"
    write_json(job_file, job)
    command = [request["blender"], "--background", "--threads", str(request["config"]["threads"]),
               "--python-exit-code", "1", "--python", str(DIRECTORY / "tools.py"), "--", "--job", str(job_file)]
    write_json(output_dir / "command.json", command)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": str(request["config"]["threads"]),
           "OPENBLAS_NUM_THREADS": "1"}
    CONSOLE.print(f"[cyan]Blender · {action}[/cyan]  {output_dir.name} · CPU")
    started = time.monotonic()
    with (output_dir / "blender.log").open("w") as log:
        subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    report = read_json(output_dir / "report.json")
    report["elapsed_seconds"] = time.monotonic() - started
    write_json(output_dir / "report.json", report)
    return report


def annotate_views(context, overview_dir, output_dir):
    """保留原始分辨率；稀疏编号避免文字堆叠，完整索引仍在 scene_context.json。"""
    pictures = []
    for view in context["views"]:
        source = overview_dir / f"{view['id']}.png"
        with Image.open(source) as original:
            image = original.convert("RGB")
        if list(image.size) != view["resolution"]:
            raise ValueError(f"参考图尺寸与 views.json 不一致：{source}")
        draw, occupied = ImageDraw.Draw(image), set()
        for item in context["label_positions"][view["id"]]:
            x, y = (int(value) for value in item["pixel"])
            cell = (x // 42, y // 22)
            if cell in occupied:
                continue
            occupied.add(cell)
            label = str(item["label"])
            draw.rectangle((x, y, x + 8 * len(label) + 4, y + 15), fill="#17222b")
            draw.text((x + 2, y + 1), label, fill="#ffe38a")
        target = output_dir / f"{view['id']}.indexed.png"
        image.save(target)
        pictures.extend([str(source.resolve()), str(target)])
    return pictures


def save_state(request, state):
    """每个阶段前后保存；进程中断时保留已经通过的项，其余仍是 incomplete。"""
    write_json(Path(request["output_dir"]) / "placements.json", state)


def ask_agent(request, state, stage, inputs, turn_dir, images):
    state["stage"] = stage
    save_state(request, state)
    prompt = (PROMPTS / f"{stage}.md").read_text(encoding="utf-8")
    prompt += "\n\n" + json.dumps(inputs, ensure_ascii=False, indent=2)
    turn = run_codex_turn(request, prompt, turn_dir, session_id=state["session_id"],
                          images=[Path(path) for path in images], output_schema=PROMPTS / f"{stage}.schema.json")
    state["session_id"] = turn["session_id"]
    state["codex_turns"].append({key: turn[key] for key in ("turn_dir", "status", "usage")})
    save_state(request, state)
    # Agent 不应编辑基础工程；此处检测修改，不声称提供 OS 级隔离。
    if scene_hash(request["base_scene"]) != state["scene_hash"]:
        raise RuntimeError("基础场景在 Agent 调用期间被修改，停止本次运行")
    return json.loads(turn["reply"])


def match_regions(items, expected_ids):
    """唯一且完整的候选对应，避免漏项被当作成功，也避免重复消费返修预算。"""
    ids = [item["region_id"] for item in items]
    if len(ids) != len(set(ids)) or set(ids) != set(expected_ids):
        raise ValueError(f"Agent 返回的候选不匹配：{ids}，预期 {expected_ids}")
    return {item["region_id"]: item for item in items}


def prepare_supports(request, state, regions):
    """选区后、测量前建立本轮副本；后续工具与导出统一使用这个路径和摘要。"""
    source_scene, source_hash = request["base_scene"], state["scene_hash"]
    report = run_tool(request, "prepare_supports", Path(request["output_dir"]) / "support_scene", regions=regions)
    prepared_hash = scene_hash(report["base_scene"])
    request.update(source_scene=source_scene, source_scene_hash=source_hash, base_scene=report["base_scene"],
                   support_preparation=report)
    state.update(source_scene=source_scene, source_scene_hash=source_hash, base_scene=report["base_scene"],
                 scene_hash=prepared_hash, support_preparation=report)
    write_json(Path(request["output_dir"]) / "request.json", request)
    save_state(request, state)


def run_workflow(request, state, context, images):
    """主循环在这里。每轮一批提议、一批看图；几何工具逐位置串行执行。"""
    output_dir, config = Path(request["output_dir"]), request["config"]
    selected = ask_agent(request, state, "select_regions", {
        "scene_context": str(output_dir / "scene_context.json"), "images": images,
        "max_regions": config["max_regions"], "config": config, "user_request": request["user_request"],
    }, output_dir / "select_regions", images)
    regions = selected["regions"]
    ids = [region["region_id"] for region in regions]
    match_regions(regions, ids)
    if len(regions) > config["max_regions"]:
        raise ValueError("候选数量超过 max_regions")
    state["candidates"] = [{"region": region, "status": "incomplete", "attempts": []} for region in regions]
    state["stage"] = "prepare_supports"
    save_state(request, state)
    prepare_supports(request, state, regions)
    state["stage"] = "measure"
    save_state(request, state)
    measurements = run_tool(request, "measure", output_dir / "measurement", regions=regions, views=context["views"])
    for candidate, measurement in zip(state["candidates"], measurements["regions"]):
        candidate["measurement"] = measurement
        if measurement["status"] != "measured":
            candidate.update(status="rejected", reason=measurement["status"])
    save_state(request, state)

    for revision in range(config["max_revisions"] + 1):
        active = [item for item in state["candidates"] if item["status"] == "incomplete"]
        if not active:
            break
        round_dir = output_dir / f"round_{revision:02d}"
        round_dir.mkdir()
        # 全量采样保留在文件中；给 Agent 的摘要分别保留空位和可挪物的位置。
        summaries = []
        for candidate in active:
            measurement = candidate["measurement"]
            example_sites = []
            for kind in ("free", "movable", "fixed_or_unknown"):
                sites = [site for site in measurement["sites"] if site["kind"] == kind]
                example_sites.extend(sites[::max(1, (len(sites) + 29) // 30)])
            summaries.append({"region": candidate["region"], "measurement": {**measurement, "sites": example_sites},
                              "previous_attempt": candidate["attempts"][-1] if candidate["attempts"] else None})
        proposals = ask_agent(request, state, "propose_setup", {
            "config": config, "revision": revision, "regions": summaries,
            "base_scene": request["base_scene"], "support_preparation": request["support_preparation"],
            "full_measurements": str(output_dir / "measurement" / "report.json"),
        }, round_dir / "propose_setup", [])
        by_id = match_regions(proposals["setups"], [item["region"]["region_id"] for item in active])
        reports, review_images = [], []
        for index, candidate in enumerate(active):
            region = candidate["region"]
            proposal = by_id[region["region_id"]]
            attempt = {"revision": revision, "proposal": proposal}
            candidate["attempts"].append(attempt)
            state["stage"] = "preview"
            save_state(request, state)
            report = run_tool(request, "preview", round_dir / f"candidate_{index:02d}", region=region, proposal=proposal)
            attempt["numeric_report"] = report
            reports.append(report)
            review_images.extend(report["images"].values())
            save_state(request, state)
        reviews = ask_agent(request, state, "inspect_setup", {"reports": reports, "config": config},
                            round_dir / "inspect_setup", review_images)
        by_id = match_regions(reviews["reviews"], [item["region"]["region_id"] for item in active])
        for candidate in active:
            review = by_id[candidate["region"]["region_id"]]
            attempt = candidate["attempts"][-1]
            attempt["visual_review"] = review
            report = attempt["numeric_report"]
            # 图片已作为输入发送；这里只核对 Agent 声明的查看路径，不证明视觉判断正确。
            images_acknowledged = bool(report["images"]) and set(report["images"].values()) <= set(review["inspected_images"])
            if report["numeric_pass"] and images_acknowledged and review["decision"] == "keep":
                candidate.update(status="proxy_ready", selected_revision=revision)
            elif review["decision"] == "drop" or revision == config["max_revisions"]:
                candidate.update(status="rejected", reason=review["reason"], numeric_reasons=report["reasons"])
        save_state(request, state)
    state.update(status="completed", stage="export", human_review="pending")
    export_placements(request, state)
    return state


def export_placements(request, state):
    """紧凑可复用位置清单 + 全部候选/尝试；不复制场景或叠加多个位置的移动。"""
    ready = []
    for candidate in state["candidates"]:
        if candidate["status"] != "proxy_ready":
            continue
        attempt = candidate["attempts"][-1]
        ready.append({"region": candidate["region"], "setup": attempt["proposal"],
                      "transforms": attempt["numeric_report"]["transforms"],
                      "support_validation": attempt["numeric_report"]["checks"]["rendered_support"],
                      "camera": attempt["numeric_report"]["camera"], "images": attempt["numeric_report"]["images"]})
    write_json(Path(request["output_dir"]) / "ready.json", {
        "base_scene": request["base_scene"], "scene_hash": state["scene_hash"], "config": request["config"],
        "source_scene": request["source_scene"], "source_scene_hash": request["source_scene_hash"],
        "support_preparation": request["support_preparation"],
        "human_review": "pending", "placements": ready,
    })
    save_state(request, state)
    CONSOLE.print(f"[green]Step 2 完成[/green]：{len(ready)} / {len(state['candidates'])} 个候选通过代理检查；等待人工审阅。")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_scene", type=Path, required=True)
    parser.add_argument("--views_file", type=Path, required=True)
    parser.add_argument("--overview_dir", type=Path, required=True, help="与基础场景对应、按 view ID 命名的现有 PNG")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--blender", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DIRECTORY / "config.json")
    parser.add_argument("--user_prompt_file", type=Path)
    parser.add_argument("--model", default="gpt-6-astra")
    parser.add_argument("--reasoning_effort", default="ultra")
    parser.add_argument("--sandbox", choices=("workspace-write", "danger-full-access"), default="workspace-write")
    parser.add_argument("--prepare_only", action="store_true", help="只导出索引与输入，不调用 Agent、不渲染")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    request = {
        "base_scene": str(args.base_scene.resolve()), "views_file": str(args.views_file.resolve()),
        "overview_dir": str(args.overview_dir.resolve()), "output_dir": str(output_dir),
        "blender": str(args.blender.resolve()), "config": read_json(args.config),
        "model": args.model, "reasoning_effort": args.reasoning_effort, "sandbox": args.sandbox,
        "developer_instructions": (PROMPTS / "CONTRACT.md").read_text(encoding="utf-8"),
        "user_request": args.user_prompt_file.read_text(encoding="utf-8") if args.user_prompt_file else "",
        "tool_sources": {str(path): scene_hash(path) for path in sorted(DIRECTORY.glob("*.py"))},
        # 兼容已有调用器的目录字段；Step 2 不下载素材。
        "runtime": {"asset_cache": str(output_dir / ".context")},
    }
    write_json(output_dir / "request.json", request)
    state = {"status": "incomplete", "stage": "describe", "base_scene": request["base_scene"],
             "scene_hash": scene_hash(args.base_scene), "config": request["config"], "session_id": None,
             "human_review": "pending", "codex_turns": [], "candidates": []}
    save_state(request, state)
    views = read_json(args.views_file)["views"]
    context = run_tool(request, "describe", output_dir / "context", views=views)
    images = annotate_views(context, args.overview_dir, output_dir / "context")
    write_json(output_dir / "scene_context.json", context)
    if args.prepare_only:
        state.update(status="prepared", stage="describe")
        save_state(request, state)
        CONSOLE.print(f"输入已准备：{output_dir}；未调用 Agent、未渲染。正式运行请使用新输出目录。")
    else:
        run_workflow(request, state, context, images)


if __name__ == "__main__":
    main()
