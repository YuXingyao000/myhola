"""普通 Python 入口：制作模板 → 外围预览 → 看图审阅 → 有界返修。

先读 run_workflow()。三个 Codex 动作在 create_template()、inspect_template()
和 repair_template()；实际渲染复用 preview.py，不在本文件实现 Blender 逻辑。
"""

import argparse
import json
from pathlib import Path

from rich.console import Console

from call_llm import DEVELOPER_INSTRUCTIONS, PROMPTS_DIR, prepare_request, run_codex_turn
from preview import file_sha256, preview_template, snapshot_inputs


CONSOLE = Console(stderr=True)
REVIEW_SCHEMA = json.loads((PROMPTS_DIR / "review.schema.json").read_text(encoding="utf-8"))


def run_workflow(request: dict, *, max_repairs: int = 2, dry_run: bool = False) -> dict:
    """每个模板新建会话；同一模板只续接该会话，最多首版加 max_repairs 轮。

    ready_for_review 仅表示可交给人看。异常、阻塞和预算耗尽均保存 incomplete，
    不重试登录或CLI错误。max_repairs 不限制Agent内部工具次数或token消耗。
    """
    template_dir = Path(request["output_dir"])
    records_dir = template_dir / "workflow"
    if max_repairs < 0:
        raise ValueError("max_repairs 不能小于0")
    if records_dir.exists() or any((template_dir / name).exists() for name in ("build_scene.py", "build_material.py")):
        raise FileExistsError("输出目录已有模板或工作流记录，请为本次任务指定新目录")
    template_dir.mkdir(parents=True, exist_ok=True)
    request = dict(request, max_repairs=max_repairs)
    # 在首次Agent动作前固定外围代码和测试组，避免把改过的检查器当作新基线。
    baseline = snapshot_inputs(template_dir, Path(request["runtime"]["panel_file"]))
    request["protected_inputs"] = {key: baseline[key] for key in ("host_files", "panel_sha256")}
    _write_json(template_dir / "request.json", request)
    if dry_run:
        return {"status": "dry_run", "request_file": str(template_dir / "request.json")}

    records_dir.mkdir()
    _write_json(records_dir / "review_schema.json", REVIEW_SCHEMA)
    result = {"status": "running", "stage": "create", "session_id": None,
              "template_dir": str(template_dir), "max_repairs": max_repairs,
              "repairs_used": 0, "human_review": "pending", "rounds": []}
    _save_result(template_dir, result)
    try:
        # 1、2. 首次任务先记录方案，再制作模板。
        session_id = create_template(request, records_dir)
        result["session_id"] = session_id
        _check_host_inputs(request)
        if request["runtime"]["blender"] is None:
            missing = _missing_deliverables(template_dir)
            return _finish(result, "incomplete" if missing else "generated_unverified", missing)

        for revision in range(max_repairs + 1):
            round_dir = records_dir / f"round_{revision:02d}"
            round_dir.mkdir()
            result.update(stage="preview", repairs_used=revision)
            _save_result(template_dir, result)

            # 3. 技术报告始终来自外围程序；失败样本保留在预定分母里。
            CONSOLE.rule(f"外围检查 · 第 {revision + 1} 版", style="cyan")
            report = _preview(request, round_dir / "preview")
            feedback = _technical_feedback(report, _missing_deliverables(template_dir))
            round_record = {"revision": revision, "report": str(round_dir / "preview" / "preview_report.json"),
                            "technical": report["technical"], "counts": report["counts"]}
            result["rounds"].append(round_record)
            _save_result(template_dir, result)
            _check_host_inputs(request)

            # 4. 技术成功且文件齐备后才花额度看图；审阅阶段不允许改模板。
            if feedback is None:
                result["stage"] = "inspect"
                _save_result(template_dir, result)
                feedback = inspect_template(request, session_id, report, round_dir)
            _write_json(round_dir / "review.json", feedback)
            round_record.update(decision=feedback["decision"], review=str(round_dir / "review.json"))

            # 5. 当前版本满足条件即可交付；需要修复才继续下一轮。
            if feedback["decision"] == "ready":
                return _finish(result, "ready_for_review", [])
            if feedback["decision"] == "blocked" or revision == max_repairs:
                return _finish(result, "incomplete", feedback["remaining_issues"])
            result["stage"] = "repair"
            result["repairs_used"] = revision + 1
            _save_result(template_dir, result)
            repair_template(request, session_id, report, feedback, round_dir, repair_number=revision + 1)
            _check_host_inputs(request)
    except Exception as error:
        # 只在任务边界收集异常：保留失败阶段和证据，CLI最终以非零状态退出。
        result["error"] = f"{type(error).__name__}: {error}"
        return _finish(result, "incomplete", [result["error"]])
    except KeyboardInterrupt:
        _finish(result, "incomplete", ["工作流被中断；已有文件与检查记录已保留"])
        raise


def create_template(request: dict, records_dir: Path) -> str:
    """新建一次会话，返回后续只能续接的明确ID。"""
    reference = request["reference_image"]
    turn = run_codex_turn(
        request, _stage_prompt("create") + request["prompt"], records_dir / "create",
        images=[Path(reference)] if reference else [],
    )
    return turn["session_id"]


def inspect_template(request: dict, session_id: str, report: dict, round_dir: Path) -> dict:
    """把全部实际预览附到审阅回合，并核对回复与当前版本的对应关系。

    前后摘要检测改动，不提供操作系统级只读隔离；图片清单也只能核对Agent
    声明的审阅覆盖，不能证明其视觉判断正确。
    """
    preview_dir = Path(report["output_dir"])
    images = [preview_dir / "renders" / item["image_file"] for item in report["results"]]
    evidence_files = {
        str(preview_dir / name): file_sha256(preview_dir / name)
        for name in ("preview_report.json", "scene.blend", "build_report.json", "request.json",
                     "panel.json", "renders/renders.jsonl", "contact_sheet.png")
    }
    if any(digest is None for digest in evidence_files.values()):
        raise ValueError("完整技术预览缺少构建产物或记录，不能进入视觉审阅")
    _write_json(round_dir / "evidence.json", evidence_files)
    evidence = {"report": _report_context(report), "images_in_attachment_order": [str(path) for path in images]}
    turn = run_codex_turn(
        request, _stage_prompt("inspect") + json.dumps(evidence, indent=2), round_dir / "inspect",
        session_id=session_id, images=images,
        output_schema=round_dir.parent / "review_schema.json",
    )
    _check_evidence_unchanged(request, report, evidence_files)
    review = json.loads(turn["reply"])
    _validate_review(review, [path.name for path in images])
    return review


def repair_template(request: dict, session_id: str, report: dict, feedback: dict, round_dir: Path,
                    *, repair_number: int) -> None:
    """沿用本次模板会话，带着具体失败或视觉问题进行一次外围返修。"""
    evidence = {"report": _report_context(report), "review": feedback,
                "repair_number": repair_number, "max_repairs": request["max_repairs"]}
    run_codex_turn(
        request, _stage_prompt("repair") + json.dumps(evidence, indent=2), round_dir / "repair",
        session_id=session_id,
    )


def _stage_prompt(stage: str) -> str:
    """阶段要求直接读Markdown；在末尾追加本轮需求或检查结果。"""
    return (PROMPTS_DIR / "stages" / f"{stage}.md").read_text(encoding="utf-8")


def _preview(request: dict, output_dir: Path) -> dict:
    runtime = request["runtime"]
    return preview_template(
        Path(request["output_dir"]), blender=Path(runtime["blender"]),
        stl_dir=Path(runtime["stl_dir"]), panel_file=Path(runtime["panel_file"]),
        preset="review", output_dir=output_dir,
        device=runtime["device"], gpu_index=runtime["gpu_index"],
    )


def _missing_deliverables(template_dir: Path) -> list[str]:
    return [f"Missing or empty file: {name}" for name in ("scene_plan.md", "build_scene.py", "build_material.py")
            if not (template_dir / name).is_file() or not (template_dir / name).read_text(encoding="utf-8").strip()]


def _technical_feedback(report: dict, missing: list[str]) -> dict | None:
    if report["technical"] == "pass" and not missing:
        return None
    problems = list(missing)
    if report["technical"] != "pass":
        problems.append("Host technical checks failed; inspect the attached build/render report and logs.")
        problems.extend(f"{item['image_file']}: {item.get('error', item['status'])}"
                        for item in report["results"] if item["status"] != "success")
    # 启动器或固定数据出错不属于布景任务，避免花模型额度反复修模板。
    host_failure = (report.get("model_inputs_match_panel") is False
                    or any(report.get(stage, {}).get("error") for stage in ("build", "render")))
    return {"decision": "blocked" if host_failure else "revise", "source": "host",
            "summary": "Host technical checks failed or deliverables are missing; visual review not run.",
            "inspected_images": [], "issues": [], "remaining_issues": problems}


def _report_context(report: dict) -> dict:
    """提示词只带结论、错误和文件位置；完整摘要/逐图记录保留在原报告。"""
    return {
        "report_file": str(Path(report["output_dir"]) / "preview_report.json"),
        "technical": report["technical"], "counts": report["counts"],
        "checks_not_run": report.get("checks_not_run", []),
        "failures": [{"image": item["image_file"], "error": item.get("error", item["status"])}
                     for item in report["results"] if item["status"] != "success"],
    }


def _validate_review(review: dict, expected_images: list[str]) -> None:
    """模型回复是外部输入；在这个边界集中检查，不把格式正确等同于质量正确。"""
    if not isinstance(review, dict) or set(review) != set(REVIEW_SCHEMA["required"]):
        raise ValueError("审阅回复字段不符合review schema")
    if review["decision"] not in ("ready", "revise", "blocked") or not isinstance(review["summary"], str):
        raise ValueError("审阅decision或summary无效")
    for name in ("inspected_images", "remaining_issues"):
        if not isinstance(review[name], list) or any(not isinstance(item, str) for item in review[name]):
            raise ValueError(f"审阅{name}必须是字符串列表")
    if sorted(review["inspected_images"]) != sorted(expected_images):
        raise ValueError("审阅图片清单没有逐一对应当前全部预览")
    issue_keys = {"image", "region", "observation", "suggested_change"}
    if not isinstance(review["issues"], list):
        raise ValueError("审阅issues必须是列表")
    for issue in review["issues"]:
        if (not isinstance(issue, dict) or set(issue) != issue_keys
                or any(not isinstance(value, str) or not value.strip() for value in issue.values())
                or issue["image"] not in expected_images):
            raise ValueError("审阅问题需要对应图片、区域、观察和修改建议")
    if review["decision"] == "ready" and (review["issues"] or review["remaining_issues"]):
        raise ValueError("ready不能同时包含待解决的问题")
    if review["decision"] != "ready" and not (review["issues"] or review["remaining_issues"]):
        raise ValueError("revise/blocked必须说明具体问题")
    if review["decision"] != "ready" and not review["remaining_issues"]:
        review["remaining_issues"] = [issue["observation"] for issue in review["issues"]]


def _check_evidence_unchanged(request: dict, report: dict, evidence_files: dict) -> None:
    current = snapshot_inputs(Path(request["output_dir"]), Path(request["runtime"]["panel_file"]))
    if current != report["snapshot_after"]:
        raise ValueError("审阅期间模板、素材、测试配置或外围代码改变；旧预览不可交付")
    for name, digest in evidence_files.items():
        if file_sha256(Path(name)) != digest:
            raise ValueError(f"审阅期间构建产物或报告改变：{name}；本轮证据无效")
    for item in report["results"]:
        image_path = Path(report["output_dir"]) / "renders" / item["image_file"]
        if file_sha256(image_path) != item["image_sha256"]:
            raise ValueError(f"审阅期间图片改变：{item['image_file']}")
        if item["kind"] == "model" and file_sha256(Path(item["model_file"])) != item["final_stl_sha256"]:
            raise ValueError(f"审阅期间源STL改变：{item['model_id']}")


def _check_host_inputs(request: dict) -> None:
    current = snapshot_inputs(Path(request["output_dir"]), Path(request["runtime"]["panel_file"]))
    if any(current[key] != value for key, value in request["protected_inputs"].items()):
        raise ValueError("外围代码或固定测试组在Agent执行后改变；停止工作流，不采用新基线")


def _finish(result: dict, status: str, remaining_issues: list[str]) -> dict:
    result.update(status=status, remaining_issues=remaining_issues)
    if result["rounds"]:
        result["last_report"] = result["rounds"][-1]["report"]
        if status == "ready_for_review":
            result["final_report"] = result["last_report"]
    _save_result(Path(result["template_dir"]), result)
    CONSOLE.print(f"工作流状态：{status}", markup=False)
    return result


def _save_result(template_dir: Path, result: dict) -> None:
    # 用量来自CLI事件；不推算账号费用，也不将重试上限解释成token预算。
    result["codex_turns"] = []
    for path in sorted((template_dir / "workflow").rglob("result.json")):
        turn = json.loads(path.read_text(encoding="utf-8"))
        result["codex_turns"].append({key: turn.get(key) for key in ("turn_dir", "status", "usage")})
    _write_json(template_dir / "workflow_result.json", result)


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    prompt = parser.add_mutually_exclusive_group(required=True)
    prompt.add_argument("--user_prompt")
    prompt.add_argument("--user_prompt_file", type=Path)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--image_file", type=Path)
    parser.add_argument("--model", default="gpt-6-astra")
    parser.add_argument("--reasoning_effort", default="ultra")
    parser.add_argument("--sandbox", choices=("workspace-write", "danger-full-access"), default="workspace-write")
    parser.add_argument("--blender", type=Path)
    parser.add_argument("--stl_dir", type=Path)
    parser.add_argument("--panel_file", type=Path, default=Path(__file__).with_name("review_models.json"))
    parser.add_argument("--asset_cache", type=Path)
    parser.add_argument("--asset_catalog", type=Path)
    parser.add_argument("--device", choices=("OPTIX", "CUDA", "CPU"), default="OPTIX")
    parser.add_argument("--gpu_index", type=int, default=0)
    parser.add_argument("--max_repairs", type=int, default=2, help="外围返修上限，默认2；0表示只检查首版")
    parser.add_argument("--dry_run", action="store_true", help="只保存输入，不调用Codex或Blender")
    args = vars(parser.parse_args())
    prompt_file = args.pop("user_prompt_file")
    if prompt_file:
        args["user_prompt"] = prompt_file.read_text(encoding="utf-8")
    max_repairs, dry_run = args.pop("max_repairs"), args.pop("dry_run")
    request = prepare_request(developer_instructions=DEVELOPER_INSTRUCTIONS, **args)
    result = run_workflow(request, max_repairs=max_repairs, dry_run=dry_run)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] == "incomplete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
