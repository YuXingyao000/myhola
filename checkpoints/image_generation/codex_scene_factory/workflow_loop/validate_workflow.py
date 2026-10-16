"""验证 workflow 的控制边界；不调用真实 Codex 或 Blender。

每个 case 使用独立目录，生成真实的小型 PNG、源文件和预览证据，再由实际
snapshot_inputs/file_sha256 检查。模型回答和外围预览使用替身；因此这些结果
只证明流程控制与证据检查，不证明模型能力、渲染质量或数据集适用性。

运行：python validate_workflow.py [--output_dir /path/to/new/check]
"""

import argparse
from contextlib import ExitStack
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch

from PIL import Image
from rich.console import Console


PACKAGE_DIR = Path(__file__).resolve().parents[4] / "packages" / "cad_scene_factory"
sys.path.insert(0, str(PACKAGE_DIR))
import workflow


SESSION_ID = "0199a213-81c0-7800-8aa1-bbab2a035a53"
CASES = (
    "first_ready",
    "technical_failure_then_ready",
    "host_launch_failure",
    "fixed_stl_mismatch",
    "visual_revise_then_ready",
    "repair_limit",
    "blocked",
    "malformed_review_json",
    "malformed_review_fields",
    "missing_image_list",
    "ready_with_issues",
    "create_missing_plan",
    "readonly_script_change",
    "readonly_asset_change",
    "readonly_image_change",
    "readonly_blend_change",
    "readonly_report_change",
    "readonly_stl_change",
    "create_panel_change",
    "repair_panel_change",
    "create_host_change",
    "repair_host_change",
    "without_blender",
    "dry_run",
)


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_case(name: str, case_dir: Path) -> dict:
    case_dir.mkdir()
    template_dir = case_dir / "template"
    panel_file = case_dir / "panel.json"
    model_file = case_dir / "fixture_part.stl"
    model_file.write_text("solid fixture\nendsolid fixture\n", encoding="utf-8")
    write_json(panel_file, {"version": "workflow-control-fixture", "models": ["fixture_part"]})
    request = {
        "output_dir": str(template_dir), "prompt": "Fixture task; do not call a model.",
        "reference_image": None, "model": "unused-fixture-model",
        "runtime": {
            "panel_file": str(panel_file),
            "blender": None if name == "without_blender" else "/unused/blender",
            "stl_dir": str(case_dir), "device": "CPU", "gpu_index": 0,
        },
    }
    events = []
    preview_reports = []
    host_changed = False
    original_snapshot = workflow.snapshot_inputs

    def fixture_snapshot(directory, panel):
        snapshot = original_snapshot(directory, panel)
        # 不触碰项目代码。只有这两个场景模拟受保护外围文件的摘要发生变化。
        if host_changed:
            snapshot["host_files"]["render.py"] = "fixture-changed-host-digest"
        return snapshot

    def write_candidate(include_plan=True):
        template_dir.mkdir(parents=True, exist_ok=True)
        (template_dir / "build_scene.py").write_text("# fixture scene\n", encoding="utf-8")
        (template_dir / "build_material.py").write_text("# fixture material\n", encoding="utf-8")
        if include_plan:
            (template_dir / "scene_plan.md").write_text("Fixture: a small part on a used bench.\n", encoding="utf-8")
        assets_dir = template_dir / "assets"
        assets_dir.mkdir(exist_ok=True)
        asset = assets_dir / "surface.bin"
        asset.write_bytes(b"fixture surface v1")
        write_json(template_dir / "assets.json", [
            {"path": "assets/surface.bin", "sha256": workflow.file_sha256(asset)},
        ])

    def review_reply(images):
        decision = "ready"
        if name in {"repair_limit", "repair_panel_change", "repair_host_change"}:
            decision = "revise"
        elif name == "visual_revise_then_ready" and len(preview_reports) == 1:
            decision = "revise"
        elif name == "blocked":
            decision = "blocked"
        issues = [] if decision == "ready" else [{
            "image": images[0].name, "region": "worktop", "observation": "fixture visible issue",
            "suggested_change": "fixture targeted change",
        }]
        review = {
            "decision": decision, "summary": "Fixture visual review, not an image-quality result.",
            "inspected_images": [path.name for path in images], "issues": issues,
            "remaining_issues": [] if decision == "ready" else ["fixture visible issue"],
        }
        if name == "malformed_review_json":
            return "not JSON"
        if name == "malformed_review_fields":
            review.pop("summary")
        if name == "missing_image_list":
            review["inspected_images"] = review["inspected_images"][:-1]
        if name == "ready_with_issues":
            review["issues"] = [{
                "image": images[0].name, "region": "worktop", "observation": "unresolved fixture issue",
                "suggested_change": "fix fixture issue",
            }]
        return json.dumps(review)

    def fake_turn(task, prompt, turn_dir, *, session_id=None, images=None, output_schema=None):
        nonlocal host_changed
        action = turn_dir.name
        events.append(action)
        assert action in {"create", "inspect", "repair"}, action
        assert session_id == (None if action == "create" else SESSION_ID)
        turn_dir.mkdir(parents=True)
        (turn_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
        if action == "create":
            write_candidate(include_plan=name != "create_missing_plan")
            if name == "create_panel_change":
                panel_file.write_text('{"version":"changed"}\n', encoding="utf-8")
            if name == "create_host_change":
                host_changed = True
            reply = "Fixture candidate created."
        elif action == "repair":
            write_candidate()
            if name == "repair_panel_change":
                panel_file.write_text('{"version":"changed"}\n', encoding="utf-8")
            if name == "repair_host_change":
                host_changed = True
            reply = "Fixture candidate repaired."
        else:
            assert len(images) == 2 and all(path.is_file() for path in images)
            assert output_schema.is_file()
            preview_dir = Path(preview_reports[-1]["output_dir"])
            mutations = {
                "readonly_script_change": template_dir / "build_scene.py",
                "readonly_asset_change": template_dir / "assets" / "surface.bin",
                "readonly_image_change": images[0],
                "readonly_blend_change": preview_dir / "scene.blend",
                "readonly_report_change": preview_dir / "preview_report.json",
                "readonly_stl_change": model_file,
            }
            if name in mutations:
                with mutations[name].open("ab") as stream:
                    stream.write(b"\nfixture unauthorized mutation")
            reply = review_reply(images)
        result = {"session_id": SESSION_ID, "reply": reply, "usage": {"input_tokens": 0, "output_tokens": 0},
                  "turn_dir": str(turn_dir), "status": "completed", "returncode": 0}
        (turn_dir / "reply.txt").write_text(reply, encoding="utf-8")
        write_json(turn_dir / "result.json", result)
        return result

    def fake_preview(task, output_dir):
        events.append("preview")
        output_dir.mkdir(parents=True)
        renders = output_dir / "renders"
        renders.mkdir()
        failed = (name == "technical_failure_then_ready" and not preview_reports
                  or name in {"host_launch_failure", "fixed_stl_mismatch"})
        rows = []
        for index, kind in enumerate(("model", "empty")):
            image_file = "000000_fixture_part.png" if kind == "model" else "empty_000.png"
            image_path = renders / image_file
            Image.new("RGB", (512, 512), color=(30 + index * 40, 70, 90)).save(image_path)
            row = {
                "kind": kind, "image_file": image_file, "image_sha256": workflow.file_sha256(image_path),
                "status": "failed" if failed and index == 0 else "success",
            }
            if kind == "model":
                row.update(model_id="fixture_part", model_file=str(model_file),
                           final_stl_sha256=workflow.file_sha256(model_file))
            if row["status"] == "failed":
                row["error"] = "fixture geometry-check failure"
            rows.append(row)
        (renders / "renders.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
        (output_dir / "scene.blend").write_bytes(b"fixture blend evidence; not a Blender file")
        for filename in ("build_report.json", "request.json", "panel.json"):
            write_json(output_dir / filename, {"fixture": True, "filename": filename})
        Image.new("RGB", (512, 256), color=(70, 70, 70)).save(output_dir / "contact_sheet.png")
        report = {
            "output_dir": str(output_dir), "technical": "fail" if failed else "pass",
            "counts": {"planned": 2, "success": 1 if failed else 2, "failed": int(failed), "missing": 0},
            "results": rows, "snapshot_after": fixture_snapshot(template_dir, panel_file),
        }
        if name == "host_launch_failure":
            report["build"] = {"returncode": None, "error": "fixture missing Blender executable"}
        if name == "fixed_stl_mismatch":
            report["model_inputs_match_panel"] = False
        write_json(output_dir / "preview_report.json", report)
        preview_reports.append(report)
        return report

    console_capture = io.StringIO()
    with ExitStack() as stack:
        stack.enter_context(patch.object(workflow, "run_codex_turn", fake_turn))
        stack.enter_context(patch.object(workflow, "_preview", fake_preview))
        stack.enter_context(patch.object(workflow, "snapshot_inputs", fixture_snapshot))
        stack.enter_context(patch.object(workflow, "CONSOLE", Console(file=console_capture)))
        # 除了上面的替身，任何遗漏的真实外部进程调用都立即失败。
        stack.enter_context(patch.object(subprocess, "run", side_effect=AssertionError("External process is forbidden in validation")))
        stack.enter_context(patch.object(subprocess, "Popen", side_effect=AssertionError("External process is forbidden in validation")))
        result = workflow.run_workflow(request, max_repairs=2, dry_run=name == "dry_run")
    (case_dir / "console.log").write_text(console_capture.getvalue(), encoding="utf-8")
    write_json(case_dir / "observed_events.json", events)

    success_cases = {"first_ready", "technical_failure_then_ready", "visual_revise_then_ready", "create_missing_plan"}
    expected = "ready_for_review" if name in success_cases else "incomplete"
    if name == "without_blender":
        expected = "generated_unverified"
    if name == "dry_run":
        expected = "dry_run"
    assert result["status"] == expected, (name, result)

    sequences = {
        "first_ready": ["create", "preview", "inspect"],
        "technical_failure_then_ready": ["create", "preview", "repair", "preview", "inspect"],
        "host_launch_failure": ["create", "preview"],
        "fixed_stl_mismatch": ["create", "preview"],
        "visual_revise_then_ready": ["create", "preview", "inspect", "repair", "preview", "inspect"],
        "create_missing_plan": ["create", "preview", "repair", "preview", "inspect"],
        "repair_limit": ["create", "preview", "inspect", "repair", "preview", "inspect", "repair", "preview", "inspect"],
        "create_panel_change": ["create"], "create_host_change": ["create"],
        "repair_panel_change": ["create", "preview", "inspect", "repair"],
        "repair_host_change": ["create", "preview", "inspect", "repair"],
        "without_blender": ["create"], "dry_run": [],
    }
    assert events == sequences.get(name, ["create", "preview", "inspect"]), (name, events)
    if name != "dry_run":
        assert result["human_review"] == "pending"
        assert result["session_id"] == SESSION_ID
        assert json.loads((template_dir / "workflow_result.json").read_text()) == result
        assert len(result["rounds"]) == len(preview_reports)
        assert all(round_["counts"]["planned"] == 2 for round_ in result["rounds"])
    if expected == "incomplete":
        assert result["remaining_issues"], (name, "missing failure explanation")
    if name == "repair_limit":
        assert result["repairs_used"] == 2 and len(result["rounds"]) == 3
    if name in {"blocked", "host_launch_failure", "fixed_stl_mismatch"}:
        assert result["repairs_used"] == 0 and result["rounds"][0]["decision"] == "blocked"
    if name == "technical_failure_then_ready":
        assert result["rounds"][0]["counts"] == {"planned": 2, "success": 1, "failed": 1, "missing": 0}
    if name == "dry_run":
        assert not (template_dir / "workflow").exists()
        assert (template_dir / "request.json").is_file()
    if name.startswith("readonly_"):
        assert "改变" in result["error"], result
    if name.endswith(("panel_change", "host_change")):
        assert "不采用新基线" in result["error"], result
    return {"case": name, "pass": True, "status": result["status"], "events": events,
            "repairs_used": result.get("repairs_used", 0), "error": result.get("error")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, help="保存独立case与results.json；默认新建临时目录")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve() if args.output_dir else Path(tempfile.mkdtemp(prefix="cad-scene-workflow-validation-"))
    if output_dir.exists() and any(output_dir.iterdir()):
        parser.error("output_dir非空，请使用新目录以保留此前验证记录")
    output_dir.mkdir(parents=True, exist_ok=True)
    outcomes = []
    for name in CASES:
        try:
            outcomes.append(run_case(name, output_dir / name))
        except Exception as error:
            outcomes.append({"case": name, "pass": False, "error": f"{type(error).__name__}: {error}"})
    result = {
        "scope": "Workflow control and evidence checks using fixtures; no Codex or Blender calls.",
        "tests": len(outcomes), "passed": sum(item["pass"] for item in outcomes),
        "failed": sum(not item["pass"] for item in outcomes), "pass": all(item["pass"] for item in outcomes),
        "output_dir": str(output_dir), "cases": outcomes,
        "source_sha256": {name: workflow.file_sha256(PACKAGE_DIR / name)
                          for name in ("workflow.py", "preview.py", "call_llm.py")},
        "prompt_sha256": {str(path.relative_to(PACKAGE_DIR)): workflow.file_sha256(path)
                          for path in sorted((PACKAGE_DIR / "prompts").rglob("*")) if path.is_file()},
    }
    write_json(output_dir / "results.json", result)
    print(json.dumps({key: value for key, value in result.items() if key != "cases"}, ensure_ascii=False, indent=2))
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
