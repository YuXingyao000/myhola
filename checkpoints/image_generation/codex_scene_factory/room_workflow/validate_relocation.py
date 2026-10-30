"""复制完整模板，在原模板路径不可用时重建并渲染六图；不调用模型。

只用于已停止运行的任务。原模板仅在检查期间改名，finally 中恢复；源 Shed 不移动。
本脚本保留全部二进制产物和报告于显式 output_dir，不写入版本管理目录。
"""

import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "packages" / "cad_scene_factory"))
import room_workflow as workflow


def validate_relocation(template_dir: Path, output_dir: Path) -> dict:
    template_dir = template_dir.resolve()
    output_dir = output_dir.resolve()
    state = workflow._read_json(template_dir / "workflow_result.json")
    if state["status"] != "ready_for_review":
        raise ValueError("只验收已完成外围检查的模板")
    request = workflow._read_json(template_dir / "request.json")
    if output_dir.exists():
        raise FileExistsError(output_dir)
    held = template_dir.with_name(template_dir.name + "__relocation_hold")
    if held.exists():
        raise FileExistsError(held)

    with workflow._task_lock(template_dir):
        output_dir.mkdir(parents=True)
        for name in ("build_scene.py", "views.json", "scene_plan.md", "assets.json"):
            shutil.copy2(template_dir / name, output_dir / name)
        shutil.copytree(template_dir / "assets", output_dir / "assets")
        copied_hashes = {
            str(path.relative_to(output_dir)): workflow._sha256(path)
            for path in output_dir.rglob("*") if path.is_file()
        }
        for relative, digest in copied_hashes.items():
            if workflow._sha256(template_dir / relative) != digest:
                raise RuntimeError(f"复制摘要不一致：{relative}")

        relocated = copy.deepcopy(request)
        relocated["output_dir"] = str(output_dir)
        relocated["runtime"]["template_dir"] = str(output_dir)
        workflow._write_json(output_dir / "relocation_request.json", relocated)
        template_dir.rename(held)
        try:
            unavailable = not template_dir.exists()
            report = workflow.preview_room(relocated, output_dir / "verification")
        finally:
            held.rename(template_dir)

    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": report["status"],
        "source_template": str(template_dir),
        "relocated_template": str(output_dir),
        "source_template_unavailable_during_check": unavailable,
        "source_template_restored": template_dir.is_dir(),
        "copied_files": copied_hashes,
        "preview_report": str(output_dir / "verification" / "preview_report.json"),
        "planned": report["planned"], "success": report["success"], "failed": report["failed"],
        "errors": report["errors"],
    }
    workflow._write_json(output_dir / "relocation_result.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    result = validate_relocation(args.template_dir, args.output_dir)
    print(json.dumps({key: value for key, value in result.items() if key != "copied_files"},
                     ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["status"] == "pass" else 1)
