"""真实Blender + 替身Codex的接线检查，不判断视觉质量，不消耗模型额度。"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "packages/cad_scene_factory"))
import workflow
from call_llm import DEVELOPER_INSTRUCTIONS, prepare_request


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_template", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--blender", type=Path, required=True)
    parser.add_argument("--stl_dir", type=Path, required=True)
    parser.add_argument("--gpu_index", type=int, default=0)
    args = parser.parse_args()
    source_hashes = {name: digest(args.source_template / name)
                     for name in ("build_scene.py", "build_material.py", "assets.json")}
    calls = []

    def fake_codex(request, prompt, turn_dir, *, session_id=None, images=None, output_schema=None):
        turn_dir.mkdir(parents=True)
        target = Path(request["output_dir"])
        if session_id is None:
            for name in source_hashes:
                shutil.copy2(args.source_template / name, target / name)
            shutil.copytree(args.source_template / "assets", target / "assets")
            (target / "scene_plan.md").write_text(
                "Integration fixture: copy the existing workshop without scene changes.\n"
                "Its original unit interpretation remains 0.25 m/unit.\n"
                "This validates workflow wiring, not the new 17.5 cm design or realism.\n"
            )
            reply = "Fixture copied; no language model was called."
        else:
            assert output_schema is not None, "The unchanged fixture should not need repair"
            assert len(images) == 15 and all(path.is_file() for path in images)
            reply = json.dumps({
                "decision": "ready", "summary": "Test stub only; no visual quality judgment.",
                "inspected_images": [path.name for path in images], "issues": [], "remaining_issues": [],
            })
        turn = {"session_id": "blender-integration-stub", "reply": reply, "usage": None,
                "status": "completed", "turn_dir": str(turn_dir), "returncode": 0}
        (turn_dir / "prompt.txt").write_text(prompt)
        (turn_dir / "reply.txt").write_text(reply)
        (turn_dir / "result.json").write_text(json.dumps(turn, indent=2))
        calls.append({"session_id": session_id, "images": len(images or []), "output_schema": str(output_schema)})
        return turn

    request = prepare_request(
        DEVELOPER_INSTRUCTIONS, "Integration fixture: reuse the supplied existing template unchanged.",
        args.output_dir, blender=args.blender, stl_dir=args.stl_dir, gpu_index=args.gpu_index,
        asset_cache=args.output_dir / ".asset_cache",
    )
    with patch.object(workflow, "run_codex_turn", fake_codex):
        result = workflow.run_workflow(request, max_repairs=0)
    after = {name: digest(args.source_template / name) for name in source_hashes}
    validation = {"agent_mode": "stub_no_visual_judgment", "real_blender": True,
                  "source_unchanged": source_hashes == after, "source_hashes": source_hashes,
                  "calls": calls, "workflow_status": result["status"], "rounds": result["rounds"]}
    (args.output_dir / "validation.json").write_text(json.dumps(validation, indent=2))
    assert source_hashes == after
    assert result["status"] == "ready_for_review", result
    assert result["rounds"][0]["counts"] == {"planned": 15, "success": 15, "failed": 0, "missing": 0}
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
