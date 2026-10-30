"""不调用模型/Blender，检查真实主循环的准备时序、场景切换和验收导出。"""

import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "packages/cad_scene_factory/step2"))
import workflow

output = Path(sys.argv[1]).resolve()
output.mkdir(parents=True, exist_ok=False)
source = output / "source.blend"
source.write_bytes(b"original")
request = {"base_scene": str(source), "output_dir": str(output), "config": {"max_regions": 1, "max_revisions": 0},
           "user_request": ""}
state = {"scene_hash": workflow.scene_hash(source), "candidates": []}
region = {"region_id": "table", "support_object_id": "Table", "movable_groups": []}
events = []


def agent(request, state, stage, inputs, turn_dir, images):
    events.append(stage)
    if stage == "select_regions":
        return {"regions": [region]}
    if stage == "propose_setup":
        assert inputs["base_scene"] == request["base_scene"] != str(source)
        return {"setups": [{"region_id": "table"}]}
    return {"reviews": [{"region_id": "table", "decision": "keep", "reason": "test only",
                          "inspected_images": ["before.png", "after.png"]}]}


def tool(request, action, output_dir, **inputs):
    events.append(action)
    if action == "prepare_supports":
        assert request["base_scene"] == str(source)
        output_dir.mkdir()
        prepared = output_dir / "prepared_scene.blend"
        prepared.write_bytes(b"prepared")
        return {"base_scene": str(prepared), "material_changes": []}
    assert Path(request["base_scene"]).read_bytes() == b"prepared"
    assert workflow.scene_hash(request["base_scene"]) == state["scene_hash"]
    if action == "measure":
        return {"regions": [{"region_id": "table", "status": "measured", "sites": []}]}
    return {"numeric_pass": True, "transforms": {}, "camera": {}, "reasons": [],
            "images": {"before": "before.png", "after": "after.png"},
            "checks": {"rendered_support": {"pass": True, "expected_height_m": .75,
                                             "rendered_height_m": {"median": .75}, "tolerance_m": .002}}}


with patch.object(workflow, "ask_agent", agent), patch.object(workflow, "run_tool", tool):
    workflow.run_workflow(request, state, {"views": []}, [])
assert events == ["select_regions", "prepare_supports", "measure", "propose_setup", "preview", "inspect_setup"]
ready = workflow.read_json(output / "ready.json")
assert ready["base_scene"] == request["base_scene"] == state["base_scene"]
assert ready["scene_hash"] == workflow.scene_hash(ready["base_scene"])
assert ready["source_scene"] == str(source) and source.read_bytes() == b"original"
assert ready["placements"][0]["support_validation"]["pass"]
workflow.write_json(output / "checks.json", {"order": events, "prepared_scene_used_everywhere": True,
                                              "source_unchanged": True, "validation_exported": True})
print("SUPPORT_LOOP_CHECK_PASSED")
