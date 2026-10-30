"""替换 Agent/Blender 边界，验证真实外围循环；不声称检验了 LLM 的选位质量。"""

import copy
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / "packages/cad_scene_factory/step2"))
import workflow

OUTPUT = Path(sys.argv[1]).resolve()
OUTPUT.mkdir(parents=True, exist_ok=False)
CONFIG = workflow.read_json(ROOT / "packages/cad_scene_factory/step2/config.json")
base = OUTPUT / "base.blend"
base.write_bytes(b"loop test fixture; never opened by Blender")
request = {"output_dir": str(OUTPUT), "base_scene": str(base), "config": CONFIG, "user_request": ""}
state = {"status":"incomplete", "stage":"select", "scene_hash":workflow.scene_hash(base),
         "session_id":None, "codex_turns":[], "candidates":[], "human_review":"pending"}
regions = [{"region_id": f"r{i}", "support_object_id":"Table", "movable_groups":[]} for i in (1,2,3)]
calls, previews = [], []


def agent(request, prompt, turn_dir, *, session_id=None, images=None, output_schema=None):
    stage = turn_dir.name
    assert session_id == ("scripted-placement-session" if calls else None)
    calls.append(stage)
    if stage == "select_regions":
        reply = {"regions": regions}
    elif stage == "propose_setup":
        active = [item["region"]["region_id"] for item in state["candidates"] if item["status"] == "incomplete"]
        reply = {"setups": [{"region_id": region_id} for region_id in active]}
    else:
        reviews = []
        for item in state["candidates"]:
            if item["status"] != "incomplete":
                continue
            attempt = item["attempts"][-1]
            region_id = item["region"]["region_id"]
            # r3 总是数值失败但 Agent 错误 keep；外围必须否决。
            decision = "revise" if region_id == "r2" and attempt["revision"] == 0 else "keep"
            reviews.append({"region_id":region_id, "decision":decision, "reason":"scripted test",
                            "suggested_change":"", "inspected_images":list(attempt["numeric_report"]["images"].values())})
        reply = {"reviews": reviews}
    return {"session_id":"scripted-placement-session", "turn_dir":str(turn_dir), "status":"completed",
            "usage":{"test_stub":True}, "reply":json.dumps(reply)}


def tool(request, action, output_dir, **inputs):
    if action == "measure":
        return {"regions":[{"region_id":item["region_id"],"status":"measured","sites":[]} for item in regions]}
    region_id = inputs["region"]["region_id"]
    previews.append(region_id)
    return {"region_id":region_id,"numeric_pass":region_id != "r3", "reasons":[] if region_id != "r3" else ["collision"],
            "images":{"before":str(output_dir/"before.png"),"after":str(output_dir/"after.png")},
            "transforms":{},"camera":{}}


with patch.object(workflow, "prepare_scene_evidence", return_value=({"views":[]}, [])), \
        patch.object(workflow, "run_codex_turn", agent), patch.object(workflow, "run_tool", tool):
    workflow.run_workflow(request, state)
assert calls == ["select_regions"] + ["propose_setup", "inspect_setup"] * 3
assert previews == ["r1", "r2", "r3", "r2", "r3", "r3"]
assert [item["status"] for item in state["candidates"]] == ["proxy_ready", "proxy_ready", "rejected"]
assert [len(item["attempts"]) for item in state["candidates"]] == [1, 2, 3]
assert len(workflow.read_json(OUTPUT/"ready.json")["placements"]) == 2

# 工具故障向外抛出；已有状态保留 incomplete，而非把基础设施故障记成几何失败。
failed_output = OUTPUT / "tool_failure"
failed_output.mkdir()
failed_request = {**request, "output_dir":str(failed_output)}
failed_state = {**copy.deepcopy(state), "status":"incomplete", "session_id":None,"codex_turns":[],"candidates":[]}
calls.clear()
with patch.object(workflow, "prepare_scene_evidence", return_value=({"views":[]}, [])), \
        patch.object(workflow, "run_codex_turn", agent), patch.object(workflow, "run_tool", side_effect=RuntimeError("tool failure")):
    try:
        workflow.run_workflow(failed_request, failed_state)
    except RuntimeError as error:
        assert str(error) == "tool failure"
    else:
        raise AssertionError("tool failure should surface")
assert workflow.read_json(failed_output/"placements.json")["status"] == "incomplete"
workflow.write_json(OUTPUT/"checks.json", {"finite_rounds":True,"passed_candidates_not_retried":True,
    "numeric_veto":True,"one_explicit_session":True,"failure_keeps_incomplete":True,"real_llm_calls":0})
print("STEP2_LOOP_CHECKS_PASSED")
