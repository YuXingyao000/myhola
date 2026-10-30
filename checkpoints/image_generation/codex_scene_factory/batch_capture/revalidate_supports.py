"""固定原五个位置，不调用 Agent：旧场景负例 → 修复副本 → 重新测量与试拍。"""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "packages/cad_scene_factory/step2"))
from workflow import read_json, write_json, run_tool, prepare_supports, export_placements, scene_hash

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--ready_file", type=Path, required=True)
parser.add_argument("--output_dir", type=Path, required=True)
parser.add_argument("--blender", required=True)
args = parser.parse_args()
old = read_json(args.ready_file)
output = args.output_dir.resolve()
output.mkdir(parents=True, exist_ok=False)
request = {"base_scene": old["base_scene"], "config": old["config"], "output_dir": str(output),
           "blender": args.blender, "source_ready": str(args.ready_file),
           "mode": "fixed_placement_revalidation", "agent_calls": 0}
state = {"status": "incomplete", "stage": "negative_control", "base_scene": old["base_scene"],
         "scene_hash": old["scene_hash"], "config": old["config"], "human_review": "pending",
         "mode": "fixed_placement_revalidation", "agent_calls": 0, "candidates": []}

first = old["placements"][0]
negative = run_tool(request, "preview", output / "old_scene_negative_control",
                    region=first["region"], proposal=first["setup"])
assert not negative["numeric_pass"] and "rendered_support_height" in negative["reasons"]
print("旧场景位移负例已被新检查拒绝。", flush=True)

regions = [placement["region"] for placement in old["placements"]]
prepare_supports(request, state, regions)
assert scene_hash(old["base_scene"]) == old["scene_hash"]
context = read_json(args.ready_file.parent / "scene_context.json")
measurements = run_tool(request, "measure", output / "measurement", regions=regions, views=context["views"])
for index, (placement, measurement) in enumerate(zip(old["placements"], measurements["regions"])):
    assert measurement["status"] == "measured"
    report = run_tool(request, "preview", output / f"candidate_{index:02d}",
                      region=placement["region"], proposal=placement["setup"])
    candidate = {"region": placement["region"], "measurement": measurement,
                 "status": "proxy_ready" if report["numeric_pass"] else "rejected",
                 "attempts": [{"proposal": placement["setup"], "numeric_report": report,
                               "visual_review": {"decision": "pending", "reviewer": "human",
                                                 "note": "数值复验；未调用 Agent 对新图片评议"}}]}
    state["candidates"].append(candidate)
    write_json(output / "placements.json", state)
    check = report["checks"]["rendered_support"]
    print(placement["setup"]["region_id"], "pass=", report["numeric_pass"],
          "max_error_mm=", check.get("max_height_error_m", float("nan")) * 1000, flush=True)
state.update(status="completed", stage="export")
export_placements(request, state)
write_json(output / "validation_summary.json", {
    "negative_control_rejected": True, "source_scene_unchanged": scene_hash(old["base_scene"]) == old["scene_hash"],
    "planned": len(regions), "passed": sum(c["status"] == "proxy_ready" for c in state["candidates"]),
    "human_review": "pending", "agent_calls": 0,
})
assert all(c["status"] == "proxy_ready" for c in state["candidates"])
