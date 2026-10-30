"""使用本地小型文件和模拟 Agent/Blender 验证工作流状态，不调用真实服务。"""

import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
from rich.console import Console


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "packages" / "cad_scene_factory"))
import room_workflow as workflow


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


class RoomWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="room-workflow-check-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.template = self.root / "template"
        self.source = self.root / "source"
        self.source.mkdir()
        (self.source / "base.blend").write_bytes(b"fixture-blend")
        (self.source / "wood.png").write_bytes(b"fixture-texture")
        self.asset_manifest = self.root / "source_assets.json"
        write_json(self.asset_manifest, {"files": [
            {"path": path.name, "bytes": path.stat().st_size, "sha256": workflow._sha256(path)}
            for path in sorted(self.source.iterdir())
        ]})
        self.reference_image = self.root / "reference_workbench.png"
        Image.new("RGB", (32, 24), "white").save(self.reference_image)
        self.reference = self.root / "reference.json"
        write_json(self.reference, {
            "id": "test_reference", "asset_directory": str(self.source),
            "asset_subdir": "the_shed", "source_blend": str(self.source / "base.blend"),
            "source": {"sha256": workflow._sha256(self.source / "base.blend"), "license": "CC0"},
            "images": [{"path": str(self.reference_image)}],
            "asset_manifest": {"path": str(self.asset_manifest), "sha256": workflow._sha256(self.asset_manifest)},
        })
        self.host_file = self.root / "host_fixture.py"
        self.host_file.write_text("# frozen fixture\n")
        self.calls = []
        self.previews = []
        self.reviews = ["ready"]
        self.preview_statuses = []
        self.failure_stage = None
        self.inspect_mutation = None
        self.session = "fixture-exact-session"
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(workflow, "run_codex_turn", self.codex))
        self.stack.enter_context(patch.object(workflow, "preview_room", self.preview))
        self.stack.enter_context(patch.object(workflow, "_host_snapshot", lambda: {
            str(self.host_file): workflow._sha256(self.host_file),
        }))
        self.stack.enter_context(patch.object(workflow, "CONSOLE", Console(file=io.StringIO())))

    def generate(self, **kwargs):
        return workflow.generate_room(
            "Adapt the supplied workshop.", self.reference, self.template,
            Path("/fixture/blender"), **kwargs,
        )

    def write_candidate(self):
        (self.template / "build_scene.py").write_text("def build_scene(template_dir):\n    pass\n")
        (self.template / "scene_plan.md").write_text("Fixture adapted workbench.\n")
        write_json(self.template / "views.json", {"views": [
            {"id": name, "position": [1, 1, 1], "target": [0, 0, 0],
             "vertical_fov_degrees": 50, "resolution": [32, 24]}
            for name in sorted(workflow.VIEW_IDS)
        ]})

    def review(self, decision):
        result = {
            "decision": decision, "summary": "Fixture visual judgment; human review pending.",
            "inspected_images": [name + ".png" for name in sorted(workflow.VIEW_IDS)],
            "reference_images": [self.reference_image.name],
            "reference_comparison": {name: "Fixture observation" for name in
                                     ("layout", "surfaces", "lighting", "adaptation")},
            "issues": [], "remaining_issues": [],
        }
        if decision == "revise":
            result["issues"] = [{"image": "workbench_context.png", "region": "tool shelf",
                                 "observation": "Fixture floating tool", "suggested_change": "Move onto shelf"}]
            result["remaining_issues"] = ["Fixture floating tool"]
        if decision == "blocked":
            result["remaining_issues"] = ["Fixture required resource absent"]
        return result

    def codex(self, request, prompt, turn_dir, *, session_id=None, images=None, output_schema=None):
        stage = "inspect" if output_schema else ("create" if session_id is None else "repair")
        if turn_dir.name.startswith("create_resume_"):
            stage = "create_resume"
        self.calls.append({"stage": stage, "session_id": session_id, "turn_dir": str(turn_dir)})
        turn_dir.mkdir(parents=True, exist_ok=False)
        (turn_dir / "prompt.txt").write_text(prompt)
        result = {"session_id": self.session, "turn_dir": str(turn_dir),
                  "status": "completed", "returncode": 0,
                  "usage": {"input_tokens": 10, "output_tokens": 10}, "reply": "Fixture delivered"}
        if stage == self.failure_stage:
            result.update(status="failed", returncode=1, error="fixture transport failure")
            write_json(turn_dir / "result.json", result)
            raise RuntimeError("fixture transport failure")
        if stage in {"create", "create_resume"}:
            self.write_candidate()
        if stage == "inspect":
            decision = self.reviews.pop(0) if self.reviews else "ready"
            review = self.review(decision) if isinstance(decision, str) else copy.deepcopy(decision)
            result["reply"] = json.dumps(review)
            if self.inspect_mutation:
                self.inspect_mutation()
        write_json(turn_dir / "result.json", result)
        (turn_dir / "reply.txt").write_text(result["reply"])
        return result

    def preview(self, request, output_dir):
        self.previews.append(str(output_dir))
        output_dir.mkdir(parents=True, exist_ok=False)
        status = self.preview_statuses.pop(0) if self.preview_statuses else "pass"
        report = {"status": status, "images": [], "errors": [], "planned": 6,
                  "success": 6 if status == "pass" else 0,
                  "failed": 0 if status == "pass" else 6}
        if status == "pass":
            blend = output_dir / "room.blend"
            blend.write_bytes(b"fixture-rendered-blend")
            write_json(output_dir / "build_report.json", {"status": "pass"})
            write_json(output_dir / "scene_inventory.json", {"objects": []})
            (output_dir / "views.json").write_bytes((self.template / "views.json").read_bytes())
            (output_dir / "renders").mkdir()
            write_json(output_dir / "renders" / "render_report.json", {"status": "pass", "success": 6, "failed": 0})
            images = []
            for index, name in enumerate(sorted(workflow.VIEW_IDS)):
                path = output_dir / (name + ".png")
                Image.new("RGB", (32, 24), (index * 20, 60, 90)).save(path)
                images.append(str(path))
            report.update(images=images, blend_file=str(blend))
        else:
            report["errors"] = ["Fixture Blender build failed"]
        write_json(output_dir / "preview_report.json", report)
        return report

    def test_create_ready(self):
        result = self.generate()
        self.assertEqual(result["status"], "ready_for_review")
        self.assertEqual(result["human_review"], "pending")
        self.assertEqual([call["stage"] for call in self.calls], ["create", "inspect"])
        self.assertIsNone(self.calls[0]["session_id"])
        self.assertEqual(self.calls[1]["session_id"], self.session)
        self.assertTrue(Path(result["room_blend"]).is_file())

    def test_revise_repair_ready_same_session(self):
        self.reviews = ["revise", "ready"]
        result = self.generate(max_repairs=1)
        self.assertEqual(result["status"], "ready_for_review")
        self.assertEqual(result["repairs_used"], 1)
        self.assertEqual([call["stage"] for call in self.calls], ["create", "inspect", "repair", "inspect"])
        self.assertTrue(all(call["session_id"] == self.session for call in self.calls[1:]))
        self.assertEqual([record["decision"] for record in result["rounds"]], ["revise", "ready"])

    def test_technical_failure_skips_inspect_then_repairs(self):
        self.preview_statuses = ["failed", "pass"]
        result = self.generate(max_repairs=1)
        self.assertEqual(result["status"], "ready_for_review")
        self.assertEqual([call["stage"] for call in self.calls], ["create", "repair", "inspect"])
        old_report = Path(result["rounds"][0]["report"])
        self.assertEqual(json.loads(old_report.read_text())["status"], "failed")

    def test_exhausted_repair_budget_preserves_failure(self):
        self.preview_statuses = ["failed"]
        result = self.generate(max_repairs=0)
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["repairs_used"], 0)
        self.assertEqual(result["remaining_issues"], ["Fixture Blender build failed"])
        self.assertTrue(Path(result["rounds"][0]["review"]).is_file())

    def test_create_transport_failure_retains_records(self):
        self.failure_stage = "create"
        result = self.generate()
        self.assertEqual(result["status"], "incomplete")
        path = self.template / "workflow" / "create" / "result.json"
        self.assertEqual(json.loads(path.read_text())["status"], "failed")
        self.assertIn("transport failure", result["remaining_issues"][0])
        self.assertEqual(self.previews, [])

    def test_incomplete_candidate_image_coverage_rejected(self):
        review = self.review("ready")
        review["inspected_images"].pop()
        self.reviews = [review]
        result = self.generate()
        self.assertEqual(result["status"], "incomplete")
        self.assertIn("inspected_images", result["remaining_issues"][0])

    def test_duplicate_candidate_image_coverage_rejected(self):
        review = self.review("ready")
        review["inspected_images"][-1] = review["inspected_images"][0]
        self.reviews = [review]
        result = self.generate()
        self.assertEqual(result["status"], "incomplete")

    def test_missing_reference_image_coverage_rejected(self):
        review = self.review("ready")
        review["reference_images"] = []
        self.reviews = [review]
        result = self.generate()
        self.assertEqual(result["status"], "incomplete")
        self.assertIn("reference_images", result["remaining_issues"][0])

    def test_ready_with_issue_rejected(self):
        review = self.review("revise")
        review["decision"] = "ready"
        self.reviews = [review]
        result = self.generate()
        self.assertEqual(result["status"], "incomplete")
        self.assertIn("ready", result["remaining_issues"][0])

    def test_blocked_remains_incomplete(self):
        self.reviews = ["blocked"]
        result = self.generate()
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["repairs_used"], 0)
        self.assertEqual(result["remaining_issues"], ["Fixture required resource absent"])

    def test_builder_mutation_during_inspect_rejected(self):
        self.inspect_mutation = lambda: (self.template / "build_scene.py").write_text("# modified in inspection")
        result = self.generate()
        self.assertEqual(result["status"], "incomplete")
        self.assertIn("只读", result["remaining_issues"][0])

    def test_host_mutation_during_inspect_rejected(self):
        self.inspect_mutation = lambda: self.host_file.write_text("# altered host")
        result = self.generate()
        self.assertEqual(result["status"], "incomplete")
        self.assertIn("外围代码", result["remaining_issues"][0])

    def test_resume_ready_without_feedback_does_not_call_agent(self):
        self.generate()
        count = len(self.calls)
        result = workflow.resume_room(self.template)
        self.assertEqual(result["status"], "ready_for_review")
        self.assertEqual(len(self.calls), count)

    def test_resume_feedback_keeps_history_and_session(self):
        first = self.generate(max_repairs=1)
        first_path = Path(first["rounds"][0]["review"])
        first_hash = workflow._sha256(first_path)
        result = workflow.resume_room(self.template, feedback="Repair the floating toolbox.")
        self.assertEqual(result["status"], "ready_for_review")
        self.assertEqual(result["repairs_used"], 1)
        self.assertEqual(len(result["rounds"]), 2)
        self.assertEqual(workflow._sha256(first_path), first_hash)
        self.assertEqual([call["stage"] for call in self.calls], ["create", "inspect", "repair", "inspect"])
        self.assertTrue(all(call["session_id"] == self.session for call in self.calls[1:]))

    def test_resume_failed_inspect_uses_original_session(self):
        self.failure_stage = "inspect"
        first = self.generate()
        self.assertEqual(first["status"], "incomplete")
        self.failure_stage = None
        result = workflow.resume_room(self.template)
        self.assertEqual(result["status"], "ready_for_review")
        self.assertEqual(sum(call["stage"] == "create" for call in self.calls), 1)
        self.assertTrue(all(call["session_id"] == self.session for call in self.calls[1:]))

    def test_resume_unfinished_create_refuses_new_session(self):
        self.failure_stage = "create"
        self.generate()
        turn_path = self.template / "workflow" / "create" / "result.json"
        turn = json.loads(turn_path.read_text())
        turn["returncode"] = None
        write_json(turn_path, turn)
        count = len(self.calls)
        with self.assertRaises(RuntimeError):
            workflow.resume_room(self.template)
        self.assertEqual(len(self.calls), count)

    def test_resume_failed_create_continues_exact_session(self):
        self.failure_stage = "create"
        first = self.generate()
        turn_path = self.template / "workflow" / "create" / "result.json"
        old_hash = workflow._sha256(turn_path)
        self.assertTrue(first["pending_creation"])
        self.failure_stage = None
        with patch.object(workflow, "_stage_assets", side_effect=AssertionError("assets copied again")):
            result = workflow.resume_room(self.template)
        self.assertEqual(result["status"], "ready_for_review")
        self.assertNotIn("pending_creation", result)
        self.assertEqual(workflow._sha256(turn_path), old_hash)
        self.assertEqual([call["stage"] for call in self.calls], ["create", "create_resume", "inspect"])
        self.assertTrue(all(call["session_id"] == self.session for call in self.calls[1:]))
        self.assertEqual(len(result["codex_turns"]), 3)

    def test_resume_recovers_legacy_failed_create_session(self):
        self.failure_stage = "create"
        self.generate()
        result_path = self.template / "workflow_result.json"
        result = json.loads(result_path.read_text())
        result.update(session_id=None, codex_turns=[])
        result.pop("pending_creation")
        result.pop("active_turn_dir")
        write_json(result_path, result)
        self.failure_stage = None
        result = workflow.resume_room(self.template)
        self.assertEqual(result["status"], "ready_for_review")
        self.assertEqual(self.calls[1]["stage"], "create_resume")
        self.assertEqual(self.calls[1]["session_id"], self.session)

    def test_locked_resume_does_not_change_budget(self):
        self.generate()
        request_path = self.template / "request.json"
        old_hash = workflow._sha256(request_path)
        with workflow._task_lock(self.template):
            with self.assertRaises(RuntimeError):
                workflow.resume_room(self.template, max_repairs=12)
        self.assertEqual(workflow._sha256(request_path), old_hash)


if __name__ == "__main__":
    unittest.main(verbosity=2)
