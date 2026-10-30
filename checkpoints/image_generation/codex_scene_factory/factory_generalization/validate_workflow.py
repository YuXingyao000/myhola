"""验证房间工作流的质量参考模式和素材接口；仅模拟 Agent/Blender，不运行服务。"""

import importlib.util
import json
from pathlib import Path
import shutil
import sys
import unittest
from unittest.mock import patch

from PIL import Image


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "packages" / "cad_scene_factory"))
import assets
import room_reference
import room_workflow as workflow

# 复用已有的临时目录与会话模拟器，不重复执行或修改原来的 19 项测试。
BASELINE_FILE = Path(__file__).resolve().parents[1] / "room_workflow/validate_workflow.py"
spec = importlib.util.spec_from_file_location("baseline_room_validation", BASELINE_FILE)
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


class FactoryWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.fixture = baseline.RoomWorkflowTests()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.setUp()
        self.root = self.fixture.root
        self.template = self.fixture.template

        original = read_json(self.fixture.reference)
        original["images"][0].update(
            id="bench_context", description="A quality example, not a factory layout.",
            render_record={"source_blend": "/old/source.blend", "camera": [4, 5, 6]},
        )
        original["audit_sources"] = [{"path": "/old/scene_inventory.json"}]
        original["source"].update(project_url="https://example.invalid/reference",
                                  license_url="https://example.invalid/license")
        write_json(self.fixture.reference, original)
        self.quality_dir = self.root / "quality_reference"
        self.quality = room_reference.prepare_quality_reference(self.fixture.reference, self.quality_dir)
        self.quality_file = self.quality_dir / "reference.json"

        self.catalog = self.root / "local_catalog.json"
        write_json(self.catalog, [{
            "provider": "local", "asset_id": "machine_fixture", "kind": "model",
            "name": "Machine fixture", "source_dir": str(self.fixture.source),
            "source_url": "local-fixture", "license": "CC0-1.0",
        }])
        self.cache = self.root / "shared_asset_cache"

    def generate(self, **kwargs):
        return workflow.generate_room(
            "Build a small machine shop from quality examples.", self.quality_file,
            self.template, Path("/fixture/blender"), reference_mode="quality_only", **kwargs,
        )

    def request(self):
        return read_json(self.template / "request.json")

    def test_quality_pack_omits_source_engineering_and_layout(self):
        forbidden = {"source_blend", "asset_directory", "asset_manifest", "asset_subdir",
                     "render_record", "audit_sources", "semantic_groups", "lighting_summary"}

        def check_keys(value):
            if isinstance(value, dict):
                self.assertFalse(forbidden.intersection(value))
                for item in value.values():
                    check_keys(item)
            elif isinstance(value, list):
                for item in value:
                    check_keys(item)

        check_keys(self.quality)
        files = {path.relative_to(self.quality_dir).as_posix()
                 for path in self.quality_dir.rglob("*") if path.is_file()}
        self.assertEqual(files, {"reference.json", "overview.md",
                                 "images/" + self.fixture.reference_image.name})
        self.assertEqual(self.quality["source"]["project_url"], "https://example.invalid/reference")
        self.assertEqual(self.quality["source"]["license"], "CC0")
        for item in self.quality["images"] + self.quality["documents"]:
            self.assertEqual(workflow._sha256(Path(item["path"])), item["sha256"])

    def test_quality_only_finishes_without_source_assets_or_blend(self):
        # 物理移除原工程、清单及原参考，证明执行时不需要回读它们。
        shutil.rmtree(self.fixture.source)
        self.fixture.asset_manifest.unlink()
        self.fixture.reference.unlink()
        with patch.object(workflow.shutil, "copytree", side_effect=AssertionError("Source copied")):
            result = self.generate()
        self.assertEqual(result["status"], "ready_for_review")
        self.assertEqual(result["human_review"], "pending")
        self.assertEqual(read_json(self.template / "assets.json")["files"], [])
        self.assertEqual(list((self.template / "assets").rglob("*")), [])
        self.assertEqual([call["stage"] for call in self.fixture.calls], ["create", "inspect"])
        self.assertNotIn("source_blend", self.request()["runtime"]["reference_bundle"])

    def test_quality_only_rejects_full_project_reference(self):
        with self.assertRaisesRegex(ValueError, "仅含图片与说明"):
            workflow.generate_room("Independent factory", self.fixture.reference, self.template,
                                   Path("/fixture/blender"), reference_mode="quality_only")
        self.assertFalse(self.template.exists())
        self.assertEqual(self.fixture.calls, [])

    def test_default_adapt_still_copies_and_verifies_assets(self):
        result = self.fixture.generate()
        self.assertEqual(result["status"], "ready_for_review")
        self.assertEqual(self.request()["runtime"]["reference_mode"], "adapt")
        source = self.fixture.source / "base.blend"
        copied = self.template / "assets/the_shed/base.blend"
        self.assertEqual(copied.read_bytes(), source.read_bytes())
        copied.write_bytes(b"changed base engineering")
        with self.assertRaisesRegex(RuntimeError, "只读基础资产已变化"):
            workflow._check_host(self.request())

    def test_adapt_rejects_changed_source_before_agent(self):
        (self.fixture.source / "base.blend").write_bytes(b"changed source engineering")
        result = self.fixture.generate()
        self.assertEqual(result["status"], "incomplete")
        self.assertIn("参考源工程摘要变化", result["remaining_issues"][0])
        self.assertEqual(self.fixture.calls, [])

    def test_runtime_contains_asset_tool_shared_cache_catalog_and_guide(self):
        result = self.generate(dry_run=True, asset_cache=self.cache, asset_catalog=self.catalog)
        self.assertEqual(result["status"], "dry_run")
        request = self.request()
        runtime = request["runtime"]
        self.assertEqual(runtime["assets_script"], str(workflow.DIRECTORY / "assets.py"))
        self.assertEqual(runtime["asset_cache"], str(self.cache.resolve()))
        self.assertEqual(runtime["asset_catalog"], str(self.catalog.resolve()))
        guide = (workflow.PROMPTS.parent / "ASSET_TOOLS.md").read_text(encoding="utf-8")
        self.assertEqual(request["instruction_files"]["ASSET_TOOLS.md"], guide)
        self.assertIn(guide, request["developer_instructions"])
        self.assertIn(str(self.cache), request["developer_instructions"])
        self.assertIn(str(self.catalog), request["developer_instructions"])

    def test_changed_quality_image_is_detected(self):
        self.generate(dry_run=True)
        image = Path(self.quality["images"][0]["path"])
        Image.new("RGB", (32, 24), "red").save(image)
        with self.assertRaisesRegex(RuntimeError, "参考图片或说明已变化"):
            workflow._check_host(self.request())

    def test_changed_quality_document_is_detected(self):
        self.generate(dry_run=True)
        document = Path(self.quality["documents"][0]["path"])
        document.write_text("Changed quality expectations", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "参考图片或说明已变化"):
            workflow._check_host(self.request())

    def prepare_downloaded_candidate(self, flat_manifest=False):
        self.generate(dry_run=True)
        self.fixture.write_candidate()
        (self.template / "workflow").mkdir()
        workflow._stage_assets(self.request())
        assets.fetch_asset("local", "machine_fixture", self.template, self.cache, catalog=self.catalog)
        if flat_manifest:
            manifest = read_json(self.template / "assets.json")
            write_json(self.template / "assets.json", manifest["files"])

    def blender_fixture(self, request, arguments, log_file):
        """只落下 preview_room 所需的构建／图片证据，不启动子进程。"""
        output = Path(arguments[arguments.index("--output_dir") + 1])
        if arguments[0] == "build":
            write_json(output / "build_report.json", {"status": "pass"})
            (output / "room.blend").write_bytes(b"fixture-only-blend")
            return
        output.mkdir()
        results = []
        for index, view_id in enumerate(sorted(workflow.VIEW_IDS)):
            Image.new("RGB", (32, 24), (index * 20, 40, 80)).save(output / (view_id + ".png"))
            results.append({"status": "success", "image_file": view_id + ".png"})
        write_json(output / "render_report.json", {"success": 6, "failed": 0, "results": results})

    def run_actual_preview(self, output_name):
        # 旧 fixture 替换了 preview_room；此处显式调用保存下来的真实实现。
        return ACTUAL_PREVIEW(self.request(), self.root / output_name)

    def test_corrupt_download_is_rejected_before_blender(self):
        self.prepare_downloaded_candidate()
        record = read_json(self.template / "assets.json")["files"][0]
        (self.template / record["path"]).write_bytes(b"corrupted downloaded asset")
        with patch.object(workflow, "_blender") as blender:
            report = self.run_actual_preview("rejected_preview")
        blender.assert_not_called()
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["success"], 0)
        self.assertEqual(report["failed"], 6)
        self.assertIn("素材摘要不符", report["errors"][0])

    def check_valid_preview(self, flat_manifest):
        self.prepare_downloaded_candidate(flat_manifest=flat_manifest)
        with patch.object(workflow, "_blender", side_effect=self.blender_fixture) as blender:
            report = self.run_actual_preview("accepted_preview")
        self.assertEqual(blender.call_count, 2)
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["asset_files_checked"], 2)
        self.assertEqual(report["success"], 6)
        self.assertEqual(report["failed"], 0)

    def test_room_manifest_downloads_pass_preview_gate(self):
        self.check_valid_preview(flat_manifest=False)

    def test_flat_manifest_downloads_pass_preview_gate(self):
        self.check_valid_preview(flat_manifest=True)


ACTUAL_PREVIEW = workflow.preview_room


if __name__ == "__main__":
    unittest.main(verbosity=2)
