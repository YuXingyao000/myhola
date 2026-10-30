"""CPU-only workflow tests; no real checkpoint, reconstruction or GPU jobs."""

import os
from pathlib import Path
import signal
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from src.eval import preference_ablation as workflow
from tools.vae_compare_post import atomic_json, fingerprint, read_json, sha256_file


class PreferenceAblationWorkflowTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="preference-ablation-workflow-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.protocol = self.make_protocol("workflow")

    def make_protocol(self, name):
        output = self.root / name
        config = {
            "output_dir": str(output), "old_run": str(self.root / "old"),
            "old_audit": str(self.root / "old_audit"), "font": None,
            "expected_count": 2, "bootstrap_repeats": 100,
        }
        comparison = {
            "config": {
                "run_dir": str(output / "comparison"), "original_checkpoint": str(self.root / "original.ckpt"),
                "preference_checkpoint": str(self.root / "new.ckpt"), "split_list": str(self.root / "validation.txt"),
                "expected_count": 2, "reuse_run_dir": config["old_run"], "reuse_methods": ["baseline"],
                "raw_data_root": str(self.root / "raw"), "condition_data_root": str(self.root / "images"),
                "gpus": [0, 1, 2, 3], "cpu_workers": 16, "seed": 42,
            },
            "identities": ["one", "two"],
            "checkpoint_sha256": {"original_checkpoint": "original-sha", "preference_checkpoint": "new-sha"},
        }
        old_audit = {"config": {
            "workers": 16, "timeout": 120., "distance_threshold": .01,
            "resolution": 224, "mesh_deflection": .002, "mask_threshold": 245,
        }}
        protocol = {"config": config, "comparison": comparison, "old_audit": old_audit,
                    "checkpoint_metadata": {"global_step": 5000}}
        protocol["fingerprint"] = fingerprint(protocol)
        return protocol

    def fake_validate(self, stage, protocol):
        # A real file makes receipt hashing and mutation detection meaningful.
        path = Path(protocol["config"]["output_dir"]) / stage / "result.json"
        self.assertTrue(path.is_file())
        return {str(path): sha256_file(path)}

    def fake_child(self, command, log):
        stage = log.stem
        directory = log.parent.parent / stage
        atomic_json(directory / "status.json", {"status": "complete"})
        atomic_json(directory / "result.json", {"stage": stage, "count": 2})

    def git_output(self, command, **kwargs):
        return "fixture-commit\n" if kwargs.get("text") else b""

    def complete_fixture(self, protocol=None):
        protocol = protocol or self.protocol
        with patch.object(workflow, "run_child", side_effect=self.fake_child) as child, \
                patch.object(workflow, "validate_stage", side_effect=self.fake_validate), \
                patch.object(workflow.subprocess, "check_output", side_effect=self.git_output):
            workflow.run_workflow(protocol)
            self.assertEqual(child.call_count, 3)

    def test_commands_reuse_only_baseline_and_audit_only_new_dpo(self):
        commands = workflow.commands(self.protocol)
        self.assertEqual(list(commands), ["comparison", "observation", "report"])
        comparison, audit = commands["comparison"], commands["observation"]
        self.assertEqual(comparison[comparison.index("--reuse-methods") + 1], "baseline")
        self.assertNotIn("dpo_5000", comparison)
        self.assertEqual(comparison[comparison.index("--reuse-run-dir") + 1], self.protocol["config"]["old_run"])
        self.assertEqual(audit[audit.index("--methods") + 1], "dpo_5000")
        self.assertNotIn("baseline", audit)
        self.assertEqual(audit[audit.index("--timeout") + 1], "120.0")
        self.assertEqual(commands["report"][commands["report"].index("--old-run") + 1], self.protocol["config"]["old_run"])

    def test_check_only_performs_preflight_without_launch_or_output(self):
        arguments = ["preference_ablation", "--check-only"]
        for name in ("old-run", "old-audit", "original-checkpoint", "preference-checkpoint", "split-list",
                     "raw-data-root", "condition-data-root", "output-dir"):
            path = Path(self.protocol["config"]["output_dir"]) if name == "output-dir" else self.root / name
            arguments.extend(["--" + name, str(path)])
        with patch.object(sys, "argv", arguments), patch.dict(os.environ), \
                patch.object(workflow, "preflight", return_value=self.protocol) as preflight, \
                patch.object(workflow, "run_workflow") as run:
            workflow.main()
            preflight.assert_called_once()
            run.assert_not_called()
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "")
        self.assertFalse(Path(self.protocol["config"]["output_dir"]).exists())

    def test_workflow_lock_rejects_concurrent_runner(self):
        with workflow.workflow_lock(self.root):
            with self.assertRaisesRegex(RuntimeError, "already in use"):
                with workflow.workflow_lock(self.root):
                    self.fail("Concurrent workflow acquired the lock")

    def test_completed_receipts_skip_all_children_and_stage_revalidation(self):
        self.complete_fixture()
        root = Path(self.protocol["config"]["output_dir"])
        with patch.object(workflow, "run_child") as child, patch.object(workflow, "validate_stage") as validate:
            workflow.run_workflow(self.protocol, resume=True)
            child.assert_not_called()
            validate.assert_not_called()
        self.assertEqual(read_json(root / "status.json")["status"], "complete")
        self.assertEqual(len(list((root / "receipts").glob("*.json"))), 3)

    def test_changed_completed_artifact_is_rejected_without_child(self):
        self.complete_fixture()
        root = Path(self.protocol["config"]["output_dir"])
        atomic_json(root / "comparison/result.json", {"changed": True})
        with patch.object(workflow, "run_child") as child:
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                workflow.run_workflow(self.protocol, resume=True)
            child.assert_not_called()
        self.assertEqual(read_json(root / "status.json")["status"], "failed")
        self.assertEqual(read_json(root / "status.json")["stage"], "comparison")

    def test_changed_external_source_in_receipt_is_rejected(self):
        self.complete_fixture()
        root = Path(self.protocol["config"]["output_dir"])
        source = self.root / "immutable-input.bin"
        source.write_bytes(b"original source")
        path = root / "receipts/comparison.json"
        receipt = read_json(path)
        receipt["artifacts"][str(source)] = sha256_file(source)
        receipt.pop("fingerprint")
        receipt["fingerprint"] = fingerprint(receipt)
        atomic_json(path, receipt)
        source.write_bytes(b"changed source")
        with patch.object(workflow, "run_child") as child:
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                workflow.run_workflow(self.protocol, resume=True)
            child.assert_not_called()

    def test_receipts_must_belong_to_current_workflow_and_stage(self):
        self.complete_fixture()
        root = Path(self.protocol["config"]["output_dir"])
        path = root / "receipts/comparison.json"
        original = read_json(path)
        for change in ({"workflow_fingerprint": "other-workflow"}, {"stage": "report"}):
            row = {**original, **change}
            row.pop("fingerprint")
            row["fingerprint"] = fingerprint(row)
            atomic_json(path, row)
            with self.subTest(change=change), patch.object(workflow, "run_child") as child:
                with self.assertRaises(ValueError):
                    workflow.run_workflow(self.protocol, resume=True)
                child.assert_not_called()

    def test_failed_and_interrupted_workflows_record_stage_and_restore_signals(self):
        handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
        for name, error, status in (("failed", RuntimeError("child failed"), "failed"),
                                    ("interrupted", KeyboardInterrupt("stop"), "interrupted")):
            protocol = self.make_protocol(name)
            with self.subTest(status=status), \
                    patch.object(workflow, "run_child", side_effect=error), \
                    patch.object(workflow.subprocess, "check_output", side_effect=self.git_output):
                with self.assertRaises(type(error)):
                    workflow.run_workflow(protocol)
            row = read_json(Path(protocol["config"]["output_dir"]) / "status.json")
            self.assertEqual(row["status"], status)
            self.assertEqual(row["stage"], "comparison")
            self.assertEqual({sig: signal.getsignal(sig) for sig in handlers}, handlers)

    def test_resume_existing_incomplete_stage_gets_resume_flag(self):
        root = Path(self.protocol["config"]["output_dir"])
        root.mkdir()
        atomic_json(root / "protocol.json", self.protocol)
        atomic_json(root / "comparison/status.json", {"status": "interrupted"})
        atomic_json(root / "comparison/metadata/protocol.json", self.protocol["comparison"])
        with patch.object(workflow, "run_child", side_effect=self.fake_child) as child, \
                patch.object(workflow, "validate_stage", side_effect=self.fake_validate):
            workflow.run_workflow(self.protocol, resume=True)
        commands = [call.args[0] for call in child.call_args_list]
        self.assertIn("--resume", commands[0])
        self.assertNotIn("--resume", commands[1])
        self.assertNotIn("--resume", commands[2])

    def test_empty_interrupted_initializations_are_preserved_and_started_fresh(self):
        root = Path(self.protocol["config"]["output_dir"])
        root.mkdir()
        atomic_json(root / "protocol.json", self.protocol)
        for stage in workflow.STAGES:
            (root / stage / "metadata").mkdir(parents=True)
        (root / "comparison/.comparison.lock").write_text("preserve comparison lock")
        (root / "report/.lock").write_text("preserve report lock")
        with patch.object(workflow, "run_child", side_effect=self.fake_child) as child, \
                patch.object(workflow, "validate_stage", side_effect=self.fake_validate):
            workflow.run_workflow(self.protocol, resume=True)
        self.assertEqual(child.call_count, 3)
        for call in child.call_args_list:
            self.assertNotIn("--resume", call.args[0])
        archived = list((root / "interrupted_initializations").iterdir())
        self.assertEqual(len(archived), 3)
        self.assertEqual({path.name.split("-", 1)[0] for path in archived}, set(workflow.STAGES))
        previous_comparison = next(path for path in archived if path.name.startswith("comparison-"))
        self.assertEqual((previous_comparison / ".comparison.lock").read_text(), "preserve comparison lock")
        self.assertEqual(read_json(root / "status.json")["status"], "complete")

    def test_artifact_without_stage_protocol_is_rejected_and_preserved(self):
        root = Path(self.protocol["config"]["output_dir"])
        root.mkdir()
        atomic_json(root / "protocol.json", self.protocol)
        artifact = root / "comparison/raw_predictions/candidate.npy"
        artifact.parent.mkdir(parents=True)
        artifact.write_bytes(b"existing generated artifact")
        with patch.object(workflow, "run_child") as child:
            with self.assertRaisesRegex(ValueError, "artifacts but no frozen protocol"):
                workflow.run_workflow(self.protocol, resume=True)
            child.assert_not_called()
        self.assertEqual(artifact.read_bytes(), b"existing generated artifact")
        self.assertFalse((root / "interrupted_initializations").exists())

    def test_stage_directory_symlink_is_rejected_without_touching_target(self):
        root = Path(self.protocol["config"]["output_dir"])
        root.mkdir()
        atomic_json(root / "protocol.json", self.protocol)
        target = self.root / "external-comparison"
        atomic_json(target / "status.json", {"status": "complete"})
        (root / "comparison").symlink_to(target, target_is_directory=True)
        original = (target / "status.json").read_bytes()
        with patch.object(workflow, "run_child") as child:
            with self.assertRaisesRegex(ValueError, "must be a local directory"):
                workflow.run_workflow(self.protocol, resume=True)
            child.assert_not_called()
        self.assertTrue((root / "comparison").is_symlink())
        self.assertEqual((target / "status.json").read_bytes(), original)
        self.assertEqual([path.name for path in target.iterdir()], ["status.json"])

    def test_empty_stage_recovery_requires_exclusive_stage_lock(self):
        directory = self.root / "comparison"
        directory.mkdir()
        with (directory / ".comparison.lock").open("a") as lock:
            workflow.fcntl.flock(lock.fileno(), workflow.fcntl.LOCK_EX | workflow.fcntl.LOCK_NB)
            with self.assertRaises((BlockingIOError, RuntimeError)):
                workflow.stage_needs_resume(directory, "comparison")
        self.assertTrue(directory.is_dir())
        self.assertFalse((self.root / "interrupted_initializations").exists())

    def test_completed_stage_without_receipt_is_validated_without_relaunch(self):
        root = Path(self.protocol["config"]["output_dir"])
        root.mkdir()
        atomic_json(root / "protocol.json", self.protocol)
        for stage in workflow.STAGES:
            atomic_json(root / stage / "status.json", {"status": "complete"})
            atomic_json(root / stage / "result.json", {"stage": stage})
        with patch.object(workflow, "run_child") as child, \
                patch.object(workflow, "validate_stage", side_effect=self.fake_validate) as validate:
            workflow.run_workflow(self.protocol, resume=True)
            child.assert_not_called()
            self.assertEqual(validate.call_count, 3)

    def test_run_child_cleans_up_cpu_grandchild_on_failed_coordinator(self):
        pid_file = self.root / "grandchild.pid"
        script = (
            "import pathlib,subprocess,sys; "
            "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); "
            f"pathlib.Path({str(pid_file)!r}).write_text(str(child.pid)); "
            "sys.exit(7)"
        )
        try:
            with self.assertRaisesRegex(RuntimeError, "code 7"):
                workflow.run_child([sys.executable, "-c", script], self.root / "cpu-child.log")
            pid = int(pid_file.read_text())
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                status = Path(f"/proc/{pid}/stat")
                try:
                    state = status.read_text().split()[2]
                except (FileNotFoundError, ProcessLookupError):
                    break
                if state == "Z":
                    break
                time.sleep(.02)
            else:
                self.fail("CPU grandchild survived failed stage cleanup")
        finally:
            if pid_file.is_file():
                try:
                    os.kill(int(pid_file.read_text()), signal.SIGKILL)
                except ProcessLookupError:
                    pass


if __name__ == "__main__":
    unittest.main()
