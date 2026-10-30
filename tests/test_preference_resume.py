"""Scheduling changes preserve the frozen scientific protocol and its provenance."""

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.preference import io
from src.preference import generate as generation
from src.preference.selection import SelectionConfig
from src.common.io import atomic_json, sha256_file


class PreferenceRuntimeRevisionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="hola-preference-resume-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.repo = self.root / "repo"
        self.sources = sorted(io.SCHEDULING_SOURCES | {
            "src/models/diffusion/legacy.py",
            "src/eval/preference.py",
            "src/preference/selection.py",
        })
        for relative in self.sources:
            path = self.repo / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"# original {relative}\n")
        (self.root / "checkpoint.ckpt").write_bytes(b"fixed model")
        (self.root / "train.txt").write_text("train_a\ntrain_b\n")
        (self.root / "validation.txt").write_text("val_a\n")
        dino = self.root / "dino"
        dino.mkdir()
        (dino / "hubconf.py").write_text("# fixed DINO\n")
        self.config = {
            "checkpoint": str(self.root / "checkpoint.ckpt"), "dino_repo": str(dino),
            "train_list": str(self.root / "train.txt"),
            "validation_list": str(self.root / "validation.txt"),
            "raw_data_root": str(self.root / "cad"),
            "condition_data_root": str(self.root / "images"),
            "seed": 42, "candidates": 8, "gpu_cpu_threads": 2,
            "gpus": [4, 5, 6, 7], "workers_per_gpu": 1,
            "cpu_workers": 16, "queue_size": 32,
            "selection": asdict(SelectionConfig()),
            "post": {"timeout_seconds": 180, "max_optimize_iter": 200},
        }
        for mock in (
            patch.object(io, "REPO_ROOT", self.repo),
            patch.object(io, "source_hashes", self.hashes),
            patch.object(io.subprocess, "check_output", side_effect=self.command_output),
        ):
            mock.start()
            self.addCleanup(mock.stop)
        self.run = self.root / "run"
        self.original = io.prepare_run(self.config, self.run, resume=False)
        self.protocol_path = self.run / "metadata/protocol.json"
        self.original_bytes = self.protocol_path.read_bytes()

    def hashes(self):
        return {relative: sha256_file(self.repo / relative) for relative in self.sources}

    @staticmethod
    def command_output(command, **kwargs):
        if command[:2] == ["git", "rev-parse"]:
            return "fixture-commit\n"
        if command[0] == "git":
            return ""
        if command[0] == "nvidia-smi":
            return "CPU-only test fixture\n"
        raise AssertionError(f"Unexpected command: {command}")

    def test_strict_resume_rejects_changed_scheduler_without_writing_revision(self):
        source = self.repo / "src/preference/generate.py"
        source.write_text("# two workers per GPU\n")
        with self.assertRaisesRegex(ValueError, "Resume protocol"):
            io.prepare_run(self.config, self.run, resume=True)
        self.assertEqual(self.protocol_path.read_bytes(), self.original_bytes)
        self.assertFalse((self.run / "metadata/runtime_revisions.json").exists())

    def test_explicit_scheduler_revision_preserves_protocol_and_saves_exact_source(self):
        relative = "src/preference/generate.py"
        (self.repo / relative).write_text("# two workers per GPU\n")
        resources = {**self.config, "gpus": list(range(8)), "workers_per_gpu": 2,
                     "cpu_workers": 32, "queue_size": 64}
        result = io.prepare_run(resources, self.run, resume=True, accept_scheduling_update=True)
        self.assertEqual(result, self.original)
        self.assertEqual(self.protocol_path.read_bytes(), self.original_bytes)
        revisions = json.loads((self.run / "metadata/runtime_revisions.json").read_text())["revisions"]
        self.assertEqual(len(revisions), 1)
        row = revisions[0]
        self.assertEqual(row["protocol_fingerprint"], self.original["fingerprint"])
        self.assertEqual(row["resources"]["gpus"], list(range(8)))
        self.assertEqual(row["resources"]["workers_per_gpu"], 2)
        self.assertEqual(row["source_changes"], {relative: {
            "before": self.original["source_sha256"][relative],
            "after": sha256_file(self.repo / relative),
        }})
        runtime = self.run / row["runtime_directory"]
        self.assertEqual((runtime / "source" / relative).read_bytes(), (self.repo / relative).read_bytes())
        self.assertIn("+# two workers per GPU", (runtime / "scheduling.diff").read_text())

    def test_resource_only_resume_is_recorded_and_revisions_are_appended(self):
        for workers in (2, 3):
            result = io.prepare_run({**self.config, "workers_per_gpu": workers}, self.run, resume=True)
            self.assertEqual(result, self.original)
        revisions = json.loads((self.run / "metadata/runtime_revisions.json").read_text())["revisions"]
        self.assertEqual([row["resources"]["workers_per_gpu"] for row in revisions], [2, 3])
        self.assertTrue(all(row["source_changes"] == {} for row in revisions))
        self.assertNotEqual(revisions[0]["runtime_directory"], revisions[1]["runtime_directory"])
        self.assertEqual(self.protocol_path.read_bytes(), self.original_bytes)

    def test_scheduler_permission_does_not_allow_scientific_config_changes(self):
        variants = []
        for key, value in (("candidates", 16), ("seed", 43), ("gpu_cpu_threads", 4)):
            variants.append({**self.config, key: value})
        post = deepcopy(self.config)
        post["post"]["timeout_seconds"] = 360
        variants.append(post)
        selection = deepcopy(self.config)
        selection["selection"]["cd_improvement"] = .01
        variants.append(selection)
        for config in variants:
            with self.subTest(config=config), self.assertRaisesRegex(ValueError, "Resume protocol"):
                io.prepare_run(config, self.run, resume=True, accept_scheduling_update=True)
        self.assertFalse((self.run / "metadata/runtime_revisions.json").exists())

    def test_scheduler_permission_does_not_allow_model_metric_or_selection_source_changes(self):
        for relative in set(self.sources) - io.SCHEDULING_SOURCES:
            source = self.repo / relative
            original = source.read_bytes()
            source.write_text("# changed science\n")
            with self.subTest(source=relative), self.assertRaisesRegex(ValueError, "Resume protocol"):
                io.prepare_run(self.config, self.run, resume=True, accept_scheduling_update=True)
            source.write_bytes(original)
        self.assertEqual(self.protocol_path.read_bytes(), self.original_bytes)

    def test_scheduler_permission_does_not_allow_added_or_removed_source_paths(self):
        original_sources = list(self.sources)
        self.sources.remove("src/preference/generate.py")
        with self.assertRaisesRegex(ValueError, "Resume protocol"):
            io.prepare_run(self.config, self.run, resume=True, accept_scheduling_update=True)
        self.sources = original_sources
        added = "src/preference/other.py"
        (self.repo / added).write_text("# additional behavior\n")
        self.sources.append(added)
        with self.assertRaisesRegex(ValueError, "Resume protocol"):
            io.prepare_run(self.config, self.run, resume=True, accept_scheduling_update=True)

    def test_changed_checkpoint_or_split_is_rejected_even_with_permission(self):
        for path, changed in ((self.root / "checkpoint.ckpt", b"changed model"),
                              (self.root / "train.txt", b"train_b\ntrain_a\n")):
            original = path.read_bytes()
            path.write_bytes(changed)
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "Resume protocol"):
                io.prepare_run(self.config, self.run, resume=True, accept_scheduling_update=True)
            path.write_bytes(original)

    def test_corrupt_original_fingerprint_is_not_accepted_as_scheduling_migration(self):
        changed = {**self.original, "fingerprint": "not-the-original-fingerprint"}
        self.protocol_path.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, "Resume protocol"):
            io.prepare_run(self.config, self.run, resume=True, accept_scheduling_update=True)

    def test_missing_original_protocol_is_rejected(self):
        self.protocol_path.unlink()
        with self.assertRaisesRegex(ValueError, "Resume protocol is missing"):
            io.prepare_run(self.config, self.run, resume=True, accept_scheduling_update=True)


class PreferenceCompletedResumeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="hola-preference-completed-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.run = self.root / "run"
        self.run.mkdir()
        self.config = {
            "raw_data_root": str(self.root / "cad"),
            "condition_data_root": str(self.root / "images"),
            "candidates": 8, "seed": 42, "selection": asdict(SelectionConfig()),
        }

    def image_fixture(self, image_id="part", split="train"):
        image_root = self.run / "candidates" / split / image_id
        image_root.mkdir(parents=True)
        gt = Path(self.config["raw_data_root"]) / image_id
        gt.mkdir(parents=True)
        condition = Path(self.config["condition_data_root"]) / image_id / "imgs.npz"
        condition.parent.mkdir(parents=True)
        sources = {}
        for path in (condition, gt / "data.npz", gt / "normalized_shape.step"):
            path.write_bytes(f"fixture {path.name}".encode())
            sources[str(path)] = sha256_file(path)
        atomic_json(image_root / "sources.json", sources)
        cache = self.run / "gt_cache" / f"{image_id}_0"
        cache.mkdir(parents=True)
        (cache / "features.npy").write_bytes(b"original GT latent fixture")
        atomic_json(cache / "source.json", {
            "gt_npz_sha256": sources[str(gt / "data.npz")],
            "output_sha256": {"features.npy": sha256_file(cache / "features.npy")},
        })
        records = []
        for index in range(self.config["candidates"]):
            candidate_id = f"c{index:02d}"
            directory = image_root / candidate_id
            directory.mkdir()
            if index >= 2:
                row = generation._unknown(image_id, candidate_id, "generation_error", "fixture failure")
                atomic_json(directory / "inference.json", row)
                records.append(row)
                continue
            outputs = {}
            for name in ("data.npz", "latent.npy", "mappings.npz"):
                path = directory / name
                path.write_bytes(f"{image_id}/{candidate_id}/{name}".encode())
                outputs[name] = sha256_file(path)
            atomic_json(directory / "inference.json", {
                "status": "ok", "image_id": image_id, "candidate_id": candidate_id,
                "output_sha256": outputs,
            })
            evaluation = directory / "evaluation"
            evaluation.mkdir()
            metrics = {"face_cd_sq": 1., "edge_cd_sq": 1., "face_f1": .9}
            row = {
                "image_id": image_id, "candidate_id": candidate_id,
                "status": "ok", "validity": "valid" if index == 0 else "invalid",
                "raw": metrics, "post": [], "protocol_fingerprint": f"fixture-{candidate_id}",
            }
            atomic_json(evaluation / "metrics.json", row)
            atomic_json(evaluation / "protocol.json", {
                "protocol_fingerprint": row["protocol_fingerprint"],
                "input_sha256": {
                    "raw_path": outputs["data.npz"], "gt_npz": sources[str(gt / "data.npz")],
                    "gt_step": sources[str(gt / "normalized_shape.step")],
                },
            })
            records.append(row)
        return image_root, records

    def test_worker_shards_cover_images_once_and_keep_same_gpu_slots_distinct(self):
        identities = [f"part_{index}" for index in range(29)]
        assignments = generation.worker_assignments(list(range(8)), 2, identities)
        self.assertEqual(len(assignments), 16)
        self.assertEqual(len({(gpu, slot) for gpu, slot, _ in assignments}), 16)
        flattened = [image_id for _, _, ids in assignments for image_id in ids]
        self.assertEqual(sorted(flattened), sorted(identities))
        self.assertEqual(len(flattened), len(set(flattened)))
        for gpu in range(8):
            self.assertEqual({slot for assigned_gpu, slot, _ in assignments if assigned_gpu == gpu}, {0, 1})

    def test_short_shards_use_all_available_gpus_before_second_slot(self):
        assignments = generation.worker_assignments([4, 5, 6, 7], 3, ["a", "b", "c"])
        self.assertEqual(assignments, [(4, 0, ["a"]), (5, 0, ["b"]), (6, 0, ["c"])])
        self.assertEqual(generation.worker_assignments([0, 1], 2, []), [])

    def test_complete_image_restores_all_terminal_candidates(self):
        _, records = self.image_fixture()
        restored = generation._completed_records(self.run, "train", "part", self.config)
        self.assertEqual(restored, records)
        self.assertEqual(len(restored), 8)
        self.assertEqual(sum(row["validity"] == "unknown" for row in restored), 6)

    def test_partial_or_running_image_is_left_for_normal_resume(self):
        root, records = self.image_fixture()
        terminal_path = root / "c01/evaluation/metrics.json"
        atomic_json(terminal_path, {**records[1], "status": "running"})
        self.assertIsNone(generation._completed_records(self.run, "train", "part", self.config))
        atomic_json(terminal_path, records[1])
        terminal_path.unlink()
        self.assertIsNone(generation._completed_records(self.run, "train", "part", self.config))
        atomic_json(terminal_path, records[1])
        (root / "c07/inference.json").unlink()
        self.assertIsNone(generation._completed_records(self.run, "train", "part", self.config))

    def test_complete_image_still_rejects_changed_latent_source_and_gt_cache(self):
        root, _ = self.image_fixture()
        paths = (
            root / "c00/latent.npy",
            Path(self.config["condition_data_root"]) / "part/imgs.npz",
            self.run / "gt_cache/part_0/features.npy",
        )
        for path in paths:
            original = path.read_bytes()
            path.write_bytes(b"changed artifact")
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "artifact changed"):
                generation._completed_records(self.run, "train", "part", self.config)
            path.write_bytes(original)

    def test_complete_image_rejects_changed_metric_input_mapping_or_fingerprint(self):
        root, _ = self.image_fixture()
        path = root / "c00/evaluation/protocol.json"
        original = json.loads(path.read_text())
        for key in ("raw_path", "gt_npz", "gt_step", "protocol_fingerprint"):
            changed = deepcopy(original)
            if key == "protocol_fingerprint":
                changed[key] = "different metric protocol"
            else:
                changed["input_sha256"][key] = "different input"
            atomic_json(path, changed)
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "evaluation inputs differ"):
                generation._completed_records(self.run, "train", "part", self.config)
        atomic_json(path, original)

    def test_completed_phase_restores_counts_pairs_and_elapsed_without_processes(self):
        self.image_fixture("part_a")
        self.image_fixture("part_b")
        atomic_json(self.run / "status.json", {
            "status": "failed", "phase": "train", "elapsed_seconds": 1234.5,
        })
        with patch.object(generation.mp, "get_context", side_effect=AssertionError("No workers needed")):
            summary = generation.run_phase("train", ["part_a", "part_b"], self.config, self.run)
        self.assertEqual(summary["images"], 2)
        self.assertEqual(summary["candidates"], 16)
        self.assertEqual(summary["pairs"], 2)
        self.assertEqual(summary["pairs_validity"], 2)
        self.assertEqual(summary["validity_valid"], 2)
        self.assertEqual(summary["validity_invalid"], 2)
        self.assertEqual(summary["validity_unknown"], 12)
        self.assertEqual(summary["elapsed_seconds"], 1234.5)
        rows = [json.loads(line) for line in (self.run / "pairs.jsonl").read_text().splitlines()]
        self.assertEqual([row["image_id"] for row in rows], ["part_a", "part_b"])
        self.assertFalse((self.run / "pairs.in_progress.jsonl").exists())
        self.assertEqual(json.loads((self.run / "metadata/train_summary.json").read_text()), summary)

    def test_completed_validation_does_not_overwrite_train_elapsed_or_status(self):
        self.image_fixture("val_a", split="validation")
        old_status = {"status": "failed", "phase": "train", "elapsed_seconds": 5678., "images": 19}
        atomic_json(self.run / "status.json", old_status)
        atomic_json(self.run / "metadata/validation_summary.json", {"elapsed_seconds": 90.})
        before = (self.run / "status.json").read_bytes()
        with patch.object(generation.mp, "get_context", side_effect=AssertionError("No workers needed")):
            summary = generation.run_phase("validation", ["val_a"], self.config, self.run)
        self.assertEqual(summary["elapsed_seconds"], 90.)
        self.assertEqual(summary["images"], 1)
        self.assertEqual((self.run / "status.json").read_bytes(), before)
        self.assertTrue((self.run / "validation_pairs.jsonl").is_file())


if __name__ == "__main__":
    unittest.main()
