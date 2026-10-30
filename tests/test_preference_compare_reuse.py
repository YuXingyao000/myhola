"""CPU-only reuse fixtures: provenance, immutable failures and method scheduling."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from src.preference.generate import candidate_seed
from src.eval.preference_compare import (
    METHODS, build_protocol, comparison_lock, evaluation_jobs, generation_assignments,
    reuse_completed_run, validate_reuse_protocol, verify_completed_comparison,
)
from tools.vae_compare_post import atomic_json, fingerprint, read_json, sha256_file


class PreferenceCompareReuseTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="preference-compare-reuse-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source, self.output = self.root / "source", self.root / "new"
        self.source.mkdir()
        self.output.mkdir()
        self.identities = ["one", "two", "failed"]
        self.split = self.source / "validation.txt"
        self.split.write_text("\n".join(self.identities) + "\n")
        for name in ("original.ckpt", "old.ckpt", "new.ckpt"):
            (self.root / name).write_text(name)
        old_config = {
            "run_dir": str(self.source), "original_checkpoint": str(self.root / "original.ckpt"),
            "preference_checkpoint": str(self.root / "old.ckpt"),
            "raw_data_root": str(self.root / "raw"), "condition_data_root": str(self.root / "images"),
            "split_list": str(self.split), "expected_count": len(self.identities),
            "reuse_run_dir": None, "gpus": [0, 1], "cpu_workers": 1, "seed": 42,
        }
        for image_id in self.identities:
            inputs = {
                "image": self.root / "images" / image_id / "imgs.npz",
                "gt_npz": self.root / "raw" / image_id / "data.npz",
                "gt_step": self.root / "raw" / image_id / "normalized_shape.step",
            }
            for path in inputs.values():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(str(path))
            for method in METHODS:
                raw = self.source / "raw_predictions" / method / image_id
                raw.mkdir(parents=True)
                inference = {
                    "image_id": image_id, "method": method,
                    "seed": candidate_seed(42, image_id, 0),
                    "status": "generation_error" if image_id == "failed" else "ok", "seconds": 1.,
                }
                if image_id == "failed":
                    atomic_json(raw / "inference.json", inference)
                    continue
                for name in ("data.npz", "latent.npy"):
                    (raw / name).write_text(f"generated {method} {image_id} {name}")
                inference.update(input_sha256={name: sha256_file(path) for name, path in inputs.items()},
                                 output_sha256={name: sha256_file(raw / name) for name in ("data.npz", "latent.npy")})
                atomic_json(raw / "inference.json", inference)
                evaluation = self.source / "evaluations" / method / image_id
                evaluation.mkdir(parents=True)
                evaluation_protocol = {
                    "version": "svr_preference_metrics_v1", "image_id": image_id, "candidate_id": "c00",
                    "raw_path": str(raw / "data.npz"), "output_dir": str(evaluation),
                    "gt_npz": str(inputs["gt_npz"]), "gt_step": str(inputs["gt_step"]),
                    "seed": 42, "timeout_seconds": 180, "max_optimize_iter": 200, "repeats": 3,
                    "input_sha256": {"raw_path": inference["output_sha256"]["data.npz"],
                                     "gt_npz": inference["input_sha256"]["gt_npz"],
                                     "gt_step": inference["input_sha256"]["gt_step"]},
                }
                evaluation_protocol["protocol_fingerprint"] = fingerprint(evaluation_protocol)
                atomic_json(evaluation / "protocol.json", evaluation_protocol)
                step = evaluation / "post/c00/recon_brep.step"
                step.parent.mkdir(parents=True)
                step.write_text(f"native step {image_id} {method}")
                atomic_json(evaluation / "metrics.json", {
                    "image_id": image_id, "candidate_id": "c00", "status": "ok", "validity": "valid",
                    "protocol_fingerprint": evaluation_protocol["protocol_fingerprint"],
                    "raw": {"face_cd_sq": .1, "edge_cd_sq": .2, "face_f1": .9},
                    "step_path": str(step), "step_sha256": sha256_file(step), "post": [],
                })
        self.old_protocol = build_protocol(old_config, self.identities)
        atomic_json(self.source / "metadata/protocol.json", self.old_protocol)
        atomic_json(self.source / "status.json", {"status": "complete", "intended_per_model": 3})
        config = {**old_config, "run_dir": str(self.output), "reuse_run_dir": str(self.source),
                  "reuse_methods": ["baseline"], "preference_checkpoint": str(self.root / "new.ckpt")}
        self.protocol = build_protocol(config, self.identities)

    def test_partial_reuse_allows_new_dpo_and_never_mutates_source(self):
        original = {str(path): sha256_file(path) for path in self.source.rglob("*") if path.is_file()}
        reused = reuse_completed_run(self.protocol)
        self.assertEqual(reused, {"baseline": set(self.identities), "dpo_5000": set()})
        self.assertFalse((self.output / "raw_predictions/dpo_5000").exists())
        for image_id in self.identities:
            self.assertTrue((self.output / "raw_predictions/baseline" / image_id).is_symlink())
        failed = read_json(self.output / "raw_predictions/baseline/failed/inference.json")
        self.assertEqual(failed["status"], "generation_error")
        self.assertEqual(reuse_completed_run(self.protocol), reused)
        self.assertEqual(original, {str(path): sha256_file(path) for path in self.source.rglob("*") if path.is_file()})

    def test_legacy_default_reuses_both_and_checks_both_checkpoints(self):
        self.protocol["config"].pop("reuse_methods")
        with self.assertRaisesRegex(ValueError, "preference_checkpoint"):
            reuse_completed_run(self.protocol)
        self.protocol["checkpoint_sha256"]["preference_checkpoint"] = self.old_protocol["checkpoint_sha256"]["preference_checkpoint"]
        reused = reuse_completed_run(self.protocol)
        self.assertEqual(reused, {method: set(self.identities) for method in METHODS})
        self.assertEqual(generation_assignments([0, 1], self.identities, reused), [])
        self.assertEqual(read_json(self.output / "metadata/reuse.json")["reused_per_model"], 3)

    def test_original_checkpoint_is_required_even_when_only_dpo_reused(self):
        self.protocol["config"]["reuse_methods"] = ["dpo_5000"]
        self.protocol["checkpoint_sha256"]["original_checkpoint"] = "different"
        with self.assertRaisesRegex(ValueError, "original_checkpoint"):
            validate_reuse_protocol(self.protocol)

    def test_protocol_source_and_split_changes_are_rejected(self):
        original = deepcopy(self.protocol)
        self.protocol["source_sha256"]["src/models/diffusion/legacy.py"] = "changed"
        with self.assertRaisesRegex(ValueError, "Generation or evaluation source changed"):
            validate_reuse_protocol(self.protocol)
        self.protocol = original
        self.split.write_text("two\none\nfailed\n")
        with self.assertRaisesRegex(ValueError, "frozen validation list changed"):
            validate_reuse_protocol(self.protocol)
        self.split.write_text("\n".join(self.identities) + "\n")
        old = read_json(self.source / "metadata/protocol.json")
        old["sampling"] = "changed"
        atomic_json(self.source / "metadata/protocol.json", old)
        with self.assertRaisesRegex(ValueError, "Protocol fingerprint mismatch"):
            validate_reuse_protocol(self.protocol)

    def test_recorded_inputs_outputs_and_step_must_match(self):
        for path in (self.root / "images/one/imgs.npz",
                     self.source / "raw_predictions/baseline/one/latent.npy",
                     self.source / "evaluations/baseline/one/post/c00/recon_brep.step"):
            original = path.read_bytes()
            path.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                reuse_completed_run(self.protocol)
            path.write_bytes(original)
        self.assertFalse((self.output / "raw_predictions").exists())

    def test_resume_rejects_modified_metric_record(self):
        reuse_completed_run(self.protocol)
        path = self.source / "evaluations/baseline/one/metrics.json"
        row = read_json(path)
        row["raw"]["face_f1"] = .5
        atomic_json(path, row)
        with self.assertRaisesRegex(ValueError, "Previously reused source records"):
            reuse_completed_run(self.protocol)

    def test_missing_hashes_unfinished_records_and_wrong_seed_rejected(self):
        path = self.source / "raw_predictions/baseline/one/inference.json"
        original = read_json(path)
        for change in ({"output_sha256": {}}, {"status": "running"}, {"seed": 1}):
            atomic_json(path, {**original, **change})
            with self.assertRaises(ValueError):
                reuse_completed_run(self.protocol)
        atomic_json(path, original)

    def test_reuse_accepts_chained_read_only_symlinks(self):
        earlier = self.root / "earlier"
        for stage in ("raw_predictions", "evaluations"):
            source = self.source / stage / "baseline/one"
            target = earlier / stage / "baseline/one"
            target.parent.mkdir(parents=True)
            source.rename(target)
            source.symlink_to(target, target_is_directory=True)
        reused = reuse_completed_run(self.protocol)
        self.assertEqual(reused["baseline"], set(self.identities))
        self.assertEqual((self.output / "raw_predictions/baseline/one").resolve(), earlier / "raw_predictions/baseline/one")

    def test_assignments_use_all_gpus_only_for_new_method(self):
        ids = [str(index) for index in range(17)]
        reused = {"baseline": set(ids), "dpo_5000": set()}
        jobs = generation_assignments(list(range(8)), ids, reused)
        self.assertEqual({job[0] for job in jobs}, {"dpo_5000"})
        self.assertEqual({job[1] for job in jobs}, set(range(8)))
        self.assertCountEqual([image_id for _, _, shard in jobs for image_id in shard], ids)
        old_jobs = generation_assignments([0, 1, 2, 3], ids, {method: set() for method in METHODS})
        self.assertEqual([(method, gpu) for method, gpu, _ in old_jobs],
                         [("baseline", 0), ("baseline", 2), ("dpo_5000", 1), ("dpo_5000", 3)])
        with self.assertRaises(ValueError):
            generation_assignments([0], ids, {method: set() for method in METHODS})

    def test_cpu_jobs_skip_only_reused_method_and_keep_failed_terminal(self):
        config = {**self.protocol["config"], "run_dir": str(self.source)}
        reused = {"baseline": set(self.identities), "dpo_5000": set()}
        self.assertEqual(evaluation_jobs(config, self.identities, reused),
                         [("dpo_5000", "one"), ("dpo_5000", "two")])
        self.assertEqual(evaluation_jobs(config, self.identities,
                                         {method: set(self.identities) for method in METHODS}), [])

    def test_output_lock_rejects_concurrent_coordinator(self):
        with comparison_lock(self.output):
            with self.assertRaisesRegex(RuntimeError, "already in use"):
                with comparison_lock(self.output):
                    self.fail("Second coordinator acquired lock")

    def test_completed_stage_verification_is_read_only_and_rejects_unfinished(self):
        original = {str(path): sha256_file(path) for path in self.source.rglob("*") if path.is_file()}
        artifacts = verify_completed_comparison(self.old_protocol)
        self.assertIn(str(self.source / "raw_predictions/baseline/failed/inference.json"), artifacts)
        self.assertIn(str(self.source / "evaluations/dpo_5000/one/post/c00/recon_brep.step"), artifacts)
        self.assertEqual(original, {str(path): sha256_file(path) for path in self.source.rglob("*") if path.is_file()})
        atomic_json(self.source / "status.json", {"status": "evaluating"})
        with self.assertRaisesRegex(ValueError, "not complete"):
            verify_completed_comparison(self.old_protocol)


if __name__ == "__main__":
    unittest.main()
