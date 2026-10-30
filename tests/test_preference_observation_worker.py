"""CPU fixtures for immutable candidate inputs, shared rendering and resume."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.preference.observation_worker import evaluate_image, SourceIntegrityError
from src.common.io import atomic_json, fingerprint, read_json, sha256_file


class PreferenceObservationWorkerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        self.output = self.root / "derived"
        self.image_id = "object_a"
        self.gt = self.root / "raw" / self.image_id
        self.image = self.root / "condition" / self.image_id / "imgs.npz"
        self.gt.mkdir(parents=True)
        self.image.parent.mkdir(parents=True)
        np.savez(self.gt / "data.npz", face=np.zeros((2, 2, 2, 3)))
        (self.gt / "normalized_shape.step").write_text("fake GT STEP")
        rgb = np.full((1, 16, 16, 3), 255, dtype=np.uint8)
        rgb[:, 4:12, 4:12] = 100
        np.savez(self.image, svr_imgs=rgb)
        source_root = self.source / "candidates/validation" / self.image_id
        self.candidates = [source_root / f"c{index:02d}" for index in range(3)]
        atomic_json(source_root / "sources.json", {
            str(path): sha256_file(path)
            for path in (self.image, self.gt / "data.npz", self.gt / "normalized_shape.step")
        })
        cache = self.source / "gt_cache" / f"{self.image_id}_0"
        cache.mkdir(parents=True)
        np.save(cache / "features.npy", np.zeros((2, 64), np.float32))
        atomic_json(cache / "source.json", {
            "gt_npz_sha256": sha256_file(self.gt / "data.npz"),
            "output_sha256": {"features.npy": sha256_file(cache / "features.npy")},
            "precision": "float32", "layout": "mean32,std32",
        })
        for index, directory in enumerate(self.candidates[:2]):
            directory.mkdir(parents=True)
            np.save(directory / "latent.npy", np.zeros((30, 32), np.float32))
            np.savez(directory / "data.npz", face=np.zeros((2, 2, 2, 3)))
            np.savez(directory / "mappings.npz", index=np.arange(2))
            identity = {"image_id": self.image_id, "candidate_id": directory.name}
            inference = {**identity, "status": "ok", "output_sha256": {
                name: sha256_file(directory / name) for name in ("latent.npy", "data.npz", "mappings.npz")
            }}
            atomic_json(directory / "inference.json", inference)
            evaluation = {
                **identity, "version": "svr_preference_metrics_v1",
                "raw_path": str(directory / "data.npz"),
                "gt_npz": str(self.gt / "data.npz"),
                "gt_step": str(self.gt / "normalized_shape.step"),
                "input_sha256": {
                    "raw_path": sha256_file(directory / "data.npz"),
                    "gt_npz": sha256_file(self.gt / "data.npz"),
                    "gt_step": sha256_file(self.gt / "normalized_shape.step"),
                },
            }
            evaluation["protocol_fingerprint"] = fingerprint(evaluation)
            atomic_json(directory / "evaluation/protocol.json", evaluation)
            step = directory / "evaluation/post/c00/recon_brep.step"
            step.parent.mkdir(parents=True)
            step.write_text(f"fake candidate {index}")
            atomic_json(directory / "evaluation/metrics.json", {
                **identity, "protocol_fingerprint": evaluation["protocol_fingerprint"],
                "status": "ok", "validity": "valid" if index == 0 else "invalid",
                "raw": {"face_cd_sq": .1, "edge_cd_sq": .2, "face_f1": .8},
                "step_path": str(step), "step_sha256": sha256_file(step),
            })
        self.protocol = {
            "fingerprint": "derived-fingerprint", "candidates": 3,
            "attempt_id": "first-attempt",
            "raw_data_root": str(self.root / "raw"),
            "condition_data_root": str(self.root / "condition"),
            "observation": {"resolution": 16, "view_id": 0, "image_key": "svr_imgs",
                            "mesh_deflection": .002, "mask_threshold": 245},
        }
        self.depth = np.full((16, 16), np.nan)
        self.depth[4:12, 4:12] = 3.

    def run_worker(self):
        return evaluate_image(self.source, self.output, "validation", self.image_id, self.protocol)

    def test_shared_gt_resume_and_fixed_denominator(self):
        with patch("src.preference.observation_worker.metrics.render_step", return_value=self.depth) as render:
            result = self.run_worker()
            self.assertEqual(render.call_count, 3)  # One GT and two saved candidates.
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["attempt_id"], "first-attempt")
        self.assertEqual(len(result["candidates"]), 3)
        self.assertEqual([row["validity"] for row in result["candidates"]], ["valid", "invalid", "unknown"])
        self.assertEqual(result["candidates"][0]["observation"]["silhouette_iou"], 1.)
        self.assertEqual(result["candidates"][2]["observation"]["status"], "error")
        self.protocol["attempt_id"] = "resumed-attempt"
        with patch("src.preference.observation_worker.metrics.render_step") as render, \
                patch("src.preference.observation_worker.load_image_view") as load:
            resumed = self.run_worker()
            render.assert_not_called()
            load.assert_not_called()
        self.assertEqual(result["candidates"], resumed["candidates"])
        self.assertEqual(resumed["attempt_id"], "resumed-attempt")
        checkpoint = self.output / "cases/validation" / self.image_id / "metrics.json"
        self.assertEqual(read_json(checkpoint)["status"], "complete")

    def test_source_bytes_change_is_fatal_even_on_resume(self):
        with patch("src.preference.observation_worker.metrics.render_step", return_value=self.depth):
            self.run_worker()
        (self.candidates[0] / "data.npz").write_bytes(b"changed")
        with self.assertRaisesRegex(SourceIntegrityError, "SHA256 mismatch"):
            self.run_worker()

    def test_source_metric_edit_rejects_cached_observation(self):
        with patch("src.preference.observation_worker.metrics.render_step", return_value=self.depth):
            self.run_worker()
        path = self.candidates[0] / "evaluation/metrics.json"
        original = read_json(path)
        original["raw"]["face_f1"] = .9
        atomic_json(path, original)
        with self.assertRaisesRegex(SourceIntegrityError, "Cached image protocol or source changed"):
            self.run_worker()

    def test_evaluation_fingerprint_is_checked(self):
        path = self.candidates[0] / "evaluation/protocol.json"
        original = read_json(path)
        original["input_sha256"]["raw_path"] = "wrong"
        atomic_json(path, original)
        with self.assertRaisesRegex(SourceIntegrityError, "Evaluation protocol or inputs mismatch"):
            self.run_worker()

    def test_bad_latent_and_failed_render_are_unavailable_not_invalid(self):
        directory = self.candidates[0]
        np.save(directory / "latent.npy", np.zeros((2, 32), np.float32))
        inference = read_json(directory / "inference.json")
        inference["output_sha256"]["latent.npy"] = sha256_file(directory / "latent.npy")
        atomic_json(directory / "inference.json", inference)
        with patch("src.preference.observation_worker.metrics.render_step",
                   side_effect=[self.depth, ValueError("untriangulable shape")]):
            result = self.run_worker()
        first, second = result["candidates"][:2]
        self.assertEqual(first["latent_status"], "invalid")
        self.assertEqual(first["validity"], "valid")
        self.assertIsNone(first["latent_path"])
        self.assertEqual(first["observation"]["stage"], "latent")
        self.assertEqual(second["validity"], "invalid")
        self.assertEqual(second["observation"]["status"], "error")
        with patch("src.preference.observation_worker.metrics.render_step") as render:
            self.run_worker()
            render.assert_not_called()

    def test_gt_failure_keeps_all_candidates(self):
        with patch("src.preference.observation_worker.metrics.render_step", side_effect=ValueError("bad GT")):
            result = self.run_worker()
        self.assertEqual(len(result["candidates"]), 3)
        self.assertTrue(all(row["observation"]["status"] == "error" for row in result["candidates"]))
        self.assertEqual(result["candidates"][0]["observation"]["stage"], "gt_or_input")


if __name__ == "__main__":
    unittest.main()
