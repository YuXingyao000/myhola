"""Fixed denominators, deterministic native geometry and hard job isolation."""
from pathlib import Path
import os
import tempfile
import time
import unittest

import numpy as np

from tools.vae_compare_post import (
    SAMPLING, atomic_json, execute_job, load_gt_samples, native_validity,
    read_json, read_step_at_precision, run_isolated_jobs, sample_native_geometry,
    squared_chamfer, stable_seed, summarize,
)


def slow_worker(job, status_path, log_path):
    os.setsid()
    atomic_json(status_path, {"status": "running", "stage": "deliberate_sleep"})
    time.sleep(10)


def crashed_worker(job, status_path, log_path):
    os.setsid()
    os._exit(17)


class VaeComparePostTests(unittest.TestCase):
    def setUp(self):
        base = Path(__file__).resolve().parents[3] / "checkpoints/joint_boundary/test_tmp"
        base.mkdir(parents=True, exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(prefix="compare-post-", dir=base)
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def job(self):
        return {"sample_id": "sample", "method": "joint", "mode": "fit_only",
                "protocol_fingerprint": "test", "raw_sha256": None,
                "raw_root": str(self.root / "raw"), "post_root": str(self.root / "post"),
                "data_root": str(self.root / "gt"), "cache_root": str(self.root / "cache"),
                "max_optimize_iter": 200}

    def test_squared_chamfer_self_translation_and_missing(self):
        points = np.array([[0., 0., 0.], [1., 0., 0.]])
        self.assertEqual(squared_chamfer(points, points)["cd"], 0)
        result = squared_chamfer(np.array([[0., 0., 0.]]), np.array([[0., 0., 3.]]))
        self.assertEqual(result, {"accuracy": 9., "completeness": 9., "cd": 18.})
        for invalid in (np.empty((0, 3)), np.array([[np.nan, 0, 0]])):
            with self.assertRaises(ValueError):
                squared_chamfer(invalid, points)

    def test_missing_prediction_is_a_record_not_origin_geometry(self):
        output = self.root / "row.json"
        result = execute_job(self.job(), output)
        self.assertEqual(result["status"], "missing_raw")
        self.assertFalse(result["step_exists"])
        self.assertNotIn("face_cd", result)
        self.assertEqual(read_json(output)["sample_id"], "sample")

    def test_denominators_and_paired_common_subset(self):
        metrics = {key: 1. for key in ("face_cd", "face_accuracy", "face_completeness", "edge_cd", "edge_accuracy", "edge_completeness")}
        rows = [
            {"sample_id": "a", "method": "legacy", "status": "ok", "geometry_status": "ok", "native_valid_solid": True, **metrics},
            {"sample_id": "a", "method": "joint", "status": "ok", "geometry_status": "ok", "native_valid_solid": True, **{key: value / 2 for key, value in metrics.items()}},
            {"sample_id": "b", "method": "legacy", "status": "timeout"},
            {"sample_id": "b", "method": "joint", "status": "missing_raw"},
        ]
        result = summarize(rows, 2, 3)
        self.assertEqual(result["paired_common_geometry"]["count"], 1)
        self.assertEqual(result["methods"]["legacy"]["rates_over_intended"]["native_valid_solid"], .5)
        self.assertEqual(result["methods"]["joint"]["rates_over_original_test"]["native_valid_solid"], 1 / 3)
        self.assertEqual(result["paired_common_geometry"]["metrics"]["face_cd"]["joint_minus_legacy"]["mean"], -.5)

    def test_worker_timeout_and_crash_preserve_id(self):
        for name, target, expected in (("slow", slow_worker, "timeout"), ("crash", crashed_worker, "crash")):
            metadata = self.root / name
            start = time.monotonic()
            run_isolated_jobs([self.job()], metadata, workers=1, timeout=.2, process_target=target)
            self.assertLess(time.monotonic() - start, 3)
            row = read_json(metadata / "samples/joint/sample.json")
            self.assertEqual(row["status"], expected)
            self.assertEqual(row["sample_id"], "sample")
            if expected == "crash":
                self.assertEqual(row["worker_exit_code"], 17)

    def test_native_geometry_is_deterministic_and_gt_cache_is_atomic(self):
        from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
        from OCC.Core.Interface import Interface_Static
        from OCC.Extend.DataExchange import write_step_file
        path = self.root / "box.step"
        write_step_file(BRepPrimAPI_MakeBox(.4, .5, .6).Shape(), str(path))
        Interface_Static.SetIVal("read.precision.mode", 1)
        shape = read_step_at_precision(path)
        self.assertEqual(Interface_Static.IVal("read.precision.mode"), 1)
        self.assertTrue(native_validity(shape)["native_valid_solid"])
        seed = stable_seed("box", "geometry")
        first = sample_native_geometry(shape, seed)
        second = sample_native_geometry(read_step_at_precision(path), seed)
        np.testing.assert_array_equal(first["face_points"], second["face_points"])
        np.testing.assert_array_equal(first["edge_points"], second["edge_points"])
        self.assertEqual(len(first["face_points"]), 6 * SAMPLING["face_points_per_face"])
        self.assertEqual(squared_chamfer(first["face_points"], second["face_points"])["cd"], 0.)
        cached1 = load_gt_samples(path, self.root / "cache", "box")
        cached2 = load_gt_samples(path, self.root / "cache", "box")
        np.testing.assert_array_equal(cached1["face_points"], cached2["face_points"])
        self.assertEqual(len(list((self.root / "cache").glob("*.npz"))), 1)
        self.assertFalse(list((self.root / "cache").glob("*.tmp")))


if __name__ == "__main__":
    unittest.main()
