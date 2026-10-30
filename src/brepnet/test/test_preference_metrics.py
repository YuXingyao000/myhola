"""Metric meaning and bounded OCC execution; no model checkpoints required."""

import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np

from src.brepnet.eval.preference import (
    _confirmed_no_solid, evaluate_candidate, face_detection, metric_seed, score_geometry,
    score_post, score_raw,
)
from tools import vae_compare_post as post


def slow_worker(job, status_path, log_path):
    os.setsid()
    post.atomic_json(status_path, {
        "image_id": job["image_id"], "candidate_id": job["candidate_id"],
        "protocol_fingerprint": job["protocol_fingerprint"], "status": "running",
        "validity": "invalid", "raw": {"face_cd_sq": 1.}, "post": []})
    time.sleep(10)


class PreferenceMetricsTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="hola-preference-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def test_squared_cd_and_primitive_f1_have_distinct_meanings(self):
        face = np.array([[[0., 0., 0.], [0., 1., 0.]]])
        self.assertEqual(face_detection(face, face)["face_f1"], 1.)
        duplicated = np.concatenate([face, face])
        values = score_geometry(duplicated, face, face, face)
        self.assertEqual(values["face_cd_sq"], 0.)
        self.assertEqual(values["face_f1"], 2 / 3)
        self.assertEqual(face_detection(face + 10, face)["face_f1"], 0.)

    def test_raw_identity_and_missing_geometry_do_not_use_origin_fallback(self):
        face = np.zeros((1, 2, 2, 3))
        edge = np.zeros((1, 2, 3))
        raw, gt = self.root / "raw.npz", self.root / "gt.npz"
        np.savez(raw, pred_face=face, pred_edge=edge)
        np.savez(gt, sample_points_faces=face, sample_points_lines=edge)
        values = score_raw(raw, gt)
        self.assertEqual((values["face_cd_sq"], values["edge_cd_sq"], values["face_f1"]),
                         (0., 0., 1.))
        np.savez(raw, pred_face=face, pred_edge=np.zeros((0, 2, 3)))
        with self.assertRaisesRegex(ValueError, "nonempty"):
            score_raw(raw, gt)

    def test_missing_step_only_labels_completed_geometric_rejection(self):
        self.assertTrue(_confirmed_no_solid({
            "status": "no_accepted_solid",
            "paths": [{"status": "native_rejected"}, {"status": "incomplete_faces"}]}))
        for diagnostics in ({}, {"status": "budget_exhausted"}, {
                "status": "no_accepted_solid", "paths": [{"status": "error"}]}):
            self.assertFalse(_confirmed_no_solid(diagnostics))

    def test_hard_timeout_is_unknown_and_resume_does_not_retry(self):
        kwargs = dict(raw_path=self.root / "raw.npz", gt_npz=self.root / "gt.npz",
                      gt_step=self.root / "gt.step", output_dir=self.root / "evaluation",
                      image_id="part", timeout_seconds=.2, start_method="fork")
        start = time.monotonic()
        with patch("src.brepnet.eval.preference._candidate_process", slow_worker):
            row = evaluate_candidate(**kwargs)
        self.assertLess(time.monotonic() - start, 3.)
        self.assertEqual((row["status"], row["validity"]), ("timeout", "unknown"))
        self.assertEqual(evaluate_candidate(**kwargs), row)
        with self.assertRaisesRegex(ValueError, "changed"):
            evaluate_candidate(**{**kwargs, "timeout_seconds": .3})

    def test_native_step_repeats_reuse_gt_and_recover_face_groups(self):
        from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
        from OCC.Extend.DataExchange import write_step_file
        shape = BRepPrimAPI_MakeBox(1., 1., 1.).Shape()
        step = self.root / "box.step"
        write_step_file(shape, str(step))
        cache = self.root / "cache"
        a = score_post(shape, step, cache, "box", "a", 0, 42)
        b = score_post(shape, step, cache, "box", "b", 0, 42)
        repeat = score_post(shape, step, cache, "box", "a", 1, 42)
        self.assertEqual(a["gt_seed"], b["gt_seed"])
        self.assertNotEqual(a["pred_seed"], b["pred_seed"])
        self.assertNotEqual(a["gt_seed"], repeat["gt_seed"])
        self.assertEqual(len(list(cache.glob("*.npz"))), 2)
        self.assertEqual((a["pred_face_count"], a["gt_face_count"], a["face_f1"]), (6, 6, 1.))
        self.assertLess(a["face_cd_sq"], .01)
        self.assertEqual(a["edge_cd_sq"], 0.)
        self.assertEqual(a, score_post(shape, step, cache, "box", "a", 0, 42))


if __name__ == "__main__":
    unittest.main()
