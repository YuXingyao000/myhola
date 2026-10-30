"""Geometry repetition and within-condition diversity contracts; CPU only."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.eval.metrics.conditional_diversity import (
    evaluate_conditional_diversity,
    geometry_fingerprints,
    summarize_fingerprints,
)


class ConditionalDiversityTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="hola-conditional-diversity-")
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def save(self, name, *, x=0., faces=None, edges=None):
        path = self.root / f"{name}.npz"
        if faces is None:
            faces = np.array([[[[x, 0., 0.]]]])
        if edges is None:
            edges = np.array([[[x, 0., 0.]]])
        np.savez(path, pred_face=faces, pred_edge=edges)
        return path

    def record(self, image_id, candidate_id, path, *, validity="valid", method="baseline"):
        return {"method": method, "image_id": image_id, "candidate_id": candidate_id,
                "raw_path": str(path), "validity": validity}

    def test_fingerprints_ignore_point_order_primitive_order_and_duplicate_points(self):
        face = np.arange(24, dtype=np.float64).reshape(2, 2, 2, 3) / 30
        edge = np.arange(18, dtype=np.float64).reshape(2, 3, 3) / 30
        original = self.save("original", faces=face, edges=edge)
        permuted = self.save("permuted", faces=face[::-1, ::-1].swapaxes(1, 2), edges=edge[::-1, ::-1])
        repeated = self.save("repeated", faces=np.concatenate([face, face]), edges=np.concatenate([edge, edge]))
        self.assertEqual(geometry_fingerprints(original), geometry_fingerprints(permuted))
        self.assertEqual(geometry_fingerprints(original), geometry_fingerprints(repeated))

    def test_common_frame_keeps_translation_and_quantization_sensitivity(self):
        original = geometry_fingerprints(self.save("original"))
        nearby = geometry_fingerprints(self.save("nearby", x=.0004))
        translated = geometry_fingerprints(self.save("translated", x=.25))
        self.assertNotEqual(original["0.0001"], nearby["0.0001"])
        self.assertEqual(original["0.001"], nearby["0.001"])
        self.assertEqual(original["0.01"], nearby["0.01"])
        for step in original:
            self.assertNotEqual(original[step], translated[step])

    def test_fingerprint_failures_are_explicit(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            geometry_fingerprints(self.save("good"), (0.,))
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            geometry_fingerprints(self.save("nan", x=float("nan")))
        with self.assertRaisesRegex(ValueError, "nonempty"):
            geometry_fingerprints(self.save("empty", edges=np.empty((0, 16, 3))))

    def test_fingerprint_denominators_and_valid_subset(self):
        fingerprint = geometry_fingerprints(self.save("one"))
        records = [
            {**self.record("a", "c00", self.root / "one.npz"), "raw_status": "ok", "raw_fingerprints": fingerprint},
            {**self.record("b", "c00", self.root / "one.npz", validity="invalid"), "raw_status": "ok", "raw_fingerprints": fingerprint},
            {**self.record("c", "c00", self.root / "missing.npz", validity="unknown"), "raw_status": "missing_file", "raw_fingerprints": {}},
        ]
        result = summarize_fingerprints(records)["methods"]["baseline"]
        overall = result["all"]["quantization"]["0.001"]
        valid = result["valid"]["quantization"]["0.001"]
        self.assertEqual(result["intended_count"], 3)
        self.assertEqual(overall["measured_count"], 2)
        self.assertEqual(overall["missing_count"], 1)
        self.assertEqual(overall["unique_fingerprint_ratio"], .5)
        self.assertEqual(overall["repeated_member_ratio"], 1.)
        self.assertEqual(overall["duplicate_pair_ratio"], 1.)
        self.assertEqual(overall["unmeasured"][0]["image_id"], "c")
        self.assertEqual(valid["unique_fingerprint_ratio"], 1.)
        self.assertEqual(valid["repeated_member_ratio"], 0.)
        self.assertIsNone(valid["duplicate_pair_ratio"])

    def test_k1_is_not_available_and_images_methods_never_mix(self):
        path = self.save("one")
        records = [self.record("a", "c00", path), self.record("b", "c00", path),
                   self.record("a", "c00", path, method="dpo")]
        result = evaluate_conditional_diversity(records, workers=1)
        self.assertEqual(len(result["group_rows"]), 3)
        self.assertEqual(result["pair_rows"], [])
        for group in result["group_rows"]:
            self.assertEqual(group["status"], "not_available")
            self.assertEqual(group["reason"], "fewer_than_two_intended_candidates")
            self.assertIsNone(group["all"]["face"]["mean_pairwise_cd_sq"])
        self.assertIsNone(result["methods"]["baseline"]["all"]["face"]["mean_pairwise_cd_sq"])

    def test_identical_and_translated_k2_have_known_squared_chamfer(self):
        original, shifted = self.save("one"), self.save("two", x=1.)
        records = [self.record("same", "c00", original), self.record("same", "c01", original),
                   self.record("different", "c00", original), self.record("different", "c01", shifted)]
        result = evaluate_conditional_diversity(records, workers=1)
        groups = {row["image_id"]: row for row in result["group_rows"]}
        self.assertEqual(groups["same"]["all"]["face"]["mean_pairwise_cd_sq"], 0.)
        self.assertEqual(groups["different"]["all"]["face"]["mean_pairwise_cd_sq"], 2.)
        self.assertEqual(result["methods"]["baseline"]["all"]["face"]["mean_pairwise_cd_sq"], 1.)

    def test_images_have_equal_weight_even_when_candidate_counts_differ(self):
        points = [self.save(f"x{i}", x=float(i)) for i in range(5)]
        records = [self.record("k2", "c00", points[0]), self.record("k2", "c01", points[1])]
        records += [self.record("k3", f"c{i:02d}", points[x]) for i, x in enumerate([0, 2, 4])]
        result = evaluate_conditional_diversity(records, workers=1)
        # First image: 2. Second image: (8 + 32 + 8) / 3 = 16.
        self.assertEqual(result["methods"]["baseline"]["all"]["face"]["mean_pairwise_cd_sq"], 9.)
        self.assertEqual(result["methods"]["baseline"]["all"]["face"]["measured_pair_count"], 4)

    def test_invalid_and_missing_candidates_preserve_pair_budget(self):
        original, shifted = self.save("one"), self.save("two", x=1.)
        records = [self.record("a", "c00", original),
                   self.record("a", "c01", shifted, validity="invalid"),
                   self.record("a", "c02", self.root / "missing.npz", validity="unknown")]
        result = evaluate_conditional_diversity(records, workers=1)
        group = result["group_rows"][0]
        self.assertEqual(group["intended_candidate_count"], 3)
        self.assertEqual(group["status"], "partial")
        self.assertEqual(group["all"]["face"]["intended_pair_count"], 3)
        self.assertEqual(group["all"]["face"]["measured_pair_count"], 1)
        self.assertEqual(group["all"]["face"]["mean_pairwise_cd_sq"], 2.)
        self.assertEqual(group["valid"]["face"]["intended_pair_count"], 0)
        self.assertIsNone(group["valid"]["face"]["mean_pairwise_cd_sq"])
        self.assertEqual(len(result["pair_rows"]), 3)
        self.assertEqual(result["candidate_rows"][2]["face_status"], "missing_file")
        summary = result["methods"]["baseline"]["all"]["face"]
        self.assertEqual(summary["partial_image_count"], 1)
        self.assertIsNone(summary["complete_images_mean_pairwise_cd_sq"])

    def test_empty_curves_preserve_available_face_distance(self):
        valid = self.save("valid")
        empty = self.save("empty", x=1., edges=np.empty((0, 16, 3)))
        result = evaluate_conditional_diversity([
            self.record("a", "c00", valid), self.record("a", "c01", empty),
        ], workers=1)
        pair = result["pair_rows"][0]
        self.assertEqual(pair["face_cd_sq"], 2.)
        self.assertIsNone(pair["edge_cd_sq"])
        self.assertEqual(result["candidate_rows"][1]["edge_status"], "geometry_error")

    def test_parallel_output_matches_serial_and_duplicate_candidates_are_rejected(self):
        path = self.save("one")
        records = [self.record("a", "c00", path), self.record("a", "c01", path)]
        self.assertEqual(evaluate_conditional_diversity(records, workers=1),
                         evaluate_conditional_diversity(records, workers=2))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            evaluate_conditional_diversity([records[0], records[0]], workers=1)


if __name__ == "__main__":
    unittest.main()
