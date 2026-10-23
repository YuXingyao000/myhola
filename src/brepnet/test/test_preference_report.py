"""Denominators, missing measurements and paired CAD summaries; CPU only."""

import json
from pathlib import Path
import tempfile
import unittest

from src.brepnet.eval.preference_report import (
    METHODS, load_rows, paired_statistics, plot_comparison, summarize, write_csv,
)


class PreferenceReportTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="hola-preference-report-")
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.ids = ["one", "two", "three", "four"]
        (self.root / "validation100.txt").write_text("\n".join(self.ids) + "\n")
        for method in METHODS:
            for index, image_id in enumerate(self.ids):
                raw_dir = self.root / "raw_predictions" / method / image_id
                raw_dir.mkdir(parents=True)
                inference = {"status": "generation_error" if index == 3 else "ok", "seed": index, "seconds": 1.}
                (raw_dir / "inference.json").write_text(json.dumps(inference))
                if index == 3:
                    continue
                for filename in ("data.npz", "latent.npy"):
                    (raw_dir / filename).touch()
                evaluation_dir = self.root / "evaluations" / method / image_id
                evaluation_dir.mkdir(parents=True)
                validity = "valid" if index < 2 else "invalid" if method == "baseline" else "unknown"
                raw = {"face_cd_sq": .1, "edge_cd_sq": .2, "face_f1": .8,
                       "pred_face_count": 7, "gt_face_count": 9}
                repeat_values = ([.1, .2, .3] if index == 0 else [.4, .5, .6]) if method == "baseline" else [.3, .6, .9]
                post = [{"repeat": repeat, "face_cd_sq": value, "edge_cd_sq": value * 2, "face_f1": .8}
                        for repeat, value in enumerate(repeat_values)] if index < 2 else []
                status = "ok"
                if method == "dpo_5000" and index == 1:
                    post = post[:2]
                    status = "evaluation_error"
                if validity == "unknown":
                    status = "timeout"
                evaluation = {"status": status, "validity": validity, "raw": raw,
                              "post": post, "wall_seconds": 2., "error": "", "native": {"step_readable": validity != "unknown"}}
                (evaluation_dir / "metrics.json").write_text(json.dumps(evaluation))
                if validity != "unknown":
                    step_dir = evaluation_dir / "post/c00"
                    step_dir.mkdir(parents=True)
                    (step_dir / "recon_brep.step").touch()

    def load(self):
        return load_rows(self.root, expected_count=4)

    def test_full_denominator_unknown_and_partial_post_are_preserved(self):
        ids, rows = self.load()
        result = summarize(ids, rows)
        baseline = result["methods"]["baseline"]
        dpo = result["methods"]["dpo_5000"]
        self.assertEqual(result["intended_count"], 4)
        self.assertEqual(baseline["rates_over_intended"]["validity_valid"], .5)
        self.assertEqual(dpo["counts"]["validity_valid"], 2)
        self.assertEqual(dpo["counts"]["validity_unknown"], 2)
        self.assertEqual(dpo["counts"]["partial_post"], 1)
        self.assertEqual(dpo["counts"]["valid_without_complete_post"], 1)
        self.assertEqual(dpo["counts"]["inference_errors"], 1)
        self.assertEqual(dpo["counts"]["timeouts"], 1)
        self.assertEqual(rows["dpo_5000"][1]["post_face_cd_sq"], None)
        self.assertEqual(rows["dpo_5000"][2]["raw_face_cd_sq"], .1)
        self.assertEqual(rows["dpo_5000"][3]["raw_face_cd_sq"], None)
        self.assertEqual(result["validity_transitions"]["invalid_to_unknown"]["ids"], ["three"])
        self.assertEqual(sum(r["n"] for r in result["validity_transitions"].values()), 4)

    def test_post_averages_per_cad_and_common_subset(self):
        ids, rows = self.load()
        result = summarize(ids, rows)
        metric = result["metrics"]["post_face_cd_sq"]
        self.assertEqual(metric["own"]["baseline"]["n"], 2)
        self.assertAlmostEqual(metric["own"]["baseline"]["mean"], .35)
        self.assertEqual(metric["paired_common"]["ids"], ["one"])
        self.assertAlmostEqual(metric["paired_common"]["baseline_mean"], .2)
        self.assertAlmostEqual(metric["paired_common"]["dpo_5000_mean"], .6)
        self.assertAlmostEqual(metric["paired_common"]["dpo_minus_baseline"], .4)
        self.assertEqual(result["metrics"]["raw_face_count_abs_error"]["own"]["baseline"]["mean"], 2.)
        self.assertEqual(result["validity_paired"]["n"], 4)

    def test_empty_common_geometry_has_null_means_and_no_bootstrap(self):
        result = paired_statistics({"a": 1.}, {"b": 2.})
        self.assertEqual(result["n"], 0)
        self.assertIsNone(result["dpo_minus_baseline"])
        self.assertIsNone(result["ci95"])

    def test_paired_bootstrap_preserves_pairing_and_reproducibility(self):
        left = {"a": 0., "b": 100.}
        right = {"a": 1., "b": 101.}
        result = paired_statistics(left, right)
        self.assertEqual(result["dpo_minus_baseline"], 1.)
        self.assertEqual(result["ci95"], [1., 1.])
        self.assertEqual(result, paired_statistics(left, right))

    def test_missing_or_running_terminal_records_are_rejected(self):
        path = self.root / "evaluations/baseline/one/metrics.json"
        original = path.read_text()
        path.unlink()
        with self.assertRaisesRegex(ValueError, "Missing evaluation terminal"):
            self.load()
        value = json.loads(original)
        value["status"] = "running"
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "Unfinished evaluation"):
            self.load()
        path.write_text(original)
        (self.root / "raw_predictions/baseline/one/inference.json").unlink()
        with self.assertRaisesRegex(ValueError, "Missing inference terminal"):
            self.load()

    def test_changed_seed_and_incomplete_list_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "100 distinct"):
            load_rows(self.root)
        path = self.root / "raw_predictions/dpo_5000/one/inference.json"
        value = json.loads(path.read_text())
        value["seed"] = 999
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "unequal paired"):
            self.load()

    def test_report_artifacts_render_without_gpu_or_geometry_dependencies(self):
        ids, rows = self.load()
        result = summarize(ids, rows)
        write_csv(self.root / "per_object.csv", ids, rows)
        plot_comparison(result, self.root)
        for name in ("per_object.csv", "comparison.png", "comparison.svg"):
            self.assertGreater((self.root / name).stat().st_size, 0)
        self.assertEqual(len((self.root / "per_object.csv").read_text().splitlines()), 5)


if __name__ == "__main__":
    unittest.main()
