"""Audit population, paired coverage, and immutable reuse of saved evidence."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from src.eval.diffusion_audit import _audit_bounded, comparison_cases
from src.eval.diffusion_audit_report import summarize
from tools.vae_compare_post import sha256_file


def audit_row(method, image_id, validity="unknown", *, observation=None, flag=None):
    row = {"method": method, "image_id": image_id, "candidate_id": "c00", "status": "ok",
           "validity": validity, "trace": {"primary_stage": "valid_solid" if validity == "valid" else "unresolved"},
           "raw_consistency": {"status": "ok", "distance_threshold": .01,
                               "observation_flags": {"reciprocal_curve_gap": flag}}}
    if observation is not None:
        row["observation"] = {"status": "ok", "silhouette_iou": observation}
    return row


class DiffusionAuditReportTests(unittest.TestCase):
    def test_missing_measurements_keep_denominator_and_pair_only_common_ids(self):
        rows = [audit_row("baseline", "a", "valid", observation=.7, flag=True),
                audit_row("baseline", "b", "invalid", observation=.3, flag=False),
                audit_row("baseline", "c", flag=None),
                audit_row("dpo_5000", "a", "valid", observation=.8, flag=False),
                audit_row("dpo_5000", "b", "valid", flag=None),
                audit_row("dpo_5000", "c", flag=None)]
        result = summarize(rows)
        self.assertEqual(result["methods"]["baseline"]["intended_count"], 3)
        self.assertEqual(sum(result["methods"]["baseline"]["validity_counts"].values()), 3)
        signal = result["methods"]["baseline"]["signals"]["reciprocal_curve_gap"]["all"]
        self.assertEqual((signal["intended_count"], signal["measured_count"], signal["positive_count"]), (3, 2, 1))
        self.assertEqual(signal["positive_rate"], .5)
        metric = result["observation_metrics"]["silhouette_iou"]
        self.assertEqual(metric["own"]["baseline"]["n"], 2)
        self.assertEqual(metric["paired_common"]["ids"], ["a"])
        self.assertAlmostEqual(metric["paired_common"]["baseline_mean"], .7)
        self.assertAlmostEqual(metric["paired_common"]["dpo_5000_mean"], .8)

    def test_new_validity_measurement_is_not_a_disagreement_with_unknown(self):
        unknown = audit_row("baseline", "a")
        known_invalid = audit_row("baseline", "b", "invalid")
        for row in (unknown, known_invalid):
            row["step_check"] = {"status": "ok", "native": {"native_valid_solid": True}}
        result = summarize([unknown, known_invalid])["methods"]["baseline"]
        self.assertEqual(result["saved_step_check_disagreement_count"], 1)

    def test_duplicate_candidates_are_rejected(self):
        row = audit_row("baseline", "a")
        with self.assertRaisesRegex(ValueError, "unique"):
            summarize([row, row.copy()])

    def test_multiple_candidates_cannot_overwrite_observation_scores(self):
        first = audit_row("baseline", "a", observation=.7)
        second = {**audit_row("baseline", "a", observation=.9), "candidate_id": "c01"}
        with self.assertRaisesRegex(ValueError, "requires K=1"):
            summarize([first, second])

    def test_mixed_distance_thresholds_are_not_aggregated(self):
        first, second = audit_row("baseline", "a"), audit_row("baseline", "b")
        second["raw_consistency"]["distance_threshold"] = .02
        with self.assertRaisesRegex(ValueError, "different thresholds"):
            summarize([first, second])
        self.assertEqual(summarize([second])["distance_threshold"], .02)


class DiffusionAuditReuseTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="hola-diffusion-audit-")
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def test_adapter_keeps_missing_cases_from_frozen_population(self):
        run = self.root / "run"
        (run / "metadata").mkdir(parents=True)
        (run / "metadata/protocol.json").write_text(json.dumps({"identities": ["a", "b"], "config": {
            "raw_data_root": str(self.root / "gt"), "condition_data_root": str(self.root / "images")}}))
        (run / "raw_predictions/baseline/a").mkdir(parents=True)
        cases = comparison_cases(run, ["baseline"])
        self.assertEqual([case["image_id"] for case in cases], ["a", "b"])
        self.assertTrue(cases[0]["gt_npz_path"].endswith("/a/data.npz"))
        with self.assertRaisesRegex(ValueError, "Unknown comparison method"):
            comparison_cases(run, ["typo"])

    def saved_case(self, source):
        case = {"method": "baseline", "image_id": "a", "candidate_id": "c00", "raw_path": str(source)}
        config = {"output_dir": str(self.root / "audit"), "fingerprint": "same-protocol"}
        destination = self.root / "audit/cases/baseline/a/audit.json"
        destination.parent.mkdir(parents=True)
        destination.write_text(json.dumps({**case, "audit_fingerprint": config["fingerprint"], "status": "ok",
            "sources": {"raw_path": {"path": str(source.resolve()), "sha256": sha256_file(source)}}}))
        return case, config

    def test_reuse_rejects_changed_source_content(self):
        source = self.root / "data.npz"
        source.write_bytes(b"first")
        case, config = self.saved_case(source)
        self.assertEqual(_audit_bounded(case, config)["status"], "ok")
        source.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "source changed"):
            _audit_bounded(case, config)

    def test_reuse_rejects_retargeted_source_symlink(self):
        old, new, link = [self.root / name for name in ("old.npz", "new.npz", "data.npz")]
        old.write_bytes(b"first")
        new.write_bytes(b"second")
        link.symlink_to(old)
        case, config = self.saved_case(link)
        link.unlink()
        link.symlink_to(new)
        with self.assertRaisesRegex(ValueError, "source"):
            _audit_bounded(case, config)


if __name__ == "__main__":
    unittest.main()
