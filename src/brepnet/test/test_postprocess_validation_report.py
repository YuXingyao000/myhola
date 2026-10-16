"""Post-policy reports keep fixed cohorts and distinguish validity from geometry."""
from contextlib import redirect_stdout
from copy import deepcopy
from io import StringIO
from pathlib import Path
import sys
import tempfile
import unittest
import warnings
from unittest.mock import patch

from tools.postprocess_validation_report import (
    POLICIES, aggregate, draw_results, main, validate_policy_records,
)
from tools.vae_compare_post import atomic_json, fingerprint, read_json, sha256_file


class PostprocessReportTests(unittest.TestCase):
    def fixture(self):
        ids = ["a", "b", "c", "d", "e"]
        rows = []
        for index, sample_id in enumerate(ids):
            rows.append({
                "sample_id": sample_id,
                "raw_sha256": None if sample_id == "d" else sample_id * 64,
                "status": "missing_raw" if sample_id == "d" else "ok",
                "native_valid_solid": sample_id in {"a", "c"},
                "geometry_status": "ok" if sample_id in {"a", "b", "e"} else "not_run",
                "face_cd": float(index + 1), "edge_cd": float(2 * (index + 1)),
            })
        records = {policy: deepcopy(rows) for policy in POLICIES}
        candidate = records["roundtrip_retry"]
        candidate[0].update(face_cd=0.5, edge_cd=1.0)
        candidate[1].update(native_valid_solid=True, face_cd=1.0, edge_cd=2.0)
        candidate[2].update(native_valid_solid=False, geometry_status="ok")
        return records, ids

    def test_fixed_denominators_and_transition_conservation(self):
        records, ids = self.fixture()
        result = aggregate(records, ids, original_count=6)
        baseline = result["policies"]["legacy"]
        self.assertEqual(baseline["native_valid_count"], 2)
        self.assertEqual(baseline["rate_accepted"], 2 / 5)
        self.assertEqual(baseline["rate_original"], 2 / 6)
        self.assertEqual(baseline["status_counts"]["missing_raw"], 1)
        pair = result["paired"]["roundtrip_retry"]
        self.assertEqual(pair["counts"], {"both_valid": 1, "rescued": 1, "regressed": 1, "both_failed": 2})
        self.assertEqual(sum(pair["counts"].values()), len(ids))
        self.assertEqual(pair["ids"]["rescued"], ["b"])
        self.assertEqual(pair["ids"]["regressed"], ["c"])

    def test_geometry_computable_and_common_valid_are_separate_cohorts(self):
        records, ids = self.fixture()
        result = aggregate(records, ids, original_count=6)
        geometry = result["paired"]["roundtrip_retry"]["geometry"]
        self.assertEqual(geometry["common_computable"]["count"], 3)
        self.assertEqual(geometry["common_valid"]["count"], 1)
        self.assertEqual(geometry["common_computable"]["face_cd"]["legacy"]["count"], 3)
        self.assertAlmostEqual(geometry["common_computable"]["face_cd"]["legacy"]["mean"], 8 / 3)
        self.assertEqual(geometry["common_valid"]["face_cd"]["candidate_minus_baseline"]["mean"], -0.5)

    def test_timeout_after_native_check_keeps_validity_but_not_incomplete_geometry(self):
        records, ids = self.fixture()
        records["legacy"][0].update(status="timeout", stage="geometry", geometry_status="running")
        result = aggregate(records, ids, original_count=6)
        self.assertEqual(result["policies"]["legacy"]["native_valid_count"], 2)
        self.assertEqual(result["policies"]["legacy"]["status_counts"]["timeout"], 1)
        pair = result["paired"]["roundtrip_retry"]
        self.assertEqual(pair["counts"]["both_valid"], 1)
        self.assertEqual(pair["geometry"]["common_valid"]["count"], 0)

    def test_duplicate_missing_and_unexpected_record_ids_are_rejected(self):
        for change in ("duplicate", "missing", "unexpected"):
            records, ids = self.fixture()
            rows = records["fit_score_v2"]
            if change == "duplicate":
                rows[-1] = deepcopy(rows[0])
            elif change == "missing":
                rows.pop()
            else:
                rows[-1]["sample_id"] = "foreign"
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "Record IDs differ"):
                aggregate(records, ids, original_count=6)

    def test_nonterminal_record_is_rejected(self):
        records, ids = self.fixture()
        records["recover_v1"][1]["status"] = "running"
        with self.assertRaisesRegex(ValueError, "Nonterminal record"):
            aggregate(records, ids, original_count=6)

    def test_same_id_must_have_same_raw_hash_in_all_policies(self):
        records, ids = self.fixture()
        records["fit_score_v2"][1]["raw_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "Raw predictions differ"):
            aggregate(records, ids, original_count=6)

    def test_invalid_intended_cohort_is_rejected(self):
        records, ids = self.fixture()
        for chosen, count in (([], 6), (ids + [ids[0]], 6), (ids, 4)):
            with self.subTest(ids=chosen, count=count), self.assertRaisesRegex(ValueError, "Invalid intended"):
                aggregate(records, chosen, original_count=count)

    def test_explicit_two_policies_preserve_metrics_and_default_requires_all_four(self):
        records, ids = self.fixture()
        full = aggregate(records, ids, original_count=6)
        policies = ("legacy", "recover_v1")
        selected = {p: records[p] for p in policies}
        result = aggregate(selected, ids, original_count=6, policies=policies)
        self.assertEqual(tuple(result["policies"]), policies)
        self.assertEqual(result["policies"], {p: full["policies"][p] for p in policies})
        self.assertEqual(result["paired"], {"recover_v1": full["paired"]["recover_v1"]})
        with self.assertRaisesRegex(ValueError, "Missing policy records"):
            aggregate(selected, ids, original_count=6)

    def test_invalid_policy_selections_are_rejected(self):
        records, ids = self.fixture()
        for policies, error in (((), "legacy"), (("recover_v1",), "legacy"),
                                (("legacy",), "candidate"),
                                (("legacy", "recover_v1", "legacy"), "Duplicate"),
                                (("legacy", "unknown"), "Unknown")):
            with self.subTest(policies=policies), self.assertRaisesRegex(ValueError, error):
                aggregate(records, ids, original_count=6, policies=policies)

    def test_two_policy_chart_uses_selected_labels_and_test_split(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        records, ids = self.fixture()
        result = aggregate(records, ids, original_count=6, policies=("legacy", "recover_v1"))
        with tempfile.TemporaryDirectory() as temporary, warnings.catch_warnings(), \
                patch("tools.postprocess_validation_report.configure_plot", return_value=plt), \
                patch("matplotlib.figure.Figure.savefig"), patch.object(plt, "close"):
            warnings.filterwarnings("ignore", message="Glyph .* missing from font")
            draw_results(result, Path(temporary), Path("unused-font"), split_label="test")
            figure = plt.gcf()
            self.assertEqual([t.get_text() for t in figure.axes[0].get_yticklabels()],
                             ["原后处理", "原流程优先＋失败再修复"])
            self.assertEqual([t.get_text() for t in figure.axes[1].get_xticklabels()], ["失败再修复"])
            self.assertEqual(len(figure.axes[0].patches), 2)
            self.assertIn("5 个预定 test 样本", figure.axes[0].get_xlabel())
            self.assertNotIn("validation", figure.axes[0].get_xlabel())
        plt.close(figure)

    def report_fixture(self, root, policies=POLICIES):
        records, ids = self.fixture()
        split = root / "split.txt"
        split.write_text("\n".join(ids) + "\n")
        common = {
            "split_sha256": sha256_file(split), "intended_count": len(ids), "original_test_count": 6,
            "mode": "optimized",
            "sampling": {"face": 1024, "edge": 128}, "native_validity": "read mode 0",
            "timeout_seconds": 180, "max_optimize_iter": 200, "post_algorithm_files": {"test.py": "sha"},
        }
        for policy in policies:
            protocol = {**common, "post_policy": policy}
            directory = root / "post_variants" / policy / "metadata/post/optimized"
            atomic_json(directory / "protocol.json", protocol)
            for row in records[policy]:
                row["protocol_fingerprint"] = fingerprint(protocol)
                row.update(post_policy=policy, method="intersection", mode="optimized")
                atomic_json(directory / "samples/intersection" / (row["sample_id"] + ".json"), row)
        return split

    def test_validate_records_accepts_matching_protocol_and_rejects_record_identity_changes(self):
        protocol = {"split_sha256": "split", "intended_count": 1, "original_test_count": 2,
                    "mode": "optimized", "post_policy": "recover_v1"}
        row = {"sample_id": "sample", "method": "intersection", "mode": "optimized",
               "post_policy": "recover_v1", "protocol_fingerprint": fingerprint(protocol)}
        self.assertIsNone(validate_policy_records(protocol, [row], "recover_v1", "split", 1, 2))
        for field, wrong in (("method", "point_feedback"), ("mode", "fit_only"),
                             ("post_policy", "roundtrip_retry"), ("protocol_fingerprint", "stale")):
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_policy_records(protocol, [{**row, field: wrong}], "recover_v1", "split", 1, 2)

    def test_validate_records_rejects_protocol_mismatch_against_requested_report(self):
        protocol = {"split_sha256": "split", "intended_count": 1, "original_test_count": 2,
                    "mode": "optimized", "post_policy": "recover_v1"}
        for field, wrong in (("split_sha256", "foreign"), ("intended_count", 9), ("original_test_count", 9),
                             ("mode", "fit_only"), ("post_policy", "legacy")):
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_policy_records({**protocol, field: wrong}, [], "recover_v1", "split", 1, 2)

    def run_report(self, root, split, original_count=6, extra_args=()):
        argv = ["postprocess_validation_report.py", "--run-root", str(root), "--split-list", str(split),
                "--original-count", str(original_count), *extra_args]
        with patch.object(sys, "argv", argv), patch("tools.postprocess_validation_report.draw_explanations"), \
                patch("tools.postprocess_validation_report.draw_results") as draw, redirect_stdout(StringIO()):
            main()
        return draw

    def test_report_rejects_cli_subset_that_differs_from_frozen_split(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            split = self.report_fixture(root)
            split.write_text("a\nb\n")
            with self.assertRaisesRegex(ValueError, "[Ss]plit|intended_count"):
                self.run_report(root, split)

    def test_report_accepts_the_frozen_cohort(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            split = self.report_fixture(root)
            self.run_report(root, split)
            result = read_json(root / "report/summary.json")
            self.assertEqual(result["intended_count"], 5)
            self.assertEqual(result["original_count"], 6)
            self.assertEqual(result["policies"]["legacy"]["native_valid_count"], 2)

    def test_report_accepts_only_selected_two_policies_and_labels_test(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            policies = ("legacy", "recover_v1")
            split = self.report_fixture(root, policies=policies)
            draw = self.run_report(root, split, extra_args=("--policies", *policies, "--split-label", "test"))
            result = read_json(root / "report/summary.json")
            self.assertEqual(tuple(result["policies"]), policies)
            self.assertEqual(tuple(result["protocols"]), policies)
            self.assertEqual(result["split_label"], "test")
            self.assertEqual(draw.call_args.kwargs["split_label"], "test")

    def test_report_rejects_invalid_policy_cli_before_creating_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for policies in (("recover_v1",), ("legacy", "legacy"), ("legacy", "unknown")):
                with self.subTest(policies=policies), self.assertRaises(ValueError):
                    self.run_report(root, root / "missing.txt", extra_args=("--policies", *policies))
                self.assertFalse((root / "report").exists())

    def test_report_rejects_cli_original_denominator_that_differs_from_protocol(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            split = self.report_fixture(root)
            with self.assertRaisesRegex(ValueError, "original"):
                self.run_report(root, split, original_count=7)

    def test_report_rejects_rows_from_a_different_policy_or_budget_protocol(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            split = self.report_fixture(root)
            path = root / "post_variants/recover_v1/metadata/post/optimized/samples/intersection/a.json"
            row = read_json(path)
            row["protocol_fingerprint"] = "stale-budget-or-policy"
            atomic_json(path, row)
            with self.assertRaisesRegex(ValueError, "[Pp]rotocol|fingerprint"):
                self.run_report(root, split)


if __name__ == "__main__":
    unittest.main()
