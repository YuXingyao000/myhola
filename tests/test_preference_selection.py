"""Preference thresholds, invalidity semantics, and repeated-measurement labels."""

from copy import deepcopy
import unittest

from src.preference.selection import (
    SelectionConfig, compare_metrics, pair_decision, select_pair, select_pair_with_report,
)


def metrics(face=1., edge=1., f1=.8):
    return {"face_cd_sq": face, "edge_cd_sq": edge, "face_f1": f1}


def candidate(name, *, validity="valid", raw=None, post=None):
    return {"image_id": "part", "candidate_id": name, "validity": validity,
            "raw": raw if raw is not None else metrics(),
            "post": [{"repeat": i, **values} for i, values in enumerate(post or [metrics()] * 3)]}


class PreferenceSelectionTests(unittest.TestCase):
    def setUp(self):
        self.config = SelectionConfig()

    def test_inclusive_relative_cd_and_absolute_f1_thresholds(self):
        loser = metrics(face=2., edge=4., f1=.9)
        winner = metrics(face=1.9, edge=4.08, f1=.89)
        decision = compare_metrics(winner, loser, self.config)
        self.assertTrue(decision["no_regression"])
        self.assertTrue(decision["improvements"]["face_cd_sq"])
        self.assertFalse(decision["improvements"]["face_f1"])
        for changed in (metrics(1.9001, 4., .9), metrics(2., 4., .9199)):
            self.assertFalse(compare_metrics(changed, loser, self.config)["improved"])
        self.assertTrue(compare_metrics(metrics(2., 4., .92), loser, self.config)["improved"])
        for changed in (metrics(1.9, 4.0801, .9), metrics(1.9, 4., .8899)):
            self.assertFalse(compare_metrics(changed, loser, self.config)["no_regression"])

    def test_two_wins_cannot_override_large_third_metric_regression(self):
        winner = candidate("a", post=[metrics(.8, .8, .4)] * 3)
        loser = candidate("b", post=[metrics(1., 1., .9)] * 3)
        self.assertEqual(pair_decision(winner, loser, self.config)["reason"], "post_regression")
        self.assertIsNone(select_pair([winner, loser]))

    def test_validity_gain_accepts_equal_raw_geometry_not_unknown_failure(self):
        winner = candidate("a")
        invalid = candidate("b", validity="invalid")
        pair = select_pair([winner, invalid])
        self.assertEqual((pair["winner_id"], pair["loser_id"], pair["pair_type"]),
                         ("a", "b", "validity"))
        invalid["validity"] = "unknown"
        invalid["status"] = "timeout"
        self.assertIsNone(select_pair([winner, invalid]))

    def test_validity_uses_raw_and_geometry_uses_post(self):
        winner = candidate("a", raw=metrics(2.), post=[metrics(.8)] * 3)
        loser = candidate("b", raw=metrics(1.), post=[metrics(1.)] * 3)
        self.assertEqual(select_pair([winner, loser])["winner_id"], "a")
        loser["validity"] = "invalid"
        self.assertIsNone(select_pair([winner, loser]))

    def test_every_post_repeat_must_show_clear_same_direction_preference(self):
        winner = candidate("a", post=[metrics(.8), metrics(.8), metrics(.99)])
        loser = candidate("b")
        self.assertEqual(pair_decision(winner, loser, self.config)["reason"],
                         "post_improvement_unstable_or_small")
        self.assertIsNone(select_pair([winner, loser]))
        winner["post"][2] = {"repeat": 2, **metrics(1.2)}
        self.assertIsNone(select_pair([winner, loser]))

    def test_missing_nonfinite_and_zero_metrics_are_not_fake_improvements(self):
        for value in (None, float("nan"), float("inf"), -1.):
            broken = candidate("a", validity="invalid", raw=metrics(face=value))
            self.assertIsNone(select_pair([candidate("b"), broken]))
        winner, loser = candidate("a"), candidate("b")
        winner["post"] = winner["post"][:2]
        self.assertIsNone(select_pair([winner, loser]))
        self.assertFalse(compare_metrics(metrics(0., 0., 1.), metrics(0., 0., 1.), self.config)["improved"])
        self.assertFalse(compare_metrics(metrics(1e-8, 0., 1.), metrics(0., 0., 1.), self.config)["no_regression"])

    def test_all_valid_all_invalid_and_validity_priority(self):
        better = candidate("a", post=[metrics(.8)] * 3)
        ordinary = candidate("b")
        invalid = candidate("c", validity="invalid")
        self.assertEqual(select_pair([better, ordinary])["pair_type"], "geometry")
        self.assertIsNone(select_pair([invalid, candidate("d", validity="invalid")]))
        pair, report = select_pair_with_report([better, ordinary, invalid])
        self.assertEqual(pair["pair_type"], "validity")
        self.assertEqual(report["eligible_validity_pairs"], 2)
        self.assertEqual(report["eligible_geometry_pairs"], 1)

    def test_selection_reproducible_under_record_order_and_requires_same_image(self):
        records = [candidate("a"), candidate("b", validity="invalid"),
                   candidate("c", validity="invalid")]
        self.assertEqual(select_pair(records, seed=17), select_pair(records[::-1], seed=17))
        mixed = deepcopy(records)
        mixed[0]["image_id"] = "other_part"
        with self.assertRaisesRegex(ValueError, "one image"):
            select_pair(mixed)


if __name__ == "__main__":
    unittest.main()
