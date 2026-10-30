"""Observation-aware preference decisions, missingness, and stable selection."""

from copy import deepcopy
import json

import pytest

from src.preference.observation_selection import (
    SelectionConfig, compare_observation, pair_decision, select_pair, select_pair_with_report,
)
from src.preference.selection import select_pair as select_old_pair


def observation(iou=.8, cd=.01, boundary=.02):
    return {"status": "ok", "silhouette_iou": iou, "visible_face_cd_sq": cd,
            "silhouette_boundary_distance": boundary}


def candidate(name, *, validity="valid", obs=None, raw=None):
    return {"image_id": "part", "candidate_id": name, "validity": validity,
            "observation": observation() if obs is None else obs,
            "raw": {"face_cd_sq": .01, "edge_cd_sq": .02, "face_f1": .8} if raw is None else raw}


@pytest.mark.parametrize("obs", [observation(iou=.82), observation(cd=.0095),
                                 observation(boundary=.019)])
def test_each_observation_can_establish_a_valid_valid_preference(obs):
    winner, loser = candidate("a", obs=obs, raw={}), candidate("b", raw={})
    pair = select_pair([winner, loser])
    assert pair["winner_id"] == "a"
    assert pair["pair_type"] == "geometry"
    assert pair["rule"] == "observation"
    assert "raw" not in pair["checks"]


def test_thresholds_are_inclusive_and_guards_are_conjunctive():
    base = observation()
    boundary = observation(iou=.79, cd=.0095, boundary=.0204)
    check = compare_observation(boundary, base, SelectionConfig())
    assert check["no_regression"] and check["improved"]
    for changed in (observation(iou=.789999, cd=.0095),
                    observation(iou=.82, cd=.01020001),
                    observation(iou=.82, boundary=.02040001)):
        assert pair_decision(candidate("a", obs=changed), candidate("b"))["reason"] == "observation_regression"
    for changed in (observation(iou=.819999), observation(cd=.00950001),
                    observation(boundary=.01900001)):
        assert pair_decision(candidate("a", obs=changed), candidate("b"))["reason"] == "observation_improvement_too_small"


def test_conflicting_metrics_and_hidden_geometry_ties_abstain():
    winner = candidate("a", obs=observation(iou=.84, cd=.02), raw={"face_cd_sq": 0.})
    loser = candidate("b")
    assert select_pair([winner, loser]) is None
    winner["observation"] = deepcopy(loser["observation"])
    winner["post"] = [{"repeat": 0, "face_cd_sq": 0.0}]
    assert select_pair([winner, loser]) is None


def test_validity_gain_needs_no_observation_improvement_but_keeps_raw_guards():
    winner, loser = candidate("a"), candidate("b", validity="invalid")
    winner["raw"] = {"face_cd_sq": .0102, "edge_cd_sq": .0204, "face_f1": .79}
    assert pair_decision(winner, loser)["accepted"]
    for key, value in (("face_cd_sq", .01020001), ("edge_cd_sq", .02040001), ("face_f1", .789999)):
        broken = deepcopy(winner)
        broken["raw"][key] = value
        assert pair_decision(broken, loser)["reason"] == "raw_regression"
    winner["observation"]["silhouette_iou"] = .78
    assert pair_decision(winner, loser)["reason"] == "observation_regression"


@pytest.mark.parametrize("validity", ["unknown", "invalid", None])
def test_unknown_or_invalid_winner_never_wins(validity):
    assert not pair_decision(candidate("a", validity=validity), candidate("b"))["accepted"]


def test_unknown_and_missing_measurements_are_never_automatic_losers():
    winner = candidate("a", obs=observation(iou=.9))
    for loser in (candidate("b", validity="unknown"), candidate("b", obs={}),
                  candidate("b", obs={"status": "timeout"}),
                  candidate("b", validity="invalid", raw={})):
        assert select_pair([winner, loser]) is None
    winner["observation"]["status"] = "render_error"
    assert select_pair([winner, candidate("b", validity="invalid")]) is None


@pytest.mark.parametrize("key,values", [
    ("silhouette_iou", [None, float("nan"), float("inf"), -.01, 1.01, True, ".8"]),
    ("visible_face_cd_sq", [None, float("nan"), float("inf"), -.01, False]),
    ("silhouette_boundary_distance", [None, float("nan"), float("inf"), -.01, 2.01]),
])
def test_invalid_observation_numbers_abstain_and_report_is_json_safe(key, values):
    for value in values:
        broken = candidate("b", validity="invalid")
        broken["observation"][key] = value
        pair, report = select_pair_with_report([candidate("a"), broken])
        assert pair is None
        json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("key,value", [("face_cd_sq", float("nan")),
                                        ("edge_cd_sq", -.01), ("face_f1", 1.01),
                                        ("face_f1", True)])
def test_invalid_raw_metrics_reject_validity_pair_only(key, value):
    loser = candidate("b", validity="invalid")
    loser["raw"][key] = value
    assert pair_decision(candidate("a"), loser)["reason"] == "raw_metrics_missing_or_invalid"
    loser["validity"] = "valid"
    assert pair_decision(candidate("a", obs=observation(iou=.82)), loser)["accepted"]


def test_zero_distances_neither_create_improvement_nor_tolerate_nonzero_regression():
    perfect = observation(iou=1., cd=0., boundary=0.)
    assert select_pair([candidate("a", obs=perfect), candidate("b", obs=perfect)]) is None
    for key in ("visible_face_cd_sq", "silhouette_boundary_distance"):
        worse = deepcopy(perfect)
        worse[key] = 1e-30
        assert not compare_observation(worse, perfect, SelectionConfig())["no_regression"]
        assert compare_observation(perfect, worse, SelectionConfig())["improvements"][key]


def test_validity_priority_deterministic_choice_and_complete_denominator():
    records = [candidate("a", obs=observation(iou=.82)), candidate("b"),
               candidate("c", validity="invalid"), candidate("d", validity="invalid")]
    pair, report = select_pair_with_report(records, seed=17)
    assert pair["pair_type"] == "validity"
    assert report["eligible_validity_pairs"] == 4
    assert report["eligible_geometry_pairs"] == 1
    assert report["candidate_count"] == 4
    assert sum(report["ordered_pair_reasons"].values()) == 12
    assert len(report["ordered_pair_decisions"]) == 12
    assert (pair, report) == select_pair_with_report(records[::-1], seed=17)
    # With the same eligible validity set, preserve the previous deterministic draw.
    old = select_old_pair(records, seed=17)
    assert (pair["winner_id"], pair["loser_id"]) == (old["winner_id"], old["loser_id"])


def test_empty_single_and_invalid_sets_abstain_and_ids_must_be_unique_same_image():
    for records in ([], [candidate("a")], [candidate("a", validity="invalid"),
                                          candidate("b", validity="invalid")]):
        assert select_pair(records) is None
    with pytest.raises(ValueError, match="unique"):
        select_pair([candidate("a"), candidate("a")])
    other = candidate("b")
    other["image_id"] = "other"
    with pytest.raises(ValueError, match="one image"):
        select_pair([candidate("a"), other])
    assert pair_decision(candidate("a"), candidate("a"))["reason"] == "same_candidate"


@pytest.mark.parametrize("kwargs", [{"distance_improvement": float("nan")},
                                     {"distance_allowed_regression": -1.},
                                     {"silhouette_iou_improvement": 1.01},
                                     {"raw_f1_allowed_regression": True},
                                     {"validity_priority": False}])
def test_invalid_selection_configuration_rejected(kwargs):
    with pytest.raises(ValueError):
        SelectionConfig(**kwargs)
