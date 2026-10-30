"""Select at most one same-image preference pair using the frozen metric rule."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import math
import random
from typing import Any


@dataclass(frozen=True)
class SelectionConfig:
    """CD changes are relative; F1 changes are absolute (0.02 = 2 points)."""

    version: str = "svr_preference_v3"
    cd_improvement: float = 0.05
    cd_allowed_regression: float = 0.02
    f1_improvement: float = 0.02
    f1_allowed_regression: float = 0.01
    post_repeats: int = 3
    validity_priority: bool = True

    def __post_init__(self):
        values = (self.cd_improvement, self.cd_allowed_regression,
                  self.f1_improvement, self.f1_allowed_regression)
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("Selection thresholds must be finite and nonnegative")
        if self.post_repeats < 2:
            raise ValueError("Post geometry requires at least two sampling repeats")
        if not self.validity_priority:
            raise ValueError("This protocol selects validity pairs before geometry pairs")


def _complete(metrics: dict) -> bool:
    values = [metrics.get(key) for key in ("face_cd_sq", "edge_cd_sq", "face_f1")]
    return (all(isinstance(value, (int, float)) and math.isfinite(value) for value in values)
            and values[0] >= 0 and values[1] >= 0 and 0 <= values[2] <= 1)


def _at_most(left: float, right: float) -> bool:
    # Inclusive thresholds, allowing only floating-point arithmetic roundoff.
    return left <= right or math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-15)


def compare_metrics(winner: dict, loser: dict, config: SelectionConfig) -> dict:
    """Return each guard and improvement decision; never combine metrics."""
    if not _complete(winner) or not _complete(loser):
        return {"complete": False, "no_regression": False, "improved": False}
    guards, improvements = {}, {}
    for key in ("face_cd_sq", "edge_cd_sq"):
        guards[key] = _at_most(winner[key], loser[key] * (1 + config.cd_allowed_regression))
        # Equal zero is not an improvement, and no arbitrary denominator epsilon
        # turns a perfect candidate into a usable relative-improvement target.
        improvements[key] = (winner[key] < loser[key] and
                             _at_most(winner[key], loser[key] * (1 - config.cd_improvement)))
    guards["face_f1"] = _at_most(loser["face_f1"] - config.f1_allowed_regression, winner["face_f1"])
    improvements["face_f1"] = (
        winner["face_f1"] > loser["face_f1"] and
        _at_most(loser["face_f1"] + config.f1_improvement, winner["face_f1"]))
    return {"complete": True, "no_regression": all(guards.values()),
            "improved": any(improvements.values()), "guards": guards,
            "improvements": improvements}


def pair_decision(winner: dict, loser: dict, config: SelectionConfig) -> dict:
    """Explain why this ordered pair is accepted or rejected."""
    if winner["image_id"] != loser["image_id"]:
        raise ValueError("Preference candidates must come from the same image")
    if winner["candidate_id"] == loser["candidate_id"]:
        return {"accepted": False, "reason": "same_candidate"}
    if winner.get("validity") != "valid":
        return {"accepted": False, "reason": "winner_not_valid"}
    if loser.get("validity") not in ("valid", "invalid"):
        return {"accepted": False, "reason": "loser_validity_unknown"}

    if loser["validity"] == "invalid":
        check = compare_metrics(winner.get("raw", {}), loser.get("raw", {}), config)
        reason = ("raw_metrics_missing" if not check["complete"] else
                  "raw_regression" if not check["no_regression"] else "accepted")
        return {"accepted": reason == "accepted", "reason": reason,
                "pair_type": "validity", "checks": [check]}

    positive, negative = winner.get("post", []), loser.get("post", [])
    expected = list(range(config.post_repeats))
    if ([entry.get("repeat") for entry in positive] != expected or
            [entry.get("repeat") for entry in negative] != expected):
        return {"accepted": False, "reason": "post_repeats_missing", "pair_type": "geometry"}
    checks = [compare_metrics(a, b, config) for a, b in zip(positive, negative)]
    if not all(check["complete"] for check in checks):
        reason = "post_metrics_missing"
    elif not all(check["no_regression"] for check in checks):
        reason = "post_regression"
    elif not all(check["improved"] for check in checks):
        reason = "post_improvement_unstable_or_small"
    else:
        reason = "accepted"
    return {"accepted": reason == "accepted", "reason": reason,
            "pair_type": "geometry", "checks": checks}


def select_pair_with_report(candidates: list[dict], *, seed: int = 42,
                            config: SelectionConfig | None = None) -> tuple[dict | None, dict]:
    """Enumerate ordered pairs, prefer validity, then select reproducibly."""
    config = config or SelectionConfig()
    records = sorted(candidates, key=lambda record: str(record["candidate_id"]))
    image_ids = {record["image_id"] for record in records}
    if len(image_ids) > 1:
        raise ValueError("Selection operates on exactly one image at a time")
    if len({record["candidate_id"] for record in records}) != len(records):
        raise ValueError("Candidate IDs must be unique within an image")
    eligible: dict[str, list[dict]] = {"validity": [], "geometry": []}
    reasons: Counter = Counter()
    for winner in records:
        for loser in records:
            if winner is loser:
                continue
            decision = pair_decision(winner, loser, config)
            reasons[decision["reason"]] += 1
            if decision["accepted"]:
                eligible[decision["pair_type"]].append({
                    "image_id": winner["image_id"], "winner_id": winner["candidate_id"],
                    "loser_id": loser["candidate_id"], **decision,
                })
    report: dict[str, Any] = {
        "candidate_count": len(records), "ordered_pair_reasons": dict(reasons),
        "eligible_validity_pairs": len(eligible["validity"]),
        "eligible_geometry_pairs": len(eligible["geometry"]), "selection": asdict(config),
    }
    choices = eligible["validity"] or eligible["geometry"]
    if not choices:
        return None, report
    image_id = next(iter(image_ids))
    stable = int.from_bytes(hashlib.sha256(f"{seed}:{image_id}".encode()).digest()[:8], "little")
    pair = random.Random(stable).choice(choices)
    pair.update(selection_seed=seed, selection=asdict(config))
    return pair, report


def select_pair(candidates: list[dict], *, seed: int = 42,
                config: SelectionConfig | None = None) -> dict | None:
    return select_pair_with_report(candidates, seed=seed, config=config)[0]
