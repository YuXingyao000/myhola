"""Select one observation-aware pair from the saved same-image candidates.

Full-GT raw geometry only guards a validity preference. Among two valid
candidates, visible evidence alone establishes a preference; a hidden-surface
match to the single training GT is not a tie breaker.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import math
from numbers import Real
import random
from typing import Any


@dataclass(frozen=True)
class SelectionConfig:
    """Distance changes are relative; IoU/F1 changes are absolute fractions."""

    version: str = "svr_observation_v1"
    silhouette_iou_improvement: float = 0.02
    silhouette_iou_allowed_regression: float = 0.01
    distance_improvement: float = 0.05
    distance_allowed_regression: float = 0.02
    raw_cd_allowed_regression: float = 0.02
    raw_f1_allowed_regression: float = 0.01
    validity_priority: bool = True

    def __post_init__(self) -> None:
        values = (
            self.silhouette_iou_improvement, self.silhouette_iou_allowed_regression,
            self.distance_improvement, self.distance_allowed_regression,
            self.raw_cd_allowed_regression, self.raw_f1_allowed_regression,
        )
        if any(not _finite_number(value) or value < 0 for value in values):
            raise ValueError("Selection thresholds must be finite and nonnegative")
        if any(value > 1 for value in (
            self.silhouette_iou_improvement, self.silhouette_iou_allowed_regression,
            self.distance_improvement, self.raw_f1_allowed_regression,
        )):
            raise ValueError("Fractional improvement and IoU/F1 thresholds cannot exceed one")
        if self.validity_priority is not True:
            raise ValueError("This protocol selects validity pairs before observation pairs")


def _finite_number(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


def _metric_errors(metrics: Any, ranges: dict[str, tuple[float, float | None]]) -> dict:
    if not isinstance(metrics, dict):
        return {"metrics": "missing"}
    errors = {}
    for key, (lower, upper) in ranges.items():
        value = metrics.get(key)
        if value is None:
            errors[key] = "missing"
        elif not _finite_number(value):
            errors[key] = "not_finite_number"
        elif value < lower or (upper is not None and value > upper):
            errors[key] = "out_of_range"
    return errors


def _at_most(left: float, right: float) -> bool:
    # Permit arithmetic roundoff at inclusive thresholds, never a nonzero
    # distance against a perfect zero reference.
    return left <= right or math.isclose(left, right, rel_tol=1e-12, abs_tol=0.0)


def compare_observation(winner: Any, loser: Any, config: SelectionConfig) -> dict:
    """Check all observation guards and each independent improvement signal."""
    ranges = {
        "silhouette_iou": (0.0, 1.0),
        "visible_face_cd_sq": (0.0, None),
        # Sum of two directed distances, each normalized by the image diagonal.
        "silhouette_boundary_distance": (0.0, 2.0),
    }
    errors = {}
    for side, metrics in (("winner", winner), ("loser", loser)):
        side_errors = _metric_errors(metrics, ranges)
        if not isinstance(metrics, dict) or metrics.get("status") != "ok":
            side_errors["status"] = "not_ok"
        if side_errors:
            errors[side] = side_errors
    if errors:
        return {"complete": False, "no_regression": False, "improved": False,
                "errors": errors}
    guards = {
        "silhouette_iou": _at_most(
            loser["silhouette_iou"] - config.silhouette_iou_allowed_regression,
            winner["silhouette_iou"],
        ),
    }
    improvements = {
        "silhouette_iou": (
            winner["silhouette_iou"] > loser["silhouette_iou"] and
            _at_most(loser["silhouette_iou"] + config.silhouette_iou_improvement,
                     winner["silhouette_iou"])
        ),
    }
    for key in ("visible_face_cd_sq", "silhouette_boundary_distance"):
        guards[key] = _at_most(winner[key], loser[key] * (1 + config.distance_allowed_regression))
        improvements[key] = (
            winner[key] < loser[key] and
            _at_most(winner[key], loser[key] * (1 - config.distance_improvement))
        )
    return {"complete": True, "no_regression": all(guards.values()),
            "improved": any(improvements.values()), "guards": guards,
            "improvements": improvements}


def _raw_guard(winner: Any, loser: Any, config: SelectionConfig) -> dict:
    ranges = {"face_cd_sq": (0.0, None), "edge_cd_sq": (0.0, None), "face_f1": (0.0, 1.0)}
    errors = {side: error for side, metrics in (("winner", winner), ("loser", loser))
              if (error := _metric_errors(metrics, ranges))}
    if errors:
        return {"complete": False, "no_regression": False, "errors": errors}
    guards = {
        key: _at_most(winner[key], loser[key] * (1 + config.raw_cd_allowed_regression))
        for key in ("face_cd_sq", "edge_cd_sq")
    }
    guards["face_f1"] = _at_most(
        loser["face_f1"] - config.raw_f1_allowed_regression, winner["face_f1"],
    )
    return {"complete": True, "no_regression": all(guards.values()), "guards": guards}


def pair_decision(winner: dict, loser: dict, config: SelectionConfig | None = None) -> dict:
    """Explain acceptance of an ordered pair without treating missing data as loss."""
    config = config or SelectionConfig()
    if winner["image_id"] != loser["image_id"]:
        raise ValueError("Preference candidates must come from the same image")
    if winner["candidate_id"] == loser["candidate_id"]:
        return {"accepted": False, "reason": "same_candidate"}
    if winner.get("validity") != "valid":
        return {"accepted": False, "reason": "winner_not_valid"}
    if loser.get("validity") not in ("valid", "invalid"):
        return {"accepted": False, "reason": "loser_validity_unknown"}

    pair_type = "validity" if loser["validity"] == "invalid" else "geometry"
    observation = compare_observation(winner.get("observation"), loser.get("observation"), config)
    checks = {"observation": observation}
    if not observation["complete"]:
        reason = "observation_metrics_missing_or_invalid"
    elif not observation["no_regression"]:
        reason = "observation_regression"
    elif pair_type == "validity":
        raw = _raw_guard(winner.get("raw"), loser.get("raw"), config)
        checks["raw"] = raw
        reason = ("raw_metrics_missing_or_invalid" if not raw["complete"] else
                  "raw_regression" if not raw["no_regression"] else "accepted")
    else:
        reason = "accepted" if observation["improved"] else "observation_improvement_too_small"
    return {"accepted": reason == "accepted", "reason": reason, "pair_type": pair_type,
            "rule": "observation", "checks": checks}


def select_pair_with_report(
    candidates: list[dict], *, seed: int = 42, config: SelectionConfig | None = None,
) -> tuple[dict | None, dict]:
    """Prefer eligible validity pairs, then uniformly choose using the old seed rule."""
    config = config or SelectionConfig()
    records = sorted(candidates, key=lambda record: str(record["candidate_id"]))
    image_ids = {record["image_id"] for record in records}
    if len(image_ids) > 1:
        raise ValueError("Selection operates on exactly one image at a time")
    if len({record["candidate_id"] for record in records}) != len(records):
        raise ValueError("Candidate IDs must be unique within an image")
    eligible: dict[str, list[dict]] = {"validity": [], "geometry": []}
    reasons: Counter = Counter()
    decisions = []
    for winner in records:
        for loser in records:
            if winner is loser:
                continue
            decision = pair_decision(winner, loser, config)
            reasons[decision["reason"]] += 1
            ordered = {"winner_id": winner["candidate_id"], "loser_id": loser["candidate_id"]}
            # Preserve each rejection/acceptance without repeating the candidate
            # metrics or full checks 56 times per eight-candidate image.
            decisions.append({**ordered, **{key: value for key, value in decision.items()
                                           if key != "checks"}})
            if decision["accepted"]:
                eligible[decision["pair_type"]].append({
                    "image_id": winner["image_id"], **ordered, **decision,
                })
    report: dict[str, Any] = {
        "candidate_count": len(records), "ordered_pair_reasons": dict(reasons),
        "ordered_pair_decisions": decisions,
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
