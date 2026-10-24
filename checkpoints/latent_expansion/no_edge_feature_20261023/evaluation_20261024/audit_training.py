"""Reproduce the frozen training-history audit from downloaded W&B JSONL.

Offline verification (stdlib only; does not alter training_audit.json)::

    python checkpoints/latent_expansion/no_edge_feature_20261023/evaluation_20261024/audit_training.py

The original collection used ``wandb.Api(timeout=90).run(path).scan_history(
page_size=1000)`` without sampling for both runs. This script preserves the exact
compaction, full-epoch intersection, and 25k--30k arithmetic-mean calculations.
Run state and process-exit evidence remain time-stamped observations in the audit;
they cannot be inferred from history alone. No network calls are made here.

Optionally recheck the trusted local checkpoint on 142 with its existing torch /
OmegaConf environment: append ``--checkpoint /absolute/path/to/last.ckpt``.
``--output NEW.json`` writes the verification result, refusing to overwrite.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics


HERE = Path(__file__).resolve().parent
LABELS = ("d64_no_edge_feature", "d64_original")
METRICS = (
    "loss", "loss_epoch", "edge_feature", "weighted_edge_feature", "kl_loss",
    "weighted_kl_loss", "encoder_edge_local", "encoder_edge_bbox", "edge_xyz",
    "face_xyz", "refined_edge_local", "refined_edge_bbox", "coarse_edge_local",
    "coarse_edge_bbox", "refined_face_local", "refined_face_bbox", "uv",
    "uv_anchor", "incidence", "refined_precision", "refined_recall",
    "coarse_precision", "coarse_recall", "delta_edge_mean", "delta_face_mean",
)
WINDOW_METRICS = (
    "encoder_edge_local", "encoder_edge_bbox", "coarse_edge_local",
    "coarse_edge_bbox", "refined_edge_local", "refined_edge_bbox", "edge_xyz",
    "face_xyz", "incidence", "uv", "uv_anchor", "kl_loss", "weighted_kl_loss",
    "edge_feature", "weighted_edge_feature", "refined_precision",
    "refined_recall", "non_feature_sum",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_equal(actual: object, expected: object, label: str) -> None:
    if actual != expected:
        raise ValueError(f"Recomputed value differs from training_audit: {label}")


def compact(row: dict, split: str) -> dict:
    result = {
        key: row.get(key)
        for key in ("epoch", "trainer/global_step", "_step", "_runtime", "_timestamp")
    }
    result.update({key: row[f"{split}/{key}"] for key in METRICS if f"{split}/{key}" in row})
    weighted = {
        key: value for key, value in row.items()
        if key.startswith(f"{split}/weighted_") and value is not None
    }
    result["weighted_count"] = len(weighted)
    result["weighted_sum"] = sum(weighted.values())
    result["non_feature_sum"] = result["weighted_sum"] - result["weighted_edge_feature"]
    result["non_feature_non_kl_sum"] = result["non_feature_sum"] - result["weighted_kl_loss"]
    result["weighted_sum_minus_total"] = result["weighted_sum"] - result[
        "loss_epoch" if split == "train" else "loss"
    ]
    result["full_epoch"] = (int(result["trainer/global_step"]) + 1) % 363 == 0
    return result


def check_checkpoint(path: Path, expected: dict) -> dict:
    # Checkpoints are trusted research artifacts; this mirrors the original audit.
    import torch
    from omegaconf import OmegaConf

    torch.set_num_threads(2)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    nonfinite = [
        key for key, value in checkpoint["state_dict"].items()
        if torch.is_tensor(value)
        and (value.is_floating_point() or value.is_complex())
        and not torch.isfinite(value).all().item()
    ]
    optimizers = checkpoint.get("optimizer_states", [])
    observed = {
        "sha256": sha256(path), "bytes": path.stat().st_size,
        "global_step": checkpoint["global_step"], "epoch": checkpoint["epoch"],
        "config": OmegaConf.to_container(
            OmegaConf.create(checkpoint["hyper_parameters"]), resolve=True
        ),
        "state_tensor_count": len(checkpoint["state_dict"]),
        "nonfinite_tensor_keys": nonfinite, "all_state_tensors_finite": not nonfinite,
        "optimizer_count": len(optimizers),
        "optimizer_state_count": [len(item["state"]) for item in optimizers],
        "optimizer_steps": [
            sorted({int(state["step"]) for state in item["state"].values() if "step" in state})
            for item in optimizers
        ],
        "optimizer_learning_rates": [
            [group["lr"] for group in item["param_groups"]] for item in optimizers
        ],
    }
    for key, value in observed.items():
        require_equal(value, expected[key], f"checkpoint.{key}")
    return {"path": str(path), "sha256": observed["sha256"], "matches_audit": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=HERE)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        raise FileExistsError(args.output)
    audit_path = args.directory / "training_audit.json"
    audit = json.loads(audit_path.read_text())
    expected = audit["history"]
    runs = {}
    result = {"matches_audit": True, "audit_sha256": sha256(audit_path), "inputs": {}}
    for label in LABELS:
        path = args.directory / f"{label}_history.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        require_equal(len(rows), expected["runs"][label]["history_rows"], f"{label}.history_rows")
        splits = {
            split: [
                compact(row, split) for row in rows
                if row.get(f"{split}/{total}") is not None
                and row.get("trainer/global_step") is not None
            ]
            for split, total in (("train", "loss_epoch"), ("val", "loss"))
        }
        require_equal(splits, expected["runs"][label]["split_rows"], f"{label}.split_rows")
        duplicate_keys = {
            split: [list(key) for key, count in Counter(
                (row["epoch"], row["trainer/global_step"]) for row in values
            ).items() if count > 1]
            for split, values in splits.items()
        }
        nonfinite = {
            split: [
                {"step": row["trainer/global_step"], "key": key, "value": value}
                for row in values for key, value in row.items()
                if isinstance(value, (int, float)) and not math.isfinite(value)
            ]
            for split, values in splits.items()
        }
        errors = {
            split: max(abs(row["weighted_sum_minus_total"]) for row in values)
            for split, values in splits.items()
        }
        for key, value in (("duplicate_epoch_step_keys", duplicate_keys),
                           ("nonfinite_epoch_metrics", nonfinite),
                           ("max_weighted_sum_error", errors)):
            require_equal(value, expected["runs"][label][key], f"{label}.{key}")
        if any(duplicate_keys.values()) or any(nonfinite.values()):
            raise ValueError(f"Invalid epoch history: {label}")
        runs[label] = splits
        result["inputs"][label] = {"sha256": sha256(path), "rows": len(rows)}

    result["aligned"] = {}
    result["comparison_25k_30k"] = {}
    for split in ("train", "val"):
        maps = {
            label: {
                (int(row["epoch"]), int(row["trainer/global_step"])): row
                for row in runs[label][split]
                if row["trainer/global_step"] <= 30000 and row["full_epoch"]
            }
            for label in LABELS
        }
        keys = sorted(set(maps[LABELS[0]]) & set(maps[LABELS[1]]))
        aligned = {
            "count": len(keys), "first_key": list(keys[0]), "last_key": list(keys[-1]),
            "rows": [{label: mapping[key] for label, mapping in maps.items()} for key in keys],
        }
        require_equal(aligned, expected["aligned"][split], f"aligned.{split}")
        window = [row for row in aligned["rows"]
                  if 25000 <= row[LABELS[0]]["trainer/global_step"] <= 30000]
        metrics = {}
        for metric in WINDOW_METRICS:
            means = {
                label: statistics.mean(row[label][metric] for row in window)
                for label in ("d64_original", "d64_no_edge_feature")
                if all(metric in row[label] for row in window)
            }
            if len(means) == 2:
                means["relative_change_pct"] = (
                    (means[LABELS[0]] / means[LABELS[1]] - 1) * 100
                    if means[LABELS[1]] else None
                )
                metrics[metric] = means
        comparison = {
            "n_epochs": len(window),
            "steps": [row[LABELS[0]]["trainer/global_step"] for row in window],
            "metrics": metrics,
        }
        require_equal(comparison, expected["comparison_25k_30k"][split], f"window.{split}")
        result["aligned"][split] = {key: value for key, value in aligned.items() if key != "rows"}
        result["comparison_25k_30k"][split] = comparison

    zero_check = {
        split: {
            "rows": len(values),
            "max_abs_weighted_edge_feature": max(abs(row["weighted_edge_feature"]) for row in values),
            "raw_edge_feature_min": min(row["edge_feature"] for row in values),
            "raw_edge_feature_max": max(row["edge_feature"] for row in values),
        }
        for split, values in runs[LABELS[0]].items()
    }
    require_equal(zero_check, expected["no_edge_feature_zero_check"], "zero_check")
    if any(value["max_abs_weighted_edge_feature"] != 0 for value in zero_check.values()):
        raise ValueError("Disabled feature objective was nonzero")
    result["zero_check"] = zero_check
    result["checkpoint_rechecked"] = (
        check_checkpoint(args.checkpoint, audit["checkpoint"])
        if args.checkpoint is not None else False
    )
    serialized = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.output is not None:
        with args.output.open("x") as stream:
            stream.write(serialized)
    print(json.dumps({
        "matches_audit": True, "aligned": result["aligned"], "zero_check": zero_check,
        "window_val_relative_change_pct": {
            key: value["relative_change_pct"]
            for key, value in result["comparison_25k_30k"]["val"]["metrics"].items()
        },
        "checkpoint_rechecked": result["checkpoint_rechecked"],
    }, indent=2))


if __name__ == "__main__":
    main()
