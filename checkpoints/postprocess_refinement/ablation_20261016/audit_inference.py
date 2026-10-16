"""Read-only independent coverage/provenance audit of the frozen decoder pair."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(REPOSITORY))

import numpy as np
from omegaconf import OmegaConf

from infer_pairs import (
    CONDITIONED_SHA256, LEGACY_SHA256, METHODS, reference_contract,
    reference_input, scalar_text,
)
from tools.intersection_evaluate_inference import audit_encoder_state, legacy_configuration, validate_prediction
from tools.vae_compare_inference import file_sha256, json_digest, normalized_weights, read_ids, safe_checkpoint

SPLITS = {"validation": ("val", 3495, 3505), "test": ("test", 2417, 2424)}


def read_json(path):
    return json.loads(path.read_text())


def source_audit(protocol, launch, completion, stage_root):
    checks = {}
    for name, digest in protocol["source_sha256"].items():
        checks[name] = {"actual_matches_protocol": file_sha256(REPOSITORY / name) == digest}
    for name, digest in launch["source_sha256"].items():
        checks.setdefault(name, {}).update(
            actual_matches_launch=file_sha256(REPOSITORY / name) == digest,
            archived_source_matches_launch=file_sha256(stage_root / "source" / name) == digest,
            driver_completion_unchanged=completion["source_unchanged"].get(name) is True,
        )
    return {"passed": all(all(row.values()) for row in checks.values()), "files": checks}


def checkpoint_audit(protocol):
    paths = {"legacy": Path(protocol["legacy_checkpoint"]), "conditioned": Path(protocol["conditioned_checkpoint"])}
    actual_hashes = {method: file_sha256(path) for method, path in paths.items()}
    expected = {"legacy": LEGACY_SHA256, "conditioned": CONDITIONED_SHA256}
    if actual_hashes != expected:
        raise ValueError("Actual checkpoint hashes differ from the frozen plan")
    old, new = (safe_checkpoint(paths[method]) for method in METHODS)
    config = OmegaConf.to_container(OmegaConf.create(new["hyper_parameters"]["model"]), resolve=True)
    audit = audit_encoder_state(normalized_weights(new), normalized_weights(old), config, legacy_configuration(old))
    if audit["encoder_equal_tensor_count"] != 243:
        raise ValueError("Unexpected number of encoder/posterior tensors")
    return {"passed": True, "actual_sha256": actual_hashes, **audit,
            "legacy_epoch": old.get("epoch"), "legacy_global_step": old.get("global_step"),
            "conditioned_epoch": new.get("epoch"), "conditioned_global_step": new.get("global_step")}


def split_overlap_audit():
    lists, hashes = {}, {}
    for name, prefix in (("train", "train"), ("validation", "val"), ("test", "test")):
        path = REPOSITORY / f"checkpoints/joint_boundary/validation/{prefix}_contract_audit/accepted.txt"
        lists[name] = set(read_ids(path))
        hashes[name] = file_sha256(path)
    pairs = (("train", "validation"), ("train", "test"), ("validation", "test"))
    overlap = {f"{a}_and_{b}": len(lists[a] & lists[b]) for a, b in pairs}
    return {"passed": not any(overlap.values()), "counts": {k: len(v) for k, v in lists.items()},
            "sha256": hashes, "intersection_counts": overlap,
            "scope": "Current accepted lists only; this does not independently certify historical training-list contents."}


def audit_split(root, split):
    prefix, expected_count, original_count = SPLITS[split]
    split_path = REPOSITORY / f"checkpoints/joint_boundary/validation/{prefix}_contract_audit/accepted.txt"
    ids = read_ids(split_path)
    if len(ids) != expected_count:
        raise ValueError("Unexpected accepted split count")
    destination = root / split
    metadata = destination / "metadata/inference/posterior_mean"
    protocol_path = metadata / "protocol.json"
    protocol = read_json(protocol_path)
    protocol_hash = json_digest(protocol)
    stage_root = root / "metadata" / f"inference_{split}"
    launch, completion = (read_json(stage_root / name) for name in ("launch.json", "completion.json"))
    if len(completion["returncodes"]) != 8 or any(completion["returncodes"].values()):
        raise ValueError("All eight inference shards must have completed successfully")
    if (protocol["split_sha256"] != file_sha256(split_path) or protocol["intended_count"] != len(ids)
            or protocol["num_shards"] != 8 or protocol.get("smoke_subset") is not False
            or protocol["methods"] != list(METHODS) or protocol["decoder_gt_topology"] is not False
            or protocol["decoder_face_deduplication"] is not False
            or protocol["legacy_checkpoint_sha256"] != LEGACY_SHA256
            or protocol["conditioned_checkpoint_sha256"] != CONDITIONED_SHA256
            or protocol["model_audit"].get("strict_load") is not True
            or protocol["model_audit"].get("encoder_equal") is not True
            or protocol["model_audit"].get("encoder_equal_tensor_count") != 243):
        raise ValueError("Frozen inference protocol mismatch")
    reference_root, data_root = Path(protocol["reference_root"]), Path(protocol["data_root"])
    reference, reference_path, reference_file_hash = reference_contract(reference_root, ids, data_root, False)
    if (protocol["reference_protocol_file_sha256"] != reference_file_hash or
            protocol["reference_protocol_sha256"] != json_digest(reference)):
        raise ValueError("Reference protocol digest mismatch")
    frozen_sources = source_audit(protocol, launch, completion, stage_root)
    violations = Counter()
    rows, means = {}, []
    for shard in range(8):
        shard_rows = [json.loads(line) for line in (metadata / f"shard-{shard:03d}.jsonl").read_text().splitlines() if line]
        expected_ids = ids[shard::8]
        if [row["model_id"] for row in shard_rows] != expected_ids:
            violations["shard_ID_order_or_coverage"] += 1
        invocation = read_json(metadata / f"shard-{shard:03d}-invocation.json")
        if invocation["protocol_sha256"] != protocol_hash or invocation["shard_ids"] != expected_ids:
            violations["invocation_protocol_or_IDs"] += 1
        mean = read_json(metadata / f"shard-{shard:03d}-mean-audit.json")
        if (mean["passed"] is not True or [row["model_id"] for row in mean["samples"]] != expected_ids[:3]
                or mean["atol"] != 1e-5 or mean["rtol"] != 1e-5):
            violations["mean_audit_contract"] += 1
        for entry in mean["samples"]:
            if set(entry["methods"]) != set(METHODS):
                violations["mean_audit_methods"] += 1
            for value in entry["methods"].values():
                if value["passed"] is not True or not np.isfinite(value["max_abs_error"]):
                    violations["mean_audit_failed_or_nonfinite"] += 1
                means.append(value["max_abs_error"])
        for row in shard_rows:
            if row["model_id"] in rows:
                violations["duplicate_status_ID"] += 1
            rows[row["model_id"]] = row
    result = {"intended_count": len(ids), "original_count": original_count,
              "split_sha256": file_sha256(split_path), "protocol_sha256": protocol_hash,
              "protocol_file_sha256": file_sha256(protocol_path),
              "reference_protocol_file_sha256": reference_file_hash,
              "shards": {"count": 8, "all_returncodes_zero": True, "status_unique_ID_count": len(rows)},
              "mean_audits": {"object_method_checks": len(means), "maximum_absolute_difference": max(means, default=None)},
              "source_freeze": frozen_sources, "methods": {},
              "exact_common_reference_latent_count": 0, "exact_between_method_latent_count": 0}
    if set(rows) != set(ids):
        violations["global_status_ID_coverage"] += 1
    for method in METHODS:
        paths = list((destination / "raw_predictions/posterior_mean" / method).glob("*/data.npz"))
        raw_ids = [p.parent.name for p in paths]
        result["methods"][method] = {"raw_count": len(raw_ids), "unique_ID_count": len(set(raw_ids)),
                                      "missing_count": len(set(ids) - set(raw_ids)), "unexpected_count": len(set(raw_ids) - set(ids)),
                                      "valid_shape_finite_provenance_count": 0}
        if set(raw_ids) != set(ids) or len(raw_ids) != len(ids):
            violations["raw_ID_coverage"] += 1
    for index, model_id in enumerate(ids):
        try:
            expected, provenance = reference_input(reference_root, data_root, model_id, reference)
            status = rows[model_id]
            if status["status"] != "ok" or set(status["methods"]) != set(METHODS):
                raise ValueError("Incomplete method status")
            for field, value in provenance.items():
                if status.get(field) != value:
                    raise ValueError("Status provenance mismatch")
            pair = []
            for method in METHODS:
                path = destination / "raw_predictions/posterior_mean" / method / model_id / "data.npz"
                if status["methods"][method]["status"] != "ok" or file_sha256(path) != status["methods"][method]["raw_sha256"]:
                    raise ValueError("Actual raw hash or method status mismatch")
                with np.load(path, allow_pickle=False) as archive:
                    arrays = {key: archive[key] for key in archive.files}
                validate_prediction(arrays, expected)
                for value in arrays.values():
                    if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
                        raise ValueError("Nonfinite numeric output")
                checks = {**provenance, "protocol_sha256": protocol_hash, "method": method,
                          "comparison_mode": "posterior_mean",
                          "checkpoint_sha256": LEGACY_SHA256 if method == "legacy" else CONDITIONED_SHA256}
                for field, value in checks.items():
                    if scalar_text(arrays, field) != value:
                        raise ValueError("Raw provenance field mismatch")
                if arrays["pred_face_adj"].size != len(expected) ** 2:
                    raise ValueError("Adjacency size mismatch")
                pair.append(arrays["face_latents"])
                result["methods"][method]["valid_shape_finite_provenance_count"] += 1
            result["exact_between_method_latent_count"] += int(np.array_equal(*pair))
            result["exact_common_reference_latent_count"] += int(all(np.array_equal(a, expected) for a in pair))
        except Exception as error:
            violations[f"per_object_{type(error).__name__}"] += 1
        if (index + 1) % 500 == 0 or index + 1 == len(ids):
            print(json.dumps({"audit_split": split, "processed": index + 1, "total": len(ids), "violations": sum(violations.values())}), flush=True)
    if file_sha256(reference_path) != reference_file_hash:
        violations["reference_protocol_changed_during_audit"] += 1
    if not frozen_sources["passed"]:
        violations["source_freeze"] += 1
    result.update(passed=not violations, violations=dict(violations))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.run_root.resolve()
    # Require both completed runs before certifying a full two-split export.
    for split in SPLITS:
        if not (root / "metadata" / f"inference_{split}" / "completion.json").is_file():
            raise ValueError(f"Inference has not completed: {split}")
    initial_protocol = read_json(root / "validation/metadata/inference/posterior_mean/protocol.json")
    result = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "audit_script_sha256": file_sha256(Path(__file__)),
              "scope": "Read-only raw prediction and provenance audit; no postprocessing, CAD rendering, test failure inspection, or strategy changes.",
              "checkpoint_audit": checkpoint_audit(initial_protocol), "accepted_split_overlap": split_overlap_audit(),
              "splits": {split: audit_split(root, split) for split in SPLITS}}
    result["passed"] = result["checkpoint_audit"]["passed"] and result["accepted_split_overlap"]["passed"] and all(v["passed"] for v in result["splits"].values())
    output = root / "integrity/inference_integrity.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(output)
    print(json.dumps({"output": str(output), "passed": result["passed"], "counts": {s: v["exact_common_reference_latent_count"] for s, v in result["splits"].items()}}), flush=True)
    return int(not result["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
