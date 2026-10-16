"""Decode the frozen AE pair from identical, verified 2026-10-16 means."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

REPOSITORY = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY))

import numpy as np
import torch

from tools.intersection_evaluate_inference import (
    MEAN_ATOL, MEAN_RTOL, encode_identity_mean, legacy_configuration,
    load_intersection, validate_method, validate_prediction,
)
from tools.vae_compare_inference import (
    atomic_npz, claim_protocol, configure_fp32, decode_legacy, file_sha256,
    json_digest, latent_sha256, normalized_weights, read_ids, safe_checkpoint,
)

LEGACY_SHA256 = "8b714943d24c3f3f18f3a1e64222b1f3577323ddbf42e4b90d603186ab7e917a"
CONDITIONED_SHA256 = "4c7910b8169206c74fcf38c358cf171359149e0429c8debbc227d391aa56613f"
REFERENCE_SHA256 = "9b18509de2a21f50d8fad41b8c37695842fe2e92745dc779713b0ecac06e2107"
METHODS = ("legacy", "conditioned")


def scalar_text(arrays, key):
    value = np.asarray(arrays[key])
    if value.ndim != 0:
        raise ValueError(f"Expected scalar provenance: {key}")
    return str(value.item())


def reference_contract(root, ids, data_root, allow_subset):
    path = root / "metadata/inference/posterior_mean/protocol.json"
    protocol = json.loads(path.read_text())
    expected = {
        "version": "intersection_evaluate_inference_v3_standalone_posterior",
        "mode": "posterior_mean", "method": "intersection",
        "checkpoint_sha256": REFERENCE_SHA256,
        "legacy_checkpoint_sha256": LEGACY_SHA256,
        "decoder_gt_topology": False, "decoder_face_deduplication": False,
        "identity_view": True, "encoder_gt_topology": True,
        "frozen_legacy_encoder_required": True, "edge_threshold": .5,
    }
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise ValueError(f"Reference protocol mismatch: {key}")
    if Path(protocol["data_root"]).resolve() != data_root.resolve():
        raise ValueError("Reference canonical CAD root differs")
    numeric = protocol["numeric_settings"]
    for key, value in {"dtype": "float32", "autocast": False, "seed": 42,
                       "matmul_allow_tf32": False, "cudnn_allow_tf32": False}.items():
        if numeric.get(key) != value:
            raise ValueError(f"Reference numerical contract mismatch: {key}")
    audit = protocol["model_audit"]
    if audit.get("encoder_equal") is not True or audit.get("encoder_equal_tensor_count") != 243:
        raise ValueError("Reference must document the frozen, legacy-equal encoder/posterior")
    split = Path(protocol["split_list"])
    if file_sha256(split) != protocol["split_sha256"]:
        raise ValueError("Reference split changed")
    reference_ids = read_ids(split)
    if len(reference_ids) != protocol["intended_count"] or not set(ids) <= set(reference_ids):
        raise ValueError("Requested IDs are outside the reference split")
    if ids != reference_ids and not allow_subset:
        raise ValueError("A smoke subset requires --allow-reference-subset")
    return protocol, path, file_sha256(path)


def reference_input(root, data_root, model_id, protocol):
    path = root / "raw_predictions/posterior_mean/intersection" / model_id / "data.npz"
    raw_hash = file_sha256(path)
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}  # Validate every ZIP CRC.
    if file_sha256(path) != raw_hash:
        raise ValueError("Reference raw changed while reading")
    checks = {
        "protocol_sha256": json_digest(protocol), "comparison_mode": "posterior_mean",
        "representation": "legacy_half_edges", "input_source_kind": "checkpoint_posterior_mean",
        "encoder_checkpoint_sha256": protocol["checkpoint_sha256"],
    }
    for key, value in checks.items():
        if scalar_text(arrays, key) != value:
            raise ValueError(f"Reference raw provenance mismatch: {key}")
    source = data_root / model_id / "data.npz"
    if Path(scalar_text(arrays, "input_source_path")).resolve() != source.resolve():
        raise ValueError("Reference raw names a different source CAD")
    source_hash = file_sha256(source)
    if source_hash != scalar_text(arrays, "input_source_sha256"):
        raise ValueError("Canonical source CAD changed since reference inference")
    with np.load(source, allow_pickle=False) as archive:
        face_shape = archive["sample_points_faces"].shape
    latents = arrays["face_latents"]
    if (len(face_shape) != 4 or face_shape[1:] != (16, 16, 6) or
            latents.shape != (face_shape[0], 32) or not len(latents) or
            latents.dtype != np.float32 or not np.isfinite(latents).all()):
        raise ValueError("Reference means violate canonical float32 [F,32] geometry contract")
    digest = latent_sha256(latents)
    if digest != scalar_text(arrays, "input_latent_sha256"):
        raise ValueError("Reference latent hash mismatch")
    validate_prediction(arrays, latents)
    return latents.copy(), {
        "reference_raw_path": str(path.resolve()), "reference_raw_sha256": raw_hash,
        "reference_protocol_sha256": json_digest(protocol),
        "input_source_path": str(source.resolve()), "input_source_sha256": source_hash,
        "input_source_kind": "verified_common_posterior_mean",
        "encoder_checkpoint_sha256": LEGACY_SHA256, "input_latent_sha256": digest,
    }


def load_pair(legacy_path, conditioned_path, device, ids):
    from src.brepnet.models.vae import AutoEncoder_light

    if file_sha256(legacy_path) != LEGACY_SHA256 or file_sha256(conditioned_path) != CONDITIONED_SHA256:
        raise ValueError("Checkpoint hash differs from the frozen ablation plan")
    conditioned, audit = load_intersection(conditioned_path, legacy_path, device, ids,
                                           input_source="reference_latents")
    validate_method(conditioned, "conditioned")
    if audit["encoder_equal_tensor_count"] != 243:
        raise ValueError("Unexpected encoder/posterior tensor count")
    checkpoint = safe_checkpoint(legacy_path)
    config = legacy_configuration(checkpoint)
    with torch.device("meta"):
        legacy = AutoEncoder_light(config)
    legacy.load_state_dict(normalized_weights(checkpoint), strict=True, assign=True)
    legacy = legacy.float().eval().to(device)
    legacy.requires_grad_(False)
    return {"legacy": legacy, "conditioned": conditioned}, audit


def mean_audit(models, ids, reference_root, data_root, reference, device):
    rows = []
    for model_id in ids[:3]:
        expected, _ = reference_input(reference_root, data_root, model_id, reference)
        row = {"model_id": model_id, "methods": {}}
        for method, model in models.items():
            actual = encode_identity_mean(model, data_root, model_id, device)
            same_shape = actual.shape == expected.shape
            passed = same_shape and bool(np.allclose(actual, expected, atol=MEAN_ATOL, rtol=MEAN_RTOL))
            row["methods"][method] = {
                "passed": passed,
                "max_abs_error": float(np.abs(actual.astype(np.float64) - expected).max()) if same_shape else None,
            }
        rows.append(row)
    return {"selection": "first three IDs of each shard", "atol": MEAN_ATOL, "rtol": MEAN_RTOL,
            "actual_decoder_input": "unchanged reference mean; recomputed means are audit only",
            "passed": all(m["passed"] for row in rows for m in row["methods"].values()), "samples": rows}


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for name in ("legacy-checkpoint", "split-list", "reference-root", "output-root"):
        result.add_argument("--" + name, required=True, type=Path)
    result.add_argument("--checkpoint", "--conditioned-checkpoint", dest="conditioned_checkpoint", required=True, type=Path)
    result.add_argument("--data-root", type=Path, default=Path("/mnt/d/data/deepcad_v7"))
    result.add_argument("--shard-index", type=int, default=0)
    result.add_argument("--num-shards", type=int, default=1)
    result.add_argument("--device", default="cuda:0")
    result.add_argument("--allow-reference-subset", action="store_true", help="Explicit smoke-only subset")
    return result


def main():
    args = parser().parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Require 0 <= shard-index < num-shards")
    if args.output_root.resolve() == args.reference_root.resolve():
        raise ValueError("Outputs must not overwrite the reference run")
    ids = read_ids(args.split_list)
    shard_ids = ids[args.shard_index::args.num_shards]
    reference, reference_path, reference_file_hash = reference_contract(
        args.reference_root, ids, args.data_root, args.allow_reference_subset)
    metadata = args.output_root / "metadata/inference/posterior_mean"
    status_path = metadata / f"shard-{args.shard_index:03d}.jsonl"
    if status_path.exists():
        raise FileExistsError(f"Shard already started; use a fresh output root: {status_path}")
    for model_id in shard_ids:
        for method in METHODS:
            path = args.output_root / "raw_predictions/posterior_mean" / method / model_id / "data.npz"
            if path.exists():
                raise FileExistsError(f"Prediction already exists: {path}")
    numeric = configure_fp32()
    device = torch.device(args.device)
    models, audit = load_pair(args.legacy_checkpoint, args.conditioned_checkpoint, device, ids)
    sources = [Path(__file__), REPOSITORY / "tools/intersection_evaluate_inference.py",
               REPOSITORY / "tools/vae_compare_inference.py", REPOSITORY / "src/brepnet/models/vae.py",
               REPOSITORY / "src/brepnet/models/blocks.py", REPOSITORY / "src/brepnet/dataset.py",
               REPOSITORY / "src/brepnet/data/shared_boundaries.py"]
    protocol = {
        "version": "frozen_model_post_ablation_inference_v1", "mode": "posterior_mean",
        "methods": list(METHODS), "numeric_settings": numeric, "model_audit": audit,
        "legacy_checkpoint": str(args.legacy_checkpoint.resolve()), "legacy_checkpoint_sha256": LEGACY_SHA256,
        "conditioned_checkpoint": str(args.conditioned_checkpoint.resolve()), "conditioned_checkpoint_sha256": CONDITIONED_SHA256,
        "data_root": str(args.data_root.resolve()), "split_list": str(args.split_list.resolve()),
        "split_sha256": file_sha256(args.split_list), "intended_count": len(ids), "num_shards": args.num_shards,
        "reference_root": str(args.reference_root.resolve()), "reference_protocol_sha256": json_digest(reference),
        "reference_protocol_file_sha256": reference_file_hash, "reference_split_sha256": reference["split_sha256"],
        "smoke_subset": args.allow_reference_subset, "identity_view": True,
        "encoder_gt_topology": True, "decoder_gt_topology": False, "decoder_face_deduplication": False,
        "edge_threshold": .5, "input_contract": "exact_same_verified_reference_posterior_mean",
        "candidate_contract": {"legacy": "all N*N ordered pairs including self", "conditioned": "N*(N-1) ordered pairs, existing twin-logit averaging"},
        "source_sha256": {str(p.relative_to(REPOSITORY)): file_sha256(p) for p in sources},
    }
    protocol_hash = claim_protocol(metadata / "protocol.json", protocol)
    invocation = {"argv": sys.argv, "device": str(device), "shard_ids": shard_ids,
                  "protocol_sha256": protocol_hash,
                  "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True).strip(),
                  "git_status": subprocess.check_output(["git", "status", "--short"], cwd=REPOSITORY, text=True)}
    (metadata / f"shard-{args.shard_index:03d}-invocation.json").write_text(json.dumps(invocation, indent=2) + "\n")
    (metadata / f"shard-{args.shard_index:03d}-git.diff").write_text(subprocess.check_output(["git", "diff", "--binary"], cwd=REPOSITORY, text=True))
    failures = 0
    with status_path.open("x") as stream, torch.inference_mode():
        audit_path = metadata / f"shard-{args.shard_index:03d}-mean-audit.json"
        means = mean_audit(models, shard_ids, args.reference_root, args.data_root, reference, device)
        audit_path.write_text(json.dumps(means, indent=2) + "\n")
        if not means["passed"]:
            raise ValueError("Recomputed canonical means failed the numerical audit")
        print(json.dumps({"ready": True, "shard": args.shard_index, "count": len(shard_ids),
                          "encoder_equal_tensor_count": audit["encoder_equal_tensor_count"]}), flush=True)
        for index, model_id in enumerate(shard_ids):
            started = time.monotonic()
            row = {"model_id": model_id, "mode": "posterior_mean", "shard": args.shard_index, "methods": {}}
            try:
                values, provenance = reference_input(args.reference_root, args.data_root, model_id, reference)
            except Exception as error:
                row.update(status="failed", stage="input_provenance", error=f"{type(error).__name__}: {error}")
                stream.write(json.dumps(row) + "\n"); stream.flush()
                raise  # Never continue after input/hash conflicts.
            row.update(provenance)
            latent = torch.from_numpy(values).to(device)
            for method, model in models.items():
                path = args.output_root / "raw_predictions/posterior_mean" / method / model_id / "data.npz"
                try:
                    prediction = decode_legacy(model, latent) if method == "legacy" else model.decode_latents(latent)
                    validate_prediction(prediction, values)
                    prediction.update({key: np.asarray(value) for key, value in provenance.items()})
                    prediction.update(protocol_sha256=np.asarray(protocol_hash), comparison_mode=np.asarray("posterior_mean"),
                                      method=np.asarray(method), checkpoint_sha256=np.asarray(LEGACY_SHA256 if method == "legacy" else CONDITIONED_SHA256))
                    if path.exists():
                        raise FileExistsError(f"Refusing to overwrite prediction: {path}")
                    atomic_npz(path, prediction)
                    row["methods"][method] = {"status": "ok", "raw_sha256": file_sha256(path),
                                               "faces": len(prediction["pred_face"]), "edges": len(prediction["pred_edge"])}
                except FileExistsError:
                    raise
                except Exception as error:
                    row["methods"][method] = {"status": "failed", "error": f"{type(error).__name__}: {error}"}
                    failures += 1
            row.update(status="ok" if all(v["status"] == "ok" for v in row["methods"].values()) else "failed",
                       seconds=time.monotonic() - started)
            stream.write(json.dumps(row, allow_nan=False) + "\n"); stream.flush()
            if (index + 1) % 25 == 0 or index + 1 == len(shard_ids):
                print(json.dumps({"shard": args.shard_index, "processed": index + 1, "total": len(shard_ids), "failures": failures}), flush=True)
    if file_sha256(reference_path) != reference_file_hash:
        raise ValueError("Reference protocol changed during inference")
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
