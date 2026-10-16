"""Evaluate a strictly loaded intersection VAE with an explicit input contract.

This runner writes only the new method. Legacy exports and their original
protocol remain read-only references. Explicit standalone posterior inference
needs only canonical CAD inputs and a verified frozen legacy encoder; GT is
never used to select decoder edges.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from tools.vae_compare_inference import (
    REPOSITORY, atomic_npz, claim_protocol, configure_fp32, file_sha256,
    json_digest, latent_sha256, load_identity_geometry, normalized_weights,
    read_ids, safe_checkpoint, training_split_provenance,
)


PROTOCOL_VERSION = "intersection_evaluate_inference_v1"
ENCODER_COMPONENTS = {
    "face_coords", "edge_coords", "face_attn_proj_in", "face_attn",
    "face_attn_proj_out", "graph_face_edge", "global_feature1", "global_feature2",
}
ENCODER_CONFIG_KEYS = (
    "in_channels", "latent_channels", "hidden_channels", "norm", "gaussian_weights",
    "sigmoid", "num_gat_layers", "bottleneck_dim", "num_encoder_layers", "nhead",
    "primitive_blocks_per_stage", "kl_reduction",
)
MEAN_AUDIT_COUNT = 3
# The unchanged legacy encoder itself varied by up to 4.29e-6 versus its
# archived CUDA FP32 mean (24 IDs x 5 repeats); CPU old/new paths were exact.
# This tolerance applies only to the numerical sanity check. Weight equality,
# reference hashes and the replayed input arrays remain exact requirements.
MEAN_ATOL = 1e-5
MEAN_RTOL = 1e-5


def legacy_configuration(checkpoint: dict) -> dict:
    from src.brepnet.models.vae import AutoEncoder_light

    historical = checkpoint.get("hyper_parameters", {}).get("model", {})
    if historical.get("name") not in {"AutoEncoder_1119_light", "AutoEncoder_light"}:
        raise ValueError("Reference checkpoint must identify the original light VAE")
    config = dict(AutoEncoder_light.DEFAULTS)
    config.update(norm="layer", in_channels=6)
    aliases = {"dim_shape": "hidden_channels", "dim_latent": "latent_channels",
               "encoder_decoder_sigmoid": "sigmoid"}
    for name, value in historical.items():
        name = aliases.get(name, name)
        if name in config:
            config[name] = value
    config.update(checkpoint=None, mode="frozen_inference")
    return config


def audit_encoder_state(new_weights: dict, legacy_weights: dict,
                        new_config: dict, legacy_config: dict) -> dict:
    """Compare renamed complete encoder/posterior state and nonweight options."""
    from src.brepnet.models.vae import AutoEncoder_light

    mapped = {}
    for name, value in new_weights.items():
        if name.startswith("encoder."):
            key = name.removeprefix("encoder.")
        elif name.startswith("posterior.projection."):
            key = "gaussian_proj." + name.removeprefix("posterior.projection.")
        elif name.startswith("posterior."):
            raise ValueError(f"Unknown posterior state component {name}")
        else:
            continue
        if key in mapped:
            raise ValueError(f"Ambiguous encoder mapping {key}")
        mapped[key] = value
    reference = {
        name: value for name, value in legacy_weights.items()
        if name.split(".")[0] in ENCODER_COMPONENTS | {"gaussian_proj"}
    }
    if not reference or set(reference) != set(mapped):
        raise ValueError("Encoder/posterior key sets differ from the reference")
    different = [key for key in reference if not torch.equal(mapped[key], reference[key])]
    if different:
        raise ValueError(f"Encoder/posterior weights differ: {different}")
    resolved = {**AutoEncoder_light.DEFAULTS, **new_config}
    for key in ENCODER_CONFIG_KEYS:
        if resolved.get(key) != legacy_config.get(key):
            raise ValueError(f"Encoder/posterior configuration differs: {key}")
    return {"encoder_equal_tensor_count": len(reference), "encoder_equal": True,
            "posterior_config": {key: resolved[key] for key in ENCODER_CONFIG_KEYS}}


def load_intersection(checkpoint_path: Path, legacy_path: Path, device: torch.device,
                      ids: list[str], *, input_source: str = "reference_latents",
                      require_legacy_encoder: bool = False) -> tuple[torch.nn.Module, dict]:
    from omegaconf import OmegaConf
    from src.brepnet.models.vae import AutoEncoder_intersection

    checkpoint, legacy_checkpoint = safe_checkpoint(checkpoint_path), safe_checkpoint(legacy_path)
    config = checkpoint.get("hyper_parameters", {}).get("model")
    if config is None or config.get("name") != "AutoEncoder_intersection":
        raise ValueError("New checkpoint must record AutoEncoder_intersection model configuration")
    config = OmegaConf.to_container(OmegaConf.create(config), resolve=True)
    if input_source not in {"reference_latents", "checkpoint_posterior"}:
        raise ValueError(f"Unknown input source: {input_source}")
    if (input_source == "reference_latents" or require_legacy_encoder) and not config.get("freeze_encoder", False):
        raise ValueError("This fixed posterior replay requires a frozen encoder checkpoint")
    if float(config.get("edge_threshold", -1)) != .5:
        raise ValueError("The fixed protocol requires the recorded edge threshold to be 0.5")
    new_weights, legacy_weights = normalized_weights(checkpoint), normalized_weights(legacy_checkpoint)
    legacy_config = legacy_configuration(legacy_checkpoint)
    if input_source == "reference_latents" or require_legacy_encoder:
        audit = audit_encoder_state(new_weights, legacy_weights, config, legacy_config)
    else:
        audit = {"input_source": input_source, "encoder_equal": None,
                 "encoder_equal_tensor_count": None,
                 "note": "Encode each canonical CAD using this checkpoint; old means are not decoder inputs."}
    audit.update(strict_load=True, checkpoint_model_config=config, legacy_config=legacy_config)
    for name, source, weights in (("intersection", checkpoint, new_weights),
                                  ("legacy", legacy_checkpoint, legacy_weights)):
        audit[name] = {
            "epoch": source.get("epoch"), "global_step": source.get("global_step"),
            "state_tensor_count": len(weights),
            "training_split": training_split_provenance(source, ids),
        }
    runtime_config = {**config, "checkpoint": None, "init_legacy_checkpoint": None,
                      "init_refinement_checkpoint": None,
                      "init_curve_feedback_checkpoint": None,
                      "init_encoder_edge_value_checkpoint": None,
                      "restoring_checkpoint": True, "mode": "frozen_inference"}
    with torch.device("meta"):
        model = AutoEncoder_intersection(runtime_config)
    model.load_state_dict(new_weights, strict=True, assign=True)
    model = model.float().eval().to(device)
    model.requires_grad_(False)
    return model, OmegaConf.to_container(OmegaConf.create(audit), resolve=True)


def encode_identity_mean(model: torch.nn.Module, data_root: Path, model_id: str,
                         device: torch.device) -> np.ndarray:
    """Encode the canonical global samples; topology is available only to encoder."""
    batch, _ = load_identity_geometry(data_root / model_id / "data.npz", model_id, device)
    features = model.encode(batch, v_test=True)["face_features"]
    values = model.sample(features, v_is_test=True)[0].detach().cpu().numpy()
    if values.dtype != np.float32 or values.shape != (len(batch["face_points"]), 32):
        raise ValueError("Checkpoint posterior mean must preserve canonical faces as float32 [F,32]")
    if not np.isfinite(values).all():
        raise ValueError("Checkpoint posterior mean contains nonfinite values")
    return values


def read_standalone_input(model: torch.nn.Module, data_root: Path, model_id: str,
                          device: torch.device, checkpoint_digest: str,
                          output_path: Path, resume: bool) -> tuple[np.ndarray, dict]:
    """Bind canonical CAD and stored/actual means without a reference export."""
    source = data_root / model_id / "data.npz"
    source_digest = file_sha256(source)
    with np.load(source, allow_pickle=False) as arrays:
        faces = arrays["sample_points_faces"]
        if faces.ndim != 4 or faces.shape[1:] != (16, 16, 6) or not len(faces):
            raise ValueError("Standalone source must contain nonempty [F,16,16,6] canonical faces")
        expected_shape = (len(faces), 32)
    if resume and output_path.exists():
        # CUDA re-encoding need not be bit-identical. A resumed output must keep
        # its original means and match current source/checkpoint provenance below.
        with np.load(output_path, allow_pickle=False) as saved:
            latents = saved["face_latents"].copy()
            recorded_digest = _scalar_text(saved, "input_latent_sha256")
        if latent_sha256(latents) != recorded_digest:
            raise ValueError("Existing checkpoint mean has an invalid latent digest")
    else:
        latents = encode_identity_mean(model, data_root, model_id, device)
    if latents.dtype != np.float32 or latents.shape != expected_shape or not np.isfinite(latents).all():
        raise ValueError("Standalone posterior mean changed canonical shape/dtype or contains nonfinite values")
    if file_sha256(source) != source_digest:
        raise ValueError("Canonical CAD source changed during posterior inference")
    return latents, {
        "input_source_path": str(source.resolve()), "input_source_sha256": source_digest,
        "input_source_kind": "checkpoint_posterior_mean",
        "encoder_checkpoint_sha256": checkpoint_digest,
        "input_latent_sha256": latent_sha256(latents),
    }


def read_reference_protocol(reference_root: Path, mode: str, split_hash: str,
                            intended_count: int, legacy_hash: str) -> tuple[dict, str]:
    path = reference_root / "metadata" / "inference" / mode / "protocol.json"
    protocol = json.loads(path.read_text())
    expected = {
        "version": "vae_compare_inference_v1", "mode": mode,
        "split_sha256": split_hash, "intended_count": intended_count,
        "legacy_checkpoint_sha256": legacy_hash,
        "identity_view": mode == "posterior_mean",
        "decoder_gt_topology": False, "decoder_face_deduplication": False,
        "legacy_threshold": .5,
    }
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise ValueError(f"Reference protocol mismatch for {key}")
    numeric = protocol.get("numeric_settings", {})
    for key, value in {"dtype": "float32", "autocast": False,
                       "matmul_allow_tf32": False, "cudnn_allow_tf32": False}.items():
        if numeric.get(key) != value:
            raise ValueError(f"Reference numeric protocol mismatch for {key}")
    return protocol, json_digest(protocol)


def _scalar_text(arrays: dict, key: str) -> str:
    value = np.asarray(arrays[key])
    if value.ndim != 0:
        raise ValueError(f"Expected scalar provenance field {key}")
    return str(value.item())


def read_reference_input(reference_root: Path, data_root: Path, model_id: str,
                         mode: str, reference_protocol: dict, reference_digest: str) -> tuple[np.ndarray, dict]:
    path = reference_root / "raw_predictions" / mode / "legacy" / model_id / "data.npz"
    # Read every field to trigger ZIP CRC validation, not merely the latent field.
    with np.load(path, allow_pickle=False) as source:
        arrays = {name: source[name] for name in source.files}
    if _scalar_text(arrays, "protocol_sha256") != reference_digest:
        raise ValueError("Reference raw protocol digest differs from its original protocol")
    if _scalar_text(arrays, "comparison_mode") != mode:
        raise ValueError("Reference raw input mode mismatch")
    if _scalar_text(arrays, "representation") != "legacy_half_edges":
        raise ValueError("Reference output must be legacy_half_edges")
    for field in ("pred_face", "pred_edge", "pred_edge_face_connectivity"):
        if field not in arrays:
            raise ValueError(f"Reference raw output missing {field}")
    latents = arrays["face_latents"]
    if latents.dtype != np.float32 or latents.ndim != 2 or latents.shape[1] != 32 or not len(latents):
        raise ValueError("Reference latent must be a nonempty float32 [F,32] array")
    if not np.isfinite(latents).all():
        raise ValueError("Reference latent contains nonfinite values")
    input_digest = latent_sha256(latents)
    if input_digest != _scalar_text(arrays, "input_latent_sha256"):
        raise ValueError("Reference latent digest mismatch")
    if mode == "posterior_mean":
        original_source = data_root / model_id / "data.npz"
    else:
        latent_root = reference_protocol.get("latent_root")
        if not latent_root:
            raise ValueError("Archived reference protocol must identify its original latent_root")
        original_source = Path(latent_root) / model_id / "data.npz"
    source_digest = file_sha256(original_source)
    if source_digest != _scalar_text(arrays, "input_source_sha256"):
        raise ValueError("Original source file has changed since reference inference")
    if mode == "archived_latents":
        with np.load(original_source, allow_pickle=False) as original:
            if not np.array_equal(original["face_latents"], latents):
                raise ValueError("Archived reference latent differs from its original source array")
    provenance = {
        "reference_raw_path": str(path.resolve()), "reference_raw_sha256": file_sha256(path),
        "reference_protocol_sha256": reference_digest,
        "input_source_path": str(original_source.resolve()), "input_source_sha256": source_digest,
        "input_latent_sha256": input_digest,
    }
    return latents.copy(), provenance


def validate_prediction(prediction: dict, latents: np.ndarray) -> None:
    if not np.array_equal(prediction["face_latents"], latents):
        raise ValueError("Decoder changed the input latent set/order")
    if _scalar_text(prediction, "representation") != "legacy_half_edges":
        raise ValueError("Intersection output must preserve directed half edges")
    faces, edges = prediction["pred_face"], prediction["pred_edge"]
    if faces.shape != (len(latents), 16, 16, 3):
        raise ValueError("Decoder changed the input face count or output grid")
    if edges.ndim != 3 or edges.shape[1:] != (16, 3):
        raise ValueError("Decoder curves must have shape [E,16,3]")
    if not np.isfinite(faces).all() or not np.isfinite(edges).all():
        raise ValueError("Decoder produced nonfinite geometry")
    conn = prediction["pred_edge_face_connectivity"]
    if conn.shape != (len(edges), 3) or not np.issubdtype(conn.dtype, np.integer):
        raise ValueError("Decoder connectivity must be integer [E,3]")
    if not np.array_equal(np.sort(conn[:, 0]), np.arange(len(edges))):
        raise ValueError("Decoder connectivity does not cover output edges exactly once")
    if (conn[:, 1:] < 0).any() or (conn[:, 1:] >= len(faces)).any():
        raise ValueError("Decoder connectivity has invalid face indices")


def completed_prediction(path: Path, protocol_digest: str, provenance: dict,
                         latents: np.ndarray, mode: str) -> bool:
    if not path.exists():
        return False
    with np.load(path, allow_pickle=False) as source:
        arrays = {name: source[name] for name in source.files}
    expected = {**provenance, "protocol_sha256": protocol_digest, "comparison_mode": mode}
    for key, value in expected.items():
        if _scalar_text(arrays, key) != value:
            raise ValueError(f"Existing prediction provenance mismatch: {key}")
    if latent_sha256(arrays["face_latents"]) != provenance["input_latent_sha256"]:
        raise ValueError("Existing prediction latent digest mismatch")
    validate_prediction(arrays, latents)
    return True


def audit_identity_means(model: torch.nn.Module, ids: list[str], reference_root: Path,
                         data_root: Path, reference_protocol: dict, reference_digest: str,
                         device: torch.device) -> dict:
    rows = []
    with torch.inference_mode():
        for model_id in ids[:MEAN_AUDIT_COUNT]:
            expected, provenance = read_reference_input(
                reference_root, data_root, model_id, "posterior_mean", reference_protocol, reference_digest
            )
            batch, _ = load_identity_geometry(data_root / model_id / "data.npz", model_id, device)
            features = model.encode(batch, v_test=True)["face_features"]
            actual = model.sample(features, v_is_test=True)[0].detach().cpu().numpy()
            if actual.shape != expected.shape:
                raise ValueError(f"Identity posterior mean shape mismatch for {model_id}")
            difference = np.abs(actual.astype(np.float64) - expected)
            matches = bool(np.allclose(actual, expected, atol=MEAN_ATOL, rtol=MEAN_RTOL))
            rows.append({"model_id": model_id, "max_abs_error": float(difference.max()),
                         "exact_equal": bool(np.array_equal(actual, expected)), "within_tolerance": matches,
                         "reference_raw_sha256": provenance["reference_raw_sha256"]})
    return {"atol": MEAN_ATOL, "rtol": MEAN_RTOL, "selection": "first three shard IDs",
            "count": len(rows), "passed": all(row["within_tolerance"] for row in rows),
            "max_abs_error": max((row["max_abs_error"] for row in rows), default=0.0), "samples": rows}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", "--new-checkpoint", dest="checkpoint", type=Path, required=True)
    for name in ("legacy-checkpoint", "data-root", "split-list", "output-root"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path)
    parser.add_argument("--standalone", action="store_true",
                        help="Encode canonical CAD without reference exports; require a frozen legacy-equal encoder")
    parser.add_argument("--mode", choices=("posterior_mean", "archived_latents"), required=True)
    parser.add_argument("--method", choices=("intersection", "local", "conditioned", "point_feedback",
                                            "encoder_edge_value", "encoder_gat_control"), default="intersection")
    parser.add_argument("--input-source", choices=("reference_latents", "checkpoint_posterior"),
                        default="reference_latents")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    return parser


def validate_method(model: torch.nn.Module, method: str) -> None:
    """Keep experimental identity explicit while using the same strict loader."""
    kind = model.decoder.intersection_kind
    conditioned = model.decoder.curve_conditioning_enabled
    feedback = model.decoder.curve_point_feedback_enabled
    expected = {"intersection": ("legacy", False, False), "local": ("local_attention", False, False),
                "conditioned": ("legacy", True, False), "point_feedback": ("legacy", True, True),
                "encoder_edge_value": ("legacy", True, False), "encoder_gat_control": ("legacy", True, False)}
    if method not in expected or (kind, conditioned, feedback) != expected[method]:
        raise ValueError(f"Method {method} does not match checkpoint intersection={kind}, "
                         f"curve_conditioning={conditioned}, curve_point_feedback={feedback}")
    if method in {"conditioned", "point_feedback", "encoder_edge_value", "encoder_gat_control"} and not model.decoder.refinement_enabled:
        raise ValueError(f"Method {method} requires enabled refinement")
    has_edge_values = any("value_net" in name for name, _ in model.encoder.named_parameters())
    if has_edge_values != (method == "encoder_edge_value"):
        raise ValueError(f"Method {method} does not match encoder edge-value branch={has_edge_values}")
    if method in {"encoder_edge_value", "encoder_gat_control"} and model.freeze_encoder:
        raise ValueError(f"Method {method} requires a checkpoint trained with an unfrozen encoder")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    fresh_posterior = args.input_source == "checkpoint_posterior"
    if fresh_posterior and args.mode != "posterior_mean":
        raise ValueError("Checkpoint posterior input only supports posterior_mean, never archived latents")
    if args.standalone:
        if not fresh_posterior:
            raise ValueError("Standalone inference requires --input-source checkpoint_posterior")
        if args.reference_root is not None:
            raise ValueError("Standalone inference must not provide --reference-root")
    elif args.reference_root is None:
        raise ValueError("Reference inference requires --reference-root; use --standalone explicitly")
    if args.method in {"encoder_edge_value", "encoder_gat_control"} and not fresh_posterior:
        raise ValueError("Changed encoder methods require --input-source checkpoint_posterior")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Require 0 <= shard-index < num-shards")
    if args.reference_root is not None and args.output_root.resolve() == args.reference_root.resolve():
        raise ValueError("New outputs must not use the historical reference root")
    ids = read_ids(args.split_list)
    shard_ids = ids[args.shard_index::args.num_shards]
    split_digest, legacy_digest = file_sha256(args.split_list), file_sha256(args.legacy_checkpoint)
    if not args.standalone:
        reference, reference_digest = read_reference_protocol(
            args.reference_root, args.mode, split_digest, len(ids), legacy_digest
        )
        posterior_reference, posterior_digest = read_reference_protocol(
            args.reference_root, "posterior_mean", split_digest, len(ids), legacy_digest
        )
    numeric = configure_fp32()
    device = torch.device(args.device)
    model, model_audit = load_intersection(args.checkpoint, args.legacy_checkpoint, device, ids,
                                         input_source=args.input_source,
                                         require_legacy_encoder=args.standalone)
    validate_method(model, args.method)
    source_files = ["tools/intersection_evaluate_inference.py", "tools/vae_compare_inference.py",
                    "src/brepnet/models/vae.py", "src/brepnet/models/blocks.py",
                    "src/brepnet/dataset.py",
                    "src/brepnet/data/shared_boundaries.py"]
    protocol = {
        "version": PROTOCOL_VERSION, "mode": args.mode, "method": args.method,
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": file_sha256(args.checkpoint),
        "legacy_checkpoint": str(args.legacy_checkpoint.resolve()), "legacy_checkpoint_sha256": legacy_digest,
        "data_root": str(args.data_root.resolve()), "split_list": str(args.split_list.resolve()),
        "split_sha256": split_digest, "intended_count": len(ids), "num_shards": args.num_shards,
        "numeric_settings": numeric, "model_audit": model_audit,
        "decoder_gt_topology": False, "decoder_face_deduplication": False,
        "edge_threshold": .5, "identity_view": args.mode == "posterior_mean",
        "posterior_audit": {"count_per_shard": MEAN_AUDIT_COUNT, "atol": MEAN_ATOL, "rtol": MEAN_RTOL},
        "source_sha256": {name: file_sha256(REPOSITORY / name) for name in source_files},
    }
    if not args.standalone:
        protocol.update(
            reference_root=str(args.reference_root.resolve()),
            reference_protocol_sha256=reference_digest,
            reference_protocol_file_sha256=file_sha256(args.reference_root / "metadata/inference" / args.mode / "protocol.json"),
            posterior_audit_protocol_sha256=posterior_digest,
        )
    if fresh_posterior:
        protocol.update(version="intersection_evaluate_inference_v2_checkpoint_posterior",
                        input_source="checkpoint_posterior", input_contract="same_geometry_source",
                        encoder_gt_topology=True, canonical_face_order=True,
                        posterior_audit={"kind": "finite_float32_canonical_face_count_every_object",
                                         "reference_latent_equality_required": False})
    if args.standalone:
        protocol.update(version="intersection_evaluate_inference_v3_standalone_posterior",
                        input_contract="standalone_geometry_source", standalone=True,
                        frozen_legacy_encoder_required=True)
    metadata = args.output_root / "metadata/inference" / args.mode
    protocol_digest = claim_protocol(metadata / "protocol.json", protocol)
    status_path = metadata / f"shard-{args.shard_index:03d}.jsonl"
    if status_path.exists() and not args.resume:
        raise FileExistsError(f"Shard already exists: {status_path}; use --resume")
    invocation = {
        "argv": sys.argv if argv is None else argv, "device": str(device), "shard_ids": shard_ids,
        "protocol_sha256": protocol_digest, "resume": args.resume,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], cwd=REPOSITORY, text=True),
    }
    run_id = time.time_ns()
    if args.standalone:
        diff_path = metadata / f"shard-{args.shard_index:03d}-git-{run_id}.diff"
        diff_path.write_text(subprocess.check_output(["git", "diff", "--binary"], cwd=REPOSITORY, text=True))
        invocation["git_diff_path"] = str(diff_path.resolve())
    (metadata / f"shard-{args.shard_index:03d}-invocation-{run_id}.json").write_text(json.dumps(invocation, indent=2) + "\n")
    try:
        mean_audit = ({"passed": True, "max_abs_error": None,
                       "note": "Each object uses its own checkpoint mean; old mean equality is not applicable."}
                      if fresh_posterior else audit_identity_means(
                          model, shard_ids, args.reference_root, args.data_root,
                          posterior_reference, posterior_digest, device))
    except Exception as error:
        mean_audit = {"passed": False, "error": f"{type(error).__name__}: {error}"}
    (metadata / f"shard-{args.shard_index:03d}-mean-audit-{run_id}.json").write_text(json.dumps(mean_audit, indent=2) + "\n")
    if not mean_audit["passed"]:
        with status_path.open("a" if args.resume else "x") as stream:
            for model_id in shard_ids:
                stream.write(json.dumps({"model_id": model_id, "mode": args.mode, "status": "failed",
                                         "stage": "posterior_audit", "error": mean_audit}) + "\n")
        raise ValueError("Identity posterior audit failed; refusing to label old latents as this encoder's posterior")
    print(json.dumps({"ready": True, "mode": args.mode, "shard": args.shard_index,
                      "count": len(shard_ids), "encoder_equal_tensor_count": model_audit["encoder_equal_tensor_count"],
                      "posterior_mean_max_abs_error": mean_audit["max_abs_error"]}), flush=True)
    failures = 0
    with status_path.open("a" if args.resume else "x") as stream, torch.inference_mode():
        for index, model_id in enumerate(shard_ids):
            started = time.monotonic()
            row = {"model_id": model_id, "mode": args.mode, "method": args.method,
                   "shard": args.shard_index, "status": "failed"}
            path = args.output_root / "raw_predictions" / args.mode / args.method / model_id / "data.npz"
            try:
                if args.standalone:
                    latents, provenance = read_standalone_input(
                        model, args.data_root, model_id, device, protocol["checkpoint_sha256"], path, args.resume
                    )
                else:
                    latents, provenance = read_reference_input(
                        args.reference_root, args.data_root, model_id, args.mode, reference, reference_digest
                    )
                if fresh_posterior and not args.standalone:
                    provenance.update(reference_input_latent_sha256=provenance.pop("input_latent_sha256"),
                                      input_source_kind="checkpoint_posterior_mean",
                                      encoder_checkpoint_sha256=protocol["checkpoint_sha256"])
                    if args.resume and path.exists():
                        # Stored means are immutable decoder inputs. Re-encoding on
                        # CUDA is not required to be bit-identical across invocations.
                        with np.load(path, allow_pickle=False) as saved:
                            saved_latents = saved["face_latents"].copy()
                            recorded_digest = _scalar_text(saved, "input_latent_sha256")
                        if saved_latents.shape != latents.shape or saved_latents.dtype != np.float32:
                            raise ValueError("Existing checkpoint mean changed canonical shape/dtype")
                        if latent_sha256(saved_latents) != recorded_digest or not np.isfinite(saved_latents).all():
                            raise ValueError("Existing checkpoint mean has an invalid latent digest")
                        latents = saved_latents
                    else:
                        latents = encode_identity_mean(model, args.data_root, model_id, device)
                    provenance["input_latent_sha256"] = latent_sha256(latents)
                row.update(provenance)
                if path.exists() and not args.resume:
                    raise FileExistsError("Prediction already exists; use --resume or a new output root")
                if args.resume and completed_prediction(path, protocol_digest, provenance, latents, args.mode):
                    row["status"] = "resumed_skip"
                else:
                    prediction = model.decode_latents(torch.from_numpy(latents).to(device))
                    validate_prediction(prediction, latents)
                    prediction.update({key: np.asarray(value) for key, value in provenance.items()})
                    prediction.update(protocol_sha256=np.asarray(protocol_digest), comparison_mode=np.asarray(args.mode))
                    atomic_npz(path, prediction)
                    row.update(status="ok", faces=len(prediction["pred_face"]), edges=len(prediction["pred_edge"]))
            except Exception as error:
                row["error"] = f"{type(error).__name__}: {error}"
            row["seconds"] = time.monotonic() - started
            failures += int(row["status"] == "failed")
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            if (index + 1) % 25 == 0 or index + 1 == len(shard_ids):
                print(json.dumps({"mode": args.mode, "shard": args.shard_index,
                                  "processed": index + 1, "total": len(shard_ids), "failures": failures}), flush=True)
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
