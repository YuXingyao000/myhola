"""Replay identical latent sets through the frozen legacy and joint VAE decoders.

Run from the repository root with ``python -m tools.vae_compare_inference``.
The two modes deliberately never invoke a GT-conditioned decoder or face dedup.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time
import typing

import numpy as np
import torch


REPOSITORY = Path(__file__).resolve().parents[1]
PROTOCOL_VERSION = "vae_compare_inference_v1"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def latent_sha256(latents: np.ndarray) -> str:
    values = np.ascontiguousarray(latents, dtype=np.float32)
    digest = hashlib.sha256(str(values.shape).encode())
    digest.update(values.tobytes())
    return digest.hexdigest()


def read_ids(path: Path) -> list[str]:
    ids = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("split must be nonempty with unique IDs")
    if any(Path(item).name != item or item in {".", ".."} for item in ids):
        raise ValueError("split entries must be individual IDs, not paths")
    return ids


def safe_checkpoint(path: Path) -> dict:
    """Allow known OmegaConf data containers, never arbitrary pickle fallback."""
    from omegaconf import DictConfig
    from omegaconf.base import ContainerMetadata, Metadata
    from omegaconf.nodes import AnyNode

    allowed = [DictConfig, ContainerMetadata, Metadata, AnyNode, typing.Any,
               dict, collections.defaultdict]
    previous = list(torch.serialization.get_safe_globals())
    try:
        torch.serialization.add_safe_globals(allowed)
        checkpoint = torch.load(path, weights_only=True, map_location="cpu", mmap=True)
    finally:
        torch.serialization.clear_safe_globals()
        torch.serialization.add_safe_globals(previous)
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError("Expected a Lightning checkpoint with state_dict and model metadata")
    return checkpoint


def normalized_weights(checkpoint: dict) -> dict:
    weights = {}
    for key, value in checkpoint["state_dict"].items():
        name = key.removeprefix("model.")
        if name in weights or not isinstance(value, torch.Tensor):
            raise ValueError(f"Ambiguous or nontensor state key: {name}")
        weights[name] = value
    return weights


def training_split_provenance(checkpoint: dict, intended_ids: list[str]) -> dict:
    """A currently existing path is evidence, not proof of historical contents."""
    data = checkpoint.get("hyper_parameters", {}).get("dataset", {})
    value = data.get("train_dataset")
    result = {"recorded_train_list": value, "recorded_data_root": data.get("data_root", data.get("raw_data_root")),
              "historical_list_contents_verified": False}
    if value:
        path = Path(value)
        path = path if path.is_absolute() else REPOSITORY / path
        if path.is_file():
            current = {line.strip() for line in path.read_text().splitlines() if line.strip()}
            overlap = sorted(current.intersection(intended_ids))
            result.update(current_path=str(path), current_sha256=file_sha256(path),
                          current_count=len(current), intended_overlap_count=len(overlap),
                          intended_overlap_ids=overlap)
    return result


def load_models(legacy_path: Path, joint_path: Path, device: torch.device,
                intended_ids: list[str]):
    from omegaconf import OmegaConf
    from src.brepnet.models.vae import AutoEncoder_light
    from src.brepnet.models.vae import AutoEncoder_joint

    legacy_checkpoint, joint_checkpoint = safe_checkpoint(legacy_path), safe_checkpoint(joint_path)
    historical = legacy_checkpoint.get("hyper_parameters", {}).get("model", {})
    if historical.get("name") not in {"AutoEncoder_1119_light", "AutoEncoder_light"}:
        raise ValueError(f"Unsupported legacy architecture: {historical.get('name')}")
    legacy_config = dict(AutoEncoder_light.DEFAULTS)
    legacy_config.update(norm="layer", in_channels=6)
    aliases = {"dim_shape": "hidden_channels", "dim_latent": "latent_channels",
               "encoder_decoder_sigmoid": "sigmoid"}
    for key, value in historical.items():
        target = aliases.get(key, key)
        if target in legacy_config or target in {"norm", "in_channels"}:
            legacy_config[target] = value
    legacy_config.update(checkpoint=None, mode="frozen_inference")
    joint_config = joint_checkpoint.get("hyper_parameters", {}).get("model")
    if joint_config is None or joint_config.get("name") != "AutoEncoder_joint":
        raise ValueError("Joint checkpoint must record AutoEncoder_joint model configuration")
    joint_config = dict(joint_config)
    joint_config.update(checkpoint=None, init_encoder_checkpoint=None, mode="frozen_inference")
    # Meta construction avoids random allocation and then replacing huge decoder weights.
    with torch.device("meta"):
        legacy = AutoEncoder_light(legacy_config)
        joint = AutoEncoder_joint(joint_config)
    legacy_weights, joint_weights = normalized_weights(legacy_checkpoint), normalized_weights(joint_checkpoint)
    legacy.load_state_dict(legacy_weights, strict=True, assign=True)
    joint.load_state_dict(joint_weights, strict=True, assign=True)
    encoder_keys = [key for key in joint_weights if key.split(".")[0] in joint.ENCODER_MODULES]
    legacy_encoder_keys = {key for key in legacy_weights if key.split(".")[0] in joint.ENCODER_MODULES}
    if set(encoder_keys) != legacy_encoder_keys:
        raise ValueError("Encoder key sets differ; shared-latent comparison contract fails")
    different = [key for key in encoder_keys if not torch.equal(joint_weights[key], legacy_weights[key])]
    if different:
        raise ValueError(f"Encoder/posterior weights differ: {different}")
    for key in ("in_channels", "latent_channels", "hidden_channels", "norm", "gaussian_weights", "sigmoid",
                "num_gat_layers", "bottleneck_dim", "num_encoder_layers", "nhead", "primitive_blocks_per_stage"):
        if legacy_config[key] != joint_config.get(key, joint.DEFAULTS.get(key)):
            raise ValueError(f"Encoder/posterior configuration differs: {key}")
    audit = {"encoder_equal_tensor_count": len(encoder_keys), "strict_load": True,
             "legacy_config": legacy_config, "joint_config": joint_config}
    for method, checkpoint, weights in (("legacy", legacy_checkpoint, legacy_weights),
                                         ("joint", joint_checkpoint, joint_weights)):
        audit[method] = {"epoch": checkpoint.get("epoch"), "global_step": checkpoint.get("global_step"),
                         "state_tensor_count": len(weights), "parameter_count": sum(v.numel() for v in weights.values()),
                         "training_split": training_split_provenance(checkpoint, intended_ids)}
    # Preserve historical metadata as plain JSON-safe data, never pickle it again.
    audit = OmegaConf.to_container(OmegaConf.create(audit), resolve=True)
    legacy = legacy.float().eval().to(device)
    joint = joint.float().eval().to(device)
    legacy.requires_grad_(False)
    joint.requires_grad_(False)
    return legacy, joint, audit


def configure_fp32(seed: int = 42) -> dict:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_default_dtype(torch.float32)
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return {"dtype": "float32", "autocast": False, "seed": seed,
            "matmul_precision": "highest", "matmul_allow_tf32": False,
            "cudnn_allow_tf32": False, "cudnn_benchmark": False, "cudnn_deterministic": True,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
            "numpy_version": np.__version__}


def load_identity_geometry(path: Path, model_id: str, device: torch.device) -> tuple[dict, dict]:
    from src.brepnet.data.shared_boundaries import build_shared_boundary_targets
    with np.load(path, allow_pickle=False) as archive:
        faces = archive["sample_points_faces"].copy()
        edges = archive["sample_points_lines"].copy()
        connectivity = archive["edge_face_connectivity"].copy()
    target = build_shared_boundary_targets(
        faces, edges, connectivity, sample_id=model_id, reciprocal_tolerance=.01,
        closed_tolerance=1e-5, allow_closed_phase=True,
    )
    if faces.shape[1:3] != (16, 16) or edges.shape[1] != 16:
        raise ValueError("The frozen VAE geometry contract requires 16x16 faces and 16-point curves")
    # encode() consumes these raw global six-channel arrays, not face_norm/edge_norm.
    # This is exactly the identity-view, one-object subset of the legacy collator.
    count = len(faces)
    batch = {"face_points": torch.as_tensor(faces, dtype=torch.float32, device=device),
             "edge_points": torch.as_tensor(edges, dtype=torch.float32, device=device),
             "edge_face_connectivity": torch.as_tensor(connectivity, dtype=torch.long, device=device),
             "attn_mask": torch.zeros((count, count), dtype=torch.bool, device=device),
             "face_counts": torch.tensor([count], dtype=torch.long, device=device)}
    truth = {"gt_face": faces[..., :3].astype(np.float32), "gt_edge": target["curves"],
             "gt_pairs": target["face_pairs"], "gt_edge_is_closed": target["is_closed"]}
    return batch, truth


def load_archived_latents(path: Path, device: torch.device) -> torch.Tensor:
    with np.load(path, allow_pickle=False) as archive:
        values = archive["face_latents"].copy()
    if values.dtype != np.float32 or values.ndim != 2 or values.shape[1] != 32 or not len(values):
        raise ValueError(f"Expected nonempty float32 [F,32] archived latent, got {values.shape}/{values.dtype}")
    if not np.isfinite(values).all():
        raise ValueError("Archived latents contain NaN/Inf")
    return torch.from_numpy(values).to(device)


def decode_legacy(model, latent: torch.Tensor) -> dict:
    from src.brepnet.dataset import denormalize_coord1112
    result = model.decode({"face_latents": latent}, v_data=None, v_deduplicated=False)
    array = lambda value: value.detach().cpu().numpy()
    return {"representation": np.asarray("legacy_half_edges"),
            "face_latents": array(latent.float()),
            "pred_face": array(denormalize_coord1112(result["face_points_local"], result["face_center_scale"])[..., :3].float()),
            "pred_edge": array(denormalize_coord1112(result["edge_points_local"], result["edge_center_scale"])[..., :3].float()),
            "pred_edge_face_connectivity": array(result["pred_edge_face_connectivity"].long()),
            "pred_face_adj": array(result["pred_face_adj"])}


def atomic_npz(path: Path, arrays: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".prediction-", suffix=".npz", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def claim_protocol(path: Path, protocol: dict) -> str:
    """Atomically publish a shared protocol before independent shards write files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".protocol-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(protocol, handle, sort_keys=True, indent=2)
            handle.write("\n")
        try:
            os.link(temporary, path)
        except FileExistsError:
            if json.loads(path.read_text()) != protocol:
                raise ValueError(f"Existing protocol differs: {path}; choose a fresh output root")
    finally:
        os.unlink(temporary)
    return json_digest(protocol)


def completed_pair(paths: dict[str, Path], protocol_hash: str, source_hash: str) -> bool:
    latents = []
    for path in paths.values():
        if not path.is_file():
            continue
        with np.load(path, allow_pickle=False) as archive:
            if str(archive["protocol_sha256"]) != protocol_hash or str(archive["input_source_sha256"]) != source_hash:
                raise ValueError(f"Existing prediction provenance mismatch: {path}")
            values = archive["face_latents"]
            if latent_sha256(values) != str(archive["input_latent_sha256"]):
                raise ValueError(f"Existing latent digest mismatch: {path}")
            for key in ("pred_face", "pred_edge", "pred_edge_face_connectivity", "representation"):
                archive[key]  # Force zip CRC and required-field validation before skipping.
            latents.append(values)
    if len(latents) != 2:
        return False
    if not np.array_equal(*latents):
        raise ValueError("Existing decoder outputs do not share identical input latents")
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("posterior_mean", "archived_latents"), required=True)
    for key in ("legacy-checkpoint", "joint-checkpoint", "data-root", "split-list", "output-root"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--latent-root", type=Path)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Require 0 <= shard-index < num-shards")
    if (args.mode == "archived_latents") != (args.latent_root is not None):
        raise ValueError("--latent-root is required only for archived_latents")
    ids = read_ids(args.split_list)
    shard_ids = ids[args.shard_index::args.num_shards]
    settings = configure_fp32()
    device = torch.device(args.device)
    started = time.monotonic()
    legacy, joint, audit = load_models(args.legacy_checkpoint, args.joint_checkpoint, device, ids)
    sources = ("tools/vae_compare_inference.py", "src/brepnet/models/vae.py",
               "src/brepnet/models/blocks.py", "src/brepnet/dataset.py",
               "src/brepnet/data/shared_boundaries.py")
    protocol = {"version": PROTOCOL_VERSION, "mode": args.mode, "numeric_settings": settings,
                "legacy_checkpoint": str(args.legacy_checkpoint.resolve()),
                "legacy_checkpoint_sha256": file_sha256(args.legacy_checkpoint),
                "joint_checkpoint": str(args.joint_checkpoint.resolve()),
                "joint_checkpoint_sha256": file_sha256(args.joint_checkpoint),
                "split_list": str(args.split_list.resolve()), "split_sha256": file_sha256(args.split_list),
                "intended_count": len(ids), "num_shards": args.num_shards,
                "data_root": str(args.data_root.resolve()),
                "latent_root": str(args.latent_root.resolve()) if args.latent_root else None,
                "source_sha256": {name: file_sha256(REPOSITORY / name) for name in sources},
                "model_audit": audit, "identity_view": args.mode == "posterior_mean",
                "decoder_gt_topology": False, "decoder_face_deduplication": False,
                "joint_threshold": joint.existence_threshold, "legacy_threshold": .5}
    metadata = args.output_root / "metadata" / "inference" / args.mode
    protocol_hash = claim_protocol(metadata / "protocol.json", protocol)
    status_path = metadata / f"shard-{args.shard_index:03d}.jsonl"
    if status_path.exists() and not args.resume:
        raise FileExistsError(f"Shard already started: {status_path}; use --resume")
    invocation = {"command": sys.argv if argv is None else argv, "device": str(device),
                  "shard_ids": shard_ids, "resume": args.resume, "protocol_sha256": protocol_hash,
                  "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True).strip(),
                  "git_status": subprocess.check_output(["git", "status", "--short"], cwd=REPOSITORY, text=True),
                  "startup_seconds": time.monotonic() - started}
    invocation_path = metadata / f"shard-{args.shard_index:03d}-invocation-{time.time_ns()}.json"
    invocation_path.write_text(json.dumps(invocation, indent=2) + "\n")
    print(json.dumps({"ready": True, "mode": args.mode, "shard": args.shard_index,
                      "count": len(shard_ids), "startup_seconds": invocation["startup_seconds"],
                      "encoder_equal_tensor_count": audit["encoder_equal_tensor_count"]}), flush=True)
    failures = 0
    with status_path.open("a" if args.resume else "x") as status, torch.inference_mode():
        for index, model_id in enumerate(shard_ids):
            sample_started = time.monotonic()
            paths = {method: args.output_root / "raw_predictions" / args.mode / method / model_id / "data.npz"
                     for method in ("legacy", "joint")}
            source = ((args.latent_root if args.mode == "archived_latents" else args.data_root)
                      / model_id / "data.npz")
            row = {"model_id": model_id, "mode": args.mode, "shard": args.shard_index,
                   "source": str(source.resolve()), "status": "failed"}
            try:
                source_hash = file_sha256(source)
                row["source_sha256"] = source_hash
                if any(path.exists() for path in paths.values()) and not args.resume:
                    raise FileExistsError("Prediction already exists; use --resume or a fresh output root")
                if args.resume and completed_pair(paths, protocol_hash, source_hash):
                    row["status"] = "resumed_skip"
                else:
                    if args.mode == "posterior_mean":
                        batch, truth = load_identity_geometry(source, model_id, device)
                        features = legacy.encode(batch, v_test=True)["face_features"]
                        latent = legacy.sample(features, v_is_test=True)[0]
                    else:
                        truth = {}
                        latent = load_archived_latents(source, device)
                    input_values = latent.detach().float().cpu().numpy().copy()
                    if not np.isfinite(input_values).all():
                        raise ValueError("Posterior/input latent contains NaN/Inf")
                    input_hash = latent_sha256(input_values)
                    row.update(input_latent_sha256=input_hash, num_input_faces=len(input_values), methods={})
                    for method, decoder in (("legacy", lambda: decode_legacy(legacy, latent)),
                                             ("joint", lambda: joint.decode_latents(latent))):
                        method_started = time.monotonic()
                        try:
                            prediction = decoder()
                            if not np.array_equal(prediction["face_latents"], input_values):
                                raise ValueError("Decoder changed the exported input latent set")
                            for field in ("pred_face", "pred_edge"):
                                if not np.isfinite(prediction[field]).all():
                                    raise ValueError(f"Nonfinite {field}")
                            prediction.update(truth)
                            prediction.update(protocol_sha256=np.asarray(protocol_hash),
                                              input_source_sha256=np.asarray(source_hash),
                                              input_latent_sha256=np.asarray(input_hash),
                                              comparison_mode=np.asarray(args.mode))
                            atomic_npz(paths[method], prediction)
                            row["methods"][method] = {"status": "ok", "faces": len(prediction["pred_face"]),
                                                       "edges": len(prediction["pred_edge"]),
                                                       "seconds": time.monotonic() - method_started}
                        except Exception as error:
                            row["methods"][method] = {"status": "failed", "error": f"{type(error).__name__}: {error}"}
                    row["status"] = "ok" if all(item["status"] == "ok" for item in row["methods"].values()) else "failed"
            except Exception as error:
                row["error"] = f"{type(error).__name__}: {error}"
            row["seconds"] = time.monotonic() - sample_started
            failures += int(row["status"] == "failed")
            status.write(json.dumps(row) + "\n")
            status.flush()
            if (index + 1) % 25 == 0 or index + 1 == len(shard_ids):
                print(json.dumps({"mode": args.mode, "shard": args.shard_index, "processed": index + 1,
                                  "total": len(shard_ids), "failures": failures,
                                  "elapsed_seconds": time.monotonic() - started}), flush=True)
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
