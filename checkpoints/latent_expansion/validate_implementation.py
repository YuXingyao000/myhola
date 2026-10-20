"""Check 32 -> 64 warm initialization on validation data, without optimizer steps."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.brepnet.dataset import AutoEncoder_joint_dataset
from src.brepnet.models.vae import AutoEncoder_intersection


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def move(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move(item, device) for item in value]
    return value


def output_tensors(prediction):
    result = {"uv": prediction.edge_uv}
    for name in ("coarse", "refined"):
        stage = getattr(prediction, name)
        result[f"{name}.logits"] = stage.edge_logits
        for kind in ("face", "edge"):
            geometry = getattr(stage, f"{kind}_geometry")
            for field in ("local_points", "center_scale", "points"):
                result[f"{name}.{kind}.{field}"] = getattr(geometry, field)
    return result


def check_close(actual, expected, *, atol, rtol):
    actual, expected = actual.detach().float().cpu(), expected.detach().float().cpu()
    difference = (actual - expected).abs()
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    return {
        "max_abs": float(difference.max()),
        "mean_abs": float(difference.mean()),
        "atol": atol,
        "rtol": rtol,
    }


def last_projection(encoder):
    return next(
        module for module in reversed(list(encoder.children()))
        if isinstance(module, (torch.nn.Conv1d, torch.nn.Conv2d))
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--uv-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--precisions", nargs="+", choices=("float32", "bf16-mixed"),
                        default=("float32", "bf16-mixed"))
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.set_num_threads(2)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    start = time.monotonic()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    source_model = checkpoint["hyper_parameters"]["model"]
    assert source_model["shared_curve"]["enabled"], "Use the trained shared-curve source."
    assert source_model["latent_channels"] == 8, "Source must be the 32-D shared model."
    report = {
        "purpose": "implementation_only_no_optimizer_steps",
        "status": "running",
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "git_dirty": bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True
        ).strip()),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "source_epoch": checkpoint["epoch"],
        "source_global_step": checkpoint["global_step"],
        "torch": torch.__version__,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "batch_size": args.batch_size,
        "checks": [],
    }
    del checkpoint

    def write_report():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        )

    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        wide_cfg = compose(config_name="train_vae_latent_expansion")
        control_cfg = compose(config_name="train_vae_latent_expansion_control")
    wide_cfg.model.init_latent_expansion_checkpoint = str(args.checkpoint.resolve())
    control_cfg.model.checkpoint = str(args.checkpoint.resolve())
    for cfg in (wide_cfg, control_cfg):
        cfg.dataset.raw_data_root = str(args.data_root.resolve())
        cfg.dataset.edge_uv_root = str(args.uv_root.resolve())
        cfg.dataset.val_dataset = str(
            ROOT / "checkpoints/joint_boundary/validation/val_contract_audit/accepted.txt"
        )
    assert not wide_cfg.model.freeze_encoder and not control_cfg.model.freeze_encoder
    assert wide_cfg.model.gaussian_weights == 5e-7
    assert control_cfg.model.gaussian_weights == 1e-6
    assert wide_cfg.model.loss_weights == control_cfg.model.loss_weights
    report["configuration"] = {
        "candidate": "train_vae_latent_expansion",
        "control": "train_vae_latent_expansion_control",
        "candidate_gaussian_weight": wide_cfg.model.gaussian_weights,
        "control_gaussian_weight": control_cfg.model.gaussian_weights,
        "reconstruction_loss_weights": OmegaConf.to_container(wide_cfg.model.loss_weights),
    }
    split_path = Path(wide_cfg.dataset.val_dataset)
    selected_ids = split_path.read_text().splitlines()[:max(8, args.batch_size + 2)]
    # Only the declared implementation batch needs local sidecars. The full
    # validation list remains the provenance source, without filtering failures.
    with TemporaryDirectory(prefix="hola_latent_expansion_") as temp_dir:
        subset = Path(temp_dir) / "validation_ids.txt"
        subset.write_text("\n".join(selected_ids) + "\n")
        data_cfg = OmegaConf.create(OmegaConf.to_container(wide_cfg.dataset, resolve=True))
        data_cfg.val_dataset = str(subset)
        dataset = AutoEncoder_joint_dataset("validation", data_cfg)
    loaded_count = len(dataset.data_folders)
    samples = [dataset[index] for index in range(loaded_count)]
    batch = move(dataset.collate_fn(samples[:args.batch_size]), device)
    report["validation"] = {
        "split": str(split_path),
        "split_sha256": sha256(split_path),
        "loaded_ids": dataset.data_folders[:loaded_count],
        "batch_ids": dataset.data_folders[:args.batch_size],
        "face_counts": batch["face_counts"].tolist(),
        "connection_rows": len(batch["edge_face_connectivity"]),
        "raw_data_root": str(args.data_root.resolve()),
        "uv_root": str(args.uv_root.resolve()),
    }

    print("Checking the trained 32-D reference on posterior means.", flush=True)
    control = AutoEncoder_intersection(
        OmegaConf.to_container(control_cfg.model, resolve=True)
    ).to(device).eval()
    counts = {"control": sum(parameter.numel() for parameter in control.parameters())}
    with torch.no_grad():
        old_encoding = control.encode(batch)
        old_z, old_kl, old_statistics = control.sample(old_encoding["face_features"], True)
        old_prediction = control.decode_state(old_z, batch["face_counts"])
        reference = {
            "encoding": {key: value.cpu() for key, value in old_encoding.items()},
            "z": old_z.cpu(),
            "statistics": old_statistics.cpu(),
            "weighted_kl": old_kl.cpu(),
            "outputs": {key: value.cpu() for key, value in output_tensors(old_prediction).items()},
        }
    del control, old_encoding, old_z, old_kl, old_statistics, old_prediction
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print("Checking the migrated 64-D model against the 32-D reference.", flush=True)
    model = AutoEncoder_intersection(
        OmegaConf.to_container(wide_cfg.model, resolve=True)
    ).to(device).eval()
    counts["candidate"] = sum(parameter.numel() for parameter in model.parameters())
    counts["added"] = counts["candidate"] - counts["control"]
    counts["trainable"] = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    assert counts["trainable"] == counts["candidate"]
    assert len(model.decoder.inter.layers) == 8
    assert len(model.decoder.uv_boundary.layers) == 2
    initialization = model.initialization_report
    report["initialization"] = {
        key: value for key, value in initialization.items()
        if key != "retained_keys"
    }
    report["parameters"] = counts
    report["initial_mean_evaluation"] = {}
    with torch.no_grad():
        encoding = model.encode(batch)
        z, kl, statistics = model.sample(encoding["face_features"], True)
        prediction = model.decode_state(z, batch["face_counts"])
        comparisons = report["initial_mean_evaluation"]
        for key, value in encoding.items():
            comparisons[key] = check_close(
                value, reference["encoding"][key].repeat(1, 2), atol=1e-5, rtol=1e-4
            )
        comparisons["posterior_mean"] = check_close(
            z, reference["z"].repeat(1, 2), atol=1e-5, rtol=1e-4
        )
        old_mean, old_std = reference["statistics"].chunk(2, dim=-1)
        comparisons["posterior_mean_std"] = check_close(
            statistics, torch.cat((old_mean.repeat(1, 2), old_std.repeat(1, 2)), dim=-1),
            atol=1e-5, rtol=1e-4,
        )
        comparisons["weighted_kl"] = check_close(
            kl, reference["weighted_kl"], atol=1e-7, rtol=1e-4
        )
        for key, value in output_tensors(prediction).items():
            # Widened GEMMs change FP32 reduction order. Report raw differences;
            # XYZ has a tighter absolute bound than UV/local directions/logits.
            atol = 2e-4 if key == "uv" or key.endswith("local_points") else 1e-4
            if key.endswith("logits"):
                atol = 5e-4
            comparisons[key] = check_close(
                value, reference["outputs"][key], atol=atol, rtol=1e-4
            )
        old_active = reference["outputs"]["refined.logits"] > 0
        new_active = prediction.refined.edge_logits.cpu() > 0
        report["initial_changed_edge_decisions"] = int((old_active != new_active).sum())
        assert report["initial_changed_edge_decisions"] == 0
        pure_latents = z[:int(batch["face_counts"][0])].detach().clone()
    del encoding, z, kl, statistics, prediction, reference
    gc.collect()
    write_report()

    for precision in args.precisions:
        print(f"Checking {precision} joint forward/backward; no optimizer step.", flush=True)
        model.train()
        model.zero_grad(set_to_none=True)
        torch.manual_seed(42)
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        tick = time.monotonic()
        context = (
            torch.autocast(device.type, dtype=torch.bfloat16)
            if precision == "bf16-mixed" else nullcontext()
        )
        with context:
            losses, _ = model(batch)
        assert all(torch.isfinite(value).all() for value in losses.values())
        losses["total_loss"].backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        assert gradients and all(torch.isfinite(value).all() for value in gradients)
        grouped = {}
        for name in ("encoder", "posterior", "decoder"):
            module = getattr(model, name)
            values = [parameter.grad for parameter in module.parameters() if parameter.grad is not None]
            norm = sum(float(value.detach().abs().sum()) for value in values)
            assert norm > 0, f"{name} did not receive a nonzero gradient"
            grouped[name] = {"gradient_tensors": len(values), "gradient_l1": norm}
        asymmetry = {}
        for kind in ("face", "edge"):
            projection = last_projection(getattr(model.encoder, f"{kind}_coords"))
            first, second = projection.weight.grad.chunk(2, dim=0)
            difference = (first - second).abs()
            assert float(difference.max()) > 0, f"{kind} duplicated channels remain gradient-identical"
            asymmetry[kind] = {
                "first_half_gradient_l1": float(first.abs().sum()),
                "second_half_gradient_l1": float(second.abs().sum()),
                "max_abs_gradient_difference": float(difference.max()),
                "mean_abs_gradient_difference": float(difference.mean()),
            }
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        record = {
            "precision": precision,
            "seconds": time.monotonic() - tick,
            "losses": {key: float(value.detach()) for key, value in losses.items()},
            "gradient_groups": grouped,
            "duplicated_encoder_channel_gradients": asymmetry,
            "peak_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        }
        report["checks"].append(record)
        print(json.dumps({
            "precision": precision, "seconds": record["seconds"],
            "total_loss": record["losses"]["total_loss"],
            "peak_memory_bytes": record["peak_memory_bytes"],
        }), flush=True)
        del losses, gradients, values, first, second, difference
        write_report()

    model.eval()
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        # Only z enters these calls: no GT edges, UV coordinates or topology.
        prediction = model.decode_state(pure_latents)
        reverse = prediction.pairs.reverse
        points, uv = prediction.refined.edge_geometry.points, prediction.edge_uv
        torch.testing.assert_close(points, points[reverse].flip(1), atol=0, rtol=0)
        torch.testing.assert_close(uv, uv[reverse].flip((1, 2)), atol=0, rtol=0)
        exported = model.decode_latents(pure_latents)
        assert exported["pred_edge_uv"].shape == (len(exported["pred_edge"]), 2, 16, 2)
    report["pure_latent_inference"] = {
        "latent_shape": list(pure_latents.shape),
        "candidate_pairs": len(prediction.pairs.indices),
        "exported_faces": len(exported["pred_face"]),
        "exported_edges": len(exported["pred_edge"]),
        "reciprocal_xyz_exact": True,
        "reciprocal_uv_exact": True,
        "gt_geometry_or_topology_supplied": False,
    }
    files = [
        "src/brepnet/models/vae/checkpoints.py",
        "src/brepnet/models/vae/intersection.py",
        "src/brepnet/models/vae/legacy.py",
        "src/brepnet/models/vae/encoders.py",
        "src/brepnet/models/vae/decoders/intersection.py",
        "src/brepnet/models/vae/decoders/primitives.py",
        "src/brepnet/models/vae/losses/intersection.py",
        "configs/train_vae_latent_expansion.yaml",
        "configs/train_vae_latent_expansion_control.yaml",
        "checkpoints/latent_expansion/validate_implementation.py",
    ]
    report["source_sha256"] = {name: sha256(ROOT / name) for name in files}
    report["seconds_total"] = time.monotonic() - start
    report["status"] = "passed"
    write_report()
    print(json.dumps({
        "status": report["status"], "output": str(args.output),
        "parameters": counts, "seconds_total": report["seconds_total"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
