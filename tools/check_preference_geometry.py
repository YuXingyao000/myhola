"""CPU-only fixed-train calibration and a real-batch geometry gradient check.

No Trainer, optimizer step, GPU, sampling or validation/test search is performed.
The recommended coefficient is printed and recorded, never applied automatically.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import time
from unittest.mock import patch


TIMESTEPS = [62, 187, 312, 437, 562, 687, 812, 937]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if args.threads < 1 or args.output.exists():
        parser.error("Use a positive CPU thread count and a new output file")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    import torch
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from src.brepnet.data.preference_dataset import PreferenceDataset, sha256_file
    from src.brepnet.diffusion.preference import PreferenceDiffusion, per_sample_mse
    from src.brepnet.diffusion.preference_geometry import (
        decode_supervised_points, predicted_geometry_loss, primitive_chamfer_squared,
    )

    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    started = time.monotonic()
    repo = Path(__file__).resolve().parents[1]
    with initialize_config_dir(config_dir=str(repo / "configs"), version_base="1.3"):
        cfg = compose("train_diffusion_preference_geometry", overrides=["model.geometry.weight=1.0"])
    print("Validating original train data/cache and loading the original SVR on CPU", flush=True)
    dataset = PreferenceDataset("training", cfg.dataset)
    if sha256_file(Path(cfg.model.legacy_checkpoint)) != dataset.provenance["source_checkpoint_sha256"]:
        raise ValueError("Original SVR differs from the frozen dataset source")
    model = PreferenceDiffusion(OmegaConf.to_container(cfg.model, resolve=True)).train()
    initialization = model.initialize_preference_weights(cfg.model.init_preference_checkpoint, dataset.provenance)
    # Fetch fixed train identities; no validation/test candidate selection.
    torch.manual_seed(42)
    items = [dataset[index] for index in range(len(TIMESTEPS))]
    batch = dataset.collate_fn(items)
    print(f"Loaded train IDs: {batch['v_prefix']}; checking native decoder equivalence", flush=True)
    target = batch["gt_geometry"][0]
    z = batch["gt_latent"][0, target["face_slots"]]
    ae = model.geometry_decoder
    edge_count = len(target["edge_faces"])
    with torch.no_grad():
        faces, edges = decode_supervised_points(ae, z, target["edge_faces"])
        native = ae.decode({"face_z": z, "edge_features": torch.zeros(edge_count, ae.df * 2)}, {
            "attn_mask": torch.zeros((len(z), len(z)), dtype=torch.bool),
            "edge_face_connectivity": torch.cat((torch.arange(edge_count)[:, None], target["edge_faces"]), 1),
            "zero_positions": torch.empty((0, 2), dtype=torch.long),
        })
        denormalize = ae.decode.__func__.__globals__["denormalize_coord1112"]
        native_faces = denormalize(native["face_points_local"], native["face_center_scale"])[..., :3]
        native_edges = denormalize(native["edge_points_local"], native["edge_center_scale"])[..., :3]
        torch.testing.assert_close(faces, native_faces, rtol=0, atol=1e-6)
        torch.testing.assert_close(edges, native_edges, rtol=0, atol=1e-6)
        decoder_reference = {
            "image_id": batch["v_prefix"][0],
            "surface_cd_sq": primitive_chamfer_squared(faces, target["face_points"]).mean().item(),
            "curve_cd_sq": primitive_chamfer_squared(edges, target["edge_points"]).mean().item() if edge_count else 0.,
            "native_surface_max_abs_error": (faces - native_faces).abs().max().item(),
            "native_curve_max_abs_error": (edges - native_edges).abs().max().item() if edge_count else 0.,
        }
    del native, faces, edges, native_faces, native_edges
    print("Calibrating geometry/DDPM epsilon-gradient scale on the fixed train batch", flush=True)
    torch.manual_seed(42)
    timestep = torch.tensor(TIMESTEPS)
    noise = torch.randn_like(batch["gt_latent"])
    noisy = model.noise_scheduler.add_noise(batch["gt_latent"], noise, timestep)
    with torch.no_grad():
        features = model.image_encoder(batch["gt_image"])
        prediction = model.policy(noisy, timestep, model.policy.image_condition(features))
    prediction = prediction.detach().requires_grad_()
    ddpm = per_sample_mse(prediction, noise).mean()
    geometry = predicted_geometry_loss(ae, noisy, prediction, timestep, model.noise_scheduler.alphas_cumprod,
                                      batch, min_snr=1.0, primitive_chunk_size=16)
    ddpm_gradient = torch.autograd.grad(ddpm, prediction)[0]
    geometry_gradient = torch.autograd.grad(geometry["geometry_loss"], prediction)[0]
    ddpm_norm, geometry_norm = ddpm_gradient.norm().item(), geometry_gradient.norm().item()
    if not torch.isfinite(geometry_gradient).all() or geometry_norm <= 0 or ddpm_norm <= 0:
        raise ValueError("Calibration requires finite nonzero geometry and DDPM gradients")
    coefficient = float(f"{0.1 * ddpm_norm / geometry_norm:.3g}")
    calibration = {"seed": 42, "train_ids": batch["v_prefix"], "timesteps": TIMESTEPS,
        "gradient_space": "GT epsilon prediction, not model parameter gradients", "target_ratio": .1,
        "ddpm_gradient_norm": ddpm_norm, "geometry_gradient_norm": geometry_norm,
        "geometry_active_fraction": geometry["geometry_active_fraction"].item(),
        "unweighted_losses": {key: value.item() for key, value in geometry.items()},
        "ddpm_loss": ddpm.item(), "recommended_weight": coefficient,
        "weighted_gradient_ratio": coefficient * geometry_norm / ddpm_norm}
    print(json.dumps(calibration), flush=True)
    del prediction, ddpm_gradient, geometry_gradient, geometry, ddpm, features
    gc.collect()

    print("Checking a real two-CAD full DDPM+DPO+geometry backward on CPU", flush=True)
    model.geometry["weight"] = coefficient
    real_batch = dataset.collate_fn(items[:2])
    torch.manual_seed(123)

    def fixed_noise(clean):
        t = torch.tensor([200, 750])
        n = torch.randn_like(clean)
        return model.noise_scheduler.add_noise(clean, n, t), t, n

    with patch.object(model, "_noise", side_effect=fixed_noise):
        losses = model(real_batch)
    expected = losses["ddpm_loss"] + losses["preference_loss"] + coefficient * losses["geometry_loss"]
    torch.testing.assert_close(losses["total_loss"], expected, rtol=0, atol=0)
    policy_first = next(model.policy.parameters())
    geometry_policy_gradient = torch.autograd.grad(losses["geometry_loss"], policy_first, retain_graph=True)[0]
    if not torch.isfinite(geometry_policy_gradient).all() or geometry_policy_gradient.abs().sum() == 0:
        raise ValueError("Geometry did not reach the original policy's first parameter")
    geometry_policy_norm = geometry_policy_gradient.norm().item()
    losses["total_loss"].backward()
    for name, parameter in model.policy.named_parameters():
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise ValueError(f"Missing/nonfinite policy gradient: {name}")
    for frozen in (model.reference, model.image_encoder, ae):
        if frozen.training or any(parameter.requires_grad or parameter.grad is not None for parameter in frozen.parameters()):
            raise ValueError("Frozen module participated in parameter training")
    result = {
        "purpose": "CPU implementation/gradient validation only; not a reconstruction-quality result",
        "data_provenance": dataset.provenance, "initialization": initialization,
        "original_decoder_reference": decoder_reference, "calibration": calibration,
        "real_batch": {"train_ids": real_batch["v_prefix"],
            "face_counts": [len(item["face_slots"]) for item in real_batch["gt_geometry"]],
            "edge_counts": [len(item["edge_faces"]) for item in real_batch["gt_geometry"]],
            "losses": {key: value.detach().item() for key, value in losses.items()},
            "geometry_gradient_norm_at_first_policy_parameter": geometry_policy_norm,
            "all_policy_gradients_finite": True, "frozen_parameters_have_no_grad": True},
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "policy_parameters": sum(parameter.numel() for parameter in model.policy.parameters()),
        "frozen_decoder_parameters": sum(parameter.numel() for parameter in ae.parameters()),
        "additional_trainable_parameters": 0, "gpu_used": False, "optimizer_steps": 0,
        "elapsed_seconds": time.monotonic() - started,
        "source_sha256": {str(path.relative_to(repo)): sha256_file(path) for path in (
            Path(__file__), repo / "src/brepnet/diffusion/preference.py", repo / "src/brepnet/diffusion/preference_geometry.py",
            repo / "src/brepnet/data/preference_dataset.py", repo / "src/brepnet/diffusion/legacy.py", repo / "src/brepnet/train.py")},
    }
    assert result["trainable_parameters"] == result["policy_parameters"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(f"Passed; record: {args.output}; recommended fixed lambda_geo={coefficient}", flush=True)


if __name__ == "__main__":
    main()
