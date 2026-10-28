"""Verify source migration, real CAD gradients and worst-size memory before training."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import time

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import torch

from src.brepnet.dataset import AutoEncoder_joint_dataset
from src.brepnet.models import build_model
from src.brepnet.models.vae.intersection import AutoEncoder_intersection


SOURCE_SHA256 = "ce749416c1635dda1563161b971bd4260d064cfdb0c31f1fa9d06ad0e70961ff"


def device_batch(value):
    if isinstance(value, torch.Tensor):
        return value.cuda()
    if isinstance(value, dict):
        return {key: device_batch(item) for key, item in value.items()}
    if isinstance(value, list):
        return [device_batch(item) for item in value]
    return value


def compare_decoders(reference, model, batch):
    """Identical mean latents isolate decoder arithmetic from CUDA GAT scatter."""
    with torch.no_grad():
        encoding = reference.encode(batch)
        latent = reference.sample(encoding["face_features"], v_is_test=True)[0]
        old = reference.decode(latent, batch["face_counts"])
        new = model.decode(latent, batch["face_counts"])
        differences = {}
        for stage_name in ("coarse", "refined"):
            old_stage, new_stage = getattr(old, stage_name), getattr(new, stage_name)
            tensors = {
                "face_xyz": (old_stage.face_geometry.points, new_stage.face_geometry.points),
                "edge_xyz": (old_stage.edge_geometry.points, new_stage.edge_geometry.points),
                "edge_feature": (old_stage.edge_features, new_stage.edge_features),
                "logits": (old_stage.edge_logits, new_stage.edge_logits),
            }
            for name, (expected, actual) in tensors.items():
                differences[f"{stage_name}/{name}"] = float((expected - actual).abs().max())
                print(json.dumps({"tensor": f"{stage_name}/{name}", "max_abs": differences[f"{stage_name}/{name}"]}), flush=True)
                torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)
        torch.testing.assert_close(new.edge_uv, old.edge_uv, rtol=2e-4, atol=2e-5)
        xyz = new.refined.edge_geometry.points
        twin_error = float((xyz - xyz[new.pairs.reverse].flip(1)).abs().max())
        assert twin_error == 0
        exports = model.export_items(new, latent)
        assert all(item["representation"] == "legacy_half_edges" for item in exports)
        return differences, twin_error


def train_updates(model, batch, lr):
    model.train()
    assert all(parameter.requires_grad for parameter in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    gradient_groups = {
        "encoder": "encoder.", "posterior": "posterior.",
        "local_projection": "decoder.inter.spatial_memory.",
        "reference_uv": "decoder.inter.reference_uv_head.",
        "position_q": ".query_position.", "position_k": ".key_position.",
        "query_scale": ".query_scale.",
    }
    updates = []
    torch.cuda.reset_peak_memory_stats()
    for step in range(2):
        torch.cuda.synchronize()
        started = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            losses, _ = model(batch)
        assert all(torch.isfinite(value).all() for value in losses.values())
        losses["total_loss"].backward()
        gradients = {name: 0.0 for name in gradient_groups}
        missing = []
        for name, parameter in model.named_parameters():
            if parameter.grad is None:
                missing.append(name)
                continue
            assert torch.isfinite(parameter.grad).all(), name
            for group, pattern in gradient_groups.items():
                if pattern in name:
                    gradients[group] += float(parameter.grad.float().abs().sum())
        for group in ("encoder", "posterior", "local_projection"):
            assert gradients[group] > 0, group
        if step == 1:
            assert all(value > 0 for value in gradients.values()), gradients
        norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True))
        optimizer.step()
        torch.cuda.synchronize()
        updates.append({
            "step": step + 1, "seconds": time.monotonic() - started,
            "losses": {key: float(value.detach()) for key, value in losses.items()},
            "gradient_l1": gradients, "gradient_norm_before_clip": norm,
            "parameters_without_gradient": missing,
        })
        print(json.dumps({"update": updates[-1]}), flush=True)
    return updates, torch.cuda.max_memory_allocated() / 2**30


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--largest", action="store_true")
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    torch.manual_seed(42)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("highest")
    # Compare the changed attention in FP32, without chunk-dependent TF32 CNN rounding.
    torch.backends.cudnn.allow_tf32 = False
    with initialize_config_dir(config_dir=str(root / "configs"), version_base=None):
        cfg = compose(config_name="train_vae_conditional_intersection", overrides=[
            f"model.init_conditional_intersection_checkpoint={json.dumps(args.source)}",
            "dataset.raw_data_root=/mnt/d/data/deepcad_v7",
            "dataset.edge_uv_root=/mnt/d/data/deepcad_v7_uv/v1",
            "dataset.train_dataset=checkpoints/joint_boundary/validation/train_contract_audit/accepted.txt",
            "dataset.val_dataset=checkpoints/joint_boundary/validation/val_contract_audit/accepted.txt",
            "dataset.test_dataset=checkpoints/joint_boundary/validation/test_contract_audit/accepted.txt",
        ])
    dataset = AutoEncoder_joint_dataset("validation", cfg.dataset)
    items = [dataset[0], dataset[1]]
    if args.largest:
        items = []
        for index in range(len(dataset)):
            item = dataset[index]
            if len(item[1]) == 30:
                items.append(item)
            if len(items) == 2:
                break
        assert len(items) == 2, "Expected two real 30-face validation CADs"
        # Repeat the two largest CADs only for a worst-shape allocation check.
        items = [items[index % 2] for index in range(args.batch_size)]
    else:
        items = [dataset[index] for index in range(args.batch_size)]
    batch = device_batch(dataset.collate_fn(items))
    model = build_model(cfg.model).cuda().eval()
    assert model.initialization_report["source_sha256"] == SOURCE_SHA256
    checkpoint = torch.load(args.source, map_location="cpu", weights_only=False, mmap=True)
    old_cfg = dict(checkpoint["hyper_parameters"]["model"])
    assert OmegaConf.to_container(cfg.model.loss_weights) == old_cfg["loss_weights"]
    assert cfg.model.gaussian_weights == old_cfg["gaussian_weights"]
    weights = {key.removeprefix("model."): value for key, value in checkpoint["state_dict"].items()}
    for name, old in weights.items():
        torch.testing.assert_close(model.state_dict()[name].cpu(), old, rtol=0, atol=0, msg=name)
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Implementation and memory check on validation; not a quality evaluation.",
        "repeated_largest_cads_for_memory_only": args.largest and args.batch_size > 2,
        "comparison_precision": "FP32; cuDNN TF32 disabled",
        "training_precision": "BF16 autocast; matmul=medium, cuDNN TF32 enabled, matching trainer",
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0), "seed": 42,
        "sample_ids": list(batch["v_prefix"]), "face_counts": batch["face_counts"].tolist(),
        "initialization": model.initialization_report,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    for key in old_cfg:
        if key.startswith("init_"):
            old_cfg[key] = None
    old_cfg.update(checkpoint=None, freeze_encoder=False, mode="train")
    reference = AutoEncoder_intersection(old_cfg).cuda().eval()
    reference.load_state_dict(weights, strict=True)
    report["initial_eval_max_abs"], report["shared_twin_reverse_max_abs"] = compare_decoders(reference, model, batch)
    print(json.dumps({"initial_eval_max_abs": report["initial_eval_max_abs"]}), flush=True)
    del reference, checkpoint, weights
    gc.collect()
    torch.cuda.empty_cache()
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("medium")
    report["updates"], report["peak_train_memory_gib"] = train_updates(model, batch, cfg.trainer.learning_rate)
    report["passed"] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": True, "peak_train_memory_gib": report["peak_train_memory_gib"]}), flush=True)


if __name__ == "__main__":
    main()
