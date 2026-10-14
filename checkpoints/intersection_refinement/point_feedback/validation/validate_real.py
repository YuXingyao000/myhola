"""Real-data compatibility and gradient validation; never saves trained weights."""
from __future__ import annotations

import dataclasses
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from src.brepnet.dataset import AutoEncoder_joint_dataset
from src.brepnet.models.vae import AutoEncoder_intersection

OUT = Path(__file__).resolve().parent
SOURCE = Path("/mnt/d/python/checkpoints/intersection_refinement/curve_conditioning/20261013T083729.740439Z/weights/epoch=0157-step=00057354-val_loss=0.038204.ckpt")
SEED = 20261014


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def flatten(value, prefix=""):
    if isinstance(value, torch.Tensor):
        return {prefix: value.detach().cpu().clone()}
    if dataclasses.is_dataclass(value):
        result = {}
        for field in dataclasses.fields(value):
            result.update(flatten(getattr(value, field.name), prefix + "." + field.name))
        return result
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            result.update(flatten(item, prefix + "." + key))
        return result
    raise TypeError(type(value))


def move(value):
    if isinstance(value, torch.Tensor):
        return value.cuda()
    if isinstance(value, dict):
        return {key: move(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(move(item) for item in value)
    return value


def synchronize():
    torch.cuda.synchronize()
    return time.perf_counter()


def forward_details(model, batch, amp):
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        encoded = model.encode(batch, v_test=True)
        z, kl, statistics = model.sample(encoded["face_features"], v_is_test=True)
        prediction = model.decode_state(z, batch["face_counts"])
        teacher = model.decoder.edge_geometry(encoded["edge_features"])
        losses = model.criterion(prediction, batch, encoded["edge_features"], teacher, kl, True)
    geometry = prediction.refined.edge_geometry
    with torch.autocast("cuda", enabled=False):
        rebuilt = geometry.local_points[..., :3].float() * geometry.center_scale[:, None, 3:].float()
        rebuilt = rebuilt + geometry.center_scale[:, None, :3].float()
    torch.testing.assert_close(rebuilt, geometry.points, rtol=1e-6, atol=2e-7)
    metadata = {
        "local_dtype": str(geometry.local_points.dtype),
        "global_dtype": str(geometry.points.dtype),
        "bbox_dtype": str(geometry.center_scale.dtype),
        "local_global_max_abs": float((rebuilt - geometry.points).abs().max()),
        "minimum_abs_bbox_scale": float(geometry.center_scale[:, 3:].abs().min()),
    }
    return flatten({"encoding": encoded, "z": z, "kl": kl, "statistics": statistics,
                    "prediction": prediction, "losses": losses}), metadata


def train_step(model, batch, optimizer, clip):
    model.train()
    assert not model.encoder.training and not model.posterior.training
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    start = synchronize()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        losses, _ = model(batch)
    after_forward = synchronize()
    assert all(torch.isfinite(value).all() for value in losses.values())
    losses["total_loss"].backward()
    after_backward = synchronize()
    gradients = {name: parameter.grad for name, parameter in model.named_parameters() if parameter.grad is not None}
    assert gradients and all(torch.isfinite(value).all() for value in gradients.values())
    assert all(parameter.grad is None for parameter in model.encoder.parameters())
    assert all(parameter.grad is None for parameter in model.posterior.parameters())
    feedback = model.decoder.curve_point_feedback
    measurements = {
        "delta_weight_gradient_l1": float(feedback.delta.weight.grad.abs().sum()),
        "point_encoder_gradient_l1": float(feedback.point_encoder[0].weight.grad.abs().sum()),
        "old_trainable_gradient_tensors": sum(not key.startswith("decoder.curve_point_feedback.") for key in gradients),
        "old_nonzero_gradient_tensors": sum(
            bool(value.abs().sum() > 0) for key, value in gradients.items()
            if not key.startswith("decoder.curve_point_feedback.")),
        "all_finite_gradient_tensors": len(gradients),
        "frozen_encoder_and_posterior_without_gradients": True,
        "loss_terms": len(losses), "total_loss": float(losses["total_loss"]),
    }
    grad_norm = torch.nn.utils.clip_grad_norm_(
        [parameter for parameter in model.parameters() if parameter.requires_grad], clip)
    optimizer.step()
    end = synchronize()
    measurements.update({
        "gradient_norm_before_clip": float(grad_norm),
        "forward_seconds": after_forward - start,
        "backward_seconds": after_backward - after_forward,
        "optimizer_and_audit_seconds": end - after_backward,
        "total_seconds": end - start,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024 ** 3,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024 ** 3,
    })
    return measurements



def scale_probe(model, batch):
    saved_rng, saved_cuda = torch.random.get_rng_state(), torch.cuda.get_rng_state_all()
    feedback = model.decoder.curve_point_feedback
    records = {}
    try:
        model.decoder.curve_point_feedback = None
        gt_pairs = {tuple(row) for row in batch["edge_face_connectivity"][:, 1:].cpu().tolist()}
        counts = batch["face_counts"].cpu().tolist()
        offsets = np.cumsum([0] + counts)
        for train_mode in (False, True):
            for amp in (False, True):
                # Each precision gets the same RNG state for the training probe.
                torch.random.set_rng_state(saved_rng)
                torch.cuda.set_rng_state_all(saved_cuda)
                model.train(train_mode)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                    encoding = model.encode(batch)
                    z, _, _ = model.sample(encoding["face_features"], v_is_test=not train_mode)
                    output = model.decode_state(z, batch["face_counts"])
                scales = output.refined.edge_geometry.center_scale[:, 3].float().cpu()
                pairs = output.pairs.indices.cpu().tolist()
                zero = (scales == 0).nonzero().flatten().tolist()
                nonfinite = (~torch.isfinite(scales)).nonzero().flatten().tolist()
                bad = sorted(set(zero + nonfinite))
                details = []
                for row in bad[:12]:
                    a, b = pairs[row]
                    obj = int(np.searchsorted(offsets, a, side="right") - 1)
                    details.append({"row": row, "object": batch["v_prefix"][obj],
                                    "pair": [a - int(offsets[obj]), b - int(offsets[obj])],
                                    "gt_positive": (a, b) in gt_pairs,
                                    "scale": float(scales[row])})
                records[("train" if train_mode else "eval") + ("_bf16" if amp else "_fp32")] = {
                    "dtype": str(output.refined.edge_geometry.center_scale.dtype),
                    "candidates": len(scales), "zero_count": len(zero), "nonfinite_count": len(nonfinite),
                    "zero_gt_positive": sum(tuple(pairs[row]) in gt_pairs for row in zero),
                    "zero_gt_negative": sum(tuple(pairs[row]) not in gt_pairs for row in zero),
                    "minimum_abs_scale": float(scales.abs().min()),
                    "minimum_nonzero_abs_scale": float(scales[scales != 0].abs().min()),
                    "minimum_positive_gt_abs_scale": float(torch.stack([scales[i].abs() for i, pair in enumerate(pairs) if tuple(pair) in gt_pairs]).min()),
                    "bad_examples": details,
                }
    finally:
        model.decoder.curve_point_feedback = feedback
        torch.random.set_rng_state(saved_rng)
        torch.cuda.set_rng_state_all(saved_cuda)
        model.eval()
    return records


def main():
    if (OUT / "result.json").exists():
        raise FileExistsError("Refusing to overwrite an existing final validation result")
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    overrides = [
        "dataset.raw_data_root=/mnt/d/data/deepcad_v7",
        "dataset.train_dataset=checkpoints/joint_boundary/validation/train_contract_audit/accepted.txt",
        "dataset.val_dataset=checkpoints/joint_boundary/validation/val_contract_audit/accepted.txt",
        "dataset.test_dataset=checkpoints/joint_boundary/validation/test_contract_audit/accepted.txt",
        "dataset.is_aug=0",
    ]
    with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "configs")):
        config = compose(config_name="train_vae_intersection_point_feedback", overrides=overrides)
    config.model.init_curve_feedback_checkpoint = str(SOURCE)
    model_config = OmegaConf.to_container(config.model, resolve=True)
    dataset_config = OmegaConf.to_container(config.dataset, resolve=True)
    dataset = AutoEncoder_joint_dataset("training", dataset_config)
    ids = dataset.data_folders[:16]
    assert len(ids) == len(set(ids)) == 16
    samples = [dataset[index] for index in range(16)]
    batch2 = move(dataset.collate_fn(samples[:2]))
    batch16_cpu = dataset.collate_fn(samples)
    print(json.dumps({"stage": "dataset", "ids": ids, "face_counts": batch16_cpu["face_counts"].tolist()}), flush=True)
    model = AutoEncoder_intersection(model_config)
    checkpoint = torch.load(SOURCE, map_location="cpu", weights_only=False, mmap=True)
    source = {key.removeprefix("model."): value for key, value in checkpoint["state_dict"].items()}
    actual = model.state_dict()
    for key, expected in source.items():
        torch.testing.assert_close(actual[key], expected, rtol=0, atol=0)
    new_keys = sorted(set(actual) - set(source))
    new_parameters = sum(actual[key].numel() for key in new_keys)
    assert len(source) == 736 and len(new_keys) == 6 and new_parameters == 27139
    assert all(key.startswith("decoder.curve_point_feedback.") for key in new_keys)
    frozen = [name for name, parameter in model.named_parameters() if not parameter.requires_grad]
    assert len(frozen) == 243 and all(name.startswith(("encoder.", "posterior.")) for name in frozen)
    inter_layers = model.decoder.inter.layers
    assert len(inter_layers) == 8 and inter_layers[0].self_attn.embed_dim == 512
    report = {
        "scope": "implementation validation only; not reconstruction-quality evidence",
        "source_checkpoint": str(SOURCE), "source_sha256": sha256(SOURCE),
        "source_epoch": checkpoint["epoch"], "source_global_step": checkpoint["global_step"],
        "source_state_tensors_retained_exactly": len(source),
        "new_state_keys": new_keys, "new_parameters": new_parameters,
        "frozen_tensor_count": len(frozen), "parameter_count": sum(p.numel() for p in model.parameters()),
        "inter": {"layers": len(inter_layers), "width": inter_layers[0].self_attn.embed_dim},
        "initialization_report": model.initialization_report,
        "model_config": model_config, "dataset_config": dataset_config,
        "environment": {"seed": SEED, "torch": torch.__version__, "python": sys.executable,
                        "gpu": torch.cuda.get_device_name(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                        "cpu_threads": 2, "deterministic": True, "tf32": False},
        "input_hashes": {dataset_config["train_dataset"]: sha256(dataset_config["train_dataset"]),
                         **{str(Path(dataset_config["raw_data_root"]) / sample / "data.npz"):
                            sha256(Path(dataset_config["raw_data_root"]) / sample / "data.npz") for sample in ids}},
        "code_hashes": {path: sha256(ROOT / path) for path in [
            "src/brepnet/models/blocks.py", "src/brepnet/models/vae.py", "src/brepnet/dataset.py",
            "configs/train_vae_intersection_point_feedback.yaml", "configs/model/vae_intersection.yaml"]},
        "script_sha256": sha256(__file__),
        "samples2": ids[:2], "samples16": ids,
        "face_counts2": batch2["face_counts"].tolist(),
        "face_counts16": batch16_cpu["face_counts"].tolist(),
        "candidates2": int((batch2["face_counts"] * (batch2["face_counts"] - 1)).sum()),
        "candidates16": int((batch16_cpu["face_counts"] * (batch16_cpu["face_counts"] - 1)).sum()),
        "equivalence": {},
    }
    del actual, source, checkpoint
    gc.collect()
    model.cuda().eval()
    module = model.decoder.curve_point_feedback
    for label, amp in (("fp32", False), ("bf16", True)):
        enabled, enabled_meta = forward_details(model, batch2, amp)
        try:
            model.decoder.curve_point_feedback = None
            disabled, disabled_meta = forward_details(model, batch2, amp)
        finally:
            model.decoder.curve_point_feedback = module
        assert enabled.keys() == disabled.keys()
        dtype_changes = []
        for key in enabled:
            torch.testing.assert_close(enabled[key], disabled[key], rtol=0, atol=0, check_dtype=False)
            if enabled[key].dtype != disabled[key].dtype:
                dtype_changes.append(key)
        report["equivalence"][label] = {
            "tensors_compared_exactly": len(enabled), "maximum_absolute_difference": 0.0,
            "dtype_promotions": dtype_changes, "enabled_geometry": enabled_meta, "disabled_geometry": disabled_meta,
        }
        print(json.dumps({"stage": "equivalence", "precision": label, **report["equivalence"][label]}), flush=True)
    probe_batch = move(batch16_cpu)
    report["source_scale_probe"] = scale_probe(model, probe_batch)
    print(json.dumps({"stage": "source_scale_probe", **report["source_scale_probe"]}), flush=True)
    del enabled, disabled
    gc.collect()
    learning_rate = float(config.trainer.learning_rate)
    clip = float(config.trainer.gradient_clip_val)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=learning_rate)
    old_head = model.decoder.classifier.weight.detach().clone()
    first = train_step(model, batch2, optimizer, clip)
    assert first["delta_weight_gradient_l1"] > 0 and first["point_encoder_gradient_l1"] == 0
    assert not torch.equal(old_head, model.decoder.classifier.weight)
    second = train_step(model, batch2, optimizer, clip)
    assert second["delta_weight_gradient_l1"] > 0 and second["point_encoder_gradient_l1"] > 0
    report["bf16_two_step"] = {"learning_rate": learning_rate, "gradient_clip_norm": clip,
                              "steps": [first, second], "old_classifier_updated": True}
    print(json.dumps({"stage": "bf16_two_step", **report["bf16_two_step"]}), flush=True)
    report["post_update_scale_probe"] = scale_probe(model, probe_batch)
    if "--probe-scales" in sys.argv:
        report["status"] = "scale_probe_complete"
        (OUT / "scale_probe.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({"stage": "post_update_scale_probe", **report["post_update_scale_probe"]}), flush=True)
        return
    model.eval()
    _, report["post_update_local_global_contract"] = forward_details(model, batch2, True)
    optimizer.zero_grad(set_to_none=True)
    del batch2, old_head, probe_batch
    gc.collect()
    torch.cuda.empty_cache()
    batch16 = move(batch16_cpu)
    report["bf16_batch16"] = train_step(model, batch16, optimizer, clip)
    print(json.dumps({"stage": "bf16_batch16", **report["bf16_batch16"]}), flush=True)
    report["status"] = "passed"
    (OUT / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print("VALIDATION PASSED", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        failure = {"status": "failed", "exception": traceback.format_exc()}
        (OUT / "failure.json").write_text(json.dumps(failure, ensure_ascii=False, indent=2) + "\n")
        raise
