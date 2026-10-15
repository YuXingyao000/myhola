"""Verify the agreed model on real data; this is not a quality experiment."""
from __future__ import annotations

import dataclasses
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from src.brepnet.dataset import AutoEncoder_joint_dataset
from src.brepnet.models.blocks import EdgeValueGATv2Conv
from src.brepnet.models.vae import AutoEncoder_intersection

OUT = Path(__file__).resolve().parent
SOURCE = Path("/mnt/d/python/checkpoints/intersection_refinement/curve_conditioning/20261013T083729.740439Z/weights/epoch=0157-step=00057354-val_loss=0.038204.ckpt")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def flatten(value, prefix=""):
    if isinstance(value, torch.Tensor):
        return {prefix: value.detach().cpu().clone()}
    if dataclasses.is_dataclass(value):
        value = {field.name: getattr(value, field.name) for field in dataclasses.fields(value)}
    result = {}
    for key, child in value.items():
        result.update(flatten(child, prefix + "." + key))
    return result


def forward(model, batch, amp):
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        encoded = model.encode(batch)
        z, kl, statistics = model.sample(encoded["face_features"], True)
        prediction = model.decode_state(z, batch["face_counts"])
        teacher = model.decoder.edge_geometry(encoded["edge_features"])
        losses = model.criterion(prediction, batch, encoded["edge_features"], teacher, kl, False)
    return flatten({"encoded": encoded, "z": z, "kl": kl, "statistics": statistics,
                    "prediction": prediction, "losses": losses})


def train_step(model, batch, optimizer):
    model.train()
    assert model.encoder.training and model.posterior.training and model.decoder.training
    assert all(p.requires_grad for p in model.parameters())
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        losses, _ = model(batch)
    assert all(torch.isfinite(value).all() for value in losses.values())
    losses["total_loss"].backward()
    gradients = {key: p.grad for key, p in model.named_parameters() if p.grad is not None}
    assert all(torch.isfinite(value).all() for value in gradients.values())
    modules = list(layer for layer in model.encoder.graph_face_edge if isinstance(layer, EdgeValueGATv2Conv))
    result = {
        "loss": float(losses["total_loss"]), "weighted_kl_loss": float(losses["weighted_kl_loss"]),
        "finite_gradient_tensors": len(gradients),
        "value_final_weight_gradient_l1": [float(layer.value_net[-1].weight.grad.abs().sum()) for layer in modules],
        "value_input_weight_gradient_l1": [float(layer.value_net[0].weight.grad.abs().sum()) for layer in modules],
        "encoder_face_gradient_l1": float(model.encoder.face_coords[0].weight.grad.abs().sum()),
        "encoder_edge_gradient_l1": float(model.encoder.edge_coords[0].weight.grad.abs().sum()),
        "posterior_gradient_l1": float(model.posterior.projection[0].weight.grad.abs().sum()),
        "decoder_classifier_gradient_l1": float(model.decoder.classifier.weight.grad.abs().sum()),
    }
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    assert torch.isfinite(norm)
    optimizer.step()
    torch.cuda.synchronize()
    result.update(seconds=time.perf_counter() - start,
                  gradient_norm_before_clip=float(norm),
                  peak_allocated_gib=torch.cuda.max_memory_allocated() / 1024**3,
                  peak_reserved_gib=torch.cuda.max_memory_reserved() / 1024**3)
    return result


def main():
    if (OUT / "result.json").exists():
        raise FileExistsError("Existing validation result must not be overwritten")
    torch.set_num_threads(4)
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    overrides = ["dataset.raw_data_root=/mnt/d/data/deepcad_v7", "dataset.is_aug=0"]
    for split in ("train", "val", "test"):
        overrides.append(f"dataset.{split}_dataset=checkpoints/joint_boundary/validation/{split}_contract_audit/accepted.txt")
    with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "configs")):
        config = compose(config_name="train_vae_encoder_edge_value", overrides=overrides)
    cfg = OmegaConf.to_container(config.model, resolve=True)
    cfg["init_encoder_edge_value_checkpoint"] = str(SOURCE)
    candidate = AutoEncoder_intersection(cfg)
    original = torch.load(SOURCE, map_location="cpu", weights_only=False, mmap=True)
    original = {key.removeprefix("model."): value for key, value in original["state_dict"].items()}
    actual = candidate.state_dict()
    for key, value in original.items():
        torch.testing.assert_close(actual[key], value, rtol=0, atol=0)
    added = sorted(actual.keys() - original.keys())
    assert len(original) == 736 and len(added) == 20
    assert sum(actual[key].numel() for key in added) == 51680
    report = {
        "scope": "implementation validation only", "source": str(SOURCE), "source_sha256": sha256(SOURCE),
        "retained_tensors": len(original), "new_keys": added, "new_parameters": 51680,
        "total_parameters": sum(p.numel() for p in candidate.parameters()),
        "initialization_report": candidate.initialization_report,
        "git": os.popen("git rev-parse HEAD").read().strip(),
        "code_hashes": {path: sha256(ROOT / path) for path in (
            "src/brepnet/models/vae.py", "src/brepnet/models/blocks.py", "src/brepnet/train.py",
            "configs/train_vae_encoder_edge_value.yaml", "configs/model/vae_intersection.yaml")},
        "model_config": cfg,
    }
    del actual, original
    cfg = copy.deepcopy(cfg)
    cfg["encoder_edge_value"]["enabled"] = False
    cfg["init_encoder_edge_value_checkpoint"] = None
    cfg["checkpoint"] = str(SOURCE)
    control = AutoEncoder_intersection(cfg).cuda().eval()
    candidate.cuda().eval()
    dataset = AutoEncoder_joint_dataset("training", OmegaConf.to_container(config.dataset, resolve=True))
    samples = [dataset[i] for i in range(16)]
    report["ids"] = list(dataset.data_folders[:16])
    def batch(size):
        return {key: value.cuda() if isinstance(value, torch.Tensor) else value
                for key, value in dataset.collate_fn(samples[:size]).items()}
    small = batch(2)
    report["equivalence"] = {}
    for label, amp in (("fp32", False), ("bf16", True)):
        before, after = forward(control, small, amp), forward(candidate, small, amp)
        assert before.keys() == after.keys()
        for key in before:
            torch.testing.assert_close(after[key], before[key], rtol=0, atol=0)
        report["equivalence"][label] = {"tensor_count": len(before), "max_abs_difference": 0.0}
        print(json.dumps({"stage": "equivalence", "precision": label, **report["equivalence"][label]}), flush=True)
    del control, before, after
    gc.collect()
    torch.cuda.empty_cache()
    optimizer = torch.optim.Adam(candidate.parameters(), lr=2e-5)
    report["steps"] = []
    for index in range(2):
        step = train_step(candidate, small, optimizer)
        assert all(value > 0 for value in step["value_final_weight_gradient_l1"])
        assert all((value == 0 if index == 0 else value > 0) for value in step["value_input_weight_gradient_l1"])
        report["steps"].append(step)
        print(json.dumps({"stage": "small_step", "step": index + 1, **step}), flush=True)
    del small
    gc.collect()
    torch.cuda.empty_cache()
    large = batch(16)
    report["batch16_faces"] = large["face_counts"].tolist()
    report["batch16"] = train_step(candidate, large, optimizer)
    print(json.dumps({"stage": "batch16", **report["batch16"]}), flush=True)
    report["status"] = "passed"
    (OUT / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        (OUT / "failure.json").write_text(json.dumps({"status": "failed", "exception": traceback.format_exc()}, indent=2) + "\n")
        raise
