"""Validate the no-feature-matching run with one real BF16 batch; no updates."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import torch
from pytorch_lightning import seed_everything
from omegaconf import OmegaConf

from src.brepnet.dataset import AutoEncoder_joint_dataset
from src.brepnet.models.vae import AutoEncoder_intersection


def flatten(value, prefix=""):
    if isinstance(value, dict):
        return {key: item for name, child in value.items()
                for key, item in flatten(child, f"{prefix}.{name}" if prefix else name).items()}
    return {prefix: value}


def sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def move(value):
    if isinstance(value, torch.Tensor):
        return value.cuda()
    if isinstance(value, dict):
        return {key: move(item) for key, item in value.items()}
    if isinstance(value, list):
        return [move(item) for item in value]
    if isinstance(value, tuple):
        return tuple(move(item) for item in value)
    return value


def gradient_summary(values: list[torch.Tensor]) -> dict:
    return {
        "finite": all(bool(value.isfinite().all()) for value in values),
        "l2_norm": sum(float(value.float().square().sum()) for value in values) ** 0.5,
        "tensor_count": len(values),
    }


def main():
    os.chdir(ROOT)
    directory = Path(__file__).resolve().parent
    config_path = directory / "resolved_config.yaml"
    old_config_path = ROOT / "checkpoints/latent_expansion/d64/20261020T121050.873811Z/metadata/hydra/config.yaml"
    cfg = OmegaConf.load(config_path)
    old = OmegaConf.load(old_config_path)
    new_fields = flatten(OmegaConf.to_container(cfg, resolve=True))
    old_fields = flatten(OmegaConf.to_container(old, resolve=True))
    differences = {key: {"old64": old_fields.get(key), "no_edge_feature": new_fields.get(key)}
                   for key in sorted(old_fields.keys() | new_fields.keys())
                   if old_fields.get(key) != new_fields.get(key)}
    expected = {
        "model.loss_weights.feature_weight": 0.0,
        "trainer.exp_name": "latent_expansion/d64_no_edge_feature",
        "trainer.run_dir": "/mnt/d/python/checkpoints/latent_expansion/d64_no_edge_feature/20261023T034042Z",
        "trainer.max_steps": 30000,
        "trainer.wandb.name": "vae-latent64-no-edge-feature-30k",
    }
    assert set(differences) == set(expected), differences
    for key, value in expected.items():
        assert new_fields[key] == value, (key, new_fields[key], value)
    assert old_fields["model.loss_weights.feature_weight"] == 1.0
    assert cfg.trainer.resume_from_checkpoint is None
    assert cfg.model.checkpoint is None
    assert not cfg.model.freeze_encoder and cfg.model.mode == "train"
    assert cfg.model.latent_channels == 16 and cfg.model.gaussian_weights == 5e-7
    assert cfg.trainer.seed == 42 and cfg.trainer.batch_size == 16 and cfg.trainer.devices == 8
    assert cfg.trainer.precision == "bf16-mixed" and cfg.trainer.learning_rate == 2e-5
    assert cfg.dataset.is_aug == 1
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    source = Path(cfg.model.init_latent_expansion_checkpoint)
    checkpoint_hash = sha256(source)
    assert checkpoint_hash == "ce749416c1635dda1563161b971bd4260d064cfdb0c31f1fa9d06ad0e70961ff"
    expected_splits = {
        "train_dataset": (46480, "14a00585500f1c7996411caa1b3fba796a1c39c2b935a1153ec98e30c69dc097"),
        "val_dataset": (3495, "ae607bfe5499e919d1e68a5110c0c211daf1158aedbe57a4756eda99f6f3f0ed"),
    }
    splits = {}
    for name, (expected_count, expected_hash) in expected_splits.items():
        path = ROOT / cfg.dataset[name]
        digest = sha256(path)
        count = len([line for line in path.read_text().splitlines() if line.strip()])
        assert digest == expected_hash and count == expected_count, (name, digest, count)
        splits[name] = {"path": str(path), "sha256": digest, "count": count}

    seed_everything(42, workers=True)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("medium")
    model = AutoEncoder_intersection(OmegaConf.to_container(cfg.model, resolve=True)).cuda().train()
    assert all(parameter.requires_grad for parameter in model.parameters())
    assert model.criterion.feature_weight == 0.0
    assert model.criterion.encoder_edge_weight > 0.0
    assert model.face_feature_dim == 64
    dataset = AutoEncoder_joint_dataset("training", cfg.dataset)
    assert len(dataset.data_folders) == 46480, len(dataset.data_folders)
    batch = move(dataset.collate_fn([dataset[index] for index in range(16)]))
    captured = {}

    def capture_decoder(module, inputs, output):
        captured["prediction"] = output
        captured["latent_shape"] = list(inputs[0].shape)

    def capture_encoder(module, inputs, output):
        captured["teacher_shape"] = list(output["edge_features"].shape)

    handles = [model.decoder.register_forward_hook(capture_decoder),
               model.encoder.register_forward_hook(capture_encoder)]
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            losses, _ = model(batch)
    finally:
        for handle in handles:
            handle.remove()

    assert all(bool(value.isfinite().all()) for value in losses.values())
    assert float(losses["weighted_edge_feature"].detach()) == 0.0
    assert float(losses["edge_feature"].detach()) > 0.0
    for name in ("weighted_encoder_edge_local", "weighted_encoder_edge_bbox", "weighted_kl_loss"):
        assert float(losses[name].detach()) > 0.0, (name, losses[name])
    coarse_features = captured["prediction"].coarse.edge_features
    assert coarse_features.shape[1] == 128 and captured["teacher_shape"][1] == 128
    assert captured["latent_shape"][1] == 64
    weighted_gradient, = torch.autograd.grad(
        losses["weighted_edge_feature"], coarse_features, retain_graph=True,
    )
    raw_gradient, = torch.autograd.grad(losses["edge_feature"], coarse_features, retain_graph=True)
    feature_gradients = {
        "weighted_to_coarse": gradient_summary([weighted_gradient]),
        "raw_to_coarse": gradient_summary([raw_gradient]),
    }
    assert feature_gradients["weighted_to_coarse"]["finite"]
    assert torch.count_nonzero(weighted_gradient).item() == 0
    assert feature_gradients["raw_to_coarse"]["finite"]
    assert feature_gradients["raw_to_coarse"]["l2_norm"] > 0.0

    other_weighted_names = [name for name in losses
                            if name.startswith("weighted_") and name != "weighted_edge_feature"]
    total_without_feature = torch.stack([losses[name] for name in other_weighted_names]).sum()
    torch.testing.assert_close(total_without_feature, losses["total_loss"], rtol=1e-6, atol=1e-7)
    losses["total_loss"].backward()
    gradients = {}
    for name in ("encoder", "posterior", "decoder"):
        module = getattr(model, name)
        summary = gradient_summary([parameter.grad for parameter in module.parameters()
                                    if parameter.grad is not None])
        assert module.training and summary["finite"] and summary["l2_norm"] > 0.0, (name, summary)
        gradients[name] = summary

    report = {
        "status": "passed", "forward_passes": 1, "optimizer_steps": 0,
        "python": platform.python_version(), "torch": torch.__version__,
        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
        "repo_root": str(ROOT), "seed": 42, "precision": "bf16-mixed",
        "source_checkpoint": {"path": str(source), "sha256": checkpoint_hash},
        "initialization": model.initialization_report,
        "resolved_config_sha256": sha256(config_path),
        "old64_config_sha256": sha256(old_config_path),
        "config_differences_vs_64": differences, "splits": splits,
        "train_count": len(dataset.data_folders), "batch_size": 16,
        "batch_ids": list(batch["v_prefix"]),
        "dimensions": {"latents": captured["latent_shape"],
                       "coarse_edge_features": list(coarse_features.shape),
                       "teacher_edge_features": captured["teacher_shape"]},
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()
                                    if parameter.requires_grad),
        "training": {name: getattr(model, name).training for name in gradients},
        "gradients": gradients, "feature_gradients": feature_gradients,
        "total_without_feature": float(total_without_feature.detach()),
        "total_reconstruction_abs_error": float((total_without_feature - losses["total_loss"]).detach().abs()),
        "other_weighted_terms": other_weighted_names,
        "losses": {name: float(value.detach()) for name, value in losses.items()},
    }
    (directory / "preflight.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
