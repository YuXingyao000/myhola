"""Validate full-capacity local intersection; temporary updates are not training results."""

from __future__ import annotations

import gc
import hashlib
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.brepnet.dataset import AutoEncoder_joint_dataset
from src.brepnet.models.vae import AutoEncoder_intersection


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).resolve().parent
SOURCE = ROOT / "checkpoints/intersection_refinement/frozen/20261011T123836.485257Z/weights/epoch=0010-step=00003993-val_loss=0.039476.ckpt"
HISTORICAL = ROOT / "checkpoints/intersection_local/frozen/20261012T073904.575690Z/weights/epoch=0019-step=00007260-val_loss=0.507484.ckpt"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def to_device(value):
    if isinstance(value, torch.Tensor):
        return value.cuda()
    if isinstance(value, dict):
        return {key: to_device(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_device(item) for item in value]
    if isinstance(value, tuple):
        return tuple(to_device(item) for item in value)
    return value


def main() -> None:
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    ids = ["00001000", "00002505"]
    accepted = set((ROOT / "checkpoints/joint_boundary/validation/train_contract_audit/accepted.txt").read_text().split())
    assert set(ids) <= accepted
    ids_path = OUTPUT / "smoke_ids.txt"
    ids_path.write_text("\n".join(ids) + "\n")
    with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "configs")):
        cfg = compose(config_name="train_vae_intersection_local")
    cfg.model.init_refinement_checkpoint = str(SOURCE)
    cfg.dataset.raw_data_root = "/mnt/d/data/deepcad_v7"
    for split in ("train_dataset", "val_dataset", "test_dataset"):
        cfg.dataset[split] = str(ids_path)
    resolved = OmegaConf.to_container(cfg, resolve=True)
    model_config = resolved["model"]
    assert model_config["intersection"]["block_type"] == "transformer_decoder"
    assert model_config["intersection"]["num_layers"] == 8
    assert model_config["intersection"]["hidden_dim"] == 512
    result = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Implementation validation only; temporary optimizer updates discarded; no formal training started.",
        "source_checkpoint": str(SOURCE), "source_sha256": digest(SOURCE),
        "historical_checkpoint": str(HISTORICAL), "historical_sha256": digest(HISTORICAL),
        "seed": 42, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
        "intersection_config": model_config["intersection"], "batches": [],
    }

    # The recorded historical config must still strictly load the original 2-layer model.
    historical = torch.load(HISTORICAL, map_location="cpu", weights_only=False)
    old_config = OmegaConf.to_container(OmegaConf.create(historical["hyper_parameters"]["model"]), resolve=True)
    old_config.update(checkpoint=None, init_legacy_checkpoint=None,
                      init_refinement_checkpoint=None, restoring_checkpoint=True)
    with torch.device("meta"):
        old_model = AutoEncoder_intersection(old_config)
    old_weights = {key.removeprefix("model."): value for key, value in historical["state_dict"].items()}
    old_model.load_state_dict(old_weights, strict=True, assign=True)
    assert len(old_model.decoder.inter.blocks) == 2
    assert sum(p.numel() for p in old_model.decoder.inter.parameters()) == 311616
    result["historical_small_checkpoint_strict_load"] = True
    del historical, old_weights, old_model
    gc.collect()

    print("Constructing full-capacity model and validating migration", flush=True)
    model = AutoEncoder_intersection(model_config)
    source = torch.load(SOURCE, map_location="cpu", weights_only=False)
    source_weights = {key.removeprefix("model."): value for key, value in source["state_dict"].items()}
    retained = [key for key in model.state_dict() if not key.startswith("decoder.inter.")]
    for key in retained:
        assert torch.equal(model.state_dict()[key], source_weights[key]), key
    frozen = {key: value.clone() for key, value in model.state_dict().items()
              if key.startswith(("encoder.", "posterior."))}
    result["retained_state_keys_exact"] = len(retained)
    result["initialization"] = model.initialization_report
    result["parameters"] = {
        "total": sum(p.numel() for p in model.parameters()),
        "inter": sum(p.numel() for p in model.decoder.inter.parameters()),
        "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "old_inter": sum(v.numel() for k, v in source_weights.items() if k.startswith("decoder.inter.")),
    }
    del source, source_weights
    gc.collect()
    model = model.cuda().train()
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=resolved["trainer"]["learning_rate"])
    dataset = AutoEncoder_joint_dataset("training", resolved["dataset"])
    samples = [dataset[index] for index in range(2)]
    for size in (2, 16):
        batch = to_device(dataset.collate_fn([samples[index % 2] for index in range(size)]))
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        print(f"Forward/backward/Adam: batch={size}, faces={batch['face_counts'].tolist()}", flush=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            losses, _ = model(batch)
        loss = losses["total_loss"]
        assert all(torch.isfinite(value).all() for value in losses.values())
        loss.backward()
        assert all(p.grad is None for p in model.encoder.parameters())
        assert all(p.grad is None for p in model.posterior.parameters())
        assert not model.encoder.training and not model.posterior.training
        grads = {}
        for name, module in {
            "inter": model.decoder.inter,
            "face_cnn": model.decoder.face_geometry.points_decoder,
            "face_bbox": model.decoder.face_geometry.bbox_decoder,
            "edge_cnn": model.decoder.edge_geometry.points_decoder,
            "classifier": model.decoder.classifier,
            "refiner": model.decoder.refiner,
        }.items():
            tensors = [p.grad for p in module.parameters() if p.grad is not None]
            assert tensors and all(torch.isfinite(g).all() for g in tensors), name
            grads[name] = float(sum(g.abs().sum() for g in tensors))
            assert grads[name] > 0, name
        for block in model.decoder.inter.blocks:
            assert isinstance(block, torch.nn.TransformerDecoderLayer)
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in block.parameters())
        torch.nn.utils.clip_grad_norm_(model.parameters(), resolved["trainer"]["gradient_clip_val"])
        optimizer.step()
        torch.cuda.synchronize()
        result["batches"].append({
            "batch_size": size, "ids": [ids[i % 2] for i in range(size)],
            "face_counts": batch["face_counts"].tolist(),
            "candidates": int((batch["face_counts"] * (batch["face_counts"] - 1)).sum()),
            "loss": float(loss), "step_seconds": time.perf_counter() - start,
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
            "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
            "gradient_l1": grads,
        })
        print(json.dumps(result["batches"][-1]), flush=True)
        del batch, losses, loss

    optimizer.zero_grad(set_to_none=True)
    for key, value in frozen.items():
        assert torch.equal(model.state_dict()[key].cpu(), value), key
    result["frozen_tensors_unchanged_after_steps"] = len(frozen)
    del optimizer, frozen
    gc.collect()
    torch.cuda.empty_cache()
    model.eval()
    batch = to_device(dataset.collate_fn(samples))
    with torch.no_grad():
        latent, _, _ = model.sample(model.encode(batch)["face_features"], v_is_test=True)
        prediction = model.decode_state(latent, batch["face_counts"])
        first = {"edge": prediction.refined.edge_features.clone(),
                 "face": prediction.refined.face_geometry.points.clone(),
                 "logits": prediction.refined.edge_logits.clone()}
        items = model.export_items(prediction, latent)
    assert len(items) == 2
    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    restore_cfg = dict(model_config, checkpoint=None, init_legacy_checkpoint=None,
                       init_refinement_checkpoint=None, restoring_checkpoint=True)
    with torch.device("meta"):
        restored = AutoEncoder_intersection(restore_cfg)
    restored.load_state_dict(state, strict=True, assign=True)
    restored = restored.cuda().eval()
    assert all(torch.equal(restored.state_dict()[key].cpu(), value) for key, value in state.items())
    with torch.no_grad():
        second = restored.decode_state(latent, batch["face_counts"])
    roundtrip_differences = {}
    for name, actual in {"edge": second.refined.edge_features,
                         "face": second.refined.face_geometry.points,
                         "logits": second.refined.edge_logits}.items():
        roundtrip_differences[name] = float((first[name] - actual).abs().max())
        torch.testing.assert_close(first[name], actual, atol=1e-6, rtol=1e-5)
    result["full_capacity_strict_state_roundtrip_exact"] = True
    result["roundtrip_output_max_abs_difference"] = roundtrip_differences
    result["roundtrip_output_tolerance"] = {"atol": 1e-6, "rtol": 1e-5}
    result["exported_items"] = len(items)
    result["status"] = "passed"
    result["validation_source_sha256"] = digest(Path(__file__))
    result["model_source_sha256"] = digest(ROOT / "src/brepnet/models/blocks.py")
    result["training_config_sha256"] = digest(ROOT / "configs/train_vae_intersection_local.yaml")
    (OUTPUT / "gpu_smoke.json").write_text(json.dumps(result, indent=2) + "\n")
    print("PASS: " + str(OUTPUT / "gpu_smoke.json"), flush=True)


if __name__ == "__main__":
    main()
