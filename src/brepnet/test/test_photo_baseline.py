"""Protect photo-only input selection, cache integrity and original loss semantics."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.brepnet.data.photo_dataset import PhotoFeatureDataset, photo_cache_provenance
from src.brepnet.data.preference_dataset import sha256_file
from tools.prepare_photo_baseline import InputDataset


def make_cache(root):
    manifest = {"version": "legacy_photo_rgb_v1", "complete": True, "image_key": "flux",
                "rotation_id": 0, "checkpoint_sha256": "source", "splits": {}}
    for split in ("training", "validation"):
        arrays = {"features": np.ones((1, 1024), np.float32),
                  "means": np.arange(960, dtype=np.float32).reshape(1, 30, 32),
                  "counts": np.array([3], np.int16)}
        info = {"ids": [split], "files": {}, "sha256": {}}
        for field, array in arrays.items():
            path = root / f"{split}_{field}.npy"
            np.save(path, array)
            info["files"][field] = path.name
            info["sha256"][field] = sha256_file(path)
        manifest["splits"][split] = info
    (root / "manifest.json").write_text(json.dumps(manifest))
    return {"cache_root": str(root), "epoch_scale_factor": 2}


def test_padding_uses_only_original_faces_and_keeps_all(tmp_path):
    cfg = make_cache(tmp_path)
    dataset = PhotoFeatureDataset("training", cfg)
    item = dataset[0]
    assert len(dataset) == 2 and item["gt_latent"].shape == (30, 32)
    assert torch.equal(item["gt_latent"].unique(dim=0), torch.arange(96).reshape(3, 32).float())
    assert item["image_features"].shape == (1024,)


def test_modified_arrays_rejected_even_with_same_manifest(tmp_path):
    cfg = make_cache(tmp_path)
    photo_cache_provenance(cfg)
    np.save(tmp_path / "training_features.npy", np.zeros((1, 1024), np.float32))
    with pytest.raises(ValueError, match="content changed"):
        photo_cache_provenance(cfg)


def test_reject_overlap_and_incomplete_cache(tmp_path):
    cfg = make_cache(tmp_path)
    path = tmp_path / "manifest.json"
    doc = json.loads(path.read_text())
    doc["splits"]["validation"]["ids"] = ["training"]
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="disjoint"):
        photo_cache_provenance(cfg)
    doc["complete"] = False
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="complete"):
        photo_cache_provenance(cfg)


def test_flux_only_and_source_binding(tmp_path):
    cond = tmp_path / "images" / "object"
    latent = tmp_path / "latents" / "object_0"
    cond.mkdir(parents=True)
    latent.mkdir(parents=True)
    np.savez(cond / "real_photo.npz", flux=np.full((16, 16, 3), 128, np.uint8),
             flux_masked=np.zeros((16, 16, 3), np.uint8))
    np.save(latent / "features.npy", np.ones((2, 64), np.float32))
    source = {"precision": "float32", "layout": "mean32,std32", "gt_npz_sha256": "raw",
              "output_sha256": {"features.npy": sha256_file(latent / "features.npy")}}
    (latent / "source.json").write_text(json.dumps(source))
    dataset = InputDataset([{"id": "object", "index": 0, "split": "training"}],
                           cond.parent, latent.parent)
    image, mean, count, audit = dataset[0]
    assert image.shape == (3, 224, 224) and count == 2 and mean.shape == (30, 32)
    expected = (torch.tensor([128/255]*3)-torch.tensor([.485,.456,.406]))/torch.tensor([.229,.224,.225])
    assert torch.allclose(image[:, 0, 0], expected)
    source["output_sha256"]["features.npy"] = "wrong"
    (latent / "source.json").write_text(json.dumps(source))
    with pytest.raises(ValueError, match="provenance"):
        dataset[0]


def test_photo_mapping_preserves_ae_and_rejects_changed_frozen_encoder():
    from types import SimpleNamespace
    from src.brepnet.diffusion.photo_baseline import apply_photo_state
    names = ("p_embed", "time_embed", "net1", "fc_out", "img_fc")
    legacy = SimpleNamespace(**{name: torch.nn.Linear(2, 2) for name in (*names, "img_model", "ae_model")})
    ae_before = {k: v.clone() for k, v in legacy.ae_model.state_dict().items()}
    state = {f"model.policy.{name}.{key}": torch.full_like(value, 7.)
             for name in names for key, value in getattr(legacy, name).state_dict().items()}
    state.update({"model.image_encoder."+k: v.clone() for k,v in legacy.img_model.state_dict().items()})
    apply_photo_state(legacy, state)
    assert all(torch.equal(v, ae_before[k]) for k,v in legacy.ae_model.state_dict().items())
    assert torch.equal(legacy.fc_out.weight, torch.full((2,2), 7.))
    with pytest.raises(ValueError, match="state keys"):
        apply_photo_state(legacy, {k:v for k,v in state.items() if k != "model.policy.fc_out.bias"})
    state["model.image_encoder.weight"] += 1.
    with pytest.raises(ValueError, match="Frozen photo DINO"):
        apply_photo_state(legacy, state)
