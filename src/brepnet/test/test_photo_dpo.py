"""照片DPO的来源、split、特征及policy/reference初始化合同。"""

import hashlib
import json
from unittest.mock import patch

import numpy as np
import pytest
import torch

from src.brepnet.data.photo_dataset import photo_cache_provenance
from src.brepnet.data.photo_preference_dataset import PhotoPreferenceDataset, read_photo_pairs
from src.brepnet.data.preference_dataset import preference_data_provenance, sha256_file
from src.brepnet.diffusion.preference import PreferenceDiffusion
from src.brepnet.test.test_diffusion_preference import tiny_legacy
from src.brepnet.test.test_photo_baseline import make_cache


def make_photo_pool(root):
    cache = root / "cache"
    cache.mkdir()
    cfg = make_cache(cache)
    run = root / "run"
    (run / "metadata").mkdir(parents=True)
    cfg.update(model_source="photo", pairs_manifest=str(run / "pairs.jsonl"),
               validation_pairs_manifest=str(run / "validation_pairs.jsonl"),
               cached_latent_root=str(run / "gt_cache"), condition_data_root=str(root / "images"))
    splits = {"train": ["training"], "validation": ["validation"]}
    status = {"status": "complete"}
    for split, identities in splits.items():
        identity = identities[0]
        list_path = root / f"{split}.txt"
        list_path.write_text(identity + "\n")
        cfg["train_dataset" if split == "train" else "val_dataset"] = str(list_path)
        row = {"image_id": identity, "split": split, "pair_type": "geometry",
               "winner_id": "c00", "loser_id": "c01"}
        for kind, value in (("winner", 1), ("loser", 2)):
            path = run / "candidates" / split / identity / row[f"{kind}_id"] / "latent.npy"
            path.parent.mkdir(parents=True)
            np.save(path, np.full((30, 32), value, np.float32))
            row[f"{kind}_path"] = str(path.relative_to(run))
        manifest = run / ("pairs.jsonl" if split == "train" else "validation_pairs.jsonl")
        manifest.write_text(json.dumps(row) + "\n")
        cfg["pairs_manifest_sha256" if split == "train" else "validation_pairs_manifest_sha256"] = sha256_file(manifest)
        status[split] = {"images": 1, "candidates": 8, "pairs": 1}
    condition = {"filename": "real_photo.npz", "key": "flux", "rotation_id": 0,
                 "shape": [512, 512, 3], "dtype": "uint8"}
    protocol = {"version": "photo_preference_generation_v1", "model_source": "photo",
                "condition": condition, "condition_data_root": cfg["condition_data_root"],
                "splits": splits, "photo_training_provenance": photo_cache_provenance(cfg),
                "checkpoint_sha256": "photo", "original_checkpoint_sha256": "source", "photo_global_step": 310000}
    protocol["fingerprint"] = hashlib.sha256(json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    (run / "metadata/protocol.json").write_text(json.dumps(protocol))
    (run / "status.json").write_text(json.dumps(status))
    latent = run / "gt_cache/validation_0/features.npy"
    latent.parent.mkdir(parents=True)
    np.save(latent, np.ones((3, 64), np.float32))
    image = root / "images/validation/real_photo.npz"
    image.parent.mkdir(parents=True)
    np.savez(image, flux=np.full((512, 512, 3), 127, np.uint8))
    return cfg


def test_photo_train_uses_baseline_cache_and_original_candidate_latents(tmp_path):
    cfg = make_photo_pool(tmp_path)
    train = PhotoPreferenceDataset("training", cfg)
    item = train[0]
    assert item["image_features"].shape == item["pair_image_features"].shape == (1024,)
    assert item["pair_image_id"] == "training"
    assert torch.equal(item["winner_latent"], torch.ones(30, 32))
    assert torch.equal(item["loser_latent"], torch.full((30, 32), 2.))
    assert "gt_image" not in item
    val = PhotoPreferenceDataset("validation", cfg)
    assert val[0]["gt_image"].shape == (3, 224, 224)
    assert "winner_latent" not in val[0]
    assert len(val) == 1


def test_photo_provenance_rejects_changed_pair_or_grey_input(tmp_path):
    cfg = make_photo_pool(tmp_path)
    with pytest.raises(ValueError, match="flux RGB"):
        preference_data_provenance({**cfg, "model_source": "legacy_svr"})
    cfg["pairs_manifest_sha256"] = "changed"
    with pytest.raises(ValueError, match="SHA256"):
        preference_data_provenance(cfg)


def test_photo_pair_manifest_rejects_cross_split_and_self_pair(tmp_path):
    cfg = make_photo_pool(tmp_path)
    from pathlib import Path
    manifest = Path(cfg["pairs_manifest"])
    with pytest.raises(ValueError, match="out-of-split"):
        read_photo_pairs(manifest, ["validation"], "train")
    row = json.loads(manifest.read_text())
    row["loser_id"], row["loser_path"] = row["winner_id"], row["winner_path"]
    manifest.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="must differ"):
        read_photo_pairs(manifest, ["training"], "train")


def test_cached_and_online_features_have_same_loss_and_frozen_reference():
    legacy, provenance = tiny_legacy()
    provenance["photo"] = {"global_step": 310000}
    cfg = {"model_source": "photo", "legacy_checkpoint": "original", "photo_checkpoint": "photo",
           "photo_global_step": 310000, "beta_dpo": 20., "preference_weight": 1.}
    with patch("src.brepnet.diffusion.photo_baseline.load_photo_svr", return_value=(legacy, provenance)) as load:
        model = PreferenceDiffusion(cfg).train()
    load.assert_called_once()
    batch = {"gt_image": torch.randn(2, 3, 8, 8), "pair_image": torch.randn(2, 3, 8, 8),
             "gt_latent": torch.randn(2, 30, 32), "winner_latent": torch.randn(2, 30, 32),
             "loser_latent": torch.randn(2, 30, 32)}
    with torch.no_grad():
        features = model.image_encoder(torch.cat((batch["gt_image"], batch["pair_image"])))
    cached = {k: v for k, v in batch.items() if k not in {"gt_image", "pair_image"}}
    cached.update(image_features=features[:2], pair_image_features=features[2:])
    torch.manual_seed(123)
    expected = model(batch)
    torch.manual_seed(123)
    with patch.object(model.image_encoder, "forward", side_effect=AssertionError("cached path called DINO")):
        actual = model(cached)
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    assert abs(actual["preference_loss"].item() - np.log(2)) < 1e-6
    actual["total_loss"].backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.policy.parameters())
    assert all(p.grad is None for p in model.reference.parameters())
    assert all(p.grad is None for p in model.image_encoder.parameters())


def test_photo_initialization_rejects_wrong_step():
    legacy, provenance = tiny_legacy()
    provenance["photo"] = {"global_step": 4553}
    with patch("src.brepnet.diffusion.photo_baseline.load_photo_svr", return_value=(legacy, provenance)):
        with pytest.raises(ValueError, match="initialization step"):
            PreferenceDiffusion({"model_source": "photo", "legacy_checkpoint": "original",
                                 "photo_checkpoint": "photo", "photo_global_step": 310000})
