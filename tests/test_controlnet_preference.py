"""Learned control preservation, fixed reference, online pairs and complete reload."""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.common.io import sha256_file
from src.datasets.photo import photo_cache_provenance, photo_gt_image_provenance
from src.datasets.photo_preference import PhotoPreferenceImageDataset, photo_preference_image_provenance
from src.models.diffusion.controlnet_preference import (
    ControlNetPreferenceDiffusion, check_complete_checkpoint, control_config, load_controlnet_preference,
)
from tests.test_diffusion_controlnet import bundle  # noqa: F401 - shared tiny learned model fixture


@pytest.fixture
def learned_control(bundle, tmp_path):
    source, source_cfg, _, data_cfg = bundle
    with torch.no_grad():
        for block in source.denoiser.control_blocks:
            block.zero_out.weight.normal_(std=.02)
            block.zero_out.bias.normal_(std=.02)
    saved = {"global_step": 93000, "hyper_parameters": {
        "model": source_cfg, "dataset": {"provenance": photo_gt_image_provenance(data_cfg)}},
        "state_dict": {f"model.{key}": value for key, value in source.state_dict().items()}}
    path = tmp_path / "control.ckpt"
    torch.save(saved, path)
    cfg = {"name": "ControlNetPreferenceDiffusion", "init_controlnet_checkpoint": str(path),
           "checkpoint": None, "beta_dpo": 20., "preference_weight": 1.,
           "control_config": control_config(saved), "initialization": {"sha256": sha256_file(path)}}
    return source, cfg, data_cfg


def pair_batch():
    return {"gt_latent": torch.randn(2, 30, 32), "gt_image": torch.randn(2, 3, 8, 8),
            "winner_latent": torch.randn(2, 30, 32), "loser_latent": torch.randn(2, 30, 32),
            "pair_image": torch.randn(2, 3, 8, 8)}


def test_dpo_preserves_learned_control_and_updates_only_control(learned_control):
    source, cfg, _ = learned_control
    model = ControlNetPreferenceDiffusion(cfg).train()
    for key, value in source.denoiser.state_dict().items():
        torch.testing.assert_close(model.denoiser.state_dict()[key], value, atol=0, rtol=0)
        torch.testing.assert_close(model.reference.state_dict()[key], value, atol=0, rtol=0)
    assert all(a.data_ptr() != b.data_ptr() for a, b in zip(model.denoiser.parameters(), model.reference.parameters()))
    batch = pair_batch()
    losses = model(batch)
    assert losses["preference_loss"].item() == pytest.approx(np.log(2), abs=1e-5)
    assert losses["preference_logit"].item() == pytest.approx(0, abs=1e-5)
    losses["total_loss"].backward()
    trainable = dict(model.named_parameters())
    assert all(name.startswith("denoiser.control_blocks.") for name, p in trainable.items() if p.requires_grad)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in trainable.values() if p.requires_grad)
    assert all(p.grad is None for p in trainable.values() if not p.requires_grad)
    assert not model.reference.training and not model.denoiser.backbone.training and not model.image_encoder.training
    validation = model(batch, v_test=True)
    assert set(validation) == {"ddpm_loss", "total_loss"}


def test_complete_dpo_reload_needs_no_initial_source(learned_control, tmp_path):
    _, cfg, _ = learned_control
    model = ControlNetPreferenceDiffusion(cfg)
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=1e-3)
    model(pair_batch())["total_loss"].backward()
    optimizer.step()
    saved = {"hyper_parameters": {"model": cfg},
             "state_dict": {f"model.{key}": value for key, value in model.state_dict().items()}}
    path = tmp_path / "dpo.ckpt"
    torch.save(saved, path)
    Path(cfg["init_controlnet_checkpoint"]).unlink()
    restored = load_controlnet_preference(path)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value, atol=0, rtol=0)
    assert any(not torch.equal(a, b) for a, b in zip(restored.denoiser.control_blocks.parameters(),
                                                   restored.reference.control_blocks.parameters()))
    with pytest.raises(ValueError, match="reference/objective"):
        check_complete_checkpoint({**cfg, "beta_dpo": 40}, saved)
    bad = deepcopy(cfg)
    bad["initialization"]["sha256"] = "changed"
    with pytest.raises(ValueError, match="reference/objective"):
        check_complete_checkpoint(bad, saved)


def test_online_pairs_keep_image_identity_and_original_latents(bundle, tmp_path, monkeypatch):
    _, _, _, data_cfg = bundle
    manifest = json.loads((Path(data_cfg["cache_root"]) / "manifest.json").read_text())
    identity = manifest["splits"]["training"]["ids"][0]
    rows = {"image_id": identity, "split": "train", "pair_type": "validity"}
    expected = {}
    for side, candidate in [("winner", "c00"), ("loser", "c01")]:
        relative = Path("candidates/train") / identity / candidate / "latent.npy"
        path = tmp_path / relative
        path.parent.mkdir(parents=True)
        expected[side] = np.random.default_rng(len(side)).normal(size=(30, 32)).astype(np.float32)
        np.save(path, expected[side])
        rows.update({f"{side}_id": candidate, f"{side}_path": str(relative)})
    pairs_path = tmp_path / "pairs.jsonl"
    pairs_path.write_text(json.dumps(rows) + "\n")
    pair_provenance = {"protocol_version": "photo_preference_generation_v1",
                       "photo_cache": photo_cache_provenance(data_cfg)}
    monkeypatch.setattr("src.datasets.preference.preference_data_provenance", lambda cfg: pair_provenance)
    cfg = {**data_cfg, "pairs_manifest": str(pairs_path)}
    cfg["provenance"] = photo_preference_image_provenance(cfg)
    dataset = PhotoPreferenceImageDataset("training", cfg)
    item = dataset[0]
    assert item["v_prefix"] == item["pair_image_id"] == identity
    assert item["gt_image"].shape == item["pair_image"].shape == (3, 224, 224)
    torch.testing.assert_close(item["gt_image"], item["pair_image"], atol=0, rtol=0)
    assert "image_features" not in item and "pair_image_features" not in item
    for side in expected:
        np.testing.assert_array_equal(item[f"{side}_latent"].numpy(), expected[side])
    image_path = Path(cfg["condition_data_root"]) / identity / "real_photo.npz"
    np.savez(image_path, flux=np.ones((512, 512, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="photo differs"):
        dataset[0]
