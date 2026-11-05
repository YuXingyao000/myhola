"""零门初始化、推理一致性、纯 GT 来源与完整恢复；小模型无需 CUDA/OCC。"""

from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import patch

from diffusers import DDPMScheduler
from hydra import compose, initialize_config_dir
import numpy as np
from omegaconf import OmegaConf
import pytest
import torch
from torch import nn

from src.common.io import sha256_file
from src.datasets.photo import PhotoGTOnlyDataset, photo_gt_data_provenance
from src.models.diffusion.adaln_zero import AdaLNZeroBlock, AdaLNZeroDenoiser
from src.models.diffusion.adaln_zero_photo import AdaLNZeroPhotoDiffusion, check_complete_checkpoint
from src.models.diffusion.preference import LegacySVRDenoiser


def tiny_legacy():
    legacy = nn.Module()
    legacy.dim_total = 16
    legacy.p_embed = nn.Sequential(nn.Linear(32, 12), nn.LayerNorm(12), nn.SiLU(), nn.Linear(12, 12))
    legacy.time_embed = nn.Sequential(nn.Linear(16, 16), nn.LayerNorm(16), nn.SiLU(), nn.Linear(16, 16))
    legacy.img_fc = nn.Sequential(nn.Linear(8, 8), nn.LayerNorm(8), nn.SiLU(), nn.Linear(8, 4))
    layer = nn.TransformerEncoderLayer(16, 4, dim_feedforward=24, dropout=.1,
                                       activation="relu", batch_first=True, norm_first=True)
    legacy.net1 = nn.TransformerEncoder(layer, 2, nn.LayerNorm(16), enable_nested_tensor=False)
    legacy.fc_out = nn.Sequential(nn.Linear(16, 16), nn.LayerNorm(16), nn.SiLU(), nn.Linear(16, 32))
    legacy.img_model = nn.Linear(8, 8)
    legacy.noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2",
                                         clip_sample=False)
    return legacy


@pytest.fixture
def model_bundle():
    torch.manual_seed(42)
    legacy = tiny_legacy()
    provenance = {"model_config": {"test": True}, "photo": {"global_step": 310000}}
    cfg = {"name": "AdaLNZeroPhotoDiffusion", "modulation_hidden_width": 5,
           "legacy_checkpoint": "original", "init_photo_checkpoint": "photo",
           "photo_global_step": 310000, "legacy_model_config": {"test": True}}
    with patch("src.models.diffusion.adaln_zero_photo.load_photo_svr",
               return_value=(deepcopy(legacy), deepcopy(provenance))):
        model = AdaLNZeroPhotoDiffusion(cfg)
    return model, cfg, legacy, provenance


def test_zero_gate_block_is_identity_and_discards_only_block_ln_affine(model_bundle):
    model, _, legacy, _ = model_bundle
    x, context = torch.randn(3, 7, 16), torch.randn(3, 20)
    for block in model.policy.net1.layers:
        assert isinstance(block, AdaLNZeroBlock)
        assert block.norm1.weight is None and block.norm2.bias is None
        assert block.self_attn.dropout == 0
        assert torch.equal(block(x, context), x)
    old = LegacySVRDenoiser(legacy).state_dict()
    current = model.policy.state_dict()
    discarded = set(model.initialization["discarded_block_ln_tensors"])
    assert len(discarded) == 8
    assert set(old) - set(current) == discarded
    for key in set(old) - discarded:
        assert torch.equal(old[key], current[key])
    assert not model.initialization["initial_function_preserved"]
    assert not hasattr(model, "reference")


def test_zero_gates_open_before_upstream_gradients_and_dino_stays_frozen(model_bundle):
    model, *_ = model_bundle
    model.train()
    batch = {"gt_latent": torch.randn(2, 7, 32), "gt_image": torch.randn(2, 8)}
    loss = model(batch)
    assert set(loss) == {"ddpm_loss", "total_loss"}
    loss["total_loss"].backward()
    first = model.policy.net1.layers[0]
    assert first.modulation[2].weight.grad.abs().sum() > 0
    assert first.modulation[0].weight.grad.count_nonzero() == 0
    assert first.self_attn.in_proj_weight.grad.count_nonzero() == 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.policy.parameters())
    assert all(not p.requires_grad and p.grad is None for p in model.image_encoder.parameters())
    assert not model.image_encoder.training
    torch.optim.SGD(model.policy.parameters(), lr=.01).step()
    model.zero_grad()
    model(batch)["total_loss"].backward()
    assert first.modulation[0].weight.grad.abs().sum() > 0
    assert first.self_attn.in_proj_weight.grad.abs().sum() > 0
    assert first.linear1.weight.grad.abs().sum() > 0


def test_open_gates_train_eval_cache_online_and_slot_permutation(model_bundle):
    model, *_ = model_bundle
    with torch.no_grad():
        for layer in model.policy.net1.layers:
            layer.modulation[2].weight.normal_(std=.1)
    x, time, images = torch.randn(2, 7, 32), torch.tensor([1, 900]), torch.randn(2, 8)
    features = model.image_encoder(images).detach()
    condition = model.policy.image_condition(features)
    expected = model.policy.train()(x, time, condition)
    torch.testing.assert_close(model.policy.eval()(x, time, condition), expected, atol=1e-6, rtol=1e-6)
    order = torch.randperm(7)
    torch.testing.assert_close(model.policy(x[:, order], time, condition), expected[:, order], atol=1e-6, rtol=1e-6)
    model.train()
    torch.manual_seed(29)
    online = model({"gt_latent": x, "gt_image": images})["total_loss"]
    torch.manual_seed(29)
    cached = model({"gt_latent": x, "image_features": features})["total_loss"]
    torch.testing.assert_close(online, cached, atol=0, rtol=0)


def test_complete_checkpoint_load_avoids_initial_sources(model_bundle, tmp_path):
    model, cfg, legacy, provenance = model_bundle
    saved = {"hyper_parameters": {"model": cfg},
             "state_dict": {f"model.{key}": value for key, value in model.state_dict().items()}}
    path = tmp_path / "full.ckpt"
    torch.save(saved, path)
    with patch("src.models.diffusion.adaln_zero_photo.build_legacy_svr",
               return_value=(deepcopy(legacy), provenance)), \
            patch("src.models.diffusion.adaln_zero_photo.load_photo_svr") as old_source:
        restored = AdaLNZeroPhotoDiffusion({**cfg, "restoring_checkpoint": True, "checkpoint": str(path)})
    old_source.assert_not_called()
    assert all(torch.equal(value, restored.state_dict()[key]) for key, value in model.state_dict().items())
    state = dict(model.state_dict())
    del state["policy.net1.layers.0.modulation.2.bias"]
    with pytest.raises(RuntimeError, match="Missing key"):
        restored.load_state_dict(state, strict=True)
    with pytest.raises(ValueError, match="architecture/source"):
        check_complete_checkpoint({**cfg, "photo_global_step": 5000}, saved)


def make_gt_data(root):
    cache = root / "cache"
    cache.mkdir()
    cache_manifest = {"version": "legacy_photo_rgb_v1", "complete": True, "image_key": "flux",
                      "rotation_id": 0, "checkpoint_sha256": "original", "splits": {}}
    for split, ids in (("training", ["train1", "train2"]), ("validation", ["val1"])):
        arrays = {"features": np.ones((len(ids), 1024), np.float32),
                  "means": np.arange(len(ids) * 960, dtype=np.float32).reshape(len(ids), 30, 32),
                  "counts": np.full(len(ids), 2, np.int16)}
        info = {"ids": ids, "files": {}, "sha256": {}}
        for field, array in arrays.items():
            path = cache / f"{split}_{field}.npy"
            np.save(path, array)
            info["files"][field], info["sha256"][field] = path.name, sha256_file(path)
        cache_manifest["splits"][split] = info
    (cache / "manifest.json").write_text(json.dumps(cache_manifest))
    manifest = {"version": "photo_gt_validation_v1", "original_checkpoint_sha256": "original",
                "photo_checkpoint_sha256": "photo", "photo_global_step": 310000,
                "train_cache_manifest_sha256": sha256_file(cache / "manifest.json"),
                "condition": {"filename": "real_photo.npz", "key": "flux", "rotation_id": 0,
                              "shape": [512, 512, 3], "dtype": "uint8"}, "records": []}
    for kind, ids in (("train", ["train1", "train2"]), ("validation", ["val1", "val2", "val3"])):
        path = root / f"{kind}.txt"
        path.write_text("\n".join(ids) + "\n")
        manifest[f"{kind}_split"] = {"path": str(path), "sha256": sha256_file(path), "count": len(ids)}
    for identity in ("val1", "val2", "val3"):
        image_dir = root / "images" / identity
        image_dir.mkdir(parents=True)
        image, latent = image_dir / "real_photo.npz", root / f"{identity}.npy"
        np.savez(image, flux=np.full((512, 512, 3), 128, np.uint8))
        np.save(latent, np.ones((2, 64), np.float32))
        manifest["records"].append({"image_id": identity, "image_path": str(image),
                                    "image_sha256": sha256_file(image), "latent_path": str(latent),
                                    "latent_sha256": sha256_file(latent), "num_faces": 2})
    path = root / "full_validation.json"
    path.write_text(json.dumps(manifest))
    return {"cache_root": str(cache), "condition_data_root": str(root / "images"),
            "validation_manifest": str(path), "train_dataset": str(root / "train.txt"),
            "val_dataset": str(root / "validation.txt"), "epoch_scale_factor": 2}


def test_gt_loader_uses_full_validation_and_no_pair_paths(tmp_path):
    cfg = make_gt_data(tmp_path)
    cfg["provenance"] = photo_gt_data_provenance(cfg)
    train, val = PhotoGTOnlyDataset("training", cfg), PhotoGTOnlyDataset("validation", cfg)
    assert len(train) == 4 and len(val) == 3  # Cached validation has only one object.
    assert set(train[0]) == {"v_prefix", "gt_latent", "image_features"}
    last = val[2]
    assert last["v_prefix"] == "val3" and last["gt_image"].shape == (3, 224, 224)
    assert last["gt_latent"].shape == (30, 32)
    assert cfg["provenance"]["validation_count"] == 3
    assert cfg["provenance"]["photo_cache"]["validation_count"] == 1
    Path(val.records[2]["image_path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        val[2]


def test_changed_manifest_and_incomplete_validation_are_rejected(tmp_path):
    cfg = make_gt_data(tmp_path)
    cfg["provenance"] = photo_gt_data_provenance(cfg)
    path = Path(cfg["validation_manifest"])
    manifest = json.loads(path.read_text())
    manifest["records"].pop()
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="complete split"):
        PhotoGTOnlyDataset("validation", cfg)


def test_trainer_full_resume_restores_model_and_optimizer_without_old_sources(model_bundle, tmp_path):
    from src.train import UnifiedTrainer

    model, model_cfg, legacy, provenance = model_bundle
    dataset_cfg = make_gt_data(tmp_path)
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1] / "configs"), version_base="1.3"):
        cfg = compose("train_diffusion_adaln_zero_photo", overrides=[
            f"trainer.run_dir={tmp_path / 'run'}", "trainer.devices=1", "trainer.max_steps=1",
            *[f"dataset.{key}={value}" for key, value in dataset_cfg.items()],
        ])
    for key, value in model_cfg.items():
        OmegaConf.update(cfg.model, key, value, force_add=True)
    OmegaConf.update(cfg.model, "initialization", model.initialization, force_add=True)
    OmegaConf.update(cfg.dataset, "provenance", photo_gt_data_provenance(dataset_cfg), force_add=True)
    # A real Adam update gives nonempty moments, including newly opened gates.
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=1e-5)
    model({"gt_latent": torch.randn(2, 7, 32), "image_features": torch.randn(2, 8)})["total_loss"].backward()
    optimizer.step()
    saved = {"hyper_parameters": OmegaConf.to_container(cfg, resolve=True),
             "state_dict": {f"model.{key}": value for key, value in model.state_dict().items()},
             "optimizer_states": [optimizer.state_dict()], "global_step": 1}
    path = tmp_path / "resume.ckpt"
    torch.save(saved, path)
    cfg.trainer.resume_from_checkpoint = str(path)
    cfg.model.init_photo_checkpoint = "removed-photo.ckpt"
    cfg.model.legacy_checkpoint = "removed-original.ckpt"
    with patch("src.models.diffusion.adaln_zero_photo.build_legacy_svr",
               return_value=(deepcopy(legacy), provenance)), \
            patch("src.models.diffusion.adaln_zero_photo.load_photo_svr") as old_source:
        trainer = UnifiedTrainer(cfg)
        trainer.on_load_checkpoint(saved)
        trainer.load_state_dict(saved["state_dict"], strict=True)
        restored_optimizer = trainer.configure_optimizers()
        restored_optimizer.load_state_dict(saved["optimizer_states"][0])
    old_source.assert_not_called()
    assert all(torch.equal(value, trainer.model.state_dict()[key]) for key, value in model.state_dict().items())
    for old, new in zip(optimizer.state.values(), restored_optimizer.state.values()):
        assert all(torch.equal(value, new[key]) for key, value in old.items())
    assert cfg.trainer.check_val_every_n_epoch is None and cfg.trainer.val_check_interval == 1000


def test_sampling_preserves_cpu_initial_noise_and_step_generator(model_bundle):
    model, *_ = model_bundle
    model.eval()
    scheduler = model.noise_scheduler

    def one_step(count, device):
        assert count == 1000
        scheduler.timesteps = torch.tensor([0], device=device)

    generator = torch.Generator().manual_seed(12)
    with patch.object(scheduler, "set_timesteps", side_effect=one_step), \
            patch.object(scheduler, "step", wraps=scheduler.step) as step:
        result = model.sample_latents(torch.randn(2, 8), initial_generator=generator,
                                      step_generator=generator)
    assert result.shape == (2, 30, 32) and step.call_args.kwargs["generator"] is generator
    expected = torch.randn((2, 30, 32), generator=torch.Generator().manual_seed(12))
    torch.testing.assert_close(step.call_args.args[2], expected, atol=0, rtol=0)
