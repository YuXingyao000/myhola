"""已学 AdaLN 的零增量等价、跨冻结主干反传、在线图片来源与恢复。"""

from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import patch

from hydra import compose, initialize_config_dir
import numpy as np
from omegaconf import OmegaConf
import pytest
import torch
from torch import nn
import torch.nn.functional as F

from src.common.io import sha256_file
from src.datasets.photo import PhotoGTImageDataset, photo_gt_data_provenance, photo_gt_image_provenance
from src.models.diffusion.adaln_zero_photo import AdaLNZeroPhotoDiffusion
from src.models.diffusion.controlnet_photo import (
    ControlNetPhotoDiffusion, check_controlnet_checkpoint, load_controlnet_photo,
)
from tests.test_adaln_zero_photo import make_gt_data, tiny_legacy


class TinyDino(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(3, 8)

    def forward_features(self, images):
        patches = self.projection(F.adaptive_avg_pool2d(images, (2, 2)).flatten(2).transpose(1, 2))
        return {"x_norm_clstoken": patches.mean(1), "x_norm_patchtokens": patches,
                "x_norm_regtokens": patches.new_full((len(images), 2, 8), 1000)}

    def forward(self, images):
        return self.forward_features(images)["x_norm_clstoken"]


def online_gt_data(root):
    cfg = make_gt_data(root)
    cache = Path(cfg["cache_root"])
    manifest = json.loads((cache / "manifest.json").read_text())
    inputs = []
    for index, identity in enumerate(manifest["splits"]["training"]["ids"]):
        folder = Path(cfg["condition_data_root"]) / identity
        folder.mkdir(parents=True)
        image_path = folder / "real_photo.npz"
        np.savez(image_path, flux=np.full((512, 512, 3), index * 50, dtype=np.uint8))
        inputs.append({"split": "training", "index": index, "id": identity,
                       "photo_sha256": sha256_file(image_path)})
    inputs_path = cache / "inputs.jsonl"
    inputs_path.write_text("".join(json.dumps(row) + "\n" for row in inputs))
    manifest["inputs_sha256"] = sha256_file(inputs_path)
    (cache / "manifest.json").write_text(json.dumps(manifest))
    validation_path = Path(cfg["validation_manifest"])
    validation = json.loads(validation_path.read_text())
    validation["train_cache_manifest_sha256"] = sha256_file(cache / "manifest.json")
    validation_path.write_text(json.dumps(validation))
    return cfg


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    torch.manual_seed(12)
    data_cfg = online_gt_data(tmp_path)
    legacy = tiny_legacy()
    legacy.img_model = TinyDino()
    base_cfg = {"name": "AdaLNZeroPhotoDiffusion", "modulation_hidden_width": 5,
                "photo_global_step": 310000, "legacy_model_config": {"test": True}}
    monkeypatch.setattr("src.models.diffusion.adaln_zero_photo.build_legacy_svr",
                        lambda *args, **kwargs: (deepcopy(legacy), {"model_config": {"test": True}}))
    monkeypatch.setattr("src.models.diffusion.adaln_zero_photo.load_photo_svr",
                        lambda *args, **kwargs: pytest.fail("Must not reload old initialization sources"))
    baseline = AdaLNZeroPhotoDiffusion({**base_cfg, "restoring_checkpoint": True})
    # 模拟已学 checkpoint：attention/FFN 门已打开，不能靠恒等块蒙混过关。
    with torch.no_grad():
        for layer in baseline.policy.net1.layers:
            layer.modulation[2].weight.normal_(std=.08)
            layer.modulation[2].bias.normal_(std=.08)
    source_path = tmp_path / "adaln.ckpt"
    torch.save({"global_step": 80000,
                "hyper_parameters": {"model": base_cfg, "dataset": {
                    "provenance": photo_gt_data_provenance(data_cfg)}},
                "state_dict": {f"model.{key}": value for key, value in baseline.state_dict().items()}}, source_path)
    cfg = {"name": "ControlNetPhotoDiffusion", "architecture_version": "adaln_patch_control_v1",
           "init_adaln_checkpoint": str(source_path), "checkpoint": None}
    model = ControlNetPhotoDiffusion(cfg)
    cfg.update(backbone_config=model.backbone_config, initialization=model.initialization)
    return model, cfg, baseline, data_cfg


def test_zero_controls_preserve_learned_model_and_have_independent_storage(bundle):
    model, _, baseline, _ = bundle
    images, latent, time = torch.randn(2, 3, 8, 8), torch.randn(2, 7, 32), torch.tensor([20, 700])
    condition, patches = model.image_condition(images)
    assert patches.shape == (2, 4, 8)  # 不含两枚故意放大的register tokens。
    old_condition = baseline.policy.image_condition(baseline.image_encoder(images))
    torch.testing.assert_close(condition, old_condition, atol=0, rtol=0)
    # 同一eval/no-grad模式，避免CPU MHA训练/推理kernel的舍入差异。
    baseline.eval()
    model.eval()
    with torch.no_grad():
        expected = baseline.policy(latent, time, old_condition)
        torch.testing.assert_close(model.denoiser(latent, time, condition, patches), expected, atol=0, rtol=0)
    for original, control in zip(model.denoiser.backbone.net1.layers, model.denoiser.control_blocks):
        for left, right in zip(original.parameters(), control.block.parameters()):
            assert torch.equal(left, right) and left.data_ptr() != right.data_ptr()
            assert not left.requires_grad and right.requires_grad
        assert control.zero_out.weight.count_nonzero() == control.zero_out.bias.count_nonzero() == 0


def test_gradients_open_control_without_updating_frozen_weights(bundle):
    model, _, _, _ = bundle
    model.train()
    batch = {"gt_latent": torch.randn(2, 7, 32), "gt_image": torch.randn(2, 3, 8, 8)}
    frozen = {name: value.detach().clone() for name, value in model.named_parameters() if not value.requires_grad}
    optimizer = torch.optim.SGD((p for p in model.parameters() if p.requires_grad), lr=.01)
    model(batch)["total_loss"].backward()
    for block in model.denoiser.control_blocks:
        assert block.zero_out.weight.grad.abs().sum() > 0
        assert block.cross_attention.q_proj_weight.grad.count_nonzero() == 0
        assert block.block.modulation[2].weight.grad.count_nonzero() == 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.parameters() if p.requires_grad)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    model(batch)["total_loss"].backward()
    first = model.denoiser.control_blocks[0]
    assert first.zero_out.weight.grad.abs().sum() > 0  # 梯度穿过后续冻结层。
    assert first.cross_attention.q_proj_weight.grad.abs().sum() > 0
    assert first.block.modulation[0].weight.grad.abs().sum() > 0
    for name, parameter in model.named_parameters():
        if name in frozen:
            assert parameter.grad is None and torch.equal(parameter, frozen[name])
    assert not model.image_encoder.training and not model.denoiser.backbone.training
    assert model.denoiser.control_blocks[0].training


def test_nonzero_control_is_slot_equivariant_and_responds_to_patches(bundle):
    model, *_ = bundle
    with torch.no_grad():
        for layer in model.denoiser.control_blocks:
            layer.zero_out.weight.normal_(std=.1)
    latent, time, images = torch.randn(2, 7, 32), torch.tensor([10, 900]), torch.randn(2, 3, 8, 8)
    condition, patches = model.image_condition(images)
    expected = model.denoiser(latent, time, condition, patches)
    order = torch.randperm(7)
    torch.testing.assert_close(model.denoiser(latent[:, order], time, condition, patches),
                               expected[:, order], atol=1e-6, rtol=1e-6)
    different_image = patches.flip(0)
    assert not torch.allclose(model.denoiser(latent, time, condition, different_image), expected)
    torch.testing.assert_close(model.denoiser.eval()(latent, time, condition, patches), expected)


def test_full_load_is_self_contained_and_strict(bundle, tmp_path):
    model, cfg, _, _ = bundle
    with torch.no_grad():
        model.denoiser.control_blocks[0].zero_out.weight.normal_()
    saved = {"hyper_parameters": {"model": cfg},
             "state_dict": {f"model.{key}": value for key, value in model.state_dict().items()}}
    path = tmp_path / "control.ckpt"
    torch.save(saved, path)
    Path(cfg["init_adaln_checkpoint"]).unlink()
    restored = load_controlnet_photo(path)
    for key, value in model.state_dict().items():
        assert torch.equal(value, restored.state_dict()[key])
    with pytest.raises(ValueError, match="architecture"):
        check_controlnet_checkpoint({**cfg, "architecture_version": "changed"}, saved)
    del saved["state_dict"]["model.denoiser.control_blocks.0.zero_out.bias"]
    torch.save(saved, path)
    with pytest.raises(RuntimeError, match="Missing key"):
        load_controlnet_photo(path)


def test_online_training_preserves_gt_and_full_validation(bundle):
    _, _, _, cfg = bundle
    cfg = {**cfg, "provenance": photo_gt_image_provenance(cfg)}
    train, val = PhotoGTImageDataset("training", cfg), PhotoGTImageDataset("validation", cfg)
    assert len(train) == 4 and len(val) == 3  # cache validation只有1件。
    torch.manual_seed(7)
    item = train[1]
    torch.manual_seed(7)
    cached = train.cache[1]
    assert torch.equal(item["gt_latent"], cached["gt_latent"])
    assert set(item) == {"v_prefix", "gt_latent", "gt_image"}
    assert item["gt_image"].shape == (3, 224, 224)
    assert val[2]["v_prefix"] == "val3"
    image_path = Path(cfg["condition_data_root"]) / "train2" / "real_photo.npz"
    image_path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="differs from GT source"):
        train[1]
    inputs = Path(cfg["cache_root"]) / "inputs.jsonl"
    inputs.write_text(inputs.read_text() + "\n")
    with pytest.raises(ValueError, match="manifest changed"):
        photo_gt_image_provenance(cfg)


def test_trainer_resume_restores_optimizer_without_initialization_source(bundle, tmp_path):
    from src.train import UnifiedTrainer

    _, model_cfg, _, data_cfg = bundle
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1] / "configs"), version_base="1.3"):
        cfg = compose("train_diffusion_controlnet_photo", overrides=[
            f"trainer.run_dir={tmp_path / 'run'}", "trainer.devices=1", "trainer.batch_size=2",
            *[f"dataset.{key}={value}" for key, value in data_cfg.items()],
        ])
    cfg.model.init_adaln_checkpoint = model_cfg["init_adaln_checkpoint"]
    trainer = UnifiedTrainer(cfg)
    optimizer = trainer.configure_optimizers()
    trainer.model({"gt_latent": torch.randn(2, 7, 32), "gt_image": torch.randn(2, 3, 8, 8)})["total_loss"].backward()
    optimizer.step()
    saved = {"hyper_parameters": OmegaConf.to_container(cfg, resolve=True),
             "state_dict": trainer.state_dict(), "optimizer_states": [optimizer.state_dict()], "global_step": 1}
    path = tmp_path / "resume.ckpt"
    torch.save(saved, path)
    Path(model_cfg["init_adaln_checkpoint"]).unlink()
    cfg.trainer.resume_from_checkpoint = str(path)
    restored = UnifiedTrainer(cfg)
    restored.on_load_checkpoint(saved)
    restored.load_state_dict(saved["state_dict"], strict=True)
    restored_optimizer = restored.configure_optimizers()
    restored_optimizer.load_state_dict(saved["optimizer_states"][0])
    for key, value in trainer.state_dict().items():
        assert torch.equal(value, restored.state_dict()[key])
    for old, new in zip(optimizer.state.values(), restored_optimizer.state.values()):
        assert all(torch.equal(value, new[key]) for key, value in old.items())
    saved["hyper_parameters"]["dataset"]["provenance"]["inputs_sha256"] = "changed"
    with pytest.raises(ValueError, match="provenance"):
        restored.on_load_checkpoint(saved)


def test_sampling_encodes_image_once_and_preserves_noise_generators(bundle):
    model, *_ = bundle
    model.eval()
    scheduler = model.noise_scheduler

    def one_step(count, device):
        assert count == 1000
        scheduler.timesteps = torch.tensor([0], device=device)

    generator = torch.Generator().manual_seed(12)
    with patch.object(scheduler, "set_timesteps", side_effect=one_step), \
            patch.object(scheduler, "step", wraps=scheduler.step) as step, \
            patch.object(model.image_encoder, "forward_features", wraps=model.image_encoder.forward_features) as encode:
        result = model.sample_latents(torch.randn(2, 3, 8, 8), initial_generator=generator, step_generator=generator)
    assert encode.call_count == 1
    assert result.shape == (2, 30, 32) and torch.isfinite(result).all()
    assert step.call_args.kwargs["generator"] is generator
    expected = torch.randn((2, 30, 32), generator=torch.Generator().manual_seed(12))
    torch.testing.assert_close(step.call_args.args[2], expected, atol=0, rtol=0)
