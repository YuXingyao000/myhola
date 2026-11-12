"""已学门控的完整初始化、冻结 reference，以及无需源权重的 DPO 恢复。"""

from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import pytest
import torch

from src.datasets.preference import preference_data_provenance
from src.models.diffusion.adaln_zero_photo import AdaLNZeroPhotoDiffusion
from src.models.diffusion.adaln_zero_preference import check_initialization_data
from src.train import UnifiedTrainer
from tests.test_adaln_zero_photo import tiny_legacy
from tests.test_photo_dpo import make_photo_pool


@pytest.fixture
def source_and_config(tmp_path):
    torch.manual_seed(42)
    dataset = make_photo_pool(tmp_path)
    pairs = preference_data_provenance(dataset)
    source_data = {
        "version": "photo_gt_only_v1", "photo_cache": pairs["photo_cache"],
        "original_checkpoint_sha256": pairs["original_checkpoint_sha256"],
        "train_split_sha256": pairs["train_list_sha256"],
        "validation_split_sha256": pairs["validation_list_sha256"],
        "condition_data_root": dataset["condition_data_root"],
    }
    legacy, provenance = tiny_legacy(), {"model_config": {"test": True}, "photo": {"global_step": 310000}}
    source_cfg = {"name": "AdaLNZeroPhotoDiffusion", "modulation_hidden_width": 5,
                  "legacy_checkpoint": "original", "init_photo_checkpoint": "photo",
                  "photo_global_step": 310000, "legacy_model_config": {"test": True}}
    with patch("src.models.diffusion.adaln_zero_photo.load_photo_svr",
               return_value=(deepcopy(legacy), deepcopy(provenance))):
        source = AdaLNZeroPhotoDiffusion(source_cfg)
    # 模拟 GT 训练已打开的 gates，确保 DPO 初始化不会再次将它们归零。
    with torch.no_grad():
        for layer in source.policy.net1.layers:
            layer.modulation[2].weight.normal_(std=.1)
            layer.modulation[2].bias.normal_(std=.1)
    path = tmp_path / "adaln80k.ckpt"
    torch.save({"hyper_parameters": {"model": source_cfg, "dataset": {"provenance": source_data}},
                "global_step": 80000,
                "state_dict": {f"model.{key}": value for key, value in source.state_dict().items()}}, path)
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1] / "configs"), version_base="1.3"):
        cfg = compose("train_diffusion_adaln_zero_dpo", overrides=[
            f"trainer.run_dir={tmp_path / 'training'}", "trainer.devices=1", "trainer.max_steps=1",
            f"model.init_adaln_checkpoint={path}", "model.modulation_hidden_width=5",
        ])
    for key, value in dataset.items():
        OmegaConf.update(cfg.dataset, key, value, force_add=True)
    return source, cfg, legacy, provenance, source_data, pairs


def build_trainer(cfg, legacy, provenance):
    with patch("src.models.diffusion.adaln_zero_photo.build_legacy_svr",
               return_value=(deepcopy(legacy), deepcopy(provenance))), \
            patch("src.models.diffusion.adaln_zero_photo.load_photo_svr") as old_source:
        module = UnifiedTrainer(cfg)
    old_source.assert_not_called()
    return module


def test_complete_learned_policy_is_reference_and_both_pair_paths_agree(source_and_config):
    source, cfg, legacy, provenance, _, pairs = source_and_config
    module = build_trainer(cfg, legacy, provenance)
    model = module.model.train()
    for key, value in source.state_dict().items():
        assert torch.equal(value, model.state_dict()[key])
    for key, value in model.policy.state_dict().items():
        assert torch.equal(value, model.reference.state_dict()[key])
        assert value.data_ptr() != model.reference.state_dict()[key].data_ptr()
    assert cfg.dataset.provenance.source_checkpoint_sha256 == pairs["source_checkpoint_sha256"] == "photo"
    assert cfg.model.initialization.global_step == 80000
    assert cfg.model.initialization.sha256 != pairs["source_checkpoint_sha256"]
    batch = {"gt_latent": torch.randn(2, 30, 32), "image_features": torch.randn(2, 8),
             "winner_latent": torch.randn(2, 30, 32), "loser_latent": torch.randn(2, 30, 32),
             "pair_image_features": torch.randn(2, 8)}
    result = model(batch)
    # no_grad reference may use a different MHA kernel from the gradient path.
    assert result["preference_logit"].item() == pytest.approx(0, abs=1e-5)
    assert result["preference_loss"].item() == pytest.approx(0.69314718)
    torch.testing.assert_close(result["winner_mse"], result["reference_winner_mse"], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(result["loser_mse"], result["reference_loser_mse"], atol=1e-6, rtol=1e-6)
    result["total_loss"].backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.policy.parameters())
    assert all(not p.requires_grad and p.grad is None for p in model.reference.parameters())
    assert all(not p.requires_grad and p.grad is None for p in model.image_encoder.parameters())
    assert not model.reference.training and not model.image_encoder.training
    assert all(layer.self_attn.dropout == 0 for layer in model.policy.net1.layers)


def test_full_resume_preserves_updated_policy_reference_and_optimizer(source_and_config, tmp_path):
    _, cfg, legacy, provenance, *_ = source_and_config
    module = build_trainer(cfg, legacy, provenance)
    optimizer = module.configure_optimizers()
    batch = {"gt_latent": torch.randn(2, 30, 32), "image_features": torch.randn(2, 8),
             "winner_latent": torch.randn(2, 30, 32), "loser_latent": torch.randn(2, 30, 32),
             "pair_image_features": torch.randn(2, 8)}
    module.model(batch)["total_loss"].backward()
    optimizer.step()
    saved = {"hyper_parameters": dict(module.hparams), "state_dict": module.state_dict(),
             "optimizer_states": [optimizer.state_dict()], "global_step": 1}
    path = tmp_path / "dpo.ckpt"
    torch.save(saved, path)
    Path(cfg.model.init_adaln_checkpoint).unlink()
    cfg.trainer.resume_from_checkpoint = str(path)
    restored = build_trainer(cfg, legacy, provenance)
    restored.on_load_checkpoint(saved)
    restored.load_state_dict(saved["state_dict"], strict=True)
    restored_optimizer = restored.configure_optimizers()
    restored_optimizer.load_state_dict(saved["optimizer_states"][0])
    for key, value in module.state_dict().items():
        assert torch.equal(value, restored.state_dict()[key])
    for old, new in zip(optimizer.state.values(), restored_optimizer.state.values()):
        assert all(torch.equal(value, new[key]) for key, value in old.items())
    bad_state = dict(saved["state_dict"])
    del bad_state["model.reference.net1.layers.0.modulation.2.bias"]
    with pytest.raises(RuntimeError, match="Missing key"):
        restored.load_state_dict(bad_state, strict=True)
    cfg.model.beta_dpo = 21
    with pytest.raises(ValueError, match="architecture/reference/objective"):
        build_trainer(cfg, legacy, provenance)


def test_complete_dpo_weights_support_sampling_without_initial_checkpoint(source_and_config, tmp_path):
    _, cfg, legacy, provenance, *_ = source_and_config
    module = build_trainer(cfg, legacy, provenance)
    path = tmp_path / "sample.ckpt"
    torch.save({"hyper_parameters": dict(module.hparams), "state_dict": module.state_dict()}, path)
    Path(cfg.model.init_adaln_checkpoint).unlink()
    cfg.model.init_adaln_checkpoint = None
    cfg.model.checkpoint = str(path)
    model = build_trainer(cfg, legacy, provenance).model.eval()
    scheduler = model.noise_scheduler

    def one_step(count, device):
        assert count == 1000
        scheduler.timesteps = torch.tensor([0], device=device)

    with patch.object(scheduler, "set_timesteps", side_effect=one_step), \
            patch.object(model.reference, "forward", side_effect=AssertionError("reference used during sampling")):
        output = model.sample_latents(torch.randn(2, 8), initial_generator=torch.Generator().manual_seed(12),
                                      step_generator=torch.Generator().manual_seed(34))
    assert output.shape == (2, 30, 32) and torch.isfinite(output).all()


def test_off_policy_source_identity_does_not_bypass_shared_representation(source_and_config):
    _, cfg, _, _, source, pairs = source_and_config
    check_initialization_data(source, pairs, cfg.dataset.condition_data_root)
    with pytest.raises(ValueError, match="share DINO/AE"):
        check_initialization_data(source, {**pairs, "original_checkpoint_sha256": "different-ae"},
                                  cfg.dataset.condition_data_root)
