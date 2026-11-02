"""保留旧函数、条件作用、严格迁移与恢复；小尺寸网络无需 DINO/CUDA。"""

from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import patch

from diffusers import DDPMScheduler
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import pytest
import torch
from torch import nn

from src.models.diffusion.adaln import AdaLNSVRDenoiser
from src.models.diffusion.adaln_preference import (
    AdaLNPreferenceDiffusion, check_adaln_checkpoint, check_initialization_data,
)
from src.models.diffusion.preference import LegacySVRDenoiser, PreferenceDiffusion


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
def models():
    torch.manual_seed(13)
    legacy = tiny_legacy()
    provenance = {"model_config": {"test": True}, "photo": {"global_step": 310000}}
    cfg = {"name": "AdaLNPreferenceDiffusion", "model_source": "photo", "legacy_checkpoint": "original",
           "photo_checkpoint": "photo", "photo_global_step": 310000, "beta_dpo": 20.,
           "preference_weight": 1., "modulation_hidden_width": 5, "legacy_model_config": {"test": True}}
    with patch("src.models.diffusion.photo_baseline.load_photo_svr",
               side_effect=lambda *a, **kw: (deepcopy(legacy), deepcopy(provenance))):
        baseline = PreferenceDiffusion(cfg)
        candidate = AdaLNPreferenceDiffusion(cfg)
    return baseline, candidate, cfg, legacy, provenance


@pytest.mark.parametrize("training", [False, True])
def test_zero_modulation_preserves_original_output_and_dropout(training):
    baseline = LegacySVRDenoiser(tiny_legacy()).train(training)
    candidate = AdaLNSVRDenoiser(deepcopy(baseline), hidden_width=5).train(training)
    latent, time, condition = torch.randn(2, 7, 32), torch.tensor([0, 932]), torch.randn(2, 1, 4)
    torch.manual_seed(91)
    expected = baseline(latent, time, condition)
    torch.manual_seed(91)
    actual = candidate(latent, time, condition)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    original_keys = set(baseline.state_dict())
    assert original_keys == {key for key in candidate.state_dict() if not key.startswith("modulations.")}
    for key, value in baseline.state_dict().items():
        assert torch.equal(value, candidate.state_dict()[key])


def test_nonzero_image_time_modulation_changes_output_and_preserves_permutation():
    candidate = AdaLNSVRDenoiser(LegacySVRDenoiser(tiny_legacy()), hidden_width=5).eval()
    latent, time, condition = torch.randn(2, 7, 32), torch.tensor([1, 900]), torch.randn(2, 1, 4)
    before = candidate(latent, time, condition)
    with torch.no_grad():
        for modulation in candidate.modulations:
            modulation.output.weight.normal_(std=.1)
    after = candidate(latent, time, condition)
    assert not torch.allclose(before, after)
    order = torch.randperm(7)
    torch.testing.assert_close(candidate(latent[:, order], time, condition), after[:, order], atol=1e-6, rtol=1e-6)
    context = torch.randn(2, 20)
    first = torch.cat(candidate.modulations[0](context), -1)
    for start, stop in ((0, 4), (4, 20)):
        changed = context.clone()
        changed[:, start:stop] += 1
        assert not torch.allclose(first, torch.cat(candidate.modulations[0](changed), -1))


def test_dpo_loss_and_gradients_keep_reference_frozen(models):
    baseline, candidate, *_ = models
    baseline.train()
    candidate.train()
    batch = {"gt_latent": torch.randn(2, 7, 32), "winner_latent": torch.randn(2, 7, 32),
             "loser_latent": torch.randn(2, 7, 32), "image_features": torch.randn(2, 8),
             "pair_image_features": torch.randn(2, 8)}
    torch.manual_seed(42)
    expected = baseline(batch)
    torch.manual_seed(42)
    actual = candidate(batch)
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], atol=1e-5, rtol=1e-5)
    actual["total_loss"].backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in candidate.policy.parameters())
    assert all(p.grad is None for p in candidate.reference.parameters())
    assert all(p.grad is None for p in candidate.image_encoder.parameters())
    for modulation in candidate.policy.modulations:
        assert modulation.output.weight.grad.abs().sum() > 0
        assert modulation.input.weight.grad.count_nonzero() == 0  # Zero final layer blocks upstream gradient initially.
    assert candidate.policy.net1.layers[0].self_attn.in_proj_weight.grad.abs().sum() > 0
    torch.optim.SGD(candidate.policy.parameters(), lr=.01).step()
    candidate.zero_grad()
    candidate(batch)["total_loss"].backward()
    assert candidate.policy.modulations[0].input.weight.grad.abs().sum() > 0
    assert not candidate.reference.training and not candidate.image_encoder.training
    assert set(candidate(batch, v_test=True)) == {"ddpm_loss", "total_loss"}


def provenance():
    return {"protocol_version": "photo_preference_generation_v1", "source_checkpoint_sha256": "photo",
            "original_checkpoint_sha256": "original", "photo_global_step": 310000,
            "pairs_manifest_sha256": "train-pairs", "validation_pairs_manifest_sha256": "val-pairs",
            "train_list_sha256": "train", "validation_list_sha256": "validation",
            "generation_protocol_sha256": "old-sha", "generation_fingerprint": "old-fingerprint",
            "photo_cache": {"manifest": "old/cache/manifest.json", "manifest_sha256": "cache-sha"}}


def source_checkpoint(baseline, cfg):
    return {"hyper_parameters": {"model": {**cfg, "name": "PreferenceDiffusion"},
                                  "dataset": {"provenance": provenance()}},
            "state_dict": {f"model.{key}": value.clone() for key, value in baseline.state_dict().items()},
            "global_step": 5000, "epoch": 6}


def test_initialize_inherits_dpo_policy_and_original_reference(models, tmp_path):
    baseline, candidate, cfg, *_ = models
    with torch.no_grad():
        baseline.policy.fc_out[-1].bias.add_(.2)
    saved = source_checkpoint(baseline, cfg)
    path = tmp_path / "dpo.ckpt"
    torch.save(saved, path)
    result = candidate.initialize_preference_weights(path, provenance())
    assert result["source_global_step"] == 5000 and not result["optimizer_restored"]
    for key, value in baseline.state_dict().items():
        assert torch.equal(value, candidate.state_dict()[key])
    assert not torch.equal(candidate.policy.fc_out[-1].bias, candidate.reference.fc_out[-1].bias)
    assert all(mod.output.weight.count_nonzero() == 0 for mod in candidate.policy.modulations)


@pytest.mark.parametrize("change", ["missing", "extra", "shape", "dtype", "reference", "dino", "objective"])
def test_initialize_rejects_incomplete_or_changed_source(models, tmp_path, change):
    baseline, candidate, cfg, *_ = models
    saved = source_checkpoint(baseline, cfg)
    key = "model.policy.p_embed.0.weight"
    if change == "missing":
        del saved["state_dict"][key]
    elif change == "extra":
        saved["state_dict"]["model.policy.unexpected"] = torch.zeros(1)
    elif change == "shape":
        saved["state_dict"][key] = torch.zeros(1)
    elif change == "dtype":
        saved["state_dict"][key] = saved["state_dict"][key].double()
    elif change == "reference":
        saved["state_dict"]["model.reference.fc_out.0.bias"] += 1
    elif change == "dino":
        saved["state_dict"]["model.image_encoder.bias"] += 1
    else:
        saved["hyper_parameters"]["model"]["beta_dpo"] = 1
    path = tmp_path / "bad.ckpt"
    torch.save(saved, path)
    before = {key: value.clone() for key, value in candidate.state_dict().items()}
    with pytest.raises(ValueError):
        candidate.initialize_preference_weights(path, provenance())
    assert all(torch.equal(value, before[key]) for key, value in candidate.state_dict().items())


def test_relocation_requires_same_content_and_recorded_original_protocol(tmp_path):
    previous = provenance()
    current = deepcopy(previous)
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps({"asset_relocation": {
        "source_protocol_sha256": "old-sha", "source_protocol_fingerprint": "old-fingerprint"}}))
    current.update(generation_protocol=str(protocol_path), generation_protocol_sha256="new-sha",
                   generation_fingerprint="new-fingerprint")
    current["photo_cache"]["manifest"] = "new/cache/manifest.json"
    check_initialization_data(current, previous)
    for field in ("pairs_manifest_sha256", "validation_pairs_manifest_sha256", "source_checkpoint_sha256"):
        with pytest.raises(ValueError, match="same pairs"):
            check_initialization_data({**current, field: "changed"}, previous)
    protocol_path.write_text("{}")
    with pytest.raises(ValueError, match="relocation"):
        check_initialization_data(current, previous)


def test_complete_checkpoint_strict_load_and_architecture_metadata(models, tmp_path):
    _, candidate, cfg, legacy, source_provenance = models
    checkpoint = {"hyper_parameters": {"model": cfg},
                  "state_dict": {f"model.{key}": value for key, value in candidate.state_dict().items()}}
    path = tmp_path / "adaln.ckpt"
    torch.save(checkpoint, path)
    restoring = {**cfg, "restoring_checkpoint": True, "checkpoint": str(path)}
    with patch("src.models.diffusion.legacy.build_legacy_svr", return_value=(deepcopy(legacy), source_provenance)), \
            patch("src.models.diffusion.photo_baseline.load_photo_svr") as forbidden_source:
        restored = AdaLNPreferenceDiffusion(restoring)
    forbidden_source.assert_not_called()
    assert all(torch.equal(value, restored.state_dict()[key]) for key, value in candidate.state_dict().items())
    missing = dict(candidate.state_dict())
    del missing["policy.modulations.0.output.bias"]
    with pytest.raises(RuntimeError, match="Missing key"):
        restored.load_state_dict(missing, strict=True)
    with pytest.raises(ValueError, match="architecture/source/objective"):
        check_adaln_checkpoint({**cfg, "beta_dpo": 2}, checkpoint)


def test_sampling_retains_legacy_noise_generators_and_uses_adaln(models):
    _, model, *_ = models
    model.eval()
    initial = torch.Generator().manual_seed(12)
    steps = torch.Generator().manual_seed(34)
    scheduler = model.noise_scheduler

    def one_step(count, device):
        assert count == 1000
        scheduler.timesteps = torch.tensor([0], device=device)

    with patch.object(scheduler, "set_timesteps", side_effect=one_step), \
            patch.object(scheduler, "step", wraps=scheduler.step) as step, \
            patch.object(model.policy.modulations[0], "forward", wraps=model.policy.modulations[0].forward) as modulation:
        result = model.sample_latents(torch.randn(2, 8), initial_generator=initial, step_generator=steps)
    assert result.shape == (2, 30, 32)
    assert modulation.call_count == 1 and step.call_args.kwargs["generator"] is steps
    expected = torch.randn((2, 30, 32), generator=torch.Generator().manual_seed(12))
    torch.testing.assert_close(step.call_args.args[2], expected, atol=0, rtol=0)


def test_trainer_resume_uses_only_complete_adaln_checkpoint(models, tmp_path):
    from src.train import UnifiedTrainer

    _, candidate, model_cfg, legacy, source_provenance = models
    configs = Path(__file__).resolve().parents[1] / "configs"
    with initialize_config_dir(config_dir=str(configs), version_base="1.3"):
        cfg = compose("train_diffusion_adaln_photo", overrides=[
            f"trainer.run_dir={tmp_path}", "trainer.devices=1", "trainer.max_steps=1",
            *[f"dataset.{key}=unused" for key in (
                "pairs_manifest", "pairs_manifest_sha256", "validation_pairs_manifest",
                "validation_pairs_manifest_sha256", "cached_latent_root", "cache_root", "condition_data_root")],
        ])
    for key, value in model_cfg.items():
        OmegaConf.update(cfg.model, key, value, force_add=True)
    OmegaConf.update(cfg.dataset, "provenance", provenance(), force_add=True)
    path = tmp_path / "resume.ckpt"
    torch.save({"hyper_parameters": OmegaConf.to_container(cfg, resolve=True),
                "state_dict": {f"model.{key}": value for key, value in candidate.state_dict().items()}}, path)
    cfg.trainer.resume_from_checkpoint = str(path)
    cfg.model.init_preference_checkpoint = "removed-old-dpo.ckpt"
    with patch("src.datasets.preference.preference_data_provenance", return_value=provenance()), \
            patch("src.models.diffusion.legacy.build_legacy_svr", return_value=(deepcopy(legacy), source_provenance)), \
            patch("src.models.diffusion.photo_baseline.load_photo_svr") as forbidden_source, \
            patch("src.common.io.sha256_file") as forbidden_hash:
        trainer = UnifiedTrainer(cfg)
        trainer.load_state_dict(torch.load(path, weights_only=False)["state_dict"], strict=True)
    forbidden_source.assert_not_called()
    forbidden_hash.assert_not_called()
    assert all(torch.equal(value, trainer.model.state_dict()[key]) for key, value in candidate.state_dict().items())
