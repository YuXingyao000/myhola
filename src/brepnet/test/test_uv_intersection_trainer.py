"""Training integration must preserve resume, validation, and budget semantics."""

from pathlib import Path
from types import MethodType, SimpleNamespace

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict
import pytest
import torch

from src.brepnet import train as training
from src.brepnet.models import available_models


def config(tmp_path, overrides=()):
    directory = str(Path(__file__).resolve().parents[3] / "configs")
    with initialize_config_dir(version_base="1.3", config_dir=directory):
        return compose(config_name="train_vae_uv_intersection", overrides=[
            "dataset.raw_data_root=unused", f"trainer.output_dir={tmp_path}",
            "trainer.wandb.enabled=false", "trainer.num_threads=null", *overrides,
        ])


def stub_construction(monkeypatch):
    seen = []

    def build(model_cfg, condition):
        seen.append(OmegaConf.to_container(model_cfg, resolve=True))
        return torch.nn.Linear(1, 1)

    monkeypatch.setattr(training, "build_model", build)
    monkeypatch.setattr(training, "_load_dataset_class", lambda _: object)
    return seen


def test_entry_uses_shared_boundary_dataset_new_uv_domain_and_fixed_training_budget(tmp_path):
    cfg = config(tmp_path)
    assert cfg.model.name in available_models()
    assert cfg.dataset.name == "AutoEncoder_joint_dataset"
    assert cfg.dataset.edge_uv_root is None
    for split, name in (("train", "train"), ("val", "validation"), ("test", "test")):
        assert cfg.dataset[f"{split}_dataset"].endswith(f"deepcad_v7_contract_{name}.txt")
    assert cfg.model.freeze_encoder and cfg.model.latent_channels == 16
    assert cfg.model.uv_intersection.latent_dim == 64
    assert cfg.model.uv_intersection.face_layers == cfg.model.uv_intersection.curve_layers == 8
    assert cfg.model.uv_intersection.curve_width == 512
    assert cfg.model.uv_intersection.curve_heads == 8
    assert cfg.trainer.max_steps == 30000 and cfg.trainer.seed == 42
    assert cfg.trainer.learning_rate == 2e-5 and cfg.trainer.precision == "bf16-mixed"
    assert cfg.trainer.gradient_clip_val == 1 and cfg.trainer.max_time is None
    assert cfg.trainer.float32_matmul_precision == "highest"
    assert cfg.model.loss_weights.classification_weight == cfg.model.loss_weights.incidence_weight == .1
    assert not cfg.trainer.vae_validation_inference


def test_full_resume_clears_source_files_only_in_runtime_config(tmp_path, monkeypatch):
    cfg = config(tmp_path, [
        "model.init_uv_decoder_checkpoint=missing-source.ckpt", "model.checkpoint=missing-full.ckpt",
        "trainer.resume_from_checkpoint=resume.ckpt",
    ])
    seen = stub_construction(monkeypatch)
    training.UnifiedTrainer(cfg)
    assert seen[0]["init_uv_decoder_checkpoint"] is None and seen[0]["checkpoint"] is None
    assert seen[0]["restoring_checkpoint"] is True
    assert cfg.model.init_uv_decoder_checkpoint == "missing-source.ckpt"
    assert cfg.model.checkpoint == "missing-full.ckpt"


def test_frozen_fresh_run_cannot_freeze_random_encoder(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    seen = stub_construction(monkeypatch)
    with pytest.raises(ValueError, match="refusing to freeze a random encoder"):
        training.UnifiedTrainer(cfg)
    assert not seen


@pytest.mark.parametrize("resume", ["null", "resume.ckpt"])
def test_legacy_permissive_initialization_is_rejected(tmp_path, monkeypatch, resume):
    cfg = config(tmp_path, ["trainer.init_from_checkpoint=legacy.ckpt", f"trainer.resume_from_checkpoint={resume}"])
    seen = stub_construction(monkeypatch)
    with pytest.raises(ValueError, match="rejects trainer.init_from_checkpoint"):
        training.UnifiedTrainer(cfg)
    assert not seen


@pytest.mark.parametrize("is_training", [False, True])
def test_validation_uses_mean_without_export_despite_legacy_visualization_flag(tmp_path, is_training):
    cfg = config(tmp_path, ["trainer.vae_validation_inference=true"])
    calls = []

    def model(batch, **kwargs):
        calls.append(kwargs)
        return {"total_loss": torch.tensor(0.)}, None

    fake = SimpleNamespace(stage="vae", cfg=cfg, model=model)
    training.UnifiedTrainer._forward_step(fake, {}, is_training=is_training)
    assert calls == [{"v_test": False, "v_use_mean": not is_training}]


@pytest.mark.parametrize("accumulation", [None, 3])
def test_main_passes_optimizer_accumulation_and_preserves_legacy_default(tmp_path, monkeypatch, accumulation):
    cfg = config(tmp_path)
    with open_dict(cfg.trainer):
        if accumulation is None:
            del cfg.trainer.accumulate_grad_batches
            del cfg.trainer.float32_matmul_precision
        else:
            cfg.trainer.accumulate_grad_batches = accumulation
    monkeypatch.setattr(training.hydra.core.hydra_config.HydraConfig, "get", lambda: {
        "runtime": {"output_dir": str(tmp_path)},
    })
    monkeypatch.setattr(training, "prepare_run", lambda *_: {"wandb_id": "test"})
    monkeypatch.setattr(training, "TensorBoardLogger", lambda **_: object())
    seen = {}
    monkeypatch.setattr(training.torch, "set_float32_matmul_precision",
                        lambda precision: seen.update(matmul_precision=precision))

    class StopAtTrainer(Exception):
        pass

    def capture_trainer(**kwargs):
        seen.update(kwargs)
        raise StopAtTrainer

    monkeypatch.setattr(training, "Trainer", capture_trainer)
    with pytest.raises(StopAtTrainer):
        training.main.__wrapped__(cfg)
    assert seen["accumulate_grad_batches"] == (accumulation or 1)
    assert seen["max_steps"] == 30000
    assert seen["matmul_precision"] == ("medium" if accumulation is None else "highest")


@pytest.mark.parametrize("changed_field", ["nhead", "curve_heads"])
def test_resume_hook_rejects_same_shape_attention_semantics(tmp_path, monkeypatch, changed_field):
    from src.brepnet.models.vae.uv_intersection_training import AutoEncoder_uv_intersection

    cfg = config(tmp_path, ["trainer.resume_from_checkpoint=resume.ckpt"])

    def build(model_cfg, condition):
        # No donor tensors are needed to exercise the real metadata validator.
        model = torch.nn.Linear(1, 1)
        model._configuration = OmegaConf.to_container(model_cfg, resolve=True)
        model.validate_complete_checkpoint = MethodType(
            AutoEncoder_uv_intersection.validate_complete_checkpoint, model,
        )
        return model

    monkeypatch.setattr(training, "build_model", build)
    monkeypatch.setattr(training, "_load_dataset_class", lambda _: object)
    module = training.UnifiedTrainer(cfg)
    checkpoint = {"hyper_parameters": {"model": OmegaConf.to_container(cfg.model, resolve=True)}}
    module.on_load_checkpoint(checkpoint)  # Same architecture restores normally.
    recorded = checkpoint["hyper_parameters"]["model"]
    if changed_field == "nhead":
        recorded["nhead"] = 8
    else:
        recorded["uv_intersection"]["curve_heads"] = 4
    with pytest.raises(ValueError, match="configuration differs"):
        module.on_load_checkpoint(checkpoint)
