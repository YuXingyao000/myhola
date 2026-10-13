"""Only the old inter may be replaced when starting the local experiment."""

from pathlib import Path

import pytest
import torch

from src.brepnet.models.vae import AutoEncoder_intersection
from src.brepnet.test.test_intersection_vae import encoder_batch, tiny_config


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def local_config(*, block_type=None, **changes):
    return tiny_config(
        intersection={
            "kind": "local_attention", "hidden_dim": 16, "num_heads": 2,
            **({"block_type": block_type} if block_type is not None else {}),
            "num_layers": 2, "ffn_dim": 32, "dropout": 0.,
            "candidate_chunk_size": 3, "checkpoint_candidates": True,
        },
        **changes,
    )


def save_refinement(path):
    old = AutoEncoder_intersection(tiny_config()).eval()
    # Preserve a learned, nonzero refiner rather than merely its initial zeros.
    with torch.no_grad():
        old.decoder.refiner.edge_delta.weight.normal_(std=.01)
        old.decoder.refiner.face_delta.weight.normal_(std=.01)
    torch.save({
        "epoch": 10, "global_step": 3993,
        "state_dict": {"model." + key: value for key, value in old.state_dict().items()},
    }, path)
    return old


@pytest.mark.parametrize("block_type", [None, "transformer_decoder"])
def test_migration_preserves_every_non_inter_weight_and_frozen_posterior(tmp_path, block_type):
    source_path = tmp_path / "refinement.ckpt"
    old = save_refinement(source_path)
    model = AutoEncoder_intersection(local_config(
        init_refinement_checkpoint=str(source_path), freeze_encoder=True, block_type=block_type,
    )).eval()
    for key, expected in old.state_dict().items():
        if not key.startswith("decoder.inter."):
            torch.testing.assert_close(model.state_dict()[key], expected, rtol=0, atol=0)
    assert model.initialization_report["source_global_step"] == 3993
    batch = encoder_batch()
    with torch.no_grad():
        old_z, _, old_stats = old.sample(old.encode(batch)["face_features"], True)
        new_z, _, new_stats = model.sample(model.encode(batch)["face_features"], True)
    torch.testing.assert_close(new_z, old_z, rtol=0, atol=0)
    torch.testing.assert_close(new_stats, old_stats, rtol=0, atol=0)

    model.train()
    loss, _ = model(batch)
    assert torch.isfinite(loss["total_loss"])
    loss["total_loss"].backward()
    assert all(p.grad is None for p in model.encoder.parameters())
    assert all(p.grad is None for p in model.posterior.parameters())
    for block in (model.decoder.inter, model.decoder.face_geometry,
                  model.decoder.edge_geometry, model.decoder.classifier, model.decoder.refiner):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in block.parameters())


@pytest.mark.parametrize("corruption", ["missing_retained", "extra_retained", "retained_shape", "missing_inter"])
@pytest.mark.parametrize("block_type", [None, "transformer_decoder"])
def test_migration_rejects_any_incomplete_or_incompatible_source(tmp_path, corruption, block_type):
    path = tmp_path / "source.ckpt"
    save_refinement(path)
    checkpoint = torch.load(path, weights_only=False)
    state = checkpoint["state_dict"]
    if corruption == "missing_retained":
        state.pop("model.decoder.classifier.bias")
    elif corruption == "extra_retained":
        state["model.unexpected.weight"] = torch.ones(1)
    elif corruption == "retained_shape":
        state["model.decoder.classifier.bias"] = torch.ones(2)
    else:
        state.pop(next(key for key in state if key.startswith("model.decoder.inter.")))
    torch.save(checkpoint, path)
    with pytest.raises((ValueError, RuntimeError)):
        AutoEncoder_intersection(local_config(block_type=block_type, init_refinement_checkpoint=str(path)))


@pytest.mark.parametrize("block_type", [None, "transformer_decoder"])
def test_local_full_checkpoint_loads_strictly_and_is_not_a_migration_source(tmp_path, block_type):
    model = AutoEncoder_intersection(local_config(block_type=block_type)).eval()
    path = tmp_path / "local.ckpt"
    torch.save(model.state_dict(), path)
    restored = AutoEncoder_intersection(local_config(block_type=block_type, checkpoint=str(path))).eval()
    z = torch.randn(3, model.face_feature_dim)
    with torch.no_grad():
        first = model.decode_state(z)
        second = restored.decode_state(z)
    torch.testing.assert_close(first.refined.edge_features, second.refined.edge_features, rtol=0, atol=0)
    with pytest.raises(ValueError, match="complete legacy"):
        AutoEncoder_intersection(local_config(block_type=block_type, init_refinement_checkpoint=str(path)))
    state = torch.load(path, weights_only=False)
    state.pop("decoder.classifier.bias")
    torch.save(state, path)
    with pytest.raises(RuntimeError, match="Missing key"):
        AutoEncoder_intersection(local_config(block_type=block_type, checkpoint=str(path)))


def test_initialization_sources_are_exclusive_and_migration_is_local_only():
    with pytest.raises(ValueError, match="only one"):
        AutoEncoder_intersection(local_config(checkpoint="full.ckpt", init_refinement_checkpoint="old.ckpt"))
    with pytest.raises(ValueError, match="Resume"):
        AutoEncoder_intersection(local_config(restoring_checkpoint=True, init_refinement_checkpoint="old.ckpt"))
    with pytest.raises(ValueError, match="local_attention"):
        AutoEncoder_intersection(tiny_config(init_refinement_checkpoint="old.ckpt"))


def test_trainer_resume_clears_all_initialization_paths(tmp_path, monkeypatch):
    from hydra import compose, initialize_config_dir
    from src.brepnet import train

    config_dir = str(Path(__file__).resolve().parents[3] / "configs")
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        cfg = compose(config_name="train_vae_intersection_local", overrides=[
            "dataset.raw_data_root=unused", "dataset.train_dataset=unused",
            "dataset.val_dataset=unused", "dataset.test_dataset=unused",
            f"trainer.output_dir={tmp_path}", "trainer.resume_from_checkpoint=local.ckpt",
            "model.init_refinement_checkpoint=missing-old.ckpt",
        ])
    captured = {}

    def build_model(model_cfg, condition):
        captured.update(dict(model_cfg))
        return torch.nn.Linear(2, 2)

    monkeypatch.setattr(train, "build_model", build_model)
    monkeypatch.setattr(train, "_load_dataset_class", lambda cfg: object)
    train.UnifiedTrainer(cfg)
    assert captured["restoring_checkpoint"] is True
    assert captured["init_legacy_checkpoint"] is None
    assert captured["init_refinement_checkpoint"] is None
    assert captured["checkpoint"] is None


def test_full_capacity_config_consumes_legacy_dimensions_and_preserves_training_contract():
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from src.brepnet.models.blocks import AttnIntersection
    from src.brepnet.models.blocks import LocalPairIntersection

    config_dir = str(Path(__file__).resolve().parents[3] / "configs")
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        full = compose(config_name="train_vae_intersection_local")
        historical = compose(config_name="train_vae_intersection_local_small_legacy")
        changed = compose(config_name="train_vae_intersection_local", overrides=[
            "model.intersection_dim=256", "model.intersection_layers=3",
        ])
    config = OmegaConf.to_container(full.model.intersection, resolve=True)
    assert config["block_type"] == "transformer_decoder"
    assert (config["hidden_dim"], config["num_layers"], config["num_heads"], config["ffn_dim"]) == (512, 8, 8, 2048)
    assert config["dropout"] == .1 and config["activation"] == "relu"
    assert changed.model.intersection.hidden_dim == 256 and changed.model.intersection.num_layers == 3
    assert full.trainer.exp_name == "intersection_local/full_capacity"
    assert historical.trainer.exp_name == "intersection_local/frozen"
    old_config = OmegaConf.to_container(historical.model.intersection, resolve=True)
    assert old_config["block_type"] == "cross_attention"
    assert (old_config["hidden_dim"], old_config["num_layers"], old_config["num_heads"], old_config["ffn_dim"]) == (128, 2, 4, 256)
    assert old_config["dropout"] == 0 and old_config["activation"] == "gelu"
    for key in ("batch_size", "devices", "learning_rate", "precision", "max_steps", "max_epochs"):
        assert full.trainer[key] == historical.trainer[key]
    assert full.model.loss_weights == historical.model.loss_weights
    assert full.model.freeze_encoder and full.model.trainable_scope == "all"
    with torch.device("meta"):
        local = LocalPairIntersection(32, 64, 96, config)
        legacy = AttnIntersection(32, full.model.intersection_dim, full.model.intersection_layers)
        small = LocalPairIntersection(32, 64, 96, old_config)
    assert len(local.blocks) == 8
    for block, old_block in zip(local.blocks, legacy.layers):
        assert type(block) is type(old_block) is torch.nn.TransformerDecoderLayer
        assert {k: v.shape for k, v in block.state_dict().items()} == {k: v.shape for k, v in old_block.state_dict().items()}
    count = lambda module: sum(parameter.numel() for parameter in module.parameters())
    assert count(local.blocks) == count(legacy.layers)
    assert count(local) > count(legacy) > count(small)
