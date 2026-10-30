"""Training throughput options preserve data and distributed metric contracts."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import Dataset, default_collate

from src import train as training


class _TinyDataset(Dataset):
    collate_fn = staticmethod(default_collate)

    def __init__(self, mode, cfg):
        pass

    def __len__(self):
        return 8

    def __getitem__(self, index):
        return {"v_prefix": str(index), "value": torch.tensor(float(index + 1))}


class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))

    def forward(self, batch, v_test=False):
        loss = (self.weight * batch["value"]).mean()
        return {"total_loss": loss, "diffusion_loss": loss, "t": torch.ones(2)}


@pytest.fixture(autouse=True)
def _small_cpu_thread_budget():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def tiny_module(tmp_path, monkeypatch):
    cfg = OmegaConf.create({
        "model": {"name": "Diffusion", "stage": "diffusion"},
        "trainer": {
            "batch_size": 2, "num_workers": 0, "learning_rate": 0.0,
            "output_dir": str(tmp_path), "resume_from_checkpoint": None,
            "init_from_checkpoint": None, "sync_train_step_loss": False,
            "log_every_n_steps": 2, "diffusion_validation_inference": False,
            "devices": 2, "ddp_find_unused_parameters": False,
            "ddp_gradient_as_bucket_view": True,
        },
        "dataset": {}, "condition": {},
    })
    monkeypatch.setattr(training, "build_model", lambda *_: _TinyModel())
    monkeypatch.setattr(training, "_load_dataset_class", lambda _: _TinyDataset)
    return training.UnifiedTrainer(cfg)


@pytest.mark.parametrize("rank,step,local_logged", [(0, 0, False), (0, 1, True), (1, 1, False)])
def test_local_step_logging_does_not_reduce_each_batch(tiny_module, monkeypatch, rank, step, local_logged):
    tiny_module._trainer = SimpleNamespace(global_rank=rank, global_step=step)
    calls = {}
    monkeypatch.setattr(tiny_module, "log", lambda name, value, **kwargs: calls.update({name: kwargs}))
    batch = default_collate([_TinyDataset(None, None)[0], _TinyDataset(None, None)[1]])
    loss = tiny_module.training_step(batch, 0)
    assert loss.item() == 1.5
    assert ("train/loss_step_local" in calls) is local_logged
    for name in ("train/diffusion_loss", "train/loss_epoch"):
        assert calls[name]["on_epoch"] and calls[name]["sync_dist"]
        assert not calls[name]["on_step"]
    if local_logged:
        assert not calls["train/loss_step_local"]["sync_dist"]
        assert calls["train/loss_step_local"]["rank_zero_only"]
    assert "train/t" not in calls


def test_legacy_step_metric_remains_available(tiny_module, monkeypatch):
    tiny_module.cfg.trainer.sync_train_step_loss = True
    calls = {}
    monkeypatch.setattr(tiny_module, "log", lambda name, value, **kwargs: calls.update({name: kwargs}))
    tiny_module.training_step(default_collate([_TinyDataset(None, None)[0]]), 0)
    assert calls["train/loss"]["on_step"] and calls["train/loss"]["on_epoch"]
    assert calls["train/loss"]["sync_dist"]
    assert "train/loss_epoch" not in calls and "train/loss_step_local" not in calls


@pytest.mark.parametrize("enabled,sanity,rank,batch_idx,expected", [
    (False, False, 0, 0, False), (True, True, 0, 0, False),
    (True, False, 1, 0, False), (True, False, 0, 1, False),
    (True, False, 0, 0, True),
])
def test_diffusion_sampling_is_opt_in_and_skips_sanity(tiny_module, monkeypatch, enabled, sanity, rank, batch_idx, expected):
    tiny_module.cfg.trainer.diffusion_validation_inference = enabled
    tiny_module._trainer = SimpleNamespace(global_rank=rank, sanity_checking=sanity)
    calls = []
    monkeypatch.setattr(tiny_module, "log", lambda *args, **kwargs: None)
    monkeypatch.setattr(tiny_module, "_diffusion_viz", lambda batch: calls.append(batch))
    batch = default_collate([_TinyDataset(None, None)[0]])
    assert tiny_module.validation_step(batch, batch_idx).item() == 1.0
    assert bool(calls) is expected


@pytest.mark.parametrize("method", ["train_dataloader", "val_dataloader"])
@pytest.mark.parametrize("workers", [0, 2])
def test_loader_prefetch_and_persistence_are_explicit(tiny_module, method, workers):
    tiny_module.num_workers = workers
    tiny_module.cfg.trainer.prefetch_factor = 2
    tiny_module.cfg.trainer.persistent_workers = True
    loader = getattr(tiny_module, method)()
    assert loader.num_workers == workers
    assert loader.prefetch_factor == (2 if workers else None)
    assert loader.persistent_workers is bool(workers)
    assert loader.worker_init_fn is None  # Lightning installs rank-aware worker seeds.
    tiny_module.cfg.trainer.persistent_workers = False
    assert not getattr(tiny_module, method)().persistent_workers


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "4"])
def test_invalid_thread_budgets_fail(value):
    with pytest.raises(ValueError, match="positive integer or null"):
        training._configure_num_threads(OmegaConf.create({"trainer": {"num_threads": value}}))


def test_explicit_threads_and_null_preserve_environment(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    training._configure_num_threads(OmegaConf.create({"trainer": {"num_threads": 2}}))
    assert torch.get_num_threads() == 2 and os.environ["OMP_NUM_THREADS"] == "2"
    training._configure_num_threads(OmegaConf.create({"trainer": {"num_threads": None}}))
    assert torch.get_num_threads() == 2 and os.environ["OMP_NUM_THREADS"] == "2"


def _attach_clipping_context(module, optimizer, scaler=None):
    module._trainer = SimpleNamespace(
        global_step=1089, global_rank=0,
        precision_plugin=SimpleNamespace(
            scaler=scaler,
            main_params=lambda current: (
                parameter for group in current.param_groups for parameter in group["params"]
            ),
        ),
    )
    module._train_batch_prefixes = ("sample_a", "sample_b")


def test_finite_clipping_retains_existing_rule(tiny_module):
    optimizer = tiny_module.configure_optimizers()
    _attach_clipping_context(tiny_module, optimizer)
    tiny_module.model.weight.grad = torch.tensor(7.0)
    reference = torch.nn.Parameter(torch.ones(()))
    reference.grad = torch.tensor(7.0)
    torch.nn.utils.clip_grad_norm_([reference], 1.0)
    tiny_module.configure_gradient_clipping(optimizer, 1.0, "norm")
    torch.testing.assert_close(tiny_module.model.weight.grad, reference.grad, rtol=0, atol=0)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_bad_gradient_is_rejected_before_clipping_other_gradients(tiny_module, bad):
    tiny_module.model.other = torch.nn.Parameter(torch.ones(()))
    optimizer = tiny_module.configure_optimizers()
    _attach_clipping_context(tiny_module, optimizer)
    tiny_module.model.weight.grad = torch.tensor(bad)
    tiny_module.model.other.grad = torch.tensor(2.0)
    with pytest.raises(RuntimeError, match="step=1089.*sample_a.*model.weight"):
        tiny_module.configure_gradient_clipping(optimizer, 1.0, "norm")
    assert tiny_module.model.other.grad.item() == 2.0
    assert all(torch.isfinite(parameter).all() for parameter in tiny_module.parameters())
    assert not optimizer.state


def test_grad_scaler_retains_lightning_overflow_handling(tiny_module, monkeypatch):
    optimizer = tiny_module.configure_optimizers()
    _attach_clipping_context(tiny_module, optimizer, scaler=object())
    calls = []
    monkeypatch.setattr(
        pl.LightningModule, "configure_gradient_clipping",
        lambda module, opt, value, algorithm: calls.append((opt, value, algorithm)),
    )
    tiny_module.configure_gradient_clipping(optimizer, 1.0, "norm")
    assert calls == [(optimizer, 1.0, "norm")]


@pytest.mark.parametrize("clip_value", [None, 0.0, -1.0])
def test_disabled_clipping_retains_lightning_behavior(tiny_module, monkeypatch, clip_value):
    optimizer = tiny_module.configure_optimizers()
    _attach_clipping_context(tiny_module, optimizer)
    calls = []
    monkeypatch.setattr(
        pl.LightningModule, "configure_gradient_clipping",
        lambda module, opt, value, algorithm: calls.append((opt, value, algorithm)),
    )
    tiny_module.configure_gradient_clipping(optimizer, clip_value, "norm")
    assert calls == [(optimizer, clip_value, "norm")]


def test_lightning_stops_before_adam_can_apply_nonfinite_gradient(tiny_module):
    original = tiny_module.model.weight.detach().clone()
    handle = tiny_module.model.weight.register_hook(lambda grad: grad * float("nan"))
    trainer = pl.Trainer(
        accelerator="cpu", devices=1, max_steps=1, logger=False,
        enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False,
        limit_val_batches=0, num_sanity_val_steps=0, gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
    )
    try:
        with pytest.raises(RuntimeError, match="Nonfinite gradient norm before optimizer update"):
            trainer.fit(tiny_module)
    finally:
        handle.remove()
    torch.testing.assert_close(tiny_module.model.weight, original, rtol=0, atol=0)
    assert not trainer.optimizers[0].state


def test_strategy_defaults_remain_compatible_and_svr_can_disable_unused_detection():
    cfg = OmegaConf.create({"trainer": {"devices": 1}})
    assert training._build_strategy(cfg) == "auto"
    cfg.trainer.devices = 8
    assert training._build_strategy(cfg)._ddp_kwargs == {
        "find_unused_parameters": True, "gradient_as_bucket_view": False,
    }
    cfg.trainer.ddp_find_unused_parameters = False
    cfg.trainer.ddp_gradient_as_bucket_view = True
    assert training._build_strategy(cfg)._ddp_kwargs == {
        "find_unused_parameters": False, "gradient_as_bucket_view": True,
    }


class _MetricCapture(pl.Callback):
    """Each rank reads its own reduced metrics; no rank skips a collective."""

    def __init__(self, destination):
        self.destination = Path(destination)
        self.local_batch_means = []

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        self.local_batch_means.append(batch["value"].mean().item())

    def on_train_epoch_end(self, trainer, module):
        metrics = {key: value.item() for key, value in trainer.callback_metrics.items()}
        result = {"metrics": metrics, "local_batch_means": self.local_batch_means}
        (self.destination / f"rank{trainer.global_rank}.json").write_text(json.dumps(result))


def test_two_rank_epoch_loss_is_global_while_step_loss_is_local(tiny_module, tmp_path):
    """Real CPU DDP covers Lightning reduction, not just log keyword arguments."""
    strategy = training._build_strategy(tiny_module.cfg)
    cpu_strategy = DDPStrategy(
        start_method="spawn", process_group_backend="gloo", **strategy._ddp_kwargs,
    )
    trainer = pl.Trainer(
        accelerator="cpu", devices=2, strategy=cpu_strategy, max_epochs=1,
        logger=False, enable_checkpointing=False, enable_progress_bar=False,
        enable_model_summary=False, limit_val_batches=0, num_sanity_val_steps=0,
        callbacks=[_MetricCapture(tmp_path)], default_root_dir=str(tmp_path),
    )
    trainer.fit(tiny_module)
    rank_results = [json.loads((tmp_path / f"rank{rank}.json").read_text()) for rank in range(2)]
    for result in rank_results:
        assert result["metrics"]["train/loss_epoch"] == pytest.approx(4.5)
        assert result["metrics"]["train/diffusion_loss"] == pytest.approx(4.5)
    assert rank_results[0]["metrics"]["train/loss_step_local"] == rank_results[0]["local_batch_means"][-1]
    assert "train/loss_step_local" not in rank_results[1]["metrics"]
