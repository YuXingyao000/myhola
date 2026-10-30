"""Distributed training input seeding and single-device export guards."""

from functools import partial
from datetime import timedelta
from types import SimpleNamespace

import pytest
import pytorch_lightning as pl
import torch
from lightning_fabric.utilities.data import _auto_add_worker_init_fn
from lightning_fabric.utilities.seed import pl_worker_init_function
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint, Timer
from torch.utils.data import DataLoader, Dataset, TensorDataset, default_collate

from src.train import UnifiedTrainer, _validate_evaluation_devices


@pytest.mark.parametrize("trainer_evaluate,eval_enabled", [(True, False), (False, True), (True, True)])
def test_multidevice_export_is_rejected_for_both_evaluation_switches(trainer_evaluate, eval_enabled):
    cfg = OmegaConf.create({
        "trainer": {"devices": 8, "evaluate": trainer_evaluate},
        "eval": {"enabled": eval_enabled},
    })
    with pytest.raises(ValueError, match="trainer.devices=1"):
        _validate_evaluation_devices(cfg)
    cfg.trainer.devices = 1
    _validate_evaluation_devices(cfg)


def test_multidevice_training_remains_allowed():
    _validate_evaluation_devices(OmegaConf.create({
        "trainer": {"devices": 8, "evaluate": False},
        "eval": {"enabled": False},
    }))


class _TinyDataset(Dataset):
    collate_fn = staticmethod(default_collate)

    def __init__(self, mode, cfg):
        pass

    def __len__(self):
        return 8

    def __getitem__(self, index):
        return index


@pytest.mark.parametrize("method", ["train_dataloader", "val_dataloader"])
def test_loaders_allow_lightning_to_install_rank_aware_worker_seeding(method, monkeypatch):
    module = SimpleNamespace(
        dataset_cls=_TinyDataset,
        cfg=OmegaConf.create({
            "dataset": {},
            "trainer": {"prefetch_factor": 4, "persistent_workers": True},
        }),
        batch_size=4, num_workers=2,
    )
    loader = getattr(UnifiedTrainer, method)(module)
    assert loader.worker_init_fn is None
    # seed_everything(workers=True) sets this flag; Lightning injects each rank.
    monkeypatch.setenv("PL_SEED_WORKERS", "1")
    _auto_add_worker_init_fn(loader, rank=3)
    assert isinstance(loader.worker_init_fn, partial)
    assert loader.worker_init_fn.func is pl_worker_init_function
    assert loader.worker_init_fn.keywords == {"rank": 3}


def test_intersection_resume_does_not_reopen_initialization_checkpoints(tmp_path, monkeypatch):
    from src import train as train_module

    sources = {
        "init_legacy_checkpoint": "unavailable-legacy.ckpt",
        "init_refinement_checkpoint": "unavailable-refinement.ckpt",
        "init_curve_feedback_checkpoint": "unavailable-conditioned.ckpt",
        "checkpoint": "unavailable-full.ckpt",
    }
    cfg = OmegaConf.create({
        "model": {
            "name": "AutoEncoder_intersection", "stage": "vae", "trainable_scope": "all",
            "freeze_encoder": True, "curve_point_feedback": {"enabled": True}, **sources,
        },
        "trainer": {
            "batch_size": 2, "num_workers": 0, "learning_rate": 1e-4,
            "output_dir": str(tmp_path), "resume_from_checkpoint": "resumed.ckpt",
            "init_from_checkpoint": None,
        },
        "dataset": {}, "condition": {},
    })
    seen = []

    def capture_model(config, _condition):
        seen.append(OmegaConf.to_container(config, resolve=True))
        return torch.nn.Linear(1, 1)

    monkeypatch.setattr(train_module, "build_model", capture_model)
    monkeypatch.setattr(train_module, "_load_dataset_class", lambda _: _TinyDataset)
    UnifiedTrainer(cfg)
    assert len(seen) == 1
    assert all(seen[0][name] is None for name in sources)
    assert seen[0]["restoring_checkpoint"] is True
    assert seen[0]["curve_point_feedback"]["enabled"] is True
    # Keep the recorded configuration's source provenance; only construction clears it.
    assert all(cfg.model[name] == value for name, value in sources.items())


def test_native_training_time_limit_validates_and_saves_resumable_last_checkpoint(tmp_path):
    """A mid-epoch time stop must preserve the last completed optimizer step."""
    class TimerProbe(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

        def training_step(self, batch, batch_idx):
            return self.weight.square()

        def validation_step(self, batch, batch_idx):
            self.log("val/loss", self.weight.square(), on_epoch=True)

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=1e-3)

    checkpoint = ModelCheckpoint(dirpath=tmp_path, monitor="val/loss", save_top_k=1, save_last=True)
    trainer = pl.Trainer(
        accelerator="cpu", devices=1, logger=False, callbacks=[checkpoint],
        max_epochs=2, max_time=timedelta(milliseconds=1), num_sanity_val_steps=0,
        check_val_every_n_epoch=1, enable_progress_bar=False, enable_model_summary=False,
    )
    loader = DataLoader(TensorDataset(torch.ones(1000, 1)), batch_size=1)
    trainer.fit(TimerProbe(), train_dataloaders=loader, val_dataloaders=DataLoader(TensorDataset(torch.ones(1, 1))))
    assert 0 < trainer.global_step < len(loader)
    assert "val/loss" in trainer.callback_metrics
    assert checkpoint.last_model_path
    saved = torch.load(checkpoint.last_model_path, weights_only=False)
    assert saved["global_step"] == trainer.global_step
    assert saved["optimizer_states"][0]["state"]
    timer = next(callback for callback in trainer.callbacks if isinstance(callback, Timer))
    assert saved["callbacks"][timer.state_key]["time_elapsed"]["train"] > 0
