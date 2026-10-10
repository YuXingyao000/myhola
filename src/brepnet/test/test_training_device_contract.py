"""Distributed training input seeding and single-device export guards."""

from functools import partial
from types import SimpleNamespace

import pytest
from lightning_fabric.utilities.data import _auto_add_worker_init_fn
from lightning_fabric.utilities.seed import pl_worker_init_function
from omegaconf import OmegaConf
from torch.utils.data import Dataset, default_collate

from src.brepnet.train import UnifiedTrainer, _validate_evaluation_devices


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
        dataset_cls=_TinyDataset, cfg=OmegaConf.create({"dataset": {}}),
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
