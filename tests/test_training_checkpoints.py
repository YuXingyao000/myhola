"""Fixed-step snapshots retain the final state without a validation loader."""

from pathlib import Path
import tempfile
import unittest

from omegaconf import OmegaConf
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.train import _build_checkpoint_callbacks


class TinyTrainingModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(1, 1)

    def training_step(self, batch, batch_idx):
        return self.layer(batch[0]).square().mean()

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)


class TrainingCheckpointTests(unittest.TestCase):
    @staticmethod
    def config(interval):
        return OmegaConf.create({"trainer": {
            "checkpoint_enable_version_counter": True,
            "checkpoint_every_n_train_steps": interval,
        }})

    def test_null_keeps_only_existing_validation_callback(self):
        callbacks = _build_checkpoint_callbacks(self.config(None), Path("unused"))
        self.assertEqual(len(callbacks), 1)
        self.assertEqual(callbacks[0].monitor, "val/loss")
        self.assertEqual(callbacks[0].save_top_k, 3)
        self.assertTrue(callbacks[0].save_last)

    def test_interval_requires_positive_integer(self):
        for interval in (0, -1, True, 1.5):
            with self.subTest(interval=interval), self.assertRaisesRegex(ValueError, "positive integer"):
                _build_checkpoint_callbacks(self.config(interval), Path("unused"))

    def test_final_last_saved_between_step_intervals_without_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            callbacks = _build_checkpoint_callbacks(self.config(2), root)
            self.assertNotEqual(callbacks[0].state_key, callbacks[1].state_key)
            trainer = pl.Trainer(
                accelerator="cpu", devices=1, max_steps=3, logger=False,
                callbacks=[callbacks[1]], enable_model_summary=False,
                enable_progress_bar=False, default_root_dir=directory,
            )
            loader = DataLoader(TensorDataset(torch.ones(8, 1)), batch_size=1)
            trainer.fit(TinyTrainingModel(), train_dataloaders=loader)
            weights = root / "weights" / "steps"
            snapshot = torch.load(weights / "step=00000002.ckpt", map_location="cpu", weights_only=False)
            final = torch.load(weights / "last.ckpt", map_location="cpu", weights_only=False)
            self.assertEqual(snapshot["global_step"], 2)
            self.assertEqual(final["global_step"], 3)
            self.assertIn("optimizer_states", final)
            self.assertEqual(
                {path.name for path in weights.iterdir()},
                {"step=00000002.ckpt", "last.ckpt"},
            )

    def test_bounded_step_retention_keeps_recent_and_last(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = self.config(2)
            cfg.trainer.checkpoint_step_keep_last = 2
            callback = _build_checkpoint_callbacks(cfg, Path(directory))[1]
            trainer = pl.Trainer(accelerator="cpu", devices=1, max_steps=7, logger=False,
                                 callbacks=[callback], enable_model_summary=False,
                                 enable_progress_bar=False)
            trainer.fit(TinyTrainingModel(), train_dataloaders=DataLoader(
                TensorDataset(torch.ones(10, 1)), batch_size=1))
            folder = Path(directory) / "weights/steps"
            self.assertEqual({p.name for p in folder.iterdir()},
                             {"step=00000004.ckpt", "step=00000006.ckpt", "last.ckpt"})
            self.assertEqual(torch.load(folder / "last.ckpt", weights_only=False)["global_step"], 7)


if __name__ == "__main__":
    unittest.main()
