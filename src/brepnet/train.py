"""
Unified Training Entry Point for HoLa-BRep.

Single Lightning module for VAE and diffusion training, dispatching by
``cfg.model.stage``.

Usage Examples
--------------
.. code-block:: bash

    # VAE training (Stage 1)
    python -m src.brepnet.train --config-name train_vae \
        dataset.data_root=/path/to/data

    # Diffusion with single-image condition (Stage 2)
    python -m src.brepnet.train --config-name train_diffusion_white \
        trainer.devices=8

"""

from __future__ import annotations

import importlib
import logging
import warnings
from functools import partial
from pathlib import Path
from typing import Any, Dict, Optional

import hydra
import numpy as np
import torch
from lightning_fabric import seed_everything
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
)
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from src.brepnet.models import build_model

logger = logging.getLogger(__name__)

warnings.filterwarnings(
    "ignore",
    message=r"In '.*': Defaults list is missing `_self_`.*",
    category=UserWarning,
    module=r"hydra\._internal\.defaults_list",
)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _seed_worker(worker_id: int, rank_id: int = 0):
    """Ensure reproducible randomness per dataloader worker."""
    import random

    seed = worker_id + rank_id * 10000
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _load_dataset_class(cfg: DictConfig):
    """
    Dynamically load a dataset class based on config.

    Dataset classes are resolved by the stable ``dataset.name`` config field.
    """
    name = cfg.dataset.name
    dataset_mod = importlib.import_module("src.brepnet.dataset")
    if not hasattr(dataset_mod, name):
        raise AttributeError(
            f"Dataset class '{name}' not found in src.brepnet.dataset. "
            f"Available: {[a for a in dir(dataset_mod) if not a.startswith('_')]}"
        )
    return getattr(dataset_mod, name)


# ---------------------------------------------------------------------------
# Unified Lightning Module
# ---------------------------------------------------------------------------

class UnifiedTrainer(pl.LightningModule):
    """Single Lightning module for all training modes.

    Mode determined by ``cfg.model.stage``:
      - ``"vae"``: trains AutoEncoder (stage 1)
      - ``"diffusion"``: trains diffusion model (stage 2)
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(OmegaConf.to_container(cfg, resolve=True))

        # --- Core configuration ---
        self.stage: str = cfg.model.stage
        self.batch_size: int = cfg.trainer.batch_size
        self.num_workers: int = cfg.trainer.num_workers
        self.learning_rate: float = cfg.trainer.learning_rate

        # --- Output directory ---
        self.log_root = Path(cfg.trainer.output_dir)
        self.log_root.mkdir(parents=True, exist_ok=True)

        # --- Build model ---
        logger.info("Building model for stage='%s'", self.stage)
        self.model = build_model(cfg.model, cfg.condition)
        self._init_model_from_checkpoint()
        self._apply_trainable_scope()

        # --- Dataset class (lazy loaded) ---
        self.dataset_cls = _load_dataset_class(cfg)
        logger.info("Dataset class: %s", self.dataset_cls.__name__)

        # --- Visualization state ---
        self.viz: Dict[str, Any] = {}

    def _init_model_from_checkpoint(self):
        """Load model weights without restoring optimizer/trainer state."""
        ckpt_path = (
            self.cfg.trainer.init_from_checkpoint
        )
        if not ckpt_path or str(ckpt_path).lower() in {"none", "null"}:
            return

        checkpoint = torch.load(str(ckpt_path), map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        model_state = {}
        for key, value in state_dict.items():
            if key.startswith("model."):
                model_state[key[len("model."):]] = value
            else:
                model_state[key] = value

        missing, unexpected = self.model.load_state_dict(model_state, strict=False)
        logger.info(
            "Initialized model from %s (missing=%d, unexpected=%d)",
            ckpt_path,
            len(missing),
            len(unexpected),
        )
        if missing:
            logger.warning("Missing model keys while loading %s: %s", ckpt_path, missing[:20])
        if unexpected:
            logger.warning("Unexpected model keys while loading %s: %s", ckpt_path, unexpected[:20])

    def _apply_trainable_scope(self):
        """Optionally freeze the model to a named trainable scope."""
        if self.stage != "vae":
            return
        scope = self.cfg.model.trainable_scope
        if scope == "all":
            return

        if scope == "intersection":
            trainable_prefixes = self.cfg.model.trainable_module_prefixes
        else:
            trainable_prefixes = [scope]

        for param in self.model.parameters():
            param.requires_grad = False

        trainable_count = 0
        for name, param in self.model.named_parameters():
            if any(name == prefix or name.startswith(f"{prefix}.") for prefix in trainable_prefixes):
                param.requires_grad = True
                trainable_count += param.numel()

        if trainable_count == 0:
            raise ValueError(
                f"model.trainable_scope={scope!r} matched no parameters. "
                f"Available top-level modules: {sorted({n.split('.')[0] for n, _ in self.model.named_parameters()})}"
            )
        logger.info(
            "Applied trainable scope '%s': %d parameters remain trainable (%s)",
            scope,
            trainable_count,
            ", ".join(trainable_prefixes),
        )

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def train_dataloader(self) -> DataLoader:
        dataset = self.dataset_cls("training", self.cfg.dataset)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.dataset_cls.collate_fn,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=4 if self.num_workers > 0 else None,
            worker_init_fn=partial(_seed_worker, rank_id=self.trainer.global_rank),
        )

    def val_dataloader(self) -> DataLoader:
        dataset = self.dataset_cls("validation", self.cfg.dataset)
        return DataLoader(
            dataset,
            batch_size=max(1, self.batch_size // 2),
            shuffle=False,
            collate_fn=self.dataset_cls.collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=4 if self.num_workers > 0 else None,
            worker_init_fn=partial(_seed_worker, rank_id=self.trainer.global_rank),
        )

    def test_dataloader(self) -> DataLoader:
        dataset = self.dataset_cls("testing", self.cfg.dataset)
        batch_size = self.batch_size if self.stage == "diffusion" else 1
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self.dataset_cls.collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    # ------------------------------------------------------------------
    # Optimizers
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        loss_dict = self._forward_step(batch, is_training=True)
        total_loss = loss_dict["total_loss"]

        # Log all sub-losses
        for key, value in loss_dict.items():
            if key in ("total_loss", "t"):
                continue
            if isinstance(value, torch.Tensor) and value.numel() == 1:
                self.log(
                    f"train/{key}", value,
                    prog_bar=False, logger=True,
                    on_step=False, on_epoch=True,
                    sync_dist=True, batch_size=self.batch_size,
                )

        self.log(
            "train/loss", total_loss,
            prog_bar=True, logger=True,
            on_step=True, on_epoch=True,
            sync_dist=True, batch_size=self.batch_size,
        )
        return total_loss

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        loss_dict = self._forward_step(batch, is_training=False)
        total_loss = loss_dict["total_loss"]

        # Log all sub-losses
        for key, value in loss_dict.items():
            if key in ("total_loss", "t"):
                continue
            if isinstance(value, torch.Tensor) and value.numel() == 1:
                self.log(
                    f"val/{key}", value,
                    prog_bar=False, logger=True,
                    on_step=False, on_epoch=True,
                    sync_dist=True, batch_size=self.batch_size,
                )

        self.log(
            "val/loss", total_loss,
            prog_bar=True, logger=True,
            on_step=False, on_epoch=True,
            sync_dist=True, batch_size=self.batch_size,
        )

        # Quick inference visualization on first batch (diffusion only)
        if (
            batch_idx == 0
            and self.global_rank == 0
            and self.stage == "diffusion"
        ):
            self._diffusion_viz(batch)

        return total_loss

    # ------------------------------------------------------------------
    # Test
    # ------------------------------------------------------------------

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> Optional[torch.Tensor]:
        """Run full inference and save predictions to NPZ."""
        output_dir = Path(
            self.cfg.eval.output_dir
            or self.cfg.trainer.test_output_dir
            or self.log_root / "test_outputs"
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        total_loss = None
        loss_dict = {}
        if self.stage == "vae" or "cached_latent_stats" in batch:
            loss_dict = self._forward_step(batch, is_training=False)
            total_loss = loss_dict["total_loss"]
            self.log(
                "test/loss", total_loss,
                prog_bar=True, logger=True,
                on_step=False, on_epoch=True,
                sync_dist=True, batch_size=self.batch_size,
            )

        # Run model inference for generation
        if self.stage == "vae":
            _, recon_data = self.model(batch, v_test=True)
            self._save_vae_predictions(batch, recon_data, loss_dict, output_dir)

        elif self.stage == "diffusion":
            num_items = len(batch.get("v_prefix", [None]))
            results = self.model.inference(
                num_items, self.device, v_data=batch,
            )
            self._save_diffusion_predictions(batch, results, output_dir)

        return total_loss

    # ------------------------------------------------------------------
    # Dispatch helpers
    # ------------------------------------------------------------------

    def _forward_step(self, batch: Dict[str, Any], is_training: bool) -> Dict[str, Any]:
        """Dispatch forward pass based on stage."""
        if self.stage == "vae":
            run_validation_inference = self.cfg.trainer.vae_validation_inference
            loss, _data = self.model(
                batch,
                v_test=(not is_training and run_validation_inference),
            )
            return loss

        elif self.stage == "diffusion":
            loss = self.model(batch, v_test=not is_training)
            return loss

        else:
            raise ValueError(f"Unknown stage: '{self.stage}'. Must be 'vae' or 'diffusion'.")

    # ------------------------------------------------------------------
    # Visualization & I/O helpers
    # ------------------------------------------------------------------

    def _diffusion_viz(self, batch: Dict[str, Any]):
        """Quick generation visualization during validation (rank 0 only)."""
        try:
            results = self.model.inference(1, self.device, v_data=batch)
            if results and "pred_face" in results[0]:
                pred_faces = results[0]["pred_face"]
                import trimesh
                pc = trimesh.PointCloud(pred_faces.reshape(-1, 3))
                out_path = self.log_root / f"epoch{self.current_epoch:05d}_val_viz.ply"
                pc.export(str(out_path))
                logger.info("Saved validation viz to %s", out_path)
        except Exception as e:
            logger.warning("Diffusion viz failed (non-critical): %s", e)

    def _save_vae_predictions(
        self,
        batch: Dict[str, Any],
        recon_data: Dict[str, Any],
        loss_dict: Dict[str, Any],
        output_dir: Path,
    ):
        """Save VAE reconstruction predictions to NPZ."""
        prefixes = batch.get("v_prefix", [f"sample_{0}"])
        for idx, prefix in enumerate(prefixes):
            item_dir = output_dir / str(prefix)
            item_dir.mkdir(parents=True, exist_ok=True)

            save_dict = {}
            for key in ("pred_face", "pred_edge", "gt_face", "gt_edge",
                        "pred_face_adj", "gt_face_adj", "pred_edge_face_connectivity",
                        "gt_edge_face_connectivity"):
                if key in recon_data:
                    val = recon_data[key]
                    save_dict[key] = val.cpu().numpy() if isinstance(val, torch.Tensor) else val

            # Add per-item losses
            for key in ("face_coords", "edge_coords"):
                if key in loss_dict and isinstance(loss_dict[key], torch.Tensor):
                    save_dict[f"{key}_loss"] = loss_dict[key].cpu().item()

            np.savez_compressed(str(item_dir / "data.npz"), **save_dict)

    def _save_diffusion_predictions(
        self,
        batch: Dict[str, Any],
        results: list,
        output_dir: Path,
    ):
        """Save diffusion generation predictions to NPZ."""
        prefixes = batch.get("v_prefix", [f"sample_{i}" for i in range(len(results))])
        for idx, recon_data in enumerate(results):
            prefix = prefixes[idx] if idx < len(prefixes) else f"sample_{idx}"
            item_dir = output_dir / str(prefix)
            item_dir.mkdir(parents=True, exist_ok=True)

            save_dict = {}
            for key, val in recon_data.items():
                if isinstance(val, torch.Tensor):
                    save_dict[key] = val.cpu().numpy()
                elif isinstance(val, np.ndarray):
                    save_dict[key] = val

            np.savez_compressed(str(item_dir / "data.npz"), **save_dict)


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

@hydra.main(version_base="1.3", config_path="../../configs", config_name="train")
def main(cfg: DictConfig):
    """Unified training entry point for HoLa-BRep."""
    # --- Reproducibility ---
    seed_everything(42, workers=True)
    torch.set_float32_matmul_precision("medium")
    torch.backends.cudnn.benchmark = True

    # --- Print resolved config ---
    logger.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    # --- Output directory ---
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_dir = hydra_cfg["runtime"]["output_dir"]
    exp_name = cfg.trainer.exp_name
    log_dir = str(Path(output_dir))
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Inject output_dir back into config for the module.
    OmegaConf.update(cfg, "trainer.output_dir", log_dir, force_add=True)

    # --- Logger ---
    wandb_cfg = cfg.trainer.wandb
    use_wandb = wandb_cfg.enabled

    if use_wandb:
        pl_logger = WandbLogger(
            project=wandb_cfg.project,
            name=wandb_cfg.name or exp_name,
            save_dir=log_dir,
        )
    else:
        pl_logger = TensorBoardLogger(save_dir=log_dir, name="tb_logs")

    # --- Callbacks ---
    callbacks = [
        ModelCheckpoint(
            dirpath=str(Path(log_dir) / "checkpoints"),
            monitor="val/loss",
            mode="min",
            save_top_k=3,
            save_last=True,
            filename="{epoch:04d}-{val/loss:.4f}",
        ),
        LearningRateMonitor(logging_interval="step"),
        ModelSummary(max_depth=2),
    ]

    # --- Trainer ---
    strategy = "ddp_find_unused_parameters_true" if cfg.trainer.devices > 1 else "auto"

    trainer = Trainer(
        default_root_dir=log_dir,
        logger=pl_logger,
        accelerator=cfg.trainer.accelerator,
        strategy=strategy,
        devices=cfg.trainer.devices,
        callbacks=callbacks,
        max_epochs=cfg.trainer.max_epochs,
        max_steps=cfg.trainer.max_steps,
        precision=cfg.trainer.precision,
        check_val_every_n_epoch=cfg.trainer.check_val_every_n_epoch,
        num_sanity_val_steps=cfg.trainer.num_sanity_val_steps,
        gradient_clip_algorithm="norm",
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        enable_model_summary=True,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
    )

    # --- Module ---
    module = UnifiedTrainer(cfg)

    # --- Resume from checkpoint ---
    resume_from = cfg.trainer.resume_from_checkpoint
    ckpt_path = None
    if resume_from and resume_from != "none":
        ckpt_path = resume_from
        logger.info("Resuming training from: %s", ckpt_path)

    # --- Evaluate or Train ---
    if cfg.eval.enabled or cfg.trainer.evaluate:
        eval_ckpt = ckpt_path or "best"
        logger.info("Running evaluation with checkpoint: %s", eval_ckpt)
        trainer.test(module, ckpt_path=eval_ckpt if eval_ckpt != "best" else None)
    else:
        trainer.fit(module, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
