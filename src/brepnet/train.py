"""
Unified Training Entry Point for HoLa-BRep.

Single Lightning module that handles ALL training modes (VAE, Diffusion,
Strategy) from one entry point, dispatching based on ``cfg.model.stage``.

Usage Examples
--------------
.. code-block:: bash

    # VAE training (Stage 1)
    python -m src.brepnet.train_unified model=vae_1119 experiment=train_vae \
        dataset.data_root=/path/to/data

    # Diffusion with single-image condition (Stage 2)
    python -m src.brepnet.train_unified model=diffusion_cross_attn \
        condition=single_img trainer.gpus=8

    # Knowledge Distillation via strategy
    python -m src.brepnet.train_unified model=diffusion_cross_attn \
        condition=single_img strategy=distillation \
        strategy.teacher.checkpoint=/path/to/white.ckpt

    # Feature Mapper training
    python -m src.brepnet.train_unified strategy=feature_mapper \
        experiment=train_feature_mapper
"""

from __future__ import annotations

import importlib
import logging
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

from src.brepnet.models import (
    build_condition_extractor,
    build_model,
    build_strategy,
)

logger = logging.getLogger(__name__)


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

    Resolution order:
      1. cfg.dataset._target_ (Hydra-style full path)
      2. cfg.dataset.dataset_name (legacy, from src.brepnet.dataset module)
      3. cfg.dataset.name (fallback, from src.brepnet.dataset module)
    """
    # Hydra _target_ takes priority
    target = cfg.dataset.get("_target_", None)
    if target:
        parts = target.rsplit(".", 1)
        mod = importlib.import_module(parts[0])
        return getattr(mod, parts[1])

    # Legacy: look for dataset_name or name in the dataset module
    name = cfg.dataset.get("dataset_name", None) or cfg.dataset.get("name", None)
    if name is None:
        raise ValueError(
            "Dataset config must specify '_target_', 'dataset_name', or 'name'. "
            f"Got keys: {list(cfg.dataset.keys())}"
        )

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
      - ``"strategy"``: trains via strategy module (KD, feature mapper)
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(OmegaConf.to_container(cfg, resolve=True))

        # --- Core configuration ---
        self.stage: str = cfg.model.get("stage", "vae")
        self.batch_size: int = cfg.trainer.get("batch_size", 32)
        self.num_workers: int = cfg.trainer.get("num_workers", cfg.trainer.get("num_worker", 4))
        self.learning_rate: float = cfg.trainer.get("learning_rate", 1e-4)

        # --- Output directory ---
        self.log_root = Path(cfg.trainer.get("output_dir", "./outputs"))
        self.log_root.mkdir(parents=True, exist_ok=True)

        # --- Build model ---
        logger.info("Building model for stage='%s'", self.stage)
        self.model = build_model(cfg.model)
        self._init_model_from_checkpoint()
        self._apply_trainable_scope()

        # --- Condition extractor (diffusion / strategy stages) ---
        self.condition_extractor = None
        if self.stage in ("diffusion", "strategy"):
            cond_cfg = cfg.model.get("condition", None)
            if cond_cfg is not None:
                self.condition_extractor = build_condition_extractor(cond_cfg)
                logger.info("Condition extractor built: %s", type(self.condition_extractor).__name__)

        # --- Strategy (KD, FeatureMapper, etc.) ---
        self.strategy = None
        if self.stage == "strategy":
            strategy_cfg = cfg.get("strategy", None)
            if strategy_cfg is not None:
                self.strategy = build_strategy(strategy_cfg)
                logger.info("Strategy built: %s", type(self.strategy).__name__)
            else:
                raise ValueError(
                    "Stage is 'strategy' but no strategy config provided. "
                    "Set strategy=distillation or strategy=feature_mapper"
                )

        # --- Dataset class (lazy loaded) ---
        self.dataset_cls = _load_dataset_class(cfg)
        logger.info("Dataset class: %s", self.dataset_cls.__name__)

        # --- Visualization state ---
        self.viz: Dict[str, Any] = {}

    def _init_model_from_checkpoint(self):
        """Load model weights without restoring optimizer/trainer state."""
        ckpt_path = (
            self.cfg.trainer.get("init_from_checkpoint", None)
            or self.cfg.model.get("init_from_checkpoint", None)
        )
        if not ckpt_path or str(ckpt_path).lower() in {"none", "null"}:
            return

        checkpoint = torch.load(str(ckpt_path), map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        model_state = {}
        for key, value in state_dict.items():
            if key.startswith("model."):
                model_state[key[len("model."):]] = value
            elif not key.startswith(("strategy.", "condition_extractor.")):
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
        scope = self.cfg.model.get("trainable_scope", "all")
        if scope is None or str(scope).lower() in {"all", "none", "null"}:
            return

        if scope == "intersection":
            trainable_prefixes = self.cfg.model.get(
                "trainable_module_prefixes", ["inter", "classifier"]
            )
        else:
            trainable_prefixes = scope if isinstance(scope, (list, tuple)) else [scope]

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
        return DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=self.dataset_cls.collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    # ------------------------------------------------------------------
    # Optimizers
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        # Collect only trainable parameters
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.learning_rate,
            weight_decay=self.cfg.trainer.get("weight_decay", 0.0),
        )

        # Scheduler
        scheduler_name = self.cfg.get("scheduler", {})
        if isinstance(scheduler_name, str):
            scheduler_name = {"name": scheduler_name}

        sched_type = scheduler_name.get("name", "cosine") if hasattr(scheduler_name, "get") else "cosine"

        if sched_type == "cosine":
            max_steps = self.cfg.trainer.get("max_steps", -1)
            if max_steps <= 0:
                # Estimate from max_epochs (approximate)
                max_steps = self.cfg.trainer.get("max_epochs", 100000) * 1000
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max_steps,
                eta_min=self.learning_rate * 0.01,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }
        elif sched_type == "linear":
            warmup_steps = scheduler_name.get("warmup_steps", 1000) if hasattr(scheduler_name, "get") else 1000
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.01,
                total_iters=warmup_steps,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }
        else:
            # Constant LR
            return {"optimizer": optimizer}

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
            self.cfg.eval.get("output_dir", None)
            or self.cfg.trainer.get("test_output_dir", self.log_root / "test_outputs")
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        # Compute test loss
        loss_dict = self._forward_step(batch, is_training=False)
        total_loss = loss_dict["total_loss"]
        self.log(
            "test/loss", total_loss,
            prog_bar=True, logger=True,
            on_step=False, on_epoch=True,
            sync_dist=True, batch_size=1,
        )

        # Run model inference for generation
        if self.stage == "vae":
            _, recon_data = self.model(batch, v_test=True)
            self._save_vae_predictions(batch, recon_data, loss_dict, output_dir)

        elif self.stage == "diffusion":
            num_items = len(batch.get("v_prefix", [None]))
            condition = self._extract_condition(batch)
            results = self.model.inference(
                num_items, self.device, v_data=batch,
                condition_features=condition,
            )
            self._save_diffusion_predictions(batch, results, output_dir)

        elif self.stage == "strategy":
            if hasattr(self.strategy, "test_step"):
                self.strategy.test_step(batch, output_dir)

        return total_loss

    # ------------------------------------------------------------------
    # Dispatch helpers
    # ------------------------------------------------------------------

    def _forward_step(self, batch: Dict[str, Any], is_training: bool) -> Dict[str, Any]:
        """Dispatch forward pass based on stage."""
        if self.stage == "vae":
            run_validation_inference = self.cfg.trainer.get("vae_validation_inference", False)
            loss, _data = self.model(
                batch,
                v_test=(not is_training and run_validation_inference),
            )
            return loss

        elif self.stage == "diffusion":
            condition = self._extract_condition(batch)
            loss = self.model(batch, condition_features=condition, v_test=not is_training)
            return loss

        elif self.stage == "strategy":
            loss = self.strategy.training_step(
                batch=batch,
                model=self.model,
                condition_extractor=self.condition_extractor,
                is_training=is_training,
            )
            return loss

        else:
            raise ValueError(f"Unknown stage: '{self.stage}'. Must be 'vae', 'diffusion', or 'strategy'.")

    def _extract_condition(self, batch: Dict[str, Any]) -> Optional[Dict[str, torch.Tensor]]:
        """Extract conditioning features from the batch."""
        if self.condition_extractor is None:
            return None
        return self.condition_extractor(batch)

    # ------------------------------------------------------------------
    # Visualization & I/O helpers
    # ------------------------------------------------------------------

    def _diffusion_viz(self, batch: Dict[str, Any]):
        """Quick generation visualization during validation (rank 0 only)."""
        try:
            condition = self._extract_condition(batch)
            results = self.model.inference(1, self.device, v_data=batch, condition_features=condition)
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

@hydra.main(version_base="1.3", config_path="../../configs", config_name="__init__")
def main(cfg: DictConfig):
    """Unified training entry point for HoLa-BRep."""
    # --- Reproducibility ---
    seed_everything(cfg.trainer.get("seed", 42), workers=True)
    torch.set_float32_matmul_precision("medium")
    torch.backends.cudnn.benchmark = True

    # --- Print resolved config ---
    logger.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    # --- Output directory ---
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_dir = hydra_cfg["runtime"]["output_dir"]
    exp_name = cfg.trainer.get("exp_name", "default")
    log_dir = str(Path(output_dir) / exp_name)
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Inject output_dir back into config for the module.
    OmegaConf.update(cfg, "trainer.output_dir", log_dir, force_add=True)

    # --- Logger ---
    wandb_cfg = cfg.trainer.get("wandb", {})
    use_wandb = wandb_cfg.get("enabled", False) if isinstance(wandb_cfg, dict) else False
    if hasattr(wandb_cfg, "get"):
        use_wandb = wandb_cfg.get("enabled", False)

    if use_wandb:
        pl_logger = WandbLogger(
            project=wandb_cfg.get("project", "hola-brep"),
            name=wandb_cfg.get("name", exp_name),
            save_dir=log_dir,
            entity=wandb_cfg.get("entity", None),
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
    num_gpus = cfg.trainer.get("gpus", 1)
    strategy = "ddp_find_unused_parameters_true" if num_gpus > 1 else "auto"

    trainer = Trainer(
        default_root_dir=log_dir,
        logger=pl_logger,
        accelerator="gpu",
        strategy=strategy,
        devices=num_gpus,
        callbacks=callbacks,
        max_epochs=cfg.trainer.get("max_epochs", 1000000),
        max_steps=cfg.trainer.get("max_steps", -1),
        precision=cfg.trainer.get("precision", "bf16-mixed"),
        check_val_every_n_epoch=cfg.trainer.get("check_val_every_n_epoch", 1),
        num_sanity_val_steps=cfg.trainer.get("num_sanity_val_steps", 2),
        gradient_clip_algorithm="norm",
        gradient_clip_val=cfg.trainer.get("gradient_clip_val", 1.0),
        enable_model_summary=True,
        log_every_n_steps=cfg.trainer.get("log_every_n_steps", 50),
    )

    # --- Module ---
    module = UnifiedTrainer(cfg)

    # --- Resume from checkpoint ---
    resume_from = cfg.trainer.get("resume_from", None)
    ckpt_path = None
    if resume_from and resume_from != "none":
        ckpt_path = resume_from
        logger.info("Resuming training from: %s", ckpt_path)

    # --- Evaluate or Train ---
    if cfg.eval.get("enabled", False):
        eval_ckpt = ckpt_path or "best"
        logger.info("Running evaluation with checkpoint: %s", eval_ckpt)
        trainer.test(module, ckpt_path=eval_ckpt if eval_ckpt != "best" else None)
    else:
        trainer.fit(module, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
