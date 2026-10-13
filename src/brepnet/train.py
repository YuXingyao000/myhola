"""
Unified Training Entry Point for HoLa-BRep.

Single Lightning module for VAE and diffusion training, dispatching by
``cfg.model.stage``.

Usage Examples
--------------
.. code-block:: bash

    # VAE training (Stage 1)
    python -m src.brepnet.train --config-name train_vae \
        dataset.raw_data_root=/path/to/data

    # Diffusion with single-image condition (Stage 2)
    python -m src.brepnet.train --config-name train_diffusion_white \
        trainer.devices=8

"""

from __future__ import annotations

import importlib
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

import hydra
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
from src.brepnet.prediction_export import save_item_predictions, vae_prediction_items
from src.brepnet.run_management import (
    predictions_directory, prepare_run, register_run_resolvers,
)

register_run_resolvers()

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

def _validate_evaluation_devices(cfg: DictConfig) -> None:
    """Avoid duplicated distributed test samples writing the same NPZ paths."""
    if (cfg.eval.enabled or cfg.trainer.evaluate) and cfg.trainer.devices > 1:
        raise ValueError(
            "Evaluation currently requires trainer.devices=1: the distributed test "
            "sampler can repeat objects and multiple ranks would write the same prediction files."
        )


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
        self.log_root = Path(cfg.trainer.get("run_dir") or cfg.trainer.output_dir)
        self.log_root.mkdir(parents=True, exist_ok=True)

        # --- Build model ---
        logger.info("Building model for stage='%s'", self.stage)
        model_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
        resume = cfg.trainer.resume_from_checkpoint
        has_resume = resume and str(resume).lower() not in {"none", "null"}
        if cfg.model.name == "AutoEncoder_joint":
            if cfg.trainer.init_from_checkpoint:
                raise ValueError(
                    "AutoEncoder_joint rejects trainer.init_from_checkpoint. Use "
                    "model.init_encoder_checkpoint for validated legacy migration, or "
                    "trainer.resume_from_checkpoint for a strict full-state restore."
                )
            if has_resume:
                # The checkpoint already owns encoder/decoder weights. Do not first
                # migrate a second, potentially absent legacy checkpoint.
                model_cfg.init_encoder_checkpoint = None
                model_cfg.checkpoint = None
            elif model_cfg.freeze_encoder and not model_cfg.init_encoder_checkpoint and not model_cfg.checkpoint:
                raise ValueError(
                    "A fresh frozen-encoder run needs model.init_encoder_checkpoint "
                    "or a strict model.checkpoint; refusing to freeze a random encoder."
                )
        if cfg.model.name == "AutoEncoder_intersection":
            if cfg.trainer.init_from_checkpoint:
                raise ValueError(
                    "AutoEncoder_intersection rejects trainer.init_from_checkpoint. "
                    "Use model.init_legacy_checkpoint for strict original VAE migration, "
                    "model.init_refinement_checkpoint for validated refinement migration, "
                    "model.checkpoint for new-model initialization, or trainer.resume_from_checkpoint."
                )
            if has_resume:
                model_cfg.init_legacy_checkpoint = None
                OmegaConf.update(model_cfg, "init_refinement_checkpoint", None, force_add=True)
                model_cfg.checkpoint = None
                OmegaConf.update(model_cfg, "restoring_checkpoint", True, force_add=True)
        self.model = build_model(model_cfg, cfg.condition)
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
        if self.cfg.trainer.resume_from_checkpoint:
            raise ValueError("Choose either init_from_checkpoint or resume_from_checkpoint, not both.")

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
        )

    def test_dataloader(self) -> DataLoader:
        dataset = self.dataset_cls("testing", self.cfg.dataset)
        batched_vae = self.cfg.model.name in {"AutoEncoder_joint", "AutoEncoder_intersection"}
        batch_size = self.batch_size if self.stage == "diffusion" or batched_vae else 1
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
        optimizer = torch.optim.Adam(
            (parameter for parameter in self.model.parameters() if parameter.requires_grad),
            lr=self.learning_rate,
        )
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
                    sync_dist=True, batch_size=len(batch["v_prefix"]),
                )

        self.log(
            "train/loss", total_loss,
            prog_bar=True, logger=True,
            on_step=True, on_epoch=True,
            sync_dist=True, batch_size=len(batch["v_prefix"]),
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
                    sync_dist=True, batch_size=len(batch["v_prefix"]),
                )

        self.log(
            "val/loss", total_loss,
            prog_bar=True, logger=True,
            on_step=False, on_epoch=True,
            sync_dist=True, batch_size=len(batch["v_prefix"]),
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
        output_dir = predictions_directory(self.cfg, self.log_root)
        output_dir.mkdir(parents=True, exist_ok=True)

        total_loss = None
        loss_dict = {}
        recon_data = None
        if self.stage == "vae":
            # Report the loss of the exact deterministic posterior-mean
            # reconstruction being exported, rather than a separate random
            # posterior sample controlled by the validation-only setting.
            loss_dict, recon_data = self.model(batch, v_test=True)
        elif "cached_latent_stats" in batch:
            loss_dict = self._forward_step(batch, is_training=False)
        if loss_dict:
            total_loss = loss_dict["total_loss"]
            self.log(
                "test/loss", total_loss,
                prog_bar=True, logger=True,
                on_step=False, on_epoch=True,
                sync_dist=True, batch_size=len(batch["v_prefix"]),
            )

        # Run model inference for generation
        if self.stage == "vae":
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
                out_path = self.log_root / "visualizations" / f"epoch{self.current_epoch:05d}_val_viz.ply"
                out_path.parent.mkdir(parents=True, exist_ok=True)
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
        prefixes = batch["v_prefix"]
        save_item_predictions(prefixes, vae_prediction_items(prefixes, recon_data, loss_dict), output_dir)

    def _save_diffusion_predictions(
        self,
        batch: Dict[str, Any],
        results: list,
        output_dir: Path,
    ):
        """Save diffusion generation predictions to NPZ."""
        save_item_predictions(batch["v_prefix"], results, output_dir)


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

@hydra.main(version_base="1.3", config_path="../../configs", config_name="train")
def main(cfg: DictConfig):
    """Unified training entry point for HoLa-BRep."""
    _validate_evaluation_devices(cfg)
    logger.info(
        "Device setup: devices=%s, batch_size_per_device=%s, global_train_batch_size=%s "
        "(single node, no gradient accumulation). W&B records metrics; trainer.devices selects GPUs.",
        cfg.trainer.devices, cfg.trainer.batch_size,
        cfg.trainer.devices * cfg.trainer.batch_size,
    )
    # --- Reproducibility ---
    seed_everything(cfg.trainer.seed, workers=True)
    torch.set_float32_matmul_precision("medium")
    torch.backends.cudnn.benchmark = bool(cfg.trainer.cudnn_benchmark) and not cfg.trainer.deterministic

    # --- Print resolved config ---
    logger.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    # --- Output directory ---
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_dir = hydra_cfg["runtime"]["output_dir"]
    exp_name = cfg.trainer.exp_name
    log_dir = str(Path(output_dir).resolve())
    OmegaConf.update(cfg, "trainer.run_dir", log_dir, force_add=True)
    resume_from = cfg.trainer.resume_from_checkpoint
    ckpt_path = None
    if resume_from and str(resume_from).lower() not in {"none", "null"}:
        ckpt_path = str(Path(resume_from).expanduser().resolve())
        if not Path(ckpt_path).is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {ckpt_path}")
        OmegaConf.update(cfg, "trainer.resume_from_checkpoint", ckpt_path)
    evaluating = bool(cfg.eval.enabled or cfg.trainer.evaluate)
    if evaluating and ckpt_path is None:
        raise ValueError("Evaluation requires an explicit trainer.resume_from_checkpoint; no random/best fallback.")
    if evaluating:
        OmegaConf.update(cfg, "eval.output_dir", str(predictions_directory(cfg, Path(log_dir))))
        logger.info("Raw predictions will be saved to %s", cfg.eval.output_dir)
    run_record = prepare_run(cfg, Path(log_dir))

    # --- Logger ---
    wandb_cfg = cfg.trainer.wandb
    use_wandb = wandb_cfg.enabled

    if use_wandb:
        pl_logger = WandbLogger(
            project=wandb_cfg.project,
            name=wandb_cfg.name or exp_name,
            save_dir=log_dir,
            id=run_record["wandb_id"],
            resume="allow",
            offline=wandb_cfg.offline,
            log_model=False,
        )
    else:
        pl_logger = TensorBoardLogger(save_dir=str(Path(log_dir) / "logs"), name="tensorboard")
    if "git" in run_record:
        pl_logger.log_hyperparams({"seed": cfg.trainer.seed, "git": run_record["git"]})

    # --- Callbacks ---
    callbacks = [
        ModelCheckpoint(
            dirpath=str(Path(log_dir) / "weights"),
            monitor="val/loss",
            mode="min",
            save_top_k=3,
            save_last=True,
            filename="epoch={epoch:04d}-step={step:08d}-val_loss={val/loss:.6f}",
            auto_insert_metric_name=False,
            enable_version_counter=cfg.trainer.checkpoint_enable_version_counter,
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
        deterministic=cfg.trainer.deterministic,
        limit_train_batches=cfg.trainer.limit_train_batches,
        limit_val_batches=cfg.trainer.limit_val_batches,
        limit_test_batches=cfg.trainer.limit_test_batches,
    )

    # --- Module ---
    module = UnifiedTrainer(cfg)

    # --- Resume from checkpoint ---
    if ckpt_path:
        logger.info("Resuming training from: %s", ckpt_path)

    # --- Evaluate or Train ---
    if evaluating:
        logger.info("Running evaluation with checkpoint: %s", ckpt_path)
        trainer.test(module, ckpt_path=ckpt_path)
    else:
        trainer.fit(module, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
