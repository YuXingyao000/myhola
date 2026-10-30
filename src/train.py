"""
Unified Training Entry Point for Img2Brep.

Single Lightning module for VAE and diffusion training, dispatching by
``cfg.model.stage``.

Usage Examples
--------------
.. code-block:: bash

    # VAE training (Stage 1)
    python -m src.train --config-name train_vae \
        dataset.raw_data_root=/path/to/data

    # Diffusion with single-image condition (Stage 2)
    python -m src.train --config-name train_diffusion_white \
        trainer.devices=8

"""

from __future__ import annotations

import importlib
import logging
import os
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
from pytorch_lightning.strategies import DDPStrategy

from src.models import build_model
from src.common.prediction_export import save_item_predictions, vae_prediction_items
from src.common.run_management import (
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

def _configure_num_threads(cfg: DictConfig) -> None:
    """Apply an explicit CPU thread budget to each training process."""
    num_threads = cfg.trainer.get("num_threads")
    if num_threads is None:
        return
    if isinstance(num_threads, bool) or not isinstance(num_threads, int) or num_threads < 1:
        raise ValueError("trainer.num_threads must be a positive integer or null.")
    # Lightning's DDP launcher otherwise replaces the parent process's budget.
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    torch.set_num_threads(num_threads)


def _build_strategy(cfg: DictConfig) -> DDPStrategy | str:
    """Share the configured DDP behavior between training and benchmarks."""
    if cfg.trainer.devices <= 1:
        return "auto"
    return DDPStrategy(
        find_unused_parameters=cfg.trainer.get("ddp_find_unused_parameters", True),
        gradient_as_bucket_view=cfg.trainer.get("ddp_gradient_as_bucket_view", False),
    )


def _build_checkpoint_callbacks(cfg: DictConfig, run_dir: Path) -> list[ModelCheckpoint]:
    """Keep validation-ranked weights and optional fixed-step snapshots separately."""
    callbacks = [
        ModelCheckpoint(
            dirpath=str(run_dir / "weights"),
            monitor="val/loss",
            mode="min",
            save_top_k=3,
            save_last=True,
            filename="epoch={epoch:04d}-step={step:08d}-val_loss={val/loss:.6f}",
            auto_insert_metric_name=False,
            enable_version_counter=cfg.trainer.checkpoint_enable_version_counter,
        ),
    ]
    interval = cfg.trainer.get("checkpoint_every_n_train_steps")
    if interval is not None:
        if isinstance(interval, bool) or not isinstance(interval, int) or interval < 1:
            raise ValueError("trainer.checkpoint_every_n_train_steps must be a positive integer or null.")
        keep = cfg.trainer.get("checkpoint_step_keep_last", -1)
        if isinstance(keep, bool) or not isinstance(keep, int) or keep == 0 or keep < -1:
            raise ValueError("checkpoint_step_keep_last must be -1 or a positive integer")
        callbacks.append(ModelCheckpoint(
            dirpath=str(run_dir / "weights" / "steps"),
            every_n_train_steps=interval,
            every_n_epochs=0,
            save_top_k=keep,
            monitor="step" if keep > 0 else None,
            mode="max",
            save_last=True,
            # Also retain the final state when max_steps ends between snapshots.
            save_on_train_epoch_end=True,
            filename="step={step:08d}",
            auto_insert_metric_name=False,
            enable_version_counter=cfg.trainer.checkpoint_enable_version_counter,
        ))
    return callbacks


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
    if "." in name:
        module_path, class_name = name.rsplit(".", 1)
        return getattr(importlib.import_module(module_path), class_name)
    dataset_mod = importlib.import_module("src.datasets.brep")
    if not hasattr(dataset_mod, name):
        raise AttributeError(
            f"Dataset class '{name}' not found in src.datasets.brep. "
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
        if cfg.model.name == "PhotoBaselineDiffusion":
            from src.datasets.photo import photo_cache_provenance
            provenance = photo_cache_provenance(cfg.dataset)
            expected = cfg.dataset.get("provenance")
            if expected is not None and dict(expected) != provenance:
                raise ValueError("Photo cache changed after prepare_run")
            OmegaConf.update(cfg.dataset, "provenance", provenance, force_add=True)
        if cfg.model.name == "PreferenceDiffusion":
            from src.datasets.preference import preference_data_provenance
            provenance = preference_data_provenance(cfg.dataset)
            expected = cfg.dataset.get("provenance")
            if expected is not None and provenance != dict(expected):
                raise ValueError("Preference data changed after prepare_run.")
            OmegaConf.update(cfg.dataset, "provenance", provenance, force_add=True)
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
                    "model.init_curve_feedback_checkpoint to add final point feedback, "
                    "model.init_encoder_edge_value_checkpoint to add GAT edge messages, "
                    "model.init_face_encoder_checkpoint to replace only the face CNN, "
                    "model.init_uv_boundary_checkpoint to add supervised UV paths, "
                    "model.init_shared_curve_checkpoint to initialize shared-curve UV decoding, "
                    "model.init_latent_expansion_checkpoint for channel-wise capacity expansion, "
                    "model.init_conditional_intersection_checkpoint for conditional face attention, "
                    "model.checkpoint for new-model initialization, or trainer.resume_from_checkpoint."
                )
            if has_resume:
                model_cfg.init_legacy_checkpoint = None
                OmegaConf.update(model_cfg, "init_refinement_checkpoint", None, force_add=True)
                OmegaConf.update(model_cfg, "init_curve_feedback_checkpoint", None, force_add=True)
                OmegaConf.update(model_cfg, "init_encoder_edge_value_checkpoint", None, force_add=True)
                OmegaConf.update(model_cfg, "init_face_encoder_checkpoint", None, force_add=True)
                OmegaConf.update(model_cfg, "init_uv_boundary_checkpoint", None, force_add=True)
                OmegaConf.update(model_cfg, "init_shared_curve_checkpoint", None, force_add=True)
                OmegaConf.update(model_cfg, "init_latent_expansion_checkpoint", None, force_add=True)
                OmegaConf.update(model_cfg, "init_conditional_intersection_checkpoint", None, force_add=True)
                model_cfg.checkpoint = None
                OmegaConf.update(model_cfg, "restoring_checkpoint", True, force_add=True)
            if model_cfg.get("uv_boundary", {}).get("enabled", False) and not cfg.dataset.get("edge_uv_root"):
                raise ValueError("UV boundary supervision requires dataset.edge_uv_root")
        if cfg.model.name == "AutoEncoder_uv_intersection":
            if cfg.trainer.init_from_checkpoint:
                raise ValueError(
                    "AutoEncoder_uv_intersection rejects trainer.init_from_checkpoint. "
                    "Use model.init_uv_decoder_checkpoint for validated encoder/face migration, "
                    "model.checkpoint for strict full-model initialization, or "
                    "trainer.resume_from_checkpoint for a full-state restore."
                )
            if has_resume:
                model_cfg.init_uv_decoder_checkpoint = None
                model_cfg.checkpoint = None
                OmegaConf.update(model_cfg, "restoring_checkpoint", True, force_add=True)
            elif model_cfg.freeze_encoder and not model_cfg.init_uv_decoder_checkpoint and not model_cfg.checkpoint:
                raise ValueError(
                    "A fresh frozen-encoder UV run needs model.init_uv_decoder_checkpoint "
                    "or a strict model.checkpoint; refusing to freeze a random encoder."
                )
        if cfg.model.name == "PreferenceDiffusion":
            from src.common.io import sha256_file
            from src.models.diffusion.preference_geometry import geometry_settings

            photo_source = model_cfg.get("model_source") == "photo"
            if photo_source != (cfg.dataset.provenance.get("protocol_version") == "photo_preference_generation_v1"):
                raise ValueError("Preference model and dataset photo/grey sources must agree")
            if cfg.trainer.init_from_checkpoint:
                raise ValueError("Use model.init_preference_checkpoint for a new DPO run, or resume_from_checkpoint.")
            geometry = geometry_settings(model_cfg.get("geometry"))
            if geometry["enabled"] != bool((cfg.dataset.get("geometry") or {}).get("enabled", False)):
                raise ValueError("Model and dataset geometry supervision switches must agree")
            if geometry["enabled"] and str(cfg.trainer.precision) not in {"32", "32-true"}:
                raise ValueError("The fixed geometry experiment requires FP32 training")
            if cfg.eval.enabled or cfg.trainer.evaluate or cfg.trainer.diffusion_validation_inference:
                raise ValueError("Preference training exposes sample_latents(); STEP evaluation requires the explicit original VAE pipeline.")
            if has_resume:
                saved = torch.load(str(resume), map_location="cpu", mmap=True, weights_only=False)
                if dict(cfg.dataset.provenance) != saved["hyper_parameters"]["dataset"]["provenance"]:
                    raise ValueError("Cannot resume preference training with a changed pair pool/protocol.")
                saved_model_cfg = saved["hyper_parameters"]["model"]
                if photo_source and any(model_cfg.get(key) != saved_model_cfg.get(key) for key in (
                        "model_source", "photo_global_step", "beta_dpo", "preference_weight")):
                    raise ValueError("Cannot resume photo DPO with changed source or objective")
                if geometry != geometry_settings(saved_model_cfg.get("geometry")):
                    raise ValueError("Cannot resume preference training with changed geometry settings")
                OmegaConf.update(
                    model_cfg, "legacy_model_config", saved_model_cfg["legacy_model_config"], force_add=True,
                )
                if "preference_initialization" in saved_model_cfg:
                    OmegaConf.update(cfg.model, "preference_initialization",
                                     saved_model_cfg["preference_initialization"], force_add=True)
            elif sha256_file(Path(model_cfg.photo_checkpoint if photo_source else model_cfg.legacy_checkpoint)) != cfg.dataset.provenance.source_checkpoint_sha256:
                raise ValueError("Initial SVR checkpoint differs from the preference generation checkpoint.")
            if photo_source and not has_resume:
                if (sha256_file(Path(model_cfg.legacy_checkpoint)) != cfg.dataset.provenance.original_checkpoint_sha256 or
                        model_cfg.photo_global_step != cfg.dataset.provenance.photo_global_step):
                    raise ValueError("Photo DPO original DINO/AE or selected baseline step differs from generation")
            OmegaConf.update(model_cfg, "restoring_checkpoint", bool(has_resume), force_add=True)
        if cfg.model.name == "PhotoBaselineDiffusion":
            from src.common.io import sha256_file
            if cfg.trainer.init_from_checkpoint:
                raise ValueError("Photo baseline uses strict legacy initialization or full resume")
            if cfg.eval.enabled or cfg.trainer.evaluate or cfg.trainer.diffusion_validation_inference:
                raise ValueError("Use sample_latents and the explicit original AE for downstream evaluation")
            if has_resume:
                saved = torch.load(str(resume), map_location="cpu", mmap=True, weights_only=False)
                if saved["hyper_parameters"]["model"]["name"] != "PhotoBaselineDiffusion":
                    raise ValueError("Expected a PhotoBaselineDiffusion checkpoint")
                if dict(cfg.dataset.provenance) != saved["hyper_parameters"]["dataset"]["provenance"]:
                    raise ValueError("Cannot resume with a changed photo cache")
                OmegaConf.update(model_cfg, "legacy_model_config",
                                 saved["hyper_parameters"]["model"]["legacy_model_config"], force_add=True)
            elif sha256_file(Path(model_cfg.legacy_checkpoint)) != cfg.dataset.provenance.source_checkpoint_sha256:
                raise ValueError("Photo features/latents and initial weights have different sources")
            OmegaConf.update(model_cfg, "restoring_checkpoint", bool(has_resume), force_add=True)
        if cfg.model.name == "SpatialPhotoDiffusion":
            from src.models.diffusion.spatial_model import prepare_spatial_model_config
            prepare_spatial_model_config(cfg, model_cfg, str(resume) if has_resume else None)
        self.model = build_model(model_cfg, cfg.condition)
        if cfg.model.name == "SpatialPhotoDiffusion":
            OmegaConf.update(cfg.model, "legacy_model_config", self.model.legacy_model_config, force_add=True)
            self.save_hyperparameters(OmegaConf.to_container(cfg, resolve=True))
        if cfg.model.name == "PhotoBaselineDiffusion":
            OmegaConf.update(cfg.model, "legacy_model_config", self.model.legacy_model_config, force_add=True)
            self.save_hyperparameters(OmegaConf.to_container(cfg, resolve=True))
        if cfg.model.name == "PreferenceDiffusion":
            # 恢复只依赖自身 checkpoint 的架构/权重，不重新读取旧 SVR 权重。
            OmegaConf.update(cfg.model, "legacy_model_config", self.model.legacy_model_config, force_add=True)
            if model_cfg.get("init_preference_checkpoint") and not has_resume:
                initialization = self.model.initialize_preference_weights(
                    model_cfg.init_preference_checkpoint, dict(cfg.dataset.provenance))
                OmegaConf.update(cfg.model, "preference_initialization", initialization, force_add=True)
            self.save_hyperparameters(OmegaConf.to_container(cfg, resolve=True))
        self._init_model_from_checkpoint()
        self._apply_trainable_scope()

        # --- Dataset class (lazy loaded) ---
        self.dataset_cls = _load_dataset_class(cfg)
        logger.info("Dataset class: %s", self.dataset_cls.__name__)

        # --- Visualization state ---
        self.viz: Dict[str, Any] = {}

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        if self.cfg.model.name == "AutoEncoder_uv_intersection":
            # Strict tensor loading cannot detect same-shape changes to heads,
            # dropout, or other saved architecture semantics.
            self.model.validate_complete_checkpoint(checkpoint)

    def _init_model_from_checkpoint(self):
        """Load model weights without restoring optimizer/trainer state."""
        ckpt_path = (
            self.cfg.trainer.init_from_checkpoint
        )
        if not ckpt_path or str(ckpt_path).lower() in {"none", "null"}:
            return
        if self.cfg.trainer.resume_from_checkpoint:
            raise ValueError("Choose either init_from_checkpoint or resume_from_checkpoint, not both.")

        # Project-owned Lightning checkpoints also contain configuration metadata.
        checkpoint = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)
        model_state = {}
        for key, value in state_dict.items():
            if key.startswith("model."):
                model_state[key[len("model."):]] = value
            else:
                model_state[key] = value

        # 新 diffusion 架构禁止静默遗漏 cross-attention 参数。
        missing, unexpected = self.model.load_state_dict(model_state, strict=self.stage == "diffusion")
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
            persistent_workers=self.num_workers > 0 and self.cfg.trainer.get("persistent_workers", True),
            prefetch_factor=self.cfg.trainer.get("prefetch_factor", 4) if self.num_workers > 0 else None,
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
            persistent_workers=self.num_workers > 0 and self.cfg.trainer.get("persistent_workers", True),
            prefetch_factor=self.cfg.trainer.get("prefetch_factor", 4) if self.num_workers > 0 else None,
        )

    def test_dataloader(self) -> DataLoader:
        dataset = self.dataset_cls("testing", self.cfg.dataset)
        batched_vae = self.cfg.model.name in {
            "AutoEncoder_joint", "AutoEncoder_intersection", "AutoEncoder_uv_intersection",
        }
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

    def configure_gradient_clipping(
        self, optimizer, gradient_clip_val=None, gradient_clip_algorithm=None,
    ):
        # bf16/fp32 have no GradScaler to skip a nonfinite update. Fail before
        # norm clipping spreads one bad gradient to every parameter and Adam
        # state. Finite gradients use the same norm/clipping rule as Lightning.
        precision = self.trainer.precision_plugin
        if (gradient_clip_algorithm != "norm" or gradient_clip_val is None or gradient_clip_val <= 0
                or getattr(precision, "scaler", None) is not None):
            return super().configure_gradient_clipping(
                optimizer, gradient_clip_val, gradient_clip_algorithm,
            )
        try:
            torch.nn.utils.clip_grad_norm_(
                list(precision.main_params(optimizer)), gradient_clip_val,
                error_if_nonfinite=True,
            )
        except RuntimeError as error:
            if "non-finite" not in str(error):
                raise
            # Inspect individual tensors only on failure, not on every step.
            invalid = [
                name for name, parameter in self.named_parameters()
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
            ]
            raise RuntimeError(
                "Nonfinite gradient norm before optimizer update; "
                f"step={self.global_step}, rank={self.global_rank}, "
                f"batch={getattr(self, '_train_batch_prefixes', ())}, "
                f"nonfinite_parameters={invalid[:8]} (total={len(invalid)}). "
                "Parameters and optimizer state have not been updated in this step."
            ) from error

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        self._train_batch_prefixes = tuple(batch.get("v_prefix", ()))
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

        if self.cfg.trainer.get("sync_train_step_loss", True):
            self.log(
                "train/loss", total_loss,
                prog_bar=True, logger=True,
                on_step=True, on_epoch=True,
                sync_dist=True, batch_size=len(batch["v_prefix"]),
            )
        else:
            # Keep the full, globally averaged epoch loss for comparisons.
            self.log(
                "train/loss_epoch", total_loss,
                prog_bar=False, logger=True,
                on_step=False, on_epoch=True,
                sync_dist=True, batch_size=len(batch["v_prefix"]),
            )
            log_interval = self.cfg.trainer.log_every_n_steps
            if self.global_rank == 0 and log_interval > 0 and (self.global_step + 1) % log_interval == 0:
                # Local step diagnostics avoid both all-reduce and per-batch .item().
                self.log(
                    "train/loss_step_local", total_loss,
                    prog_bar=False, logger=True,
                    on_step=True, on_epoch=False,
                    sync_dist=False, rank_zero_only=True,
                    batch_size=len(batch["v_prefix"]),
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
            and self.cfg.trainer.get("diffusion_validation_inference", False)
            and not self.trainer.sanity_checking
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
            if self.cfg.model.name == "AutoEncoder_uv_intersection":
                # Validation measures the posterior mean without building export
                # items; the legacy visualization switch must not enable export.
                loss, _data = self.model(batch, v_test=False, v_use_mean=not is_training)
                return loss
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
    """Unified training entry point for Img2Brep."""
    _configure_num_threads(cfg)
    _validate_evaluation_devices(cfg)
    accumulation = cfg.trainer.get("accumulate_grad_batches", 1)
    logger.info(
        "Device setup: devices=%s, batch_size_per_device=%s, accumulate_grad_batches=%s, "
        "nominal_objects_per_optimizer_step=%s (single node; final partial accumulation may be smaller). "
        "W&B records metrics; trainer.devices selects GPUs.",
        cfg.trainer.devices, cfg.trainer.batch_size, accumulation,
        cfg.trainer.devices * cfg.trainer.batch_size * accumulation,
    )
    # --- Reproducibility ---
    seed_everything(cfg.trainer.seed, workers=True)
    torch.set_float32_matmul_precision(cfg.trainer.get("float32_matmul_precision", "medium"))
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
    if cfg.model.name == "PreferenceDiffusion":
        from src.datasets.preference import preference_data_provenance
        OmegaConf.update(
            cfg.dataset, "provenance", preference_data_provenance(cfg.dataset), force_add=True,
        )
    if cfg.model.name == "PhotoBaselineDiffusion":
        from src.datasets.photo import photo_cache_provenance
        OmegaConf.update(cfg.dataset, "provenance", photo_cache_provenance(cfg.dataset), force_add=True)
    if cfg.model.name == "SpatialPhotoDiffusion":
        from src.datasets.spatial_photo import spatial_data_provenance
        OmegaConf.update(cfg.dataset, "provenance", spatial_data_provenance(cfg.dataset), force_add=True)
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
        *_build_checkpoint_callbacks(cfg, Path(log_dir)),
        LearningRateMonitor(logging_interval="step"),
        ModelSummary(max_depth=2),
    ]

    # --- Trainer ---
    trainer = Trainer(
        default_root_dir=log_dir,
        logger=pl_logger,
        accelerator=cfg.trainer.accelerator,
        strategy=_build_strategy(cfg),
        devices=cfg.trainer.devices,
        callbacks=callbacks,
        max_epochs=cfg.trainer.max_epochs,
        max_steps=cfg.trainer.max_steps,
        max_time=cfg.trainer.get("max_time"),
        precision=cfg.trainer.precision,
        accumulate_grad_batches=accumulation,
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
