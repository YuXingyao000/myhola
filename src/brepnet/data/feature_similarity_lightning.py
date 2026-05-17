import json
import os
from pathlib import Path

import hydra
import numpy as np
import pytorch_lightning as pl
from lightning_fabric import seed_everything
from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader

from src.brepnet.data.fiedility import (
    Diffusion_dataset_fidelity,
    build_image_encoder,
    compute_max_sketch_similarity,
    plot_similarity_histogram,
    resolve_dataset_cls,
)


SPLITS = ("training", "validation", "testing")


def build_predict_dataloader(v_cfg: DictConfig, split: str) -> DataLoader:
    dataset_cls = resolve_dataset_cls(v_cfg["dataset"]["name"])
    dataset = dataset_cls(split, v_cfg["dataset"])

    requested_workers = int(v_cfg["trainer"]["num_worker"])
    force_zero_workers = dataset_cls.__name__ == Diffusion_dataset_fidelity.__name__
    num_workers = 0 if force_zero_workers else requested_workers
    if force_zero_workers and requested_workers != 0 and os.environ.get("LOCAL_RANK", "0") == "0":
        print(
            f"Force num_workers=0 for {dataset_cls.__name__} in Lightning fidelity mode "
            f"to avoid shared-memory crashes (requested {requested_workers})."
        )

    return DataLoader(
        dataset,
        batch_size=v_cfg["trainer"]["batch_size"],
        collate_fn=dataset_cls.collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        prefetch_factor=1 if num_workers > 0 else None,
    )


def merge_rank_outputs(raw_split_dir: Path, merged_dir: Path, split: str, world_size: int) -> Path:
    merged_dir.mkdir(parents=True, exist_ok=True)

    prefixes = []
    scores = []
    for rank in range(world_size):
        rank_path = raw_split_dir / f"rank_{rank:02d}.npz"
        if not rank_path.exists():
            raise FileNotFoundError(f"Missing rank output: {rank_path}")
        with np.load(rank_path, allow_pickle=False) as payload:
            rank_prefixes = payload["prefixes"]
            rank_scores = payload["scores"].astype(np.float32, copy=False)
        if rank_prefixes.shape[0] != rank_scores.shape[0]:
            raise RuntimeError(
                f"Mismatched rank output sizes in {rank_path}: "
                f"{rank_prefixes.shape[0]} prefixes vs {rank_scores.shape[0]} scores"
            )
        prefixes.append(rank_prefixes)
        scores.append(rank_scores)

    merged_prefixes = np.concatenate(prefixes, axis=0) if prefixes else np.empty((0,), dtype=np.str_)
    merged_scores = np.concatenate(scores, axis=0) if scores else np.empty((0,), dtype=np.float32)

    merged_path = merged_dir / f"{split}.npz"
    np.savez_compressed(merged_path, prefixes=merged_prefixes, scores=merged_scores)
    return merged_path


def load_merged_scores(output_root: Path) -> dict[str, np.ndarray]:
    results = {}
    merged_dir = output_root / "merged"
    for split in SPLITS:
        merged_path = merged_dir / f"{split}.npz"
        if not merged_path.exists():
            raise FileNotFoundError(f"Missing merged split output: {merged_path}")
        with np.load(merged_path, allow_pickle=False) as payload:
            results[split] = payload["scores"].astype(np.float32, copy=False)
    return results


def summarize_scores(split_results: dict[str, np.ndarray]) -> dict[str, dict[str, float | int]]:
    summary = {}
    for split, values in split_results.items():
        summary[split] = {
            "count": int(values.size),
            "mean": float(values.mean()) if values.size > 0 else float("nan"),
            "std": float(values.std()) if values.size > 0 else float("nan"),
        }
    return summary


class LightningFidelityModule(pl.LightningModule):
    def __init__(self, hparams: DictConfig, output_root: str):
        super().__init__()
        self.hydra_conf = hparams
        self.output_root = Path(output_root)
        self.image_encoder = build_image_encoder(hparams, torch.device("cpu"))
        self.active_split = ""
        self._local_prefixes: list[str] = []
        self._local_scores: list[torch.Tensor] = []

    def set_active_split(self, split: str) -> None:
        self.active_split = split

    def _reset_prediction_buffers(self) -> None:
        self._local_prefixes = []
        self._local_scores = []

    def on_predict_start(self) -> None:
        if not self.active_split:
            raise RuntimeError("active_split must be set before trainer.predict()")
        self._reset_prediction_buffers()
        self.image_encoder.eval()

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        del batch_idx, dataloader_idx
        features = self.image_encoder(batch["conditions"])
        scores = compute_max_sketch_similarity(features)
        self._local_prefixes.extend(list(batch["v_prefix"]))
        self._local_scores.append(scores.detach().cpu())
        return None

    def on_predict_epoch_end(self) -> None:
        raw_split_dir = self.output_root / "raw" / self.active_split
        raw_split_dir.mkdir(parents=True, exist_ok=True)

        scores = torch.cat(self._local_scores, dim=0).numpy() if self._local_scores else np.empty((0,), dtype=np.float32)
        prefixes = np.asarray(self._local_prefixes, dtype=np.str_)
        rank_path = raw_split_dir / f"rank_{self.global_rank:02d}.npz"
        np.savez_compressed(rank_path, prefixes=prefixes, scores=scores.astype(np.float32, copy=False))

        self.trainer.strategy.barrier(f"saved_{self.active_split}")
        if self.trainer.is_global_zero:
            merged_path = merge_rank_outputs(
                raw_split_dir=raw_split_dir,
                merged_dir=self.output_root / "merged",
                split=self.active_split,
                world_size=self.trainer.world_size,
            )
            print(f"Merged {self.active_split} scores -> {merged_path}")
        self.trainer.strategy.barrier(f"merged_{self.active_split}")
        self._reset_prediction_buffers()


@hydra.main(config_name="train_diffusion.yaml", config_path="../../../configs/brepnet/", version_base="1.1")
def main(v_cfg: DictConfig) -> None:
    seed_everything(0, workers=True)
    torch.set_float32_matmul_precision("medium")

    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_root = Path(hydra_cfg["runtime"]["output_dir"]) / "fidelity_outputs"
    output_root.mkdir(parents=True, exist_ok=True)

    trainer_gpu = int(v_cfg["trainer"]["gpu"])
    use_gpu = torch.cuda.is_available() and trainer_gpu > 0
    trainer = pl.Trainer(
        default_root_dir=str(output_root),
        accelerator="gpu" if use_gpu else "cpu",
        devices=trainer_gpu if use_gpu else 1,
        strategy="ddp" if use_gpu and trainer_gpu > 1 else "auto",
        precision=v_cfg["trainer"]["accelerator"],
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        inference_mode=True,
        use_distributed_sampler=True,
    )

    model = LightningFidelityModule(v_cfg, str(output_root))
    for split in SPLITS:
        model.set_active_split(split)
        dataloader = build_predict_dataloader(v_cfg, split)
        trainer.predict(model, dataloaders=dataloader, return_predictions=False)

    if trainer.is_global_zero:
        split_results = load_merged_scores(output_root)
        hist_path = plot_similarity_histogram(split_results, output_root)
        summary = summarize_scores(split_results)
        summary_path = output_root / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))
        for split, values in split_results.items():
            print(
                f"{split}: count={values.size}, "
                f"mean={values.mean() if values.size > 0 else float('nan'):.4f}, "
                f"std={values.std() if values.size > 0 else float('nan'):.4f}"
            )
        print(f"Histogram saved to {hist_path}")
        print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
