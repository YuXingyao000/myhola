"""Short DDP throughput diagnostic with real data; never saves training checkpoints.

Launch with torchrun. Each rank records timing boundaries without synchronizing
each step; CUDA events are read after one final device synchronization. The
reported loader wait is a host-side proxy, not a standalone I/O profiler.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys
import time
import types


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--config-name", default="train_diffusion_layerwise_svr")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--torch-threads", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strategy", default="ddp_find_unused_parameters_true")
    parser.add_argument("--cache-state", required=True, help="Describe existing OS file-cache state; this script does not flush caches.")
    parser.add_argument("--data-mode", choices=["real_loader", "resident_batch"], default="real_loader")
    parser.add_argument(
        "--synchronize-warmup-boundary", action="store_true",
        help="Synchronize once after warmup so its queued GPU work cannot enter measured host time.",
    )
    return parser.parse_args()


def select_source(source_root: Path, repository_root: Path) -> None:
    """An archived src/brepnet may lack src/__init__.py; make precedence explicit."""
    if "src.brepnet" in sys.modules:
        raise RuntimeError("Choose the source tree before importing src.brepnet.")
    sys.path.insert(0, str(repository_root))
    sys.path.insert(0, str(source_root))
    package = types.ModuleType("src")
    package.__path__ = [str(source_root / "src"), str(repository_root / "src")]
    sys.modules["src"] = package


def tensor_hash(tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def main() -> None:
    args = arguments()
    if not 0 <= args.warmup < args.steps:
        raise ValueError("Use 0 <= warmup < steps.")
    source_root = args.source_root.resolve()
    repository_root = Path(__file__).resolve().parents[2]
    report_dir = args.report_dir.resolve()
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if (report_dir / "metadata/run.json").exists() or (report_dir / f"rank{rank:03d}.json").exists():
        raise FileExistsError(f"Refusing to overwrite an existing run: {report_dir}")
    report_dir.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        with (report_dir / "benchmark-reservation.json").open("x") as stream:
            json.dump({"command": shlex.join([sys.executable, *sys.argv])}, stream)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    select_source(source_root, repository_root)

    import numpy as np
    import torch
    from hydra import compose, initialize_config_dir
    from lightning_fabric import seed_everything
    from omegaconf import OmegaConf
    import pytorch_lightning as pl
    from torch.utils.data import DataLoader, Dataset, DistributedSampler
    import src.brepnet.train as training
    import src.brepnet.dataset as dataset_module

    imported_paths = {"train": Path(training.__file__).resolve(), "dataset": Path(dataset_module.__file__).resolve()}
    for name, path in imported_paths.items():
        if not path.is_relative_to(source_root):
            raise RuntimeError(f"{name} unexpectedly imported from {path}")
    torch.set_num_threads(args.torch_threads)
    torch.cuda.set_device(local_rank)
    seed_everything(args.seed, workers=True)
    with initialize_config_dir(version_base=None, config_dir=str(source_root / "configs")):
        cfg = compose(config_name=args.config_name)
    cfg.trainer.batch_size = args.batch_size
    cfg.trainer.num_workers = args.workers
    cfg.trainer.devices = world_size
    cfg.trainer.seed = args.seed
    cfg.trainer.max_steps = args.steps
    cfg.trainer.run_dir = str(report_dir)
    cfg.trainer.output_dir = str(report_dir)
    cfg.trainer.resume_from_checkpoint = None
    cfg.trainer.init_from_checkpoint = None
    cfg.trainer.num_sanity_val_steps = 0
    cfg.trainer.limit_val_batches = 0
    cfg.trainer.wandb.enabled = False
    if args.strategy == "config":
        if not hasattr(training, "_build_strategy"):
            raise RuntimeError("This source tree does not expose the production _build_strategy helper.")
        strategy = training._build_strategy(cfg)
        OmegaConf.update(cfg, "trainer.num_threads", args.torch_threads, force_add=True)
        training._configure_num_threads(cfg)
    else:
        strategy = args.strategy if world_size > 1 else "auto"
    # Keep file identities stable even when the source snapshot is elsewhere.
    for key in ("train_dataset", "val_dataset", "test_dataset"):
        value = Path(cfg.dataset[key])
        if not value.is_absolute():
            cfg.dataset[key] = str(source_root / value)
    torch.backends.cudnn.benchmark = bool(cfg.trainer.cudnn_benchmark)

    records = []
    events = []
    captured = {}
    timer = {"previous_end": None, "measured_start": None, "finished": None}

    class ResidentDataset(Dataset):
        def __init__(self, batch):
            self.batch = batch

        def __len__(self):
            return args.steps * world_size

        def __getitem__(self, index):
            return self.batch

    class MeasuredTrainer(training.UnifiedTrainer):
        def train_dataloader(self):
            loader = super().train_dataloader()
            if args.data_mode == "real_loader":
                return loader
            sampler = DistributedSampler(loader.dataset, num_replicas=world_size, rank=rank, seed=args.seed, shuffle=True)
            indices = list(iter(sampler))[:args.batch_size]
            batch = loader.collate_fn([loader.dataset[index] for index in indices])
            batch = self.transfer_batch_to_device(batch, self.device, 0)
            return DataLoader(ResidentDataset(batch), batch_size=None, num_workers=0)

        def on_before_batch_transfer(self, batch, dataloader_idx):
            now = time.perf_counter()
            step = len(records)
            previous = timer["previous_end"]
            record = {
                "step": step,
                "before_transfer": now,
                "loader_wait_proxy_seconds": None if previous is None else now - previous,
                "cad_ids": list(batch["v_prefix"]),
            }
            records.append(record)
            if step == args.warmup:
                timer["measured_start"] = previous if previous is not None else now
                # Retain one existing batch without hashing inside the timed loop.
                captured["images"] = batch["conditions"]["imgs"].detach()
                captured["latent_stats"] = batch["cached_latent_stats"].detach()
                captured["rotation_id"] = batch["rotation_id"].detach()
            return super().on_before_batch_transfer(batch, dataloader_idx)

        def on_after_batch_transfer(self, batch, dataloader_idx):
            records[-1]["after_transfer"] = time.perf_counter()
            return super().on_after_batch_transfer(batch, dataloader_idx)

    class Timings(pl.Callback):
        def on_train_start(self, trainer, module):
            torch.cuda.reset_peak_memory_stats(local_rank)
            if args.synchronize_warmup_boundary and args.warmup == 0:
                torch.cuda.synchronize(local_rank)
            timer["previous_end"] = time.perf_counter()

        def on_train_batch_start(self, trainer, module, batch, batch_idx):
            records[-1]["batch_start"] = time.perf_counter()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            events.append((start, end))

        def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
            events[-1][1].record()
            if args.synchronize_warmup_boundary and len(records) == args.warmup:
                torch.cuda.synchronize(local_rank)
            now = time.perf_counter()
            records[-1]["batch_end"] = now
            timer["previous_end"] = now

        def on_train_end(self, trainer, module):
            torch.cuda.synchronize(local_rank)
            timer["finished"] = time.perf_counter()

    started = time.perf_counter()
    module = MeasuredTrainer(cfg)
    trainer = pl.Trainer(
        accelerator="gpu", devices=world_size,
        strategy=strategy,
        precision=cfg.trainer.precision, max_steps=args.steps, max_epochs=1,
        logger=False, enable_checkpointing=False, enable_progress_bar=False,
        enable_model_summary=False, num_sanity_val_steps=0, limit_val_batches=0,
        callbacks=[Timings()], log_every_n_steps=cfg.trainer.log_every_n_steps,
        gradient_clip_algorithm="norm", gradient_clip_val=cfg.trainer.gradient_clip_val,
        deterministic=cfg.trainer.deterministic,
        default_root_dir=str(report_dir),
    )
    trainer.fit(module)
    if len(records) != args.steps or len(events) != args.steps:
        raise AssertionError(f"Expected {args.steps} steps; got records={len(records)}, events={len(events)}")
    for record, (start, end) in zip(records, events):
        record["cuda_core_ms"] = start.elapsed_time(end)
        record["host_core_seconds"] = record["batch_end"] - record["batch_start"]
        record["host_transfer_dispatch_seconds"] = record["after_transfer"] - record["before_transfer"]
    measured = records[args.warmup:]
    elapsed = timer["finished"] - timer["measured_start"]
    identifiers = [record["cad_ids"] for record in records]
    report = {
        "purpose": "Short throughput diagnostic; optimizer updates are discarded, no training checkpoints or scientific quality claim.",
        "rank": rank, "world_size": world_size,
        "source_root": str(source_root), "source_commit": args.source_commit,
        "source_paths": {key: str(path) for key, path in imported_paths.items()},
        "source_sha256": {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in imported_paths.items()},
        "command": shlex.join([sys.executable, *sys.argv]),
        "config": OmegaConf.to_container(cfg, resolve=True),
        "torch": torch.__version__, "device": torch.cuda.get_device_name(local_rank),
        "torch_num_threads": torch.get_num_threads(), "torch_num_interop_threads": torch.get_num_interop_threads(),
        "seed": args.seed, "steps": args.steps, "warmup_steps": args.warmup,
        "synchronize_warmup_boundary": args.synchronize_warmup_boundary,
        "batch_size_per_rank": args.batch_size, "strategy": args.strategy,
        "effective_strategy": {"class": type(strategy).__name__, "ddp_options": getattr(strategy, "_ddp_kwargs", {})},
        "data_mode": args.data_mode, "cache_state": args.cache_state,
        "os_cache_flushed": False, "pretrained_downloads_allowed": False,
        "ordinary_step_isolation": {"logger": False, "progress": False, "checkpoint": False, "sanity": False, "validation": False},
        "all_step_cad_order_sha256": hashlib.sha256(json.dumps(identifiers).encode()).hexdigest(),
        "first_measured_batch_sha256": {name: tensor_hash(value) for name, value in captured.items()},
        "first_measured_batch_shapes": {name: list(value.shape) for name, value in captured.items()},
        "measured_elapsed_seconds": elapsed,
        "measured_cuda_event_span_ms": events[args.warmup][0].elapsed_time(events[-1][1]),
        "measured_samples_per_second_per_rank": len(measured) * args.batch_size / elapsed,
        "cuda_core_ms_median": float(np.median([item["cuda_core_ms"] for item in measured])),
        "cuda_core_ms_p95": float(np.quantile([item["cuda_core_ms"] for item in measured], .95)),
        "loader_wait_proxy_seconds_total": sum(item["loader_wait_proxy_seconds"] for item in measured),
        "loader_wait_proxy_seconds_median": float(np.median([item["loader_wait_proxy_seconds"] for item in measured])),
        "loader_wait_proxy_seconds_p95": float(np.quantile([item["loader_wait_proxy_seconds"] for item in measured], .95)),
        "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated(local_rank) / 1024**3,
        "total_process_seconds": time.perf_counter() - started,
        "timing_limits": [
            "loader_wait_proxy = previous batch-end hook to next before-transfer hook; includes loader fetch and framework overhead.",
            "CUDA core events include stream dependencies/communication and CPU submission gaps; they are not active SM compute time.",
            "Final throughput includes the final CUDA synchronization and minor train-end callback overhead; no per-step barrier/synchronize was added.",
            "Without --synchronize-warmup-boundary, measured host time can include the queued GPU tail of warmup.",
        ],
        "steps_detail": records,
    }
    with (report_dir / f"rank{rank:03d}.json").open("x") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"rank": rank, "seconds": elapsed, "cuda_core_ms_median": report["cuda_core_ms_median"], "loader_wait_seconds_total": report["loader_wait_proxy_seconds_total"]}), flush=True)


if __name__ == "__main__":
    main()
