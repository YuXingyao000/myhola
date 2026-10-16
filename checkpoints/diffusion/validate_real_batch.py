"""Validate one real cached sample with local pretrained backbones; no optimizer step.

Run from the repository root with explicit asset paths. The condition file is a
FLUX synthetic photorealistic image, not a real photograph. Two-step sampling is
an interface diagnostic and must not be reported as reconstruction quality.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import sys
import tempfile
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-data-root", type=Path, required=True)
    parser.add_argument("--condition-data-root", type=Path, required=True)
    parser.add_argument("--latent-cache-root", type=Path, required=True)
    parser.add_argument("--ae-checkpoint", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config-name", default="train_diffusion_layerwise_real")
    parser.add_argument("--config-dir", type=Path, default=Path(__file__).resolve().parents[2] / "configs")
    parser.add_argument("--hf-cache-home", type=Path, help="Optional local Hugging Face cache root; downloads remain disabled.")
    parser.add_argument("--seed", type=int, default=20261016)
    parser.add_argument("--sampling-steps", type=int, default=2)
    parser.add_argument("--num-threads", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sampling_steps < 1 or args.num_threads < 1:
        raise ValueError("sampling-steps and num-threads must be positive.")
    for path in (args.raw_data_root, args.condition_data_root, args.latent_cache_root, args.ae_checkpoint):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.hf_cache_home is not None:
        os.environ["HF_HOME"] = str(args.hf_cache_home.resolve())
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    import numpy as np
    import torch
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from src.brepnet.dataset import Diffusion_dataset
    from src.brepnet.diffusion.model import Diffusion

    device = torch.device(args.device)
    torch.set_num_threads(args.num_threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()

    with initialize_config_dir(version_base=None, config_dir=str(args.config_dir.resolve())):
        cfg = compose(config_name=args.config_name)
    cfg.dataset.raw_data_root = str(args.raw_data_root.resolve())
    cfg.dataset.condition_data_root = str(args.condition_data_root.resolve())
    cfg.dataset.cached_latent_root = str(args.latent_cache_root.resolve())
    cfg.model.autoencoder.checkpoint = str(args.ae_checkpoint.resolve())
    cfg.dataset.is_aug = False
    cfg.dataset.is_overfit = False
    cfg.dataset.validation_real_photo_ratio = 1.0
    cfg.dataset.load_topology = False
    cfg.model.topology_bias.enabled = False
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    condition_cfg = OmegaConf.to_container(cfg.condition, resolve=True)

    # Validation mode with no augmentation selects the identity rotation (0).
    with tempfile.TemporaryDirectory(prefix="hola-layerwise-validation-") as temporary:
        sample_list = Path(temporary) / "samples.txt"
        sample_list.write_text(args.model_id + "\n")
        cfg.dataset.val_dataset = str(sample_list)
        data_cfg = OmegaConf.to_container(cfg.dataset, resolve=True)
        dataset = Diffusion_dataset("validation", data_cfg)
        if len(dataset) != 1:
            raise ValueError(f"Expected one accepted CAD item; found {len(dataset)}.")
        batch = Diffusion_dataset.collate_fn([dataset[0]])

    def to_device(item):
        if isinstance(item, torch.Tensor):
            return item.to(device)
        if isinstance(item, dict):
            return {key: to_device(value) for key, value in item.items()}
        return item

    batch = to_device(batch)
    model = Diffusion(model_cfg, condition_cfg).to(device).train()
    shapes = {}

    def record_shape(name):
        def hook(_module, _inputs, output):
            shapes[name] = list(output.shape)
        return hook

    hooks = [
        model.condition_encoder.register_forward_hook(record_shape("condition")),
        model.denoiser.register_forward_hook(record_shape("prediction")),
    ]
    losses = model(batch)
    losses["total_loss"].backward()
    for hook in hooks:
        hook.remove()
    trainable = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    missing = [name for name, param in trainable if param.grad is None]
    finite = all(torch.isfinite(param.grad).all().item() for _, param in trainable if param.grad is not None)
    if missing or not finite:
        raise AssertionError(f"Gradient validation failed: missing={missing}, finite={finite}.")
    assert not model.autoencoder.training
    assert not model.condition_encoder.image_encoder.img_model.training
    assert all(param.grad is None for param in model.autoencoder.parameters())
    assert all(param.grad is None for param in model.condition_encoder.image_encoder.img_model.parameters())
    cross_norms = [float(layer.attention.q_proj_weight.grad.norm()) for layer in model.denoiser.cross_attention_layers]
    assert len(cross_norms) == model_cfg["denoiser"]["num_layers"] and min(cross_norms) > 0

    report = {
        "purpose": "Implementation validation only; no optimizer update or quality claim",
        "command": shlex.join([sys.executable, *sys.argv]),
        "seed": args.seed,
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
        "model_id": args.model_id,
        "rotation_id": int(batch["rotation_id"][0]),
        "condition_source": str(args.condition_data_root.resolve() / args.model_id / "real_photo.npz") + ":flux (synthetic photorealistic)",
        "latent_source": str(args.latent_cache_root.resolve() / f"{args.model_id}_0/features.npy"),
        "ae_checkpoint": model_cfg["autoencoder"]["checkpoint"],
        "batch_shapes": {**shapes, "cached_latent_stats": list(batch["cached_latent_stats"].shape), "image": list(batch["conditions"]["imgs"].shape)},
        "layers": len(model.denoiser.backbone.layers),
        "cross_attention_layers": len(cross_norms),
        "trainable_parameters": sum(param.numel() for _, param in trainable),
        "missing_gradients": missing,
        "finite_gradients": finite,
        "cross_attention_q_gradient_norm_min": min(cross_norms),
        "frozen_backbones_no_grad_and_eval": True,
        "losses": {key: float(value.detach()) for key, value in losses.items() if key != "t"},
        "resolved_config": {"model": model_cfg, "condition": condition_cfg, "dataset": data_cfg},
        "split_override": {"mode": "validation", "cad_ids": [args.model_id], "temporary_list_removed": True},
        "offline": True,
    }
    model.zero_grad(set_to_none=True)
    model.eval()
    model.noise_scheduler.set_timesteps(args.sampling_steps)
    with torch.inference_mode():
        outputs = model.inference(1, device, v_data={"conditions": batch["conditions"]}, v_log=False)
    for name, value in outputs[0].items():
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            assert torch.isfinite(value).all(), f"Non-finite decoded output: {name}"
        elif isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.floating):
            assert np.isfinite(value).all(), f"Non-finite decoded output: {name}"
    report["diagnostic_sampling_steps"] = args.sampling_steps
    report["sampling_inputs"] = ["conditions.imgs"]
    report["decoded_outputs"] = {key: list(value.shape) for key, value in outputs[0].items() if hasattr(value, "shape")}
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        report["peak_cuda_allocated_gib"] = torch.cuda.max_memory_allocated(device) / 1024**3
    report["elapsed_seconds"] = time.monotonic() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": "passed", "report": str(args.output), "elapsed_seconds": report["elapsed_seconds"]}))


if __name__ == "__main__":
    main()
