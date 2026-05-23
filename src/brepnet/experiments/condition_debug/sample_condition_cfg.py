import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from src.brepnet.experiments.condition_debug.common import (
    add_common_model_args,
    create_dataloader,
    load_model,
    move_to_device,
    prepare_output_dir,
    save_reconstruction,
    set_seed,
    write_run_metadata,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample conditional diffusion with inference-time CFG.")
    add_common_model_args(parser)
    parser.add_argument("--cfg-scale", type=float, default=2.0)
    parser.add_argument("--cfg-mid-scale", type=float, default=1.5)
    parser.add_argument("--cfg-low-scale", type=float, default=1.0)
    parser.add_argument("--cfg-high-t", type=int, default=500)
    parser.add_argument("--cfg-mid-t", type=int, default=200)
    parser.add_argument("--cfg-schedule", choices=["constant", "high-t"], default="high-t")
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--layout", choices=["flat", "sample-subdirs"], default="flat")
    parser.add_argument("--suffix-template", default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--amp", choices=["none", "bf16"], default="bf16")
    return parser.parse_args()


def autocast_context(device: torch.device, amp: str):
    if device.type == "cuda" and amp == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return torch.autocast(device_type=device.type, enabled=False)


def scale_for_timestep(args: argparse.Namespace, timestep: int) -> float:
    if args.cfg_schedule == "constant":
        return args.cfg_scale
    if timestep >= args.cfg_high_t:
        return args.cfg_scale
    if timestep >= args.cfg_mid_t:
        return args.cfg_mid_scale
    return args.cfg_low_scale


def decode_latent_sequence(model, latent_sequence: torch.Tensor) -> list[dict]:
    mask = model.face_padder.inference_mask(latent_sequence)

    recon_data = []
    for item_idx in range(latent_sequence.shape[0]):
        face_latents = latent_sequence[item_idx:item_idx + 1][mask[item_idx:item_idx + 1]]
        face_latents = model.face_padder.postprocess(face_latents)
        recon_data.append(model.autoencoder.decode_latents(face_latents))
    return recon_data


def inference_cfg(model, batch_size: int, device: torch.device, batch: dict, args: argparse.Namespace) -> list[dict]:
    latent_sequence = torch.randn((batch_size, model.max_faces, model.latent_dim), device=device)
    condition = model.extract_condition(batch)[:batch_size]
    zero_condition = torch.zeros_like(condition)

    for timestep_tensor in tqdm(model.noise_scheduler.timesteps, leave=False):
        timestep_value = int(timestep_tensor.item())
        timesteps = timestep_tensor.reshape(-1).to(device)
        scale = scale_for_timestep(args, timestep_value)

        pred_cond = model.diffuse(
            latent_sequence,
            timesteps,
            condition=condition,
        )
        pred_zero = model.diffuse(
            latent_sequence,
            timesteps,
            condition=zero_condition,
        )
        pred = pred_zero + scale * (pred_cond - pred_zero)
        latent_sequence = model.noise_scheduler.step(pred, timestep_tensor, latent_sequence).prev_sample

    return decode_latent_sequence(model, latent_sequence)


def main() -> None:
    args = parse_args()
    if args.num_samples < 1:
        raise ValueError("--num-samples must be >= 1")
    suffix_template = args.suffix_template
    if suffix_template is None:
        suffix_template = "" if args.num_samples == 1 else "__s{sample:02d}"

    output_root = prepare_output_dir(args.output_root, force=args.force)
    (output_root / "_metadata").mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    set_seed(args.base_seed)
    model, load_metadata = load_model(args, device)
    dataloader = create_dataloader(args)

    write_run_metadata(
        output_root / "_metadata" / "sample_condition_cfg.json",
        args,
        {
            "mode": "sample_condition_cfg",
            "model_load": load_metadata,
        },
    )

    for sample_idx in range(args.num_samples):
        set_seed(args.base_seed + sample_idx)
        progress = tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            desc=f"cfg sample {sample_idx + 1}/{args.num_samples}",
        )
        for batch_idx, batch in progress:
            if args.max_batches is not None and batch_idx >= args.max_batches:
                break
            batch = move_to_device(batch, device)
            batch_size = len(batch["v_prefix"])
            with torch.no_grad(), autocast_context(device, args.amp):
                results = inference_cfg(model, batch_size, device, batch, args)
            for item_idx, recon_data in enumerate(results):
                base_prefix = batch["v_prefix"][item_idx]
                if args.layout == "sample-subdirs":
                    candidate_root = Path(output_root) / f"s{sample_idx:02d}"
                    suffix = ""
                else:
                    candidate_root = Path(output_root)
                    suffix = suffix_template.format(sample=sample_idx)
                save_reconstruction(
                    output_root=candidate_root,
                    prefix=f"{base_prefix}{suffix}",
                    recon_data=recon_data,
                    condition_batch=batch.get("conditions"),
                    batch_index=item_idx,
                )


if __name__ == "__main__":
    main()
