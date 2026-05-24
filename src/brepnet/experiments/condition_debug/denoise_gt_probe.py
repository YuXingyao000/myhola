import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.brepnet.experiments.condition_debug.common import (
    add_common_model_args,
    condition_from_mode,
    create_dataloader,
    finite_values,
    load_model,
    mean_or_none,
    median_or_none,
    move_to_device,
    prepare_output_dir,
    resolve_device,
    set_seed,
    write_csv,
    write_json,
    write_run_metadata,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe denoising from noised GT latent.")
    add_common_model_args(parser)
    parser.add_argument("--timesteps", nargs="+", type=int, default=[50, 100, 200, 500, 800])
    parser.add_argument("--condition-modes", nargs="+", default=["normal", "shuffle", "zero"])
    parser.add_argument("--max-batches", type=int, default=None)
    return parser.parse_args()


def predict_x0(model, zt: torch.Tensor, timesteps: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    prediction_type = model.noise_scheduler.config.prediction_type
    if prediction_type == "sample":
        return pred
    if prediction_type != "epsilon":
        raise NotImplementedError(f"Unsupported prediction_type for probe: {prediction_type}")
    alphas = model.noise_scheduler.alphas_cumprod.to(device=zt.device, dtype=zt.dtype)
    alpha_t = alphas[timesteps].reshape(-1, 1, 1)
    return (zt - torch.sqrt(1.0 - alpha_t) * pred) / torch.sqrt(alpha_t)


def summarize(rows: list[dict]) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["condition_mode"], row["timestep"])].append(row)
    by_mode_timestep = {}
    for (mode, timestep), items in sorted(grouped.items()):
        key = f"{mode}_t{timestep}"
        by_mode_timestep[key] = {
            "count": len(items),
            "epsilon_mse_mean": mean_or_none(finite_values(items, "epsilon_mse")),
            "epsilon_mse_median": median_or_none(finite_values(items, "epsilon_mse")),
            "x0_l1_mean": mean_or_none(finite_values(items, "x0_l1")),
            "x0_l1_median": median_or_none(finite_values(items, "x0_l1")),
            "x0_mse_mean": mean_or_none(finite_values(items, "x0_mse")),
            "x0_mse_median": median_or_none(finite_values(items, "x0_mse")),
        }
    return {
        "num_rows": len(rows),
        "by_mode_timestep": by_mode_timestep,
    }


def main() -> None:
    args = parse_args()
    output_root = prepare_output_dir(args.output_root, force=args.force)
    device = resolve_device(args.device)
    set_seed(args.base_seed)
    model, load_metadata = load_model(args, device)
    dataloader = create_dataloader(args)

    rows = []
    for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc="denoise probe"):
        if args.max_batches is not None and batch_idx >= args.max_batches:
            break
        batch = move_to_device(batch, device)
        with torch.no_grad():
            latent_sequence = model.latent_provider(batch).values
            condition = model.extract_condition(batch)
        batch_size = latent_sequence.shape[0]
        prefixes = list(batch["v_prefix"])

        for timestep in args.timesteps:
            set_seed(args.base_seed + timestep + batch_idx * 10000)
            timesteps = torch.full((batch_size,), int(timestep), device=device, dtype=torch.long)
            noise = torch.randn_like(latent_sequence)
            zt = model.noise_scheduler.add_noise(latent_sequence, noise, timesteps)
            for mode in args.condition_modes:
                with torch.no_grad():
                    cond_mode = condition_from_mode(condition, mode)
                    pred, _ = model.diffuse(zt, timesteps, cond_mode, latent_sequence)
                    x0_pred = predict_x0(model, zt, timesteps, pred)
                epsilon_mse = ((pred - noise) ** 2).mean(dim=(1, 2)).detach().cpu().numpy()
                x0_l1 = (x0_pred - latent_sequence).abs().mean(dim=(1, 2)).detach().cpu().numpy()
                x0_mse = ((x0_pred - latent_sequence) ** 2).mean(dim=(1, 2)).detach().cpu().numpy()
                for item_idx, prefix in enumerate(prefixes):
                    rows.append(
                        {
                            "prefix": prefix,
                            "batch_idx": batch_idx,
                            "condition_mode": mode,
                            "timestep": int(timestep),
                            "epsilon_mse": float(epsilon_mse[item_idx]),
                            "x0_l1": float(x0_l1[item_idx]),
                            "x0_mse": float(x0_mse[item_idx]),
                        }
                    )

    summary = summarize(rows)
    write_json(output_root / "denoise_errors.json", {"summary": summary, "rows": rows})
    write_csv(output_root / "denoise_errors.csv", rows)
    write_run_metadata(
        output_root / "denoise_errors.metadata.json",
        args,
        {"model_load": load_metadata},
    )
    print(summary)


if __name__ == "__main__":
    main()
