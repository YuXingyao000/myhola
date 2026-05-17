import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from src.brepnet.experiments.condition_debug.common import (
    add_common_model_args,
    create_dataloader,
    install_condition_mode,
    load_model,
    move_to_device,
    prepare_output_dir,
    resolve_device,
    safe_mkdir,
    save_reconstruction,
    set_seed,
    write_run_metadata,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample conditional diffusion candidates.")
    add_common_model_args(parser)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--condition-mode", choices=["normal", "shuffle", "zero"], default="normal")
    parser.add_argument(
        "--layout",
        choices=["flat", "sample-subdirs"],
        default="flat",
        help="flat writes candidates as prefix suffixes; sample-subdirs writes sXX/prefix.",
    )
    parser.add_argument(
        "--suffix-template",
        default=None,
        help="Python format string with {sample}; default is empty for K=1 and __s{sample:02d} for K>1.",
    )
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--amp", choices=["none", "bf16"], default="bf16")
    return parser.parse_args()


def autocast_context(device: torch.device, amp: str):
    if device.type == "cuda" and amp == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return torch.autocast(device_type=device.type, enabled=False)


def main() -> None:
    args = parse_args()
    if args.num_samples < 1:
        raise ValueError("--num-samples must be >= 1")
    suffix_template = args.suffix_template
    if suffix_template is None:
        suffix_template = "" if args.num_samples == 1 else "__s{sample:02d}"

    output_root = prepare_output_dir(args.output_root, force=args.force)
    safe_mkdir(output_root / "_metadata")

    device = resolve_device(args.device)
    set_seed(args.base_seed)
    model, load_metadata = load_model(args, device)
    install_condition_mode(model, args.condition_mode)
    dataloader = create_dataloader(args)

    write_run_metadata(
        output_root / "_metadata" / "sample_condition.json",
        args,
        {
            "mode": "sample_condition",
            "condition_mode": args.condition_mode,
            "num_samples": args.num_samples,
            "suffix_template": suffix_template,
            "model_load": load_metadata,
        },
    )

    for sample_idx in range(args.num_samples):
        sample_seed = args.base_seed + sample_idx
        set_seed(sample_seed)
        progress = tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            desc=f"sample {sample_idx + 1}/{args.num_samples}",
        )
        for batch_idx, batch in progress:
            if args.max_batches is not None and batch_idx >= args.max_batches:
                break
            batch = move_to_device(batch, device)
            batch_size = len(batch["v_prefix"])
            with torch.no_grad(), autocast_context(device, args.amp):
                results = model.inference(batch_size, device, v_data=batch, v_log=False)
            for item_idx, recon_data in enumerate(results):
                base_prefix = batch["v_prefix"][item_idx]
                if args.layout == "sample-subdirs":
                    candidate_root = Path(output_root) / f"s{sample_idx:02d}"
                    suffix = ""
                else:
                    candidate_root = Path(output_root)
                    suffix = suffix_template.format(sample=sample_idx)
                out_prefix = f"{base_prefix}{suffix}"
                save_reconstruction(
                    output_root=candidate_root,
                    prefix=out_prefix,
                    recon_data=recon_data,
                    condition_batch=batch.get("conditions"),
                    batch_index=item_idx,
                )


if __name__ == "__main__":
    main()
