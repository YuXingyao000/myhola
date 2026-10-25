"""Actual original-forward equivalence and BF16 gradient preflight on photo data."""

import json
from pathlib import Path
import time

import torch

from src.brepnet.data.photo_dataset import PhotoFeatureDataset
from src.brepnet.diffusion.legacy import load_legacy_svr
from src.brepnet.diffusion.photo_baseline import PhotoBaselineDiffusion


def main():
    torch.set_num_threads(4)
    torch.manual_seed(42)
    root = Path("checkpoints/diffusion/photo_baseline")
    dataset = PhotoFeatureDataset("training", {"cache_root": str(root / "cache_v1"), "epoch_scale_factor": 1})
    batch = dataset.collate_fn([dataset[i] for i in range(64)])
    checkpoint = "/mnt/d/ckpt/HoLa/Diffusion_svr_sq30_1500k.ckpt"
    old, _ = load_legacy_svr(checkpoint)
    model = PhotoBaselineDiffusion({"legacy_checkpoint": checkpoint}).eval()
    noisy = torch.randn(2, 30, 32)
    time_batch = torch.tensor([0, 731])
    features = batch["image_features"][:2]
    with torch.no_grad():
        expected = old.diffuse(noisy, time_batch, old.img_fc(features)[:, None])
        actual = model.policy(noisy, time_batch, model.policy.image_condition(features))
    error = (expected-actual).abs().max().item()
    if not torch.equal(expected, actual):
        raise ValueError(f"Original CPU forward mismatch: {error}")
    del old
    model.cuda().train()
    batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=1e-4)
    losses = []
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    for step in range(12):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(batch)["total_loss"]
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite actual photo loss")
        loss.backward()
        if step == 0:
            missing = [name for name,p in model.policy.named_parameters() if p.grad is None]
            if missing or any(p.grad is not None for p in model.image_encoder.parameters()):
                raise ValueError(f"Gradient contract failed: {missing}")
            if any(not torch.isfinite(p.grad).all() for p in model.policy.parameters()):
                raise ValueError("Nonfinite gradient")
        torch.nn.utils.clip_grad_norm_(model.policy.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        losses.append(loss.item())
    torch.cuda.synchronize()
    result = {"original_fp32_max_abs_error": error, "original_fp32_bitwise_equal": True,
              "bf16_real_photo_batch": 64, "losses": losses,
              "all_policy_parameters_have_finite_gradients": True, "dino_frozen": True,
              "seconds_including_warmup": time.monotonic()-start,
              "peak_allocated_bytes": torch.cuda.max_memory_allocated()}
    (root / "MODEL_PREFLIGHT.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
