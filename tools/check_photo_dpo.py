"""用真实照片batch检查初始化、冻结边界、缓存语义和FP32更新。"""

import json
from pathlib import Path
import time

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import torch

from src.brepnet.data.photo_preference_dataset import PhotoPreferenceDataset
from src.brepnet.data.preference_source import load_condition
from src.brepnet.diffusion.preference import PreferenceDiffusion


def main():
    torch.set_num_threads(4)
    torch.manual_seed(42)
    root = Path(__file__).resolve().parents[1]
    with initialize_config_dir(config_dir=str(root / "configs"), version_base="1.3"):
        config = OmegaConf.to_container(compose("train_diffusion_preference_photo"), resolve=True)
    data = PhotoPreferenceDataset("training", config["dataset"])
    batch = data.collate_fn([data[i] for i in range(8)])
    model = PreferenceDiffusion(config["model"])
    saved = torch.load(config["model"]["photo_checkpoint"], map_location="cpu", mmap=True, weights_only=False)
    for name, value in model.policy.state_dict().items():
        assert torch.equal(value, saved["state_dict"]["model.policy." + name]), name
        assert torch.equal(value, model.reference.state_dict()[name]), name
    for name, value in model.image_encoder.state_dict().items():
        assert torch.equal(value, saved["state_dict"]["model.image_encoder." + name]), name
    print("Photo policy/reference and frozen DINO match checkpoint exactly", flush=True)
    model.cuda().eval()
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    image = load_condition(config["dataset"], data.identities[0])[None].cuda()
    with torch.no_grad():
        online = model.image_encoder(image).cpu()[0]
    cached = batch["image_features"][0]
    feature_error = (online - cached).abs().max().item()
    torch.testing.assert_close(online, cached, rtol=1e-4, atol=1e-5)
    print(f"Online photo DINO versus frozen cache max error: {feature_error}", flush=True)
    model.train()
    torch.set_float32_matmul_precision("medium")  # Same setting as the unified trainer.
    batch = {key: value.cuda() if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=1e-5)
    losses = []
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        result = model(batch)
        assert all(torch.isfinite(value).all() for value in result.values())
        if step == 0:
            assert abs(result["preference_loss"].item() - 0.69314718056) < 1e-6
        result["total_loss"].backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.policy.parameters())
        assert all(p.grad is None for p in model.reference.parameters())
        assert all(p.grad is None for p in model.image_encoder.parameters())
        norm = torch.nn.utils.clip_grad_norm_(model.policy.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        losses.append({**{key: value.item() for key, value in result.items()}, "gradient_norm": norm.item()})
        print({"step": step + 1, **losses[-1]}, flush=True)
    for name, value in model.reference.state_dict().items():
        assert torch.equal(value.cpu(), saved["state_dict"]["model.policy." + name]), name
    for name, value in model.image_encoder.state_dict().items():
        assert torch.equal(value.cpu(), saved["state_dict"]["model.image_encoder." + name]), name
    validation = PhotoPreferenceDataset("validation", config["dataset"])
    val_batch = validation.collate_fn([validation[i] for i in range(2)])
    val_batch = {key: value.cuda() if isinstance(value, torch.Tensor) else value for key, value in val_batch.items()}
    with torch.no_grad():
        val_loss = model.eval()(val_batch, v_test=True)
    assert set(val_loss) == {"ddpm_loss", "total_loss"} and torch.isfinite(val_loss["total_loss"])
    report = {"training_images": len(data), "training_pairs": len(data.pairs),
              "validation_images": len(validation), "batch_size": 8, "optimizer_steps": 3,
              "policy_and_reference_equal_photo310k_at_init": True,
              "reference_and_dino_unchanged_after_steps": True,
              "all_policy_gradients_finite": True, "validation_excludes_pairs": True,
              "online_vs_cached_feature_max_abs": feature_error, "losses": losses,
              "validation_loss": val_loss["total_loss"].item(),
              "seconds_including_checks": time.monotonic() - started,
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "data_provenance": data.provenance}
    (root / "checkpoints/diffusion/photo_dpo/MODEL_PREFLIGHT.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
