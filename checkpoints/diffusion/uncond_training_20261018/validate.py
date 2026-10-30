"""用真实 batch 检查无条件训练接口，不替代生成质量评估。"""

import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[2]))

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from lightning_fabric import seed_everything
from lightning_fabric.utilities.apply_func import move_data_to_device
from torch.utils.data import DataLoader

from src.brepnet.dataset import Diffusion_geometry_dataset
from src.brepnet.models import build_model


def main():
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    seed_everything(42, workers=True)
    with initialize_config_dir(config_dir=str(ROOT.parents[2] / "configs"), version_base="1.3"):
        cfg = compose(config_name="train_diffusion_uncond")

    train_data = Diffusion_geometry_dataset("training", cfg.dataset)
    val_data = Diffusion_geometry_dataset("validation", cfg.dataset)
    assert len(train_data) == 46627 * 100 and len(val_data) == 3505
    loader = DataLoader(train_data, batch_size=cfg.trainer.batch_size, shuffle=True,
                        num_workers=4, collate_fn=train_data.collate_fn)
    batch = move_data_to_device(next(iter(loader)), torch.device("cuda:0"))
    assert batch["conditions"] == {}
    model = build_model(cfg.model, cfg.condition).cuda().train()
    assert model.condition_encoder is None and len(model.denoiser.cross_attention_layers) == 0
    assert not model.autoencoder.training
    assert all(not parameter.requires_grad for parameter in model.autoencoder.parameters())

    reference = torch.load(cfg.model.autoencoder.checkpoint, map_location="cpu", weights_only=False)
    ae_state = model.autoencoder.state_dict()
    assert len(ae_state) == 704
    assert all(torch.equal(ae_state[key].cpu(), value) for name, value in reference["state_dict"].items()
               for key in [name.removeprefix("model.")])
    del reference

    shapes = {}
    def capture_latents(module, inputs, result):
        shapes["latent"] = list(result.values.shape)
        assert torch.isfinite(result.values).all()
    hook = model.latent_provider.register_forward_hook(capture_latents)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=cfg.trainer.learning_rate)
    before = model.denoiser.input_projection[0].weight.detach().clone()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        losses = model(batch)
    loss = losses["total_loss"]
    assert torch.isfinite(loss)
    loss.backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in parameters)
    assert all(parameter.grad is None for parameter in model.autoencoder.parameters())
    grad_norm = torch.nn.utils.clip_grad_norm_(parameters, cfg.trainer.gradient_clip_val)
    optimizer.step()
    torch.cuda.synchronize()
    step_seconds = time.perf_counter() - started
    assert not torch.equal(before, model.denoiser.input_projection[0].weight)
    hook.remove()
    assert shapes["latent"] == [192, 30, 32]
    peak_gib = torch.cuda.max_memory_allocated() / 1024 ** 3

    # 检查完整无条件DDPM→decoder可调用；随机主干不用于质量判断。
    del optimizer, batch
    model.zero_grad(set_to_none=True)
    model.eval()
    with torch.inference_mode():
        prediction = model.inference(1, torch.device("cuda:0"), v_data=None, v_log=False)[0]
    for key in ("pred_face", "pred_edge", "pred_edge_face_connectivity"):
        value = prediction[key]
        assert torch.isfinite(value).all() if torch.is_tensor(value) else np.isfinite(value).all()

    report = {"status": "passed", "train_cads": 46627, "validation_cads": len(val_data),
              "batch_size": 192, "latent_shape": shapes["latent"], "loss": float(loss.detach()),
              "gradient_norm_before_clip": float(grad_norm), "step_seconds": step_seconds,
              "peak_allocated_gib": peak_gib, "ae_tensors_equal": 704,
              "ae_frozen_and_eval": True, "all_trainable_gradients_finite": True,
              "full_unconditional_sampling_finite": True,
              "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)}
    (ROOT / "VALIDATION.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
