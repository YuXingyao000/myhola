"""严格恢复旧无条件网络；保留 768 维特征与 256 维零条件拼接。"""

from __future__ import annotations

import fcntl
import importlib
import importlib.util
from pathlib import Path
import tempfile

from diffusers import DDPMScheduler
from omegaconf import OmegaConf
import torch


ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parents[2]
SHARED_RUNTIME = ROOT.parent / "svr_comparison_20261017/legacy_runtime.py"
SOURCE_COMMIT = "c2660c00c6a9c8f9f82bd5b9f4d3c774fa38d361"
EXPECTED_STATE_KEYS = 1018  # 308 denoiser + 6 unused validity head + 704 AE.


def _shared_helpers():
    """复用既有 Git 快照与模块隔离，不调用其有条件模型加载器。"""
    spec = importlib.util.spec_from_file_location("hola_svr_source_helpers", SHARED_RUNTIME)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load source isolation helpers: {SHARED_RUNTIME}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if module.SOURCE_COMMIT != SOURCE_COMMIT:
        raise ValueError("The shared historical source revision changed.")
    return module


def load_legacy_model(
    checkpoint_path: str | Path,
    *,
    archive_root: str | Path | None = None,
    reference_ae_checkpoint: str | Path | None = None,
) -> tuple[torch.nn.Module, dict]:
    """完整 strict 加载到 CPU；原生 inference 不需要图片或外部 AE。"""
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    before = checkpoint_path.stat()
    helpers = _shared_helpers()
    archive = Path(archive_root) if archive_root else Path(tempfile.mkdtemp(prefix="hola-uncond-legacy-"))
    archive = archive.resolve()
    archive.parent.mkdir(parents=True, exist_ok=True)
    # 多 GPU worker 可以安全共享同一个只读源码快照。
    with archive.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        source_root, source_hashes = helpers._archive_sources(REPOSITORY, archive)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", mmap=True, weights_only=False)
    hparams = checkpoint["hyper_parameters"]
    if OmegaConf.is_config(hparams):
        hparams = OmegaConf.to_container(hparams, resolve=False)
    saved_config = dict(hparams["model"])
    expected = {
        "name": "Diffusion_condition", "condition": None, "diffusion_latent": 768,
        "diffusion_type": "epsilon", "loss": "l2", "beta_schedule": "linear",
        "beta_start": 0.0001, "beta_end": 0.02, "variance_type": "fixed_small",
        "num_max_faces": 30, "pad_method": "random", "addition_tag": False,
        "autoencoder": "AutoEncoder_1119_light", "train_decoder": False,
    }
    mismatches = {key: {"expected": value, "actual": saved_config.get(key)}
                  for key, value in expected.items() if saved_config.get(key) != value}
    if mismatches:
        raise ValueError(f"Checkpoint differs from the frozen unconditional protocol: {mismatches}")
    if checkpoint["global_step"] != 1_000_000 or checkpoint["epoch"] != 324:
        raise ValueError("Expected the audited 1,000,000-step / epoch-324 checkpoint.")
    state = checkpoint["state_dict"]
    if len(state) != EXPECTED_STATE_KEYS or any(not key.startswith("model.") for key in state):
        raise ValueError(f"Expected all {EXPECTED_STATE_KEYS} model.* tensors.")
    weights = {key.removeprefix("model."): value for key, value in state.items()}

    # 新一些的旧构造器用 `in condition`；空列表与历史 null 均表示无模态。
    config = {**saved_config, "condition": [], "autoencoder_weights": None}
    added_is_aug = "is_aug" not in config
    if added_is_aug:
        config["is_aug"] = hparams["dataset"]["is_aug"]
    original_hub_load = torch.hub.load

    def reject_hub_load(*args, **kwargs):
        raise RuntimeError("Unconditional restoration must not construct or download DINO.")

    with helpers._isolated_sources(source_root, REPOSITORY):
        module = importlib.import_module("src.brepnet.diffusion_model")
        try:
            torch.hub.load = reject_hub_load
            with torch.device("meta"):
                model = module.Diffusion_condition(config)
        finally:
            torch.hub.load = original_hub_load
        loaded = model.load_state_dict(weights, strict=True, assign=True)

    model.noise_scheduler = DDPMScheduler.from_config(model.noise_scheduler.config)
    if any(tensor.is_meta for tensor in (*model.parameters(), *model.buffers())):
        raise RuntimeError("The strict load left meta tensors behind.")
    if any((model.with_img, model.with_pc, model.with_txt)) or hasattr(model, "img_model"):
        raise RuntimeError("An image/point/text condition was unexpectedly constructed.")
    if (model.dim_input, model.dim_latent, model.dim_condition, model.dim_total) != (32, 768, 256, 1024):
        raise RuntimeError("The restored latent/zero-condition dimensions differ.")
    if len(model.net1.layers) != 24 or model.net1.layers[0].self_attn.num_heads != 16:
        raise RuntimeError("The restored Transformer dimensions differ.")
    model.eval().requires_grad_(False)
    provenance = {
        "checkpoint": str(checkpoint_path), "checkpoint_bytes": before.st_size,
        "global_step": int(checkpoint["global_step"]), "epoch": int(checkpoint["epoch"]),
        "source_commit": SOURCE_COMMIT, "source_archive": str(source_root),
        "source_sha256": source_hashes,
        "model_class": "src.brepnet.diffusion_model.Diffusion_condition",
        "autoencoder_class": "src.brepnet.model.AutoEncoder_1119_light",
        "autoencoder_weight_source": "complete checkpoint model.ae_model.*",
        "saved_model_config": saved_config,
        "constructor_adaptations": {
            "condition": {"saved": None, "constructor": []},
            "autoencoder_weights": {"saved": saved_config["autoencoder_weights"], "constructor": None},
            "added_is_aug": config["is_aug"] if added_is_aug else None,
        },
        "strict_load": True, "state_key_count": len(state),
        "missing_keys": list(loaded.missing_keys), "unexpected_keys": list(loaded.unexpected_keys),
        "native_inference_preserved": True, "conditions_constructed": [],
        "architecture": {"latent_dim": 32, "input_feature_dim": 768, "zero_condition_dim": 256,
                         "transformer_dim": 1024, "layers": 24, "heads": 16, "ffn_dim": 2048},
        "scheduler": {key: getattr(model.noise_scheduler.config, key)
                      for key in ("num_train_timesteps", "beta_schedule", "prediction_type",
                                  "beta_start", "beta_end", "variance_type", "clip_sample")},
    }
    if reference_ae_checkpoint is not None:
        reference = torch.load(reference_ae_checkpoint, map_location="cpu", mmap=True, weights_only=False)
        reference_state = {key.removeprefix("model."): value for key, value in reference["state_dict"].items()}
        ae_state = {key.removeprefix("ae_model."): value for key, value in weights.items()
                    if key.startswith("ae_model.")}
        equal = (ae_state.keys() == reference_state.keys()
                 and all(torch.equal(value, reference_state[key]) for key, value in ae_state.items()))
        if not equal or len(ae_state) != 704:
            raise ValueError("The embedded AE differs from the audited current 1119 AE.")
        provenance["reference_ae_equality"] = {
            "path": str(Path(reference_ae_checkpoint).resolve()), "tensor_count": 704,
            "all_keys_and_values_equal": True,
        }
    after = checkpoint_path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError("Checkpoint changed during restoration.")
    return model, provenance


def validate_cpu_forward(model: torch.nn.Module) -> dict:
    """验证 None 与显式零条件同值；不采样、不占用 GPU。"""
    if next(model.parameters()).device.type != "cpu":
        raise ValueError("CPU validation requires a CPU model.")
    with torch.random.fork_rng(devices=[]), torch.inference_mode():
        torch.manual_seed(42)
        latent = torch.randn(1, 30, 32)
        timestep = torch.tensor([700], dtype=torch.long)
        absent = model.diffuse(latent, timestep, v_condition=None)
        explicit = model.diffuse(latent, timestep, v_condition=torch.zeros(1, 1, 256))
    if absent.shape != (1, 30, 32) or not torch.isfinite(absent).all() or not torch.equal(absent, explicit):
        raise AssertionError("None/zero-condition forward validation failed.")
    return {"device": "cpu", "output_shape": list(absent.shape), "dtype": str(absent.dtype),
            "finite": True, "none_equals_explicit_zero": True, "max_abs_difference": 0.0}
