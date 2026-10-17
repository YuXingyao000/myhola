"""恢复历史前端 cross-attention + CLIP-style alignment 的完整 checkpoint。

仅隔离加载固定 Git 源码；保留历史 inference、原生 AE 和去重行为。
历史训练 split 与当前主线不同，strict 加载不等于实验协议可直接等同。
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import importlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
from types import SimpleNamespace

from diffusers import DDPMScheduler
from omegaconf import OmegaConf
import torch
from transformers import Dinov2Config, Dinov2Model


SOURCE_COMMIT = "be94bbea8a6eadc49ef7539e2ea0c13525b71a72"
SOURCE_FILES = (
    "src/__init__.py", "src/brepnet/__init__.py",
    "src/brepnet/diffusion_model.py", "src/brepnet/model.py",
    "src/brepnet/dataset.py", "shared/__init__.py", "shared/common_utils.py",
)


def _archive_sources(repository: Path, destination: str | Path | None):
    """只提取固定源码，不 checkout 或改动当前工作区。"""
    payload = subprocess.check_output(
        ["git", "archive", SOURCE_COMMIT, *SOURCE_FILES], cwd=repository,
    )
    root = Path(destination).resolve() if destination else Path(tempfile.mkdtemp(prefix="hola-svr-alignment-"))
    root.mkdir(parents=True, exist_ok=True)
    hashes = {}
    with tarfile.open(fileobj=io.BytesIO(payload)) as archive:
        for member in archive:
            if member.isdir():
                continue
            if member.name not in SOURCE_FILES or not member.isfile():
                raise ValueError(f"Unexpected historical source: {member.name}")
            content = archive.extractfile(member).read()
            target = root / member.name
            if target.exists():
                if target.is_symlink() or target.read_bytes() != content:
                    raise ValueError(f"Historical source cache differs: {target}")
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("xb") as stream:
                    stream.write(content)
            hashes[member.name] = hashlib.sha256(content).hexdigest()
    if set(hashes) != set(SOURCE_FILES):
        raise ValueError("Incomplete historical source archive.")
    return root, hashes


def _historical_package(name: str) -> bool:
    return name in {"src", "shared"} or name.startswith(("src.", "shared."))


@contextmanager
def _isolated_sources(source_root: Path, repository: Path):
    """仅在独立 worker 启动期使用；不支持同进程并发构造。"""
    modules = {name: module for name, module in sys.modules.items() if _historical_package(name)}
    search_path = list(sys.path)
    tensor_repr = torch.Tensor.__repr__
    hf_endpoint = os.environ.get("HF_ENDPOINT")
    try:
        for name in modules:
            del sys.modules[name]
        sys.path[:0] = [str(source_root), str(repository)]
        importlib.invalidate_caches()
        yield
    finally:
        for name in list(sys.modules):
            if _historical_package(name):
                del sys.modules[name]
        sys.modules.update(modules)
        sys.path[:] = search_path
        torch.Tensor.__repr__ = tensor_repr
        if hf_endpoint is None:
            os.environ.pop("HF_ENDPOINT", None)
        else:
            os.environ["HF_ENDPOINT"] = hf_endpoint
        importlib.invalidate_caches()


def load_alignment_model(
    checkpoint_path: str | Path,
    *,
    archive_root: str | Path | None = None,
) -> tuple[torch.nn.Module, dict]:
    """返回 CPU 模型；输入 ``imgs [B,1,3,224,224]``、``img_id [B,1]``。"""
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    repository = Path(__file__).resolve().parents[3]
    source_root, source_hashes = _archive_sources(repository, archive_root)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", mmap=True, weights_only=False)
    hparams = checkpoint["hyper_parameters"]
    if OmegaConf.is_config(hparams):
        hparams = OmegaConf.to_container(hparams, resolve=False)
    config = dict(hparams["model"])
    if config["name"] != "Diffusion_condition" or config["condition"] != ["single_img"]:
        raise ValueError("Expected the historical single_img Diffusion_condition checkpoint.")
    state = checkpoint["state_dict"]
    if len(state) != 1556 or any(not key.startswith("model.") for key in state):
        raise ValueError("Expected 1556 complete model.* tensors for the HF-DINO alignment architecture.")
    weights = {key[len("model."):]: value for key, value in state.items()}

    # 两个冻结网络均来自完整 checkpoint，不另载 AE 或下载视觉权重。
    original_ae_path = config["autoencoder_weights"]
    config["autoencoder_weights"] = None
    dino_config = Dinov2Config.from_pretrained("facebook/dinov2-large", local_files_only=True)

    def construct_dino(name, *args, **kwargs):
        if name != "facebook/dinov2-large" or args or kwargs:
            raise ValueError("Unexpected historical DINO constructor request.")
        return Dinov2Model(dino_config)

    with _isolated_sources(source_root, repository):
        module = importlib.import_module("src.brepnet.diffusion_model")
        module.Dinov2Model = SimpleNamespace(from_pretrained=construct_dino)
        with torch.device("meta"):
            model = module.Diffusion_condition(config)
        result = model.load_state_dict(weights, strict=True, assign=True)

    # Scheduler 不属于 Module，其 meta 常量需要按原配置重建。
    model.noise_scheduler = DDPMScheduler.from_config(model.noise_scheduler.config)
    if any(value.is_meta for value in (*model.parameters(), *model.buffers())):
        raise RuntimeError("Historical model contains unmaterialized meta tensors.")
    model.eval()
    provenance = {
        "checkpoint": str(checkpoint_path), "checkpoint_bytes": checkpoint_path.stat().st_size,
        "global_step": checkpoint["global_step"], "epoch": checkpoint["epoch"],
        "source_commit": SOURCE_COMMIT, "source_archive": str(source_root),
        "source_sha256": source_hashes,
        "model_class": "src.brepnet.diffusion_model.Diffusion_condition",
        "autoencoder_class": "src.brepnet.model.AutoEncoder_1119_light",
        "original_autoencoder_path": original_ae_path,
        "autoencoder_weight_source": "complete checkpoint model.ae_model.*",
        "dino_weight_source": "complete checkpoint model.img_model.*",
        "dino_architecture": "facebook/dinov2-large, local configuration only",
        "dino_attention_implementation": model.img_model.config._attn_implementation,
        "strict_load": True, "state_key_count": len(weights),
        "missing_keys": list(result.missing_keys), "unexpected_keys": list(result.unexpected_keys),
        "native_inference_preserved": True,
        "input_shapes": {"conditions.imgs": "[B,1,3,224,224]", "conditions.img_id": "[B,1]"},
        "saved_experiment_name": hparams["trainer"].get("exp_name"),
        "saved_train_split": hparams["dataset"].get("train_dataset"),
        "saved_validation_split": hparams["dataset"].get("val_dataset"),
        "saved_augmentation": hparams["dataset"].get("is_aug"),
        "alignment": {"kind": "symmetric InfoNCE", "temperature": 0.07, "weight": 1.0},
        "protocol_limit": "历史保存配置的训练/验证为同一小集；SVR像素来源由归档源码支持，未保存的训练期源码修改无法从checkpoint排除。",
    }
    return model, provenance
