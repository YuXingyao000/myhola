"""严格加载指定的历史 HoLa SVR；仅供独立比较 worker 在启动时调用。

网络和原生 inference 来自固定 Git 源码。构造期间隔离历史包名，返回前
恢复当前进程的 src/shared 模块；不替换当前 diffusion 或修改旧解码行为。
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import importlib
import io
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile

from diffusers import DDPMScheduler
from omegaconf import OmegaConf
import torch


SOURCE_COMMIT = "c2660c00c6a9c8f9f82bd5b9f4d3c774fa38d361"
SOURCE_FILES = (
    "src/__init__.py",
    "src/brepnet/__init__.py",
    "src/brepnet/diffusion_model.py",
    "src/brepnet/model.py",
    "src/brepnet/dataset.py",
    "shared/__init__.py",
    "shared/common_utils.py",
)


def _archive_sources(repo_root: Path, archive_root: str | Path | None) -> tuple[Path, dict]:
    """提取固定的必要源码，不 checkout 工作区；校验复用目录里的每个文件。"""
    payload = subprocess.check_output(
        ["git", "archive", SOURCE_COMMIT, *SOURCE_FILES], cwd=repo_root,
    )
    root = Path(archive_root).resolve() if archive_root else Path(tempfile.mkdtemp(prefix="hola-svr-legacy-"))
    root.mkdir(parents=True, exist_ok=True)
    hashes = {}
    with tarfile.open(fileobj=io.BytesIO(payload)) as archive:
        for member in archive:
            if member.isdir():
                continue
            if member.name not in SOURCE_FILES or not member.isfile():
                raise ValueError(f"Unexpected historical archive member: {member.name}")
            content = archive.extractfile(member).read()
            digest = hashlib.sha256(content).hexdigest()
            target = root / member.name
            if target.exists():
                if target.is_symlink() or target.read_bytes() != content:
                    raise ValueError(f"Historical source cache differs from fixed commit: {target}")
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("xb") as stream:
                    stream.write(content)
            hashes[member.name] = digest
    if set(hashes) != set(SOURCE_FILES):
        raise ValueError("Historical source archive is incomplete.")
    return root, hashes


def _historical_package(name: str) -> bool:
    return name in {"src", "shared"} or name.startswith(("src.", "shared."))


@contextmanager
def _isolated_sources(source_root: Path, repo_root: Path):
    """旧代码的绝对 import 只在此启动期上下文生效；不支持并发调用。"""
    saved_modules = {name: module for name, module in sys.modules.items() if _historical_package(name)}
    saved_path = list(sys.path)
    saved_repr = torch.Tensor.__repr__
    try:
        for name in saved_modules:
            del sys.modules[name]
        # 当前 thirdparty 仅提供旧文件顶层导入的依赖，不替换历史网络源码。
        sys.path[:0] = [str(source_root), str(repo_root)]
        importlib.invalidate_caches()
        yield
    finally:
        for name in list(sys.modules):
            if _historical_package(name):
                del sys.modules[name]
        sys.modules.update(saved_modules)
        sys.path[:] = saved_path
        # 历史 shared.common_utils 在 import 时修改了 Tensor repr。
        torch.Tensor.__repr__ = saved_repr
        importlib.invalidate_caches()


def load_legacy_model(
    checkpoint_path: str | Path,
    *,
    archive_root: str | Path | None = None,
    dino_repo: str | Path | None = None,
) -> tuple[torch.nn.Module, dict]:
    """返回 CPU 模型和来源记录；调用方执行 ``model.to(device).eval()``。

    图像输入为 conditions.imgs [B,1,3,224,224] 和 img_id [B,1]。
    保留原生 inference、latent/geometry 去重及输出字段，不做 schema 修复。
    """
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    repo_root = Path(__file__).resolve().parents[3]
    source_root, source_hashes = _archive_sources(repo_root, archive_root)
    dino_root = Path(dino_repo).resolve() if dino_repo else Path(torch.hub.get_dir()) / "facebookresearch_dinov2_main"
    if not (dino_root / "hubconf.py").is_file():
        raise FileNotFoundError(f"A local DINOv2 source repository is required: {dino_root}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", mmap=True, weights_only=False)
    hparams = checkpoint["hyper_parameters"]
    if OmegaConf.is_config(hparams):
        hparams = OmegaConf.to_container(hparams, resolve=False)
    config = dict(hparams["model"])
    if config["name"] != "Diffusion_condition" or config["condition"] != ["single_img"]:
        raise ValueError("This backend supports only the historical Diffusion_condition single_img model.")
    weights = checkpoint["state_dict"]
    if len(weights) != 1375 or any(not key.startswith("model.") for key in weights):
        raise ValueError("Unexpected checkpoint signature: expected 1375 model.* tensors.")
    weights = {key[len("model."):]: value for key, value in weights.items()}

    # 完整 checkpoint 已带 AE；不重新读取另一份 AE，也不下载 DINO 权重。
    original_ae_path = config["autoencoder_weights"]
    config["autoencoder_weights"] = None
    added_is_aug = "is_aug" not in config
    if added_is_aug:
        config["is_aug"] = hparams["dataset"]["is_aug"]
    original_hub_load = torch.hub.load

    def load_local_dino(repository, model_name, *args, **kwargs):
        if repository != "facebookresearch/dinov2" or model_name != "dinov2_vitl14_reg" or args or kwargs:
            raise ValueError("Unexpected torch.hub request in historical constructor.")
        return original_hub_load(str(dino_root), model_name, source="local", pretrained=False)

    with _isolated_sources(source_root, repo_root):
        module = importlib.import_module("src.brepnet.diffusion_model")
        try:
            torch.hub.load = load_local_dino
            # 避免先分配/随机初始化一份完整权重，再复制 checkpoint。
            with torch.device("meta"):
                model = module.Diffusion_condition(config)
        finally:
            torch.hub.load = original_hub_load
        result = model.load_state_dict(weights, strict=True, assign=True)

    # Scheduler 非 Module/state_dict；重建同配置的 CPU 常量，避免遗留 meta 张量。
    model.noise_scheduler = DDPMScheduler.from_config(model.noise_scheduler.config)
    if any(tensor.is_meta for tensor in (*model.parameters(), *model.buffers())):
        raise RuntimeError("Historical model still contains unmaterialized meta tensors.")
    model.eval()
    provenance = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "global_step": checkpoint["global_step"], "epoch": checkpoint["epoch"],
        "source_commit": SOURCE_COMMIT, "source_archive": str(source_root),
        "source_sha256": source_hashes,
        "model_class": "src.brepnet.diffusion_model.Diffusion_condition",
        "autoencoder_class": "src.brepnet.model.AutoEncoder_1119_light",
        "original_autoencoder_path": original_ae_path,
        "autoencoder_weight_source": "model.ae_model.* in the complete diffusion checkpoint",
        "dino_repository": str(dino_root), "dino_model": "dinov2_vitl14_reg",
        "dino_weight_source": "model.img_model.* in the complete diffusion checkpoint",
        "constructor_added_is_aug": config["is_aug"] if added_is_aug else None,
        "strict_load": True, "state_key_count": len(weights),
        "missing_keys": list(result.missing_keys), "unexpected_keys": list(result.unexpected_keys),
        "native_inference_preserved": True,
        "input_shapes": {"conditions.imgs": "[B,1,3,224,224]", "conditions.img_id": "[B,1]"},
    }
    return model, provenance
