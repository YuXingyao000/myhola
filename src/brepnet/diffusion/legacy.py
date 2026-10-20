"""旧 HoLa 单图 checkpoint 的严格加载与保留完整 latent 的采样接口。

网络和原生 inference 来自固定 Git 源码。构造期间隔离历史包名，返回前
恢复当前进程的 src/shared 模块；不替换当前 diffusion 或修改旧解码行为。
历史 AE 的几何去重不缩减 pred_face 数组；映射显式保留，不修补原生输出。
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
from typing import Sequence

from diffusers import DDPMScheduler
import numpy as np
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


def _construct_legacy(
    config: dict,
    source_cache: str | Path | None,
    dino_repo: str | Path | None,
) -> tuple[torch.nn.Module, dict]:
    repo_root = Path(__file__).resolve().parents[3]
    source_root, source_hashes = _archive_sources(repo_root, source_cache)
    dino_root = Path(dino_repo).resolve() if dino_repo else Path(torch.hub.get_dir()) / "facebookresearch_dinov2_main"
    if not (dino_root / "hubconf.py").is_file():
        raise FileNotFoundError(f"A local DINOv2 source repository is required: {dino_root}")
    original_hub_load = torch.hub.load

    def load_local_dino(repository, model_name, *args, **kwargs):
        if repository != "facebookresearch/dinov2" or model_name != "dinov2_vitl14_reg" or args or kwargs:
            raise ValueError("Unexpected torch.hub request in historical constructor.")
        return original_hub_load(str(dino_root), model_name, source="local", pretrained=False)

    with _isolated_sources(source_root, repo_root):
        module = importlib.import_module("src.brepnet.diffusion_model")
        try:
            torch.hub.load = load_local_dino
            # 完整模型约 2.7 GB；避免先随机初始化再加载一份相同大小的权重。
            with torch.device("meta"):
                model = module.Diffusion_condition(config)
        finally:
            torch.hub.load = original_hub_load
    model.noise_scheduler = DDPMScheduler.from_config(model.noise_scheduler.config)
    return model, {
        "source_commit": SOURCE_COMMIT, "source_archive": str(source_root),
        "source_sha256": source_hashes, "dino_repository": str(dino_root),
    }


def build_legacy_svr(
    config: dict,
    device: str | torch.device = "cpu",
    *,
    source_cache: str | Path | None = None,
    dino_repo: str | Path | None = None,
    with_autoencoder: bool = False,
) -> tuple[torch.nn.Module, dict]:
    """仅构造恢复 checkpoint 所需的结构；返回权重未初始化，必须严格加载。

    无需原始 diffusion checkpoint。训练恢复通常省略 AE，但保留同构 DINO，
    随后用完整训练 state_dict 恢复 policy、reference 与冻结图像编码器。
    """
    config = dict(config)
    config["autoencoder_weights"] = None
    model, provenance = _construct_legacy(config, source_cache, dino_repo)
    if not with_autoencoder:
        del model.ae_model
    model.to_empty(device=device).eval()
    provenance.update(model_config=config, weights_initialized=False, with_autoencoder=with_autoencoder)
    return model, provenance


def load_legacy_svr(
    checkpoint: str | Path,
    device: str | torch.device = "cpu",
    *,
    source_cache: str | Path | None = None,
    dino_repo: str | Path | None = None,
) -> tuple[torch.nn.Module, dict]:
    """严格加载完整旧 SVR；DINO、AE 权重只来自此 checkpoint。

    ``diffuse(x, t, condition)`` 可用于训练；本函数不冻结去噪器和 img_fc。
    原生 ``extract_condition`` 的输入是 imgs [B,1,3,224,224]。
    仅在进程启动时加载；历史 import 和 torch.hub 替换不是线程安全的。
    """
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", mmap=True, weights_only=False)
    hparams = checkpoint["hyper_parameters"]
    if OmegaConf.is_config(hparams):
        hparams = OmegaConf.to_container(hparams, resolve=False)
    config = dict(hparams["model"])
    if config["name"] != "Diffusion_condition" or config["condition"] != ["single_img"]:
        raise ValueError("This backend supports only the historical Diffusion_condition single_img model.")
    contract = {
        "diffusion_latent": 768, "num_max_faces": 30, "pad_method": "random",
        "addition_tag": False, "diffusion_type": "epsilon", "loss": "l2",
        "autoencoder": "AutoEncoder_1119_light", "train_decoder": False,
    }
    if any(config[key] != value for key, value in contract.items()):
        raise ValueError(f"The historical SVR checkpoint must satisfy {contract}.")
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
    model, source_provenance = _construct_legacy(config, source_cache, dino_repo)
    result = model.load_state_dict(weights, strict=True, assign=True)
    if any(tensor.is_meta for tensor in (*model.parameters(), *model.buffers())):
        raise RuntimeError("Historical model still contains unmaterialized meta tensors.")
    model.to(device).eval()
    provenance = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "global_step": checkpoint["global_step"], "epoch": checkpoint["epoch"],
        **source_provenance,
        "model_class": "src.brepnet.diffusion_model.Diffusion_condition",
        "autoencoder_class": "src.brepnet.model.AutoEncoder_1119_light",
        "original_autoencoder_path": original_ae_path,
        "autoencoder_weight_source": "model.ae_model.* in the complete diffusion checkpoint",
        "dino_model": "dinov2_vitl14_reg",
        "dino_weight_source": "model.img_model.* in the complete diffusion checkpoint",
        "constructor_added_is_aug": config["is_aug"] if added_is_aug else None,
        "strict_load": True, "state_key_count": len(weights),
        "missing_keys": list(result.missing_keys), "unexpected_keys": list(result.unexpected_keys),
        "native_inference_preserved": True,
        "model_config": config,
        "sampling": {
            "initial_noise": "CPU torch.randn then transfer to model device",
            "step_noise": "DDPMScheduler default generator on model device",
            "candidate_batch_size": 1,
            "steps": len(model.noise_scheduler.timesteps),
            "prediction_type": model.noise_scheduler.config.prediction_type,
        },
        "input_shapes": {"conditions.imgs": "[B,1,3,224,224]", "conditions.img_id": "[B,1]"},
    }
    return model, provenance


@torch.no_grad()
def encode_original_gt(model: torch.nn.Module, gt_npz: str | Path) -> np.ndarray:
    """原版 AE 以 FP32 编码 identity GT，返回未 padding 的 [N,64]=mean/std。

    输入面、半边及 face order 直接来自原始 data.npz；去掉 self-face
    connectivity 的规则与历史 VAE dataset 相同，不做旋转或图像条件编码。
    """
    autoencoder = model.ae_model
    if autoencoder.training or next(autoencoder.parameters()).dtype != torch.float32:
        raise ValueError("GT encoding requires the original AE in eval mode and float32.")
    device = next(autoencoder.parameters()).device
    with np.load(gt_npz) as arrays:
        faces = torch.as_tensor(arrays["sample_points_faces"], dtype=torch.float32, device=device)
        edges = torch.as_tensor(arrays["sample_points_lines"], dtype=torch.float32, device=device)
        connectivity = torch.as_tensor(arrays["edge_face_connectivity"], dtype=torch.long, device=device)
    connectivity = connectivity[connectivity[:, 1] != connectivity[:, 2]]
    batch = {
        "face_points": faces,
        "edge_points": edges,
        "edge_face_connectivity": connectivity,
        "attn_mask": torch.zeros((len(faces), len(faces)), dtype=torch.bool, device=device),
        "num_face_record": torch.tensor([len(faces)], dtype=torch.long, device=device),
    }
    with torch.autocast(device_type=device.type, enabled=False):
        encoded = autoencoder.encode(batch, True)["face_features"]
        # 原 sample() 即使 use_mean 也抽噪声；直接保留同一 posterior 公式，
        # 避免 GT 缓存写入消耗候选采样的随机状态。
        posterior = autoencoder.gaussian_proj(encoded).reshape(-1, autoencoder.df, 2)
        mean = posterior[:, :, 0]
        std = torch.exp(0.5 * posterior[:, :, 1])
        statistics = torch.cat((mean, std), dim=1)
    return statistics.cpu().numpy()


def _greedy_mapping(distance: torch.Tensor, threshold: float) -> tuple[torch.Tensor, torch.Tensor]:
    """复用旧版按顺序保留首个不重复面的规则，同时记录所有输入的归属。"""
    kept: list[int] = []
    assignment: list[int] = []
    # 30 个面的小矩阵一次转 CPU，避免逐元素 GPU 同步。
    for row in distance.detach().cpu():
        duplicate = next((index for index, previous in enumerate(kept) if row[previous] < threshold), None)
        if duplicate is None:
            assignment.append(len(kept))
            kept.append(len(assignment) - 1)
        else:
            assignment.append(duplicate)
    return torch.tensor(kept, dtype=torch.long), torch.tensor(assignment, dtype=torch.long)


@torch.no_grad()
def decode_candidate(model: torch.nn.Module, latent: torch.Tensor) -> dict:
    """以旧版两级去重解码一组 [30,32]，保留原生 raw 字段及面索引映射。"""
    if latent.shape != (30, 32):
        raise ValueError(f"Expected one [30,32] latent, received {tuple(latent.shape)}.")
    distance = (latent[:, None] - latent[None, :]).abs().mean(dim=-1)
    keep, slots_to_face = _greedy_mapping(distance, 1e-2)
    face_z = latent[keep.to(latent.device)]

    # 捕获原生 decode 已经算出的曲面，避免另一次 AE 前向改变数值。
    decoded: dict[str, torch.Tensor] = {}
    handles = [
        model.ae_model.face_points_decoder.register_forward_hook(
            lambda module, inputs, output: decoded.__setitem__("local", output)),
        model.ae_model.face_center_scale_decoder.register_forward_hook(
            lambda module, inputs, output: decoded.__setitem__("bbox", output)),
    ]
    try:
        raw = model.ae_model.inference(face_z)
    finally:
        for handle in handles:
            handle.remove()
    denormalize = model.ae_model.decode.__func__.__globals__["denormalize_coord1112"]
    surface = denormalize(decoded["local"], decoded["bbox"])
    # 原 AE 的去重距离同时含 XYZ 和 normal 六通道，严格保留其语义。
    geometry_distance = torch.sqrt(
        ((surface[:, None] - surface[None, :]) ** 2).sum(dim=-1),
    ).mean(dim=(-1, -2))
    geometry_keep, face_to_geometry = _greedy_mapping(geometry_distance, 1e-3)
    return {
        "latent": latent.detach().float().cpu(),
        "raw": {
            key: value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
            for key, value in raw.items()
        },
        "latent_keep_indices": keep,
        "slot_to_face": slots_to_face,
        "geometry_keep_indices": geometry_keep,
        "face_to_geometry": face_to_geometry,
    }


@torch.no_grad()
def sample_candidates(
    model: torch.nn.Module,
    images: torch.Tensor,
    seeds: Sequence[int],
) -> list[dict]:
    """逐候选采样，保持旧版 batch=1 的 RNG 与浮点语义。

    ``images`` 是已完成旧版 resize/ImageNet normalize 的 [B,3,224,224]。
    每张图对应一个 seed；生成 K 个候选时显式重复图片和提供 K 个 seed。
    返回的 latent 未去重，可直接用于偏好训练；raw 完全沿用原 AE 输出。
    """
    if images.ndim != 4 or images.shape[1:] != (3, 224, 224):
        raise ValueError("images must have shape [B,3,224,224].")
    if len(images) != len(seeds):
        raise ValueError("Each image must have one candidate seed.")
    if model.training:
        raise ValueError("Candidate generation requires model.eval().")
    if len(model.noise_scheduler.timesteps) != 1000:
        raise ValueError("Candidate generation requires the native 1000-step DDPM schedule.")
    device = next(model.parameters()).device
    cuda_devices = [device.index] if device.type == "cuda" else []
    results = []
    for image, seed in zip(images, seeds):
        # 保存调用方 RNG；初始噪声原生在 CPU 采样，step noise 则在 model device。
        with torch.random.fork_rng(devices=cuda_devices):
            torch.random.default_generator.manual_seed(int(seed))
            if device.type == "cuda":
                torch.cuda.default_generators[device.index].manual_seed(int(seed))
            latent = torch.randn((1, model.num_max_faces, model.dim_input)).to(device)
            condition = model.extract_condition({
                "conditions": {
                    "imgs": image.to(device)[None, None],
                    "img_id": torch.zeros((1, 1), dtype=torch.long, device=device),
                },
            })
            for timestep in model.noise_scheduler.timesteps:
                prediction = model.diffuse(latent, timestep.reshape(-1).to(device), condition)
                latent = model.noise_scheduler.step(prediction, timestep, latent).prev_sample
            result = decode_candidate(model, latent[0])
            result["seed"] = int(seed)
            results.append(result)
    return results
