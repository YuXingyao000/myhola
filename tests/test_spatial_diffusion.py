"""空间条件的集合语义、梯度、冻结、恢复与数据隔离合同。"""

from copy import deepcopy
import hashlib
import json

import numpy as np
import pytest
import torch
from torch import nn
from omegaconf import OmegaConf

from src.models.diffusion.spatial_blocks import SpatialMemoryDenoiser
from src.models.diffusion.spatial_conditioning import SpatialImageMemory
from src.models.diffusion.spatial_model import SpatialPhotoDiffusion
from src.datasets.spatial_photo import SpatialPhotoDataset, spatial_data_provenance


@pytest.fixture
def config():
    cfg = OmegaConf.to_container(OmegaConf.load("configs/train_diffusion_spatial_photo.yaml").model)
    cfg["spatial"].update(width=16, grid_side=2, layers=2, heads=2, ff_width=32, dropout=0.)
    cfg["denoiser"].update(width=32, layers=2, heads=4, ff_width=64, dropout=0.)
    return cfg


def test_face_permutation_memory_and_time(config):
    torch.manual_seed(1)
    encoder = SpatialImageMemory(config["spatial"], image_width=24)
    denoiser = SpatialMemoryDenoiser(config["denoiser"], 16).eval()
    tokens = torch.randn(2, 257, 24)
    memory = encoder(tokens)
    assert memory.shape == (2, 8 + 256, 16)
    original = memory.clone()
    latent, time = torch.randn(2, 30, 32), torch.tensor([23, 750])
    permutation = torch.randperm(30)
    output = denoiser(latent, time, memory)
    torch.testing.assert_close(denoiser(latent[:, permutation], time, memory), output[:, permutation],
                               atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(memory, original, rtol=0, atol=0)
    assert not torch.allclose(output, denoiser(latent, time + 20, memory))
    assert not torch.allclose(output, denoiser(latent, time, encoder(tokens.flip(0))))
    # 坐标确实进入图像 lifting，而不只是旁存一个未使用的 metadata buffer。
    with torch.no_grad():
        encoder.coordinates.mul_(-1)
    assert not torch.allclose(original[:, :8], encoder(tokens)[:, :8])


def test_first_backward_reaches_spatial_path_and_all_blocks(config):
    torch.manual_seed(2)
    encoder = SpatialImageMemory(config["spatial"], image_width=24).train()
    denoiser = SpatialMemoryDenoiser(config["denoiser"], 16).train()
    output = denoiser(torch.randn(2, 30, 32), torch.tensor([17, 999]), encoder(torch.randn(2, 257, 24)))
    output.square().mean().backward()
    for name, parameter in list(encoder.named_parameters()) + list(denoiser.named_parameters()):
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    assert encoder.coordinate_projection[0].weight.grad.abs().sum() > 0
    for block in denoiser.blocks:
        assert block.cross_attention.attention.q_proj_weight.grad.abs().sum() > 0


class FakeDINO(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1024))
        self.calls = 0

    def forward_features(self, image):
        self.calls += 1
        cls = image.mean((1, 2, 3))[:, None] * self.weight[None]
        return {"x_norm_clstoken": cls, "x_norm_patchtokens": cls[:, None].expand(-1, 256, -1)}


@pytest.fixture
def model(config, monkeypatch):
    legacy = nn.Module()
    legacy.img_model, legacy.ae_model = FakeDINO(), nn.Linear(32, 3)
    monkeypatch.setattr("src.models.diffusion.spatial_model.load_legacy_svr",
                        lambda *args, **kwargs: (legacy, {"model_config": {"test": True}}))
    monkeypatch.setattr("src.datasets.preference.sha256_file",
                        lambda path: config["frozen_source_sha256"])
    return SpatialPhotoDiffusion(config)


def test_model_freezing_sampling_once_and_reproducibility(model):
    image = torch.randn(2, 3, 224, 224)
    model.train()
    assert not model.image_encoder.training and not model.ae_model.training
    loss = model({"gt_image": image, "gt_latent": torch.randn(2, 30, 32)})["total_loss"]
    loss.backward()
    assert all(p.grad is None and not p.requires_grad for p in model.image_encoder.parameters())
    assert all(p.grad is None and not p.requires_grad for p in model.ae_model.parameters())
    model.eval()
    before = model.image_encoder.calls
    first = model.sample_latents(image, generator=torch.Generator().manual_seed(42), num_steps=3)
    assert model.image_encoder.calls == before + 1
    second = model.sample_latents(image, generator=torch.Generator().manual_seed(42), num_steps=3)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert first.shape == (2, 30, 32) and torch.isfinite(first).all()


def test_strict_checkpoint_detects_semantics_not_only_shapes(model):
    state = deepcopy(model.state_dict())
    model.load_state_dict(state)
    with pytest.raises(ValueError, match="strict=True"):
        model.load_state_dict(state, strict=False)
    # Lightning 从父 module 递归加载时也必须触发语义检查。
    parent = nn.Module()
    parent.model = model
    parent_state = deepcopy(parent.state_dict())
    parent_state["model._extra_state"]["spatial"]["extent"] = 2.
    with pytest.raises(ValueError, match="architecture/config"):
        parent.load_state_dict(parent_state)
    state.pop("denoiser.input_projection.0.weight")
    with pytest.raises(RuntimeError, match="Missing key"):
        model.load_state_dict(state)


@pytest.fixture
def data_config(tmp_path):
    root = tmp_path / "generation"
    (root / "metadata").mkdir(parents=True)
    images = tmp_path / "images"
    for identity in ("a", "b"):
        folder = root / "gt_cache" / f"{identity}_0"
        folder.mkdir(parents=True)
        stats = np.zeros((3, 64), dtype=np.float32)
        stats[:, :32] = np.arange(3)[:, None]
        np.save(folder / "features.npy", stats)
        digest = hashlib.sha256((folder / "features.npy").read_bytes()).hexdigest()
        (folder / "source.json").write_text(json.dumps({"precision": "float32", "layout": "mean32,std32", "gt_npz_sha256": "gt",
                                                       "output_sha256": {"features.npy": digest}}))
        (images / identity).mkdir(parents=True)
        np.savez(images / identity / "real_photo.npz", flux=np.zeros((512, 512, 3), dtype=np.uint8))
        sources = root / "candidates" / ("train" if identity == "a" else "validation") / identity
        sources.mkdir(parents=True)
        photo_path = images / identity / "real_photo.npz"
        (sources / "sources.json").write_text(json.dumps({
            str(photo_path): hashlib.sha256(photo_path.read_bytes()).hexdigest(),
            str(tmp_path / "raw" / identity / "data.npz"): "gt"}))
    (tmp_path / "train.txt").write_text("a\n")
    (tmp_path / "val.txt").write_text("b\n")
    protocol = {"version": "photo_preference_generation_v1", "splits": {"train": ["a"], "validation": ["b"]},
                "condition_data_root": str(images), "raw_data_root": str(tmp_path / "raw"), "original_checkpoint_sha256": "frozen-source",
                "gt_cache": "original embedded AE / FP32 / mean32+std32 / rotation0",
                "condition": {"filename": "real_photo.npz", "key": "flux", "rotation_id": 0,
                              "shape": [512, 512, 3], "dtype": "uint8"}}
    path = root / "metadata/protocol.json"
    path.write_text(json.dumps(protocol))
    return {"source_protocol": str(path), "source_protocol_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "condition_data_root": str(images), "cached_latent_root": str(root / "gt_cache"),
            "train_dataset": str(tmp_path / "train.txt"), "val_dataset": str(tmp_path / "val.txt"),
            "epoch_scale_factor": 3}


def test_data_reads_raw_photo_and_full_validation_without_preferences(data_config):
    train = SpatialPhotoDataset("training", data_config)
    val = SpatialPhotoDataset("validation", data_config)
    assert len(train) == 3 and len(val) == 1
    item = train[0]
    assert item["gt_image"].shape == (3, 224, 224)
    assert item["gt_latent"].shape == (30, 32)
    assert len(item["gt_latent"].unique(dim=0)) == 3
    assert set(item) == {"v_prefix", "gt_image", "gt_latent"}
    assert val[0]["v_prefix"] == "b"


def test_data_rejects_changed_split_and_cache(data_config):
    from pathlib import Path
    config = dict(data_config, val_dataset=data_config["train_dataset"])
    with pytest.raises(ValueError, match="full, disjoint"):
        spatial_data_provenance(config)
    cache = Path(data_config["cached_latent_root"]) / "a_0/features.npy"
    np.save(cache, np.ones((3, 64), dtype=np.float32))
    with pytest.raises(ValueError, match="provenance mismatch"):
        SpatialPhotoDataset("training", data_config)[0]


def test_spatial_only_memory_and_shape_guards(config):
    config["spatial"]["retain_rgb"] = False
    memory = SpatialImageMemory(config["spatial"], image_width=24)
    assert memory(torch.randn(1, 257, 24)).shape == (1, 8, 16)
    with pytest.raises(ValueError, match="CLS"):
        memory(torch.randn(1, 256, 24))
    denoiser = SpatialMemoryDenoiser(config["denoiser"], 16)
    with pytest.raises(ValueError, match="latent"):
        denoiser(torch.randn(1, 29, 32), torch.tensor([1]), torch.randn(1, 8, 16))
