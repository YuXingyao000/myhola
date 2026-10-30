"""Diffusion package contracts; no pretrained model download is required.

Optional real cache check:
    HOLABREP_DIFFUSION_CACHE_SAMPLE=/path/to/features.npy python -m pytest ...

The modular baseline parity tests are preserved on codex/diffusion-modular-baseline.
These current tests verify interfaces, not reconstruction quality.
"""

from __future__ import annotations

import importlib
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn


class TinyDino(nn.Module):
    """Keep the image/token interface while avoiding pretrained weights."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.linspace(0.1, 1.0, 1024))
        self.config = SimpleNamespace(hidden_size=1024)

    def forward(self, images: torch.Tensor) -> SimpleNamespace:
        image_mean = images.mean(dim=(1, 2, 3))[:, None, None]
        offsets = torch.arange(5, dtype=images.dtype, device=images.device)[None, :, None]
        return SimpleNamespace(last_hidden_state=(image_mean + offsets) * self.scale)


class TinyAutoencoder(nn.Module):
    """Expose frozen parameters and record the exact sampled decoder input."""

    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(latent_dim, latent_dim)
        self.requires_grad_(False)

    def encode(self, batch: dict, *_args) -> dict:
        return {"face_features": self.projection(batch["face_features"])}

    def sample(self, features: torch.Tensor, v_is_test: bool):
        return features, None, None

    def decode_latents(self, latents: torch.Tensor) -> dict:
        return {"face_latents": latents.detach().clone()}


@pytest.fixture(autouse=True)
def cpu_threads():
    # Tiny attention workloads are faster and more reproducible with one thread.
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def tiny_dependencies(monkeypatch):
    from transformers import Dinov2Model
    from src.brepnet.diffusion import model

    monkeypatch.setattr(Dinov2Model, "from_pretrained", lambda *_args, **_kwargs: TinyDino())
    monkeypatch.setattr(model, "build_autoencoder", lambda cfg: TinyAutoencoder(cfg["fixture_dim"]))
    return model


def tiny_config(padding: str = "random", prediction: str = "epsilon", latent_dim: int = 4):
    model_cfg = {
        "name": "Diffusion",
        "stage": "diffusion",
        "loss": "l2",
        "latent": {"dim": latent_dim, "use_cached_latents": True, "use_mean": True},
        "padding": {"type": padding, "max_faces": 4, "valid_loss_weight": 0.01},
        "noise": {
            "prediction_type": prediction,
            "beta_schedule": "squaredcos_cap_v2",
            "beta_start": 0.0001,
            "beta_end": 0.02,
            "variance_type": "fixed_small",
            "num_train_timesteps": 4,
        },
        "denoiser": {
            "hidden_dim": 64,
            "num_layers": 3,
            "nhead_divisor": 32,
            "feedforward_dim": 128,
            "dropout": 0.1,
            "conditioning": "layerwise_cross_attention",
        },
        "topology_bias": {
            "enabled": False,
            "source": "gt_adjacency",
            "mode": "soft_bias",
            "timestep_weight": "linear_high_noise",
            "parameterization": "additive",
            "scale": 2.0,
            "max_multiplier": 2.0,
        },
        "autoencoder": {"fixture_dim": latent_dim},
    }
    condition_cfg = {
        "type": "single_img",
        "output_dim": 80,
        "image": {"backbone": "dinov2", "depth_anything_v2_ckpt": None},
        "point_cloud": {"augment_probability": 0.0, "encoder": "pointnet"},
    }
    return model_cfg, condition_cfg


def tiny_batch(latent_dim: int = 4, max_faces: int = 4) -> dict:
    generator = torch.Generator().manual_seed(17)
    mean = torch.randn(2, max_faces, latent_dim, generator=generator)
    std = torch.rand(2, max_faces, latent_dim, generator=generator) + 0.1
    return {
        "cached_latent_stats": torch.cat((mean, std), dim=-1),
        "face_mask": torch.arange(max_faces)[None] < torch.tensor([max_faces, max_faces - 1])[:, None],
        "conditions": {"imgs": torch.randn(2, 3, 8, 8, generator=generator)},
        "face_adj": torch.ones(2, max_faces, max_faces) - torch.eye(max_faces)[None],
    }


def test_registry_points_to_diffusion_package(tiny_dependencies):
    from src.brepnet.models import _MODEL_LOADERS

    assert _MODEL_LOADERS["Diffusion"]() is tiny_dependencies.Diffusion


@pytest.mark.parametrize("padding", ["random", "zero"])
@pytest.mark.parametrize("prediction", ["epsilon", "sample"])
def test_single_image_forward_backward_and_sampling_contract(tiny_dependencies, padding, prediction):
    cfg, condition_cfg = tiny_config(padding, prediction)
    model = tiny_dependencies.Diffusion(cfg, condition_cfg).train()
    batch = tiny_batch()
    assert model.extract_condition(batch).shape == (2, 1, 5, 80)
    assert not model.condition_encoder.image_encoder.img_model.training

    losses = model(batch)
    expected = {"diffusion_loss", "total_loss", "t"}
    if padding == "zero":
        expected.add("valid_loss")
    assert losses.keys() == expected
    assert losses["t"].shape == (2, 2)
    assert all(torch.isfinite(value).all() for value in losses.values())
    torch.testing.assert_close(losses["total_loss"], sum(value for name, value in losses.items() if name not in {"total_loss", "t"}))
    losses["total_loss"].backward()
    assert model.denoiser.input_projection[0].weight.grad is not None
    for layer in model.denoiser.cross_attention_layers:
        assert layer.attention.q_proj_weight.grad is not None
        assert layer.attention.k_proj_weight.grad is not None
        assert layer.attention.v_proj_weight.grad is not None
    assert model.condition_encoder.image_encoder.projection[0].weight.grad is not None
    assert all(torch.isfinite(param.grad).all() for param in model.parameters() if param.grad is not None)
    assert all(param.grad is None for param in model.autoencoder.parameters())
    assert all(param.grad is None for param in model.condition_encoder.image_encoder.img_model.parameters())

    # Exercise the overridable public denoising hook in the sampler.
    calls = []
    diffuse = model.diffuse

    def record_diffuse(*args, **kwargs):
        calls.append(args[0].shape)
        return diffuse(*args, **kwargs)

    model.diffuse = record_diffuse
    model.eval()
    with torch.no_grad():
        torch.manual_seed(31)
        first = model.inference(2, torch.device("cpu"), {"conditions": batch["conditions"]}, v_log=False)
        torch.manual_seed(31)
        second = model.inference(2, torch.device("cpu"), {"conditions": batch["conditions"]}, v_log=False)
    assert calls == [torch.Size([2, 4, 4])] * 8
    assert len(first) == len(second) == 2
    for a, b in zip(first, second):
        assert a["face_latents"].ndim == 2
        assert a["face_latents"].shape[-1] == 4
        assert torch.isfinite(a["face_latents"]).all()
        torch.testing.assert_close(a["face_latents"], b["face_latents"], rtol=0, atol=0)


@pytest.mark.parametrize("use_mean", [True, False])
def test_cached_latents_are_mean_and_standard_deviation(use_mean):
    from src.brepnet.diffusion.latents import LatentProvider
    from src.brepnet.diffusion.padding import RandomFacePadder

    batch = tiny_batch()
    provider = LatentProvider({"dim": 4, "use_cached_latents": True, "use_mean": use_mean}, RandomFacePadder(4), TinyAutoencoder(4))
    mean, std = batch["cached_latent_stats"].chunk(2, dim=-1)
    torch.manual_seed(47)
    expected = mean if use_mean else mean + std * torch.randn_like(mean)
    torch.manual_seed(47)
    actual = provider(batch)
    torch.testing.assert_close(actual.values, expected, rtol=0, atol=0)
    assert actual.face_mask is batch["face_mask"]


def test_padding_preserves_object_membership_and_duplicate_removal():
    from src.brepnet.diffusion.padding import RandomFacePadder, ZeroFacePadder

    faces = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    counts = torch.tensor([2, 3])
    random_padder = RandomFacePadder(7)
    torch.manual_seed(53)
    packed = random_padder.pack(faces, counts)
    assert packed.values.shape == (2, 7, 4)
    assert packed.face_mask.all()
    for item, source in zip(packed.values, [faces[:2], faces[2:]]):
        assert set(map(tuple, item.tolist())) == set(map(tuple, source.tolist()))
        unique = random_padder.postprocess(item)
        assert set(map(tuple, unique.tolist())) == set(map(tuple, source.tolist()))
    assert random_padder.postprocess(faces[:0]).shape == (0, 4)
    near_duplicates = torch.stack((faces[0], faces[0] + 0.009, faces[0] + 0.011))
    torch.testing.assert_close(random_padder.postprocess(near_duplicates), near_duplicates[[0, 2]])

    zero = ZeroFacePadder(4, 7).pack(faces, counts)
    torch.testing.assert_close(zero.values[zero.face_mask], faces)
    assert not zero.values[~zero.face_mask].any()


@pytest.mark.parametrize("padding", ["random", "zero"])
def test_uncached_latents_use_frozen_vae_and_face_padder(padding):
    from src.brepnet.diffusion.latents import LatentProvider
    from src.brepnet.diffusion.padding import build_face_padder

    faces = torch.randn(5, 4, requires_grad=True)
    padder = build_face_padder({"type": padding, "max_faces": 4}, 4)
    provider = LatentProvider({"dim": 4, "use_cached_latents": False, "use_mean": True}, padder, TinyAutoencoder(4))
    result = provider({"face_features": faces, "face_counts": torch.tensor([2, 3])})
    assert result.values.shape == (2, 4, 4)
    assert result.face_mask.shape == (2, 4)
    assert not result.values.requires_grad


def test_real_cache_sample_shape_loss_and_gradient(tiny_dependencies):
    path = os.environ.get("HOLABREP_DIFFUSION_CACHE_SAMPLE")
    if not path:
        pytest.skip("Set HOLABREP_DIFFUSION_CACHE_SAMPLE to check an explicit real cache file.")
    stats = torch.from_numpy(np.load(path, allow_pickle=False)).float()
    assert stats.ndim == 2 and stats.shape[1] == 64
    assert torch.isfinite(stats).all() and (stats[:, 32:] >= 0).all()
    cfg, condition_cfg = tiny_config(latent_dim=32)
    cfg["padding"]["max_faces"] = stats.shape[0]
    model = tiny_dependencies.Diffusion(cfg, condition_cfg).train()
    batch = tiny_batch(32, stats.shape[0])
    batch["cached_latent_stats"] = stats[None].repeat(2, 1, 1)
    batch["face_mask"].fill_(True)
    loss = model(batch)["total_loss"]
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(model.denoiser.input_projection[0].weight.grad).all()
