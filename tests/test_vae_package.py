"""VAE package contracts; these do not establish reconstruction quality."""

from __future__ import annotations

import importlib
import inspect
import pickle

import pytest
import torch
from omegaconf import OmegaConf

from src.models import build_model
from src.models.vae import BaseVAE, build_autoencoder
from tests.test_intersection_vae import tiny_config


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def test_base_requires_the_complete_vae_contract_and_adds_no_state():
    assert inspect.isabstract(BaseVAE)
    assert BaseVAE.__abstractmethods__ == {
        "encode", "sample", "decode", "forward", "decode_latents",
    }
    with pytest.raises(TypeError, match="abstract"):
        BaseVAE()

    class MinimalVAE(BaseVAE):
        def encode(self, data):
            return data

        def sample(self, features):
            return features

        def decode(self, latents):
            return latents

        def forward(self, data):
            return self.decode(self.sample(self.encode(data)))

        def decode_latents(self, latents):
            return {"latents": latents}

    model = MinimalVAE()
    assert list(model.state_dict()) == []
    assert list(model.parameters()) == []
    latent = torch.randn(2, 8)
    assert model.inference(latent)["latents"] is latent


@pytest.mark.parametrize("name,anchors", [
    ("AutoEncoder", ("face_coords.0.weight", "gaussian_proj.0.weight", "classifier.weight")),
    ("AutoEncoder_light", ("face_coords.0.weight", "gaussian_proj.0.weight", "classifier.weight")),
    ("AutoEncoder_light_exp", ("face_coords.0.weight", "gaussian_proj.0.weight", "classifier.weight")),
    ("AutoEncoder_intersection", (
        "encoder.face_coords.0.weight", "posterior.projection.0.weight", "decoder.classifier.weight",
    )),
    ("AutoEncoder_joint", ("face_coords.0.weight", "gaussian_proj.0.weight", "decoder_joint.edge_queries")),
])
def test_factories_keep_concrete_classes_and_checkpoint_names(name, anchors):
    config = tiny_config(
        name=name, joint_hidden_dim=16, joint_num_queries=4,
        joint_num_rounds=1, joint_num_heads=2,
    )
    torch.manual_seed(471)
    first = build_model(OmegaConf.create(config))
    torch.manual_seed(471)
    second = build_autoencoder(config)
    assert isinstance(first, BaseVAE)
    assert type(first) is type(second)
    assert not inspect.isabstract(type(first))
    assert set(anchors) <= first.state_dict().keys()
    assert list(first.state_dict()) == list(second.state_dict())
    for key, expected in first.state_dict().items():
        torch.testing.assert_close(second.state_dict()[key], expected, rtol=0, atol=0)
    assert not any(key.startswith("base.") for key in first.state_dict())
    second.load_state_dict(first.state_dict(), strict=True)
    incomplete = dict(first.state_dict())
    incomplete.pop(anchors[0])
    with pytest.raises(RuntimeError, match="Missing key"):
        second.load_state_dict(incomplete, strict=True)


@pytest.mark.parametrize("module_name,names", [
    ("src.models.vae", (
        "AutoEncoder", "AutoEncoder_light", "AutoEncoder_light_exp", "AutoEncoder_intersection",
        "AutoEncoder_joint", "HolisticEncoder", "FacePosterior", "DirectedTargets",
        "IntersectionLoss", "balanced_binary_cross_entropy", "boundary_costs",
        "joint_boundary_loss", "match_boundaries", "build_autoencoder",
    )),
    ("src.models.blocks", (
        "ResBlockXD", "UVFaceEncoder", "EdgeValueGATv2Conv", "AttnIntersection",
        "CandidatePairs", "PrimitiveGeometry", "DecoderStage", "RefinementOutput",
        "IntersectionDecoder", "IntersectionRefiner", "FacePairConditioning", "UVPathHead",
        "JointBoundaryDecoder", "closest_surface_points", "closest_polyline_points",
    )),
])
def test_old_import_and_pickle_global_paths_still_resolve(module_name, names):
    module = importlib.import_module(module_name)
    for name in names:
        expected = getattr(module, name)
        # Protocol-0 GLOBAL represents an old pickle's module/class lookup.
        restored = pickle.loads(f"c{module_name}\n{name}\n.".encode())
        assert restored is expected
    if module_name.endswith(".vae"):
        assert callable(module._map_directed_targets)


@pytest.mark.parametrize("class_name", ["AutoEncoder_intersection", "AutoEncoder_joint"])
def test_state_decoder_and_public_decode_preserve_latent_gradients(class_name):
    config = tiny_config(
        name=class_name, joint_hidden_dim=16, joint_num_queries=4,
        joint_num_rounds=1, joint_num_heads=2,
    )
    model = build_autoencoder(config).eval()
    latents = torch.randn(5, 8, requires_grad=True)
    counts = torch.tensor([2, 3])
    if class_name == "AutoEncoder_joint":
        decoded = model.decode({"face_latents": latents, "face_counts": counts})
    else:
        decoded = model.decode(latents, counts)
    old_entry = model.decode_state(latents, counts)
    if class_name == "AutoEncoder_joint":
        face_points, edge_points = decoded["face_points"], decoded["curves"]
        torch.testing.assert_close(face_points, old_entry["face_points"], rtol=0, atol=0)
        torch.testing.assert_close(edge_points, old_entry["curves"], rtol=0, atol=0)
    else:
        face_points = decoded.refined.face_geometry.points
        edge_points = decoded.refined.edge_geometry.points
        torch.testing.assert_close(face_points, old_entry.refined.face_geometry.points, rtol=0, atol=0)
        torch.testing.assert_close(edge_points, old_entry.refined.edge_geometry.points, rtol=0, atol=0)
    (face_points.square().mean() + edge_points.square().mean()).backward()
    assert latents.grad is not None and latents.grad.isfinite().all()
    assert latents.grad.abs().sum() > 0
