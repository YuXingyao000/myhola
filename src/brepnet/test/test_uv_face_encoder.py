"""Contracts for the UV face encoder; these do not establish research quality."""

from __future__ import annotations

import pytest
import torch

from src.brepnet.models.blocks import UVFaceEncoder


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def small_encoder(**overrides):
    options = {
        "hidden_dim": 32,
        "embedding_dim": 16,
        "num_layers": 2,
        "nhead": 4,
        "dim_feedforward": 64,
        "dropout": 0.0,
    }
    options.update(overrides)
    return UVFaceEncoder(**options)


def test_default_capacity_and_independent_layer_initialization():
    with torch.device("meta"):
        encoder = UVFaceEncoder()
    assert sum(parameter.numel() for parameter in encoder.parameters()) == 22_219_040
    assert len(encoder.layers) == 6
    assert encoder.layers[0].self_attn.num_heads == 8
    assert encoder.layers[0].linear1.out_features == 2048
    assert encoder.out_channels == 32

    torch.manual_seed(43)
    encoder = small_encoder()
    first = encoder.layers[0].self_attn.in_proj_weight
    second = encoder.layers[1].self_attn.in_proj_weight
    assert first.data_ptr() != second.data_ptr()
    assert not torch.equal(first, second)


def test_all_grid_positions_reach_readout_without_spatial_pooling():
    torch.manual_seed(44)
    encoder = small_encoder()
    layer_shapes = []
    handles = [
        layer.register_forward_hook(
            lambda module, inputs, output: layer_shapes.append((inputs[0].shape, output.shape))
        )
        for layer in encoder.layers
    ]
    readout_shapes = []
    handles.append(encoder.readout_attention.register_forward_pre_hook(
        lambda module, inputs: readout_shapes.append(tuple(value.shape for value in inputs))
    ))
    try:
        result = encoder(torch.randn(3, 6, 16, 16))
    finally:
        for handle in handles:
            handle.remove()
    assert result.shape == (3, 32)
    assert torch.isfinite(result).all()
    assert layer_shapes == [((3, 256, 32), (3, 256, 32))] * 2
    assert readout_shapes == [((3, 1, 32), (3, 256, 32), (3, 256, 32))]


def test_uv_positions_are_deterministic_and_affect_geometric_correspondence():
    torch.manual_seed(45)
    encoder = small_encoder().eval()
    grid = encoder.uv_grid.clone()
    assert grid.shape == (1, 256, 2)
    torch.testing.assert_close(grid[0, 0], torch.tensor([0., 0.]))
    torch.testing.assert_close(grid[0, 15], torch.tensor([0., 1.]))
    torch.testing.assert_close(grid[0, 16], torch.tensor([1 / 15, 0.]))
    torch.testing.assert_close(grid[0, -1], torch.tensor([1., 1.]))
    assert "uv_grid" not in encoder.state_dict()

    faces = torch.randn(2, 6, 16, 16)
    permutation = torch.randperm(256)
    shuffled = faces.flatten(2)[:, :, permutation].reshape_as(faces)
    with torch.no_grad():
        reference = encoder(faces)
        # Moving samples while leaving UV fixed changes the parameterized face.
        wrong_correspondence = encoder(shuffled)
        encoder.uv_grid.copy_(grid[:, permutation])
        # Moving geometry and its UV together only reorders the same tokens.
        same_correspondence = encoder(shuffled)
    assert not torch.allclose(wrong_correspondence, reference, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(same_correspondence, reference, rtol=1e-5, atol=1e-6)


def test_faces_are_independent_and_face_permutations_are_preserved():
    torch.manual_seed(46)
    encoder = small_encoder().eval()
    faces = torch.randn(3, 6, 16, 16)
    with torch.no_grad():
        packed = encoder(faces)
        separate = torch.cat([encoder(face.unsqueeze(0)) for face in faces])
        permutation = torch.tensor([2, 0, 1])
        permuted = encoder(faces[permutation])
        changed = faces.clone()
        changed[1] += 20
        after = encoder(changed)
    torch.testing.assert_close(packed, separate, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(permuted, packed[permutation], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(after[[0, 2]], packed[[0, 2]], rtol=0, atol=0)
    assert not torch.allclose(after[1], packed[1])


def test_geometry_position_query_and_attention_receive_finite_gradients():
    torch.manual_seed(47)
    encoder = small_encoder()
    faces = torch.randn(2, 6, 16, 16, requires_grad=True)
    target = torch.randn(2, 32)
    loss = (encoder(faces) - target).square().mean()
    loss.backward()
    assert torch.isfinite(faces.grad).all()
    assert torch.count_nonzero(faces.grad) > 0
    for name, parameter in encoder.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    for parameter in (
        encoder.geometry_embedding[0].weight,
        encoder.uv_embedding[0].weight,
        encoder.query,
        encoder.layers[0].self_attn.in_proj_weight,
        encoder.layers[-1].self_attn.in_proj_weight,
        encoder.readout_attention.in_proj_weight,
    ):
        assert torch.count_nonzero(parameter.grad) > 0


def test_state_restore_is_strict_and_reproduces_output():
    torch.manual_seed(48)
    original = small_encoder().eval()
    restored = small_encoder().eval()
    restored.load_state_dict(original.state_dict(), strict=True)
    faces = torch.randn(2, 6, 16, 16)
    with torch.no_grad():
        torch.testing.assert_close(restored(faces), original(faces), rtol=0, atol=0)
    torch.testing.assert_close(restored.uv_grid, original.uv_grid, rtol=0, atol=0)
    incomplete = dict(original.state_dict())
    incomplete.pop("query")
    with pytest.raises(RuntimeError, match="Missing key"):
        restored.load_state_dict(incomplete, strict=True)


@pytest.mark.parametrize("options", [
    {"in_channels": 3}, {"in_channels": 6.0}, {"in_channels": True},
    {"out_channels": 0}, {"hidden_dim": 0}, {"hidden_dim": 31},
    {"embedding_dim": -1}, {"num_layers": 0}, {"num_layers": 1.5},
    {"nhead": 0}, {"nhead": True}, {"dim_feedforward": 0},
    {"dropout": -0.1}, {"dropout": 1.0}, {"dropout": float("nan")},
])
def test_invalid_configuration_fails_explicitly(options):
    with pytest.raises(ValueError):
        small_encoder(**options)


@pytest.mark.parametrize("faces", [
    torch.zeros(1, 16, 16, 6), torch.zeros(1, 3, 16, 16),
    torch.zeros(1, 6, 8, 8), torch.zeros(1, 256, 6),
    torch.zeros(0, 6, 16, 16), torch.zeros(1, 6, 16, 16, dtype=torch.long),
    None,
])
def test_invalid_input_fails_explicitly(faces):
    with pytest.raises(ValueError, match="face_points"):
        small_encoder()(faces)
