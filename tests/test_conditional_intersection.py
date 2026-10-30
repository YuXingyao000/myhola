"""Conditional Intersection math, migration, locality and gradient checks."""

import copy

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from src.models.vae.decoders.intersection import AttnIntersection
from src.models.vae.decoders.conditional_intersection import (
    ConditionalCrossAttention,
    ConditionalIntersection,
    UVPositionEncoding,
)


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def make_model(**changes):
    legacy = AttnIntersection(8, 32, 2)
    config = {"candidate_chunk_size": 2, "checkpoint_candidates": False, **changes}
    return legacy, ConditionalIntersection(copy.deepcopy(legacy), 6, config)


def inputs(requires_grad=False):
    faces = torch.randn(4, 8, requires_grad=requires_grad)
    local = torch.randn(4, 6, 16, 16, requires_grad=requires_grad)
    pairs = torch.tensor([[0, 1], [1, 0], [0, 2], [2, 0], [2, 3], [3, 2]])
    return faces, local, pairs


def disable_dropout(model):
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0
        elif isinstance(module, (nn.MultiheadAttention, ConditionalCrossAttention)):
            module.dropout = 0.0


def test_zero_local_branch_retains_legacy_weights_and_initial_eval_output():
    torch.manual_seed(530)
    legacy, candidate = make_model()
    legacy.eval()
    candidate.eval()
    old_state, new_state = legacy.state_dict(), candidate.state_dict()
    assert old_state.keys() <= new_state.keys()
    for key, value in old_state.items():
        torch.testing.assert_close(value, new_state[key], rtol=0, atol=0)
    faces, local, pairs = inputs()
    with torch.no_grad():
        expected = legacy(faces[pairs])
        actual = candidate(faces, local, pairs)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    assert actual.shape == (6, 16)


def test_local_branch_opens_gradient_path_to_positions_and_cnn_after_update():
    torch.manual_seed(531)
    _, model = make_model(checkpoint_candidates=True)
    model.train()
    disable_dropout(model)
    faces, local, pairs = inputs(requires_grad=True)
    target = torch.randn(6, 16)
    F.mse_loss(model(faces, local, pairs), target).backward()
    projection = model.spatial_memory.local_projection
    assert projection.weight.grad.abs().sum() > 1e-6
    assert local.grad.abs().max() == 0
    first_reference_grad = sum(p.grad.abs().sum() for p in model.reference_uv_head.parameters())
    assert first_reference_grad < 1e-6  # equal V makes position irrelevant initially
    with torch.no_grad():
        projection.weight.add_(projection.weight.grad, alpha=-0.1)
    model.zero_grad(set_to_none=True)
    local.grad = None
    F.mse_loss(model(faces, local, pairs), target).backward()
    reference_grad = sum(p.grad.abs().sum() for p in model.reference_uv_head.parameters())
    assert reference_grad > 1e-8
    assert local.grad.abs().sum() > 1e-8
    for layer in model.layers:
        attention = layer.multihead_attn
        assert attention.query_position.weight.grad.abs().sum() > 1e-8
        assert attention.key_position.weight.grad.abs().sum() > 1e-8
        if attention.query_scale is not None:
            assert attention.query_scale[0].weight.grad.abs().sum() > 1e-8
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_attention_matches_concatenated_content_position_scaled_dot_product():
    torch.manual_seed(532)
    attention = ConditionalCrossAttention(nn.MultiheadAttention(32, 8, batch_first=True), False).eval()
    query, reference = torch.randn(3, 1, 32), torch.randn(3, 1, 32)
    memory, grid = torch.randn(2, 256, 32), torch.randn(256, 32)
    owners = torch.tensor([1, 0, 1])
    keys, values, key_positions = attention.project_memory(memory, grid)
    actual, weights = attention(query, reference, keys, values, key_positions, owners, True)
    q_content = attention._heads(F.linear(query, attention.in_proj_weight[:32], attention.in_proj_bias[:32]))
    q_position = attention._heads(attention.query_position(reference * attention.query_scale(query)))
    concatenated_q = torch.cat((q_content, q_position), dim=-1)
    concatenated_k = torch.cat((keys[owners], key_positions.expand(3, -1, -1, -1)), dim=-1)
    expected = F.scaled_dot_product_attention(concatenated_q, concatenated_k, values[owners])
    expected = attention.out_proj(expected.transpose(1, 2).reshape(3, 1, 32))
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(weights.sum(-1), torch.ones(3, 8, 1))


def test_checkpoint_chunking_preserves_outputs_and_all_gradients():
    torch.manual_seed(533)
    _, direct = make_model(candidate_chunk_size=100, checkpoint_candidates=False)
    with torch.no_grad():
        direct.spatial_memory.local_projection.weight.normal_(std=0.02)
    chunked = copy.deepcopy(direct)
    for layer in chunked.layers:
        layer.chunk_size = 2
        layer.checkpoint_candidates = True
    direct.train()
    chunked.train()
    disable_dropout(direct)
    disable_dropout(chunked)
    faces, local, pairs = inputs(requires_grad=True)
    other_faces, other_local = faces.detach().clone().requires_grad_(), local.detach().clone().requires_grad_()
    output = direct(faces, local, pairs)
    other_output = chunked(other_faces, other_local, pairs)
    target = torch.randn_like(output)
    (output * target).sum().backward()
    (other_output * target).sum().backward()
    torch.testing.assert_close(output, other_output, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(faces.grad, other_faces.grad, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(local.grad, other_local.grad, atol=2e-6, rtol=2e-5)
    for (name, value), (_, other) in zip(direct.named_parameters(), chunked.named_parameters()):
        assert value.grad is not None, name
        torch.testing.assert_close(value.grad, other.grad, atol=3e-6, rtol=3e-5, msg=name)


def test_pair_indices_isolate_objects_and_preserve_permutation_equivariance():
    torch.manual_seed(534)
    _, model = make_model()
    model.eval()
    with torch.no_grad():
        model.spatial_memory.local_projection.weight.normal_(std=0.02)
    faces, local, _ = inputs()
    pairs = torch.tensor([[0, 1], [1, 0], [2, 3], [3, 2]])
    permutation = torch.tensor([2, 0, 3, 1])
    inverse = permutation.argsort()
    with torch.no_grad():
        output = model(faces, local, pairs)
        isolated = model(faces[:2], local[:2], pairs[:2])
        reordered = model(faces[permutation], local[permutation], inverse[pairs])
        changed_faces, changed_local = faces.clone(), local.clone()
        changed_faces[2:] *= 50
        changed_local[2:] *= 50
        changed = model(changed_faces, changed_local, pairs)
    torch.testing.assert_close(output[:2], isolated)
    torch.testing.assert_close(output, reordered)
    torch.testing.assert_close(output[:2], changed[:2])
    assert not torch.allclose(output[2:], changed[2:])


def test_attention_inspection_and_uv_grid_axis_convention():
    torch.manual_seed(535)
    _, model = make_model()
    model.eval()
    faces, local, pairs = inputs()
    result = model.inspect_attention(faces, local, pairs[:2], layer_index=0)
    assert result["reference_uv"].shape == (2, 1, 2)
    assert result["attention"].shape == (2, 8, 1, 256)
    assert ((result["reference_uv"] >= 0) & (result["reference_uv"] <= 1)).all()
    grid = result["grid_uv"]
    torch.testing.assert_close(grid[0], torch.tensor([0.0, 0.0]))
    torch.testing.assert_close(grid[15], torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(grid[240], torch.tensor([0.0, 1.0]))
    torch.testing.assert_close(grid[-1], torch.tensor([1.0, 1.0]))
    assert UVPositionEncoding(512)(grid).shape == (256, 512)


def test_empty_pairs_return_differentiable_empty_features():
    _, model = make_model()
    faces, local, _ = inputs(requires_grad=True)
    pairs = torch.empty(0, 2, dtype=torch.long)
    output = model(faces, local, pairs)
    assert output.shape == (0, 16)
    output.sum().backward()
    assert faces.grad is not None
    assert model.inspect_attention(faces, local, pairs)["attention"].shape == (0, 8, 1, 256)


def test_full_model_migration_rejects_missing_or_unexpected_source_tensors(tmp_path):
    from src.models.vae.checkpoints import IntersectionCheckpointLoader

    legacy, conditional = make_model()
    source_model, target_model = nn.Module(), nn.Module()
    for model, inter in ((source_model, legacy), (target_model, conditional)):
        model.decoder = nn.Module()
        model.decoder.inter = inter
        model.decoder.intersection_kind = "conditional_attention" if model is target_model else "legacy"
        model.encoder = nn.Linear(8, 8)
    source = source_model.state_dict()
    path = tmp_path / "source.ckpt"
    torch.save({"state_dict": {"model." + key: value for key, value in source.items()},
                "epoch": 116, "global_step": 42471}, path)
    loader = IntersectionCheckpointLoader(target_model)
    loader.init_conditional_intersection(str(path))
    for name, tensor in source.items():
        torch.testing.assert_close(target_model.state_dict()[name], tensor, rtol=0, atol=0)
    assert target_model.initialization_report["source_global_step"] == 42471
    for invalid in ({key: value for key, value in source.items() if key != "encoder.bias"},
                    {**source, "unexpected.weight": torch.zeros(1)}):
        torch.save({"state_dict": invalid}, path)
        with pytest.raises(ValueError, match="complete matching legacy-inter"):
            loader.init_conditional_intersection(str(path))
