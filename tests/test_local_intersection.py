"""Spatial intersection contracts; passing these does not establish CAD quality."""

import pytest
import torch
from torch import nn

from src.models.blocks import IntersectionDecoder
from src.models.blocks import LocalPairIntersection
from src.models.vae import AutoEncoder_light


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def local_config(**changes):
    return {
        "kind": "local_attention", "hidden_dim": 16, "num_heads": 2,
        "num_layers": 2, "ffn_dim": 32, "dropout": 0.0,
        "candidate_chunk_size": 64, "checkpoint_candidates": False,
        **changes,
    }


def make_decoder(*, refinement=True, local_changes=None):
    legacy = AutoEncoder_light({
        **AutoEncoder_light.DEFAULTS,
        "norm": "layer", "in_channels": 6, "hidden_channels": 16,
        "latent_channels": 2, "bottleneck_dim": 16, "nhead": 2,
        "num_encoder_layers": 1, "num_decoder_layers": 1, "num_gat_layers": 1,
        "intersection_dim": 16, "intersection_layers": 1,
        "face_bbox_decoder_blocks": 1, "edge_bbox_decoder_blocks": 1,
    })
    config = {
        "enabled": refinement, "hidden_dim": 16, "geometry_dim": 8,
        "geometry_chunk_size": 4, "candidate_chunk_size": 4,
        "checkpoint_candidates": False,
    }
    return IntersectionDecoder(legacy, config, local_config(**(local_changes or {})))


def activate_refiner(model):
    with torch.no_grad():
        for head in (model.refiner.face_delta, model.refiner.edge_delta):
            head.weight.normal_(std=0.02)
            head.bias.normal_(std=0.01)


def disable_dropout(model):
    # Keep training/checkpoint execution active while removing stochastic noise.
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0
        elif isinstance(module, nn.MultiheadAttention):
            module.dropout = 0.0


@pytest.mark.parametrize("block_type", ["cross_attention", "transformer_decoder"])
def test_one_pair_query_reads_both_spatial_banks_with_xyz_gradients(block_type):
    torch.manual_seed(73)
    model = LocalPairIntersection(8, 16, 2, local_config(block_type=block_type)).eval()
    faces = torch.randn(2, 8, requires_grad=True)
    local = torch.randn(2, 2, 16, 16, requires_grad=True)
    xyz = torch.randn(2, 16, 16, 3, requires_grad=True)
    observed = []

    def record_attention(module, args, kwargs):
        query = kwargs.get("query", args[0] if len(args) > 0 else None)
        key = kwargs.get("key", args[1] if len(args) > 1 else None)
        value = kwargs.get("value", args[2] if len(args) > 2 else None)
        sequence_axis = 1 if module.batch_first else 0
        observed.append((query.shape[sequence_axis], key.shape[sequence_axis], value.shape[sequence_axis]))

    handles = [
        module.register_forward_pre_hook(record_attention, with_kwargs=True)
        for module in model.modules() if isinstance(module, nn.MultiheadAttention)
    ]
    output = model(faces, local, xyz, torch.tensor([[0, 1]]))
    for handle in handles:
        handle.remove()
    assert output.shape == (1, 16)
    expected = [(1, 512, 512)] * 2 if block_type == "cross_attention" else [(1, 1, 1), (1, 512, 512)] * 2
    assert observed == expected
    (output * torch.randn_like(output)).sum().backward()
    for tensor in (faces, local, xyz):
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all()
        assert (tensor.grad.flatten(1).abs().sum(1) > 0).all()
    # Every spatial site can contribute, instead of a single pooled KV token.
    assert (local.grad.abs().sum(1) > 0).all()


@pytest.mark.parametrize("block_type", ["cross_attention", "transformer_decoder"])
@pytest.mark.parametrize("refinement, expected_calls", [(False, 1), (True, 2)])
def test_coarse_surface_is_decoded_once_and_reused(refinement, expected_calls, block_type):
    torch.manual_seed(74)
    model = make_decoder(refinement=refinement, local_changes={"block_type": block_type}).eval()
    local_maps, geometry_inputs = [], []
    final_conv = model.face_geometry.points_decoder[-2]
    assert isinstance(final_conv, nn.Conv2d)
    handle = final_conv.register_forward_pre_hook(lambda _module, args: local_maps.append(args[0]))
    inter_handle = model.inter.register_forward_pre_hook(
        lambda _module, args: geometry_inputs.append((args[1], args[2]))
    )
    output = model(torch.randn(3, 8), torch.tensor([3]))
    handle.remove()
    inter_handle.remove()
    assert len(local_maps) == expected_calls
    assert len(geometry_inputs) == 1
    assert geometry_inputs[0][0] is local_maps[0]
    assert geometry_inputs[0][1] is output.coarse.face_geometry.points
    assert output.coarse.edge_features.shape == (6, 16)
    assert not torch.allclose(
        output.coarse.edge_features, output.coarse.edge_features[output.pairs.reverse]
    )
    torch.testing.assert_close(
        output.refined.edge_logits, output.refined.edge_logits[output.pairs.reverse]
    )
    expected_local = model.face_geometry.points_decoder(output.coarse.face_features)
    torch.testing.assert_close(output.coarse.face_geometry.local_points, expected_local)
    if not refinement:
        assert output.refined is output.coarse


@pytest.mark.parametrize("block_type", ["cross_attention", "transformer_decoder"])
def test_initial_edge_loss_backpropagates_to_face_cnn_and_predicted_xyz(block_type):
    torch.manual_seed(75)
    model = make_decoder(refinement=False, local_changes={"block_type": block_type}).train()
    captured = []

    def retain_local_grad(_module, args):
        args[0].retain_grad()
        captured.append(args[0])

    handle = model.face_geometry.points_decoder[-2].register_forward_pre_hook(retain_local_grad)
    output = model(torch.randn(2, 8), torch.tensor([2]))
    handle.remove()
    output.coarse.face_geometry.points.retain_grad()
    # Only initial edge geometry is supervised here: no direct face loss.
    target = torch.randn_like(output.coarse.edge_geometry.points)
    nn.functional.mse_loss(output.coarse.edge_geometry.points, target).backward()
    assert captured[0].grad is not None and captured[0].grad.abs().sum() > 0
    xyz_gradient = output.coarse.face_geometry.points.grad
    assert xyz_gradient is not None and (xyz_gradient.flatten(1).abs().sum(1) > 0).all()
    cnn_grads = [p.grad for p in model.face_geometry.points_decoder.parameters()]
    assert all(grad is not None and torch.isfinite(grad).all() for grad in cnn_grads)
    assert sum(grad.abs().sum() for grad in cnn_grads) > 0
    bbox_grads = [p.grad for p in model.face_geometry.bbox_decoder.parameters()]
    assert all(grad is not None and torch.isfinite(grad).all() for grad in bbox_grads)
    assert sum(grad.abs().sum() for grad in bbox_grads) > 0


@pytest.mark.parametrize("block_type", ["cross_attention", "transformer_decoder"])
def test_packed_object_isolation_and_face_permutation_equivariance(block_type):
    torch.manual_seed(76)
    model = make_decoder(local_changes={"block_type": block_type}).eval()
    activate_refiner(model)
    z = torch.randn(5, 8)
    with torch.no_grad():
        packed = model(z, torch.tensor([3, 2]))
        alone = model(z[:3], torch.tensor([3]))
        changed_z = z.clone()
        changed_z[3:] = 20 * torch.randn_like(changed_z[3:])
        changed = model(changed_z, torch.tensor([3, 2]))
        permutation = torch.tensor([2, 0, 1])
        reordered = model(z[:3][permutation], torch.tensor([3]))
    assert packed.pairs.indices.shape == (8, 2)
    torch.testing.assert_close(packed.pairs.indices[packed.pairs.reverse], packed.pairs.indices.flip(-1))
    original_rows = {tuple(pair): row for row, pair in enumerate(alone.pairs.indices.tolist())}
    rows = torch.tensor([
        original_rows[tuple(pair)] for pair in permutation[reordered.pairs.indices].tolist()
    ])
    for stage in ("coarse", "refined"):
        together, single = getattr(packed, stage), getattr(alone, stage)
        other, shuffled = getattr(changed, stage), getattr(reordered, stage)
        torch.testing.assert_close(together.face_geometry.points[:3], single.face_geometry.points, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(together.edge_features[:6], single.edge_features, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(other.edge_features[:6], together.edge_features[:6], atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(shuffled.face_geometry.points, single.face_geometry.points[permutation], atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(shuffled.edge_geometry.points, single.edge_geometry.points[rows], atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(shuffled.edge_logits, single.edge_logits[rows], atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("block_type", ["cross_attention", "transformer_decoder"])
def test_candidate_chunking_and_checkpoint_preserve_outputs_and_gradients(block_type):
    torch.manual_seed(77)
    direct = make_decoder(local_changes={"candidate_chunk_size": 64, "block_type": block_type}).train()
    activate_refiner(direct)
    disable_dropout(direct)
    checkpointed = make_decoder(local_changes={
        "candidate_chunk_size": 1, "checkpoint_candidates": True, "block_type": block_type,
    }).train()
    checkpointed.load_state_dict(direct.state_dict(), strict=True)
    disable_dropout(checkpointed)
    checkpointed.checkpoint_candidates = True
    checkpointed.chunk_size = 1
    z = torch.randn(3, 8)
    outputs, inputs = [], []
    for model in (direct, checkpointed):
        latents = z.clone().requires_grad_()
        output = model(latents, torch.tensor([3]))
        loss = (
            output.refined.face_geometry.points.square().mean()
            + output.refined.edge_geometry.points.square().mean()
            + output.refined.edge_logits.square().mean()
        )
        loss.backward()
        inputs.append(latents)
        outputs.append(output)
    torch.testing.assert_close(outputs[0].refined.edge_geometry.points, outputs[1].refined.edge_geometry.points, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(inputs[0].grad, inputs[1].grad, atol=2e-6, rtol=2e-4)
    for (name, parameter), (other_name, other) in zip(direct.named_parameters(), checkpointed.named_parameters()):
        assert name == other_name
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        torch.testing.assert_close(parameter.grad, other.grad, atol=2e-6, rtol=2e-4, msg=name)


@pytest.mark.parametrize("block_type", ["cross_attention", "transformer_decoder"])
def test_singleton_objects_and_empty_candidate_memory(block_type):
    torch.manual_seed(78)
    model = make_decoder(local_changes={"block_type": block_type}).eval()
    activate_refiner(model)
    with torch.no_grad():
        output = model(torch.randn(2, 8), torch.tensor([1, 1]))
        mixed = model(torch.randn(3, 8), torch.tensor([1, 2]))
    assert output.pairs.indices.shape == (0, 2)
    assert output.coarse.edge_features.shape == (0, 16)
    assert output.refined.edge_geometry.points.shape == (0, 16, 3)
    assert output.refined.edge_logits.shape == (0,)
    assert torch.count_nonzero(output.delta_face) == 0
    assert torch.count_nonzero(mixed.delta_face[0]) == 0
    assert mixed.coarse.edge_features.shape == (2, 16)


@pytest.mark.parametrize("block_type", ["cross_attention", "transformer_decoder"])
def test_local_decoder_bfloat16_autocast_has_finite_geometry_gradients(block_type):
    torch.manual_seed(79)
    model = make_decoder(local_changes={"checkpoint_candidates": True, "block_type": block_type}).train()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = model(torch.randn(2, 8), torch.tensor([2]))
        loss = output.refined.edge_geometry.points.square().mean() + output.refined.edge_logits.square().mean()
    loss.backward()
    assert output.coarse.face_geometry.points.dtype == torch.float32
    assert output.refined.edge_geometry.points.dtype == torch.float32
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_untagged_historical_config_keeps_exact_state_and_forward_contract():
    torch.manual_seed(80)
    historical = LocalPairIntersection(8, 16, 2, local_config()).eval()
    explicit = LocalPairIntersection(8, 16, 2, local_config(
        block_type="cross_attention", activation="gelu",
    )).eval()
    explicit.load_state_dict(historical.state_dict(), strict=True)
    assert historical.block_type == explicit.block_type == "cross_attention"
    assert "blocks.0.attention.in_proj_weight" in historical.state_dict()
    assert "blocks.0.self_attn.in_proj_weight" not in historical.state_dict()
    inputs = (torch.randn(2, 8), torch.randn(2, 2, 16, 16),
              torch.randn(2, 16, 16, 3), torch.tensor([[0, 1], [1, 0]]))
    torch.testing.assert_close(historical(*inputs), explicit(*inputs), atol=0, rtol=0)
    replacement = LocalPairIntersection(8, 16, 2, local_config(block_type="transformer_decoder"))
    with pytest.raises(RuntimeError, match="Missing key"):
        replacement.load_state_dict(historical.state_dict(), strict=True)
    with pytest.raises(RuntimeError, match="Missing key"):
        historical.load_state_dict(replacement.state_dict(), strict=True)


def test_full_capacity_layers_match_legacy_decoder_layer_contract():
    from src.models.blocks import AttnIntersection

    model = LocalPairIntersection(8, 16, 2, local_config(
        block_type="transformer_decoder", num_heads=8, ffn_dim=2048,
        dropout=.1, activation="relu",
    )).eval()
    legacy = AttnIntersection(8, 16, 1).eval()
    layer = model.blocks[0]
    assert isinstance(layer, nn.TransformerDecoderLayer)
    assert layer.norm_first and layer.self_attn.batch_first and layer.multihead_attn.batch_first
    assert layer.activation is nn.functional.relu
    assert layer.dropout.p == layer.dropout1.p == layer.dropout2.p == layer.dropout3.p == .1
    assert layer.self_attn.num_heads == layer.multihead_attn.num_heads == 8
    assert layer.linear1.out_features == 2048
    layer.load_state_dict(legacy.layers[0].state_dict(), strict=True)
    query, memory = torch.randn(3, 1, 16), torch.randn(3, 512, 16)
    torch.testing.assert_close(layer(query, memory), legacy.layers[0](query, memory), atol=0, rtol=0)
    assert model.blocks[0] is not model.blocks[1]


@pytest.mark.parametrize("changes", [
    {"block_type": "invalid"},
    {"block_type": "cross_attention", "activation": "relu"},
    {"block_type": "transformer_decoder", "activation": "invalid"},
])
def test_explicit_block_semantics_reject_silent_configuration_drift(changes):
    with pytest.raises(ValueError):
        LocalPairIntersection(8, 16, 2, local_config(**changes))


def test_transformer_checkpoint_preserves_nonzero_dropout_rng_and_gradients():
    import copy

    torch.manual_seed(81)
    direct = LocalPairIntersection(8, 16, 2, local_config(
        block_type="transformer_decoder", dropout=.1, candidate_chunk_size=2,
    )).train()
    checkpointed = copy.deepcopy(direct)
    checkpointed.checkpoint_candidates = True
    inputs = (torch.randn(2, 8), torch.randn(2, 2, 16, 16), torch.randn(2, 16, 16, 3))
    pairs = torch.tensor([[0, 1], [1, 0]])
    probe = torch.randn(2, 16)
    outputs, gradients = [], []
    for model in (direct, checkpointed):
        tensors = tuple(value.clone().requires_grad_() for value in inputs)
        torch.manual_seed(82)
        output = model(*tensors, pairs)
        (output * probe).sum().backward()
        outputs.append(output)
        gradients.append([tensor.grad for tensor in tensors])
    torch.testing.assert_close(outputs[0], outputs[1], atol=0, rtol=0)
    for first, second in zip(gradients[0], gradients[1]):
        torch.testing.assert_close(first, second, atol=1e-6, rtol=1e-5)
    for (name, parameter), (other_name, other) in zip(direct.named_parameters(), checkpointed.named_parameters()):
        assert name == other_name and parameter.grad is not None and other.grad is not None
        torch.testing.assert_close(parameter.grad, other.grad, atol=1e-6, rtol=1e-5, msg=name)
