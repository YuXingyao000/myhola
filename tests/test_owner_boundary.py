"""Owner locality, confidence routing, output contracts and explicit migration."""
from copy import deepcopy

import pytest
import torch

from src.models.vae import AutoEncoder_intersection
from src.models.vae.decoders.owner_boundary import OwnerBoundaryAttention
from tests.test_decoder_face_width import source_model
from tests.test_intersection_vae import encoder_batch
from tests.test_uv_boundary_model import add_targets
from tools.migrate_checkpoint import migrate


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def migrated(mode):
    source, cfg = source_model()
    checkpoint = {"state_dict": {"model." + k: v for k, v in source.state_dict().items()},
                  "hyper_parameters": {"model": deepcopy(cfg)}, "optimizer_states": [{}],
                  "loops": {}, "epoch": 24, "global_step": 10375}
    migrate(checkpoint, owner_readout=mode)
    model = AutoEncoder_intersection(checkpoint["hyper_parameters"]["model"])
    model.load_state_dict({k.removeprefix("model."): v for k, v in checkpoint["state_dict"].items()}, strict=True)
    return source, model.eval(), checkpoint


def test_confidence_masks_sources_not_queries_and_has_null_route():
    torch.manual_seed(51)
    block = OwnerBoundaryAttention(16, 4)
    messages = torch.randn(5, 16, requires_grad=True)
    probabilities = torch.tensor([1., 0., .7, .6, .4], requires_grad=True)
    owners = torch.tensor([0, 0, 0, 1, 1])
    output = block(messages, probabilities, owners, 3)
    changed = messages.detach().clone()
    changed[1] += torch.randn(16) * 100
    other = block(changed, probabilities, owners, 3)
    torch.testing.assert_close(other[[0, 2, 3, 4]], output[[0, 2, 3, 4]])
    assert output[1].abs().sum() > 0  # low-probability query can still receive
    output.square().sum().backward()
    assert probabilities.grad is None
    assert messages.grad.abs().sum() > 0
    zero = block(messages, torch.zeros(5), owners, 3)
    assert torch.count_nonzero(zero) == 0
    tiny = block(messages, torch.full((5,), 1e-7), owners, 3)
    assert tiny.abs().max() < 1e-5


def test_owner_attention_is_partitioned_and_order_equivariant():
    block = OwnerBoundaryAttention(16, 4).eval()
    messages, probabilities = torch.randn(5, 16), torch.rand(5)
    owners = torch.tensor([0, 0, 0, 1, 1])
    expected = block(messages, probabilities, owners, 2)
    changed = messages.clone()
    changed[3:] *= 20
    torch.testing.assert_close(block(changed, probabilities, owners, 2)[:3], expected[:3])
    permutation = torch.tensor([3, 0, 4, 2, 1])
    torch.testing.assert_close(
        block(messages[permutation], probabilities[permutation], owners[permutation], 2),
        expected[permutation],
    )


@pytest.mark.parametrize("mode", ["direct", "residual"])
def test_migration_keeps_old_heads_and_full_training_reaches_encoder(mode):
    source, model, checkpoint = migrated(mode)
    assert not model.freeze_encoder and all(p.requires_grad for p in model.parameters())
    assert "optimizer_states" not in checkpoint and "loops" not in checkpoint
    assert "already applied" in migrate(checkpoint, owner_readout=mode)[-1]
    old, new = source.state_dict(), model.state_dict()
    for key in old:
        if not key.startswith("decoder.refiner."):
            torch.testing.assert_close(new[key], old[key], rtol=0, atol=0)
    batch = add_targets(encoder_batch())
    loss, _ = model(batch)
    loss["total_loss"].backward()
    for module in (model.encoder.face_coords, model.encoder.edge_coords,
                   model.decoder.inter.pair_query, model.decoder.boundary_exchange.attention,
                   model.decoder.teacher_edge_geometry):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.decoder.refiner is None


@pytest.mark.parametrize("mode,face_calls", [("direct", 1), ("residual", 2)])
def test_final_surface_reuse_shared_curve_and_single_face(mode, face_calls):
    _, model, _ = migrated(mode)
    decoder = model.decoder
    calls = []
    hook = decoder.face_geometry.bbox_decoder.register_forward_hook(lambda *args: calls.append(1))
    with torch.no_grad():
        prediction = model.decode(torch.randn(5, model.face_feature_dim), torch.tensor([3, 2]))
    hook.remove()
    assert len(calls) == face_calls
    if mode == "direct":
        assert prediction.refined.face_geometry is prediction.coarse.face_geometry
        assert decoder.boundary_exchange.writeback is None
        assert not any("relation_encoder" in k or "face_delta" in k for k in decoder.state_dict())
    pairs = prediction.pairs
    torch.testing.assert_close(prediction.refined.edge_geometry.points,
                               prediction.refined.edge_geometry.points[pairs.reverse].flip(1))
    torch.testing.assert_close(prediction.refined.edge_logits, prediction.refined.edge_logits[pairs.reverse])
    assert prediction.edge_uv.shape == (8, 2, 16, 2)
    with torch.no_grad():
        single = model.decode(torch.randn(2, model.face_feature_dim), torch.tensor([1, 1]))
    assert single.refined.edge_geometry.points.shape == (0, 16, 3)
    assert torch.count_nonzero(single.delta_face) == 0


def test_spatial_read_uses_first_owner_only():
    _, model, _ = migrated("direct")
    inter = model.decoder.inter
    faces = torch.randn(3, model.decoder.face_dim)
    local = torch.randn(3, inter.spatial_memory.local_projection.in_features, 16, 16)
    pairs = torch.tensor([[0, 1], [1, 0]])
    expected = inter(faces, local, pairs)
    changed = local.clone()
    changed[0] += torch.randn_like(changed[0]) * 10
    actual = inter(faces, changed, pairs)
    assert not torch.allclose(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])
    view = inter.inspect_attention(faces, local, pairs)
    assert view["attention"].shape[-1] == 256


def test_both_variants_have_bitwise_identical_common_initialization():
    _, direct, _ = migrated("direct")
    _, residual, _ = migrated("residual")
    a, b = direct.state_dict(), residual.state_dict()
    for key in a.keys() & b.keys():
        torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
