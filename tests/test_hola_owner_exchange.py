"""The two invariants that isolate the new information path from HoLa."""
import torch
from torch import nn
from torch.nn import functional as F

from src.models.vae.hola_owner import AutoEncoder_hola_owner, FaceGeometryRead, OwnerBoundaryExchange


def test_identity_then_messages_only_from_same_owner():
    torch.manual_seed(42)
    exchange = OwnerBoundaryExchange(4, 8, 3, 2, 1)
    pairs = torch.cartesian_prod(torch.arange(2), torch.arange(2))
    h = torch.randn(4, 4, requires_grad=True)
    geometry = torch.randn(4, 3, requires_grad=True)
    logits = torch.randn(4, requires_grad=True)

    def forward():
        return exchange(h, geometry, logits, pairs, pairs[:, 1], 2, 2)

    torch.testing.assert_close(forward(), h, atol=0, rtol=0)
    with torch.no_grad():
        exchange.readout.weight[:, 4:].fill_(1.0)
    forward()[0].sum().backward()
    assert geometry.grad[1].abs().sum() > 0
    assert geometry.grad[2:].count_nonzero() == 0
    assert logits.grad is None


class PairFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Parameter(torch.randn(16, 4))
        with torch.no_grad():
            self.features[:, 0] = -2
            self.features[[0, 1, 10], 0] = 2

    def forward(self, faces):
        index = faces[:, 0, 0].long() * 4 + faces[:, 1, 0].long()
        return self.features[index]


def routing_model():
    """Small trainable heads exercise the actual selection and exchange forward."""
    torch.manual_seed(42)
    model = AutoEncoder_hola_owner.__new__(AutoEncoder_hola_owner)
    nn.Module.__init__(model)
    model.inter = PairFeatures()
    model.classifier = nn.Linear(4, 1, bias=False)
    with torch.no_grad():
        model.classifier.weight.copy_(torch.tensor([[1., 0., 0., 0.]]))
    model.edge_points_decoder = nn.Sequential(nn.Linear(4, 12), nn.Unflatten(-1, (4, 3)))
    model.edge_center_scale_decoder = nn.Linear(4, 4)
    model.owner_exchange = OwnerBoundaryExchange(4, 8, 3, 2, 2)
    model.face_geometry_read = None
    model.pair_chunk_size = 2
    model.runtime_mode = 'train'
    model.trainable_scope = 'all'
    model.intersection_noise_std = 0.0
    model.use_intersection_noise = False
    model.loss_fn = F.l1_loss
    faces = torch.arange(4).float()[:, None]
    plane = torch.tensor([[[0., 0., 0.], [0., 1., 0.]], [[1., 0., 0.], [1., 1., 0.]]])
    result = {
        'face_points_local': plane.repeat(4, 1, 1, 1),
        'face_center_scale': torch.tensor([[0., 0., 0., 1.]]).repeat(4, 1),
    }
    return model, faces, result


def geometry_trace(model):
    decoded, pairs = [], []
    hooks = [
        model.edge_points_decoder.register_forward_pre_hook(lambda m, a: decoded.append(a[0].detach().clone())),
        model.owner_exchange.geometry.register_forward_pre_hook(lambda m, a: pairs.append(a[2].detach().clone())),
    ]
    return decoded, pairs, hooks


def test_inference_filters_before_coarse_geometry_and_keeps_identity():
    model, faces, result = routing_model()
    model.eval()
    decoded, pairs, hooks = geometry_trace(model)
    with torch.no_grad():
        final = model._intersect_faces(faces, {}, None, result, None)
    for hook in hooks:
        hook.remove()
    expected_pairs = torch.tensor([[0, 0], [0, 1], [2, 2]])
    expected = model.inter.features[[0, 1, 10]]
    torch.testing.assert_close(torch.cat(decoded), expected)
    assert torch.equal(torch.cat(pairs), expected_pairs)
    torch.testing.assert_close(final, expected, atol=0, rtol=0)
    assert torch.equal(result['pred_edge_face_connectivity'][:, 1:], expected_pairs)
    torch.testing.assert_close(result['pred_face_adj_prob'], model.inter.features[:, 0].sigmoid())
    assert torch.equal(result['pred_face_adj'], model.inter.features[:, 0] > 0)


def test_training_gt_selection_loss_and_geometry_gradients():
    model, faces, result = routing_model()
    model.train()
    # GT includes classifier misses; sampled negatives include a high-confidence false edge.
    batch = {
        'face_counts': torch.tensor([2, 2]),
        'edge_face_connectivity': torch.tensor([[2, 3, 2], [0, 0, 1], [1, 1, 0]]),
        'zero_positions': torch.tensor([[0, 0], [2, 3]]),
    }
    encoding = {'edge_features': torch.randn(3, 4, requires_grad=True)}
    decoded, pairs, hooks = geometry_trace(model)
    with torch.no_grad():
        model.owner_exchange.readout.weight[:, 4:].fill_(0.1)
    final = model._intersect_faces(faces, encoding, batch, result, None)
    for hook in hooks:
        hook.remove()
    selected_rows = torch.tensor([14, 1, 4])
    torch.testing.assert_close(torch.cat(decoded), model.inter.features[selected_rows])
    assert torch.equal(torch.cat(pairs), batch['edge_face_connectivity'][:, 1:])
    expected_logits = model.inter.features[[14, 1, 4, 0, 11], 0]
    torch.testing.assert_close(result['loss_edge'], F.binary_cross_entropy_with_logits(
        expected_logits, torch.tensor([1., 1., 1., 0., 0.])))
    torch.testing.assert_close(result['loss_edge_feature'], F.l1_loss(
        model.inter.features[selected_rows], encoding['edge_features'][[2, 0, 1]].detach()))
    assert torch.isfinite(final).all()  # Includes an owner with no selected boundary.

    final.square().sum().backward(retain_graph=True)
    assert model.inter.features.grad[selected_rows].abs().sum() > 0
    rejected = torch.ones(16, dtype=torch.bool)
    rejected[selected_rows] = False
    assert model.inter.features.grad[rejected].count_nonzero() == 0
    assert model.classifier.weight.grad is None  # Geometry cannot lower routing confidence.
    model.zero_grad(set_to_none=True)
    (result['loss_edge'] + result['loss_edge_feature']).backward()
    assert model.inter.features.grad[[0, 11]].abs().sum() > 0
    assert model.classifier.weight.grad.abs().sum() > 0
    assert encoding['edge_features'].grad is None


def test_no_selected_edges_skips_geometry_and_exchange():
    model, faces, result = routing_model()
    model.eval()
    with torch.no_grad():
        model.inter.features[:, 0] = -2
    calls = []
    hooks = [module.register_forward_pre_hook(lambda m, a: calls.append(m)) for module in (
        model.edge_points_decoder, model.owner_exchange.geometry, model.owner_exchange,
    )]
    with torch.no_grad():
        final = model._intersect_faces(faces, {}, None, result, None)
    for hook in hooks:
        hook.remove()
    assert final.shape == (0, 4)
    assert not calls
    assert result['pred_edge_face_connectivity'].shape == (0, 3)
    assert not result['pred_face_adj'].any()


def test_geometry_classifier_initial_identity_then_bce_reaches_surfaces():
    model, faces, _ = routing_model()
    model.eval()
    reader = FaceGeometryRead(4, 8, 2, 16)
    model.face_geometry_read = reader
    original_weight = model.classifier.weight.detach().clone()
    model.classifier = nn.Linear(12, 1, bias=False)
    with torch.no_grad():
        model.classifier.weight.zero_()
        model.classifier.weight[:, :4].copy_(original_weight)
    surfaces = torch.randn(4, 2, 2, 3, requires_grad=True)
    pairs = torch.tensor([[0, 1], [1, 0]])
    h, logits = model._candidate_features(faces, pairs, reader.encode(surfaces))
    torch.testing.assert_close(logits, (h @ original_weight.T).squeeze(-1), atol=0, rtol=0)
    F.binary_cross_entropy_with_logits(logits, torch.tensor([1., 0.])).backward()
    assert model.classifier.weight.grad[:, 4:].abs().sum() > 0
    assert surfaces.grad.count_nonzero() == 0

    model.zero_grad(set_to_none=True)
    surfaces.grad = None
    with torch.no_grad():
        model.classifier.weight[:, 4:].fill_(0.1)
    _, logits = model._candidate_features(faces, pairs, reader.encode(surfaces))
    F.binary_cross_entropy_with_logits(logits, torch.tensor([1., 0.])).backward()
    assert surfaces.grad[:2].abs().sum() > 0
    assert surfaces.grad[2:].count_nonzero() == 0
    assert reader.query_projection.weight.grad.abs().sum() > 0
    assert reader.attention.in_proj_weight.grad.abs().sum() > 0


def test_geometry_read_respects_order_and_face_reindexing():
    torch.manual_seed(42)
    reader = FaceGeometryRead(4, 8, 2, 16).eval()
    h = torch.randn(1, 4).expand(2, -1)
    surfaces = torch.randn(3, 2, 2, 3)
    pairs = torch.tensor([[0, 1], [1, 0]])
    tokens = reader.encode(surfaces)
    g = reader(h, tokens, pairs)
    assert not torch.allclose(g[0], g[1])
    permutation = torch.tensor([2, 0, 1])
    inverse = permutation.argsort()
    torch.testing.assert_close(g, reader(h, reader.encode(surfaces[permutation]), inverse[pairs]))
    # Encoding/reading is per sample, with no dependence on UV flattening order.
    torch.testing.assert_close(g, reader(h, tokens.flip(1), pairs))
