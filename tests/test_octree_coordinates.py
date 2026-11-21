import copy

import torch
from torch.nn import functional as F

from src.models.vae.decoders.octree import OctreeCoordinateHead, decode_local_xyz, encode_local_xyz


def test_continuous_roundtrip_and_all_leaf_addresses():
    grid = torch.cartesian_prod(*(torch.arange(8) for _ in range(3))).float()
    centers = -1 + (grid + 0.5) * 0.25
    path, offset = encode_local_xyz(centers, 3)
    assert path.unique(dim=0).shape[0] == 512
    torch.testing.assert_close(decode_local_xyz(path, offset), centers, atol=0, rtol=0)
    points = torch.cat((torch.rand(1024, 3) * 2 - 1, torch.tensor([[-1., 0., 1.]])))
    for depth in (10, 11, 12):
        path, offset = encode_local_xyz(points, depth)
        torch.testing.assert_close(decode_local_xyz(path, offset), points, atol=1e-7, rtol=0)
        assert offset.abs().max() <= 0.5
    path, offset = encode_local_xyz(torch.tensor([[.3, -.2, .7]]), 3)
    assert path.tolist() == [[5, 3, 6]]
    torch.testing.assert_close(offset, torch.tensor([[-.3, -.3, .3]]))


def test_causal_targets_free_decoding_and_checkpoint_gradients():
    torch.manual_seed(42)
    head = OctreeCoordinateHead(feature_dim=5, width=16, blocks=2, depth=4)
    plain = copy.deepcopy(head)
    plain.checkpoint_levels = False
    x = torch.randn(2, 16, 5, requires_grad=True)
    other_x = x.detach().clone().requires_grad_()
    path, offset = encode_local_xyz(torch.rand(2, 16, 3) * 2 - 1, 4)
    out = head.teacher_forced(x, path)
    alt = plain.teacher_forced(other_x, path)
    def objective(pred):
        return F.cross_entropy(pred['logits'].flatten(0, -2), path.flatten()) + F.l1_loss(pred['offset'], offset)
    objective(out).backward()
    objective(alt).backward()
    torch.testing.assert_close(x.grad, other_x.grad)
    for left, right in zip(head.parameters(), plain.parameters()):
        assert left.grad is not None and torch.isfinite(left.grad).all()
        torch.testing.assert_close(left.grad, right.grad)
    assert head.child_classifier.weight.grad.abs().sum() > 0
    assert head.offset_regressor.weight.grad.abs().sum() > 0

    with torch.no_grad():
        changed = path.clone()
        changed[..., 2:] = (changed[..., 2:] + 1) % 8
        causal = head.teacher_forced(x, changed)
        torch.testing.assert_close(out['logits'][..., :3, :], causal['logits'][..., :3, :], atol=0, rtol=0)
        prediction = head.generate(x)
        replay = head.teacher_forced(x, prediction['path'])
        torch.testing.assert_close(prediction['offset'], replay['offset'], atol=0, rtol=0)
        torch.testing.assert_close(prediction['local_xyz'], decode_local_xyz(prediction['path'], replay['offset']))
        with torch.autocast('cpu', dtype=torch.bfloat16):
            mixed = head.generate(x.to(torch.bfloat16))
        assert mixed['local_xyz'].dtype == torch.float32
        assert head.generate(x[:0])['local_xyz'].shape == (0, 16, 3)


def test_free_path_detaches_only_discrete_decisions():
    head = OctreeCoordinateHead(feature_dim=5, width=16, blocks=2, depth=4)
    x = torch.randn(1, 16, 5, requires_grad=True)
    head.generate(x)['local_xyz'].square().mean().backward()
    assert x.grad.abs().sum() > 0
    assert head.offset_regressor.weight.grad.abs().sum() > 0
    assert head.child_classifier.weight.grad is None
