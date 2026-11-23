import pytest
import torch

from src.models.vae.pcurve_relations import assign_next, relation_loss, PcurveDecoder
from src.models.vae.pcurve_geometry import SplineGeometry


def test_assignment_accepts_closed_self_loop_and_rejects_missing_neighbor():
    owner = torch.tensor([0, 0, 1])
    scores = torch.tensor([[-5.0, 5.0, -torch.inf], [5.0, -5.0, -torch.inf], [-torch.inf, -torch.inf, 5.0]])
    null = torch.zeros(3)
    assert assign_next(scores, null, null, owner).tolist() == [1, 0, 2]
    low = scores.clone()
    low[0, :2] = -20
    assert assign_next(low, null, null, owner)[0].item() == -1
    assert torch.equal(assign_next(scores + 11, null + 11, null + 11, owner), assign_next(scores, null, null, owner))


def test_shared_corner_is_exact_and_gradients_reach_geometry_heads():
    torch.manual_seed(42)
    decoder = PcurveDecoder(width=32, depth=2, controls=8)
    h = torch.randn(6, 64, requires_grad=True)
    f = torch.randn(3, 32, requires_grad=True)
    owner = torch.tensor([0, 0, 0, 0, 1, 2])
    mate = torch.tensor([4, 3, 5, 1, 0, 2])
    nxt = torch.tensor([1, 2, 3, 0, 4, 5])
    poles = decoder(h, f, owner, mate, nxt)
    assert torch.equal(poles[:, -1], poles[nxt, 0])
    poles.square().mean().backward()
    assert h.grad.abs().sum() > 0
    assert decoder.corner.weight.grad.abs().sum() > 0
    assert decoder.interior.weight.grad.abs().sum() > 0
    with pytest.raises(ValueError, match="Incomplete next"):
        decoder(h, f, owner, mate, torch.full_like(nxt, -1))


def test_surface_bridge_preserves_grid_and_propagates_uv_gradient():
    geometry = SplineGeometry(controls=4).double()
    t = torch.linspace(0, 1, 16, dtype=torch.float64)
    v, u = torch.meshgrid(t, t, indexing="ij")
    grid = torch.stack([2 * u, 3 * v, u * v], -1)[None].requires_grad_()
    poles = torch.tensor([[[0.2, 0.1], [0.3, 0.3], [0.4, 0.5], [0.5, 0.7]]], dtype=torch.float32, requires_grad=True)
    # Module runtime intentionally uses FP32.
    geometry = geometry.float()
    uv, xyz = geometry(grid.float(), torch.tensor([0]), poles)
    expected = torch.stack([2 * uv[..., 0], 3 * uv[..., 1], uv[..., 0] * uv[..., 1]], -1)
    assert torch.allclose(xyz, expected, atol=2e-5)
    xyz.square().mean().backward()
    assert torch.isfinite(poles.grad).all() and poles.grad.abs().sum() > 0
    assert torch.isfinite(grid.grad).all() and grid.grad.abs().sum() > 0


def test_null_scores_receive_supervision_for_missing_targets():
    scores = torch.randn(3, 3, requires_grad=True)
    end = torch.zeros(3, requires_grad=True)
    start = torch.zeros(3, requires_grad=True)
    loss = relation_loss(scores, end, start, torch.tensor([1, -1, 2]), torch.tensor([-1, 0, 2]))
    loss.backward()
    assert torch.isfinite(scores.grad).all()
    assert end.grad[1] < 0 and start.grad[0] < 0


def test_native_loader_and_batch_preserve_seam_uses_and_relation_offsets(tmp_path):
    import numpy as np
    from src.datasets.native_pcurve import NativePcurveDataset
    from src.preprocess.native_pcurve import SCHEMA

    sample = tmp_path / "sample"
    sample.mkdir()
    owner = np.array([0, 0, 0, 0, 1, 2], dtype=np.int64)
    mate = np.array([4, 3, 5, 1, 0, 2], dtype=np.int64)
    nxt = np.array([1, 2, 3, 0, 4, 5], dtype=np.int64)
    uv = np.linspace(0, 1, 6 * 16 * 2, dtype=np.float32).reshape(6, 16, 2)
    rng = np.random.default_rng(3)
    np.savez(
        sample / "data.npz",
        schema=SCHEMA,
        sample_points_faces=rng.normal(size=(3, 16, 16, 6)).astype(np.float32),
        sample_points_lines=rng.normal(size=(6, 16, 6)).astype(np.float32),
        owner=owner,
        mate=mate,
        next=nxt,
        pcurve_uv=uv,
        slot=np.zeros(6, dtype=np.int64),
        side=np.array([0, 0, 0, 1, 1, 1]),
        vertex=np.array([0, 0, 1, 1, 0, 1]),
    )
    split = tmp_path / "split.txt"
    split.write_text("sample\n")
    dataset = NativePcurveDataset(
        "training", {"train_dataset": str(split), "raw_data_root": str(tmp_path), "is_aug": 1}
    )
    first, second = dataset[0], dataset[0]
    batch = dataset.collate_fn([first, second])
    assert torch.equal(batch["pcurve_uv"][:6], torch.from_numpy(uv))
    assert torch.equal(batch["next"][6:], torch.from_numpy(nxt) + 6)
    assert torch.equal(batch["mate"][6:], torch.from_numpy(mate) + 6)
    assert torch.equal(batch["owner"][6:], torch.from_numpy(owner) + 3)
    # The two seam uses survive as distinct records with the SAME owner.
    assert batch["mate"][1].item() == 3 and batch["owner"][1] == batch["owner"][3]
    assert batch["vertex"][6:].min() == 2
    assert batch["attn_mask"][:3, 3:].all()
    assert not batch["attn_mask"][:3, :3].any()


def test_multiple_edge_slot_key_is_symmetric_in_face_uses():
    import numpy as np
    from src.preprocess.native_pcurve import pair_geometry_key

    uv = np.random.default_rng(5).random((4, 16, 2))
    assert pair_geometry_key(0, 1, uv) == pair_geometry_key(1, 0, uv)
    original = sorted([(0, 1), (2, 3)], key=lambda ab: pair_geometry_key(*ab, uv))
    reversed_uses = sorted([(1, 0), (3, 2)], key=lambda ab: pair_geometry_key(*ab, uv))
    assert [frozenset(pair) for pair in original] == [frozenset(pair) for pair in reversed_uses]
