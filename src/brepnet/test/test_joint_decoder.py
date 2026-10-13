"""Contract tests, not evidence of a reconstruction-quality improvement."""

import numpy as np
import pytest
import torch

from src.brepnet.models.blocks import JointBoundaryDecoder
from src.brepnet.models.vae import boundary_costs, joint_boundary_loss, match_boundaries


LOSS_WEIGHTS = dict(face=1.0, curve=1.0, existence=0.1, ownership=0.1, consistency=0.1)
MATCH_WEIGHTS = dict(curve=1.0, ownership=0.25)


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def small_decoder(**kwargs):
    return JointBoundaryDecoder(8, hidden_dim=16, num_queries=4, num_rounds=2, num_heads=2, **kwargs)


def empty_target():
    return {"curves": torch.empty(0, 16, 3), "face_pairs": torch.empty(0, 2, dtype=torch.long)}


def test_batch_isolation_and_face_permutation_equivariance():
    torch.manual_seed(11)
    model = small_decoder().eval()
    z = torch.randn(5, 8)
    batch = model(z, torch.tensor([2, 3]))
    alone = model(z[:2], torch.tensor([2]))
    torch.testing.assert_close(batch["face_points"][0, :2], alone["face_points"][0], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(batch["curves"][0], alone["curves"][0], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(batch["ownership_logits"][0, :, :, :2], alone["ownership_logits"][0], atol=1e-6, rtol=1e-5)
    assert batch["ownership_logits"][0, :, :, 2].isneginf().all()
    perm = torch.tensor([1, 0])
    reordered = model(z[:2][perm], torch.tensor([2]))
    torch.testing.assert_close(reordered["face_points"][0], alone["face_points"][0, perm], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(reordered["curves"], alone["curves"], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(reordered["ownership_logits"], alone["ownership_logits"][..., perm], atol=1e-6, rtol=1e-5)


def test_edge_queries_really_change_faces_only_with_feedback_enabled():
    torch.manual_seed(12)
    z = torch.randn(3, 8)
    for enabled in (True, False):
        model = small_decoder(edge_to_face=enabled).eval()
        before = model(z, torch.tensor([3]))["face_points"].detach()
        with torch.no_grad():
            model.edge_queries.add_(torch.randn_like(model.edge_queries) * 2)
        after = model(z, torch.tensor([3]))["face_points"].detach()
        assert bool(torch.allclose(before, after)) is (not enabled)


def test_reversal_side_swap_and_target_permutation_preserve_matching():
    curves = torch.randn(3, 16, 3)
    logits = torch.randn(3, 2, 4)
    target = {"curves": curves[[2, 0]].clone(), "face_pairs": torch.tensor([[0, 1], [2, 3]])}
    changed = {"curves": target["curves"].flip(0).flip(1), "face_pairs": target["face_pairs"].flip(0).flip(1)}
    cost1 = boundary_costs(curves, logits, target)
    cost2 = boundary_costs(curves, logits, changed)
    for a, b in zip(cost1, cost2):
        torch.testing.assert_close(a, b.flip(1))
    rows1, cols1 = match_boundaries(curves, logits, target, 1.0, 0.25)
    rows2, cols2 = match_boundaries(curves, logits, changed, 1.0, 0.25)
    torch.testing.assert_close(rows1, rows2)
    torch.testing.assert_close(cols1, 1 - cols2)


def test_closed_curve_matching_accepts_cyclic_origin_and_reversal():
    unique = torch.randn(15, 3)
    curve = torch.cat((unique, unique[:1]))
    shifted = unique.flip(0).roll(7, dims=0)
    target = {
        "curves": torch.cat((shifted, shifted[:1]))[None],
        "face_pairs": torch.tensor([[0, 1]]), "is_closed": torch.tensor([True]),
    }
    geom, _ = boundary_costs(curve[None], torch.zeros(1, 2, 2), target)
    assert geom.item() == 0


def test_empty_targets_are_finite_and_capacity_overflow_is_explicit():
    model = small_decoder()
    state = model(torch.randn(3, 8), torch.tensor([3]))
    losses = joint_boundary_loss(state, [empty_target()], torch.randn(3, 16, 16, 3), LOSS_WEIGHTS, MATCH_WEIGHTS)
    assert all(torch.isfinite(value) for value in losses.values())
    losses["total_loss"].backward()
    assert model.existence_head.weight.grad.abs().sum() > 0
    target = {"curves": torch.randn(5, 16, 3), "face_pairs": torch.tensor([[0, 1]] * 5)}
    with pytest.raises(ValueError, match="exceed query capacity"):
        joint_boundary_loss(state, [target], torch.randn(3, 16, 16, 3), LOSS_WEIGHTS, MATCH_WEIGHTS)
    with pytest.raises(ValueError, match="nonempty"):
        model(torch.empty(0, 8), torch.tensor([0]))


def test_geometry_and_incidence_receive_finite_gradients():
    torch.manual_seed(14)
    model = small_decoder()
    z = torch.randn(3, 8, requires_grad=True)
    state = model(z, torch.tensor([3]))
    target = {"curves": torch.randn(2, 16, 3), "face_pairs": torch.tensor([[0, 1], [1, 2]])}
    losses = joint_boundary_loss(state, [target], torch.randn(3, 16, 16, 3), LOSS_WEIGHTS, MATCH_WEIGHTS)
    losses["total_loss"].backward()
    parameters = [model.rounds[-1].side_query.weight, model.curve_head[-1].weight, model.face_head[-1].weight, model.edge_queries, z]
    for parameter in parameters:
        assert parameter.grad is not None and parameter.grad.isfinite().all()
        assert parameter.grad.abs().sum() > 0


def small_vae_config(**extra):
    return dict(
        norm="layer", in_channels=6, hidden_channels=16, latent_channels=2,
        bottleneck_dim=16, nhead=2, num_encoder_layers=1, num_decoder_layers=1,
        num_gat_layers=1, intersection_dim=16, intersection_layers=1,
        primitive_blocks_per_stage=1, point_decoder_blocks_per_stage=1,
        face_bbox_decoder_blocks=1, edge_bbox_decoder_blocks=1,
        gaussian_weights=1e-6, joint_hidden_dim=16, joint_num_queries=4,
        joint_num_rounds=2, joint_num_heads=2, **extra,
    )


def synthetic_encoder_batch():
    faces = torch.randn(5, 16, 16, 6) * 0.2
    curve = torch.randn(16, 6) * 0.2
    counts = torch.tensor([2, 3])
    object_ids = torch.repeat_interleave(torch.arange(2), counts)
    return {
        "face_points": faces,
        "edge_points": torch.stack((curve, curve.flip(0))),
        "face_counts": counts,
        "edge_face_connectivity": torch.tensor([[0, 0, 1], [1, 1, 0]]),
        "attn_mask": object_ids[:, None] != object_ids[None, :],
        "boundary_targets": [
            {"curves": curve[None, :, :3], "face_pairs": torch.tensor([[0, 1]])},
            empty_target(),
        ],
        "v_prefix": ["first", "second"],
    }


def test_legacy_encoder_migration_cache_identity_frozen_eval_and_strict_reload(tmp_path):
    from src.brepnet.models.vae import AutoEncoder_joint
    from src.brepnet.models.vae import AutoEncoder_light

    torch.manual_seed(15)
    legacy = AutoEncoder_light(small_vae_config()).eval()
    legacy_path = tmp_path / "legacy.ckpt"
    torch.save({"state_dict": {"model." + key: value for key, value in legacy.state_dict().items()}}, legacy_path)
    joint = AutoEncoder_joint(small_vae_config(init_encoder_checkpoint=legacy_path)).train()
    assert joint.decoder_joint.training
    for name in joint.ENCODER_MODULES:
        assert not getattr(joint, name).training
        assert all(not p.requires_grad for p in getattr(joint, name).parameters())
    batch = synthetic_encoder_batch()
    legacy_features = legacy.encode(batch, True)["face_features"]
    joint_features = joint.encode(batch, True)["face_features"]
    torch.testing.assert_close(legacy_features, joint_features, rtol=0, atol=0)
    legacy_stats = legacy.sample(legacy_features, True)[2]
    joint_stats = joint.sample(joint_features, True)[2]
    torch.testing.assert_close(legacy_stats, joint_stats, rtol=0, atol=0)
    loss, data = joint(batch, v_test=True)
    assert len(data["items"]) == 2
    assert data["items"][0]["pred_face"].shape[0] == 2
    assert data["items"][1]["pred_face"].shape[0] == 3
    loss["total_loss"].backward()
    assert all(parameter.grad is None for name in joint.ENCODER_MODULES for parameter in getattr(joint, name).parameters())
    checkpoint = tmp_path / "joint.ckpt"
    torch.save(joint.state_dict(), checkpoint)
    restored = AutoEncoder_joint(small_vae_config(checkpoint=checkpoint)).eval()
    original_state = joint.decode_state(torch.randn(2, 8))
    assert original_state["curves"].shape == (1, 4, 16, 3)
    for key, value in joint.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[key])
    broken = dict(joint.state_dict())
    broken.pop("decoder_joint.edge_queries")
    broken_path = tmp_path / "broken.ckpt"
    torch.save(broken, broken_path)
    with pytest.raises(RuntimeError, match="Missing key"):
        AutoEncoder_joint(small_vae_config(checkpoint=broken_path))
    broken_legacy = dict(legacy.state_dict())
    broken_legacy["gaussian_proj.0.weight"] = torch.zeros(1, 1)
    torch.save(broken_legacy, broken_path)
    with pytest.raises(ValueError, match="shape_mismatch"):
        AutoEncoder_joint(small_vae_config(init_encoder_checkpoint=broken_path))


def test_finetuning_backpropagates_encoder_and_export_uses_predictions_only():
    from src.brepnet.models.vae import AutoEncoder_joint

    torch.manual_seed(16)
    model = AutoEncoder_joint(small_vae_config(freeze_encoder=False, joint_existence_threshold=0.0)).train()
    loss, data = model(synthetic_encoder_batch(), v_test=True)
    loss["total_loss"].backward()
    assert model.gaussian_proj[0].weight.grad.abs().sum() > 0
    for item in data["items"]:
        assert len(item["pred_edge"]) == 4  # including the object with NO GT boundaries
        assert item["representation"].item() == "shared_boundary_v1"
        assert item["pred_edge_confidence"].shape == (4,)
        assert item["pred_edge_face_connectivity"].dtype == np.int64
    state = model.decode_state(torch.randn(2, 8, requires_grad=True))
    state["curves"].sum().backward()  # tensor-native API retains latent derivative
    assert "inter" not in model._modules and "face_attn2" not in model._modules


def test_same_face_and_duplicate_predictions_are_preserved_raw():
    from src.brepnet.models.vae import AutoEncoder_joint

    model = AutoEncoder_joint(small_vae_config(joint_existence_threshold=0.0)).eval()
    z = torch.randn(2, 8)
    state = model.decode_state(z)
    state["ownership_logits"] = torch.tensor([[[[9.0, 0.0], [9.0, 0.0]]] * 4])
    exported = model.export_items(state, z)[0]
    assert len(exported["pred_edge"]) == 4
    assert (exported["pred_edge_face_connectivity"][:, 1:] == 0).all()
    assert exported["pred_face_adj"][0, 0]


def test_factories_and_frozen_decoder_keep_latent_derivatives():
    from omegaconf import OmegaConf
    from src.brepnet.models import build_model
    from src.brepnet.models.vae import build_autoencoder

    config = small_vae_config(name="AutoEncoder_joint", mode="frozen_inference")
    model = build_model(OmegaConf.create(config))
    assert type(build_autoencoder(config)) is type(model)
    model.train()
    assert not model.training and not model.decoder_joint.training
    assert all(not parameter.requires_grad for parameter in model.parameters())
    z = torch.randn(2, 8, requires_grad=True)
    state = model.decode_state(z)
    (state["curves"].square().mean() + state["face_points"].square().mean()).backward()
    assert z.grad is not None and z.grad.isfinite().all() and z.grad.abs().sum() > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_bfloat16_forward_backward_and_numpy_export():
    from src.brepnet.models.vae import AutoEncoder_joint

    def to_device(value):
        if isinstance(value, torch.Tensor):
            return value.cuda()
        if isinstance(value, dict):
            return {key: to_device(item) for key, item in value.items()}
        if isinstance(value, list):
            return [to_device(item) for item in value]
        return value

    model = AutoEncoder_joint(small_vae_config()).cuda().train()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss, data = model(to_device(synthetic_encoder_batch()), v_test=True)
    loss["total_loss"].backward()
    assert all(value.isfinite() for value in loss.values())
    assert all(parameter.grad is None or parameter.grad.isfinite().all() for parameter in model.parameters())
    assert data["items"][0]["cached_latent_stats"].dtype == np.float32
    assert data["items"][0]["pred_edge"].dtype == np.float32
