"""Topology roles and migration/gradient ownership, not reconstruction quality."""
from copy import deepcopy

import pytest
import torch

from src.datasets.coedge import collate_coedge_topology
from src.models.vae import AutoEncoder_intersection
from src.models.vae.coedge import CoedgeFeatureFusion
from tests.test_curve_head_roles import model_config
from tests.test_intersection_vae import encoder_batch
from tests.test_uv_boundary_model import add_targets
from tools.migrate_checkpoint import migrate


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def topology(count):
    indices = torch.arange(count)
    return {"coedge_neighbors": torch.stack((indices.roll(1), indices.roll(-1), indices), 1),
            "coedge_orientation": torch.ones(count, 1)}


def test_roles_are_distinct_and_packing_does_not_mix_objects():
    module = CoedgeFeatureFusion(2, 4)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.zero_()
        # Select the previous role's first geometric coordinate, not next/mate.
        module.update[0].weight[0, 3] = 1.
        module.update[-1].weight[0, 0] = 1.
    edges = torch.tensor([[1., 0.], [2., 0.], [4., 0.]])
    first = topology(3)
    output = module(edges, first)
    swapped = {**first, "coedge_neighbors": first["coedge_neighbors"][:, [1, 0, 2]]}
    assert not torch.allclose(module(edges, swapped), output)
    second = topology(2)
    empty = topology(0)
    packed = collate_coedge_topology([first, empty, second], [3, 0, 2])
    actual = module(torch.cat((edges, edges[:2] * 10)), packed)
    torch.testing.assert_close(actual[:3], output)
    torch.testing.assert_close(actual[3:], module(edges[:2] * 10, second))
    assert torch.equal(first["coedge_neighbors"], topology(3)["coedge_neighbors"])
    assert module(edges[:0], empty).shape == (0, 2)
    # Renumbering raw curves must commute with the fixed-role exchange.
    order = torch.tensor([2, 0, 1])
    permuted = {"coedge_neighbors": order.argsort()[first["coedge_neighbors"][order]],
                "coedge_orientation": first["coedge_orientation"][order]}
    torch.testing.assert_close(module(edges[order], permuted), output[order])


def test_direction_is_available_without_reversing_teacher_features():
    module = CoedgeFeatureFusion(2, 4)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.zero_()
        module.update[0].weight[0, 5] = 1.  # previous role's orientation
        module.update[-1].weight[0, 0] = 1.
    edges = torch.zeros(3, 2)
    labels = topology(3)
    before = module(edges, labels)
    labels["coedge_orientation"][0] = -1
    after = module(edges, labels)
    assert not torch.equal(before[1], after[1])
    torch.testing.assert_close(before[[0, 2]], after[[0, 2]])


def test_strict_migration_preserves_old_model_and_teacher_then_trains_new_path():
    config = model_config(fp32=True)
    config["loss_weights"]["feature_weight"] = 0.
    torch.manual_seed(61)
    baseline = AutoEncoder_intersection(config).eval()
    checkpoint = {"state_dict": {"model." + k: v for k, v in baseline.state_dict().items()},
                  "hyper_parameters": {"model": deepcopy(config)},
                  "optimizer_states": [{}], "loops": {}, "global_step": 12}
    checkpoint["hyper_parameters"]["model"]["checkpoint"] = "old-source.ckpt"
    rng = torch.get_rng_state()
    migrate(checkpoint, encoder_coedge_hidden_dim=12)
    assert torch.equal(rng, torch.get_rng_state())
    migrated_config = checkpoint["hyper_parameters"]["model"]
    assert migrated_config["checkpoint"] is None
    assert "optimizer_states" not in checkpoint and "loops" not in checkpoint
    candidate = AutoEncoder_intersection(migrated_config).eval()
    candidate.load_state_dict({k.removeprefix("model."): v for k, v in checkpoint["state_dict"].items()}, strict=True)
    for key, expected in baseline.state_dict().items():
        torch.testing.assert_close(candidate.state_dict()[key], expected, atol=0, rtol=0)
    assert "already applied" in migrate(checkpoint, encoder_coedge_hidden_dim=12)[-1]
    with pytest.raises(ValueError, match="already migrated"):
        migrate(checkpoint, encoder_coedge_hidden_dim=16)
    batch = add_targets(encoder_batch())
    batch.update(topology(len(batch["edge_points"])))
    with torch.no_grad():
        old_encoding, new_encoding = baseline.encode(batch), candidate.encode(batch)
        for key in old_encoding:
            torch.testing.assert_close(new_encoding[key], old_encoding[key], atol=0, rtol=0)
        old_losses, _ = baseline(batch, v_test=True)
        new_losses, _ = candidate(batch, v_test=True)
        for key in old_losses:
            torch.testing.assert_close(new_losses[key], old_losses[key], atol=0, rtol=0)
    optimizer = torch.optim.Adam(candidate.encoder.coedge_fusion.parameters(), lr=1e-3)
    for iteration in range(2):
        optimizer.zero_grad(set_to_none=True)
        # Isolate the route through GAT into the encoded face posterior input.
        loss = candidate.encode(batch)["face_features"].square().sum()
        loss.backward()
        module = candidate.encoder.coedge_fusion
        assert module.update[-1].weight.grad.abs().sum() > 0
        if iteration:
            assert module.update[0].weight.grad.abs().sum() > 0
        assert all(torch.isfinite(p.grad).all() for p in module.parameters())
        optimizer.step()
    trained = candidate.encode(batch)
    torch.testing.assert_close(trained["edge_features"], old_encoding["edge_features"], atol=0, rtol=0)
    assert not torch.equal(trained["face_features"], old_encoding["face_features"])


def test_optional_module_does_not_shift_existing_initialization_rng():
    config = model_config()
    torch.manual_seed(73)
    baseline = AutoEncoder_intersection(config)
    expected_rng = torch.get_rng_state()
    torch.manual_seed(73)
    candidate = AutoEncoder_intersection({**config, "encoder_coedge": {"enabled": True, "hidden_dim": 12}})
    assert torch.equal(torch.get_rng_state(), expected_rng)
    for key, expected in baseline.state_dict().items():
        torch.testing.assert_close(candidate.state_dict()[key], expected, atol=0, rtol=0)
