"""Edge-conditioned GAT message contracts, not research-quality conclusions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GATv2Conv

from src.brepnet.models.blocks import EdgeValueGATv2Conv
from src.brepnet.models.vae import AutoEncoder_intersection
from src.brepnet.test.test_intersection_vae import curve_config, encoder_batch


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def graph_inputs():
    generator = torch.Generator().manual_seed(105)
    nodes = torch.randn(5, 4, generator=generator)
    # Node 4 is isolated; an existing self-loop must be replaced as in PyG.
    edges = torch.tensor([[0, 2, 1, 3, 1], [1, 1, 2, 2, 1]])
    attributes = torch.randn(edges.shape[1], 8, generator=generator)
    return nodes, edges, attributes


def migrated_layers(**kwargs):
    torch.manual_seed(82)
    old = GATv2Conv(4, 4, heads=1, edge_dim=8, **kwargs)
    new = EdgeValueGATv2Conv(4, 4, heads=1, edge_dim=8, value_hidden_dim=8, **kwargs)
    report = new.load_state_dict(old.state_dict(), strict=False)
    assert not report.unexpected_keys
    assert set(report.missing_keys) == {
        "value_net.0.weight", "value_net.0.bias", "value_net.2.weight", "value_net.2.bias",
    }
    return old, new


@pytest.mark.parametrize("add_self_loops", [False, True])
@pytest.mark.parametrize("training", [False, True])
def test_zero_initialization_preserves_old_parameters_output_and_attention(add_self_loops, training):
    old, new = migrated_layers(add_self_loops=add_self_loops, dropout=.35)
    old.train(training)
    new.train(training)
    for name, value in old.state_dict().items():
        torch.testing.assert_close(new.state_dict()[name], value, rtol=0, atol=0)
    assert torch.count_nonzero(new.value_net[-1].weight) == 0
    assert torch.count_nonzero(new.value_net[-1].bias) == 0
    inputs = graph_inputs()
    torch.manual_seed(51)
    before, (before_edges, before_alpha) = old(*inputs, return_attention_weights=True)
    torch.manual_seed(51)
    after, (after_edges, after_alpha) = new(*inputs, return_attention_weights=True)
    torch.testing.assert_close(after, before, rtol=0, atol=0)
    torch.testing.assert_close(after_edges, before_edges, rtol=0, atol=0)
    torch.testing.assert_close(after_alpha, before_alpha, rtol=0, atol=0)


def test_added_initialization_does_not_change_later_legacy_rng_initialization():
    torch.manual_seed(71)
    old = GATv2Conv(4, 4, heads=1, edge_dim=8)
    old_following = nn.Linear(4, 4)
    torch.manual_seed(71)
    new = EdgeValueGATv2Conv(4, 4, heads=1, edge_dim=8, value_hidden_dim=8)
    new_following = nn.Linear(4, 4)
    for name, expected in old.state_dict().items():
        torch.testing.assert_close(new.state_dict()[name], expected, rtol=0, atol=0)
    for name, expected in old_following.state_dict().items():
        torch.testing.assert_close(new_following.state_dict()[name], expected, rtol=0, atol=0)


@pytest.mark.parametrize("add_self_loops", [False, True])
def test_empty_connectivity_and_isolated_faces_remain_defined(add_self_loops):
    old, new = migrated_layers(add_self_loops=add_self_loops)
    old.eval()
    new.eval()
    nodes = graph_inputs()[0]
    edges = torch.empty(2, 0, dtype=torch.long)
    attributes = torch.empty(0, 8)
    expected = old(nodes, edges, attributes)
    actual = new(nodes, edges, attributes)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    with torch.no_grad():
        new.value_net[-1].bias.fill_(.25)
    activated = new(nodes, edges, attributes)
    torch.testing.assert_close(activated, expected + (.25 if add_self_loops else 0))


@pytest.mark.parametrize("input_role", ["target", "source", "edge"])
def test_value_reads_raw_roles_and_uses_incoming_mean_self_loop_attributes(input_role):
    layer = EdgeValueGATv2Conv(4, 4, heads=1, edge_dim=8, value_hidden_dim=8).eval()
    # Zero legacy projections make every original message value identical.
    # Zero attention fixes alpha uniformly, independently of edge content.
    with torch.no_grad():
        for parameter in layer.parameters():
            parameter.zero_()
        offset = {"target": 0, "source": 4, "edge": 8}[input_role]
        layer.value_net[0].weight[0, offset] = 1.
        layer.value_net[-1].weight[0, 0] = 1.
    nodes = torch.zeros(4, 4)
    nodes[:, 0] = torch.tensor([1., 2., 3., 4.])
    edges = torch.tensor([[0, 2, 1, 1], [1, 1, 2, 1]])
    attributes = torch.zeros(4, 8)
    attributes[:, 0] = torch.tensor([2., 4., 6., 99.])
    output, (effective_edges, alpha) = layer(nodes, edges, attributes, return_attention_weights=True)
    # Explicit expected graph: discard input self-loop 99, then add all faces.
    expected_edges = torch.tensor([[0, 2, 1, 0, 1, 2, 3], [1, 1, 2, 0, 1, 2, 3]])
    torch.testing.assert_close(effective_edges, expected_edges, rtol=0, atol=0)
    expected_alpha = torch.tensor([1/3, 1/3, 1/2, 1., 1/3, 1/2, 1.])[:, None]
    torch.testing.assert_close(alpha, expected_alpha)
    if input_role == "target":
        expected = F.silu(torch.tensor([1., 2., 3., 4.]))
    elif input_role == "source":
        expected = torch.stack((F.silu(nodes[0, 0]), F.silu(nodes[:3, 0]).mean(),
                                F.silu(nodes[1:3, 0]).mean(), F.silu(nodes[3, 0])))
    else:
        expected = torch.tensor([0., 0., 0., 0.])
        expected[1] = F.silu(torch.tensor([2., 4., 3.])).mean()
        expected[2] = F.silu(torch.tensor(6.))
    torch.testing.assert_close(output[:, 0], expected)
    torch.testing.assert_close(output[:, 1:], torch.zeros(4, 3))


def test_edge_content_changes_values_when_legacy_values_and_attention_are_fixed():
    old, new = migrated_layers()
    old.eval()
    new.eval()
    nodes, edges, attributes = graph_inputs()
    nodes = torch.ones_like(nodes)
    with torch.no_grad():
        old.att.zero_()
        new.att.zero_()
        for parameter in new.value_net.parameters():
            parameter.zero_()
        new.value_net[0].weight[0, 8] = 1.
        new.value_net[-1].weight[0, 0] = 1.
    changed_attributes = attributes.clone()
    changed_attributes[0, 0] += 5
    old_a = old(nodes, edges, attributes)
    old_b = old(nodes, edges, changed_attributes)
    new_a, (_, alpha_a) = new(nodes, edges, attributes, return_attention_weights=True)
    new_b, (_, alpha_b) = new(nodes, edges, changed_attributes, return_attention_weights=True)
    torch.testing.assert_close(old_a, old_b, rtol=0, atol=0)
    torch.testing.assert_close(alpha_a, alpha_b, rtol=0, atol=0)
    assert not torch.allclose(new_a[1], new_b[1])
    torch.testing.assert_close(new_a[[0, 2, 3, 4]], new_b[[0, 2, 3, 4]], rtol=0, atol=0)


def test_active_values_are_permutation_equivariant_and_do_not_mix_packed_objects():
    _, layer = migrated_layers()
    layer.eval()
    with torch.no_grad():
        layer.value_net[-1].weight.normal_(std=.2)
    nodes, edges, attributes = graph_inputs()
    reference = layer(nodes, edges, attributes)
    permutation = torch.tensor([4, 2, 0, 3, 1])
    old_to_new = permutation.argsort()
    edge_permutation = torch.tensor([4, 0, 3, 1, 2])
    permuted = layer(nodes[permutation], old_to_new[edges[:, edge_permutation]], attributes[edge_permutation])
    torch.testing.assert_close(permuted, reference[permutation], atol=1e-6, rtol=1e-6)

    second_nodes = nodes[:3] * 2
    second_edges = torch.tensor([[0, 1], [1, 0]])
    second_attributes = attributes[:2] * 3
    separate = layer(second_nodes, second_edges, second_attributes)
    packed = layer(torch.cat((nodes, second_nodes)), torch.cat((edges, second_edges + len(nodes)), dim=1),
                   torch.cat((attributes, second_attributes)))
    torch.testing.assert_close(packed[:len(nodes)], reference, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(packed[len(nodes):], separate, atol=1e-6, rtol=1e-6)


def test_zero_head_learns_first_then_upstream_value_network_and_edges_receive_gradients():
    _, layer = migrated_layers()
    with torch.no_grad():
        layer.att.zero_()
    for name, parameter in layer.named_parameters():
        parameter.requires_grad_(name.startswith("value_net."))
    nodes, edges, attributes = graph_inputs()
    attributes.requires_grad_()
    optimizer = torch.optim.SGD(layer.value_net.parameters(), lr=.1)
    target = torch.linspace(-.3, .7, nodes.numel()).reshape_as(nodes)
    F.mse_loss(layer(nodes, edges, attributes), target).backward()
    assert layer.value_net[-1].weight.grad.abs().sum() > 0
    assert layer.value_net[-1].bias.grad.abs().sum() > 0
    assert torch.count_nonzero(layer.value_net[0].weight.grad) == 0
    assert torch.count_nonzero(attributes.grad) == 0
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    attributes.grad = None
    F.mse_loss(layer(nodes, edges, attributes), target).backward()
    assert layer.value_net[0].weight.grad.abs().sum() > 0
    assert attributes.grad.abs().sum() > 0
    assert all(torch.isfinite(parameter.grad).all() for parameter in layer.value_net.parameters())


def test_cpu_bfloat16_initial_equivalence_and_active_backward():
    old, new = migrated_layers()
    old.eval()
    new.eval()
    inputs = graph_inputs()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        expected = old(*inputs)
        actual = new(*inputs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    with torch.no_grad():
        new.value_net[-1].weight.normal_(std=.02)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = new(*inputs)
        loss = output.float().square().mean()
    loss.backward()
    assert torch.isfinite(output).all()
    for parameter in new.value_net.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    assert new.value_net[0].weight.grad.abs().sum() > 0


def edge_value_config(**changes):
    return curve_config(encoder_edge_value={"enabled": True, "hidden_dim": 8}, **changes)


def save_conditioned_source(path):
    model = AutoEncoder_intersection(curve_config()).eval()
    # Exercise transfer of learned decoder branches, not only their initial zeros.
    with torch.no_grad():
        model.decoder.refiner.face_delta.weight.normal_(std=.01)
        model.decoder.refiner.edge_delta.weight.normal_(std=.01)
        for parameter in model.decoder.edge_geometry.conditioning.parameters():
            parameter.normal_(std=.01)
    torch.save({
        "epoch": 12, "global_step": 1953,
        "state_dict": {"model." + key: value for key, value in model.state_dict().items()},
    }, path)
    return model


def test_model_migration_retains_every_old_tensor_prediction_and_loss(tmp_path):
    torch.manual_seed(381)
    path = tmp_path / "conditioned.ckpt"
    old = save_conditioned_source(path)
    new = AutoEncoder_intersection(edge_value_config(init_encoder_edge_value_checkpoint=str(path))).eval()
    old_state, new_state = old.state_dict(), new.state_dict()
    added = new_state.keys() - old_state.keys()
    assert len(added) == 4  # tiny_config has one GAT layer, unlike the production five.
    assert all(key.startswith("encoder.graph_face_edge.") and ".value_net." in key for key in added)
    for key, value in old_state.items():
        torch.testing.assert_close(new_state[key], value, rtol=0, atol=0)
    assert new.initialization_report["source_global_step"] == 1953
    assert new.initialization_report["initialized_encoder_edge_value_keys"] == sorted(added)
    assert new.initialization_report["initialized_encoder_edge_value_parameters"] == sum(
        new_state[key].numel() for key in added
    )
    batch = encoder_batch()
    with torch.no_grad():
        before_encoding = old.encode(batch)
        after_encoding = new.encode(batch)
        before_loss, before = old(batch, v_test=True)
        after_loss, after = new(batch, v_test=True)
    for key in ("face_features", "edge_features"):
        torch.testing.assert_close(after_encoding[key], before_encoding[key], rtol=0, atol=0)
    for key, value in before_loss.items():
        torch.testing.assert_close(after_loss[key], value, rtol=0, atol=0)
    for before_item, after_item in zip(before["items"], after["items"]):
        for key, value in before_item.items():
            np.testing.assert_array_equal(after_item[key], value)


def test_model_conversion_does_not_shift_constructor_rng_or_legacy_initialization():
    torch.manual_seed(67)
    old = AutoEncoder_intersection(curve_config())
    old_random = torch.rand(12)
    torch.manual_seed(67)
    new = AutoEncoder_intersection(edge_value_config())
    new_random = torch.rand(12)
    for name, expected in old.state_dict().items():
        torch.testing.assert_close(new.state_dict()[name], expected, rtol=0, atol=0)
    torch.testing.assert_close(new_random, old_random, rtol=0, atol=0)


@pytest.mark.parametrize("corruption", [
    "missing_encoder", "missing_decoder", "extra", "shape", "duplicate_prefix", "already_edge_value",
])
def test_model_migration_rejects_any_incomplete_or_modified_source(tmp_path, corruption):
    path = tmp_path / "source.ckpt"
    save_conditioned_source(path)
    saved = torch.load(path, weights_only=False)
    state = saved["state_dict"]
    if corruption == "missing_encoder":
        state.pop(next(key for key in state if key.startswith("model.encoder.graph_face_edge.")))
    elif corruption == "missing_decoder":
        state.pop("model.decoder.classifier.bias")
    elif corruption == "extra":
        state["model.unexpected.weight"] = torch.zeros(1)
    elif corruption == "shape":
        state["model.decoder.classifier.bias"] = torch.zeros(2)
    elif corruption == "duplicate_prefix":
        state["decoder.classifier.bias"] = state["model.decoder.classifier.bias"]
    else:
        state["model.encoder.graph_face_edge.0.value_net.0.bias"] = torch.zeros(8)
    torch.save(saved, path)
    with pytest.raises((ValueError, RuntimeError), match="complete conditioned|Ambiguous|size mismatch"):
        AutoEncoder_intersection(edge_value_config(init_encoder_edge_value_checkpoint=str(path)))


def test_model_training_and_strict_full_checkpoint_restore(tmp_path):
    path = tmp_path / "conditioned.ckpt"
    save_conditioned_source(path)
    model = AutoEncoder_intersection(edge_value_config(init_encoder_edge_value_checkpoint=str(path))).train()
    assert model.encoder.training and model.posterior.training and model.decoder.training
    assert all(parameter.requires_grad for parameter in model.parameters())
    batch = encoder_batch()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses, _ = model(batch)
    assert all(torch.isfinite(value).all() for value in losses.values())
    losses["total_loss"].backward()
    layer = next(layer for layer in model.encoder.graph_face_edge if isinstance(layer, EdgeValueGATv2Conv))
    assert layer.value_net[-1].weight.grad.abs().sum() > 0
    assert model.encoder.edge_coords[0].weight.grad.abs().sum() > 0
    assert model.posterior.projection[0].weight.grad.abs().sum() > 0
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    model(batch)[0]["total_loss"].backward()
    assert layer.value_net[0].weight.grad.abs().sum() > 0

    full_path = tmp_path / "edge_value.ckpt"
    torch.save(model.state_dict(), full_path)
    restored = AutoEncoder_intersection(edge_value_config(checkpoint=str(full_path))).eval()
    model.eval()
    with torch.no_grad():
        before_loss, before = model(batch, v_test=True)
        after_loss, after = restored(batch, v_test=True)
    torch.testing.assert_close(after_loss["total_loss"], before_loss["total_loss"], rtol=0, atol=0)
    for before_item, after_item in zip(before["items"], after["items"]):
        for key, value in before_item.items():
            np.testing.assert_array_equal(after_item[key], value)
    with pytest.raises(ValueError, match="complete conditioned"):
        AutoEncoder_intersection(edge_value_config(init_encoder_edge_value_checkpoint=str(full_path)))
    state = torch.load(full_path, weights_only=False)
    state.pop("decoder.classifier.bias")
    torch.save(state, full_path)
    with pytest.raises(RuntimeError, match="Missing key"):
        AutoEncoder_intersection(edge_value_config(checkpoint=str(full_path)))


def test_training_rejects_conflicting_initialization():
    with pytest.raises(ValueError, match="only one"):
        AutoEncoder_intersection(edge_value_config(checkpoint="full.ckpt", init_encoder_edge_value_checkpoint="source.ckpt"))
    with pytest.raises(ValueError, match="Resume"):
        AutoEncoder_intersection(edge_value_config(restoring_checkpoint=True, init_encoder_edge_value_checkpoint="source.ckpt"))


def test_trainer_resume_clears_new_migration_source(tmp_path, monkeypatch):
    from hydra import compose, initialize_config_dir
    from src.brepnet import train

    config_dir = str(Path(__file__).resolve().parents[3] / "configs")
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        cfg = compose(config_name="train_vae_encoder_edge_value", overrides=[
            "dataset.raw_data_root=unused", "dataset.train_dataset=unused",
            "dataset.val_dataset=unused", "dataset.test_dataset=unused",
            f"trainer.output_dir={tmp_path}", "trainer.resume_from_checkpoint=full.ckpt",
            "model.init_encoder_edge_value_checkpoint=missing-source.ckpt",
        ])
    captured = {}

    def build_model(model_cfg, condition):
        captured.update(dict(model_cfg))
        return nn.Linear(2, 2)

    monkeypatch.setattr(train, "build_model", build_model)
    monkeypatch.setattr(train, "_load_dataset_class", lambda config: object)
    train.UnifiedTrainer(cfg)
    assert captured["restoring_checkpoint"] is True
    for key in ("checkpoint", "init_legacy_checkpoint", "init_refinement_checkpoint",
                "init_curve_feedback_checkpoint", "init_encoder_edge_value_checkpoint"):
        assert captured[key] is None
