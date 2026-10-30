import math

import torch

from src.brepnet.diffusion.topology import (
    OracleTopologySelfAttentionMask,
    build_soft_topology_attention_bias,
)


def test_log_multiplier_maps_binary_adjacency_to_one_and_two() -> None:
    adjacency = torch.tensor([[[0.0, 1.0], [1.0, 0.0]]])
    bias = build_soft_topology_attention_bias(
        adjacency,
        torch.tensor([999]),
        num_train_timesteps=1000,
        timestep_weight="linear_high_noise",
        parameterization="log_multiplier",
        scale=2.0,
        max_multiplier=2.0,
    )

    assert torch.allclose(torch.diagonal(bias, dim1=1, dim2=2), torch.zeros(1, 2))
    assert torch.allclose(bias[0, 0, 1], torch.tensor(math.log(2.0)))
    assert torch.allclose(bias[0, 1, 0], torch.tensor(math.log(2.0)))


def test_log_multiplier_doubles_attention_odds() -> None:
    base_logits = torch.zeros(2)
    topology_bias = torch.tensor([0.0, math.log(2.0)])
    attention = torch.softmax(base_logits + topology_bias, dim=0)

    assert torch.allclose(attention[1] / attention[0], torch.tensor(2.0))


def test_additive_parameterization_preserves_historical_schedule() -> None:
    adjacency = torch.ones(1, 1, 1)
    bias = build_soft_topology_attention_bias(
        adjacency,
        torch.tensor([999]),
        num_train_timesteps=1000,
        timestep_weight="linear_high_noise",
        parameterization="additive",
        scale=2.0,
        max_multiplier=2.0,
    )

    assert torch.allclose(bias, torch.tensor([[[1.998]]]))


def test_hard_mask_still_allows_only_self_and_adjacent_faces() -> None:
    module = OracleTopologySelfAttentionMask(
        {
            "enabled": True,
            "mode": "hard_mask",
            "timestep_weight": "linear_high_noise",
            "parameterization": "log_multiplier",
            "scale": 2.0,
            "max_multiplier": 2.0,
        },
        num_train_timesteps=1000,
    )
    adjacency = torch.tensor(
        [[[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]
    )

    mask = module(adjacency, torch.tensor([999]))

    expected = torch.tensor(
        [[[False, False, True], [False, False, True], [True, True, False]]]
    )
    assert torch.equal(mask, expected)


def test_log_multiplier_bias_backpropagates_to_edge_logits() -> None:
    edge_logits = torch.tensor([[[0.0, 0.2], [0.2, 0.0]]], requires_grad=True)
    probability = torch.sigmoid(edge_logits)
    bias = build_soft_topology_attention_bias(
        probability,
        torch.tensor([999]),
        num_train_timesteps=1000,
        timestep_weight="linear_high_noise",
        parameterization="log_multiplier",
        scale=2.0,
        max_multiplier=2.0,
    )

    loss = bias[0, 0, 1] + bias[0, 1, 0]
    loss.backward()

    assert edge_logits.grad is not None
    assert torch.isfinite(edge_logits.grad).all()
    assert edge_logits.grad[0, 0, 1] > 0
