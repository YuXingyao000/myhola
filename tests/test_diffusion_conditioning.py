"""逐层图像条件网络的结构、信息流与梯度合同，不评价重建效果。"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch
from torch import nn

from src.models.diffusion.denoiser import BRepDenoiser
from src.models.blocks import sincos_embedding
from tests.test_diffusion_refactor import (
    cpu_threads,
    tiny_batch,
    tiny_config,
    tiny_dependencies,
)


def make_denoiser(condition_dim: int | None = 80) -> BRepDenoiser:
    cfg, _ = tiny_config()
    torch.manual_seed(101)
    return BRepDenoiser(4, cfg["denoiser"], condition_dim=condition_dim)


def denoising_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(103)
    return (
        torch.randn(2, 4, 4, generator=generator),
        torch.tensor([7, 83]),
        torch.randn(2, 5, 80, generator=generator),
    )


def test_layerwise_config_preserves_baseline_data_and_training_budget():
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    config_dir = Path(__file__).resolve().parents[1] / "configs"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        baseline = compose(config_name="train_diffusion_baseline_real")
        layerwise = compose(config_name="train_diffusion_layerwise_real")
    assert OmegaConf.to_container(layerwise.dataset, resolve=True) == OmegaConf.to_container(baseline.dataset, resolve=True)
    assert OmegaConf.to_container(layerwise.condition, resolve=True) == OmegaConf.to_container(baseline.condition, resolve=True)
    assert layerwise.trainer.exp_name == "diffusion_layerwise_real"
    for key in ("batch_size", "learning_rate", "devices", "max_steps"):
        assert layerwise.trainer[key] == baseline.trainer[key]
    assert layerwise.model.denoiser.conditioning == "layerwise_cross_attention"
    assert "condition_fuser" not in layerwise.model


def test_default_architecture_keeps_all_24_encoder_blocks_and_adds_24_image_blocks():
    from omegaconf import OmegaConf

    cfg_path = Path(__file__).resolve().parents[1] / "configs/model/diffusion.yaml"
    cfg = OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
    assert "condition_fuser" not in cfg
    assert cfg["denoiser"]["conditioning"] == "layerwise_cross_attention"
    # Inspect the full-size network without allocating its parameter storage.
    with torch.device("meta"):
        model = BRepDenoiser(32, cfg["denoiser"], condition_dim=1024)
    assert len(model.backbone.layers) == len(model.cross_attention_layers) == 24
    for encoder, image_layer in zip(model.backbone.layers, model.cross_attention_layers):
        assert isinstance(encoder, nn.TransformerEncoderLayer)
        assert encoder.self_attn.embed_dim == 768
        assert encoder.linear1.out_features == 2048
        assert encoder.linear2.out_features == 768
        assert isinstance(encoder.norm1, nn.LayerNorm)
        assert isinstance(encoder.norm2, nn.LayerNorm)
        assert image_layer.attention.embed_dim == 768
        assert image_layer.attention.kdim == image_layer.attention.vdim == 1024
        assert image_layer.attention.num_heads == 12
    assert isinstance(model.backbone.norm, nn.LayerNorm)


def test_each_complete_encoder_is_followed_by_independent_image_attention():
    model = make_denoiser().eval()
    noisy, timesteps, memory = denoising_inputs()
    events, first_encoder_input, memory_inputs = [], [], []

    def record_attention_memory(_module, args, kwargs):
        key = kwargs["key"] if "key" in kwargs else args[1]
        value = kwargs["value"] if "value" in kwargs else args[2]
        memory_inputs.append((key.detach().clone(), value.detach().clone()))

    handles = [model.backbone.layers[0].register_forward_pre_hook(
        lambda _module, args: first_encoder_input.append(args[0].detach().clone())
    )]
    for index, (encoder, image_layer) in enumerate(zip(model.backbone.layers, model.cross_attention_layers)):
        handles.append(encoder.register_forward_hook(
            lambda _module, _args, _output, index=index: events.append(f"encoder_{index}")
        ))
        handles.append(image_layer.register_forward_hook(
            lambda _module, _args, _output, index=index: events.append(f"image_{index}")
        ))
        handles.append(image_layer.attention.register_forward_pre_hook(record_attention_memory, with_kwargs=True))
    handles.append(model.backbone.norm.register_forward_hook(
        lambda _module, _args, _output: events.append("final_norm")
    ))
    with torch.no_grad():
        output = model(noisy, timesteps, memory)
        expected_input = model.input_projection(noisy) + model.time_embedding(
            sincos_embedding(timesteps, model.hidden_dim)
        )[:, None]
    for handle in handles:
        handle.remove()

    assert output.shape == noisy.shape
    assert events == ["encoder_0", "image_0", "encoder_1", "image_1", "encoder_2", "image_2", "final_norm"]
    torch.testing.assert_close(first_encoder_input[0], expected_input)
    assert len(memory_inputs) == 3
    for key, value in memory_inputs:
        torch.testing.assert_close(key, memory, rtol=0, atol=0)
        torch.testing.assert_close(value, memory, rtol=0, atol=0)
    for name in ("q_proj_weight", "k_proj_weight", "v_proj_weight"):
        parameters = [getattr(layer.attention, name) for layer in model.cross_attention_layers]
        assert len({id(param) for param in parameters}) == 3
        assert len({param.data_ptr() for param in parameters}) == 3


def test_each_image_attention_layer_and_memory_receive_finite_nonzero_gradients():
    model = make_denoiser().train()
    noisy, timesteps, memory = denoising_inputs()
    noisy.requires_grad_()
    memory.requires_grad_()
    torch.manual_seed(107)
    model(noisy, timesteps, memory).square().mean().backward()
    for layer in model.cross_attention_layers:
        for parameter in (
            layer.query_norm.weight,
            layer.attention.q_proj_weight,
            layer.attention.k_proj_weight,
            layer.attention.v_proj_weight,
            layer.attention.out_proj.weight,
        ):
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()
            assert parameter.grad.abs().sum() > 0
    for gradient in (noisy.grad, memory.grad):
        assert gradient is not None and torch.isfinite(gradient).all()
        assert (gradient.flatten(1).abs().sum(dim=1) > 0).all()


def test_prediction_depends_on_image_and_time_before_first_cross_attention():
    model = make_denoiser().eval()
    noisy, timesteps, memory = denoising_inputs()
    image_queries = []
    handle = model.cross_attention_layers[0].register_forward_pre_hook(
        lambda _module, args: image_queries.append(args[0].detach().clone())
    )
    with torch.no_grad():
        original = model(noisy, timesteps, memory)
        changed_time = model(noisy, timesteps + 41, memory)
        changed_image = model(noisy, timesteps, memory.flip(0))
    handle.remove()
    assert not torch.allclose(original, changed_time, atol=1e-6, rtol=1e-6)
    assert not torch.allclose(original, changed_image, atol=1e-6, rtol=1e-6)
    assert not torch.allclose(image_queries[0], image_queries[1], atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("mask_type", ["none", "soft", "hard"])
def test_face_permutation_equivariance_with_correspondingly_permuted_topology(mask_type):
    model = make_denoiser().eval()
    noisy, timesteps, memory = denoising_inputs()
    permutation = torch.tensor([2, 0, 3, 1])
    mask = None
    if mask_type == "soft":
        base = torch.arange(16, dtype=torch.float32).reshape(4, 4) / 16
        mask = (base + base.T)[None].repeat(2, 1, 1)
    elif mask_type == "hard":
        mask = ~torch.eye(4, dtype=torch.bool)[None].repeat(2, 1, 1)
        mask[:, 0, 1] = mask[:, 1, 0] = False
    permuted_mask = None if mask is None else mask[:, permutation][:, :, permutation]
    with torch.no_grad():
        original = model(noisy, timesteps, memory, self_attention_mask=mask)
        permuted = model(noisy[:, permutation], timesteps, memory, self_attention_mask=permuted_mask)
    torch.testing.assert_close(permuted, original[:, permutation], rtol=2e-5, atol=2e-6)


def test_singleton_view_axis_and_three_dimensional_memory_are_equivalent():
    model = make_denoiser().eval()
    noisy, timesteps, memory = denoising_inputs()
    with torch.no_grad():
        direct = model(noisy, timesteps, memory)
        singleton_view = model(noisy, timesteps, memory[:, None])
    torch.testing.assert_close(direct, singleton_view, rtol=0, atol=0)


def test_float_topology_bias_has_same_semantics_in_grad_and_inference_modes():
    model = make_denoiser().eval()
    noisy, timesteps, memory = denoising_inputs()
    bias = (torch.ones(4, 4) - torch.eye(4))[None].repeat(2, 1, 1) * 0.5
    reference = model(noisy, timesteps, memory, self_attention_mask=bias).detach()
    with torch.no_grad():
        inference = model(noisy, timesteps, memory, self_attention_mask=bias)
    assert torch.isfinite(inference).all()
    torch.testing.assert_close(inference, reference, rtol=2e-5, atol=2e-6)

    # Native EncoderLayer with grad enabled bypasses its faulty fused path.
    layer = model.backbone.layers[0]
    hidden = model.input_projection(noisy).detach().requires_grad_()
    layer_bias = bias[:, None].expand(-1, layer.self_attn.num_heads, -1, -1)
    layer_bias = layer_bias.reshape(-1, 4, 4).clone().requires_grad_()
    with torch.enable_grad():
        native = layer(hidden, src_mask=layer_bias)
    with torch.no_grad():
        explicit = model._encoder_with_bias(layer, hidden, layer_bias)
    torch.testing.assert_close(explicit, native.detach(), rtol=2e-5, atol=2e-6)
    model._encoder_with_bias(layer, hidden, layer_bias).square().mean().backward()
    assert layer_bias.grad is not None and torch.isfinite(layer_bias.grad).all()
    assert layer_bias.grad.abs().sum() > 0


@pytest.mark.parametrize("condition_shape", [None, (2, 80), (2, 2, 5, 80), (1, 5, 80), (2, 5, 79)])
def test_missing_or_malformed_single_image_condition_is_rejected(condition_shape):
    model = make_denoiser().eval()
    noisy, timesteps, _ = denoising_inputs()
    condition = None if condition_shape is None else torch.zeros(condition_shape)
    with pytest.raises(ValueError):
        model(noisy, timesteps, condition)


def test_unconditional_route_needs_no_image_memory():
    model = make_denoiser(condition_dim=None).eval()
    noisy, timesteps, _ = denoising_inputs()
    with torch.no_grad():
        output = model(noisy, timesteps, None)
    assert len(model.cross_attention_layers) == 0
    assert output.shape == noisy.shape
    assert torch.isfinite(output).all()


def test_diffuse_ignores_legacy_clean_argument_and_has_no_alignment_parameters(tiny_dependencies):
    cfg, condition_cfg = tiny_config()
    model = tiny_dependencies.Diffusion(cfg, condition_cfg).eval()
    noisy, timesteps, memory = denoising_inputs()
    clean = torch.full_like(noisy, float("nan"), requires_grad=True)
    ordinary, alignment = model.diffuse(noisy, timesteps, memory)
    with_clean, alignment_with_clean = model.diffuse(noisy, timesteps, memory, clean)
    torch.testing.assert_close(with_clean, ordinary, rtol=0, atol=0)
    assert alignment is alignment_with_clean is None
    with_clean.square().mean().backward()
    assert clean.grad is None
    assert "clean_latent_sequence" not in inspect.signature(model.denoiser.forward).parameters
    assert all(not any(old in name for old in ("alignment", "condition_fuser", "camera_embedding")) for name in model.state_dict())


def test_training_and_image_only_sampling_never_pass_clean_latent_to_denoiser(tiny_dependencies, monkeypatch):
    cfg, condition_cfg = tiny_config()
    model = tiny_dependencies.Diffusion(cfg, condition_cfg)
    diffuse = model.diffuse
    calls = []

    def checked_diffuse(noisy, timesteps, condition, clean_latent_sequence=None, self_attention_mask=None):
        assert clean_latent_sequence is None
        calls.append(noisy.shape)
        return diffuse(noisy, timesteps, condition, self_attention_mask=self_attention_mask)

    monkeypatch.setattr(model, "diffuse", checked_diffuse)
    batch = tiny_batch()
    losses = model(batch)
    assert "align_loss" not in losses
    assert len(calls) == 1

    def forbidden_ground_truth(*_args, **_kwargs):
        raise AssertionError("Image-only inference must not read a latent cache or encode GT geometry.")

    monkeypatch.setattr(model.latent_provider, "forward", forbidden_ground_truth)
    monkeypatch.setattr(model.autoencoder, "encode", forbidden_ground_truth)
    model.eval()
    with torch.no_grad():
        samples = model.inference(2, torch.device("cpu"), {"conditions": batch["conditions"]}, v_log=False)
    assert len(calls) == 5
    assert len(samples) == 2
    assert all(torch.isfinite(sample["face_latents"]).all() for sample in samples)


def test_legacy_fuser_config_is_explicitly_rejected(tiny_dependencies):
    cfg, condition_cfg = tiny_config()
    cfg["condition_fuser"] = {"type": "cross_attention", "alignment": {"enabled": True}}
    with pytest.raises(ValueError, match="condition_fuser"):
        tiny_dependencies.Diffusion(cfg, condition_cfg)


def test_new_checkpoint_loads_strictly_and_legacy_fuser_keys_are_rejected():
    model = make_denoiser()
    new_state = model.state_dict()
    make_denoiser().load_state_dict(new_state, strict=True)
    # Represent the old prefix plus absent new layers; no permissive migration.
    legacy_state = {name: value for name, value in new_state.items() if not name.startswith("cross_attention_layers.")}
    legacy_state["condition_fuser.query_projection.weight"] = torch.zeros(64, 64)
    with pytest.raises(RuntimeError, match="condition_fuser"):
        model.load_state_dict(legacy_state, strict=True)


def test_full_model_forbids_permissive_checkpoint_loading(tiny_dependencies):
    cfg, condition_cfg = tiny_config()
    model = tiny_dependencies.Diffusion(cfg, condition_cfg)
    with pytest.raises(ValueError, match="strict=True"):
        model.load_state_dict(model.state_dict(), strict=False)
    result = model.load_state_dict(model.state_dict(), strict=True)
    assert not result.missing_keys and not result.unexpected_keys


def test_inference_checkpoint_loader_preserves_vae_and_loads_all_other_parameters(tiny_dependencies):
    cfg, condition_cfg = tiny_config()
    torch.manual_seed(109)
    model = tiny_dependencies.Diffusion(cfg, condition_cfg)
    original_vae = {name: value.clone() for name, value in model.autoencoder.state_dict().items()}
    torch.manual_seed(113)
    checkpoint_model = tiny_dependencies.Diffusion(cfg, condition_cfg)
    checkpoint = {name: value.clone() for name, value in checkpoint_model.state_dict().items()}
    prefixes = ("autoencoder.", "latent_provider.autoencoder.", "latent_codec.autoencoder.", "ae_model.")
    # All historical VAE aliases must be discarded, even with incompatible values.
    for name in list(checkpoint):
        if name.startswith(prefixes):
            checkpoint[name] = torch.tensor([float("nan")])
    checkpoint["latent_codec.autoencoder.legacy.weight"] = torch.tensor([float("nan")])
    checkpoint["ae_model.legacy.weight"] = torch.tensor([float("nan")])
    result = model.load_diffusion_state_dict(checkpoint)
    assert not result.missing_keys and not result.unexpected_keys
    for name, value in model.autoencoder.state_dict().items():
        torch.testing.assert_close(value, original_vae[name], rtol=0, atol=0)
    for name, value in model.state_dict().items():
        if not name.startswith(prefixes):
            torch.testing.assert_close(value, checkpoint[name], rtol=0, atol=0)


@pytest.mark.parametrize("problem", ["missing_layer", "legacy_fuser", "wrong_shape"])
def test_inference_checkpoint_loader_rejects_incomplete_or_incompatible_network(tiny_dependencies, problem):
    cfg, condition_cfg = tiny_config()
    model = tiny_dependencies.Diffusion(cfg, condition_cfg)
    checkpoint = dict(model.state_dict())
    key = "denoiser.cross_attention_layers.0.attention.q_proj_weight"
    if problem == "missing_layer":
        del checkpoint[key]
    elif problem == "legacy_fuser":
        checkpoint["denoiser.condition_fuser.query_projection.weight"] = torch.zeros(64, 64)
    else:
        checkpoint[key] = torch.zeros(3, 3)
    with pytest.raises(RuntimeError):
        model.load_diffusion_state_dict(checkpoint)
