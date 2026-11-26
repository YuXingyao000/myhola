"""A frozen encoder is bit-identical before and after decoder training steps."""
import torch

from src.models.vae import AutoEncoder_light
from tests.test_intersection_vae import encoder_batch, tiny_config


def frozen_pair(tmp_path):
    torch.manual_seed(7)
    reference = AutoEncoder_light(tiny_config())
    checkpoint = tmp_path / "legacy.ckpt"
    torch.save({"state_dict": {f"model.{k}": v for k, v in reference.state_dict().items()}}, checkpoint)
    frozen = AutoEncoder_light(tiny_config(checkpoint=str(checkpoint), freeze_encoder=True))
    return reference, frozen


def encoder_keys(model):
    names = set()
    for module in model._encoder_modules():
        for name, _ in module.named_parameters():
            names.add(name)
    return {key for key, _ in model.named_parameters()
            if any(key == f"{prefix}.{n}" for prefix in ENCODER_PREFIXES for n in [key.split(".", 1)[1]] if key.startswith(prefix + "."))}


ENCODER_PREFIXES = (
    "face_coords", "edge_coords", "face_attn_proj_in", "face_attn", "face_attn_proj_out",
    "graph_face_edge", "global_feature1", "global_feature2", "gaussian_proj",
)


def is_encoder_key(key):
    return any(key.startswith(prefix + ".") for prefix in ENCODER_PREFIXES)


def test_frozen_encoder_is_eval_without_grad_and_unchanged_by_training(tmp_path):
    reference, frozen = frozen_pair(tmp_path)
    frozen.train()
    for key, parameter in frozen.named_parameters():
        assert parameter.requires_grad is not is_encoder_key(key), key
    assert not frozen.face_attn.training and not frozen.gaussian_proj.training
    assert frozen.face_attn2.training and frozen.inter.training

    before = {k: v.clone() for k, v in frozen.state_dict().items()}
    optimizer = torch.optim.Adam((p for p in frozen.parameters() if p.requires_grad), lr=1e-2)
    for _ in range(2):
        loss, _ = frozen(encoder_batch())
        assert loss["kl_loss"].grad_fn is None
        optimizer.zero_grad()
        loss["total_loss"].backward()
        for key, parameter in frozen.named_parameters():
            assert (parameter.grad is None) is is_encoder_key(key), key
        optimizer.step()
    after = frozen.state_dict()
    changed = [k for k in before if not torch.equal(before[k], after[k])]
    assert changed and not any(is_encoder_key(k) for k in changed)
    assert any(k.startswith("inter.") for k in changed) and any(k.startswith("face_attn2.") for k in changed)


def test_frozen_and_reference_agree_in_eval_mode(tmp_path):
    reference, frozen = frozen_pair(tmp_path)
    reference.eval()
    frozen.eval()
    with torch.no_grad():
        expected, _ = reference(encoder_batch(), v_test=True)
        actual, _ = frozen(encoder_batch(), v_test=True)
    for key, value in expected.items():
        torch.testing.assert_close(actual[key], value, rtol=0, atol=0)
