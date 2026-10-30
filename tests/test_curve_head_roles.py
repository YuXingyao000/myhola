"""Parameter ownership, migration and UV gradient routing (not CAD quality)."""
from copy import deepcopy

import pytest
import torch
from torch import nn

from src.models.vae import AutoEncoder_intersection
from tests.test_intersection_decoder import activate_residuals
from tests.test_intersection_vae import encoder_batch
from tests.test_shared_curve_uv import config
from tests.test_uv_boundary_model import add_targets
from tools.migrate_checkpoint import migrate


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def model_config(mode="independent", detach=True, fp32=False):
    cfg = config(geometry_heads={"curve_mode": mode, "fp32": fp32})
    cfg["uv_boundary"]["detach_curve_input"] = detach
    return cfg


def has_gradient(module):
    return any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())


@pytest.mark.parametrize("mode", ["teacher_split", "independent"])
def test_migration_preserves_predictions_but_separates_parameter_storage(mode):
    torch.manual_seed(581)
    original = AutoEncoder_intersection(model_config("shared", False)).eval()
    checkpoint = {
        "state_dict": {"model." + k: v for k, v in original.state_dict().items()},
        "hyper_parameters": {"model": model_config("shared", False)},
        "optimizer_states": [{"state": {}}], "loops": {}, "callbacks": {},
        "epoch": 18, "global_step": 7885,
    }
    migrate(checkpoint, mode)
    migrated = AutoEncoder_intersection(model_config(mode)).eval()
    migrated.load_state_dict({k.removeprefix("model."): v for k, v in checkpoint["state_dict"].items()}, strict=True)
    assert "optimizer_states" not in checkpoint and "loops" not in checkpoint
    assert checkpoint["initialization_only"]["source_global_step"] == 7885
    assert "already applied" in migrate(checkpoint, mode)[-1]
    decoder = migrated.decoder
    heads = [decoder.teacher_edge_geometry]
    if mode == "independent":
        heads.append(decoder.coarse_edge_geometry)
    else:
        assert decoder.coarse_edge_geometry is None
    original_parameters = dict(decoder.edge_geometry.named_parameters())
    for head in heads:
        assert head.conditioning is None
        for key, parameter in head.named_parameters():
            source = original_parameters[key]
            assert source.data_ptr() != parameter.data_ptr()
            torch.testing.assert_close(parameter, source, rtol=0, atol=0)
    batch = add_targets(encoder_batch())
    with torch.no_grad():
        old_losses, old = original(batch, v_test=True)
        new_losses, new = migrated(batch, v_test=True)
    for key in old_losses:
        torch.testing.assert_close(new_losses[key], old_losses[key], rtol=0, atol=0)
    for a, b in zip(old["items"], new["items"]):
        for field in ("pred_face", "pred_edge", "pred_edge_uv"):
            torch.testing.assert_close(torch.from_numpy(a[field]), torch.from_numpy(b[field]), rtol=0, atol=0)


def test_teacher_loss_trains_edge_encoder_without_final_geometry_parameters():
    model = AutoEncoder_intersection(model_config()).eval()
    encoding = model.encode(encoder_batch())
    teacher = model.decoder.decode_teacher(encoding["edge_features"])
    (teacher.local_points.square().mean() + teacher.center_scale.square().mean()).backward()
    assert has_gradient(model.encoder.edge_coords)
    assert has_gradient(model.decoder.teacher_edge_geometry)
    assert not has_gradient(model.decoder.edge_geometry)
    assert not has_gradient(model.decoder.coarse_edge_geometry)


def test_independent_final_head_is_used_without_refinement_or_shared_curve():
    cfg = model_config()
    cfg["refinement"]["enabled"] = False
    cfg["shared_curve"]["enabled"] = False
    cfg["curve_conditioning"]["enabled"] = False
    model = AutoEncoder_intersection(cfg).eval()
    result = model.decode(torch.randn(3, model.face_feature_dim), torch.tensor([3]))
    result.refined.edge_geometry.points.square().mean().backward()
    assert has_gradient(model.decoder.edge_geometry)
    assert not has_gradient(model.decoder.coarse_edge_geometry)


def test_uv_detach_preserves_values_and_still_trains_shared_features():
    torch.manual_seed(582)
    direct = AutoEncoder_intersection(model_config(detach=False)).eval()
    activate_residuals(direct.decoder)
    detached = deepcopy(direct)
    detached.decoder.detach_uv_curve_input = True
    z, counts = torch.randn(3, direct.face_feature_dim), torch.tensor([3])
    first, second = direct.decode(z, counts), detached.decode(z, counts)
    torch.testing.assert_close(first.edge_uv, second.edge_uv, rtol=0, atol=0)
    first.edge_uv.square().mean().backward()
    second.edge_uv.square().mean().backward()
    assert has_gradient(direct.decoder.edge_geometry)
    assert not has_gradient(detached.decoder.edge_geometry)
    assert has_gradient(detached.decoder.uv_boundary)
    assert has_gradient(detached.decoder.shared_curve)
    assert has_gradient(detached.decoder.inter)
    assert has_gradient(detached.decoder.face_features)
    # h/g/F still depend on coarse geometry through refinement; keep this path.
    assert has_gradient(detached.decoder.coarse_edge_geometry)


def test_final_and_anchor_geometry_losses_keep_both_geometry_heads_connected():
    torch.manual_seed(583)
    model = AutoEncoder_intersection(model_config()).eval()
    activate_residuals(model.decoder)
    losses, _ = model(add_targets(encoder_batch()))
    losses["uv_anchor"].backward()
    assert has_gradient(model.decoder.edge_geometry)
    assert has_gradient(model.decoder.face_geometry)
    assert has_gradient(model.decoder.coarse_edge_geometry)
    assert not has_gradient(model.decoder.uv_boundary)


@pytest.mark.parametrize("checkpointed", [False, True])
def test_fp32_covers_convolution_bbox_conditioning_and_local_face_map(checkpointed):
    cfg = model_config(fp32=True)
    # Conditional attention exercises forward_with_local_features for coarse faces.
    cfg["intersection"] = {"kind": "conditional_attention", "candidate_chunk_size": 2,
                           "checkpoint_candidates": checkpointed}
    cfg["refinement"]["checkpoint_candidates"] = checkpointed
    model = AutoEncoder_intersection(cfg).train()
    outputs = []
    handles = []
    for head in (model.decoder.face_geometry, model.decoder.edge_geometry,
                 model.decoder.coarse_edge_geometry, model.decoder.teacher_edge_geometry):
        for module in head.modules():
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                handles.append(module.register_forward_hook(lambda mod, args, out: outputs.append(out.dtype)))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        losses, _ = model(add_targets(encoder_batch()))
    losses["total_loss"].backward()
    for handle in handles:
        handle.remove()
    assert outputs and set(outputs) == {torch.float32}
    assert torch.isfinite(losses["total_loss"])
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_split_heads_checkpoint_recompute_uses_the_same_head():
    torch.manual_seed(584)
    direct = AutoEncoder_intersection(model_config(fp32=True)).eval()
    activate_residuals(direct.decoder)
    recompute = deepcopy(direct)
    direct.train()
    recompute.train()
    direct.decoder.checkpoint_candidates = False
    recompute.decoder.checkpoint_candidates = True
    for model in (direct, recompute):
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.p = 0.0
            elif isinstance(module, nn.MultiheadAttention):
                module.dropout = 0.0
    batch = add_targets(encoder_batch())
    results = []
    for model in (direct, recompute):
        torch.manual_seed(585)  # same posterior sampling
        losses, _ = model(batch)
        losses["total_loss"].backward()
        results.append(losses["total_loss"])
    torch.testing.assert_close(*results)
    for (name, value), (other_name, other) in zip(direct.named_parameters(), recompute.named_parameters()):
        assert name == other_name
        if value.grad is None:
            assert other.grad is None, name
        else:
            torch.testing.assert_close(value.grad, other.grad, atol=1e-6, rtol=1e-4, msg=name)
