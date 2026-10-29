"""照片 DPO 来源、空间采样接口与固定分母的针对性回归检查。"""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from src.brepnet.eval.photo_model_loading import SpatialSamplingAdapter, apply_photo_preference_state
from src.brepnet.eval.photo_model_report import KEYS, METHODS, summarize


def fake_legacy():
    model = nn.Module()
    for name in ("p_embed", "time_embed", "net1", "fc_out", "img_fc", "img_model", "ae_model"):
        setattr(model, name, nn.Linear(2, 2))
    return model


def preference_state(model):
    state = {}
    for name in ("p_embed", "time_embed", "net1", "fc_out", "img_fc"):
        for key, value in getattr(model, name).state_dict().items():
            state[f"model.reference.{name}.{key}"] = value.clone()
            state[f"model.policy.{name}.{key}"] = value.clone() + 1
    state.update({f"model.image_encoder.{k}": v.clone() for k, v in model.img_model.state_dict().items()})
    return state


def test_photo_reference_and_vae_are_retained():
    model = fake_legacy()
    state = preference_state(model)
    ae = deepcopy(model.ae_model.state_dict())
    apply_photo_preference_state(model, state)
    assert torch.equal(model.net1.weight, state["model.policy.net1.weight"])
    assert all(torch.equal(value, ae[key]) for key, value in model.ae_model.state_dict().items())


@pytest.mark.parametrize("corruption", ["reference", "dino", "missing", "extra", "nan"])
def test_reject_invalid_photo_dpo_before_mutating_policy(corruption):
    model = fake_legacy()
    before = model.net1.weight.clone()
    state = preference_state(model)
    if corruption == "reference":
        state["model.reference.net1.weight"] += .1
    elif corruption == "dino":
        state["model.image_encoder.weight"] += .1
    elif corruption == "missing":
        del state["model.policy.net1.weight"]
    elif corruption == "extra":
        state["model.ae_model.weight"] = torch.ones(2, 2)
    else:
        state["model.policy.net1.weight"][0, 0] = float("nan")
    with pytest.raises(ValueError):
        apply_photo_preference_state(model, state)
    assert torch.equal(model.net1.weight, before)


def test_spatial_adapter_only_changes_condition_call_signature():
    photo = torch.randn(1, 1, 3, 224, 224)
    latent, time, memory = torch.randn(1, 30, 32), torch.tensor([42]), torch.randn(1, 513, 512)
    spatial = SimpleNamespace(ae_model=object(), noise_scheduler=object(),
                              extract_condition=lambda images: images,
                              denoiser=lambda z, t, c: (z, t, c))
    adapter = SpatialSamplingAdapter(spatial)
    assert torch.equal(adapter.extract_condition({"conditions": {"imgs": photo}}), photo[:, 0])
    out = adapter.diffuse(latent, time, memory)
    assert all(a is b for a, b in zip(out, (latent, time, memory)))
    assert adapter.ae_model is spatial.ae_model and adapter.noise_scheduler is spatial.noise_scheduler
    with pytest.raises(ValueError):
        adapter.extract_condition({"conditions": {"imgs": photo[:, 0]}})


def fake_rows():
    return {method: [{"image_id": mid, "seed": i, "inference_status": "generation_error",
                     "evaluation_status": "not_run", "validity": "unknown", "post_complete": False,
                     "step_exists": False, "step_readable": False, "inference_seconds": 1.,
                     "evaluation_seconds": None, **{key: None for key in KEYS}}
                    for i, mid in enumerate(("first", "second"))] for method in METHODS}


def test_failures_stay_in_denominator_and_empty_geometry_is_null():
    rows = fake_rows()
    row = rows[METHODS[0]][0]
    row.update(inference_status="ok", validity="valid", raw_face_cd_sq=.02)
    summary = summarize(rows)
    assert summary["intended_per_method"] == 2
    assert summary["methods"][METHODS[0]]["valid_solids"] == 1
    assert summary["methods"][METHODS[1]]["metrics"]["raw_face_cd_sq"] == {"n": 0, "mean": None, "median": None}
    assert summary["pairwise"][f"{METHODS[1]}_minus_{METHODS[0]}"]["valid"]["mean_delta"] == -.5


def test_reject_different_cad_order_or_seed():
    rows = fake_rows()
    rows[METHODS[-1]].reverse()
    with pytest.raises(ValueError, match="ordered population"):
        summarize(rows)
    rows = fake_rows()
    rows[METHODS[-1]][0]["seed"] = 100
    with pytest.raises(ValueError, match="seeds"):
        summarize(rows)
