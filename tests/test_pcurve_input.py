"""Owner-local UV reading must preserve the published path at initialization."""
import copy
import hashlib
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch import nn
from omegaconf import OmegaConf

from src.datasets.brep import AutoEncoder_dataset3
from src.datasets.pcurve_input import PcurveInputDataset
from src.models.vae.pcurve_input import PcurveFaceInput
from src.train import UnifiedTrainer
from tests.test_shared_boundaries import make_boundary_arrays


@pytest.fixture
def example(tmp_path):
    (tmp_path / "ids.txt").write_text("one\ntwo\n")
    for cad_id in ("one", "two"):
        folder = tmp_path / cad_id
        folder.mkdir()
        faces, edges, conn = make_boundary_arrays()
        adj = np.eye(2, dtype=bool) == 0
        np.savez(folder / "data.npz", sample_points_faces=faces, sample_points_lines=edges,
                 edge_face_connectivity=conn, face_adj=adj, zero_positions=np.argwhere(~adj))
        native = tmp_path / "native" / cad_id
        native.mkdir(parents=True)
        np.savez(native / "data.npz", schema="native_pcurve_v4",
                 source_npz_sha256=hashlib.sha256((folder / "data.npz").read_bytes()).hexdigest(),
                 pcurve_uv=np.arange(5 * 16 * 2, dtype=np.float32).reshape(5, 16, 2) / 160,
                 owner=np.array([1, 0, 1, 0, 0], dtype=np.int64))
    return dict(raw_data_root=str(tmp_path), pcurve_input_root=str(tmp_path / "native"),
                train_dataset=str(tmp_path / "ids.txt"), val_dataset=str(tmp_path / "ids.txt"),
                test_dataset=str(tmp_path / "ids.txt"), is_aug=1, is_overfit=False,
                condition_name=[], cached_condition=False, num_points=10000)


def test_legacy_geometry_rng_rotation_and_native_owner_packing(example):
    legacy = AutoEncoder_dataset3("training", example)
    added = PcurveInputDataset("training", example)
    np.random.seed(23)
    old = legacy.collate_fn([legacy[0], legacy[1]])
    np.random.seed(23)
    new = added.collate_fn([added[0], added[1]])
    for key, value in old.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, new[key]), key
        else:
            assert value == new[key]
    assert len(new["pcurve_uv"]) == 10  # Extra native uses never replace legacy edges.
    assert new["pcurve_indices"].tolist() == [[1, 3, 4], [0, 2, -1], [6, 8, 9], [5, 7, -1]]
    assert torch.equal(new["pcurve_padding_mask"], new["pcurve_indices"] < 0)
    test_set = PcurveInputDataset("testing", example)
    unrotated, rotated = test_set[0], test_set[2]
    assert not torch.equal(unrotated[0][1], rotated[0][1])
    assert torch.equal(unrotated[1], rotated[1])


def test_bad_native_provenance_or_empty_owner_fails(example):
    dataset = PcurveInputDataset("validation", example)
    path = dataset.pcurve_root / "one/data.npz"
    with np.load(path) as f:
        fields = dict(f)
    np.savez(path, **{**fields, "source_npz_sha256": "wrong"})
    with pytest.raises(ValueError, match="hash mismatch"):
        dataset[0]
    np.savez(path, **{**fields, "owner": np.zeros(5, dtype=np.int64)})
    with pytest.raises(ValueError, match="missing face boundaries"):
        dataset[0]


def module_and_inputs():
    torch.manual_seed(19)
    module = PcurveFaceInput(8, width=32, curve_depth=4, layers=4, heads=4, dropout=0).eval()
    face = torch.randn(3, 8)
    uv = torch.rand(6, 16, 2)
    indices = torch.tensor([[0, 1, 2], [3, -1, -1], [4, 5, -1]])
    return module, (face, uv, indices, indices < 0)


def test_zero_initialization_and_internal_gradients_after_second_update():
    module, inputs = module_and_inputs()
    assert torch.equal(module(*inputs), inputs[0])
    optimizer = torch.optim.AdamW(module.parameters(), lr=1e-3)
    target = torch.randn_like(inputs[0])
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        nn.functional.mse_loss(module(*inputs), target).backward()
        optimizer.step()
    for name, parameter in module.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
    assert module.curve_encoder[0].weight.grad.abs().max() > 0
    for layer in module.layers:
        assert layer.attention.in_proj_weight.grad.abs().max() > 0


def test_owner_isolation_padding_and_boundary_permutation():
    module, (face, uv, indices, padding) = module_and_inputs()
    nn.init.normal_(module.output_projection.weight, std=.05)
    expected = module(face, uv, indices, padding)
    changed = uv.clone()
    changed[3] += 10
    actual = module(face, changed, indices, padding)
    torch.testing.assert_close(actual[[0, 2]], expected[[0, 2]], rtol=0, atol=0)
    assert not torch.allclose(actual[1], expected[1])
    torch.testing.assert_close(module(face, uv, indices.flip(1), padding.flip(1)), expected)
    extended = torch.cat([indices, torch.full((3, 2), -1)], dim=1)
    torch.testing.assert_close(module(face, uv, extended, extended < 0), expected)


def test_optimizer_defaults_unchanged_and_explicit_published_adamw():
    for config, expected, decay in [({}, torch.optim.Adam, 0),
                                    ({"optimizer": "adamw", "weight_decay": .01}, torch.optim.AdamW, .01)]:
        stub = SimpleNamespace(model=nn.Linear(2, 1), learning_rate=1e-4,
                               cfg=OmegaConf.create({"trainer": config}))
        optimizer = UnifiedTrainer.configure_optimizers(stub)
        assert isinstance(optimizer, expected)
        assert optimizer.param_groups[0]["weight_decay"] == decay


@pytest.mark.parametrize("config_name", ["train_vae_pcurve_input", "train_vae_hola_owner_wide"])
def test_resume_does_not_reopen_published_initialization(tmp_path, config_name):
    from hydra import compose, initialize_config_dir
    from pathlib import Path
    with initialize_config_dir(version_base=None, config_dir=str(Path(__file__).resolve().parents[1] / "configs")):
        overrides = [
            f"trainer.run_dir={tmp_path}", "trainer.resume_from_checkpoint=owned.ckpt",
            "model.init_hola_checkpoint=absent.ckpt", "dataset.raw_data_root=unused",
            "dataset.train_dataset=unused",
            "dataset.val_dataset=unused", "dataset.test_dataset=unused",
        ]
        if config_name == "train_vae_pcurve_input":
            overrides.append("dataset.pcurve_input_root=unused")
        cfg = compose(config_name=config_name, overrides=overrides)
    observed = []
    def build(config, condition):
        observed.append(copy.deepcopy(config))
        return nn.Linear(2, 1)
    with patch("src.train.build_model", side_effect=build):
        UnifiedTrainer(cfg)
    assert observed[0].init_hola_checkpoint is None
    assert observed[0].checkpoint is None
