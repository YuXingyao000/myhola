"""Geometry objective, GT correspondence, freezing and checkpoint contracts (CPU)."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from src.datasets.preference import PreferenceDataset, random_duplicate_padding
from src.models.diffusion.preference import PreferenceDiffusion
from src.models.diffusion.preference_geometry import (
    decode_supervised_points, geometry_settings, predicted_clean_latent,
    predicted_geometry_loss, primitive_chamfer_squared,
)
from tests import test_diffusion_preference as preference_fixtures
from tests import test_preference_legacy_loading as loading_fixtures


class TinyAttention(nn.Linear):
    def forward(self, value, mask):
        return super().forward(value) + value.mean(0, keepdim=True)


class TinyPoints(nn.Linear):
    def __init__(self, surface):
        super().__init__(32, 6)
        self.surface = surface

    def forward(self, value):
        points = super().forward(value)
        return (points[:, None, None].expand(-1, 2, 2, -1) if self.surface
                else points[:, None].expand(-1, 2, -1))


class TinyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.face_attn_proj_in2 = nn.Linear(32, 32)
        self.face_attn2 = TinyAttention(32, 32)
        self.face_attn_proj_out2 = nn.Linear(32, 32)
        self.face_points_decoder = TinyPoints(True)
        self.face_center_scale_decoder = nn.Linear(32, 4)
        self.inter = nn.Sequential(nn.Flatten(1), nn.Linear(64, 32))
        self.edge_points_decoder = TinyPoints(False)
        self.edge_center_scale_decoder = nn.Linear(32, 4)


def legacy_with_decoder():
    legacy, provenance = preference_fixtures.tiny_legacy()
    legacy.ae_model = TinyDecoder()
    return legacy, provenance


def geometry_target(face_slots=(0, 4), with_edges=True):
    return {
        "face_slots": torch.tensor(face_slots),
        "face_points": torch.zeros(len(face_slots), 2, 2, 3),
        "edge_points": torch.zeros(1 if with_edges else 0, 2, 3),
        "edge_faces": torch.tensor([[0, 1]] if with_edges else [], dtype=torch.long).reshape(-1, 2),
    }


class GeometryMathTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.decoder = TinyDecoder().requires_grad_(False).eval()

    def test_epsilon_to_x0_recovers_gt_and_retains_gradients(self):
        clean, noise = torch.randn(2, 30, 32), torch.randn(2, 30, 32)
        alpha = torch.tensor([.8, .2])
        noisy = alpha[:, None, None].sqrt() * clean + (1 - alpha[:, None, None]).sqrt() * noise
        prediction = noise.clone().requires_grad_()
        recovered = predicted_clean_latent(noisy, prediction, torch.tensor([0, 1]), alpha)
        torch.testing.assert_close(recovered, clean)
        recovered.square().mean().backward()
        self.assertGreater(prediction.grad.abs().sum().item(), 0)

    def test_chamfer_squared_definition_permutation_and_zero_gradient(self):
        pred = torch.zeros(1, 1, 3, requires_grad=True)
        target = torch.tensor([[[1., 2., 2.]]])
        loss = primitive_chamfer_squared(pred, target)
        torch.testing.assert_close(loss, torch.tensor([18.]))
        loss.sum().backward()
        torch.testing.assert_close(pred.grad, torch.tensor([[[-4., -8., -8.]]]))
        points = torch.randn(3, 7, 3, requires_grad=True)
        equal = primitive_chamfer_squared(points, points.detach().flip(1), chunk_size=2)
        torch.testing.assert_close(equal, torch.zeros(3), rtol=0, atol=0)
        equal.sum().backward()
        self.assertTrue(torch.isfinite(points.grad).all())

    def test_gate_full_batch_denominator_empty_edges_and_selected_slots(self):
        noisy = torch.randn(2, 30, 32)
        pred = torch.randn_like(noisy, requires_grad=True)
        timestep, alphas = torch.tensor([0, 1]), torch.tensor([.8, .2])
        batch = {"v_prefix": ["a", "b"], "gt_geometry": [geometry_target(with_edges=False), geometry_target()]}
        combined = predicted_geometry_loss(self.decoder, noisy, pred, timestep, alphas, batch)
        single = predicted_geometry_loss(self.decoder, noisy[:1], pred[:1], timestep[:1], alphas,
                                         {"v_prefix": ["a"], "gt_geometry": batch["gt_geometry"][:1]})
        torch.testing.assert_close(combined["geometry_loss"] * 2, single["geometry_loss"])
        self.assertEqual(combined["geometry_active_fraction"].item(), .5)
        self.assertEqual(combined["geometry_curve_loss"].item(), 0)
        combined["geometry_loss"].backward()
        self.assertTrue(torch.isfinite(pred.grad).all())
        self.assertGreater(pred.grad[0, [0, 4]].abs().sum().item(), 0)
        unused = [index for index in range(30) if index not in (0, 4)]
        self.assertEqual(pred.grad[0, unused].abs().sum().item(), 0)
        self.assertEqual(pred.grad[1].abs().sum().item(), 0)
        self.assertTrue(all(parameter.grad is None for parameter in self.decoder.parameters()))

    def test_all_high_noise_returns_differentiable_zero(self):
        pred = torch.randn(2, 30, 32, requires_grad=True)
        with patch("src.models.diffusion.preference_geometry.decode_supervised_points") as forbidden:
            losses = predicted_geometry_loss(self.decoder, torch.randn_like(pred), pred, torch.ones(2, dtype=torch.long),
                torch.tensor([.8, .2]), {"v_prefix": ["a", "b"], "gt_geometry": [geometry_target(), geometry_target()]})
        forbidden.assert_not_called()
        losses["geometry_loss"].backward()
        self.assertEqual(losses["geometry_loss"].item(), 0)
        self.assertEqual(pred.grad.abs().sum().item(), 0)

    def test_relabelled_face_order_with_remapped_owners_preserves_geometry_loss(self):
        noisy = torch.randn(1, 30, 32)
        pred = torch.randn_like(noisy, requires_grad=True)
        target = geometry_target((0, 4, 8))
        target["face_points"] = torch.randn(3, 2, 2, 3)
        target["edge_points"] = torch.randn(1, 2, 3)
        perm = torch.tensor([2, 0, 1])
        reordered = {**target, "face_slots": target["face_slots"][perm],
                     "face_points": target["face_points"][perm],
                     "edge_faces": perm.argsort()[target["edge_faces"]]}
        values = [predicted_geometry_loss(self.decoder, noisy, pred, torch.tensor([0]), torch.tensor([.8]),
                  {"v_prefix": ["a"], "gt_geometry": [item]}) for item in (target, reordered)]
        torch.testing.assert_close(values[0]["geometry_loss"], values[1]["geometry_loss"])

    def test_frozen_decoder_boundary_and_invalid_settings(self):
        for config in ({"weight": float("nan")}, {"min_snr": .1}, {"primitive_chunk_size": 0}, {"typo": 1}):
            with self.assertRaises(ValueError):
                geometry_settings(config)
        self.decoder.train()
        with self.assertRaisesRegex(ValueError, "frozen decoder"):
            decode_supervised_points(self.decoder, torch.randn(2, 32), torch.tensor([[0, 1]]))


class GeometryModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.legacy, self.provenance = legacy_with_decoder()
        self.config = {"legacy_checkpoint": "unused", "beta_dpo": 20., "preference_weight": 1.,
                       "geometry": {"enabled": True, "weight": .3, "min_snr": 1.}}

    def build(self, config=None):
        with patch("src.models.diffusion.legacy.load_legacy_svr", side_effect=lambda *a, **kw:
                   (deepcopy(self.legacy), deepcopy(self.provenance))):
            return PreferenceDiffusion(config or self.config)

    def batch(self):
        return {"v_prefix": ["a", "b"], "gt_image": torch.randn(2, 3, 8, 8),
                "pair_image": torch.randn(2, 3, 8, 8), "gt_latent": torch.randn(2, 30, 32),
                "winner_latent": torch.randn(2, 30, 32), "loser_latent": torch.randn(2, 30, 32),
                "gt_geometry": [geometry_target(), geometry_target((1, 7, 9))]}

    def test_original_losses_trainable_parameters_and_rng_unchanged(self):
        control = self.build({**self.config, "geometry": {"enabled": False}}).train()
        geometry = self.build().train()
        self.assertEqual([key for key, value in control.named_parameters() if value.requires_grad],
                         [key for key, value in geometry.named_parameters() if value.requires_grad])
        batch = self.batch()
        torch.manual_seed(1)
        before = control(batch)
        rng = torch.get_rng_state()
        torch.manual_seed(1)
        after = geometry(batch)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for key in before:
            if key != "total_loss":
                torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)
        torch.testing.assert_close(after["total_loss"], before["total_loss"] + .3 * after["geometry_loss"])
        self.assertGreater(after["geometry_loss"].item(), 0)
        decoder_before = deepcopy(geometry.geometry_decoder.state_dict())
        after["total_loss"].backward()
        for module in (geometry.reference, geometry.image_encoder, geometry.geometry_decoder):
            self.assertFalse(module.training)
            self.assertTrue(all(parameter.grad is None for parameter in module.parameters()))
        self.assertTrue(all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                            for parameter in geometry.policy.parameters()))
        optimizer = torch.optim.Adam((p for p in geometry.parameters() if p.requires_grad), lr=1e-5)
        optimizer.step()
        for key, value in decoder_before.items():
            torch.testing.assert_close(value, geometry.geometry_decoder.state_dict()[key], rtol=0, atol=0)
        self.assertEqual(set(geometry(batch, v_test=True)), {"ddpm_loss", "total_loss"})

    def test_initialize_keeps_reference_and_adds_only_frozen_decoder(self):
        original = self.build({**self.config, "geometry": {"enabled": False}})
        with torch.no_grad():
            original.policy.fc_out.bias.add_(.25)
        data = {"source_checkpoint_sha256": "fixed", "pairs_manifest_sha256": "same-pairs"}
        saved = {"global_step": 5000, "epoch": 6, "hyper_parameters": {
            "model": {"name": "PreferenceDiffusion", "legacy_model_config": self.provenance["model_config"],
                      "beta_dpo": 20., "preference_weight": 1.}, "dataset": {"provenance": data}},
            "state_dict": {f"model.{key}": value for key, value in original.state_dict().items()}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old.ckpt"
            torch.save(saved, path)
            model = self.build()
            info = model.initialize_preference_weights(path, {**data, "geometry_supervision": {"new": True}})
            self.assertFalse(info["optimizer_restored"])
            self.assertEqual(info["source_global_step"], 5000)
            for key, value in original.state_dict().items():
                torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)
            with self.assertRaisesRegex(ValueError, "same pair pool"):
                model.initialize_preference_weights(path, {**data, "pairs_manifest_sha256": "changed"})
            saved["state_dict"]["model.reference.fc_out.bias"] = saved["state_dict"]["model.reference.fc_out.bias"] + 1
            torch.save(saved, path)
            with self.assertRaisesRegex(ValueError, "frozen reference"):
                self.build().initialize_preference_weights(path, data)

    def test_geometry_state_can_strictly_resume_without_initial_checkpoint(self):
        original = self.build()
        with patch("src.models.diffusion.legacy.build_legacy_svr", side_effect=lambda *a, **kw:
                   (deepcopy(self.legacy), deepcopy(self.provenance))) as build, \
                patch("src.models.diffusion.legacy.load_legacy_svr") as forbidden:
            restored = PreferenceDiffusion({**self.config, "restoring_checkpoint": True,
                                            "legacy_model_config": self.provenance["model_config"]})
            restored.load_state_dict(original.state_dict(), strict=True)
        forbidden.assert_not_called()
        self.assertTrue(build.call_args.kwargs["with_autoencoder"])
        for key, value in original.state_dict().items():
            torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)

    def test_trainer_resume_rejects_changed_geometry_and_skips_initialization(self):
        from hydra import compose, initialize_config_dir
        from src.train import UnifiedTrainer

        with tempfile.TemporaryDirectory() as directory:
            configs = Path(__file__).resolve().parents[1] / "configs"
            with initialize_config_dir(config_dir=str(configs), version_base="1.3"):
                cfg = compose("train_diffusion_preference_geometry", overrides=[
                    "model.geometry.weight=0.3", "model.init_preference_checkpoint=null",
                    f"trainer.run_dir={directory}", "trainer.max_steps=1",
                ])
            data = {"source_checkpoint_sha256": "fixed", "geometry_supervision": {"version": "test"}}
            with patch("src.models.diffusion.legacy.load_legacy_svr", side_effect=lambda *a, **kw:
                       (deepcopy(self.legacy), deepcopy(self.provenance))), \
                    patch("src.datasets.preference.preference_data_provenance", return_value=data), \
                    patch("src.datasets.preference.sha256_file", return_value="fixed"):
                original = UnifiedTrainer(cfg)
            path = Path(directory) / "resume.ckpt"
            torch.save({"hyper_parameters": dict(original.hparams), "state_dict": original.state_dict()}, path)
            cfg.trainer.resume_from_checkpoint = str(path)
            cfg.model.init_preference_checkpoint = "/removed/old_dpo.ckpt"
            cfg.model.legacy_checkpoint = "/removed/svr.ckpt"
            with patch("src.models.diffusion.legacy.build_legacy_svr", side_effect=lambda *a, **kw:
                       (deepcopy(self.legacy), deepcopy(self.provenance))), \
                    patch("src.models.diffusion.legacy.load_legacy_svr") as forbidden, \
                    patch("src.datasets.preference.preference_data_provenance", return_value=data), \
                    patch.object(PreferenceDiffusion, "initialize_preference_weights") as no_init:
                restored = UnifiedTrainer(cfg)
                restored.load_state_dict(original.state_dict(), strict=True)
                cfg.model.geometry.weight = .4
                with self.assertRaisesRegex(ValueError, "changed geometry settings"):
                    UnifiedTrainer(cfg)
            forbidden.assert_not_called()
            no_init.assert_not_called()


class GeometryDatasetTests(unittest.TestCase):
    def setUp(self):
        preference_fixtures.PreferenceDatasetTests.setUp(self)
        self.config["geometry"] = {"enabled": True, "raw_data_root": self.root / "raw"}
        self.protocol["raw_data_root"] = str(self.root / "raw")
        self.protocol.pop("fingerprint")
        self.protocol["fingerprint"] = hashlib.sha256(json.dumps(
            self.protocol, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        (self.root / "metadata/protocol.json").write_text(json.dumps(self.protocol))
        for image_id in ("a", "b", "c"):
            directory = self.root / "raw" / image_id
            directory.mkdir(parents=True)
            faces = np.zeros((3, 16, 16, 6), np.float32)
            faces[..., :3] = np.arange(3)[:, None, None, None]
            edges = np.zeros((4, 16, 6), np.float32)
            edges[..., :3] = np.arange(4)[:, None, None]
            np.savez(directory / "data.npz", sample_points_faces=faces, sample_points_lines=edges,
                     edge_face_connectivity=np.array([[2, 1, 0], [0, 0, 1], [3, 2, 2]]))
            cache = self.root / "gt_cache" / f"{image_id}_0"
            (cache / "source.json").write_text(json.dumps({
                "precision": "float32", "layout": "mean32,std32",
                "gt_npz_sha256": hashlib.sha256((directory / "data.npz").read_bytes()).hexdigest(),
                "output_sha256": {"features.npy": hashlib.sha256((cache / "features.npy").read_bytes()).hexdigest()}}))

    def test_padding_retains_original_rng_and_one_copy_per_face(self):
        values = torch.arange(3 * 32).reshape(3, 32)
        torch.manual_seed(5)
        old = random_duplicate_padding(values)
        rng = torch.get_rng_state()
        torch.manual_seed(5)
        new, slots = random_duplicate_padding(values, return_face_slots=True)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        torch.testing.assert_close(old, new, rtol=0, atol=0)
        torch.testing.assert_close(new[slots], values, rtol=0, atol=0)

    def test_geometry_preserves_data_draws_and_ordered_targets(self):
        control = PreferenceDataset("training", {**self.config, "geometry": {"enabled": False}})
        geometry = PreferenceDataset("training", self.config)
        torch.manual_seed(10)
        before = control[0]
        rng = torch.get_rng_state()
        torch.manual_seed(10)
        after = geometry[0]
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for key in before:
            if torch.is_tensor(before[key]):
                torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)
            else:
                self.assertEqual(before[key], after[key])
        target = after["gt_geometry"]
        torch.testing.assert_close(after["gt_latent"][target["face_slots"], 0], torch.arange(3).float())
        torch.testing.assert_close(target["edge_faces"], torch.tensor([[1, 0], [0, 1]]))
        torch.testing.assert_close(target["edge_points"][:, 0, 0], torch.tensor([2., 0.]))
        batch = geometry.collate_fn([after, geometry[1]])
        self.assertEqual(len(batch["gt_geometry"]), 2)
        self.assertEqual(batch["gt_latent"].shape, (2, 30, 32))
        (self.root / "raw/c/data.npz").unlink()
        self.assertNotIn("gt_geometry", PreferenceDataset("validation", self.config)[0])

    def test_gt_or_cache_tampering_is_rejected(self):
        dataset = PreferenceDataset("training", self.config)
        for path in (self.root / "raw/a/data.npz", self.root / "gt_cache/a_0/features.npy"):
            original = path.read_bytes()
            path.write_bytes(original + b"changed")
            with self.assertRaisesRegex(ValueError, "source mismatch"):
                dataset[0]
            path.write_bytes(original)


class GeometryInferenceLoadingTests(unittest.TestCase):
    def setUp(self):
        self.fixture = loading_fixtures.PreferenceLegacyLoadingTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.checkpoint["hyper_parameters"]["model"]["geometry"] = {"enabled": True, "weight": .1}
        self.fixture.checkpoint["state_dict"].update({f"model.geometry_decoder.{key}": value.clone()
            for key, value in self.fixture.original.ae_model.state_dict().items()})

    def test_geometry_checkpoint_native_inference_still_uses_original_decoder(self):
        model, provenance = self.fixture.load()
        value = torch.randn(3, 2)
        torch.testing.assert_close(model.inference(value), self.fixture.original.inference(value), rtol=0, atol=0)
        self.assertTrue(provenance["preference"]["geometry_decoder_equal_original"])

    def test_changed_or_undeclared_geometry_decoder_is_rejected(self):
        self.fixture.checkpoint["state_dict"]["model.geometry_decoder.weight"].add_(1)
        with self.assertRaisesRegex(ValueError, "Frozen DPO reference/DINO/decoder"):
            self.fixture.load()
        self.fixture.checkpoint["hyper_parameters"]["model"]["geometry"]["enabled"] = False
        with self.assertRaisesRegex(ValueError, "state keys differ"):
            self.fixture.load()


if __name__ == "__main__":
    unittest.main()
