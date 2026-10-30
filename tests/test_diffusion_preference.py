"""偏好目标方向、冻结边界及原始候选 latent 的训练合同。"""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from diffusers import DDPMScheduler
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch import nn

from src.datasets.preference import PreferenceDataset, preference_data_provenance
from src.models.diffusion.preference import PreferenceDiffusion, diffusion_dpo_loss


class TinyDino(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(3, 8)

    def forward(self, images):
        return self.projection(images.mean((2, 3)))


def tiny_legacy():
    model = nn.Module()
    model.dim_total = 8
    model.img_model = TinyDino()
    model.p_embed = nn.Linear(32, 6)
    model.time_embed = nn.Linear(8, 8)
    model.img_fc = nn.Sequential(nn.Linear(8, 2), nn.Dropout(0.7))
    model.net1 = nn.Sequential(nn.Linear(8, 8), nn.Dropout(0.7), nn.ReLU())
    model.fc_out = nn.Linear(8, 32)
    model.noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2", clip_sample=False,
    )
    return model, {"model_config": {"test": True}}


class PreferenceLossTests(unittest.TestCase):
    def test_sampling_uses_separate_cpu_initial_and_step_generators(self):
        with patch("src.models.diffusion.legacy.load_legacy_svr", side_effect=lambda *a, **kw: tiny_legacy()):
            model = PreferenceDiffusion({"legacy_checkpoint": "unused", "beta_dpo": 20, "preference_weight": 1}).eval()
        images = torch.randn(1, 3, 8, 8)
        initial_generator = torch.Generator(device="cpu").manual_seed(12)
        step_generator = torch.Generator(device="cpu").manual_seed(34)
        expected_initial = torch.randn((1, 30, 32), generator=torch.Generator().manual_seed(12))
        observed = []
        scheduler = model.noise_scheduler
        native_step = scheduler.step

        def one_step_schedule(steps, device):
            self.assertEqual(steps, 1000)
            scheduler.timesteps = torch.tensor([0], device=device)

        def observe(prediction, timestep, latent, *, generator):
            observed.append(latent.clone())
            self.assertIs(generator, step_generator)
            return native_step(prediction, timestep, latent, generator=generator)

        with patch.object(scheduler, "set_timesteps", side_effect=one_step_schedule), patch.object(scheduler, "step", side_effect=observe):
            model.sample_latents(images, initial_generator=initial_generator, step_generator=step_generator)
        torch.testing.assert_close(observed[0], expected_initial, rtol=0, atol=0)

    def test_improving_winner_or_worsening_loser_reduces_preference_loss(self):
        winner = torch.tensor([1.0], requires_grad=True)
        loser = torch.tensor([1.0], requires_grad=True)
        reference = torch.tensor([1.0], requires_grad=True)
        loss, logits = diffusion_dpo_loss(winner, loser, reference, reference, 20.0)
        self.assertAlmostEqual(loss.item(), np.log(2), places=6)
        self.assertEqual(logits.item(), 0)
        loss.backward()
        self.assertGreater(winner.grad.item(), 0)
        self.assertLess(loser.grad.item(), 0)
        self.assertIsNone(reference.grad)
        improved, _ = diffusion_dpo_loss(winner - 0.1, loser, reference, reference, 20.0)
        self.assertLess(improved.item(), loss.item())

    def test_full_forward_freezes_reference_dino_and_disables_pair_dropout(self):
        with patch("src.models.diffusion.legacy.load_legacy_svr", side_effect=lambda *a, **kw: tiny_legacy()):
            model = PreferenceDiffusion({"legacy_checkpoint": "unused", "beta_dpo": 20, "preference_weight": 1})
        model.train()
        frozen = deepcopy(model.reference.state_dict())
        batch = {
            "gt_image": torch.randn(2, 3, 8, 8),
            "pair_image": torch.randn(2, 3, 8, 8),
            "gt_latent": torch.randn(2, 30, 32),
            "winner_latent": torch.randn(2, 30, 32),
            "loser_latent": torch.randn(2, 30, 32),
        }
        losses = model(batch)
        self.assertAlmostEqual(losses["preference_loss"].item(), np.log(2), places=6)
        self.assertAlmostEqual(losses["preference_logit"].item(), 0, places=7)
        losses["total_loss"].backward()
        self.assertTrue(all(p.grad is None for p in model.reference.parameters()))
        self.assertTrue(all(p.grad is None for p in model.image_encoder.parameters()))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.policy.parameters()))
        self.assertFalse(model.reference.training)
        self.assertFalse(model.image_encoder.training)
        self.assertTrue(model.policy.training)
        optimizer = torch.optim.SGD(model.policy.parameters(), lr=0.001)
        optimizer.step()
        for key, value in model.reference.state_dict().items():
            torch.testing.assert_close(value, frozen[key], rtol=0, atol=0)
        self.assertEqual(set(model(batch, v_test=True)), {"ddpm_loss", "total_loss"})

    def test_unified_checkpoint_resume_uses_saved_architecture_not_initial_weights(self):
        from src.train import UnifiedTrainer

        with tempfile.TemporaryDirectory() as directory:
            configs = Path(__file__).resolve().parents[1] / "configs"
            with initialize_config_dir(config_dir=str(configs), version_base="1.3"):
                cfg = compose("train_diffusion_preference", overrides=[
                    "dataset.pairs_manifest=/unused/pairs.jsonl",
                    "dataset.cached_latent_root=/unused/gt_cache",
                    "trainer.max_steps=1", f"trainer.run_dir={directory}",
                ])
            with (
                patch("src.models.diffusion.legacy.load_legacy_svr", side_effect=lambda *a, **kw: tiny_legacy()),
                patch("src.datasets.preference.preference_data_provenance", return_value={"source_checkpoint_sha256": "fixed"}),
                patch("src.datasets.preference.sha256_file", return_value="fixed") as source_hash,
            ):
                original = UnifiedTrainer(cfg)
            source_hash.assert_called_once_with(Path(cfg.model.legacy_checkpoint))
            # 模拟 policy 已更新、reference 保持最初权重的训练状态。
            with torch.no_grad():
                next(original.model.policy.parameters()).add_(1)
            checkpoint = Path(directory) / "resume.ckpt"
            torch.save({"hyper_parameters": dict(original.hparams), "state_dict": original.state_dict()}, checkpoint)
            cfg.trainer.resume_from_checkpoint = str(checkpoint)
            cfg.model.legacy_checkpoint = "/removed/original.ckpt"
            with (
                patch("src.models.diffusion.legacy.build_legacy_svr", side_effect=lambda *a, **kw: tiny_legacy()) as build,
                patch("src.models.diffusion.legacy.load_legacy_svr") as forbidden_load,
                patch("src.datasets.preference.preference_data_provenance", return_value={"source_checkpoint_sha256": "fixed"}),
                patch("src.datasets.preference.sha256_file") as forbidden_hash,
            ):
                resumed = UnifiedTrainer(cfg)
                state = torch.load(checkpoint, weights_only=False)["state_dict"]
                resumed.load_state_dict(state, strict=True)
            forbidden_load.assert_not_called()
            forbidden_hash.assert_not_called()
            self.assertEqual(build.call_args.args[0], {"test": True})
            self.assertEqual(OmegaConf.to_container(cfg.model.legacy_model_config), {"test": True})
            for key, value in original.state_dict().items():
                torch.testing.assert_close(value, resumed.state_dict()[key], rtol=0, atol=0)

    def test_same_architecture_different_initial_checkpoint_is_rejected(self):
        from src.train import UnifiedTrainer

        with tempfile.TemporaryDirectory() as directory:
            configs = Path(__file__).resolve().parents[1] / "configs"
            with initialize_config_dir(config_dir=str(configs), version_base="1.3"):
                cfg = compose("train_diffusion_preference", overrides=[
                    "dataset.pairs_manifest=/unused/pairs.jsonl",
                    "dataset.cached_latent_root=/unused/gt_cache",
                    "trainer.max_steps=1", f"trainer.run_dir={directory}",
                ])
            checkpoint = Path(directory) / "different.ckpt"
            checkpoint.write_bytes(b"different weights")
            cfg.model.legacy_checkpoint = str(checkpoint)
            with (
                patch("src.datasets.preference.preference_data_provenance", return_value={"source_checkpoint_sha256": "original-digest"}),
                patch("src.models.diffusion.legacy.load_legacy_svr") as forbidden_load,
            ):
                with self.assertRaisesRegex(ValueError, "differs from the preference generation checkpoint"):
                    UnifiedTrainer(cfg)
            forbidden_load.assert_not_called()


class PreferenceDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "train.txt").write_text("a\nb\n")
        (self.root / "val.txt").write_text("c\n")
        self.config = {
            "train_dataset": self.root / "train.txt", "val_dataset": self.root / "val.txt",
            "test_dataset": self.root / "val.txt", "cached_latent_root": self.root / "gt_cache",
            "condition_data_root": self.root / "images", "epoch_scale_factor": 1,
            "pairs_manifest": self.root / "pairs.jsonl",
        }
        for image_id in ("a", "b", "c"):
            cache = self.root / "gt_cache" / f"{image_id}_0"
            cache.mkdir(parents=True)
            stats = np.zeros((3, 64), np.float32)
            stats[:, :32] = np.arange(3)[:, None]
            stats[:, 32:] = 999  # std 不能误入 mean 训练输入。
            np.save(cache / "features.npy", stats)
            images = self.root / "images" / image_id
            images.mkdir(parents=True)
            np.savez_compressed(images / "imgs.npz", svr_imgs=np.zeros((1, 16, 16, 3), np.uint8))
        self.winner = np.arange(30 * 32, dtype=np.float32).reshape(30, 32)
        np.save(self.root / "winner.npy", self.winner)
        np.save(self.root / "loser.npy", -self.winner)
        self.pair = {"image_id": "a", "winner_path": "winner.npy", "loser_path": "loser.npy", "pair_type": "geometry"}
        (self.root / "pairs.jsonl").write_text(json.dumps(self.pair) + "\n")
        self.protocol = {
            "version": "svr_preference_generation_v1",
            "splits": {"train": ["a", "b"], "validation": ["c"]},
            "condition_data_root": str(self.root / "images"),
            "checkpoint_sha256": "test-checkpoint",
        }
        self.protocol["fingerprint"] = hashlib.sha256(
            json.dumps(self.protocol, sort_keys=True, separators=(",", ":")).encode(),
        ).hexdigest()
        (self.root / "metadata").mkdir()
        (self.root / "metadata/protocol.json").write_text(json.dumps(self.protocol))
        (self.root / "status.json").write_text('{"status":"complete"}')

    def test_generated_latents_keep_exact_order_gt_uses_all_identities(self):
        (self.root / "pairs.jsonl").write_text(json.dumps({**self.pair, "metrics_evidence": {"large": list(range(100))}}))
        dataset = PreferenceDataset("training", self.config)
        self.assertEqual(len(dataset), 2)
        item = dataset[1]
        self.assertEqual(item["v_prefix"], "b")
        self.assertEqual(item["pair_image_id"], "a")
        torch.testing.assert_close(item["winner_latent"], torch.from_numpy(self.winner), rtol=0, atol=0)
        self.assertEqual(set(item["gt_latent"][:, 0].tolist()), {0.0, 1.0, 2.0})
        self.assertEqual(item["gt_latent"].shape, (30, 32))
        self.assertEqual(dataset.collate_fn([item, item])["winner_latent"].shape, (2, 30, 32))
        self.assertEqual(set(dataset.pairs[0]), {"image_id", "winner_path", "loser_path", "pair_type"})

    def test_validation_never_reads_preference_pool(self):
        (self.root / "winner.npy").unlink()
        (self.root / "loser.npy").unlink()
        item = PreferenceDataset("validation", self.config)[0]
        self.assertEqual(item["v_prefix"], "c")
        self.assertNotIn("winner_latent", item)

    def test_manifest_and_protocol_are_fingerprinted_and_frozen(self):
        provenance = preference_data_provenance(self.config)
        self.assertEqual(provenance["generation_fingerprint"], self.protocol["fingerprint"])
        self.assertEqual(
            provenance["pairs_manifest_sha256"],
            hashlib.sha256((self.root / "pairs.jsonl").read_bytes()).hexdigest(),
        )
        self.config["provenance"] = provenance
        (self.root / "pairs.jsonl").write_text(json.dumps({**self.pair, "audit": "changed"}))
        with self.assertRaisesRegex(ValueError, "changed"):
            PreferenceDataset("training", self.config)

    def test_missing_or_tampered_protocol_and_in_progress_run_rejected(self):
        path = self.root / "metadata/protocol.json"
        path.unlink()
        with self.assertRaises(FileNotFoundError):
            preference_data_provenance(self.config)
        path.write_text(json.dumps({**self.protocol, "checkpoint_sha256": "tampered"}))
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            preference_data_provenance(self.config)
        path.write_text(json.dumps(self.protocol))
        (self.root / "status.json").write_text('{"status":"running"}')
        with self.assertRaisesRegex(ValueError, "complete"):
            preference_data_provenance(self.config)

    def test_validation_identity_and_duplicate_pairs_rejected(self):
        for rows in ([{**self.pair, "image_id": "c"}], [self.pair, self.pair]):
            (self.root / "pairs.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
            with self.assertRaises(ValueError):
                PreferenceDataset("training", self.config)

    def test_deduplicated_candidate_is_not_re_padded(self):
        np.save(self.root / "winner.npy", self.winner[:8])
        with self.assertRaisesRegex(ValueError, "original float32"):
            PreferenceDataset("training", self.config)[0]


if __name__ == "__main__":
    unittest.main()
