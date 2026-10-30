"""DPO 推理只更新 policy；原始 reference、DINO 和 AE 合同保持严格。"""

from copy import deepcopy
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

from src.models.diffusion.legacy import load_preference_svr


class TinyNativeSVR(nn.Module):
    """用小网络验证加载边界；真实旧网络的数值等价另做 GPU 验证。"""

    def __init__(self):
        super().__init__()
        for name in ("p_embed", "time_embed", "net1", "fc_out", "img_fc", "img_model", "ae_model"):
            setattr(self, name, nn.Linear(2, 2))

    def inference(self, value):
        latent = self.p_embed(value) + self.time_embed(value)
        condition = self.img_fc(self.img_model(value))
        return self.fc_out(self.net1(latent + condition))


class PreferenceLegacyLoadingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.original_path = Path(self.directory.name) / "original.ckpt"
        self.preference_path = Path(self.directory.name) / "preference.ckpt"
        self.original_path.write_bytes(b"original checkpoint fixture")
        self.original = TinyNativeSVR().eval()
        self.config = {"name": "Diffusion_condition", "beta_schedule": "squaredcos_cap_v2"}
        policy = {
            key: value.clone() for key, value in self.original.state_dict().items()
            if key.split(".")[0] not in {"img_model", "ae_model"}
        }
        self.checkpoint = {
            "hyper_parameters": {
                "model": {"name": "PreferenceDiffusion", "legacy_model_config": self.config},
                "dataset": {"provenance": {
                    "source_checkpoint_sha256": hashlib.sha256(self.original_path.read_bytes()).hexdigest(),
                }},
            },
            "global_step": 5000, "epoch": 6,
            "state_dict": {
                **{f"model.policy.{key}": value.clone() for key, value in policy.items()},
                **{f"model.reference.{key}": value.clone() for key, value in policy.items()},
                **{f"model.image_encoder.{key}": value.clone()
                   for key, value in self.original.img_model.state_dict().items()},
            },
        }

    def load(self):
        torch.save(self.checkpoint, self.preference_path)
        self.loaded_original = deepcopy(self.original)
        with patch("src.models.diffusion.legacy.load_legacy_svr", return_value=(
            self.loaded_original, {"model_config": deepcopy(self.config), "native_inference_preserved": True},
        )) as loader:
            result = load_preference_svr(self.original_path, self.preference_path)
        loader.assert_called_once_with(self.original_path, "cpu", source_cache=None, dino_repo=None)
        return result

    def test_reference_policy_matches_original_native_output_exactly(self):
        model, provenance = self.load()
        value = torch.randn(3, 2)
        self.assertTrue(torch.equal(model.inference(value), self.original.inference(value)))
        for key, tensor in self.original.state_dict().items():
            self.assertTrue(torch.equal(model.state_dict()[key], tensor), key)
        self.assertTrue(provenance["native_inference_preserved"])
        self.assertTrue(provenance["preference"]["reference_equal_original"])
        self.assertTrue(provenance["preference"]["dino_equal_original"])
        self.assertEqual(provenance["preference"]["policy_tensor_count"], 10)
        self.assertEqual(provenance["preference"]["global_step"], 5000)

    def test_only_trained_policy_is_replaced(self):
        self.checkpoint["state_dict"]["model.policy.fc_out.bias"].add_(1)
        model, _ = self.load()
        self.assertFalse(model.training)
        for name in ("ae_model", "img_model"):
            for key, tensor in getattr(self.original, name).state_dict().items():
                self.assertTrue(torch.equal(getattr(model, name).state_dict()[key], tensor))
        value = torch.randn(3, 2)
        torch.testing.assert_close(model.inference(value), self.original.inference(value) + 1)

    def test_modified_frozen_reference_or_dino_is_rejected(self):
        for key in ("model.reference.net1.weight", "model.image_encoder.weight"):
            with self.subTest(key=key):
                original = self.checkpoint["state_dict"][key].clone()
                self.checkpoint["state_dict"][key].add_(1)
                with self.assertRaisesRegex(ValueError, "Frozen DPO reference/DINO"):
                    self.load()
                self.checkpoint["state_dict"][key] = original

    def test_missing_extra_policy_keys_and_tensor_mismatch_are_rejected(self):
        original = deepcopy(self.checkpoint["state_dict"])
        for mismatch in ("missing", "extra", "shape", "dtype"):
            with self.subTest(mismatch=mismatch):
                weights = deepcopy(original)
                if mismatch == "missing":
                    del weights["model.policy.p_embed.weight"]
                elif mismatch == "extra":
                    weights["model.policy.untrained.weight"] = torch.zeros(2)
                elif mismatch == "shape":
                    weights["model.policy.p_embed.weight"] = torch.zeros(3, 2)
                else:
                    weights["model.policy.p_embed.weight"] = weights["model.policy.p_embed.weight"].double()
                self.checkpoint["state_dict"] = weights
                with self.assertRaisesRegex(ValueError, "DPO (state keys|tensor shape/dtype) differ"):
                    self.load()
                for key, tensor in self.original.state_dict().items():
                    self.assertTrue(torch.equal(self.loaded_original.state_dict()[key], tensor), key)

    def test_different_source_checkpoint_is_rejected(self):
        self.original_path.write_bytes(b"another original checkpoint")
        with self.assertRaisesRegex(ValueError, "source checkpoint hash differs"):
            self.load()

    def test_different_model_configuration_is_rejected(self):
        self.checkpoint["hyper_parameters"]["model"]["legacy_model_config"] = {
            **self.config, "beta_schedule": "linear",
        }
        with self.assertRaisesRegex(ValueError, "model configuration differs"):
            self.load()


if __name__ == "__main__":
    unittest.main()
