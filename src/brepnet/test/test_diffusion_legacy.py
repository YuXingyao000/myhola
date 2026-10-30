"""旧采样的去重规则；真实 GPU 等价测试由显式环境变量启用。"""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from src.brepnet.diffusion.legacy import (
    _greedy_mapping, encode_original_gt, load_legacy_svr, sample_candidates,
)


class LegacyDeduplicationTests(unittest.TestCase):
    def test_greedy_duplicates_compare_retained_faces_only(self):
        # 0≈1、1≈2、0≉2：旧规则保留 0 和 2，不做传递聚类。
        values = torch.tensor([0.0, 0.006, 0.012, 0.020])
        distance = (values[:, None] - values[None, :]).abs()
        keep, mapping = _greedy_mapping(distance, 0.01)
        self.assertEqual(keep.tolist(), [0, 2])
        self.assertEqual(mapping.tolist(), [0, 0, 1, 1])

    def test_threshold_equality_is_not_duplicate(self):
        keep, mapping = _greedy_mapping(torch.tensor([[0.0, 0.01], [0.01, 0.0]]), 0.01)
        self.assertEqual(keep.tolist(), [0, 1])
        self.assertEqual(mapping.tolist(), [0, 1])


class OriginalGTEncodingTests(unittest.TestCase):
    def test_mean_std_and_native_encoder_inputs(self):
        class FakeEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(()))
                self.df = 32
                self.gaussian_proj = torch.nn.Identity()

            def encode(self, batch, is_test):
                self.batch = batch
                self.grad_enabled = torch.is_grad_enabled()
                posterior = torch.zeros((2, 32, 2), device=self.anchor.device)
                posterior[:, :, 0] = torch.arange(32)
                posterior[:, :, 1] = 2 * np.log(2)
                return {"face_features": posterior.reshape(2, 64)}

        encoder = FakeEncoder().eval()
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "data.npz"
            np.savez(source, sample_points_faces=np.zeros((2, 16, 16, 6)),
                     sample_points_lines=np.zeros((3, 16, 6)),
                     edge_face_connectivity=np.array([[0, 0, 0], [1, 0, 1], [2, 1, 0]]))
            statistics = encode_original_gt(SimpleNamespace(ae_model=encoder), source)
        np.testing.assert_array_equal(statistics[:, :32], np.broadcast_to(np.arange(32), (2, 32)))
        np.testing.assert_allclose(statistics[:, 32:], 2.0)
        self.assertEqual(statistics.dtype, np.float32)
        self.assertEqual(encoder.batch["edge_face_connectivity"].tolist(), [[1, 0, 1], [2, 1, 0]])
        self.assertEqual(encoder.batch["num_face_record"].tolist(), [2])
        self.assertFalse(encoder.batch["attn_mask"].any())
        self.assertFalse(encoder.grad_enabled)


@unittest.skipUnless(os.environ.get("HOLA_LEGACY_TEST_CHECKPOINT"), "Explicit original checkpoint required.")
class LegacyNativeSamplingTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("HOLA_LEGACY_TEST_GT"), "Explicit original GT required.")
    def test_original_gt_matches_native_posterior(self):
        torch.set_num_threads(1)
        model, _ = load_legacy_svr(os.environ["HOLA_LEGACY_TEST_CHECKPOINT"], "cuda:0")
        captured = {}
        hook = model.ae_model.gaussian_proj.register_forward_pre_hook(
            lambda module, inputs: captured.__setitem__("features", inputs[0]),
        )
        cpu_rng = torch.random.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state()
        try:
            encoded = encode_original_gt(model, os.environ["HOLA_LEGACY_TEST_GT"])
        finally:
            hook.remove()
        self.assertTrue(torch.equal(cpu_rng, torch.random.get_rng_state()))
        self.assertTrue(torch.equal(cuda_rng, torch.cuda.get_rng_state()))
        with torch.no_grad():
            mean, _, native = model.ae_model.sample(captured["features"], True)
        np.testing.assert_array_equal(encoded, native.cpu().numpy())
        np.testing.assert_array_equal(encoded[:, :32], mean.cpu().numpy())
        self.assertEqual(encoded.shape[1], 64)
        self.assertTrue(np.isfinite(encoded).all())
        self.assertTrue((encoded[:, 32:] > 0).all())

    def test_same_seed_matches_native_inference(self):
        import torchvision.transforms as transforms

        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        device = torch.device("cuda:0")
        checkpoint = os.environ["HOLA_LEGACY_TEST_CHECKPOINT"]
        image_path = Path(os.environ["HOLA_LEGACY_TEST_IMAGE"])
        with np.load(image_path) as archive:
            image = archive["svr_imgs"][0]
        transform = transforms.Compose([
            transforms.ToPILImage(), transforms.Resize((224, 224)), transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        image = transform(image)[None].to(device)
        model, provenance = load_legacy_svr(checkpoint, device)
        seed = 486731
        original_step = model.noise_scheduler.step
        captured = {}

        def capture_final_latent(*args, **kwargs):
            result = original_step(*args, **kwargs)
            captured["latent"] = result.prev_sample
            return result

        model.noise_scheduler.step = capture_final_latent
        torch.manual_seed(seed)
        started = time.perf_counter()
        with torch.no_grad():
            native = model.inference(1, device, v_data={"conditions": {
                "imgs": image[:, None], "img_id": torch.zeros((1, 1), dtype=torch.long, device=device),
            }})[0]
        native_latent = captured["latent"][0].cpu()
        native_seconds = time.perf_counter() - started
        model.noise_scheduler.step = original_step
        cpu_rng = torch.random.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state(device)
        started = time.perf_counter()
        sample = sample_candidates(model, image, [seed])[0]
        torch.cuda.synchronize(device)
        sample_seconds = time.perf_counter() - started
        self.assertTrue(torch.equal(cpu_rng, torch.random.get_rng_state()))
        self.assertTrue(torch.equal(cuda_rng, torch.cuda.get_rng_state(device)))
        self.assertTrue(torch.equal(native_latent, sample["latent"]))
        self.assertEqual(sample["latent"].shape, (30, 32))
        for key, value in native.items():
            expected = value.cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
            np.testing.assert_array_equal(expected, sample["raw"][key], err_msg=key)
        np.testing.assert_array_equal(
            sample["latent"][sample["latent_keep_indices"]].numpy(), sample["raw"]["face_features"],
        )
        result = {
            "source_commit": provenance["source_commit"], "checkpoint": checkpoint,
            "seed": seed, "image": str(image_path), "latent_bitwise_equal": True,
            "raw_bitwise_equal": True, "caller_rng_preserved": True,
            "native_seconds": native_seconds, "sample_seconds": sample_seconds,
            "latent_keep_indices": sample["latent_keep_indices"].tolist(),
            "geometry_keep_indices": sample["geometry_keep_indices"].tolist(),
            "raw_shapes": {key: list(value.shape) for key, value in sample["raw"].items()},
        }
        report = os.environ.get("HOLA_LEGACY_TEST_REPORT")
        if report:
            Path(report).write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    unittest.main()
