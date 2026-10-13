"""Strict loading, source provenance and replay behavior for VAE evaluation."""

from copy import deepcopy
import json

import numpy as np
import pytest
import torch

from src.brepnet.models.vae import AutoEncoder_intersection
from src.brepnet.models.vae import AutoEncoder_light
from src.brepnet.test.test_intersection_vae import tiny_config
from src.brepnet.test.test_shared_boundaries import make_boundary_arrays
from tools.intersection_evaluate_inference import (
    audit_encoder_state, audit_identity_means, completed_prediction, legacy_configuration,
    load_intersection, main, read_reference_input, read_reference_protocol,
)
from tools.vae_compare_inference import (
    atomic_npz, file_sha256, json_digest, latent_sha256, load_identity_geometry,
)


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def make_models(tmp_path):
    config = tiny_config(name="AutoEncoder_light", latent_channels=8)
    legacy = AutoEncoder_light(config).eval()
    legacy_path = tmp_path / "legacy.ckpt"
    legacy_checkpoint = {
        "state_dict": {f"model.{key}": value for key, value in legacy.state_dict().items()},
        "hyper_parameters": {"model": config}, "epoch": 1, "global_step": 10,
    }
    torch.save(legacy_checkpoint, legacy_path)
    new_config = {**config, "name": "AutoEncoder_intersection", "freeze_encoder": True,
                  "init_legacy_checkpoint": str(legacy_path)}
    new = AutoEncoder_intersection(new_config).eval()
    path = tmp_path / "intersection.ckpt"
    torch.save({
        "state_dict": {f"model.{key}": value for key, value in new.state_dict().items()},
        "hyper_parameters": {"model": new_config}, "epoch": 2, "global_step": 20,
    }, path)
    return legacy, new, legacy_path, path, legacy_checkpoint, new_config


def make_reference(tmp_path, model, legacy_path):
    model_id = "example"
    data_root, reference_root = tmp_path / "data", tmp_path / "reference"
    source = data_root / model_id / "data.npz"
    source.parent.mkdir(parents=True)
    faces, edges, conn = make_boundary_arrays()
    np.savez(source, sample_points_faces=faces, sample_points_lines=edges, edge_face_connectivity=conn)
    split = tmp_path / "ids.txt"
    split.write_text(model_id + "\n")
    protocol = {
        "version": "vae_compare_inference_v1", "mode": "posterior_mean",
        "split_sha256": file_sha256(split), "intended_count": 1,
        "legacy_checkpoint_sha256": file_sha256(legacy_path),
        "identity_view": True, "decoder_gt_topology": False,
        "decoder_face_deduplication": False, "legacy_threshold": .5,
        "numeric_settings": {"dtype": "float32", "autocast": False,
                             "matmul_allow_tf32": False, "cudnn_allow_tf32": False},
    }
    protocol_path = reference_root / "metadata/inference/posterior_mean/protocol.json"
    protocol_path.parent.mkdir(parents=True)
    protocol_path.write_text(json.dumps(protocol))
    batch, _ = load_identity_geometry(source, model_id, torch.device("cpu"))
    with torch.inference_mode():
        latent = model.sample(model.encode(batch, v_test=True)["face_features"], v_is_test=True)[0]
        prediction = model.decode_latents(latent)
    arrays = {**prediction, "protocol_sha256": np.asarray(json_digest(protocol)),
              "comparison_mode": np.asarray("posterior_mean"),
              "input_source_sha256": np.asarray(file_sha256(source)),
              "input_latent_sha256": np.asarray(latent_sha256(latent.numpy()))}
    raw = reference_root / "raw_predictions/posterior_mean/legacy" / model_id / "data.npz"
    atomic_npz(raw, arrays)
    return reference_root, data_root, split, protocol, arrays


def test_strict_meta_loading_and_renamed_encoder_audit(tmp_path):
    _, expected, legacy_path, new_path, legacy_checkpoint, config = make_models(tmp_path)
    actual, audit = load_intersection(new_path, legacy_path, torch.device("cpu"), ["example"])
    assert audit["encoder_equal_tensor_count"] > 0
    assert audit["encoder_equal"] and audit["strict_load"]
    assert not actual.training and not any(param.requires_grad for param in actual.parameters())
    for name, value in actual.state_dict().items():
        torch.testing.assert_close(value, expected.state_dict()[name], atol=0, rtol=0)
    old = legacy_configuration(legacy_checkpoint)
    weights = dict(expected.state_dict())
    key = next(name for name in weights if name.startswith("encoder."))
    weights[key] = weights[key] + 1
    with pytest.raises(ValueError, match="weights differ"):
        audit_encoder_state(weights, {k.removeprefix("model."): v for k, v in legacy_checkpoint["state_dict"].items()}, config, old)
    with pytest.raises(ValueError, match="gaussian_weights"):
        audit_encoder_state(expected.state_dict(), {k.removeprefix("model."): v for k, v in legacy_checkpoint["state_dict"].items()}, {**config, "gaussian_weights": 1e-4}, old)
    damaged = torch.load(new_path, weights_only=True)
    damaged["state_dict"].pop(next(key for key in damaged["state_dict"] if key.startswith("model.decoder.")))
    torch.save(damaged, new_path)
    with pytest.raises(RuntimeError, match="Missing key"):
        load_intersection(new_path, legacy_path, torch.device("cpu"), [])


def test_local_checkpoint_replay_does_not_require_old_migration_source(tmp_path):
    _, _, legacy_path, refinement_path, _, config = make_models(tmp_path)
    local_config = {
        **deepcopy(config), "init_legacy_checkpoint": None,
        "init_refinement_checkpoint": str(refinement_path),
        "intersection": {
            "kind": "local_attention", "hidden_dim": 16, "num_heads": 2,
            "num_layers": 2, "ffn_dim": 32, "dropout": 0.,
            "candidate_chunk_size": 3, "checkpoint_candidates": True,
        },
    }
    expected = AutoEncoder_intersection(local_config).eval()
    local_path = tmp_path / "local.ckpt"
    torch.save({
        "state_dict": {f"model.{key}": value for key, value in expected.state_dict().items()},
        "hyper_parameters": {"model": local_config}, "epoch": 3, "global_step": 30,
    }, local_path)
    refinement_path.unlink()
    assert legacy_path.is_file()  # The independent encoder audit still needs its baseline.
    actual, audit = load_intersection(local_path, legacy_path, torch.device("cpu"), ["example"])
    assert audit["strict_load"] and audit["encoder_equal"]
    assert audit["checkpoint_model_config"]["init_refinement_checkpoint"] == str(refinement_path)
    assert actual.decoder.intersection_kind == "local_attention"
    assert not actual.training and not any(param.requires_grad for param in actual.parameters())
    for name, value in actual.state_dict().items():
        torch.testing.assert_close(value, expected.state_dict()[name], atol=0, rtol=0)
    latent = torch.randn(3, expected.face_feature_dim)
    with torch.inference_mode():
        first, second = expected.decode_latents(latent), actual.decode_latents(latent)
    assert first.keys() == second.keys()
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])


def test_reference_hashes_and_actual_source_are_checked(tmp_path):
    _, model, legacy_path, _, _, _ = make_models(tmp_path)
    reference, data, split, protocol, arrays = make_reference(tmp_path, model, legacy_path)
    loaded_protocol, digest = read_reference_protocol(reference, "posterior_mean", file_sha256(split), 1, file_sha256(legacy_path))
    z, provenance = read_reference_input(reference, data, "example", "posterior_mean", loaded_protocol, digest)
    np.testing.assert_array_equal(z, arrays["face_latents"])
    assert provenance["input_latent_sha256"] == latent_sha256(z)
    path = reference / "raw_predictions/posterior_mean/legacy/example/data.npz"
    for field, text in (("input_latent_sha256", "latent digest"),
                        ("protocol_sha256", "protocol digest"),
                        ("input_source_sha256", "source file")):
        atomic_npz(path, {**arrays, field: np.asarray("changed")})
        with pytest.raises(ValueError, match=text):
            read_reference_input(reference, data, "example", "posterior_mean", protocol, digest)
    atomic_npz(path, arrays)
    with (data / "example/data.npz").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="source file"):
        read_reference_input(reference, data, "example", "posterior_mean", protocol, digest)


def test_archived_source_and_duplicate_latents_are_preserved(tmp_path):
    _, model, legacy_path, _, _, _ = make_models(tmp_path)
    reference, data, _, protocol, arrays = make_reference(tmp_path, model, legacy_path)
    original = tmp_path / "archive/example/data.npz"
    original.parent.mkdir(parents=True)
    z = arrays["face_latents"].copy()
    z[1] = z[0]
    np.savez(original, face_latents=z)
    protocol = {**protocol, "mode": "archived_latents", "identity_view": False,
                "latent_root": str(tmp_path / "archive")}
    digest = json_digest(protocol)
    archived = {**arrays, "face_latents": z, "input_latent_sha256": np.asarray(latent_sha256(z)),
                "input_source_sha256": np.asarray(file_sha256(original)),
                "protocol_sha256": np.asarray(digest), "comparison_mode": np.asarray("archived_latents")}
    atomic_npz(reference / "raw_predictions/archived_latents/legacy/example/data.npz", archived)
    actual, _ = read_reference_input(reference, data, "example", "archived_latents", protocol, digest)
    np.testing.assert_array_equal(actual, z)
    assert len(actual) == 2


def test_mean_audit_records_numerical_error_and_resume_rejects_changed_inputs(tmp_path):
    _, model, legacy_path, _, _, _ = make_models(tmp_path)
    reference, data, _, protocol, arrays = make_reference(tmp_path, model, legacy_path)
    digest = json_digest(protocol)
    audit = audit_identity_means(model, ["example"], reference, data, protocol, digest, torch.device("cpu"))
    assert audit["passed"] and audit["max_abs_error"] == 0
    z, provenance = read_reference_input(reference, data, "example", "posterior_mean", protocol, digest)
    output = tmp_path / "output.npz"
    values = {**arrays, **{key: np.asarray(value) for key, value in provenance.items()},
              "protocol_sha256": np.asarray("new-protocol")}
    assert not completed_prediction(output, "new-protocol", provenance, z, "posterior_mean")
    atomic_npz(output, values)
    assert completed_prediction(output, "new-protocol", provenance, z, "posterior_mean")
    with pytest.raises(ValueError, match="provenance mismatch"):
        completed_prediction(output, "new-protocol", {**provenance, "reference_raw_sha256": "changed"}, z, "posterior_mean")
    with torch.no_grad():
        next(model.posterior.parameters()).add_(1)
    audit = audit_identity_means(model, ["example"], reference, data, protocol, digest, torch.device("cpu"))
    assert not audit["passed"] and audit["max_abs_error"] > 0


@pytest.mark.parametrize("method", ["intersection", "local"])
def test_cpu_entrypoint_replay_resume_and_preserved_reference(tmp_path, method):
    _, model, legacy_path, checkpoint, _, config = make_models(tmp_path)
    if method == "local":
        local_config = {**deepcopy(config), "init_legacy_checkpoint": None,
                        "init_refinement_checkpoint": str(checkpoint),
                        "intersection": {"kind": "local_attention", "hidden_dim": 16,
                                         "num_heads": 2, "num_layers": 1, "ffn_dim": 32,
                                         "dropout": 0., "candidate_chunk_size": 3}}
        model = AutoEncoder_intersection(local_config).eval()
        checkpoint = tmp_path / "local.ckpt"
        torch.save({"state_dict": {f"model.{key}": value for key, value in model.state_dict().items()},
                    "hyper_parameters": {"model": local_config}, "epoch": 3, "global_step": 30}, checkpoint)
    reference, data, split, _, arrays = make_reference(tmp_path, model, legacy_path)
    output = tmp_path / "evaluation"
    source = reference / "raw_predictions/posterior_mean/legacy/example/data.npz"
    reference_sha = file_sha256(source)
    args = ["--checkpoint", str(checkpoint), "--legacy-checkpoint", str(legacy_path),
            "--reference-root", str(reference), "--data-root", str(data), "--split-list", str(split),
            "--output-root", str(output), "--mode", "posterior_mean", "--device", "cpu", "--method", method]
    assert main(args) == 0
    result_path = output / "raw_predictions/posterior_mean" / method / "example/data.npz"
    protocol = json.loads((output / "metadata/inference/posterior_mean/protocol.json").read_text())
    assert protocol["method"] == method
    if method == "local":
        assert "src/brepnet/models/blocks.py" in protocol["source_sha256"]
    with np.load(result_path, allow_pickle=False) as actual:
        np.testing.assert_array_equal(actual["face_latents"], arrays["face_latents"])
        assert str(actual["reference_protocol_sha256"]) == str(arrays["protocol_sha256"])
        assert str(actual["protocol_sha256"]) != str(arrays["protocol_sha256"])
    assert main([*args, "--resume"]) == 0
    statuses = [json.loads(line) for line in (output / "metadata/inference/posterior_mean/shard-000.jsonl").read_text().splitlines()]
    assert [row["status"] for row in statuses] == ["ok", "resumed_skip"]
    assert file_sha256(source) == reference_sha
    with pytest.raises(FileExistsError):
        main(args)


def test_mean_audit_tolerates_cuda_roundoff_but_rejects_real_drift(tmp_path):
    _, model, legacy_path, _, _, _ = make_models(tmp_path)
    reference, data, _, protocol, arrays = make_reference(tmp_path, model, legacy_path)
    expected = arrays["face_latents"].copy()
    expected[0, 0] = 0.0  # Exercise absolute tolerance where relative error cannot help.
    arrays = {**arrays, "face_latents": expected,
              "input_latent_sha256": np.asarray(latent_sha256(expected))}
    atomic_npz(reference / "raw_predictions/posterior_mean/legacy/example/data.npz", arrays)

    class ControlledMean:
        def __init__(self, perturbation):
            self.perturbation = perturbation

        def encode(self, batch, v_test):
            assert v_test is True
            return {"face_features": torch.from_numpy(expected.copy())}

        def sample(self, features, v_is_test):
            assert v_is_test is True
            result = features.clone()
            result[0, 0] += self.perturbation
            return result, None, None

    for perturbation, should_pass in ((4.3e-6, True), (2e-5, False), (1e-3, False)):
        audit = audit_identity_means(
            ControlledMean(perturbation), ["example"], reference, data,
            protocol, json_digest(protocol), torch.device("cpu"),
        )
        assert audit["passed"] is should_pass
        assert audit["max_abs_error"] == pytest.approx(perturbation)
