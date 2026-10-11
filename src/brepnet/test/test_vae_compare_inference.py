"""Fair-decoding and resume contracts; these do not test research quality."""
from pathlib import Path
import pickle

import numpy as np
import pytest
import torch

from tools.vae_compare_inference import (
    atomic_npz, claim_protocol, completed_pair, decode_legacy, latent_sha256,
    load_archived_latents, load_identity_geometry, normalized_weights, read_ids, safe_checkpoint,
)
from src.brepnet.test.test_shared_boundaries import make_boundary_arrays


def test_legacy_decode_never_receives_truth_or_deduplicates():
    latent = torch.arange(96, dtype=torch.float32).reshape(3, 32)

    class FakeLegacy:
        def decode(self, encoding, v_data, v_deduplicated):
            assert set(encoding) == {"face_latents"}
            assert encoding["face_latents"] is latent
            assert v_data is None and v_deduplicated is False
            return {"face_points_local": torch.zeros(3, 16, 16, 6),
                    "face_center_scale": torch.tensor([[1., 2., 3., 2.]]).repeat(3, 1),
                    "edge_points_local": torch.zeros(2, 16, 6),
                    "edge_center_scale": torch.tensor([[4., 5., 6., 2.]]).repeat(2, 1),
                    "pred_edge_face_connectivity": torch.tensor([[0, 0, 1], [1, 1, 0]]),
                    "pred_face_adj": torch.zeros(9, dtype=torch.bool)}

    result = decode_legacy(FakeLegacy(), latent)
    np.testing.assert_array_equal(result["face_latents"], latent.numpy())
    np.testing.assert_array_equal(result["pred_face"][0, 0, 0], [1, 2, 3])
    np.testing.assert_array_equal(result["pred_edge"][0, 0], [4, 5, 6])
    assert result["pred_edge_face_connectivity"].dtype == np.int64
    assert str(result["representation"]) == "legacy_half_edges"


def test_identity_batch_keeps_exact_encoder_arrays_and_local_graph(tmp_path):
    faces, edges, conn = make_boundary_arrays()
    path = tmp_path / "data.npz"
    # No adjacency/negative-pair arrays are needed by encode(), nor any GT by decode().
    np.savez(path, sample_points_faces=faces, sample_points_lines=edges, edge_face_connectivity=conn)
    batch, truth = load_identity_geometry(path, "example", torch.device("cpu"))
    np.testing.assert_array_equal(batch["face_points"].numpy(), faces)
    np.testing.assert_array_equal(batch["edge_points"].numpy(), edges)
    np.testing.assert_array_equal(batch["edge_face_connectivity"].numpy(), conn)
    assert batch["face_counts"].tolist() == [2]
    assert not batch["attn_mask"].any()
    np.testing.assert_array_equal(truth["gt_pairs"], [[0, 1]])
    assert len(truth["gt_edge"]) == 1


def test_archived_replay_preserves_duplicates_and_variable_count(tmp_path):
    path = tmp_path / "data.npz"
    values = np.zeros((5, 32), dtype=np.float32)
    np.savez(path, face_latents=values)
    loaded = load_archived_latents(path, torch.device("cpu"))
    assert tuple(loaded.shape) == (5, 32)
    np.testing.assert_array_equal(loaded.numpy(), values)
    for invalid in (values.astype(np.float64), np.zeros((0, 32), np.float32),
                    np.full((5, 32), np.nan, np.float32), np.zeros((5, 64), np.float32)):
        np.savez(path, face_latents=invalid)
        with pytest.raises(ValueError):
            load_archived_latents(path, torch.device("cpu"))


def test_protocol_rejects_mutation_and_ids_reject_ambiguity(tmp_path):
    protocol = tmp_path / "protocol.json"
    value = {"decoder_dedup": False, "mode": "posterior_mean"}
    digest = claim_protocol(protocol, value)
    assert claim_protocol(protocol, value) == digest
    with pytest.raises(ValueError, match="differs"):
        claim_protocol(protocol, {**value, "decoder_dedup": True})
    split = tmp_path / "ids.txt"
    for bad in ("", "one\none\n", "../one\n"):
        split.write_text(bad)
        with pytest.raises(ValueError):
            read_ids(split)


def test_resume_requires_both_outputs_same_provenance_and_same_latents(tmp_path):
    paths = {method: tmp_path / method / "data.npz" for method in ("legacy", "joint")}
    z = np.zeros((2, 32), np.float32)
    payload = {"face_latents": z, "input_latent_sha256": np.asarray(latent_sha256(z)),
               "protocol_sha256": np.asarray("protocol"), "input_source_sha256": np.asarray("source"),
               "pred_face": np.zeros((2, 16, 16, 3)), "pred_edge": np.zeros((0, 16, 3)),
               "pred_edge_face_connectivity": np.zeros((0, 3), np.int64), "representation": np.asarray("test")}
    atomic_npz(paths["legacy"], payload)
    assert not completed_pair(paths, "protocol", "source")
    atomic_npz(paths["joint"], payload)
    assert completed_pair(paths, "protocol", "source")
    with pytest.raises(ValueError, match="provenance"):
        completed_pair(paths, "other", "source")
    changed = z + 1
    atomic_npz(paths["joint"], {**payload, "face_latents": changed,
                              "input_latent_sha256": np.asarray(latent_sha256(changed))})
    with pytest.raises(ValueError, match="identical"):
        completed_pair(paths, "protocol", "source")


def test_safe_checkpoint_rejects_unrecognized_pickle_and_ambiguous_keys(tmp_path):
    # pathlib is harmless here but deliberately outside the exact allowed data containers.
    path = tmp_path / "unknown.ckpt"
    torch.save({"state_dict": {"model.weight": torch.ones(1)}, "unknown": Path("example")}, path)
    previous = list(torch.serialization.get_safe_globals())
    with pytest.raises(pickle.UnpicklingError):
        safe_checkpoint(path)
    assert torch.serialization.get_safe_globals() == previous
    with pytest.raises(ValueError, match="Ambiguous"):
        normalized_weights({"state_dict": {"model.weight": torch.ones(1), "weight": torch.ones(1)}})


def test_export_representation_and_fields_are_accepted_by_post_loader(tmp_path):
    from src.brepnet.post.construct_brep import load_prediction_shape
    faces, edges, conn = make_boundary_arrays()
    for name, representation, keep in (("legacy", "legacy_half_edges", 2),
                                        ("joint", "shared_boundary_v1", 1)):
        path = tmp_path / name / "data.npz"
        payload = {"representation": np.asarray(representation), "pred_face": faces[..., :3],
                   "pred_edge": edges[:keep, ..., :3], "pred_edge_face_connectivity": conn[:keep]}
        if name == "joint":
            payload["pred_edge_confidence"] = np.ones(keep, np.float32)
        atomic_npz(path, payload)
        shape = load_prediction_shape(path)
        assert shape.representation == representation
        np.testing.assert_array_equal(shape.recon_face_points, faces[..., :3])
        np.testing.assert_array_equal(shape.recon_edge_points, edges[:keep, ..., :3])
