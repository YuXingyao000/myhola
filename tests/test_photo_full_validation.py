"""A saved candidate is reusable only with the frozen identity and artifacts."""

import pytest

from src.preference.generate import candidate_seed
from src.eval.photo_full_validation import verify_generation
from src.common.io import atomic_json, sha256_file


def test_generation_reuse_rejects_changed_input_and_artifact(tmp_path):
    mid, method = "00000001", "original_svr"
    inputs = {"image": "photo", "gt_npz": "npz", "gt_step": "step"}
    protocol = {"fingerprint": "frozen", "input_sha256": {mid: inputs}}
    for name in ("latent.npy", "data.npz"):
        (tmp_path / name).write_bytes(b"frozen output")
    row = {"method": method, "image_id": mid, "seed": candidate_seed(42, mid, 0),
           "status": "ok", "protocol_fingerprint": "frozen", "input_sha256": inputs,
           "output_sha256": {name: sha256_file(tmp_path / name) for name in ("latent.npy", "data.npz")}}
    atomic_json(tmp_path / "inference.json", row)
    assert verify_generation(protocol, method, mid, tmp_path) == row
    changed = {**protocol, "input_sha256": {mid: {**inputs, "image": "different photo"}}}
    with pytest.raises(ValueError, match="identity/input/protocol"):
        verify_generation(changed, method, mid, tmp_path)
    (tmp_path / "data.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="artifact changed"):
        verify_generation(protocol, method, mid, tmp_path)
