"""Validate coverage and provenance of the complete no-feature-matching evaluation."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[4]
    split = root / "checkpoints/joint_boundary/validation/val_contract_audit/accepted.txt"
    intended = [line.strip() for line in split.read_text().splitlines() if line.strip()]
    split_sha = hashlib.sha256(split.read_bytes()).hexdigest()
    receipt = json.loads((args.run / "launch_receipt.json").read_text())
    assert receipt["status"] == "complete"
    assert len(receipt["jobs"]) == 8 and all(j["exit_code"] == 0 for j in receipt["jobs"])
    protocols, rows = [], []
    for shard in sorted((args.run / "no_edge_feature_last64").glob("shard_*")):
        protocol = json.loads((shard / "protocol.json").read_text())
        current = [json.loads(line) for line in (shard / "metrics.jsonl").read_text().splitlines()]
        assert protocol["status"] == "complete", shard
        assert [r["id"] for r in current] == protocol["intended_ids"], shard
        assert protocol["split_sha256"] == split_sha
        assert protocol["checkpoint_global_step"] == 30000
        assert protocol["checkpoint_sha256"] == "4262f4998fe047e19c3ca21a835d74c183b7430f7a656c709ec8ed045262ada4"
        assert protocol["face_feature_dim"] == 64
        assert protocol["effective_model_config"]["freeze_encoder"] is False
        assert protocol["effective_model_config"]["loss_weights"]["feature_weight"] == 0.0
        assert protocol["latent"] == "posterior_mean" and protocol["tf32"] is False
        assert protocol["effective_dataset_config"]["is_aug"] is False
        if protocols:
            for key in ("source_hashes", "source_git_commit", "checkpoint_sha256", "effective_model_config"):
                assert protocol[key] == protocols[0][key], (shard, key)
        protocols.append(protocol)
        rows.extend(current)
    assert Counter(r["id"] for r in rows) == Counter(intended)
    assert len(intended) == 3495
    counts = Counter(r["status"] for r in rows)
    ok = [r for r in rows if r["status"] == "ok"]
    assert sum(r["edge_count"] for r in ok) == 199092
    assert all(r["edge_count"] == r["teacher_input_edge_count"] for r in ok)
    compose_max, reconstruction_max, matched_max_relative = 0.0, 0.0, 0.0
    for row in ok:
        for name, path in row["paths"].items():
            compose_max = max(compose_max, path["stats"]["compose_max_abs"])
            reconstruction_max = max(reconstruction_max, path["stats"]["gt_reconstruction_max_abs"])
            before, after = row[f"path_{name}_matched_cd_sq"], path["cases"]["pred_all"]["cd_sq"]
            assert math.isclose(before, after, rel_tol=5e-5, abs_tol=1e-10)
            matched_max_relative = max(matched_max_relative, abs(before - after) / max(abs(before), 1e-30))
    assert compose_max == 0 and reconstruction_max <= 2e-6
    result = {
        "status": "passed", "intended_count": len(intended), "record_count": len(rows),
        "status_counts": dict(counts), "gt_edge_count": sum(r["edge_count"] for r in ok),
        "checkpoint_sha256": protocols[0]["checkpoint_sha256"], "checkpoint_global_step": 30000,
        "split_sha256": split_sha, "source_git_commit": protocols[0]["source_git_commit"],
        "all_shards_identical_code_and_model": True,
        "compose_max_abs": compose_max, "gt_reconstruction_max_abs": reconstruction_max,
        "matched_cd_decomposition_max_relative_difference": matched_max_relative,
        "started_utc": receipt["started_utc"], "completed_utc": receipt["completed_utc"],
        "run": str(args.run.resolve()), "source_hashes": protocols[0]["source_hashes"],
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "source_hashes"}))


if __name__ == "__main__":
    main()
