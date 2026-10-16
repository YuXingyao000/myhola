"""Inspect existing DDP smoke checkpoints; does not train or run inference."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-step", type=int, choices=(1, 2), required=True)
    args = parser.parse_args()
    plan = json.loads((ROOT / "commands/plan.json").read_text())
    run = Path(plan["run_dir"])
    path = run / "weights/last.ckpt"
    checkpoint = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    source_path = Path(plan["source"])
    source = torch.load(source_path, map_location="cpu", mmap=True, weights_only=False)
    assert checkpoint["global_step"] == args.expected_step
    config = checkpoint["hyper_parameters"]
    inter = config["model"]["intersection"]
    expected = {"kind": "local_attention", "block_type": "transformer_decoder",
                "hidden_dim": 512, "num_heads": 8, "num_layers": 8,
                "ffn_dim": 2048, "dropout": 0.1, "activation": "relu"}
    assert all(inter[key] == value for key, value in expected.items())
    assert config["trainer"]["devices"] == 8
    assert config["trainer"]["batch_size"] == 16
    assert config["trainer"]["precision"] == "bf16-mixed"
    assert config["trainer"]["wandb"]["enabled"] is False
    assert config["model"]["freeze_encoder"] is True
    state = checkpoint["state_dict"]
    frozen = [key for key in state if key.startswith(("model.encoder.", "model.posterior."))]
    assert len(frozen) == 243
    assert all(torch.equal(state[key], source["state_dict"][key]) for key in frozen)
    inter_state = {key: value for key, value in state.items() if key.startswith("model.decoder.inter.")}
    assert len(inter_state) == 157
    assert sum(value.numel() for value in inter_state.values()) == 34015296
    optimizer = checkpoint["optimizer_states"]
    assert len(optimizer) == 1 and optimizer[0]["state"]
    steps = sorted({int(item["step"].item()) for item in optimizer[0]["state"].values()})
    assert steps == [args.expected_step]
    if args.expected_step == 2:
        assert config["model"]["init_refinement_checkpoint"] == "/nonexistent/full-capacity-migration-source.ckpt"
        assert config["trainer"]["resume_from_checkpoint"] == str(path)
    else:
        assert config["model"]["init_refinement_checkpoint"] == str(source_path)
        assert config["trainer"]["resume_from_checkpoint"] is None
    record = {
        "status": "passed", "checked_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(path), "checkpoint_sha256_at_validation": sha(path),
        "epoch": checkpoint["epoch"], "global_step": checkpoint["global_step"],
        "optimizer_steps": steps, "optimizer_state_count": len(optimizer[0]["state"]),
        "intersection": inter, "intersection_state_keys": len(inter_state),
        "intersection_parameters": sum(value.numel() for value in inter_state.values()),
        "frozen_encoder_posterior_tensors_equal_to_source": len(frozen),
        "saved_initialization_source": config["model"]["init_refinement_checkpoint"],
        "saved_resume_checkpoint": config["trainer"]["resume_from_checkpoint"],
    }
    (ROOT / f"step{args.expected_step}_audit.json").write_text(json.dumps(record, indent=2) + "\n")
    if args.expected_step == 2:
        source_hash = sha(source_path)
        assert source_hash == "9b18509de2a21f50d8fad41b8c37695842fe2e92745dc779713b0ecac06e2107"
        summary = {
            "status": "passed", "completed_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "Eight-GPU one-step save and resume to cumulative step two; implementation validation only, no production training or quality claim.",
            "run_dir": str(run), "source_checkpoint": str(source_path), "source_sha256": source_hash,
            "devices": 8, "batch_per_gpu": 16, "precision": "bf16-mixed", "seed": 42,
            "wandb_enabled": False, "initial_max_steps": 1, "resumed_global_step": 2,
            "optimizer_steps": steps, "resume_ignored_nonexistent_migration_source": True,
            "frozen_encoder_posterior_tensors_equal_to_source": len(frozen),
            "command_plan": str(ROOT / "commands/plan.json"), "command_scripts": plan["command_scripts"],
            "lists": plan["lists"],
            "invocation_metadata": [str(p) for p in sorted((run / "metadata/invocations").glob("*/command.json"))],
            "first_checkpoint": json.loads((ROOT / "step1_audit.json").read_text()),
            "resumed_checkpoint": record,
        }
        assert len(summary["invocation_metadata"]) == 2
        (ROOT.parent / "ddp_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
