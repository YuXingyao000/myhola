"""Audit validation-selected curve-conditioning weights without running inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import subprocess
from pathlib import Path

import torch
from wandb.proto import wandb_internal_pb2
from wandb.sdk.internal.datastore import DataStore


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False, mmap=True)


def model_checkpoint_state(value: dict) -> dict:
    candidates = [
        state for key, state in value["callbacks"].items()
        if key.startswith("ModelCheckpoint") and state.get("monitor") == "val/loss"
    ]
    if len(candidates) != 1:
        raise ValueError("Expected exactly one val/loss ModelCheckpoint callback")
    return candidates[0]


def history(run: Path) -> tuple[list[dict], list[dict], Path]:
    paths = sorted({path.resolve() for path in run.glob("wandb/*/*.wandb")})
    if len(paths) != 1:
        raise ValueError(f"Expected one local W&B stream, found {paths}")
    store = DataStore()
    store.open_for_scan(str(paths[0]))
    rows = []
    while True:
        raw = store.scan_data()
        if raw is None:
            break
        record = wandb_internal_pb2.Record()
        record.ParseFromString(raw)
        if record.WhichOneof("record_type") == "history":
            rows.append({
                item.key or "/".join(item.nested_key): json.loads(item.value_json)
                for item in record.history.item
            })
    return rows, [row for row in rows if "val/loss" in row], paths[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)

    latest = checkpoint(args.run / "weights/last.ckpt")
    selector = model_checkpoint_state(latest)
    selected_path = Path(selector["best_model_path"])
    if selected_path.parent != args.run / "weights":
        raise ValueError("Best checkpoint points outside the requested run")
    selected = checkpoint(selected_path)
    baseline = checkpoint(args.baseline)
    selected_state, baseline_state = selected["state_dict"], baseline["state_dict"]
    added = sorted(set(selected_state) - set(baseline_state))
    removed = sorted(set(baseline_state) - set(selected_state))
    expected_prefix = "model.decoder.edge_geometry.conditioning."
    assert len(added) == 6 and all(key.startswith(expected_prefix) for key in added)
    assert not removed
    assert all(selected_state[key].shape == baseline_state[key].shape for key in baseline_state)
    frozen_keys = [key for key in baseline_state if key.startswith(("model.encoder.", "model.posterior."))]
    frozen_changed = [key for key in frozen_keys if not torch.equal(selected_state[key], baseline_state[key])]
    assert frozen_keys and not frozen_changed
    changed_old_keys = [key for key in baseline_state if not torch.equal(selected_state[key], baseline_state[key])]
    conditioning = {
        key: {
            "shape": list(selected_state[key].shape),
            "numel": selected_state[key].numel(),
            "nonzero": torch.count_nonzero(selected_state[key]).item(),
            "abs_max": selected_state[key].abs().max().item(),
            "l2_norm": selected_state[key].float().norm().item(),
        }
        for key in added
    }
    assert sum(item["numel"] for item in conditioning.values()) == 239424
    assert all(item["nonzero"] > 0 for item in conditioning.values())

    rows, validations, wandb_path = history(args.run)
    best_validation = min(validations, key=lambda row: row["val/loss"])
    callback_score = float(selector["best_model_score"])
    assert best_validation["epoch"] == selected["epoch"]
    assert best_validation["trainer/global_step"] + 1 == selected["global_step"]
    assert best_validation["val/loss"] == callback_score
    _, baseline_validations, baseline_wandb = history(args.baseline.parent.parent)
    baseline_row = next(row for row in baseline_validations if row["epoch"] == baseline["epoch"])
    assert baseline_row["trainer/global_step"] + 1 == baseline["global_step"]

    run_meta = json.loads((args.run / "metadata/run.json").read_text())
    invocation = args.run / "metadata/invocations" / run_meta["initial_invocation_id"]
    git_meta = json.loads((invocation / "git.json").read_text())
    diff_text = (invocation / "git.diff").read_text()
    training_paths = ["src/brepnet/models/vae.py", "src/brepnet/models/blocks.py", "src/brepnet/train.py", "src/brepnet/dataset.py"]
    source_checks = {}
    for relative in training_paths:
        source = subprocess.check_output(["git", "show", f"{git_meta['commit']}:{relative}"], cwd=args.repo)
        current = (args.repo / relative).read_bytes()
        source_checks[relative] = {
            "training_commit_sha256": hashlib.sha256(source).hexdigest(),
            "evaluation_sha256": hashlib.sha256(current).hexdigest(),
            "exact_match": source == current,
            "mentioned_in_training_dirty_diff": f"diff --git a/{relative} b/{relative}" in diff_text,
        }
        assert source == current
        assert not source_checks[relative]["mentioned_in_training_dirty_diff"]
    hp = selected["hyper_parameters"]
    train_cfg, dataset_cfg, model_cfg = hp["trainer"], hp["dataset"], hp["model"]
    splits = {}
    for split in ("train", "val", "test"):
        path = Path(dataset_cfg[f"{split}_dataset"])
        if not path.is_absolute():
            path = args.repo / path
        values = [line.strip() for line in path.read_text().splitlines() if line.strip()]
        splits[split] = {"path": str(path), "sha256": sha256(path), "count": len(values), "unique_count": len(set(values))}
    inter_layers = sorted({int(match.group(1)) for key in selected_state if (match := re.match(r"model\.decoder\.inter\.layers\.(\d+)\.", key))})
    assert inter_layers == list(range(8))
    global_batch = train_cfg["devices"] * train_cfg["batch_size"]
    curves_columns = ["epoch", "trainer/global_step", "val/loss", "val/edge_xyz", "val/face_xyz", "val/incidence", "val/twin", "val/refined_edge_bbox", "val/refined_edge_local", "val/refined_face_bbox", "val/refined_face_local"]
    comparisons = {
        key: {"baseline": value, "selected": best_validation[key], "relative_change_percent": (best_validation[key] / value - 1) * 100 if value else None}
        for key, value in baseline_row.items() if key.startswith("val/") and key in best_validation
    }
    report = {
        "selection_policy": "Use last.ckpt ModelCheckpoint(val/loss, min) best_model_path; no test metric used for checkpoint selection.",
        "selected": {"path": str(selected_path), "sha256": sha256(selected_path), "epoch": selected["epoch"], "global_step": selected["global_step"], "validation_loss": callback_score, "validation_metrics": {key: value for key, value in best_validation.items() if key.startswith("val/")}},
        "last_saved": {"path": str(args.run / "weights/last.ckpt"), "epoch": latest["epoch"], "global_step": latest["global_step"], "last_full_validation_loss": validations[-1]["val/loss"]},
        "saved_top_k": [{"path": path, "validation_loss": float(score)} for path, score in sorted(selector["best_k_models"].items(), key=lambda item: float(item[1]))],
        "baseline": {"path": str(args.baseline), "sha256": sha256(args.baseline), "epoch": baseline["epoch"], "global_step": baseline["global_step"], "validation_loss": baseline_row["val/loss"]},
        "state_compatibility": {"baseline_keys": len(baseline_state), "selected_keys": len(selected_state), "all_old_keys_present_and_same_shapes": True, "frozen_encoder_posterior_keys": len(frozen_keys), "frozen_keys_bitwise_equal": True, "old_decoder_keys_changed": len(changed_old_keys), "total_parameters": sum(value.numel() for value in selected_state.values()), "conditioning_parameters": conditioning},
        "actual_intersection_tensor_shapes": {"layer_indices": inter_layers, "attention_in_proj": list(selected_state['model.decoder.inter.layers.0.self_attn.in_proj_weight'].shape), "ffn_linear1": list(selected_state['model.decoder.inter.layers.0.linear1.weight'].shape)},
        "exposure": {"global_train_batch": global_batch, "steps_per_completed_epoch": selected['loops']['fit_loop']['epoch_loop.batch_progress']['current']['completed'], "selected_completed_epochs": selected["epoch"] + 1, "selected_additional_steps": selected["global_step"], "selected_additional_object_presentations": selected["global_step"] * global_batch, "last_saved_completed_epochs": latest["epoch"] + 1, "last_saved_additional_steps": latest["global_step"], "last_saved_additional_object_presentations": latest["global_step"] * global_batch, "latest_logged_train_step_zero_based": max(row.get("trainer/global_step", -1) for row in rows), "note": "Fresh optimizer after baseline initialization. This is not an equal-training-budget ablation of conditioning; the baseline checkpoint had 3993 steps. Training loader uses drop_last=True. Log step is zero-based relative to saved global_step. Some logged training is newer than last saved checkpoint."},
        "validation_vs_baseline": comparisons,
        "validation_history": {"source": str(wandb_path), "baseline_source": str(baseline_wandb), "num_complete_validations": len(validations), "first_validation_loss": validations[0]["val/loss"], "best_validation_loss": best_validation["val/loss"], "last_validation_loss": validations[-1]["val/loss"], "best_runtime_seconds": best_validation.get("_runtime"), "last_validation_runtime_seconds": validations[-1].get("_runtime"), "curve_columns": curves_columns, "curve_rows": [[row.get(key) for key in curves_columns] for row in validations]},
        "splits": splits,
        "model_config": model_cfg,
        "dataset_config": dataset_cfg,
        "trainer_config": train_cfg,
        "provenance": {"training_commit": git_meta["commit"], "training_branch": git_meta["branch"], "training_dirty_diff_sha256": sha256(invocation / "git.diff"), "source_checks": source_checks, "optimizer_type_from_training_source": "Adam", "optimizer_param_groups": [{key: value for key, value in group.items() if key != "params"} | {"parameter_tensor_count": len(group["params"])} for group in selected["optimizer_states"][0]["param_groups"]]},
        "limitations": ["Validation losses are training objectives, not CAD tolerance or valid-solid metrics.", "Current evaluation will determine downstream geometry and solid validity; no improvement conclusion follows from this audit.", "Comparison to the original refinement checkpoint includes substantial additional training exposure."],
    }
    # Keep the existing evaluation report's audit contract while retaining the
    # additional checkpoint/tensor/source checks above.
    report["checkpoint_recommendation"] = {
        key: report["selected"][key]
        for key in ("path", "sha256", "epoch", "global_step")
    } | {"best_model_score": callback_score}
    report["last_checkpoint"] = report["last_saved"]
    report["last_logged_global_step"] = report["exposure"]["latest_logged_train_step_zero_based"]
    curve_summary = report.pop("validation_history")
    curve_summary.pop("curve_columns")
    curve_summary.pop("curve_rows")
    report["validation_history_summary"] = curve_summary
    report["validation_history"] = [
        {key: row.get(key) for key in curves_columns} for row in validations
    ]
    losses = [row["val/loss"] for row in validations]
    report["val_trends"] = {"loss": {
        "first_three_mean": statistics.mean(losses[:3]),
        "last_three_mean": statistics.mean(losses[-3:]),
        "last_vs_first_three_pct": (statistics.mean(losses[-3:]) / statistics.mean(losses[:3]) - 1) * 100,
        "last_eight_std_population": statistics.pstdev(losses[-8:]),
    }}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"report": str(args.output), "selected": report["selected"] | {"validation_metrics": "see report"}, "frozen_keys": len(frozen_keys), "old_decoder_keys_changed": len(changed_old_keys)}, indent=2))


if __name__ == "__main__":
    main()
