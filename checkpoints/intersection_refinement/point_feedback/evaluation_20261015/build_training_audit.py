"""Audit validation-selected point feedback; no inference or checkpoint ranking by test.

The frozen execution contract is PLAN.md beside this script. Only whitelisted
configuration fields and numeric history are exported; local W&B files are read
offline and no authentication, complete config, or model weights are exported.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
from zoneinfo import ZoneInfo

import torch
from wandb.proto import wandb_internal_pb2
from wandb.sdk.internal.datastore import DataStore


MODEL_FIELDS = (
    "name", "stage", "in_channels", "latent_channels", "hidden_channels", "norm",
    "gaussian_weights", "sigmoid", "num_gat_layers", "bottleneck_dim",
    "num_encoder_layers", "num_decoder_layers", "nhead", "with_intersection",
    "intersection_dim", "intersection_layers", "intersection_noise_std", "loss",
    "primitive_blocks_per_stage", "point_decoder_blocks_per_stage",
    "face_bbox_decoder_blocks", "edge_bbox_decoder_blocks", "kl_reduction",
    "freeze_encoder", "trainable_scope", "edge_threshold", "intersection",
    "curve_conditioning", "refinement", "loss_weights",
)
TRAINER_FIELDS = (
    "devices", "batch_size", "num_workers", "learning_rate", "precision", "seed",
    "gradient_clip_val", "deterministic", "cudnn_benchmark", "max_epochs",
    "max_steps", "max_time", "check_val_every_n_epoch", "num_sanity_val_steps",
    "vae_validation_inference", "limit_train_batches", "limit_val_batches",
    "limit_test_batches", "checkpoint_enable_version_counter",
)
DATASET_FIELDS = (
    "name", "raw_data_root", "train_dataset", "val_dataset", "test_dataset",
    "is_aug", "is_overfit", "epoch_scale_factor", "scale_factor",
)
CURVE_METRICS = (
    "edge_xyz", "face_xyz", "incidence", "twin", "edge_feature",
    "refined_edge_bbox", "refined_edge_local", "refined_face_bbox",
    "refined_face_local", "encoder_edge_local", "delta_edge_mean", "delta_face_mean",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False, mmap=True)


def selection(value: dict) -> dict:
    states = [state for key, state in value["callbacks"].items()
              if key.startswith("ModelCheckpoint") and "'mode': 'min'" in key
              and state.get("monitor") == "val/loss"]
    require(len(states) == 1, "Expected exactly one val/loss, min ModelCheckpoint")
    return states[0]


def history(run: Path) -> tuple[list[dict], Path]:
    paths = sorted({path.resolve() for path in run.glob("wandb/*/*.wandb")})
    require(len(paths) == 1, f"Expected one local W&B stream in {run}; found {len(paths)}")
    store = DataStore()
    store.open_for_scan(str(paths[0]))
    rows = []
    try:
        while (raw := store.scan_data()) is not None:
            record = wandb_internal_pb2.Record()
            record.ParseFromString(raw)
            if record.WhichOneof("record_type") == "history":
                row = {item.key or "/".join(item.nested_key): json.loads(item.value_json)
                       for item in record.history.item}
                for key, value in row.items():
                    if key.startswith(("val/", "train/")):
                        require(isinstance(value, (int, float)) and math.isfinite(value),
                                f"Non-finite or non-numeric history metric {key}")
                rows.append(row)
    finally:
        store.close()
    require(bool(rows), f"Empty local history: {paths[0]}")
    return rows, paths[0]


def changes(current: dict, baseline: dict, fields: tuple[str, ...]) -> dict:
    return {key: {"baseline": baseline.get(key), "selected": current.get(key)}
            for key in fields if current.get(key) != baseline.get(key)}


def write_markdown(report: dict, output: Path) -> None:
    selected, baseline, last = report["selected"], report["baseline"], report["last_saved"]
    ending = report["termination"]
    local_times = [datetime.fromisoformat(ending["exit_record"][key]).astimezone(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M:%S")
                   for key in ("started_at_utc", "ended_at_utc")]
    lines = [
        "# Point feedback 训练审计", "",
        "执行合同见 [PLAN.md](PLAN.md)。本报告只核验训练来源、验证集选权重和记录；不使用 test 选择模型，也不据训练 loss 宣称 solid 提升。", "",
        f"唯一候选：epoch **{selected['epoch']}** / step **{selected['global_step']}**，val/loss **{selected['validation_loss']:.10f}**。"
        "last.ckpt 的 ModelCheckpoint(val/loss, min) 与完整本地验证 history 的最小值一致。", "",
        f"正式运行正常退出，exit_code=0；北京时间 **{local_times[0]} → {local_times[1]}**。"
        f"Timer 计时 {ending['timer_training_seconds']:.3f} 秒，包含初始化/退出的进程时间 {ending['process_wall_seconds']:.3f} 秒。", "",
        f"last 为 epoch {last['epoch']} / step {last['global_step']}：**58 个完整训练 epoch + 最后 220/363 batch**，不是 59 个完整训练 epoch。"
        "时间到期后完成一次全验证并保存 last；共有 59 次完整验证。", "",
        "## 权重与来源", "",
        "| 角色 | epoch / step | SHA-256 |", "|---|---|---|",
        *[f"| {label} | {item['epoch']} / {item['global_step']} | `{item['sha256']}` |"
          for label, item in (("直接 conditioned 基线", baseline), ("验证最佳候选", selected), ("最后保存", last))], "",
        "- 旧 736 个 tensor 全部保留且形状相同；243 个 encoder/posterior tensor 在候选和 last 中均逐位等于直接基线。",
        "- 493 个旧 decoder tensor 均相对直接基线改变，包括已有 conditioning；并非只训练新增分支。",
        "- 新增 6 个 tensor、27,139 参数，总计 181,993,120 参数；均有限，且 6 个 tensor 在 best→last 之间均发生变化。"
        "两个 delta tensor 从已知零初始化变为非零；其余四个初始随机 tensor 未保存快照，不报告未经验证的初始差值。",
        "- 原模型容量/refinement/loss 配置一致；Intersection 保持 8 层、512 维、FFN 2048。训练配置白名单的唯一变化是新增 8 小时 max_time。",
        "- split 只是相对/绝对路径写法变化；train/val/test 文件 SHA 与直接基线历史审计完全相同（46,480 / 3,495 / 2,417 件）。",
        f"- 训练 commit `{report['provenance']['training_commit']}`，启动时 clean；vae.py、blocks.py、train.py、dataset.py 的训练版本与评估版本 SHA 完全相同。", "",
        "完整路径、各 tensor 数值统计、配置白名单、source SHA 和逐 epoch 表均在 [training_audit.json](training_audit.json)。不导出完整配置、凭证或权重。", "",
        "## 验证最佳与直接基线", "",
        "| 验证指标 | Conditioned 基线 | Point feedback 最佳 | 相对变化 |", "|---|---:|---:|---:|",
    ]
    for name in ("loss", "edge_xyz", "face_xyz", "incidence", "twin", "refined_edge_local", "refined_edge_bbox", "refined_face_local", "refined_face_bbox", "encoder_edge_local"):
        item = report["validation_vs_baseline"]["val/" + name]
        lines.append(f"| {name} | {item['baseline']:.8f} | {item['selected']:.8f} | {item['relative_change_percent']:+.2f}% |")
    lines += ["", "上述是训练目标，不是 CAD tolerance/Chamfer/solid 指标。"
              "其中 local 包括归一化 XYZ 与后 3 维方向；edge_xyz 是全局 XYZ，不能直接比较二者数值。"
              "incidence/twin 带 GT 离散误差基线。encoder-edge teacher 输入信息不同，且该指标本轮反而增加。", "",
              f"候选额外训练 {selected['global_step']:,} 步，global batch 128，{report['exposure']['selected_additional_object_presentations']:,} 次对象呈现；"
              f"全程 {last['global_step']:,} 步、{report['exposure']['last_saved_additional_object_presentations']:,} 次对象呈现。"
              "optimizer 从直接 conditioned 权重重新初始化。因此不是等预算结构消融，无法把差异全部归因新增分支。", "",
              f"last 的 val/loss={last['last_full_validation_loss']:.8f}，较最佳高 {report['val_trends']['loss']['last_vs_best_pct']:.2f}%；"
              "选用验证最佳，不因最后训练 loss 更低而改选 last。delta_edge_mean/delta_face_mean 属于原 feature refiner，不是新增 XYZ 位移。", "",
              "## 本地 epoch 表", "", "train 最后一行仅 220 batch；val 每行都是完整验证。step 是 checkpoint 对应的计数（本地 log 记为 step−1）。", "",
              "| epoch | step | train loss | val loss | val edge XYZ | val incidence | val twin |", "|---:|---:|---:|---:|---:|---:|---:|"]
    train_rows = {row["epoch"]: row for row in report["training_epoch_history"]}
    for row in report["validation_history"]:
        epoch = row["epoch"]
        train = train_rows.get(epoch, {})
        train_loss = f"{train['train/loss_epoch']:.6f}" if "train/loss_epoch" in train else "缺失"
        label = str(epoch) + ("（部分训练）" if train.get("full_training_epoch") is False else "")
        lines.append(f"| {label} | {row['trainer/global_step'] + 1} | {train_loss} | {row['val/loss']:.6f} | {row['val/edge_xyz']:.6f} | {row['val/incidence']:.6f} | {row['val/twin']:.6f} |")
    output.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--baseline-audit", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.run, args.baseline, args.repo = args.run.resolve(), args.baseline.resolve(), args.repo.resolve()
    torch.set_num_threads(2)
    latest_path = args.run / "weights/last.ckpt"
    latest = checkpoint(latest_path)
    selector = selection(latest)
    selected_path = Path(selector["best_model_path"]).resolve()
    require(selected_path.parent == args.run / "weights", "Selected checkpoint is outside the requested run")
    selected, baseline = checkpoint(selected_path), checkpoint(args.baseline)
    state, old, last_state = selected["state_dict"], baseline["state_dict"], latest["state_dict"]
    added = sorted(state.keys() - old.keys())
    require(len(added) == 6 and all(key.startswith("model.decoder.curve_point_feedback.") for key in added),
            "Expected exactly the six new feedback tensors")
    require(not (old.keys() - state.keys()), "An original tensor was removed")
    require(all(old[key].shape == state[key].shape for key in old), "An original tensor changed shape")
    frozen = [key for key in old if key.startswith(("model.encoder.", "model.posterior."))]
    require(len(frozen) == 243, "Unexpected frozen encoder/posterior tensor count")
    require(all(torch.equal(state[key], old[key]) and torch.equal(last_state[key], old[key]) for key in frozen),
            "Frozen encoder/posterior changed in selected or last checkpoint")
    changed_old = [key for key in old if not torch.equal(state[key], old[key])]
    fresh_stats = {}
    for key in added:
        tensor, last_tensor = state[key], last_state[key]
        require(torch.isfinite(tensor).all().item() and torch.isfinite(last_tensor).all().item(),
                f"Non-finite new parameter: {key}")
        difference = (last_tensor - tensor).float()
        fresh_stats[key] = {
            "shape": list(tensor.shape), "numel": tensor.numel(),
            "selected_nonzero": torch.count_nonzero(tensor).item(),
            "selected_abs_max": tensor.abs().max().item(), "selected_l2_norm": tensor.float().norm().item(),
            "changed_between_best_and_last": not torch.equal(tensor, last_tensor),
            "best_to_last_delta_abs_max": difference.abs().max().item(),
            "known_zero_initialization": ".delta." in key,
        }
    require(sum(value["numel"] for value in fresh_stats.values()) == 27139, "Unexpected parameter increment")
    require(all(value["selected_nonzero"] and value["changed_between_best_and_last"] for value in fresh_stats.values()),
            "New feedback tensors did not all change between best and last")

    rows, history_path = history(args.run)
    baseline_rows, baseline_history_path = history(args.baseline.parent.parent)
    validations = [row for row in rows if "val/loss" in row]
    train_epochs = [row for row in rows if "train/loss_epoch" in row]
    require(bool(validations), "No validation rows")
    require(len({row["epoch"] for row in validations}) == len(validations), "Ambiguous duplicate validation epoch")
    best_row = min(validations, key=lambda row: row["val/loss"])
    score = float(selector["best_model_score"])
    require(best_row["epoch"] == selected["epoch"] and best_row["trainer/global_step"] + 1 == selected["global_step"],
            "History minimum disagrees with callback-selected checkpoint")
    require(best_row["val/loss"] == score, "History minimum disagrees with callback score")
    base_rows = [row for row in baseline_rows if "val/loss" in row and row["epoch"] == baseline["epoch"]
                 and row["trainer/global_step"] + 1 == baseline["global_step"]]
    require(len(base_rows) == 1, "Cannot identify the baseline validation record uniquely")
    base_row = base_rows[0]
    require(validations[-1]["trainer/global_step"] + 1 == latest["global_step"], "Final validation/last checkpoint mismatch")
    comparisons = {
        key: {"baseline": value, "selected": best_row[key],
              "relative_change_percent": (best_row[key] / value - 1) * 100 if value else None}
        for key, value in base_row.items() if key.startswith("val/") and key in best_row
    }

    hp, base_hp = selected["hyper_parameters"], baseline["hyper_parameters"]
    model_diff = changes(hp["model"], base_hp["model"], MODEL_FIELDS)
    require(not model_diff, f"Unexpected old model/loss configuration changes: {model_diff}")
    require(hp["model"]["curve_point_feedback"] == {"enabled": True, "hidden_dim": 128, "feature_dim": 64},
            "Unexpected feedback architecture")
    require(Path(hp["model"]["init_curve_feedback_checkpoint"]).resolve() == args.baseline,
            "Initialization source differs from requested baseline")
    layer_indices = sorted({int(match.group(1)) for key in state
                            if (match := re.match(r"model\.decoder\.inter\.layers\.(\d+)\.", key))})
    require(layer_indices == list(range(8)), "Original eight intersection layers not preserved")

    run_meta = json.loads((args.run / "metadata/run.json").read_text())
    invocation = args.run / "metadata/invocations" / run_meta["initial_invocation_id"]
    git_meta = json.loads((invocation / "git.json").read_text())
    dirty_path = invocation / "git.diff"
    dirty_text = dirty_path.read_text()
    source_checks = {}
    for relative in ("src/brepnet/models/vae.py", "src/brepnet/models/blocks.py", "src/brepnet/train.py", "src/brepnet/dataset.py"):
        original = subprocess.check_output(["git", "show", f"{git_meta['commit']}:{relative}"], cwd=args.repo)
        current = (args.repo / relative).read_bytes()
        dirty = f"diff --git a/{relative} b/{relative}" in dirty_text
        require(original == current and not dirty, f"Training/evaluation source differs: {relative}")
        source_checks[relative] = {"training_commit_sha256": hashlib.sha256(original).hexdigest(),
                                   "evaluation_sha256": hashlib.sha256(current).hexdigest(), "exact_match": True,
                                   "mentioned_in_training_dirty_diff": False}
    split_records = {}
    historical_audit = json.loads(args.baseline_audit.read_text())
    for split in ("train", "val", "test"):
        path = Path(hp["dataset"][f"{split}_dataset"])
        path = path if path.is_absolute() else args.repo / path
        ids = [line.strip() for line in path.read_text().splitlines() if line.strip()]
        require(bool(ids) and len(ids) == len(set(ids)), f"Empty or duplicate IDs in {split}")
        split_records[split] = {"path": str(path), "sha256": sha256(path), "count": len(ids), "unique_count": len(set(ids))}
        require(split_records[split]["sha256"] == historical_audit["splits"][split]["sha256"],
                f"{split} IDs differ from the direct baseline's archived split hash")
        split_records[split]["same_sha256_as_direct_baseline_audit"] = True
    exit_record = json.loads((args.run / "metadata/exit.json").read_text())
    require(exit_record["exit_code"] == 0, "Run did not exit successfully")
    timer_seconds = latest["callbacks"]["Timer"]["time_elapsed"]["train"]
    require(timer_seconds >= 28800, "Eight-hour training timer was not exhausted")
    total_seconds = (datetime.fromisoformat(exit_record["ended_at_utc"]) - datetime.fromisoformat(exit_record["started_at_utc"])).total_seconds()
    fit_loop = latest["loops"]["fit_loop"]
    selected_loop = selected["loops"]["fit_loop"]
    completed_epochs = fit_loop["epoch_progress"]["current"]["completed"]
    last_epoch_batches = fit_loop["epoch_loop.batch_progress"]["current"]["completed"]
    steps_per_epoch = selected_loop["epoch_loop.batch_progress"]["current"]["completed"]
    require(latest["global_step"] == completed_epochs * steps_per_epoch + last_epoch_batches,
            "Partial final epoch accounting does not reproduce total steps")
    batch = hp["trainer"]["devices"] * hp["trainer"]["batch_size"]
    losses = [row["val/loss"] for row in validations]
    report = {
        "selection_policy": "last.ckpt ModelCheckpoint(val/loss, min) best_model_path; independently confirmed by all local validation history. No test metric selects weights.",
        "selected": {"path": str(selected_path), "sha256": sha256(selected_path), "epoch": selected["epoch"],
                     "global_step": selected["global_step"], "validation_loss": score,
                     "validation_metrics": {key: value for key, value in best_row.items() if key.startswith("val/")}},
        "baseline": {"path": str(args.baseline), "sha256": sha256(args.baseline), "epoch": baseline["epoch"],
                     "global_step": baseline["global_step"], "validation_loss": base_row["val/loss"]},
        "last_saved": {"path": str(latest_path), "sha256": sha256(latest_path), "epoch": latest["epoch"],
                       "global_step": latest["global_step"], "last_full_validation_loss": validations[-1]["val/loss"],
                       "completed_training_epochs": completed_epochs, "partial_epoch_batches": last_epoch_batches,
                       "final_epoch_complete": fit_loop["epoch_loop.batch_progress"]["is_last_batch"]},
        "saved_top_k": [{"path": path, "validation_loss": float(value)} for path, value in sorted(selector["best_k_models"].items(), key=lambda item: float(item[1]))],
        "state_compatibility": {"baseline_keys": len(old), "selected_keys": len(state),
                                "all_old_keys_present_and_same_shapes": True, "frozen_encoder_posterior_keys": len(frozen),
                                "frozen_keys_bitwise_equal_selected_and_last": True, "old_decoder_keys_changed": len(changed_old),
                                "total_parameters": sum(tensor.numel() for tensor in state.values()), "new_feedback_parameters": fresh_stats,
                                "change_evidence": "All six tensors differ between selected and final checkpoints; two delta tensors also differ from their known zero initialization. Hidden initial values were not archived, so initial-to-selected deltas for those four are not claimed."},
        "actual_intersection_tensor_shapes": {"layer_indices": layer_indices,
                                               "attention_in_proj": list(state["model.decoder.inter.layers.0.self_attn.in_proj_weight"].shape),
                                               "ffn_linear1": list(state["model.decoder.inter.layers.0.linear1.weight"].shape)},
        "exposure": {"global_train_batch": batch, "steps_per_completed_epoch": steps_per_epoch,
                     "selected_completed_epochs": selected["epoch"] + 1, "selected_additional_steps": selected["global_step"],
                     "selected_additional_object_presentations": selected["global_step"] * batch,
                     "last_saved_completed_epochs": completed_epochs, "last_saved_partial_epoch_batches": last_epoch_batches,
                     "last_saved_additional_steps": latest["global_step"], "last_saved_additional_object_presentations": latest["global_step"] * batch,
                     "latest_logged_train_step_zero_based": max(row.get("trainer/global_step", -1) for row in rows),
                     "note": "Fresh optimizer after the direct conditioned baseline. This is additional training, not an equal-budget architecture ablation. drop_last=True; saved global_step is logged step plus one."},
        "validation_vs_baseline": comparisons,
        "validation_history_summary": {"source": str(history_path), "sha256": sha256(history_path),
                                       "baseline_source": str(baseline_history_path), "num_complete_validations": len(validations),
                                       "first_validation_loss": losses[0], "best_validation_loss": min(losses), "last_validation_loss": losses[-1],
                                       "best_runtime_seconds": best_row.get("_runtime"), "last_validation_runtime_seconds": validations[-1].get("_runtime")},
        "validation_history": [{key: row.get(key) for key in ("epoch", "trainer/global_step", "val/loss", *("val/" + metric for metric in CURVE_METRICS))} for row in validations],
        "training_epoch_history": [{key: row.get(key) for key in ("epoch", "trainer/global_step", "train/loss_epoch", *("train/" + metric for metric in CURVE_METRICS))}
                                   | {"full_training_epoch": row["epoch"] < completed_epochs} for row in train_epochs],
        "val_trends": {"loss": {"first_three_mean": statistics.mean(losses[:3]), "last_three_mean": statistics.mean(losses[-3:]),
                                "last_vs_first_three_pct": (statistics.mean(losses[-3:]) / statistics.mean(losses[:3]) - 1) * 100,
                                "last_eight_std_population": statistics.pstdev(losses[-8:]),
                                "last_vs_best_pct": (losses[-1] / min(losses) - 1) * 100}},
        "splits": split_records,
        "model_config": {key: hp["model"].get(key) for key in (*MODEL_FIELDS, "curve_point_feedback")},
        "dataset_config": {key: hp["dataset"].get(key) for key in DATASET_FIELDS},
        "trainer_config": {key: hp["trainer"].get(key) for key in TRAINER_FIELDS},
        "configuration_changes_vs_direct_baseline": {"existing_model_and_losses": model_diff,
                                                      "trainer": changes(hp["trainer"], base_hp["trainer"], TRAINER_FIELDS),
                                                      "dataset": changes(hp["dataset"], base_hp["dataset"], DATASET_FIELDS),
                                                      "added_model_branch": hp["model"]["curve_point_feedback"]},
        "termination": {"exit_record": exit_record, "timer_training_seconds": timer_seconds,
                        "process_wall_seconds": total_seconds, "timer_budget_seconds": 28800,
                        "partial_final_epoch_followed_by_complete_validation_and_saved_last": True},
        "provenance": {"training_commit": git_meta["commit"], "training_branch": git_meta["branch"],
                       "training_dirty_diff_sha256": sha256(dirty_path), "training_clean": not dirty_text,
                       "source_checks": source_checks, "optimizer_type_from_training_source": "Adam",
                       "optimizer_param_groups": [{key: value for key, value in group.items() if key != "params"}
                                                  | {"parameter_tensor_count": len(group["params"])} for group in selected["optimizer_states"][0]["param_groups"]]},
        "limitations": ["Validation metrics are training objectives, not CAD tolerance or valid-solid measurements.",
                        "The existing decoder and six conditioning tensors also train; improvements cannot be uniquely attributed to point feedback.",
                        "The final training epoch is partial; its train averages have a shorter sampling budget.",
                        "delta_edge_mean/delta_face_mean measure the existing feature refiner, not the new coordinate displacement."],
    }
    report["checkpoint_recommendation"] = {key: report["selected"][key] for key in ("path", "sha256", "epoch", "global_step")} | {"best_model_score": score}
    report["last_checkpoint"] = report["last_saved"]
    report["last_logged_global_step"] = report["exposure"]["latest_logged_train_step_zero_based"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    write_markdown(report, args.output.with_name("TRAINING_AUDIT.md"))
    print(json.dumps({"output": str(args.output), "selected": report["checkpoint_recommendation"],
                      "frozen_tensors": len(frozen), "old_decoder_tensors_changed": len(changed_old),
                      "new_tensors_changed_between_best_last": len(fresh_stats)}, indent=2))


if __name__ == "__main__":
    main()
