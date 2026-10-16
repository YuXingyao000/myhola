"""Audit the already-finished resume; never trains or mutates checkpoints.

Reuses the previous task's checkpoint/hash/parameter audit helpers. The two
local W&B streams have different logging-step offsets, checked independently
against complete epochs and saved optimizer counters.
"""

from __future__ import annotations

from datetime import datetime
import importlib.util
import json
import math
from pathlib import Path
import statistics
import subprocess
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
from matplotlib import font_manager
import pytorch_lightning as pl
import torch
import yaml
from wandb.proto import wandb_internal_pb2
from wandb.sdk.internal.datastore import DataStore


OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
TOPIC = OUT.parent
RUN = TOPIC / "candidate/20261015T073112.923985Z"
HELPER_PATH = TOPIC / "evaluation_20261015/build_training_audit.py"
SPEC = importlib.util.spec_from_file_location("previous_training_audit", HELPER_PATH)
HELPER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HELPER)
sha256, load, require = HELPER.sha256, HELPER.load, HELPER.require
CHECKPOINTS = {
    "previous_last": ("evaluated_snapshots/e9120c78254d9b94aaaef88333c0f0656fd6a01346225d092fb8341297bff2bf.ckpt", "e9120c78254d9b94aaaef88333c0f0656fd6a01346225d092fb8341297bff2bf", 14520, 39),
    "last": ("last.ckpt", "040c05a45dd7edae825f2e6dd8e2cb3489a418146f9a7b2122cc54d38070618c", 28677, 78),
    "resume_best": ("epoch=0075-step=00027588-val_loss=0.041263.ckpt", "e1e579e7bc6bdf64b3c078d6b5414e33630222de989b180ca8f3eb22b59c6ccf", 27588, 75),
}
METRICS = ("loss", "weighted_kl_loss", "non_kl_objective", "edge_feature", "face_xyz", "edge_xyz", "incidence", "twin", "refined_face_local", "refined_face_bbox", "refined_edge_local", "refined_edge_bbox", "refined_classification")


def flatten(value, prefix=""):
    return {k: v for key, item in value.items() for k, v in
            (flatten(item, prefix + key + ".").items() if isinstance(item, dict) else [(prefix + key, item)])}


def stream_history(path, expected_offset):
    store, rows = DataStore(), []
    store.open_for_scan(str(path))
    try:
        while (raw := store.scan_data()) is not None:
            record = wandb_internal_pb2.Record()
            record.ParseFromString(raw)
            if record.WhichOneof("record_type") == "history":
                row = {item.key or "/".join(item.nested_key): json.loads(item.value_json) for item in record.history.item}
                numeric = {k: v for k, v in row.items() if k.startswith(("train/", "val/")) or k in ("epoch", "trainer/global_step")}
                require(all(isinstance(v, (int, float)) and math.isfinite(v) for v in numeric.values()), "Nonfinite/non-numeric history")
                rows.append(numeric)
    finally:
        store.close()
    result = {"path": str(path), "sha256": sha256(path), "numeric_rows": len(rows), "optimizer_step_minus_logged_step": expected_offset}
    for prefix, total in (("train", "train/loss_epoch"), ("val", "val/loss")):
        selected = [dict(row) for row in rows if total in row]
        require(len({row["epoch"] for row in selected}) == len(selected), "Duplicate complete epoch")
        for row in selected:
            row["optimizer_step"] = (int(row["epoch"]) + 1) * 363
            require(row["optimizer_step"] - row["trainer/global_step"] == expected_offset, "Wrong stream step offset")
            row[prefix + "/loss"] = row[total]
            row[prefix + "/non_kl_objective"] = row[total] - row[prefix + "/weighted_kl_loss"]
            weighted = sum(v for k, v in row.items() if k.startswith(prefix + "/weighted_"))
            require(abs(weighted - row[total]) < 2e-7, "Weighted components do not sum to objective")
        result[prefix + "_epochs"] = selected
    require(len(result["train_epochs"]) == len(result["val_epochs"]), "Incomplete epoch history")
    return result


def differences(new, reference, prefix="val/"):
    return {key: {"reference": reference[prefix + key], "value": new[prefix + key],
                  "relative_change_percent": 100 * (new[prefix + key] / reference[prefix + key] - 1)} for key in METRICS}


def plots(report):
    destination = OUT / "visualizations"
    destination.mkdir(exist_ok=True)
    font_path = Path("/root/.cache/hola-browser-libs/fonts/NotoSansCJKsc-Regular.otf")
    font_manager.fontManager.addfont(str(font_path))
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=str(font_path)).get_name(), "font.size": 11,
                         "axes.unicode_minus": False, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    keys = ("loss", "edge_feature", "weighted_kl_loss", "face_xyz", "edge_xyz", "incidence")
    titles = ("总训练目标", "边特征 L1（目标随训练变化）", "加权 KL（sum 归约）", "面全局 XYZ L1", "边全局 XYZ L1", "边面贴合目标")
    for ax, key, title in zip(axes.flat, keys, titles):
        for prefix, style, color, label in (("val", "-", "#bd6544", "验证"), ("train", "--", "#277e83", "训练")):
            for i, phase in enumerate(("original", "resume")):
                rows = report["history"][phase][prefix + "_epochs"]
                ax.plot([r["optimizer_step"] for r in rows], [r[prefix + "/" + key] for r in rows], style,
                        color=color, label=label if i == 0 else None, linewidth=1.5)
        ax.axvline(14520, color="#64748b", linestyle=":", label="开始续训" if key == "loss" else None)
        ax.axvline(27588, color="#b8bfc8", linestyle="--", linewidth=0.8, label="续训验证最优" if key == "loss" else None)
        ax.set(title=title, xlabel="实际 optimizer step")
        ax.grid(alpha=0.18)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    axes[0, 0].legend(fontsize=9)
    fig.suptitle("Encoder 边内容消息：原训练与续训的完整数值曲线\n实线为验证，虚线为训练；两段日志横轴已分别校正。训练／验证的 batch 与增强不同，不直接横比总值。", fontsize=14)
    for suffix in ("png", "svg"):
        fig.savefig(destination / f"resumed_training_history.{suffix}", dpi=150)
    plt.close(fig)


def markdown(report):
    rows = report["checkpoints"]
    lines = ["# Encoder 边内容消息：续训结束审计", "", "协议见 [PLAN](PLAN.md)。本审计只解释训练状态和验证曲线；模型是否更好由固定测试集的几何与 solid 结果判断。", "",
             "**训练正常退出，但没有完成 29,040 步预算。实际最后保存 28,677 步／epoch78，续训增加 14,157 步，即 39 个完整训练轮次。** 退出原因是 Lightning 的 `max_epochs=80` 先触发；不是 OOM、数据耗尽或用户停止。本轮只测试，不补训。", "",
             f"北京时间 {report['termination']['started_at_beijing']} → {report['termination']['finished_at_beijing']}；进程 exit code 0，runner 正确保留 `incomplete`。", "",
             "| 权重身份 | optimizer step | epoch 标签 | validation total |",
             "|---|---:|---:|---:|", f"| 全程验证最优（原训练，已有测试） | 1,089 | 2 | {report['global_best']['validation_loss']:.8f} |"]
    for role, label in (("previous_last", "续训起点（已有测试）"), ("resume_best", "续训窗口验证最优（补充测试）"), ("last", "实际最后权重（主测试）")):
        row = rows[role]
        lines.append(f"| {label} | {row['global_step']:,} | {row['epoch']} | {row['validation_metrics']['val/loss']:.8f} |")
    lines += ["", "27,588 只在这次续训窗口内最优，尚未超过全程 1,089 的最优值。两份新权重在测试前按此规则固定，均完整报告；不根据 test 再挑权重。", "", "## 少一轮的直接证据", "",
              "本地版本为 PyTorch Lightning 2.5.0.post0。恢复的旧 checkpoint 已完成 40×363 次优化，但保存时 `epoch_progress` 为 ready/started/processed=40、completed=39。续训先执行旧轮末尾的恢复/收尾流程，`processed` 比实际完整轮数多计一次。最后 checkpoint 为 ready/started=79、completed=78、processed=80；优化器计数和训练 batch 数均为 28,677，即 79×363。", "",
              "安装源码 `fit_loop.py:187` 用 `epoch_progress.current.processed` 判断 epoch 上限，`on_advance_end()` 的行462会增加此计数。日志明确写 `Trainer.fit stopped: max_epochs=80 reached`，且真实训练只有 epoch40…78。源码恢复路径与 checkpoint 计数共同说明，这是轮末恢复的计数问题；没有证据表明丢了一个已经执行的优化 epoch。", "",
              "同一源码行431也用 `processed` 设置 DistributedSampler 的 epoch，因而计数偏差可能影响 shuffle 编号。未逐 batch 回放采样序列，不能声称续训与不中断训练逐步等价；这不影响对已保存快照的固定输入测试。", "",
              "启动时 `ModelCheckpoint` 曾在首个新 batch 前提示缺少 `val/loss`，仅一次；随后39个完整验证和保存正常。`_batches_that_stepped` 也比优化器计数多1，因此原 W&B 流是 logged step+1=optimizer step，续训流则 logged step=optimizer step。曲线按各自偏移校正，并用 checkpoint 中真实 optimizer 计数验证。", "", "## 验证变化", "",
              "| 指标 | 14,520 | 27,588 | 28,677 | 续训前5轮→后5轮均值 |", "|---|---:|---:|---:|---:|"]
    for key in METRICS:
        numbers = [rows[r]["validation_metrics"]["val/" + key] for r in ("previous_last", "resume_best", "last")]
        lines.append(f"| {key} | {numbers[0]:.8f} | {numbers[1]:.8f} | {numbers[2]:.8f} | {report['resume_window_trends']['val'][key]['relative_change_percent']:+.2f}% |")
    lines += ["", "续训窗口中，feature L1 和边面贴合目标下降；前后5轮均值的面／边全局 XYZ 分别上升约3.39%／1.33%。但实际最后权重相对偏差较大的14,520端点仍有所改善。这两个说法对应不同参照，不矛盾；不能把单个噪声较大的端点或总 loss 当作整体精度结论。", "",
              "训练与验证的 KL 均按 packed face×32维求和，batch 分别16和8，训练还做旋转增强；总 loss 不宜直接横比。edge-feature teacher 随解冻 Encoder 一起改变，即使扣除 KL，仍不是固定几何目标。模型验证使用 posterior 采样，本次几何测试使用各 checkpoint 自己的 posterior mean。", "", "![完整训练曲线](visualizations/resumed_training_history.png)", "", "## 配置、参数与数据", "",
              "原配置与实际续训配置仅4项不同：清空组件初始化、设置完整resume路径、步数上限14,520→29,040、轮数上限40→80。架构、loss、Adam学习率2e−5、optimizer参数组、8×16 batch、bf16、seed42、数据split和增强保持原配置。四个生产模型／训练／数据文件与原训练commit `0b145d5d` 字节一致。", "",
              "两段全部数值日志未发现 NaN/Inf；最后模型756个state tensor与优化器756个parameter state均有限，未发现损坏权重。参数变化只说明训练在更新模型，不代表下游效果。训练、验证、测试split哈希仍一致，未做重分割。", "", "## 权重归档与可复核数据", ""]
    for role in ("last", "resume_best"):
        row = rows[role]
        lines += [f"- {role}: `{row['archive_path']}`", f"  - 原文件 `{row['source_path']}`", f"  - SHA256 `{row['sha256']}`；独立 copy2，源文件与历史metadata未修改。"]
    lines += ["", "[training_audit.json](training_audit.json) 包含两段完整 train/val epoch、选择依据、配置差异、源码与输入摘要、归档映射和参数变化。任务脚本复用昨日审计的读取／哈希／参数比较函数；复现命令：", "", "```bash", f"/mnt/d/anaconda3/envs/torch/bin/python {OUT / 'build_training_audit.py'}", "```", ""]
    (OUT / "TRAINING_AUDIT.md").write_text("\n".join(lines))


def main():
    torch.set_num_threads(2)
    active_path = TOPIC / "metadata/active_pair.json"
    active = json.loads(active_path.read_text())
    require(active["launch_id"] == "20261015T133540.963165Z" and active["arms"]["candidate"]["exit_code"] == 0, "Wrong resume or failed process")
    streams = {"original": RUN / "wandb/run-20261015_073122-376247fb7478/run-376247fb7478.wandb",
               "resume": RUN / "wandb/run-20261015_133550-376247fb7478/run-376247fb7478.wandb"}
    history = {name: stream_history(path, 1 if name == "original" else 0) for name, path in streams.items()}
    require([r["epoch"] for r in history["original"]["val_epochs"]] == list(range(40)), "Wrong original epochs")
    require([r["epoch"] for r in history["resume"]["val_epochs"]] == list(range(40, 79)), "Wrong resumed epochs")
    validation = history["original"]["val_epochs"] + history["resume"]["val_epochs"]
    by_step = {r["optimizer_step"]: r for r in validation}
    report = {"plan_sha256": sha256(OUT / "PLAN.md"), "audit_script_sha256": sha256(Path(__file__)), "reused_helper_sha256": sha256(HELPER_PATH),
              "checked_at_beijing": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), "history": history, "checkpoints": {},
              "active_metadata_sha256": sha256(active_path), "lightning_version": pl.__version__, "limitations": ["No downstream accuracy claim from training curves.", "No checkpoint or runtime metadata modified by this audit.", "Training was not resumed to fill the missing epoch.", "The processed counter is also used for sampler epoch; no batchwise replay establishes equivalence to uninterrupted training."]}
    loaded = {}
    for role, (name, digest, step, epoch) in CHECKPOINTS.items():
        source = RUN / "weights" / name
        archive = RUN / "weights/evaluated_snapshots" / f"{digest}.ckpt"
        require(sha256(source) == sha256(archive) == digest, "Checkpoint/archive mismatch")
        ckpt = load(archive)
        require(ckpt["global_step"] == step and ckpt["epoch"] == epoch, "Checkpoint metadata mismatch")
        require(ckpt["loops"]["fit_loop"]["epoch_loop.automatic_optimization.optim_progress"]["optimizer"]["step"]["total"]["completed"] == step, "Optimizer loop mismatch")
        require(all(torch.isfinite(t).all() for t in ckpt["state_dict"].values()), "Nonfinite model state")
        optimizer = ckpt["optimizer_states"][0]
        require(all(torch.isfinite(t).all() for state in optimizer["state"].values() for t in state.values() if torch.is_tensor(t)), "Nonfinite optimizer state")
        require(len(optimizer["state"]) == len(ckpt["state_dict"]) == 756, "Changed state layout")
        loaded[role] = ckpt
        report["checkpoints"][role] = {"source_path": str(source), "archive_path": str(archive), "sha256": digest, "epoch": epoch, "global_step": step,
             "independent_copy": source.stat().st_ino != archive.stat().st_ino if role != "previous_last" else None,
             "validation_metrics": by_step[step], "epoch_progress": ckpt["loops"]["fit_loop"]["epoch_progress"], "optimizer_state_count": len(optimizer["state"]),
             "logged_batches_that_stepped": ckpt["loops"]["fit_loop"]["epoch_loop.state_dict"]["_batches_that_stepped"]}
    last = loaded["last"]
    selector = HELPER.callback(last)
    global_best = min(validation, key=lambda r: r["val/loss"])
    resume_best = min(history["resume"]["val_epochs"], key=lambda r: r["val/loss"])
    require(global_best["optimizer_step"] == 1089 and global_best["val/loss"] == float(selector["best_model_score"]), "Wrong global best")
    require(resume_best["optimizer_step"] == 27588, "Wrong resumed-window best")
    report["global_best"] = {"optimizer_step": 1089, "validation_loss": global_best["val/loss"], "path": selector["best_model_path"], "sha256": sha256(Path(selector["best_model_path"]))}
    report["retained_top_k"] = [{"path": path, "validation_loss": float(score)} for path, score in sorted(selector["best_k_models"].items(), key=lambda x: float(x[1]))]
    original_cfg_path = RUN / "metadata/invocations/20261015T073119.621506Z/config.yaml"
    resumed_cfg_path = RUN / "metadata/invocations/20261015T133547.612738Z/config.yaml"
    original, resumed = (flatten(yaml.safe_load(p.read_text())) for p in (original_cfg_path, resumed_cfg_path))
    changes = {key: {"old": original.get(key), "new": resumed.get(key)} for key in original.keys() | resumed.keys() if original.get(key) != resumed.get(key)}
    require(set(changes) == {"model.init_encoder_edge_value_checkpoint", "trainer.max_steps", "trainer.max_epochs", "trainer.resume_from_checkpoint"}, "Unexpected configuration changes")
    require(loaded["previous_last"]["optimizer_states"][0]["param_groups"] == last["optimizer_states"][0]["param_groups"], "Optimizer groups changed")
    report["configuration"] = {"original_path": str(original_cfg_path), "original_sha256": sha256(original_cfg_path), "resume_path": str(resumed_cfg_path), "resume_sha256": sha256(resumed_cfg_path), "changes": changes}
    report["production_sources"] = {}
    for name in HELPER.SOURCE_FILES:
        old = subprocess.check_output(["git", "show", "0b145d5d2fe4d0e3140bd114bb1675047093ca97:" + name], cwd=ROOT)
        require(old == (ROOT / name).read_bytes(), "Production source changed")
        report["production_sources"][name] = sha256(ROOT / name)
    report["splits"] = active["resume_plan"]["provenance"]["splits"]
    for split in report["splits"].values():
        path = Path(split["path"])
        require(sha256(path) == split["sha256"] and len(path.read_text().splitlines()) == split["count"], "Split changed")
    arm = active["arms"]["candidate"]
    console_path = Path(arm["console_log"])
    console = console_path.read_text()
    require("`Trainer.fit` stopped: `max_epochs=80` reached." in console and "Traceback (most recent call last)" not in console, "Unexpected termination")
    require(not Path(active["stop_marker"]).exists(), "Stop marker present")
    report["termination"] = {k: arm[k] for k in ("exit_code", "state", "started_at_utc", "finished_at_utc")}
    report["termination"].update({"started_at_beijing": datetime.fromisoformat(arm["started_at_utc"]).astimezone(ZoneInfo("Asia/Shanghai")).isoformat(), "finished_at_beijing": datetime.fromisoformat(arm["finished_at_utc"]).astimezone(ZoneInfo("Asia/Shanghai")).isoformat(), "requested_steps": 29040, "actual_steps": 28677, "shortfall_steps": 363, "additional_steps": 14157, "resumed_complete_epochs": 39, "actual_total_complete_epochs": 79, "console_path": str(console_path), "console_sha256": sha256(console_path), "monitor_missing_warning_count": console.count("could not find the monitored key"), "stop_reason": "max_epochs reached with processed epoch counter one ahead of actual complete epochs"})
    lightning_source = Path(pl.__file__).parent / "loops/fit_loop.py"
    report["loop_source"] = {"path": str(lightning_source), "sha256": sha256(lightning_source), "max_epochs_check_line": 187, "processed_increment_line": 462, "checkpoint_counters_directly_verified": True}
    report["resume_window_trends"] = {}
    for prefix in ("train", "val"):
        series = history["resume"][prefix + "_epochs"]
        first = {prefix + "/" + key: statistics.mean(r[prefix + "/" + key] for r in series[:5]) for key in METRICS}
        final = {prefix + "/" + key: statistics.mean(r[prefix + "/" + key] for r in series[-5:]) for key in METRICS}
        report["resume_window_trends"][prefix] = differences(final, first, prefix + "/")
    report["last_vs_previous_last"] = differences(by_step[28677], by_step[14520])
    report["resume_best_vs_previous_last"] = differences(by_step[27588], by_step[14520])
    report["parameter_changes_last_vs_resume_source"] = HELPER.parameter_changes(last["state_dict"], loaded["previous_last"]["state_dict"])
    (OUT / "training_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    plots(report)
    markdown(report)
    print(json.dumps({"status": "passed", "actual_steps": 28677, "resume_best_step": 27588, "complete_val_epochs": len(validation), "shortfall_steps": 363}))


if __name__ == "__main__":
    main()
