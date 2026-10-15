"""Offline training audit; see the frozen PLAN.md beside this file.

Only numeric history and whitelisted configuration fields are exported. No
network access, credential collection, training, or checkpoint mutation occurs.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from wandb.proto import wandb_internal_pb2
from wandb.sdk.internal.datastore import DataStore


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
TOPIC = OUT.parent
SOURCE = Path("/mnt/d/python/checkpoints/intersection_refinement/curve_conditioning/20261013T083729.740439Z/weights/epoch=0157-step=00057354-val_loss=0.038204.ckpt")
SHA = {
    "source": "4c7910b8169206c74fcf38c358cf171359149e0429c8debbc227d391aa56613f",
    "candidate": "d9e911e23a5dfc0e45fb90735a806706c9c89329898f0873df06927c8d2fb62d",
    "control": "f1de7f11fee749e51921e0fb45d32ae7b6b7d7899316b248560bab4501d6c279",
}
METRICS = ("loss", "weighted_kl_loss", "non_kl_objective", "edge_feature", "face_xyz", "edge_xyz", "incidence", "twin", "refined_face_local", "refined_face_bbox", "refined_edge_local", "refined_edge_bbox", "refined_classification", "encoder_edge_local", "encoder_edge_bbox", "refined_precision", "refined_recall")
MODEL_FIELDS = ("name", "in_channels", "latent_channels", "hidden_channels", "num_gat_layers", "num_encoder_layers", "num_decoder_layers", "intersection_dim", "intersection_layers", "nhead", "freeze_encoder", "gaussian_weights", "kl_reduction", "encoder_edge_value", "curve_conditioning", "curve_point_feedback", "loss_weights")
TRAINER_FIELDS = ("devices", "batch_size", "learning_rate", "precision", "seed", "gradient_clip_val", "max_epochs", "max_steps", "max_time", "check_val_every_n_epoch", "vae_validation_inference", "deterministic", "cudnn_benchmark")
SOURCE_FILES = ("src/brepnet/models/vae.py", "src/brepnet/models/blocks.py", "src/brepnet/train.py", "src/brepnet/dataset.py")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False, mmap=True)


def history(run):
    paths = sorted({p.resolve() for p in run.glob("wandb/*/*.wandb")})
    require(len(paths) == 1, f"Expected one history stream: {run}")
    store, rows = DataStore(), []
    store.open_for_scan(str(paths[0]))
    try:
        while (raw := store.scan_data()) is not None:
            record = wandb_internal_pb2.Record()
            record.ParseFromString(raw)
            if record.WhichOneof("record_type") != "history":
                continue
            row = {item.key or "/".join(item.nested_key): json.loads(item.value_json)
                   for item in record.history.item}
            numeric = {}
            for key, value in row.items():
                if key.startswith(("train/", "val/")) or key in ("epoch", "trainer/global_step", "_runtime"):
                    require(isinstance(value, (int, float)) and math.isfinite(value), f"Invalid metric: {key}")
                    numeric[key] = value
            rows.append(numeric)
    finally:
        store.close()
    require(rows, "Empty history")
    return rows, paths[0]


def epoch_rows(rows, prefix):
    total = prefix + ("/loss_epoch" if prefix == "train" else "/loss")
    selected = [dict(row) for row in rows if total in row]
    require(selected and len({row["epoch"] for row in selected}) == len(selected), "Missing/duplicate epoch")
    for row in selected:
        row[prefix + "/loss"] = row[total]
        row["optimizer_step"] = int(row["trainer/global_step"]) + 1
        row[prefix + "/non_kl_objective"] = row[total] - row[prefix + "/weighted_kl_loss"]
        weighted = sum(value for key, value in row.items() if key.startswith(prefix + "/weighted_"))
        require(abs(weighted - row[total]) < 2e-7, "Weighted losses do not reproduce total")
    return selected


def callback(checkpoint):
    states = [v for k, v in checkpoint["callbacks"].items()
              if k.startswith("ModelCheckpoint") and v.get("monitor") == "val/loss"]
    require(len(states) == 1, "Ambiguous callback")
    return states[0]


def parameter_changes(current, reference):
    missing = sorted(set(reference) - set(current))
    require(not missing, f"Missing old keys: {missing}")
    groups = defaultdict(lambda: {"tensor_count": 0, "changed_tensors": 0, "numel": 0, "reference_squared_norm": 0.0, "delta_squared_norm": 0.0, "delta_abs_max": 0.0})
    for key, old in reference.items():
        new = current[key]
        require(new.shape == old.shape and torch.isfinite(new).all(), f"Invalid state: {key}")
        group = groups[".".join(key.split(".")[:3])]
        delta = (new - old).float()
        group["tensor_count"] += 1
        group["changed_tensors"] += int(not torch.equal(new, old))
        group["numel"] += new.numel()
        group["reference_squared_norm"] += float(old.float().square().sum())
        group["delta_squared_norm"] += float(delta.square().sum())
        group["delta_abs_max"] = max(group["delta_abs_max"], float(delta.abs().max()))
    for group in groups.values():
        group["delta_l2_norm"] = math.sqrt(group.pop("delta_squared_norm"))
        group["reference_l2_norm"] = math.sqrt(group.pop("reference_squared_norm"))
        group["relative_l2_change"] = group["delta_l2_norm"] / group["reference_l2_norm"] if group["reference_l2_norm"] else None
    return dict(groups)


def branch_stats(best, last, source):
    keys = sorted(set(best) - set(source))
    require(len(keys) == 20 and all(".value_net." in key for key in keys), "Unexpected new state")
    require(sum(best[k].numel() for k in keys) == 51680, "Wrong new parameter count")
    result = {}
    for key in keys:
        a, b = best[key].float(), last[key].float()
        require(torch.isfinite(a).all() and torch.isfinite(b).all(), "Nonfinite branch")
        result[key] = {"shape": list(a.shape), "numel": a.numel(), "selected_l2_norm": float(a.norm()), "last_l2_norm": float(b.norm()), "selected_nonzero": int(torch.count_nonzero(a)), "known_zero_initialization": ".value_net.2." in key, "selected_to_last_l2": float((b - a).norm()), "changed_best_to_last": not torch.equal(a, b)}
    require(all(row["selected_nonzero"] and row["changed_best_to_last"] for row in result.values()), "Unchanged value branch")
    return result


def relative(a, b):
    return (a / b - 1) * 100 if b else None


def comparisons(a, b):
    return {key: {"value": a["val/" + key], "reference": b["val/" + key], "relative_change_percent": relative(a["val/" + key], b["val/" + key])} for key in METRICS}


def write_plots(report):
    destination = OUT / "visualizations"
    destination.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.2, "svg.fonttype": "none"})
    colors = {"candidate": "#d6691c", "control": "#2877a9"}
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for ax, metric, title in zip(axes.flat, ("loss", "non_kl_objective", "weighted_kl_loss", "edge_feature", "face_xyz", "edge_xyz"), ("Total objective", "Total minus weighted KL", "Weighted KL (sum reduction)", "Edge-feature L1 (moving teacher)", "Face global XYZ L1", "Edge global XYZ L1")):
        for arm, color in colors.items():
            for prefix, style in (("val", "-"), ("train", "--")):
                rows = report["arms"][arm]["validation_history" if prefix == "val" else "training_epoch_history"]
                ax.plot([r["optimizer_step"] for r in rows], [r[prefix + "/" + metric] for r in rows], style, color=color, lw=1.6, label=arm + " " + prefix)
        ax.axvline(1089, color="#64748b", ls=":", lw=1)
        ax.set(title=title, xlabel="Optimizer steps")
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    axes[0, 0].legend(fontsize=9)
    fig.suptitle("Encoder edge-value: full logged training history\nTrain uses rotated CAD / 16 objects per GPU; val uses canonical CAD / 8. KL is not batch-normalized.", fontsize=13)
    for suffix in ("png", "svg"):
        fig.savefig(destination / f"training_history.{suffix}", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for ax, metric, title in zip(axes.flat, ("non_kl_objective", "face_xyz", "edge_xyz", "incidence", "twin", "refined_classification"), ("Non-KL objective", "Face XYZ L1", "Edge XYZ L1", "Owner incidence objective", "Reciprocal-edge objective", "Existence classification BCE")):
        for arm, color in colors.items():
            rows = [r for r in report["arms"][arm]["validation_history"] if r["optimizer_step"] <= report["common_window_steps"]]
            ax.plot([r["optimizer_step"] for r in rows], [r["val/" + metric] for r in rows], "o-", color=color, lw=1.6, label=arm)
        ax.axhline(report["source"]["validation_metrics"]["val/" + metric], color="#64748b", ls="--", lw=1, label="frozen source (historical)")
        ax.axvline(1089, color="#64748b", ls=":", lw=1)
        ax.set(title=title, xlabel="Optimizer steps (common completed window)")
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    axes[0, 0].legend(fontsize=9)
    fig.suptitle("Equal observed window: both validation minima at step 1,089\nThe old source has a frozen teacher; non-KL objective is still not a fixed geometry-only score.", fontsize=13)
    for suffix in ("png", "svg"):
        fig.savefig(destination / f"training_common_window.{suffix}", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    deltas = report["candidate_best_to_last_weighted_contributions"]
    ordered = sorted(deltas, key=lambda k: abs(deltas[k]), reverse=True)
    axes[0].barh([k.removeprefix("val/weighted_") for k in ordered], [deltas[k] for k in ordered], color=["#bd4d39" if deltas[k] > 0 else "#2f8b70" for k in ordered])
    axes[0].invert_yaxis()
    axes[0].set(title="Candidate validation: best -> last\nAdditive contributions to total change", xlabel="Weighted loss delta (positive = higher)")
    keys = ("face_xyz", "edge_xyz", "incidence", "twin", "edge_feature")
    for key in keys:
        rows = report["arms"]["candidate"]["validation_history"]
        selected = report["arms"]["candidate"]["selected"]["validation_metrics"]["val/" + key]
        axes[1].plot([r["optimizer_step"] for r in rows], [100 * (r["val/" + key] / selected - 1) for r in rows], label=key)
    axes[1].axhline(0, color="#64748b", lw=1)
    axes[1].axvline(1089, color="#64748b", ls=":", lw=1)
    axes[1].set(title="Candidate validation relative to selected epoch\nGeometry and compatibility can move differently", xlabel="Optimizer steps", ylabel="Change from selected checkpoint (%)")
    axes[1].legend(fontsize=9)
    for suffix in ("png", "svg"):
        fig.savefig(destination / f"training_deterioration_components.{suffix}", dpi=160)
    plt.close(fig)


def write_markdown(report):
    candidate, control = report["arms"]["candidate"], report["arms"]["control"]
    source = report["source"]
    lines = ["# Encoder 边内容消息：训练审计", "", "按 [PLAN](PLAN.md) 只用 validation 选择权重，以下尚不是下游 solid 结果。所有曲线来自本地 W&B 数值流，不从截图估读。", "", "候选已完成 **14,520 步 / 40 epoch**，随后启动的原 GAT 解冻对照被中断；对照最后完整保存 **1,815 步 / 5 epoch**。停止时 console 到 epoch5 的20/363 batch，没有包含这些额外步的 checkpoint，因此不能把这20个 batch 宣称为可恢复状态。", "", "两组恰好都在 **epoch2 / step1,089** 达到各自已观察 validation 最优，且也是共同 **1,815步** 窗口的最优。本轮用这两个权重比较，checkpoint 训练曝光相同；完整运行预算不同，不据此声称完成了原定两组14,520步实验。", "", "| 模型 | selected step | val total | weighted KL | total−KL | last saved step / val total |", "|---|---:|---:|---:|---:|---:|"]
    for name, arm in (("原 conditioning（冻结 Encoder）", source), ("原 GAT 解冻对照", control["selected"]), ("边内容消息候选", candidate["selected"])):
        row = arm["validation_metrics"]
        ending = "历史源" if arm is source else (f"{control['last_saved']['global_step']} / {control['validation_history'][-1]['val/loss']:.8f}" if arm is control["selected"] else f"{candidate['last_saved']['global_step']} / {candidate['validation_history'][-1]['val/loss']:.8f}")
        lines.append(f"| {name} | {arm['global_step']} | {row['val/loss']:.8f} | {row['val/weighted_kl_loss']:.8f} | {row['val/non_kl_objective']:.8f} | {ending} |")
    lines += ["", "两组解冻后恢复了原 KL 项，旧冻结模型的 weighted KL 为0，因此总 loss 不能直接横比。KL 本身按每卡 packed face×32 维 **sum×1e−6**；训练每卡16件、验证每卡8件，又有旋转增强差别，不能把 train/val 总值落差直接称为过拟合。即使减去 KL，仍含指向随训练变化的 edge CNN teacher 的 feature L1，不是固定几何误差。", "", "| 验证指标（raw） | 冻结源 | 原 GAT 解冻 | 边内容消息 | 候选对对照 | 候选 last |", "|---|---:|---:|---:|---:|---:|"]
    for key in METRICS[3:]:
        a, b, s = candidate["selected"]["validation_metrics"]["val/" + key], control["selected"]["validation_metrics"]["val/" + key], source["validation_metrics"]["val/" + key]
        lines.append(f"| {key} | {s:.8f} | {b:.8f} | {a:.8f} | {relative(a,b):+.2f}% | {candidate['validation_history'][-1]['val/' + key]:.8f} |")
    lines += ["", "候选 best→last 总值增加约0.002873，其中 KL 增加约0.000095；edge-feature 项增加约0.001698。这一对端点中 face/edge XYZ 升高，incidence 与 twin 略降。但 early-best 本身经过选择，不能将这一对端点当成验证几何持续退化的证据。", "", "用前5个与后5个完整 epoch 均值检查整体趋势：", "", "| 候选指标 | train 后5/前5变化 | val 后5/前5变化 |", "|---|---:|---:|"]
    for key in ("loss", "non_kl_objective", "edge_feature", "face_xyz", "edge_xyz", "incidence", "twin"):
        values = report["candidate_window_trends"]
        lines.append(f"| {key} | {values['train'][key]['relative_change_percent']:+.2f}% | {values['val'][key]['relative_change_percent']:+.2f}% |")
    lines += ["", "窗口均值下，验证面/边 XYZ 仍有小幅改善，而总 loss 和 edge-feature loss 上升。因此当前证据更直接指向 **特征对齐项与几何目标的变化不一致**，尚不能称为整体几何过拟合。edge CNN 现在会训练，teacher 的坐标系/尺度也可能改变；这是解释假说，实际 feature 分布和梯度冲突未在此审计中测量。也不能凭 feature 项数值大就判断其梯度一定占主导。", "", "local 包含归一化 XYZ 与方向共6通道；bbox 是 center3+scale1。incidence/twin 有 GT 离散采样基线和容差处理，其损失值不是 CAD 最大偏差。分类 precision/recall 是训练日志批次归约，不是整件拓扑全对率。`delta_edge_mean` 是既有 refiner 的特征增量，与新增 GAT value 消息的输出幅度不同。", "", "![完整训练曲线](visualizations/training_history.png)", "", "![共同验证窗口](visualizations/training_common_window.png)", "", "![退化分解](visualizations/training_deterioration_components.png)", "", "## 参数与来源", "", "两组旧 Encoder、posterior 和 Decoder 均确实发生参数变化；不能再用旧 latent cache 评价新 Encoder。每件 GT 必须经各自 Encoder 重新计算 posterior mean。旧 diffusion 与新 latent 的语义兼容性没有经过验证。", "", "| 参数组 | tensor数量 | 候选 best 已改变 | 原 GAT best 已改变 | 候选 best 相对L2变动 |", "|---|---:|---:|---:|---:|"]
    for key, group in candidate["selected_changes_vs_source"].items():
        c = control["selected_changes_vs_source"][key]
        lines.append(f"| {key} | {group['tensor_count']} | {group['changed_tensors']} | {c['changed_tensors']} | {group['relative_l2_change']:.6g} |")
    lines += ["", "候选新增20个 tensor／51,680参数。5层 value MLP 的零初始化末层权重和 bias 在 best 已非零；所有20个 tensor 在 best→last 之间继续改变。这里只报告参数范数，不能由参数范数直接推断实际消息输出或几何位移幅度。", "", "| Value层 | best末层weight L2 | last末层weight L2 | best末层bias L2 |", "|---|---:|---:|---:|"]
    for i in (0, 2, 4, 6, 8):
        prefix = f"model.encoder.graph_face_edge.{i}.value_net.2."
        w, b = report["new_value_parameters"][prefix + "weight"], report["new_value_parameters"][prefix + "bias"]
        lines.append(f"| {i//2+1} | {w['selected_l2_norm']:.8f} | {w['last_l2_norm']:.8f} | {b['selected_l2_norm']:.8f} |")
    lines += ["", "源commit `0b145d5d2fe4d0e3140bd114bb1675047093ca97`；两次启动均clean，四个生产模型/训练/数据源码哈希仍一致。相同源 checkpoint、accepted split、seed42、8×16 batch、Adam2e−5、bf16、clip1；没有 scheduler。selected 每组新增139,392次对象呈现（1,089×128），不等于139,392件独立CAD。", ""]
    for name, arm in report["arms"].items():
        times = [datetime.fromisoformat(arm["termination"][key]).astimezone(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M:%S") for key in ("started_at_utc", "finished_at_utc")]
        selected = arm["selected"]
        lines += [f"- {name}：北京时间 {times[0]} → {times[1]}；exit={arm['termination']['exit_code']}；运行状态 {arm['termination']['state']}。", f"  - selected: `{selected['path']}`", f"  - SHA256: `{selected['sha256']}`"]
    lines += ["", "全数值、所有完整训练/验证epoch、top-k保留、源码/输入哈希和参数统计见 [training_audit.json](training_audit.json)。复现：`/mnt/d/anaconda3/envs/torch/bin/python checkpoints/encoder_edge_value/evaluation_20261015/build_training_audit.py`。", ""]
    (OUT / "TRAINING_AUDIT.md").write_text("\n".join(lines))


def main():
    torch.set_num_threads(2)
    active_path = TOPIC / "metadata/active_pair.json"
    active = json.loads(active_path.read_text())
    require(active["git"]["commit"] == "0b145d5d2fe4d0e3140bd114bb1675047093ca97", "Unexpected training source")
    source = load(SOURCE)
    require(sha256(SOURCE) == SHA["source"], "Changed source checkpoint")
    source_history, source_stream = history(SOURCE.parent.parent)
    source_rows = epoch_rows(source_history, "val")
    source_rows = [row for row in source_rows if row["optimizer_step"] == source["global_step"] and row["epoch"] == source["epoch"]]
    require(len(source_rows) == 1, "Ambiguous historical source validation")
    report = {"spec_sha256": sha256(OUT / "PLAN.md"), "selection_policy": "Minimum complete val/loss in shared completed-validation window; no test selection.", "source": {"path": str(SOURCE), "sha256": SHA["source"], "epoch": source["epoch"], "global_step": source["global_step"], "validation_metrics": source_rows[0], "history_path": str(source_stream), "history_sha256": sha256(source_stream)}, "arms": {}, "input_metadata_sha256": sha256(active_path)}
    for name in ("candidate", "control"):
        run = Path(active["arms"][name]["run_dir"])
        last_path = run / "weights/last.ckpt"
        last = load(last_path)
        selector = callback(last)
        best_path = Path(selector["best_model_path"])
        best = load(best_path)
        require(sha256(best_path) == SHA[name], "Selected checkpoint changed")
        rows, stream = history(run)
        validation, training = epoch_rows(rows, "val"), epoch_rows(rows, "train")
        best_row = min(validation, key=lambda row: (row["val/loss"], row["optimizer_step"]))
        require(best_row["optimizer_step"] == best["global_step"] == 1089 and best_row["val/loss"] == float(selector["best_model_score"]), "Selection mismatch")
        require(validation[-1]["optimizer_step"] == last["global_step"], "Saved validation step mismatch")
        require(len(validation) == len(training) and last["global_step"] == 363 * len(validation), "Unexpected complete epoch accounting")
        require(last["loops"]["fit_loop"]["epoch_loop.batch_progress"]["is_last_batch"], "Last saved epoch incomplete")
        hp = best["hyper_parameters"]
        require(not hp["model"]["freeze_encoder"] and hp["trainer"]["batch_size"] == 16 and hp["trainer"]["devices"] == 8, "Unexpected training scope")
        require(hp["model"]["encoder_edge_value"]["enabled"] == (name == "candidate"), "Wrong GAT variant")
        require(hp["model"]["loss_weights"] == source["hyper_parameters"]["model"]["loss_weights"], "Loss weights changed")
        source_field = "init_encoder_edge_value_checkpoint" if name == "candidate" else "checkpoint"
        require(Path(hp["model"][source_field]).resolve() == SOURCE, "Initialization source changed")
        require(hp["trainer"]["resume_from_checkpoint"] is None, "Unexpected resumed run")
        meta = json.loads((run / "metadata/run.json").read_text())
        invocation = run / "metadata/invocations" / meta["initial_invocation_id"]
        git = json.loads((invocation / "git.json").read_text())
        dirty = (invocation / "git.diff").read_text()
        require(not dirty and git["commit"] == active["git"]["commit"], "Dirty/different training source")
        code = {}
        for path in SOURCE_FILES:
            original = subprocess.check_output(["git", "show", f"{git['commit']}:{path}"], cwd=ROOT)
            current = (ROOT / path).read_bytes()
            require(original == current, f"Production code changed: {path}")
            code[path] = hashlib.sha256(current).hexdigest()
        state = best["state_dict"]
        indices = sorted({int(match[1]) for key in state if (match := re.match(r"model\.decoder\.inter\.layers\.(\d+)\.", key))})
        require(indices == list(range(8)), "Intersection layer count changed")
        console = Path(active["arms"][name]["console_log"]).read_text()
        progress = re.findall(r"Epoch (\d+):[^\r\n]*?\b(\d+)/363", console)
        arm = {"run_dir": str(run), "selected": {"path": str(best_path), "sha256": SHA[name], "epoch": best["epoch"], "global_step": best["global_step"], "validation_loss": best_row["val/loss"], "validation_metrics": best_row, "additional_object_presentations": best["global_step"] * 128}, "last_saved": {"path": str(last_path), "sha256": sha256(last_path), "epoch": last["epoch"], "global_step": last["global_step"], "complete_epochs": len(validation), "additional_object_presentations": last["global_step"] * 128}, "retained_top_k": [{"path": path, "validation_loss": float(value), "exists": Path(path).is_file()} for path, value in sorted(selector["best_k_models"].items(), key=lambda pair: float(pair[1]))], "validation_history": validation, "training_epoch_history": training, "history_path": str(stream), "history_sha256": sha256(stream), "selected_changes_vs_source": parameter_changes(state, source["state_dict"]), "last_changes_vs_source": parameter_changes(last["state_dict"], source["state_dict"]), "termination": {k: active["arms"][name][k] for k in ("state", "started_at_utc", "finished_at_utc", "exit_code")}, "last_console_epoch_batch": [int(x) for x in progress[-1]] if progress else None, "console_keyboard_interrupt": "Detected KeyboardInterrupt" in console, "console_sha256": sha256(Path(active["arms"][name]["console_log"])), "model_config": {k: hp["model"].get(k) for k in MODEL_FIELDS}, "trainer_config": {k: hp["trainer"].get(k) for k in TRAINER_FIELDS}, "source_commit": git["commit"], "training_clean": True, "source_file_hashes": code}
        report["arms"][name] = arm
        arm["initialization"] = {"field": source_field, "source": hp["model"][source_field], "fresh_optimizer": True}
        arm["configuration_changes_vs_source"] = {
            field: {"source": source["hyper_parameters"]["model"].get(field), "current": hp["model"].get(field)}
            for field in MODEL_FIELDS if source["hyper_parameters"]["model"].get(field) != hp["model"].get(field)
        }
        arm["optimizer_param_groups"] = [{key: value for key, value in group.items() if key != "params"}
                                         | {"parameter_tensor_count": len(group["params"])}
                                         for group in best["optimizer_states"][0]["param_groups"]]
        if name == "candidate":
            report["new_value_parameters"] = branch_stats(state, last["state_dict"], source["state_dict"])
        else:
            require(set(state) == set(source["state_dict"]), "Control changed architecture keys")
    report["common_window_steps"] = min(a["last_saved"]["global_step"] for a in report["arms"].values())
    for arm in report["arms"].values():
        window = [row for row in arm["validation_history"] if row["optimizer_step"] <= report["common_window_steps"]]
        best = min(window, key=lambda row: (row["val/loss"], row["optimizer_step"]))
        require(best["optimizer_step"] == arm["selected"]["global_step"], "Best not in shared window")
        arm["common_window_validation_count"] = len(window)
        arm["selected_comparison_vs_source"] = comparisons(arm["selected"]["validation_metrics"], report["source"]["validation_metrics"])
    a, b = (report["arms"][name]["selected"]["validation_metrics"] for name in ("candidate", "control"))
    last = report["arms"]["candidate"]["validation_history"][-1]
    report["candidate_selected_vs_control_selected"] = comparisons(a, b)
    report["candidate_last_vs_selected"] = comparisons(last, a)
    report["candidate_best_to_last_weighted_contributions"] = {key: last[key] - value for key, value in a.items() if key.startswith("val/weighted_")}
    require(abs(sum(report["candidate_best_to_last_weighted_contributions"].values()) - (last["val/loss"] - a["val/loss"])) < 2e-7, "Decomposition does not close")
    report["candidate_window_trends"] = {}
    for prefix, field in (("train", "training_epoch_history"), ("val", "validation_history")):
        rows = report["arms"]["candidate"][field]
        trends = {}
        for key in METRICS:
            early = statistics.mean(row[prefix + "/" + key] for row in rows[:5])
            late = statistics.mean(row[prefix + "/" + key] for row in rows[-5:])
            trends[key] = {"first_five_mean": early, "last_five_mean": late, "relative_change_percent": relative(late, early)}
        report["candidate_window_trends"][prefix] = trends
    report["splits"] = {}
    for name, record in active["splits"].items():
        path = Path(record["path"])
        ids = path.read_text().splitlines()
        require(len(ids) == len(set(ids)) == record["count"] and sha256(path) == record["sha256"], "Split changed")
        report["splits"][name] = record
    report["limitations"] = ["Training/validation sum-KL use different per-device batch sizes (16/8); training CAD is cube24 augmented, validation canonical.", "Both new encoders move the detached edge-feature teacher; total minus KL is not a fixed geometry-only score.", "Validation samples posterior; reconstruction evaluation uses each checkpoint's own mean. Old latent cache/diffusion are not assumed compatible.", "Same selected step and shared validation window do not complete the unequal full-run budgets, nor eliminate single-seed uncertainty.", "New branch parameter changes demonstrate learning, not output magnitude or causal contribution to final geometry.", "No test metric selected checkpoints. Native solid/failure conclusions belong to downstream reports."]
    write_plots(report)
    write_markdown(report)
    (OUT / "training_audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"output": str(OUT), "selected_steps": {name: arm["selected"]["global_step"] for name, arm in report["arms"].items()}, "common_window_steps": report["common_window_steps"], "new_parameters": sum(r["numel"] for r in report["new_value_parameters"].values())}))


if __name__ == "__main__":
    main()
