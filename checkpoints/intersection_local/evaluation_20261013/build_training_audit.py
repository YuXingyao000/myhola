"""Read-only training audit; regenerate compact records and static curve figures."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import torch
from wandb.proto import wandb_internal_pb2
from wandb.sdk.internal.datastore import DataStore


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
RUN = REPO / "checkpoints/intersection_local/full_capacity/20261012T141134.581695Z"
PREVIOUS = REPO / (
    "checkpoints/intersection_refinement/frozen/20261011T123836.485257Z/weights/"
    "epoch=0010-step=00003993-val_loss=0.039476.ckpt"
)
LEGACY = Path("/mnt/d/data/ae_checkpoints/1119_deepcad_aug1_11k.ckpt")
WAND_FILE = RUN / "wandb/run-20261012_141137-f4822e2bc26c/run-f4822e2bc26c.wandb"
SELECTED = RUN / "weights/epoch=0042-step=00015609-val_loss=0.098798.ckpt"
FONT = Path("/root/.cache/hola-browser-libs/fonts/NotoSansCJKsc-Regular.otf")
DISPLAY_ZONE = ZoneInfo("Asia/Shanghai")
SMOKE = REPO / "checkpoints/intersection_local/validation_full_capacity/gpu_smoke.json"
OUTPUT_LOG = WAND_FILE.parent / "files/output.log"


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": file_hash(path)}


def epoch_history() -> list[dict]:
    store = DataStore()
    store.open_for_scan(str(WAND_FILE))
    epochs = {}
    while (data := store.scan_data()) is not None:
        record = wandb_internal_pb2.Record()
        record.ParseFromString(data)
        if not record.HasField("history"):
            continue
        row = {
            item.key or "/".join(item.nested_key): json.loads(item.value_json)
            for item in record.history.item
        }
        if "val/loss" not in row and "train/loss_epoch" not in row:
            continue
        epoch = int(row["epoch"])
        chosen = {
            key: value
            for key, value in row.items()
            if key.startswith(("train/", "val/"))
            or key in ("epoch", "trainer/global_step")
        }
        if "val/loss" in row:
            logged = datetime.fromtimestamp(row["_timestamp"], tz=timezone.utc)
            chosen["validation_logged_utc"] = logged.isoformat()
            chosen["validation_logged_display"] = logged.astimezone(DISPLAY_ZONE).isoformat()
        previous = epochs.setdefault(epoch, {})
        for key in previous.keys() & chosen.keys():
            if previous[key] != chosen[key]:
                raise ValueError(f"Conflicting epoch record: {epoch} {key}")
        previous.update(chosen)
    return [epochs[epoch] for epoch in sorted(epochs)]


def json_scalar(value):
    if isinstance(value, torch.Tensor):
        return value.item() if value.numel() == 1 else value.tolist()
    if isinstance(value, dict):
        return {key: json_scalar(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_scalar(item) for item in value]
    return value


def checkpoint_record(path: Path, checkpoint: dict) -> dict:
    return {
        **file_record(path),
        "epoch": checkpoint["epoch"],
        "global_step": checkpoint["global_step"],
        "state_key_count": len(checkpoint["state_dict"]),
        "callbacks": json_scalar(checkpoint.get("callbacks", {})),
    }


def metric_trend(rows: list[dict], key: str) -> dict:
    points = [(row["epoch"], row[key]) for row in rows]
    best = min(points, key=lambda item: item[1])
    return {
        "first": points[0][1],
        "last": points[-1][1],
        "minimum": best[1],
        "minimum_epoch": best[0],
        "last_vs_minimum_percent": (points[-1][1] / best[1] - 1) * 100,
        "strictly_decreasing": all(b[1] < a[1] for a, b in zip(points, points[1:])),
    }


def draw_curves(rows: list[dict], destination: Path, *, late: bool = False) -> None:
    font_manager.fontManager.addfont(str(FONT))
    family = font_manager.FontProperties(fname=FONT).get_name()
    plt.rcParams.update({
        "font.family": family, "axes.unicode_minus": False,
        "font.size": 12, "axes.titlesize": 15, "axes.labelsize": 11,
        "figure.facecolor": "#f4f6f9", "axes.facecolor": "white",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#cbd5e1", "text.color": "#17243b",
        "axes.labelcolor": "#475569", "xtick.color": "#64748b",
        "ytick.color": "#64748b", "svg.fonttype": "path",
    })
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.3))
    panels = [
        ("总训练目标", "train/loss_epoch", "val/loss"),
        ("边特征对齐", "train/edge_feature", "val/edge_feature"),
        ("边 bbox：中心与尺度", "train/refined_edge_bbox", "val/refined_edge_bbox"),
        ("边的局部几何", "train/refined_edge_local", "val/refined_edge_local"),
        ("面的局部几何", "train/refined_face_local", "val/refined_face_local"),
        ("边面贴合目标", "train/incidence", "val/incidence"),
    ]
    all_rows = rows
    if late:
        rows = rows[-12:]
    epochs = [row["epoch"] for row in rows]
    selected_epoch = min(all_rows, key=lambda row: row["val/loss"])["epoch"]
    for ax, (title, training, validation) in zip(axes.flat, panels):
        ax.plot(epochs, [row[training] for row in rows], color="#3e7cb1", lw=2, label="训练 epoch 均值")
        ax.plot(epochs, [row[validation] for row in rows], color="#d97836", lw=2.2,
                marker="o", markersize=3.5, label="验证 epoch 均值")
        ax.set_title(title, loc="left", pad=12)
        ax.set_xlabel("epoch（从 0 计数）")
        ax.set_ylabel("原始 loss（越低越好）")
        ax.set_xticks([32, 34, 36, 38, 40, 42, 43] if late else [0, 8, 16, 24, 32, 40, 43])
        ax.axvline(selected_epoch, color="#94a3b8", linewidth=1, linestyle="--", alpha=0.65)
        ax.grid(axis="y", color="#e2e8f0", linewidth=0.7)
        ax.tick_params(axis="both", length=0)
        ax.margins(x=0.03, y=0.16)
        final = rows[-1][validation]
        ax.annotate(f"末轮 {final:.5f}", xy=(epochs[-1], final), xytext=(-6, 10),
                    textcoords="offset points", ha="right", color="#a45121", fontsize=10)
        if validation in ("val/loss", "val/refined_face_local", "val/incidence"):
            best = min(all_rows, key=lambda row: row[validation])
            if best["epoch"] < epochs[0]:
                continue
            ax.scatter([best["epoch"]], [best[validation]], color="#198579", s=40, zorder=5)
            left_edge = best["epoch"] == epochs[0]
            ax.annotate(f"最低 epoch {best['epoch']}", (best["epoch"], best[validation]),
                        xytext=(7 if left_edge else -7, -24), textcoords="offset points",
                        ha="left" if left_edge else "right", color="#137066", fontsize=10)
    axes[0, 0].legend(frameon=False, fontsize=10, loc="upper right")
    axes[1, 1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    span = "最后 12 轮" if late else "44 轮完整验证记录"
    fig.suptitle(f"完整容量局部 intersection：{span}", x=0.055, y=0.97, ha="left", fontsize=22)
    fig.text(0.055, 0.915, "512维 × 8层；按验证总目标选 epoch 42 / step 15609。末轮总目标回升1.79%，贴合目标继续下降。", fontsize=12, color="#475569")
    fig.text(0.055, 0.045, "验证使用 posterior 采样；训练均值来自一轮内不断变化的权重。两者不等同于本次固定 mean 的测试指标。", fontsize=10, color="#64748b")
    fig.text(0.055, 0.023, "单轮验证反弹不足以证明持续过拟合或收敛。显示时区：Asia/Shanghai；起始时间为10月12日22:11。", fontsize=10, color="#64748b")
    fig.subplots_adjust(left=0.065, right=0.978, top=0.84, bottom=0.12, hspace=0.42, wspace=0.28)
    destination.mkdir(parents=True, exist_ok=True)
    filename = "training_curves_late" if late else "training_curves"
    fig.savefig(destination / (filename + ".png"), dpi=180)
    fig.savefig(destination / (filename + ".svg"))
    plt.close(fig)


def main() -> None:
    torch.set_num_threads(1)
    rows = epoch_history()
    selected = torch.load(SELECTED, map_location="cpu", weights_only=False, mmap=True)
    baseline = torch.load(PREVIOUS, map_location="cpu", weights_only=False, mmap=True)
    legacy = torch.load(LEGACY, map_location="cpu", weights_only=False, mmap=True)
    state = selected["state_dict"]
    frozen_keys = [key for key in state if key.startswith(("model.encoder.", "model.posterior."))]
    comparisons = []
    for key in frozen_keys:
        legacy_key = key.replace("model.encoder.", "model.").replace(
            "model.posterior.projection.", "model.gaussian_proj.",
        )
        comparisons.append({
            "key": key, "legacy_key": legacy_key,
            "shape": list(state[key].shape), "dtype": str(state[key].dtype),
            "equal_previous_refinement": key in baseline["state_dict"] and torch.equal(state[key], baseline["state_dict"][key]),
            "equal_legacy": legacy_key in legacy["state_dict"] and torch.equal(state[key], legacy["state_dict"][legacy_key]),
        })
    if not all(item["equal_previous_refinement"] and item["equal_legacy"] for item in comparisons):
        raise ValueError("Frozen encoder/posterior mismatch")
    config = selected["hyper_parameters"]
    inter_config = config["model"]["intersection"]
    expected_config = {"block_type": "transformer_decoder", "hidden_dim": 512,
                       "num_heads": 8, "num_layers": 8, "ffn_dim": 2048,
                       "dropout": 0.1, "activation": "relu"}
    assert all(inter_config[key] == value for key, value in expected_config.items())
    inter_state = {key: value for key, value in state.items() if key.startswith("model.decoder.inter.")}
    assert len(inter_state) == 157 and sum(value.numel() for value in inter_state.values()) == 34015296
    non_inter = {key: value for key, value in state.items() if not key.startswith("model.decoder.inter.")}
    old_non_inter = {key: value for key, value in baseline["state_dict"].items() if not key.startswith("model.decoder.inter.")}
    assert non_inter.keys() == old_non_inter.keys() and len(non_inter) == 581
    assert all(value.shape == old_non_inter[key].shape for key, value in non_inter.items())
    smoke = json.loads(SMOKE.read_text())
    assert smoke["status"] == "passed" and smoke["retained_state_keys_exact"] == 581
    assert config["model"]["init_refinement_checkpoint"] == str(PREVIOUS)
    assert config["trainer"]["resume_from_checkpoint"] is None
    callback = next(value for value in selected["callbacks"].values() if value.get("monitor") == "val/loss")
    if Path(callback["best_model_path"]) != SELECTED:
        raise ValueError("Selected checkpoint is not callback best")
    if min(rows, key=lambda row: row["val/loss"])["epoch"] != selected["epoch"]:
        raise ValueError("History and callback selection disagree")
    last_path = RUN / "weights/last.ckpt"
    last = torch.load(last_path, map_location="cpu", weights_only=False, mmap=True)
    last_record = checkpoint_record(last_path, last)
    selected_record = checkpoint_record(SELECTED, selected)
    last_record["file_identical_to_selected"] = last_record["sha256"] == selected_record["sha256"]
    last_record["state_identical_to_selected"] = set(last["state_dict"]) == set(state) and all(
        torch.equal(value, last["state_dict"][key]) for key, value in state.items()
    )
    last_record["frozen_tensors_equal_previous_refinement"] = all(
        torch.equal(last["state_dict"][key], baseline["state_dict"][key]) for key in frozen_keys
    )
    assert last_record["frozen_tensors_equal_previous_refinement"]
    metric_names = ["train/loss_epoch", "val/loss", "val/edge_feature", "val/refined_edge_bbox",
                    "val/refined_edge_local", "val/edge_xyz", "train/refined_face_local",
                    "val/refined_face_local", "val/incidence", "val/twin"]
    audit = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "created_display_time": datetime.now(DISPLAY_ZONE).isoformat(),
        "display_timezone": "Asia/Shanghai",
        "run_started_utc": "2026-10-12T14:11:34.581695+00:00",
        "run_started_display_time": "2026-10-12T22:11:34.581695+08:00",
        "run": str(RUN),
        "checkpoint_recommendation": {
            "path": str(SELECTED), "epoch": selected["epoch"], "global_step": selected["global_step"],
            "best_model_score": float(callback["best_model_score"]),
            "selection_rule": "Minimum recorded val/loss; checkpoint callback and complete local W&B history agree; no test-based selection.",
            "sha256": selected_record["sha256"],
        },
        "selected_checkpoint": selected_record,
        "last_checkpoint": last_record,
        "previous_refinement_checkpoint": checkpoint_record(PREVIOUS, baseline),
        "legacy_source_checkpoint": {**file_record(LEGACY), "epoch": legacy["epoch"], "global_step": legacy["global_step"]},
        "sources": {
            "history": file_record(WAND_FILE),
            "hydra_config": file_record(RUN / "metadata/hydra/config.yaml"),
            "wandb_config": file_record(RUN / "wandb/run-20261012_141137-f4822e2bc26c/files/config.yaml"),
            "invocation": file_record(RUN / "metadata/invocations/20261012T141134.581695Z/command.json"),
            "train_split": file_record(REPO / "checkpoints/joint_boundary/validation/train_contract_audit/accepted.txt"),
            "val_split": file_record(REPO / "checkpoints/joint_boundary/validation/val_contract_audit/accepted.txt"),
            "test_split": file_record(REPO / "checkpoints/joint_boundary/validation/test_contract_audit/accepted.txt"),
            "audit_builder": file_record(Path(__file__).resolve()),
            "font": file_record(FONT),
            "initialization_smoke": file_record(SMOKE),
            "training_output_log": file_record(OUTPUT_LOG),
        },
        "training_protocol": {
            "devices": 8, "per_device_batch_size": 16, "learning_rate": 2e-5,
            "precision": "bf16-mixed", "seed": 42, "freeze_encoder": True,
            "validation_uses_posterior_mean": False, "recorded_complete_epochs": len(rows),
            "configured_max_epochs": 10000, "configured_max_steps": 1000000,
            "initialization": "previous refinement, excluding the replaced local intersection weights",
            "global_step_note": "W&B epoch-end trainer/global_step is zero based (selected 15608, last complete 15971); checkpoints count completed optimizer steps (15609 and 15972).",
            "run_finished_observation": {
                "checked_utc_approx": "2026-10-13T04:16:00Z",
                "training_process_running": False, "gpu_processes": 0,
                "gpus": "8 NVIDIA RTX 4090, 24 GB each, all idle at observation",
                "termination_reason": "Saved W&B output.log records Detected KeyboardInterrupt, attempting graceful shutdown.",
                "partial_epoch_at_interrupt": 44, "displayed_batches_in_partial_epoch": "41/363",
                "saved_last_scope": "last.ckpt is epoch43/step15972; partial epoch44 is not represented by that saved checkpoint.",
                "configured_budget_reached": False, "convergence_established": False,
            },
        },
        "architecture_audit": {
            "intersection_config": inter_config, "intersection_state_keys": len(inter_state),
            "intersection_state_elements": sum(value.numel() for value in inter_state.values()),
            "previous_intersection_state_elements": sum(value.numel() for key, value in baseline["state_dict"].items() if key.startswith("model.decoder.inter.")),
            "non_inter_keys_and_shapes_match_baseline": len(non_inter),
            "non_inter_identical_values_at_trained_checkpoint": sum(torch.equal(value, old_non_inter[key]) for key, value in non_inter.items()),
            "initialization_contract": "Source migration preserves all 581 non-inter tensors exactly; local inter is newly initialized. This is documented and tested by the linked pretraining smoke, not inferred from equality of trained decoder tensors.",
            "capacity_limit": "Transformer blocks recover the legacy 512x8 capacity; local query/memory/output projections make inter totals 34,015,296 vs 33,683,008, not exactly equal total parameter counts.",
        },
        "frozen_encoder_check": {
            "comparison": "torch.equal on every encoder and posterior state tensor; no tolerance",
            "encoder_tensor_count": sum(key.startswith("model.encoder.") for key in frozen_keys),
            "posterior_tensor_count": sum(key.startswith("model.posterior.") for key in frozen_keys),
            "all_equal_previous_refinement": True, "all_equal_legacy": True,
            "legacy_mapping": "model.encoder.X -> model.X; model.posterior.projection.X -> model.gaussian_proj.X",
            "scope_limit": "This establishes weight identity only; latent input, normalization and posterior/cache protocol require the separate evaluation source audit.",
            "tensors": comparisons,
        },
        "validation_history": [
            {key: value for key, value in row.items() if key.startswith(("val/", "validation_logged_")) or key in ("epoch", "trainer/global_step")}
            for row in rows
        ],
        "trends": {key: metric_trend(rows, key) for key in metric_names},
        "interpretation": {
            "observations_zh": [
                "有44轮完整验证（epoch0–43）；训练总目标每轮下降，验证总目标总体下降并伴随末段波动。",
                "验证总目标在epoch42最低，末轮比最低高1.79%；边特征、bbox、局部几何和global XYZ也在42最低，43回升。",
                "贴合与twin目标在epoch43达到记录内最低；不是所有验证目标同时退步。",
                "面局部几何的训练和验证最低值均在epoch0，末轮分别比最低高4.91%和1.90%。",
            ],
            "limitations_zh": [
                "最后一轮部分验证目标回升与早期过拟合相容，但单轮反弹不足以证明持续泛化恶化，也不能证明收敛。",
                "部分目标回升与多目标权衡/遗忘相容，但曲线本身不能建立机制因果。",
                "训练epoch均值使用一轮内变化的权重；验证使用epoch末权重与posterior采样。不能把train低/高于val简单解释为泛化差距。",
                "这些训练目标不能替代固定latent的raw几何、拓扑、原生STEP solid和失败统计。",
                "本次完整容量模型增加了训练曝光；与refinement或上轮小local相比都不是等预算的纯架构消融。",
            ],
        },
    }
    (ROOT / "training_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    fields = ["epoch", "trainer/global_step"] + sorted(set().union(*(row.keys() for row in rows)) - {"epoch", "trainer/global_step"})
    with (ROOT / "training_history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    draw_curves(rows, ROOT / "visualizations")
    draw_curves(rows, ROOT / "visualizations", late=True)
    print(json.dumps({"checkpoint": audit["checkpoint_recommendation"], "epochs": len(rows), "output": str(ROOT)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
