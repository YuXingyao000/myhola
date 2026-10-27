"""真实批次检查：严格权重、Adam续接、新目标及全模型梯度；不评价研究效果。"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import torch

from src.brepnet.dataset import AutoEncoder_joint_dataset
from src.brepnet.models import build_model


def to_device(value):
    if isinstance(value, torch.Tensor):
        return value.cuda()
    if isinstance(value, dict):
        return {key: to_device(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(to_device(item) for item in value)
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--parameters-sha256", required=True)
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--ids", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    torch.manual_seed(42)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("highest")
    with initialize_config_dir(config_dir=str(root / "configs"), version_base=None):
        cfg = compose(config_name="train_vae_uv_intersection_supervision_v2", overrides=[
            f"model.checkpoint={args.source}", f"dataset.uv_curve_target_root={args.cache}",
            f"dataset.raw_data_root={args.raw_root}", f"dataset.val_dataset={args.ids}",
            "dataset.is_aug=0",
            f"dataset.uv_curve_target_parameters_sha256={args.parameters_sha256}",
        ])
    dataset = AutoEncoder_joint_dataset("validation", cfg.dataset)
    batch = to_device(dataset.collate_fn([dataset[0], dataset[1]]))
    model = build_model(cfg.model)
    checkpoint = torch.load(args.source, map_location="cpu", weights_only=False, mmap=True)
    source_state = {name.removeprefix("model."): value for name, value in checkpoint["state_dict"].items()}
    assert model.state_dict().keys() == source_state.keys()
    assert all(torch.equal(value, source_state[name]) for name, value in model.state_dict().items())
    assert all(parameter.requires_grad for parameter in model.parameters())
    model = model.cuda().train()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.trainer.learning_rate)
    optimizer.load_state_dict(checkpoint["optimizer_states"][0])
    initial_steps = sorted({int(state["step"]) for state in optimizer.state.values()})
    assert initial_steps == [checkpoint["global_step"]] == [30000]
    assert len(optimizer.state) == len(list(model.parameters()))
    optimizer.zero_grad(set_to_none=True)
    probes = {
        name: next(module.parameters()) for name, module in (
            ("encoder", model.encoder), ("posterior", model.posterior),
            ("face_decoder", model.decoder.face_decoder),
            ("uv_head", model.decoder.intersection.uv_head),
        )
    }
    before = {name: parameter.detach().clone() for name, parameter in probes.items()}
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        losses, data = model(batch)
    assert data == {}
    assert all(torch.isfinite(value).all() for value in losses.values())
    weighted = {key: value for key, value in losses.items() if key.startswith("weighted_")}
    torch.testing.assert_close(sum(weighted.values()), losses["total_loss"])
    uv_grad = torch.autograd.grad(losses["curve_uv"], probes["uv_head"], retain_graph=True)[0]
    assert torch.isfinite(uv_grad).all() and uv_grad.abs().sum() > 0
    losses["total_loss"].backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    gradient_probes = {name: float(p.grad.abs().sum()) for name, p in probes.items()}
    assert all(value > 0 for value in gradient_probes.values())
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    torch.cuda.synchronize()
    final_steps = sorted({int(state["step"]) for state in optimizer.state.values()})
    assert final_steps == [30001]
    changes = {name: float((p.detach() - before[name]).abs().max()) for name, p in probes.items()}
    assert all(value > 0 for value in changes.values())
    report = {
        "passed": True, "scope": "Implementation and continuation check, not research effectiveness.",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source": model.initialization_report, "all_source_tensors_bitwise_preserved_before_update": True,
        "source_step": checkpoint["global_step"], "adam_steps_before": initial_steps,
        "adam_steps_after": final_steps, "adam_parameter_states": len(optimizer.state),
        "sample_ids": list(batch["v_prefix"]), "face_counts": batch["face_counts"].tolist(),
        "parameters": sum(p.numel() for p in model.parameters()), "frozen_parameters": 0,
        "losses": {key: float(value.detach()) for key, value in losses.items()},
        "weighted_sum_matches_total": True, "all_gradients_finite": True,
        "probe_gradient_l1": gradient_probes, "probe_max_parameter_update": changes,
        "gradient_norm_before_clip": float(norm),
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "step_seconds": time.perf_counter() - started,
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key not in {"config", "source"}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
