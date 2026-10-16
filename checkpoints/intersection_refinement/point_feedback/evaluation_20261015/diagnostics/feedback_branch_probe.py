"""Fixed two-case final-feedback before/after probe, never a retraining ablation."""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
import sys

import numpy as np
import torch

OUT = Path(__file__).resolve().parent
EVAL = OUT.parent
REPO = EVAL.parents[3]
sys.path.insert(0, str(REPO))

from tools.intersection_evaluate_inference import load_intersection, validate_method
from tools.vae_compare_inference import configure_fp32, file_sha256, latent_sha256
from tools.vae_compare_metrics import squared_chamfer
from src.brepnet.data.shared_boundaries import build_shared_boundary_targets
from src.brepnet.models.blocks import closest_surface_points, symmetric_pair_logits

CASES = {"00335203": (0, 6), "00326127": (3, 11)}


def snapshot(value, prefix=""):
    if isinstance(value, torch.Tensor):
        return {prefix: value.detach().cpu().clone()}
    result = {}
    for field in dataclasses.fields(value):
        result.update(snapshot(getattr(value, field.name), prefix + "." + field.name))
    return result


def compare_exact(left, right, label):
    if left.keys() != right.keys():
        raise ValueError(f"Different fields: {label}")
    for key in left:
        torch.testing.assert_close(left[key], right[key], rtol=0, atol=0,
                                   msg=lambda message: f"{label}/{key}: {message}")
    return {"tensor_count": len(left), "exact": True}


def displacement_stats(delta):
    lengths = np.linalg.norm(np.asarray(delta, dtype=np.float64), axis=-1).reshape(-1)
    if not len(lengths):
        return {"point_count": 0, "mean": None, "median": None, "p90": None, "max": None}
    return {"point_count": len(lengths), "mean": float(lengths.mean()),
            "median": float(np.median(lengths)), "p90": float(np.quantile(lengths, .9)),
            "max": float(lengths.max())}


def owner_distances(curve, faces, owners):
    surfaces = np.asarray(faces[list(owners)], dtype=np.float64)
    points = np.repeat(np.asarray(curve, dtype=np.float64)[None], len(owners), axis=0)
    with torch.inference_mode():
        _, distances = closest_surface_points(torch.from_numpy(points), torch.from_numpy(surfaces), 2)
    return {str(owner): {"samples": len(row), "mean": float(row.mean()), "max": float(row.max())}
            for owner, row in zip(owners, distances.numpy())}


def main():
    target = OUT / "feedback_branch_probe.json"
    if target.exists():
        raise FileExistsError(target)
    protocol_path = EVAL / "metadata/inference/posterior_mean/protocol.json"
    protocol = json.loads(protocol_path.read_text())
    checkpoint = Path(protocol["checkpoint"])
    assert protocol["method"] == "point_feedback"
    assert file_sha256(checkpoint) == protocol["checkpoint_sha256"]
    settings = configure_fp32()
    for name in ("dtype", "autocast", "seed", "matmul_precision", "matmul_allow_tf32",
                 "cudnn_allow_tf32", "cudnn_benchmark", "cudnn_deterministic", "deterministic_algorithms"):
        assert settings[name] == protocol["numeric_settings"][name], name
    model, audit = load_intersection(checkpoint, Path(protocol["legacy_checkpoint"]), torch.device("cuda:0"), list(CASES))
    validate_method(model, "point_feedback")
    sources = {str(path): file_sha256(path) for path in (
        Path(__file__), OUT / "FEEDBACK_BRANCH_SPEC.md", OUT / "feedback_branch_initial_check.json",
        OUT / "outlier_pair_probe.json", protocol_path,
        REPO / "src/brepnet/models/blocks.py", REPO / "src/brepnet/models/vae.py",
        REPO / "tools/intersection_evaluate_inference.py", REPO / "tools/vae_compare_metrics.py",
        REPO / "src/brepnet/data/shared_boundaries.py")}
    result = {"scope": "fixed trained network final-branch comparison; not a retraining causal ablation",
              "checkpoint": str(checkpoint), "checkpoint_sha256": protocol["checkpoint_sha256"],
              "model_epoch": audit["intersection"]["epoch"], "global_step": audit["intersection"]["global_step"],
              "numeric_settings": settings, "device": torch.cuda.get_device_name(),
              "comparison_protocol": "one full decode; final module input/output share identical upstream execution",
              "initial_check_record": str(OUT / "feedback_branch_initial_check.json"),
              "fixed_cases": {key: list(pair) for key, pair in CASES.items()},
              "source_sha256": sources, "samples": {}}
    arrays_dir = OUT / "feedback_branch_arrays"
    arrays_dir.mkdir(exist_ok=True)
    for sid, pair in CASES.items():
        raw_path = EVAL / "raw_predictions/posterior_mean/point_feedback" / sid / "data.npz"
        latent_path = Path(protocol["reference_root"]) / "raw_predictions/posterior_mean/legacy" / sid / "data.npz"
        gt_path = Path(protocol["data_root"]) / sid / "data.npz"
        for path in (raw_path, latent_path, gt_path):
            sources[str(path)] = file_sha256(path)
        with np.load(raw_path, allow_pickle=False) as archive:
            raw = {key: archive[key] for key in archive.files}
        with np.load(latent_path, allow_pickle=False) as archive:
            latent_array = archive["face_latents"].copy()
        np.testing.assert_array_equal(latent_array, raw["face_latents"])
        assert latent_sha256(latent_array) == str(raw["input_latent_sha256"])
        captured = {}
        module = model.decoder.curve_point_feedback

        def before_hook(_module, args):
            edge, face, edge_features, face_features, owners = args
            captured.update(before=snapshot(edge), face=snapshot(face),
                            edge_input_object=edge,
                            edge_features=edge_features.detach().cpu().clone(),
                            face_features=face_features.detach().cpu().clone(),
                            pairs=owners.detach().cpu().clone())

        def after_hook(_module, args, output):
            captured["input_checks"] = {
                "edge_geometry": compare_exact(captured["before"], snapshot(args[0]), "edge input not mutated"),
                "face_geometry": compare_exact(captured["face"], snapshot(args[1]), "face input not mutated"),
                "features_and_pairs": compare_exact(
                    {"E": captured["edge_features"], "F": captured["face_features"], "pairs": captured["pairs"]},
                    {"E": args[2].detach().cpu(), "F": args[3].detach().cpu(), "pairs": args[4].detach().cpu()},
                    "features and pairs not mutated"),
            }
            captured["after"] = snapshot(output)

        def refiner_hook(_module, _args, output):
            captured["refiner_delta_face"] = output[0].detach().cpu().clone()
            captured["refiner_delta_edge"] = output[1].detach().cpu().clone()

        def classifier_hook(_module, _args, output):
            # Overwrite coarse output; the last call is final refined logits.
            captured["classifier_logits"] = output.detach().clone().squeeze(-1)

        before_handle = module.register_forward_pre_hook(before_hook)
        after_handle = module.register_forward_hook(after_hook)
        refiner_handle = model.decoder.refiner.register_forward_hook(refiner_hook)
        classifier_handle = model.decoder.classifier.register_forward_hook(classifier_hook)
        latent = torch.from_numpy(latent_array).cuda()
        try:
            with torch.inference_mode():
                enabled = model.decode_state(latent)
        finally:
            before_handle.remove()
            after_handle.remove()
            refiner_handle.remove()
            classifier_handle.remove()
        # Both states share the exact same prefix. Only the final edge geometry
        # is replaced by its captured input; no second CUDA decode is performed.
        bypassed = dataclasses.replace(enabled, refined=dataclasses.replace(
            enabled.refined, edge_geometry=captured["edge_input_object"]))
        enabled_state, bypassed_state = snapshot(enabled), snapshot(bypassed)
        mutable = {".refined.edge_geometry.points", ".refined.edge_geometry.local_points"}
        invariants = compare_exact({k: v for k, v in enabled_state.items() if k not in mutable},
                                   {k: v for k, v in bypassed_state.items() if k not in mutable}, "branch invariants")
        compare_exact(captured["before"], snapshot(bypassed.refined.edge_geometry), "hook input remains unchanged")
        hook_invariants = compare_exact(
            {"F": captured["face_features"], "E": captured["edge_features"], "pairs": captured["pairs"],
             "delta_face": captured["refiner_delta_face"], "delta_edge": captured["refiner_delta_edge"],
             "logits": symmetric_pair_logits(captured["classifier_logits"], enabled.pairs.reverse).cpu()},
            {"F": enabled.refined.face_features.cpu(), "E": enabled.refined.edge_features.cpu(),
             "pairs": enabled.pairs.indices.cpu(), "delta_face": enabled.delta_face.cpu(),
             "delta_edge": enabled.delta_edge.cpu(), "logits": enabled.refined.edge_logits.cpu()},
            "captured prefix equals final output")
        compare_exact(captured["face"], snapshot(enabled.refined.face_geometry), "face geometry stays unchanged")
        torch.testing.assert_close(enabled.refined.edge_geometry.local_points[..., 3:],
                                   bypassed.refined.edge_geometry.local_points[..., 3:], rtol=0, atol=0)
        exported = model.export_items(enabled, latent)[0]
        replay = {}
        for key in ("pred_face", "pred_edge", "pred_edge_face_connectivity", "pred_face_adj",
                    "pred_face_adj_prob", "all_candidate_pairs", "all_candidate_prob", "coarse_candidate_prob"):
            left, right = exported[key], raw[key]
            if left.shape != right.shape:
                raise ValueError(f"Raw replay shape changed: {sid}/{key}")
            if left.dtype.kind in "biu":
                np.testing.assert_array_equal(left, right)
                replay[key] = {"exact": True, "max_abs_error": 0.}
            else:
                error = float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64)))) if left.size else 0.
                if error > 1e-5:
                    raise ValueError(f"Raw replay error exceeded spec: {sid}/{key}: {error}")
                replay[key] = {"exact": bool(np.array_equal(left, right)), "max_abs_error": error}
        pairs = captured["pairs"].numpy()
        probabilities = enabled.refined.edge_logits.float().sigmoid().cpu().numpy()
        active = probabilities > model.edge_threshold
        before = captured["before"][".points"].numpy()
        after = captured["after"][".points"].numpy()
        delta = after - before
        faces = enabled.refined.face_geometry.points.cpu().numpy()
        with np.load(gt_path, allow_pickle=False) as gt:
            targets = build_shared_boundary_targets(gt["sample_points_faces"], gt["sample_points_lines"],
                gt["edge_face_connectivity"], sample_id=sid, reciprocal_tolerance=.01,
                closed_tolerance=1e-5, allow_closed_phase=True)
        gt_indices = [index for index, owners in enumerate(targets["face_pairs"]) if tuple(owners) == pair]
        if len(gt_indices) != 1:
            raise ValueError("Expected exactly one GT boundary for the fixed pair")
        gt_index = gt_indices[0]
        pair_rows = [index for index, owners in enumerate(pairs) if tuple(sorted(owners)) == pair]
        assert len(pair_rows) == 2
        active_lookup = {int(row): edge_id for edge_id, row in enumerate(np.flatnonzero(active))}
        curves = []
        for row in pair_rows:
            curves.append({"candidate_row": row, "directed_pair": pairs[row].tolist(),
                "raw_edge_id": active_lookup.get(row), "active_before_and_after": bool(active[row]),
                "probability_before_and_after": float(probabilities[row]),
                "delta_norm": displacement_stats(delta[row]),
                "centroid_before": before[row].mean(0).tolist(), "centroid_after": after[row].mean(0).tolist(),
                "own_gt_cd_sq_before": squared_chamfer(before[row], targets["curves"][gt_index])[2],
                "own_gt_cd_sq_after": squared_chamfer(after[row], targets["curves"][gt_index])[2],
                "owner_distances_before": owner_distances(before[row], faces, pair),
                "owner_distances_after": owner_distances(after[row], faces, pair)})
        array_path = arrays_dir / f"{sid}.npz"
        if array_path.exists():
            raise FileExistsError(array_path)
        np.savez_compressed(array_path, latent=latent_array, pairs=pairs, probability=probabilities, active=active,
            curve_before=before, curve_after=after, global_xyz_delta=delta,
            local_curve_before=captured["before"][".local_points"].numpy(),
            local_curve_after=captured["after"][".local_points"].numpy(),
            curve_center_scale=captured["before"][".center_scale"].numpy(),
            face_points=faces, edge_features=captured["edge_features"].numpy(),
            face_features=captured["face_features"].numpy(),
            refiner_delta_face=enabled.delta_face.cpu().numpy(), refiner_delta_edge=enabled.delta_edge.cpu().numpy())
        sample = {"faces": len(faces), "candidate_count": len(pairs), "active_halfedge_count": int(active.sum()),
                  "input_latent_sha256": latent_sha256(latent_array), "exact_unchanged_tensor_checks": invariants,
                  "direction_channels_exact_unchanged": True, "shared_prefix_inputs_not_mutated": captured["input_checks"],
                  "captured_prefix_matches_final_state": hook_invariants,
                  "raw_export_replay": replay, "all_candidate_delta": displacement_stats(delta),
                  "active_candidate_delta": displacement_stats(delta[active]),
                  "selected_gt_pair": list(pair), "gt_source_edge_id": int(targets["source_edge_ids"][gt_index]),
                  "pair_is_exported_before_and_after": bool(active[pair_rows].all()),
                  "selected_pair_twin_cd_sq_before": squared_chamfer(before[pair_rows[0]], before[pair_rows[1]])[2],
                  "selected_pair_twin_cd_sq_after": squared_chamfer(after[pair_rows[0]], after[pair_rows[1]])[2],
                  "curves": curves, "arrays_path": str(array_path), "arrays_sha256": file_sha256(array_path)}
        result["samples"][sid] = sample
        print(json.dumps({"id": sid, "delta": sample["active_candidate_delta"], "curves": curves}), flush=True)
    for path, expected in sources.items():
        assert file_sha256(Path(path)) == expected, path
    assert file_sha256(checkpoint) == protocol["checkpoint_sha256"]
    result["status"] = "passed"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    lines = ["# 固定训练后网络：最终逐点反馈前后诊断", "",
             "只使用正式 best checkpoint 和两个预定 posterior latent；每件仅完整推理一次，以末端模块的共同输入/输出比较反馈前后，不重新训练。下表反映本次推理的末端修正，不能隔离该模块在整个微调过程中的作用。", "",
             "| 对象 | 有向面对 / 导出半边ID | 存在概率 | delta mean / max | 自身GT边CD² 前 → 后 | owner面距离mean 前 → 后 |",
             "|---|---|---:|---:|---:|---|"]
    for sid, sample in result["samples"].items():
        for curve in sample["curves"]:
            owners = "; ".join(f"面{owner}: {value['mean']:.7g} → {curve['owner_distances_after'][owner]['mean']:.7g}"
                               for owner, value in curve["owner_distances_before"].items())
            edge = curve['raw_edge_id'] if curve['raw_edge_id'] is not None else '未导出'
            lines.append(f"| {sid} | {curve['directed_pair']} / {edge} | {curve['probability_before_and_after']:.7g} | {curve['delta_norm']['mean']:.7g} / {curve['delta_norm']['max']:.7g} | {curve['own_gt_cd_sq_before']:.7g} → {curve['own_gt_cd_sq_after']:.7g} | {owners} |")
    lines += ["", "两个案例均使用同一次上游计算；hook确认所有模块输入未被原地修改，预测面/bbox/方向通道不变；此前捕获的classifier输出、face/edge feature及refiner delta与最终输出严格一致。因此前后候选和存在概率相同，只比较末端曲线坐标。原始导出逐字段复核误差见JSON，连通性完全一致。", "",
              "首次尝试分别完整解码启用/旁路版本，上游严格相等检查发现2.38e-7差异并停止。为避免把重复CUDA执行差异当成模块效应，本结果改用共同prefix，保持正式数值配置；首次检查保存在feedback_branch_initial_check.json，详见FEEDBACK_BRANCH_SPEC.md。", "",
              "delta单位为归一化global XYZ；GT边CD为双向平方距离之和。owner距离来自16点曲线至16×16预测面网格/450三角片，并非精确光滑CAD曲面误差。未导出的面对仍有候选几何，其测量不表示它参与最终B-Rep。", "",
              "完整max距离、全部/active候选位移分布、来源SHA与保存数组见 [feedback_branch_probe.json](feedback_branch_probe.json)。本诊断不改变主指标或失败标签。", ""]
    (OUT / "FEEDBACK_BRANCH_PROBE.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
