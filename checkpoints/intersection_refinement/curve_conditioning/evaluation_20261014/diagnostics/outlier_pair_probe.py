"""Read-only two-case geometric probe; see OUTLIER_PAIR_SPEC.md."""
from __future__ import annotations

from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "diagnostics"
REPOSITORY = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPOSITORY))
from src.brepnet.data.shared_boundaries import build_shared_boundary_targets
from src.brepnet.models.blocks import closest_surface_points
from tools.vae_compare_metrics import squared_chamfer
from raw_distribution_analysis import digest, number

METHODS = ("intersection", "conditioned")
IDS = ("00936203", "00101718")


def surface_distances(curve, surfaces):
    points = torch.from_numpy(np.repeat(curve[None], len(surfaces), axis=0).astype(np.float64))
    with torch.inference_mode():
        _, distances = closest_surface_points(points, torch.from_numpy(surfaces.astype(np.float64)), chunk_size=2)
    return [{"sample_count": len(row), "mean": float(row.mean()), "max": float(row.max()),
             "values": row.tolist()} for row in distances.numpy()]


def read_raw(path):
    with np.load(path, allow_pickle=False) as z:
        raw = {key: z[key] for key in z.files}
    groups = defaultdict(list)
    for edge, a, b in raw["pred_edge_face_connectivity"]:
        if a == b or not 0 <= edge < len(raw["pred_edge"]):
            raise ValueError("Unexpected selected pair or edge index")
        groups[tuple(sorted((int(a), int(b))))].append((int(edge), int(a), int(b)))
    twins = {}
    for pair, rows in groups.items():
        if len(rows) != 2 or rows[0][1:] != rows[1][1:][::-1]:
            raise ValueError("Expected exactly two reciprocal rows")
        twins[pair] = squared_chamfer(raw["pred_edge"][rows[0][0]], raw["pred_edge"][rows[1][0]])[2]
    return raw, groups, twins


def main():
    torch.set_num_threads(1)
    outputs = [OUT / "outlier_pair_probe.json", OUT / "OUTLIER_PAIR_PROBE.md"]
    if any(path.exists() for path in outputs):
        raise FileExistsError("Probe already exists")
    source = ROOT / "raw_metrics/posterior_mean/per_object.csv"
    with source.open() as handle:
        csv_rows = {(row["method"], row["model_id"]): row for row in csv.DictReader(handle)}
    sources = {str(source): digest(source)}
    result = {"sample_ids": IDS, "source_sha256": sources, "samples": {}, "checks": [],
              "script_sha256": digest(Path(__file__)), "spec_sha256": digest(OUT / "OUTLIER_PAIR_SPEC.md"),
              "geometry_helper_sha256": {name: digest(REPOSITORY / name) for name in ("src/brepnet/models/blocks.py", "src/brepnet/data/shared_boundaries.py", "tools/vae_compare_metrics.py")},
              "geometry_contract": "16 curve samples; 16x16 surface grid / 450 triangles; global normalized XYZ, float64 CPU, existing squared_chamfer and closest_surface_points"}
    for item in IDS:
        gt_path = Path("/mnt/d/data/deepcad_v7") / item / "data.npz"
        sources[str(gt_path)] = digest(gt_path)
        with np.load(gt_path, allow_pickle=False) as gt:
            gt_faces = gt["sample_points_faces"][..., :3]
            targets = build_shared_boundary_targets(gt["sample_points_faces"], gt["sample_points_lines"], gt["edge_face_connectivity"], sample_id=item,
                                                    reciprocal_tolerance=.01, closed_tolerance=1e-5, allow_closed_phase=True)
        gt_lookup = {tuple(pair): index for index, pair in enumerate(targets["face_pairs"])}
        sample = {"num_faces": len(gt_faces), "gt_boundary_count": len(targets["curves"]), "methods": {}, "selected_pairs": []}
        loaded = {}
        for method in METHODS:
            path = ROOT / "raw_predictions/posterior_mean" / method / item / "data.npz"
            sources[str(path)] = digest(path)
            raw, groups, twins = read_raw(path)
            loaded[method] = (raw, groups, twins)
            worst = max(twins, key=twins.get)
            observed = csv_rows[method, item]
            mean, maximum = float(np.mean(list(twins.values()))), max(twins.values())
            for key, actual in (("legacy_reciprocal_cd_sq_mean", mean), ("legacy_reciprocal_cd_sq_max", maximum)):
                if not math.isclose(actual, float(observed[key]), rel_tol=1e-10, abs_tol=1e-14):
                    raise ValueError(f"CSV mismatch {item}/{method}/{key}")
                result["checks"].append({"id": item, "method": method, "metric": key, "passed": True})
            sample["methods"][method] = {"reciprocal_mean": mean, "reciprocal_max": maximum, "worst_pair": list(worst),
                                          "raw_edge_count": len(raw["pred_edge"]), "reciprocal_pair_count": len(twins),
                                          "merged_all_gt_edge_cd_sq": float(observed["edge_cd_sq"])}
            sample["selected_pairs"].append(worst)
        if str(loaded[METHODS[0]][0]["input_latent_sha256"]) != str(loaded[METHODS[1]][0]["input_latent_sha256"]):
            raise ValueError("Latent inputs differ")
        selected = sorted(set(sample.pop("selected_pairs")))
        sample["pairs"] = []
        for pair in selected:
            index = gt_lookup.get(pair)
            record = {"face_pair": list(pair), "gt_adjacent": index is not None,
                      "gt_boundary_index": index, "gt_source_edge_id": int(targets["source_edge_ids"][index]) if index is not None else None,
                      "methods": {}}
            for method, (raw, groups, twins) in loaded.items():
                if pair not in groups:
                    record["methods"][method] = {"status": "pair_not_predicted"}
                    continue
                curves = []
                for edge, a, b in groups[pair]:
                    curve = raw["pred_edge"][edge]
                    scores = [squared_chamfer(curve, reference)[2] for reference in targets["curves"]]
                    nearest = int(np.argmin(scores))
                    curves.append({"edge_id": edge, "directed_pair": [a, b], "sample_count": len(curve),
                                   "own_pair_gt_cd_sq": float(scores[index]) if index is not None else None,
                                   "nearest_gt_boundary_index": nearest, "nearest_gt_face_pair": targets["face_pairs"][nearest].tolist(),
                                   "nearest_gt_source_edge_id": int(targets["source_edge_ids"][nearest]), "nearest_gt_boundary_cd_sq": float(scores[nearest]),
                                   "curve_to_merged_gt_mean_sq": squared_chamfer(curve, targets["curves"].reshape(-1, 3))[0],
                                   "owner_predicted_surface_distance": dict(zip(map(str, pair), surface_distances(curve, raw["pred_face"][list(pair)]))),
                                   "owner_gt_surface_distance": dict(zip(map(str, pair), surface_distances(curve, gt_faces[list(pair)]))),
                                   "centroid": curve.mean(axis=0).tolist()})
                record["methods"][method] = {"status": "ok", "twin_cd_sq": twins[pair], "curves": curves}
            sample["pairs"].append(record)
        result["samples"][item] = sample
    for path, expected in sources.items():
        if digest(Path(path)) != expected:
            raise ValueError("Input changed during case probe")
    lines = ["# 两个互反差距极端案例：具体 face pair 核验", "",
             "只读两件 raw 与 GT，使用原 CD 和投影 helper。每条曲线16点，所属面各16×16网格/450三角片；投影距离为该离散面片的欧氏距离，不是精确CAD曲面误差。下列编号均为NPZ内零起始索引。", ""]
    for item, sample in result["samples"].items():
        lines += [f"## {item}", ""]
        for method, summary in sample["methods"].items():
            lines.append(f"{method}：整体边对GT平方CD={number(summary['merged_all_gt_edge_cd_sq'])}；互反mean={number(summary['reciprocal_mean'])}、max={number(summary['reciprocal_max'])}，最差面对={summary['worst_pair']}。")
        for pair in sample["pairs"]:
            lines += ["", f"面对 {pair['face_pair']}：GT邻接存在={pair['gt_adjacent']}；GT代表edge ID={pair['gt_source_edge_id']}。", "",
                      "| 方法 | 有向pair / edge ID | Twin CD² | 自身pair GT CD² | 最近GT pair / CD² | 到所属预测面：mean / max | 到所属GT面：mean / max |", "|---|---|---:|---:|---|---|---|"]
            for method, data in pair["methods"].items():
                if data["status"] != "ok":
                    lines.append(f"| {method} | 未选中 | — | — | — | — | — |")
                    continue
                for curve in data["curves"]:
                    def distances(name):
                        return "; ".join(f"面{k}: {number(v['mean'])}/{number(v['max'])}" for k, v in curve[name].items())
                    lines.append(f"| {method} | {curve['directed_pair']} / {curve['edge_id']} | {number(data['twin_cd_sq'])} | {number(curve['own_pair_gt_cd_sq'])} | {curve['nearest_gt_face_pair']} / {number(curve['nearest_gt_boundary_cd_sq'])} | {distances('owner_predicted_surface_distance')} | {distances('owner_gt_surface_distance')} |")
    lines += ["", "**解释范围**：整体边 CD 将所有曲线点合并后寻找任一GT边最近点，因此不会要求某face-pair曲线匹配该pair的GT边，也不要求互反曲线重合。最近另一GT边只是几何观测，不代表网络内部发生了显式重分配。自身pair误差、最近任意GT边误差和所属面距离必须分开看。", "",
              "两件全部互反mean/max逐项复核与已发布CSV一致；本诊断不改变主指标、失败标签或权重选择，不归因于有序条件拼接。完整逐点距离、来源SHA与原值见 [outlier_pair_probe.json](outlier_pair_probe.json)。", ""]
    values = (json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", "\n".join(lines))
    for path, text in zip(outputs, values):
        with path.open("x") as handle:
            handle.write(text)
    print(json.dumps({"files": [str(path) for path in outputs], "checks": len(result["checks"])}, indent=2))


if __name__ == "__main__":
    main()
