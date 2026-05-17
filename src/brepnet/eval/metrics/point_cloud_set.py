"""Point-cloud set metrics.

职责：在点云集合层面比较生成分布与参考分布，输出 MMD-CD、COV-CD、JSD。
这类指标需要先把 mesh/solid 采样为固定点数点云；采样本身是预处理，不算
metric。
"""

from __future__ import annotations

import multiprocessing
import os
import random
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
from chamfer_distance import ChamferDistance
from plyfile import PlyData
from scipy.stats import entropy
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


N_POINTS = 2000


def read_ply(path: str | Path) -> np.ndarray:
    with Path(path).open("rb") as f:
        plydata = PlyData.read(f)
        return np.stack(
            [
                np.array(plydata["vertex"]["x"]),
                np.array(plydata["vertex"]["y"]),
                np.array(plydata["vertex"]["z"]),
            ],
            axis=1,
        )


def downsample_pc(points: np.ndarray, n: int) -> np.ndarray:
    if points.shape[0] <= n:
        return points
    return points[random.sample(list(range(points.shape[0])), n)]


def normalize_pc(points: np.ndarray) -> np.ndarray:
    points = points - np.mean(points, axis=0)
    scale = np.max(np.abs(points))
    return points / scale if scale > 0 else points


def align_pc(points: np.ndarray) -> np.ndarray:
    centroid = np.mean(points, axis=0)
    centered_points = points - centroid
    dimensions = np.max(centered_points, axis=0) - np.min(centered_points, axis=0)
    axis_order = np.argsort(dimensions)[::-1]
    perm_matrix = np.zeros((3, 3))
    perm_matrix[0, axis_order[0]] = 1
    perm_matrix[1, axis_order[2]] = 1
    perm_matrix[2, axis_order[1]] = 1
    aligned_points = np.dot(centered_points, perm_matrix.T)
    if np.mean(aligned_points[:, 2]) < 0:
        aligned_points[:, 2] *= -1
    return aligned_points


def collect_pc(path: str | Path, *, align: bool = True) -> np.ndarray:
    pc = downsample_pc(read_ply(path), N_POINTS)
    pc = normalize_pc(pc)
    return align_pc(pc) if align else pc


def load_point_clouds(root: str | Path, suffix: str = ".ply") -> tuple[list[Path], np.ndarray]:
    paths = sorted(Path(root).rglob(f"*{suffix}"))
    num_cpus = multiprocessing.cpu_count()
    pcs = []
    with multiprocessing.Pool(num_cpus) as pool:
        for pc in tqdm(pool.imap(collect_pc, paths), total=len(paths), desc="Load point clouds"):
            if len(pc) > 0:
                pcs.append(pc)
    return paths, np.stack(pcs, axis=0)


def _pairwise_cd(sample_pcs: torch.Tensor, ref_pcs: torch.Tensor, batch_size: int) -> torch.Tensor:
    n_sample, n_ref = sample_pcs.shape[0], ref_pcs.shape[0]
    chamfer_dist = ChamferDistance()
    all_cd = []
    matched_gt = []
    for sample_idx in tqdm(range(n_sample), desc="Pairwise CD"):
        sample_batch = sample_pcs[sample_idx]
        cd_lst = []
        for ref_start in range(0, n_ref, batch_size):
            ref_batch = ref_pcs[ref_start : min(n_ref, ref_start + batch_size)]
            sample_exp = sample_batch.view(1, -1, 3).expand(ref_batch.size(0), -1, -1).contiguous()
            dl, dr, _, _ = chamfer_dist(sample_exp, ref_batch)
            cd_lst.append((dl.mean(dim=1) + dr.mean(dim=1)).view(1, -1))
        cd_lst = torch.cat(cd_lst, dim=1)
        all_cd.append(cd_lst)
        matched_gt.append(int(np.argmin(cd_lst.detach().cpu().numpy()[0])))
    return torch.cat(all_cd, dim=0)


def compute_cov_mmd(sample_pcs: torch.Tensor, ref_pcs: torch.Tensor, batch_size: int) -> tuple[dict[str, float], np.ndarray]:
    all_dist = _pairwise_cd(sample_pcs, ref_pcs, batch_size)
    _, min_idx = torch.min(all_dist, dim=1)
    min_val, _ = torch.min(all_dist, dim=0)
    return {
        "MMD-CD": float(min_val.mean().item()),
        "COV-CD": float(min_idx.unique().view(-1).size(0) / float(all_dist.size(1))),
    }, min_idx.cpu().numpy()


def unit_cube_grid_point_cloud(resolution: int, clip_sphere: bool = False):
    grid = np.ndarray((resolution, resolution, resolution, 3), np.float32)
    spacing = 2.0 / float(resolution - 1)
    for i in range(resolution):
        for j in range(resolution):
            for k in range(resolution):
                grid[i, j, k, 0] = i * spacing - 1.0
                grid[i, j, k, 1] = j * spacing - 1.0
                grid[i, j, k, 2] = k * spacing - 1.0
    if clip_sphere:
        grid = grid.reshape(-1, 3)
        grid = grid[np.linalg.norm(grid, axis=1) <= 0.5]
    return grid, spacing


def entropy_of_occupancy_grid(pclouds: np.ndarray, grid_resolution: int, in_sphere: bool = False):
    epsilon = 10e-4
    bound = 1 + epsilon
    if abs(np.max(pclouds)) > bound or abs(np.min(pclouds)) > bound:
        warnings.warn("Point-clouds are not in unit cube.")
    if in_sphere and np.max(np.sqrt(np.sum(pclouds**2, axis=2))) > bound:
        warnings.warn("Point-clouds are not in unit sphere.")

    grid_coordinates, _ = unit_cube_grid_point_cloud(grid_resolution, in_sphere)
    grid_coordinates = grid_coordinates.reshape(-1, 3)
    grid_counters = np.zeros(len(grid_coordinates))
    grid_bernoulli_rvars = np.zeros(len(grid_coordinates))
    nn = NearestNeighbors(n_neighbors=1).fit(grid_coordinates)

    for pc in pclouds:
        _, indices = nn.kneighbors(pc)
        for idx in np.squeeze(indices):
            grid_counters[idx] += 1
        for idx in np.unique(indices):
            grid_bernoulli_rvars[idx] += 1

    acc_entropy = 0.0
    n = float(len(pclouds))
    for g in grid_bernoulli_rvars:
        if g > 0:
            p = float(g) / n
            acc_entropy += entropy([p, 1.0 - p])
    return acc_entropy / len(grid_counters), grid_counters


def jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
    if np.any(p < 0) or np.any(q < 0):
        raise ValueError("Negative values.")
    if len(p) != len(q):
        raise ValueError("Non equal size.")
    p = p / np.sum(p)
    q = q / np.sum(q)
    return float(entropy((p + q) / 2.0, base=2) - (entropy(p, base=2) + entropy(q, base=2)) / 2.0)


def jsd_between_point_cloud_sets(sample_pcs: np.ndarray, ref_pcs: np.ndarray, in_unit_sphere: bool, resolution: int = 28) -> float:
    sample_grid_var = entropy_of_occupancy_grid(sample_pcs, resolution, in_unit_sphere)[1]
    ref_grid_var = entropy_of_occupancy_grid(ref_pcs, resolution, in_unit_sphere)[1]
    return jensen_shannon_divergence(sample_grid_var, ref_grid_var)


def evaluate_uniformity_nnd(points: np.ndarray) -> dict[str, Any]:
    diff = points[:, None, :] - points[None, :, :]
    distances = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(distances, np.inf)
    min_distances = np.min(distances, axis=1)
    density = len(points) / np.prod(np.max(points, axis=0) - np.min(points, axis=0))
    hist, bins = np.histogram(min_distances, bins="auto", density=True)
    return {
        "mean_nnd": float(np.mean(min_distances)),
        "std_nnd": float(np.std(min_distances)),
        "cv_nnd": float(np.std(min_distances) / np.mean(min_distances)),
        "min_nnd": float(np.min(min_distances)),
        "max_nnd": float(np.max(min_distances)),
        "density": float(density),
        "clark_evans_r": float(np.mean(min_distances) / (0.5 / np.sqrt(density))),
        "hist_values": hist,
        "hist_bins": bins,
    }


def evaluate_point_cloud_sets(
    fake_root: str | Path,
    real_root: str | Path,
    *,
    n_test: int = 1000,
    multi: float = 3.0,
    times: int = 10,
    batch_size: int = 64,
) -> dict[str, float]:
    random.seed(0)
    _, ref_pcs = load_point_clouds(real_root)
    _, sample_pcs = load_point_clouds(fake_root)
    result_list = []
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    for _ in range(times):
        sample_idx = random.sample(list(range(len(sample_pcs))), int(multi * n_test))
        ref_idx = random.sample(list(range(len(ref_pcs))), n_test)
        rand_sample_pcs = sample_pcs[sample_idx]
        rand_ref_pcs = ref_pcs[ref_idx]
        result = {"JSD": jsd_between_point_cloud_sets(rand_sample_pcs, rand_ref_pcs, in_unit_sphere=False)}
        with torch.no_grad():
            result_cd, _ = compute_cov_mmd(
                torch.tensor(rand_sample_pcs).to(device).float(),
                torch.tensor(rand_ref_pcs).to(device).float(),
                batch_size=batch_size,
            )
        result.update(result_cd)
        result_list.append(result)
    return {f"avg-{key}": float(np.mean([item[key] for item in result_list])) for key in result_list[0].keys()}
