"""Light Field Distance (LFD) metric wrapper.

职责：把外部 LFD 工具链的 pickle 结果整理成统一 summary 和可视化图。
真正的 LFD feature 提取和矩阵计算仍保留在 `eval/lfd/evaluation_scripts/`，
该目录来自外部实现，不在普通 Python metric 中重写。
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np

from src.brepnet.eval.metrics.validity import check_step_valid_soild


def remove_outliers_zscore(data: np.ndarray, threshold: float = 3) -> tuple[np.ndarray, np.ndarray]:
    if len(data) == 0:
        return data, np.zeros(0, dtype=bool)
    std = np.std(data)
    if std == 0:
        return data, np.ones_like(data, dtype=bool)
    z_scores = np.abs((data - np.mean(data)) / std)
    mask = z_scores < threshold
    return data[mask], mask


def load_lfd_pickle(path: str | Path) -> tuple[list[str], list[str], np.ndarray]:
    src_folder_list, nearest_name, lfd_matrix = pickle.load(Path(path).open("rb"))
    return list(src_folder_list), list(nearest_name), np.asarray(lfd_matrix)


def summarize_lfd_pickle(
    pkl_path: str | Path,
    *,
    output_png: str | Path | None = None,
    src_step_root: str | Path | None = None,
    sample_size: int = 1000,
    repeats: int = 10,
    seed: int = 0,
) -> dict[str, Any]:
    src_folder_list, nearest_name, lfd_matrix = load_lfd_pickle(pkl_path)
    if src_step_root is not None:
        src_step_root = Path(src_step_root)
        valid_mask = np.array(
            [check_step_valid_soild(src_step_root / folder / "recon_brep.step") for folder in src_folder_list],
            dtype=bool,
        )
        src_folder_list = [src_folder_list[i] for i in range(len(src_folder_list)) if valid_mask[i]]
        nearest_name = [nearest_name[i] for i in range(len(nearest_name)) if valid_mask[i]]
        lfd_matrix = lfd_matrix[valid_mask]

    nearest_lfd = lfd_matrix.min(axis=1)
    rng = np.random.default_rng(seed)
    mean_list, median_list, p75_list = [], [], []
    for _ in range(repeats):
        local_size = min(sample_size, len(nearest_lfd))
        sampled = rng.choice(nearest_lfd, local_size, replace=False if local_size == len(nearest_lfd) else True)
        sampled, _ = remove_outliers_zscore(sampled, threshold=3)
        mean_list.append(float(np.mean(sampled)))
        median_list.append(float(np.median(sampled)))
        p75_list.append(float(np.percentile(sampled, 75)))

    if output_png is not None:
        import matplotlib.pyplot as plt

        output_png = Path(output_png)
        output_png.parent.mkdir(parents=True, exist_ok=True)
        data, _ = remove_outliers_zscore(nearest_lfd, threshold=3)
        hist, bin_edges = np.histogram(data, bins=45, range=(0, 4500))
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.set_xlim(0, 100)
        ax.barh(bin_edges[:-1], hist, height=50)
        ax.set_title("Light Field Distance (LFD) Distribution")
        ax.set_xlabel("Frequency")
        ax.set_ylabel("Light Field Distance (LFD)")
        fig.savefig(output_png, dpi=600)
        plt.close(fig)

    return {
        "num_shapes": len(src_folder_list),
        "mean_lfd": float(np.mean(nearest_lfd)) if len(nearest_lfd) else float("nan"),
        "median_lfd": float(np.median(nearest_lfd)) if len(nearest_lfd) else float("nan"),
        "p75_lfd": float(np.percentile(nearest_lfd, 75)) if len(nearest_lfd) else float("nan"),
        "mean_lfd_resampled": float(np.mean(mean_list)) if mean_list else float("nan"),
        "median_lfd_resampled": float(np.mean(median_list)) if median_list else float("nan"),
        "p75_lfd_resampled": float(np.mean(p75_list)) if p75_list else float("nan"),
        "nearest_name": nearest_name,
        "src_folder": src_folder_list,
    }
