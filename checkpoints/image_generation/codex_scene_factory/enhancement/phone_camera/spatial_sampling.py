"""背景先编辑，CAD 面内部后释放；孔隙、边界和周边支撑区域持续保留。

这是固定单图、8× VAE、2×2 latent packing 的研究模块。最终 strict 与
material 两种合成必须分开保存：后者允许材质变化，没有几何正确性保证。
"""

from math import ceil

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt


def _image(values):
    return Image.fromarray(np.rint(np.asarray(values) * 255).astype(np.uint8))


def _whole_latent_cells(pixels):
    """每个 8×8 像素块必须全部可编辑，避免最近邻缩放漏过细边界。"""
    height, width = pixels.shape
    cells = pixels.reshape(height // 8, 8, width // 8, 8).all(axis=(1, 3))
    aligned_pixels = cells.repeat(8, axis=0).repeat(8, axis=1)
    return cells, aligned_pixels


def build_spatial_plan(original, mask, config):
    """返回初始采样 mask、两种最终合成 mask、可审阅摘要及内部网格。

    白色表示可编辑/使用生成图；strict_protected_mask 的白色则表示
    必须恢复原图。Canny 只用于 CAD 内部的保守边缘代理，不当作几何 GT。
    """
    target = np.asarray(mask.convert("L")) > 0
    height, width = target.shape
    if original.size != mask.size or height % 16 or width % 16:
        raise ValueError("RGB/mask 必须同尺寸，且边长能被 16 整除")
    if not target.any() or target.all():
        raise ValueError("目标 mask 必须非空且不占满画面")

    rows, columns = np.nonzero(target)
    bbox_side = max(int(columns.max() - columns.min() + 1), int(rows.max() - rows.min() + 1))
    halo = max(config.get("halo_min_px", 24), ceil(config.get("halo_fraction", 0.4) * bbox_side))
    points = np.column_stack((columns, rows)).astype(np.int32)
    hull = np.zeros_like(target, dtype=np.uint8)
    cv2.fillConvexPoly(hull, cv2.convexHull(points), 1)
    protected = distance_transform_edt(hull == 0) <= halo
    background_cells, background = _whole_latent_cells(~protected)
    if not background.any():
        raise ValueError("扩大保护区后没有可编辑背景")

    # 原图 CAD 没有复杂贴图，内部强边缘可作为首轮保守代理。它仍会
    # 同时包含部分高光/阴影，且可能漏掉低对比孔槽；必须保留这个限制。
    gray = np.asarray(original.convert("L"))
    edges = cv2.Canny(gray, config.get("canny_low", 40), config.get("canny_high", 100)) > 0
    edges &= target
    away_from_edge = (
        distance_transform_edt(~edges) > config.get("cad_edge_px", 3)
        if edges.any() else np.ones_like(target)
    )
    interior = (distance_transform_edt(target) > config.get("boundary_px", 4)) & away_from_edge
    interior_cells, interior = _whole_latent_cells(interior)

    strict_blend = np.clip(distance_transform_edt(background) / config.get("feather_px", 16), 0, 1)
    material_weight = np.clip(
        distance_transform_edt(interior) / config.get("interior_feather_px", 4), 0, 1,
    )
    material_blend = np.maximum(strict_blend, material_weight)
    release_sigma = float(config.get("interior_release_sigma", 0.3))
    release_map = np.zeros_like(gray, dtype=np.float32)
    release_map[background] = 1.0
    release_map[interior] = release_sigma

    summary = {
        "resolution": [width, height],
        "bbox_max_side_px": bbox_side,
        "halo_px": halo,
        "target_pixels": int(target.sum()),
        "protected_pixels": int((~background).sum()),
        "background_pixels": int(background.sum()),
        "cad_interior_pixels": int(interior.sum()),
        "cad_interior_fraction": float(interior.sum() / target.sum()),
        "cad_interior_latent_cells": int(interior_cells.sum()),
        "release_sigma": release_sigma,
        "material_edit_available": bool(interior.any()),
        "latent_cell_px": 8,
        "edge_source": "Canny RGB inside CAD; proxy, not geometric truth",
    }
    return {
        "initial_edit_mask": _image(background),
        "strict_protected_mask": _image(~background),
        "strict_blend_mask": _image(strict_blend),
        "material_blend_mask": _image(material_blend),
        "cad_interior_mask": _image(interior),
        "internal_edges_mask": _image(edges),
        "release_map": _image(release_map),
        "release_sigma": release_sigma,
        "background_cells": background_cells,
        "interior_cells": interior_cells,
        "summary": summary,
    }


def make_callback(pipe, plan):
    """返回 callback 与逐步日志；传入 callback_tensor_inputs=['mask']。

    官方管线在 callback 前已完成当前步的原图 latent 回填。因此此处
    更新的 mask 只作用于下一步，不会逆向开放刚完成的步骤。scheduler
    step_index 此时指向下一次去噪输入的 sigma，不能使用截断前的索引 i。
    """
    import torch

    log = []
    released_mask = None

    def callback(pipeline, step_index, timestep, tensors):
        nonlocal released_mask
        current_mask = tensors["mask"]
        if released_mask is None:
            cells = plan["background_cells"] | plan["interior_cells"]
            cells = torch.as_tensor(cells, device=current_mask.device, dtype=current_mask.dtype)
            # packed 最后一维是 channels × 2 × 2；保持官方通道/子块顺序。
            channels = current_mask.shape[-1] // 4
            grid = cells[None, None].repeat(1, channels, 1, 1)
            released_mask = pipe._pack_latents(grid, 1, channels, cells.shape[0], cells.shape[1])

        scheduler_index = pipeline.scheduler.step_index
        next_sigma = float(pipeline.scheduler.sigmas[scheduler_index])
        has_next_step = step_index + 1 < pipeline.num_timesteps
        release_next = has_next_step and next_sigma <= plan["release_sigma"]
        log.append({
            "step": int(step_index),
            "current_timestep": float(timestep),
            "next_sigma": next_sigma,
            "cad_open_this_step": bool(
                log and log[-1]["cad_open_next_step"]
            ),
            "cad_open_next_step": bool(release_next and plan["summary"]["material_edit_available"]),
        })
        return {"mask": released_mask if release_next else current_mask}

    return callback, log
