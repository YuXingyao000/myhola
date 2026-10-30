"""将配对渲染差值转移到增强空场景；这是 display-linear 近似。

sRGB EOTF 不会撤销已烘焙的 AgX，因此本模块不声称恢复了 scene-linear
辐射值。配对图必须同相机、同色管；增强改变局部几何/光照时仍需拒绝。
"""

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt


def srgb_to_display_linear(rgb):
    """输入/输出均为 [0,1] 浮点值；输出不是原始场景辐射。"""
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def display_linear_to_rgb(values):
    """最终显示量化；调用前另行记录越界值，不能用裁剪掩盖问题。"""
    values = np.clip(values, 0, 1)
    encoded = np.where(values <= 0.0031308, 12.92 * values, 1.055 * values ** (1 / 2.4) - 0.055)
    return np.rint(encoded * 255).astype(np.uint8)


def transfer_delta(with_part, empty, enhanced, mode):
    """浮点合成核心，保留越界值。两种公式在 enhanced=empty 时还原 with_part。"""
    delta = with_part - empty
    if mode == "additive":
        return enhanced + delta
    if mode == "relative_darkening":
        negative = np.minimum(delta, 0)
        positive = np.maximum(delta, 0)
        ratio = 1 + negative / np.maximum(empty, 1e-4)
        # near-black 分母保护会改变 ratio*empty；补偿该误差以保留恒等性。
        correction = negative - empty * (ratio - 1)
        return ratio * enhanced + positive + correction
    raise ValueError(f"Unknown compositing mode: {mode}")


def _edge_signals(empty_rgb, enhanced_rgb, target):
    """强边缘位移定位信号。纹理/光照也会改边缘，不能据此判定几何正确。"""
    source = cv2.Canny(cv2.cvtColor(empty_rgb, cv2.COLOR_RGB2GRAY), 40, 100) > 0
    edited = cv2.Canny(cv2.cvtColor(enhanced_rgb, cv2.COLOR_RGB2GRAY), 40, 100) > 0
    distance_to_source = distance_transform_edt(~source) if source.any() else np.full(source.shape, np.inf)
    distance_to_edited = distance_transform_edt(~edited) if edited.any() else np.full(edited.shape, np.inf)
    rows, columns = np.nonzero(target)
    near = np.zeros_like(target)
    near[max(0, rows.min() - 64):rows.max() + 65, max(0, columns.min() - 64):columns.max() + 65] = True

    metrics = {}
    for name, region in (("full_frame", np.ones_like(target)), ("target_neighborhood", near)):
        before, after = source & region, edited & region
        metrics[name] = {
            "source_edge_pixels": int(before.sum()),
            "edited_edge_pixels": int(after.sum()),
            "source_supported_within_2px": float((distance_to_edited[before] <= 2).mean()) if before.any() else None,
            "edited_unmatched_beyond_2px": float((distance_to_source[after] > 2).mean()) if after.any() else None,
        }
    overlay = np.zeros((*target.shape, 3), dtype=np.uint8)
    overlay[source, 0] = 255
    overlay[edited, 1:] = 255
    return metrics, Image.fromarray(overlay)


def compose(with_part_rgb, empty_rgb, enhanced_empty, mask, mode):
    """返回 protected PIL、指标、诊断图/数组；mode 为 additive/relative_darkening。

    三张彩色输入必须为 8 位 RGB PIL 图像，不接受隐式颜色/精度转换。
    主输出仅把目标 mask+2 px 回贴为原图；孔隙可显示新背景。hull_protected
    是额外诊断，不是默认训练输出。raw 不回贴，用于查看边缘与合成变化。
    """
    if any(image.mode != "RGB" for image in (with_part_rgb, empty_rgb, enhanced_empty)):
        raise ValueError("配对图和增强图必须为 8 位 RGB PIL 图像")
    if not with_part_rgb.size == empty_rgb.size == enhanced_empty.size == mask.size:
        raise ValueError("配对图、增强图和 mask 必须同尺寸")
    mask_values = np.asarray(mask.convert("L"))
    target = mask_values > 0
    if not np.isin(mask_values, [0, 255]).all() or not target.any() or target.all():
        raise ValueError("CAD mask 必须为非空且不占满图像的二值图")

    original = np.asarray(with_part_rgb)
    background = np.asarray(empty_rgb)
    edited = np.asarray(enhanced_empty)
    i, b, e = [srgb_to_display_linear(rgb.astype(np.float64) / 255) for rgb in (original, background, edited)]
    candidate = transfer_delta(i, b, e, mode)
    if not np.isfinite(candidate).all():
        raise ValueError("合成结果包含非有限数值")
    raw = display_linear_to_rgb(candidate)

    protected = distance_transform_edt(~target) <= 2
    output = raw.copy()
    output[protected] = original[protected]

    # 只供诊断：恢复凸包内背景可保留孔隙内容，也会重新引入旧纹理岛。
    rows, columns = np.nonzero(target)
    points = np.column_stack((columns, rows)).astype(np.int32)
    hull = np.zeros_like(mask_values)
    cv2.fillConvexPoly(hull, cv2.convexHull(points), 1)
    hull = distance_transform_edt(hull == 0) <= 2
    hull_output = raw.copy()
    hull_output[hull] = original[hull]

    delta = i - b
    absolute_delta = np.abs(delta).max(axis=-1)
    # 固定显示范围 0..0.1，CAD+2px 在热图中留黑，以便看外部阴影/反射。
    heat = np.clip(absolute_delta / 0.1, 0, 1)
    heat[protected] = 0
    edge_metrics, edge_overlay = _edge_signals(background, edited, target)
    identity = transfer_delta(i, b, b, mode)
    identity_rgb = display_linear_to_rgb(identity)
    outside_delta = absolute_delta[~protected]
    metrics = {
        "mode": mode,
        "working_domain": "display-linear sRGB EOTF; AgX remains baked; not scene-linear",
        "before_clip_min": float(candidate.min()),
        "before_clip_max": float(candidate.max()),
        "negative_channel_fraction": float((candidate < 0).mean()),
        "above_one_channel_fraction": float((candidate > 1).mean()),
        "identity_max_abs_float_error": float(np.abs(identity - i).max()),
        "identity_max_abs_uint8_error": int(np.abs(identity_rgb.astype(int) - original.astype(int)).max()),
        "target_max_abs_uint8_error": int(np.abs(output[target].astype(int) - original[target].astype(int)).max()),
        "protected_max_abs_uint8_error": int(np.abs(output[protected].astype(int) - original[protected].astype(int)).max()),
        "protected_pixels": int(protected.sum()),
        "protected_source": "with_part_rgb, target binary mask dilated by Euclidean radius 2px",
        "outside_target_delta": {
            "mean": float(outside_delta.mean()), "p50": float(np.quantile(outside_delta, .5)),
            "p95": float(np.quantile(outside_delta, .95)), "max": float(outside_delta.max()),
        },
        "delta_heatmap_range": [0, 0.1],
        "alignment_signals_not_a_gate": edge_metrics,
    }
    diagnostics = {
        "raw": Image.fromarray(raw),
        "hull_protected": Image.fromarray(hull_output),
        "protected_mask": Image.fromarray(protected.astype(np.uint8) * 255),
        "delta_heatmap": Image.fromarray(np.rint(heat * 255).astype(np.uint8)),
        "edge_alignment": edge_overlay,
        "delta_abs_display": absolute_delta.astype(np.float32),
    }
    return Image.fromarray(output), metrics, diagnostics
