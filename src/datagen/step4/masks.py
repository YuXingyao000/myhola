"""零件保护、背景编辑与最终合成；白色编辑 mask 表示允许生成的像素。"""

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy.ndimage import distance_transform_edt
from scipy.spatial import ConvexHull


def read_inputs(rgb_file, mask_file):
    with Image.open(rgb_file) as image:
        rgb = image.convert("RGB")
    with Image.open(mask_file) as image:
        mask = image.convert("L")
    if rgb.size != (512, 512) or mask.size != rgb.size:
        raise ValueError("RGB 与 mask 必须同为 512×512")
    values = np.asarray(mask)
    if not np.isin(values, [0, 255]).all() or not 0 < np.count_nonzero(values) < values.size:
        raise ValueError("零件 mask 必须为非空且不占满画面的二值图")
    return rgb, mask


def make_masks(object_mask, padding_px, feather_px, region="silhouette"):
    """先扩张保护区。采样使用硬 mask，合成仅在保护区外侧渐变。

    原始 Object Index 没有抗锯齿，外扩少量像素可保护 RGB 的混合边缘。
    不向零件内部羽化；否则最终输出也会改变零件像素。
    """
    if region == "convex_hull":
        # 一起保护孔洞、拱形内部和外轮廓之间的空隙；这些虽是背景像素，
        # 仍承载零件的结构证据。保留原背景是代价，需看合成边缘是否明显。
        points = np.argwhere(np.asarray(object_mask) > 0)[:, ::-1]
        hull = points[ConvexHull(points).vertices]
        object_mask = Image.new("L", object_mask.size, 0)
        ImageDraw.Draw(object_mask).polygon([tuple(point) for point in hull], fill=255)
    protected = object_mask.filter(ImageFilter.MaxFilter(2 * padding_px + 1))
    outside = np.asarray(protected) == 0
    if not outside.any():
        raise ValueError("扩张保护区后没有可增强的背景")
    edit_mask = Image.fromarray(outside.astype(np.uint8) * 255)
    weight = np.clip(distance_transform_edt(outside) / feather_px, 0, 1) if feather_px else outside
    blend_mask = Image.fromarray(np.rint(weight * 255).astype(np.uint8))
    return protected, edit_mask, blend_mask


def composite_and_check(original, generated, protected, blend_mask):
    """保存模型原始输出之外，再合成一份原始零件像素严格不变的训练图。"""
    if generated.size != original.size:
        raise ValueError(f"模型改变了输出尺寸：{generated.size}")
    result = Image.composite(generated.convert("RGB"), original, blend_mask)
    difference = np.abs(np.asarray(result).astype(np.int16) - np.asarray(original).astype(np.int16))
    protected_pixels = np.asarray(protected) > 0
    max_error = int(difference[protected_pixels].max())
    if max_error != 0:
        raise ValueError(f"保护区域发生变化：{max_error}")
    return result, {
        "protected_max_abs_difference": max_error,
        "protected_pixels": int(protected_pixels.sum()),
        "background_mean_abs_difference": float(difference[~protected_pixels].mean()),
        "background_changed_fraction": float((difference[~protected_pixels].max(axis=-1) > 0).mean()),
        "pass": True,
    }
