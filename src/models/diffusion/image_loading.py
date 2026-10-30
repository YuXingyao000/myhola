"""读取 NPZ 中的图像；可选只解压 C-order RGB 数组的首视图。"""

from __future__ import annotations

import math
from pathlib import Path
import warnings
import zipfile

import numpy as np


def _fallback(reason: str) -> None:
    warnings.warn(
        f"Single-view NPZ reader falls back to np.load: {reason}",
        RuntimeWarning,
        stacklevel=3,
    )


def _read_first_image(path: Path, key: str) -> np.ndarray | None:
    """支持 NPY v1/v2 的 uint8 [V,H,W,3]；其他布局明确回退。"""
    member_name = f"{key}.npy"
    with zipfile.ZipFile(path) as archive:
        if member_name not in archive.namelist():
            _fallback("the array is not stored as a .npy member")
            return None
        info = archive.getinfo(member_name)
        if info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
            _fallback("unsupported ZIP compression method")
            return None

        with archive.open(info) as member:
            version = np.lib.format.read_magic(member)
            if version == (1, 0):
                shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(member)
            elif version == (2, 0):
                shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(member)
            else:
                _fallback(f"unsupported NPY version {version}")
                return None

            if fortran_order or dtype != np.dtype(np.uint8):
                _fallback("expected C-order uint8 images")
                return None
            if len(shape) != 4 or shape[-1] != 3 or any(size <= 0 for size in shape):
                _fallback("expected a nonempty [views, height, width, 3] array")
                return None

            image_bytes = math.prod(shape[1:]) * dtype.itemsize
            expected_size = member.tell() + shape[0] * image_bytes
            if info.file_size < expected_size:
                raise ValueError(f"Truncated image array in {path}:{member_name}")
            content = member.read(image_bytes)
            if len(content) != image_bytes:
                raise ValueError(f"Incomplete first image in {path}:{member_name}")

    # 独立的可写数组，不让单张图持有完整 24-view 数组的底层内存。
    return np.frombuffer(content, dtype=dtype).reshape(shape[1:]).copy()


def load_image_view(
    path: str | Path,
    key: str,
    rotation_id: int,
    *,
    read_single_view: bool = False,
) -> np.ndarray:
    """保持像素不变地读取一个视图，不做缩放、归一化或随机采样。

    仅视图 0 启用局部解压；后续视图的压缩流 seek 可能比整数组读取更慢。
    快速路径不扫描未读取的尾部，因而不保证整个 ZIP member 的 CRC 校验；
    文件头、尺寸及实际读取错误直接抛出，完整校验可关闭此选项。
    """
    path = Path(path)
    if read_single_view and rotation_id == 0:
        image = _read_first_image(path, key)
        if image is not None:
            return image
    with np.load(path) as images:
        return images[key][rotation_id]
