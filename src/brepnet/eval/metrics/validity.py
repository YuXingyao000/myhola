"""Validity metric.

职责：检查后处理输出的 STEP 文件是否存在、是否是 OCC 认为有效的 solid，
并统计基础拓扑数量。这个指标只回答“生成物能不能作为 solid 打开和使用”，
不比较 GT 几何误差。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from OCC.Core.BRepCheck import BRepCheck_Analyzer
from OCC.Core.Interface import Interface_Static
from OCC.Core.STEPControl import STEPControl_AsIs, STEPControl_Writer
from OCC.Core.ShapeFix import ShapeFix_ShapeTolerance
from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_SOLID, TopAbs_VERTEX
from OCC.Extend.DataExchange import read_step_file

from shared.occ_utils import get_primitives
from src.brepnet.eval.protocol import EvalSample, first_existing_file
from src.brepnet.post.utils import get_tolerance


Interface_Static.SetIVal("read.precision.mode", 1)
Interface_Static.SetRVal("read.precision.val", 1e-1)
Interface_Static.SetIVal("write.precision.mode", 2)
Interface_Static.SetRVal("write.precision.val", 1e-1)


def save_step_file(step_file: str | Path, shape: Any) -> None:
    step_writer = STEPControl_Writer()
    step_writer.SetTolerance(get_tolerance(shape, TopAbs_SOLID))
    step_writer.Model(True)
    step_writer.Transfer(shape, STEPControl_AsIs)
    step_writer.Write(str(step_file))


def check_step_valid_solid(
    step_file: str | Path,
    precision: float = 1e-1,
    return_shape: bool = False,
) -> bool | tuple[bool, Any | None]:
    try:
        shape = read_step_file(str(step_file), as_compound=False, verbosity=False)
    except Exception:
        return (False, None) if return_shape else False
    if shape.ShapeType() != TopAbs_SOLID:
        return (False, shape) if return_shape else False
    ShapeFix_ShapeTolerance().SetTolerance(shape, precision)
    is_valid = BRepCheck_Analyzer(shape).IsValid()
    return (is_valid, shape) if return_shape else is_valid


def check_step_valid_soild(
    step_file: str | Path,
    precision: float = 1e-1,
    return_shape: bool = False,
) -> bool | tuple[bool, Any | None]:
    """Backward-compatible alias for the historical misspelling."""
    return check_step_valid_solid(step_file, precision=precision, return_shape=return_shape)


def load_data_with_prefix(
    root_folder: str | Path,
    prefix: str,
    folder_list_txt: str | Path | None = None,
) -> list[str]:
    root_folder = Path(root_folder)
    folder_list = None
    if folder_list_txt is not None:
        with Path(folder_list_txt).open("r", encoding="utf-8") as f:
            folder_list = set(f.read().splitlines())

    data_files: list[str] = []
    for root, _, files in os.walk(root_folder):
        if folder_list is not None and Path(root).name not in folder_list:
            continue
        for filename in files:
            if filename.endswith(prefix):
                data_files.append(str(Path(root) / filename))
    return sorted(data_files)


def evaluate_step_file(step_file: str | Path, success_marker: str | Path | None = None) -> dict[str, Any]:
    step_file = Path(step_file)
    has_step = step_file.exists()
    result: dict[str, Any] = {
        "has_step": has_step,
        "is_valid_solid": False,
        "has_success_marker": Path(success_marker).exists() if success_marker else False,
        "num_faces": 0,
        "num_edges": 0,
        "num_vertices": 0,
        "step_file": str(step_file),
    }
    if not has_step:
        return result

    is_valid, shape = check_step_valid_solid(step_file, return_shape=True)
    result["is_valid_solid"] = bool(is_valid)
    if shape is None:
        return result
    result["num_faces"] = len(get_primitives(shape, TopAbs_FACE, v_remove_half=True))
    result["num_edges"] = len(get_primitives(shape, TopAbs_EDGE, v_remove_half=True))
    result["num_vertices"] = len(get_primitives(shape, TopAbs_VERTEX, v_remove_half=True))
    return result


def evaluate_sample(sample: EvalSample) -> dict[str, Any]:
    step_file = first_existing_file(
        [
            sample.pred_dir / "recon_brep.step",
            *sorted(sample.pred_dir.glob("*.step")),
        ]
    )
    if step_file is None:
        step_file = sample.pred_dir / "recon_brep.step"
    return evaluate_step_file(step_file, success_marker=sample.pred_dir / "success.txt")
