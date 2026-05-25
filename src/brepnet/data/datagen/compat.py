"""Legacy compatibility shim for run_lists.py.

run_lists.py imports PATHS, RUNTIME, LIST_SAMPLING from .config — this module
provides those symbols so that script continues to work without changes.

All NEW code should use Hydra configs (configs/datagen/) instead.
"""
from dataclasses import dataclass
from pathlib import Path

from .config import DATAGEN_ROOT, PROJECT_ROOT


@dataclass(frozen=True)
class _LegacyPaths:
    """Minimal path set used only by run_lists.py and split_model_lists.py."""
    project_root: Path = DATAGEN_ROOT
    list_dir: Path = PROJECT_ROOT / "src" / "brepnet" / "data" / "list"
    test_list_dir: Path = DATAGEN_ROOT / "test_list"
    render_list_dir: Path = DATAGEN_ROOT / "render_lists"
    machine_list_dir: Path = DATAGEN_ROOT / "machine_lists"
    model_root: Path = Path("/mnt/d/data/deepcad_v6")
    mesh_name: str = "mesh.stl"
    materials_root: Path = DATAGEN_ROOT / "materials"
    logs_root: Path = DATAGEN_ROOT / "logs"


@dataclass(frozen=True)
class _LegacyRuntime:
    python: str = "python3"
    num_gpus: int = 8
    list_seed: int = 42


@dataclass(frozen=True)
class _LegacyListSampling:
    num_train: int = 750
    num_val: int = 125
    num_test: int = 125


PATHS = _LegacyPaths()
RUNTIME = _LegacyRuntime()
LIST_SAMPLING = _LegacyListSampling()
