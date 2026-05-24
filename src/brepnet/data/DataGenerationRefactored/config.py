from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path("/mnt/d/python/src/brepnet/data/DataGenerationRefactored")
OLD_PROJECT_ROOT = Path("/mnt/d/python/src/brepnet/data/DataGeneration")


@dataclass(frozen=True)
class Paths:
    project_root: Path = PROJECT_ROOT
    old_project_root: Path = OLD_PROJECT_ROOT

    list_dir: Path = OLD_PROJECT_ROOT / "list"
    test_list_dir: Path = PROJECT_ROOT / "test_list"
    render_list_dir: Path = PROJECT_ROOT / "render_lists"
    machine_list_dir: Path = PROJECT_ROOT / "machine_lists"

    model_root: Path = Path("/mnt/d/data/deepcad_v6")
    mesh_name: str = "mesh.stl"

    blender_bin: Path = OLD_PROJECT_ROOT / "blender-4.2.0-linux-x64" / "blender"
    materials_root: Path = PROJECT_ROOT / "materials"

    blender_single_view_out: Path = PROJECT_ROOT / "outputs" / "blender_single_view"
    blender_single_view_metal010_out: Path = PROJECT_ROOT / "outputs" / "blender_single_view_metal010"
    blender_cube24_out: Path = PROJECT_ROOT / "outputs" / "blender_cube24"
    flux_single_view_out: Path = PROJECT_ROOT / "outputs" / "flux_single_view"
    flux_cube24_out: Path = PROJECT_ROOT / "outputs" / "flux_cube24_dynamic"

    cond_single_view_target: Path = Path("/mnt/d/data/deepcad_v6_cond")
    cond_cube24_target: Path = Path("/mnt/d/data/deepcad_v6_cond_cube24")

    logs_root: Path = PROJECT_ROOT / "logs"

    transformer_path: Path = Path("/mnt/d/model/Flux1_Kontext_dev_GGUF/flux1-kontext-dev-Q8_0.gguf")
    base_model_path: Path = Path("/mnt/d/model/Flux1_Kontext_dev")


@dataclass(frozen=True)
class Runtime:
    python: str = "python3"
    num_gpus: int = 8
    list_seed: int = 42

    samples: int = 32
    png_compression: int = 0
    blender_device: str = "GPU"
    require_gpu: bool = True
    skip_existing: bool = True

    flux_seed: int = 42
    num_steps: int = 40
    guidance_scale: float = 3.5
    true_cfg_scale: float = 2.0

    pack_workers: int = 64
    pack_overwrite: bool = False


@dataclass(frozen=True)
class ListSampling:
    num_train: int = 750
    num_val: int = 125
    num_test: int = 125


PATHS = Paths()
RUNTIME = Runtime()
LIST_SAMPLING = ListSampling()
