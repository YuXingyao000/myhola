"""Single-threaded CPU setup for process-isolated post-processing workers."""
from __future__ import annotations

import os


def configure_cpu() -> None:
    # Set before importing numerical libraries, and before any child is forked.
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


def preload_cpu_dependencies() -> None:
    configure_cpu()
    import torch
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    # Import once in the single-threaded parent: every job still gets its own
    # process and OCC state, without paying thousands of heavyweight imports.
    from src.postprocess.construct_brep import construct_brep_for_sample  # noqa: F401
    from scipy.spatial import cKDTree  # noqa: F401
    import trimesh  # noqa: F401
