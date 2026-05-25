"""Project root detection. All config values come from Hydra YAML (configs/datagen/)."""
from pathlib import Path


def find_project_root() -> Path:
    """Walk up from this file to find the repository root (contains configs/ directory)."""
    current = Path(__file__).resolve().parent
    for _ in range(10):
        if (current / "configs").is_dir():
            return current
        current = current.parent
    # Fallback: DataGenerationRefactored is at src/brepnet/data/DataGenerationRefactored
    return Path(__file__).resolve().parents[4]


PROJECT_ROOT = find_project_root()
DATAGEN_ROOT = Path(__file__).resolve().parent


def resolve_path(path_str: str) -> Path:
    """Resolve a path from config: absolute stays absolute, relative is relative to DATAGEN_ROOT."""
    p = Path(path_str)
    if p.is_absolute():
        return p
    return DATAGEN_ROOT / p
