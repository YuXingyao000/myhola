"""Repository hygiene checks run by pre-commit and CI.

    python tools/check_repo.py layout   # top-level whitelist, no data/binary files
    python tools/check_repo.py dead     # every src module reachable from an entry point, CLI or test

Exit status is non-zero when a check fails; each failure prints one line.
"""
from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TOP_LEVEL = {
    ".github", ".gitignore", ".pre-commit-config.yaml", "AGENTS.md", "CLAUDE.md", "README.md",
    "configs", "docs", "environment.yml", "packages", "pyproject.toml", "src", "tests", "thirdparty", "tools",
}
DATA_SUFFIXES = {
    ".npz", ".npy", ".pt", ".pth", ".ckpt", ".safetensors", ".pkl", ".h5", ".hdf5",
    ".png", ".jpg", ".jpeg", ".webp", ".exr", ".hdr", ".tif", ".tiff", ".gif", ".mp4", ".pdf",
    ".zip", ".tar", ".gz", ".7z", ".blend", ".ply", ".obj", ".stl", ".step", ".stp", ".glb", ".fbx",
    ".exe", ".dll", ".so", ".jsonl", ".csv",
}
# Vendored third-party evaluation code ships its own sample meshes and binaries.
DATA_ALLOWED_PREFIXES = ("src/eval/lfd/evaluation_scripts/", "docs/")

ENTRY_POINTS = ("src.train", "src.inference", "src.evaluate")
# Loaded by string in a registry, so static import analysis cannot see them.
DYNAMIC_MODULES = {
    "src.models.vae.legacy", "src.models.vae.intersection", "src.models.vae.joint",
    "src.models.vae.uv_intersection_training", "src.models.vae.profiling",
}
SKIP_PREFIXES = ("src.eval.lfd.evaluation_scripts",)


def tracked_files() -> list[str]:
    return subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines()


def check_layout() -> list[str]:
    errors = []
    for name in tracked_files():
        top = name.split("/", 1)[0]
        if top not in TOP_LEVEL:
            errors.append(f"{name}: top-level entry '{top}' is not in the whitelist (tools/check_repo.py)")
        if Path(name).suffix.lower() in DATA_SUFFIXES and not name.startswith(DATA_ALLOWED_PREFIXES):
            errors.append(f"{name}: data/binary files belong in /mnt/d/data, runs or research records, not git")
    return errors


def module_name(path: Path) -> str:
    parts = path.relative_to(ROOT).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def imported_modules(path: Path, package: str) -> set[str]:
    found = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                anchor = package.split(".")[: len(package.split(".")) - node.level + 1]
                base = ".".join(anchor + ([base] if base else []))
            found.add(base)
            found.update(f"{base}.{alias.name}" for alias in node.names)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.startswith("src."):
            found.add(node.value)
    return found


def has_main_guard(path: Path) -> bool:
    return any(
        isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name) and node.test.left.id == "__name__"
        for node in ast.parse(path.read_text(encoding="utf-8")).body
    )


def check_dead_modules() -> list[str]:
    files = [ROOT / name for name in tracked_files() if name.endswith(".py")]
    modules = {module_name(path): path for path in files}
    edges = {}
    for name, path in modules.items():
        package = name if path.name == "__init__.py" else name.rpartition(".")[0]
        targets = set()
        for imported in imported_modules(path, package):
            parts = imported.split(".")
            for end in range(len(parts), 0, -1):
                candidate = ".".join(parts[:end])
                if candidate in modules:
                    targets.add(candidate)
                    break
        # Importing a module executes every parent package __init__.
        for target in list(targets):
            parts = target.split(".")
            targets.update(".".join(parts[:end]) for end in range(1, len(parts)) if ".".join(parts[:end]) in modules)
        edges[name] = targets
    roots = {name for name in modules if name in ENTRY_POINTS or name in DYNAMIC_MODULES}
    roots |= {name for name, path in modules.items()
              if name.startswith(("tests.", "tools.")) or (name.startswith("src.") and has_main_guard(path))}
    reached, stack = set(), list(roots)
    while stack:
        name = stack.pop()
        if name not in reached:
            reached.add(name)
            stack.extend(edges.get(name, ()))
    return [f"{modules[name].relative_to(ROOT)}: not reachable from an entry point, CLI, tool or test"
            for name in sorted(modules)
            if name.startswith("src.") and name not in reached and not name.startswith(SKIP_PREFIXES)
            and modules[name].name != "__init__.py"]


def main() -> int:
    checks = {"layout": check_layout, "dead": check_dead_modules}
    selected = sys.argv[1:] or list(checks)
    errors = [error for name in selected for error in checks[name]()]
    print("\n".join(errors))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
