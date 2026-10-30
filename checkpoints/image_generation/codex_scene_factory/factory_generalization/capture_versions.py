"""本次运行的旁路留存：不修改 Agent 文件或冻结的外围工具。

每当四个交付文件或自编 Python 模块的内容变化，保存一个不可变副本。时间表示观测时间，
不是 Agent 编辑时间；正式轮次可通过 evidence.json 的摘要匹配版本。
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time


FILES = ("build_scene.py", "views.json", "scene_plan.md", "assets.json")
EXCLUDED_DIRECTORIES = {"assets", "workflow", "previews", "__pycache__"}


def _read_authored_files(template_dir: Path) -> dict[str, bytes]:
    paths = {template_dir / name for name in FILES if (template_dir / name).is_file()}
    for directory, subdirectories, filenames in os.walk(template_dir):
        subdirectories[:] = sorted(
            name for name in subdirectories
            if name not in EXCLUDED_DIRECTORIES and not name.startswith(".")
        )
        paths.update(Path(directory) / name for name in filenames if name.endswith(".py"))
    return {path.relative_to(template_dir).as_posix(): path.read_bytes()
            for path in sorted(paths)}


def capture_versions(template_dir: Path) -> None:
    output = template_dir / "workflow" / "authored_versions"
    output.mkdir(exist_ok=True)
    previous = None
    while True:
        try:
            content = _read_authored_files(template_dir)
            # 不把写文件中途的内容或变化中的文件集合当作稳定版本。
            time.sleep(0.5)
            stable = content == _read_authored_files(template_dir)
        except FileNotFoundError:
            time.sleep(0.5)
            continue  # Agent 正在替换或删除文件，跳过本次观测。
        digests = {name: hashlib.sha256(value).hexdigest()
                   for name, value in content.items()}
        if stable and content and digests != previous:
            destination = output / f"version_{len(list(output.glob('version_*'))):03d}"
            destination.mkdir()
            for name, value in content.items():
                (destination / name).parent.mkdir(parents=True, exist_ok=True)
                (destination / name).write_bytes(value)
            metadata = {
                "observed_at_utc": datetime.now(timezone.utc).isoformat(),
                "sha256": digests,
                "source": "external_observer; authored files unchanged",
            }
            (destination / "snapshot.json").write_text(
                json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
            previous = digests
            print(destination, flush=True)
        try:
            state = json.loads((template_dir / "workflow_result.json").read_text())
        except json.JSONDecodeError:
            continue  # 外围正在写状态，下一次循环读取完整内容。
        if state["status"] != "running":
            return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("template_dir", type=Path)
    capture_versions(parser.parse_args().template_dir)
