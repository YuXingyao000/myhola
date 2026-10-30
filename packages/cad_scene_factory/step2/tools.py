"""Blender 工具入口：describe / prepare_supports / measure / preview。JSON 输入输出。

blender --background --python step2/tools.py -- --job /absolute/job.json
每个 job 重新打开 base_scene；只向 output_dir 写产物。
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preview import preview_placement
from scene import describe_scene, measure_region, open_scene, prepare_support_scene
from spatial import SceneGeometry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    job = json.loads(args.job.read_text())
    output_dir = Path(job["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    open_scene(job["base_scene"])
    config = job["config"]
    if job["action"] == "describe":
        result = describe_scene(SceneGeometry(config), job["views"])
    elif job["action"] == "prepare_supports":
        result = prepare_support_scene(job["regions"], output_dir / "prepared_scene.blend")
    elif job["action"] == "measure":
        geometry = SceneGeometry(config)
        views = {view["id"]: view for view in job["views"]}
        result = {"regions": [measure_region(geometry, region, views[region["view_id"]], config)
                              for region in job["regions"]]}
    elif job["action"] == "preview":
        result = preview_placement(job["region"], job["proposal"], config, output_dir)
    else:
        raise ValueError(job["action"])
    (output_dir / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
