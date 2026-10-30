"""普通 Python 入口：冻结抽样清单 → 启动 Blender 批量拍照 → 汇总结果。

本阶段没有 LLM 调用。Blender 内的「换零件、拍照」循环见 capture.py。
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys

from PIL import Image, ImageDraw


HERE = Path(__file__).resolve().parent
FACTORY = HERE.parent


def run_workflow(args):
    request = prepare_request(args)
    save_request(request, args.output_dir)
    returncode = launch_blender(request, args.output_dir)
    summary = summarize_results(request, args.output_dir, returncode)
    make_overview(request, args.output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_candidates(stl_dir, split):
    """只取源 split 内成功的 STL；同一字节摘要只保留一个 ID。"""
    unique = {}
    with (stl_dir / "manifest.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["status"] == "success" and split in row["splits"]:
                unique.setdefault(row["stl_sha256"], row)
    return sorted(unique.values(), key=lambda row: row["model_id"])


def prepare_request(args):
    ready = json.loads(args.ready_file.read_text())
    config = json.loads(args.config.read_text())
    placements = ready["placements"]
    if not placements:
        raise ValueError("ready.json 没有可用位置")
    for placement in placements:
        validation = placement.get("support_validation")
        if validation is None or not validation["pass"]:
            raise ValueError("ready.json 缺少通过的渲染支撑面验收；请先重新执行 Step 2 验收")
    if not 0 < config["longest_side_m"][0] <= config["longest_side_m"][1] <= ready["config"]["proxy_size_m"]:
        raise ValueError("零件最长边必须位于 Step 2 代理尺寸内")
    if sha256(ready["base_scene"]) != ready["scene_hash"]:
        raise ValueError("基础场景与 Step 2 的摘要不一致")
    candidates = read_candidates(args.stl_dir, config["split"])
    if not 0 < args.count <= len(candidates):
        raise ValueError(f"抽样数必须位于 1..{len(candidates)}")

    rng = random.Random(config["seed"])
    selected = rng.sample(candidates, args.count)
    placement_order = list(range(len(placements)))
    rng.shuffle(placement_order)
    jobs = []
    for index, row in enumerate(selected):
        stl_file = args.stl_dir / row["stl_file"]
        if sha256(stl_file) != row["stl_sha256"]:
            raise ValueError(f"STL 与原清单摘要不一致：{stl_file}")
        jobs.append({
            "index": index, "model_id": row["model_id"], "source": row,
            "stl_file": str(stl_file),
            "placement_index": placement_order[index % len(placements)],
            "longest_side_m": rng.uniform(*config["longest_side_m"]),
            "seed": config["seed"] + index,
        })
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "ready_file": str(args.ready_file), "ready_sha256": sha256(args.ready_file),
        "base_scene": ready["base_scene"], "scene_hash": ready["scene_hash"],
        "step2_config": ready["config"], "placements": placements,
        "config": config, "eligible_unique_stl_count": len(candidates),
        "manifest_file": str(args.stl_dir / "manifest.jsonl"),
        "manifest_sha256": sha256(args.stl_dir / "manifest.jsonl"),
        "blender": str(args.blender), "device": args.device, "gpu_index": args.gpu_index,
        "output_dir": str(args.output_dir),
        "command": [sys.executable, *sys.argv], "jobs": jobs,
        "step2_human_review": ready["human_review"], "human_review": "pending",
    }


def save_request(request, output_dir):
    """保存本次实际使用的源码，含尚未提交的文件；不只记录 Git HEAD。"""
    output_dir.mkdir(parents=True, exist_ok=False)
    files = [*HERE.glob("*.py"), *HERE.glob("*.json"),
             *(FACTORY / "step2").glob("*.py"),
             FACTORY / "geometry.py", FACTORY / "visibility.py"]
    source_hashes = {}
    for path in sorted(files):
        relative = path.relative_to(FACTORY)
        target = output_dir / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        source_hashes[str(relative)] = sha256(path)
    request["source_hashes"] = source_hashes
    request["git_commit"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=FACTORY, text=True).strip()
    diff = subprocess.check_output(["git", "diff", "HEAD", "--", str(FACTORY)], cwd=FACTORY)
    (output_dir / "git.diff").write_bytes(diff)
    (output_dir / "request.json").write_text(json.dumps(request, ensure_ascii=False, indent=2) + "\n")


def launch_blender(request, output_dir):
    # 执行刚刚冻结的源码副本；长任务期间工作区改动不会改变本次实现。
    command = [request["blender"], "--background", "--factory-startup", "--python-exit-code", "1",
               "--threads", str(request["config"]["threads"]), "--python",
               str(output_dir / "source" / "step3" / "capture.py"),
               "--", "--request", str(output_dir / "request.json")]
    visible_gpu = str(request["gpu_index"]) if request["device"] == "OPTIX" else ""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=visible_gpu, CUDA_DEVICE_ORDER="PCI_BUS_ID",
               HIP_VISIBLE_DEVICES="", ONEAPI_DEVICE_SELECTOR="*:cpu",
               OMP_NUM_THREADS=str(request["config"]["threads"]))
    (output_dir / "blender_command.json").write_text(json.dumps(command, indent=2) + "\n")
    device_label = f"OPTIX GPU {visible_gpu}" if visible_gpu else "CPU"
    print(f"拍摄 {len(request['jobs'])} 个零件 · {device_label} · {request['config']['threads']} CPU 线程\n"
          f"日志：{output_dir / 'blender.log'}", flush=True)
    with (output_dir / "blender.log").open("w") as log:
        return subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT).returncode


def summarize_results(request, output_dir, returncode):
    records_file = output_dir / "renders.jsonl"
    records = [json.loads(line) for line in records_file.read_text().splitlines()] if records_file.exists() else []
    by_id = {record["model_id"]: record for record in records}
    counts = Counter(record["status"] for record in records)
    missing = [job["model_id"] for job in request["jobs"] if job["model_id"] not in by_id]
    summary = {
        "status": "completed" if returncode == 0 and counts["success"] == len(request["jobs"]) else "incomplete",
        "planned": len(request["jobs"]), "success": counts["success"], "failed": counts["failed"],
        "not_run": len(missing), "not_run_ids": missing, "blender_exit_code": returncode,
        "human_review": "pending", "output_dir": str(output_dir),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def make_overview(request, output_dir):
    """按预定列表展示所有样本；没有图片的位置保留空格，不删失败样本。"""
    size, columns = 192, 5
    rows = (len(request["jobs"]) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * size, rows * (size + 24)), "#e8ebed")
    draw = ImageDraw.Draw(sheet)
    records_file = output_dir / "renders.jsonl"
    records = [json.loads(line) for line in records_file.read_text().splitlines()] if records_file.exists() else []
    statuses = {record["model_id"]: record["status"] for record in records}
    for index, job in enumerate(request["jobs"]):
        x, y = (index % columns) * size, (index // columns) * (size + 24)
        path = output_dir / job["model_id"] / "rgb.png"
        if path.exists():
            with Image.open(path) as image:
                sheet.paste(image.convert("RGB").resize((size, size)), (x, y))
        draw.text((x + 6, y + size + 4), f"{job['model_id']} / P{job['placement_index'] + 1}", fill="black")
        status = statuses.get(job["model_id"], "not_run")
        if status != "success":
            draw.rectangle((x, y, x + size, y + 22), fill="#963b35")
            draw.text((x + 6, y + 5), status.upper(), fill="white")
    sheet.save(output_dir / "overview.jpg", quality=92)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ready_file", type=Path, required=True)
    parser.add_argument("--stl_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--blender", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--device", choices=("CPU", "OPTIX"), default="CPU")
    parser.add_argument("--gpu_index", type=int, help="OPTIX 使用的物理 GPU 编号；其余 GPU 对 Blender 隐藏")
    args = parser.parse_args()
    if args.device == "OPTIX" and args.gpu_index is None:
        parser.error("--device OPTIX 需要显式指定 --gpu_index")
    for name in ("ready_file", "stl_dir", "output_dir", "blender", "config"):
        setattr(args, name, getattr(args, name).resolve())
    summary = run_workflow(args)
    return 0 if summary["status"] == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
