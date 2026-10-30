"""Step 4：冻结输入 → 加载模型 → 逐图增强与合成 → 保存比较结果。

普通 Python 批处理，不调用 LLM 或 ComfyUI。每个 run 使用一个场景的固定提示词。
先用 --prepare_only 审阅输入，再对保存的 request.json 使用 --run_request 执行。
"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

from masks import read_inputs, make_masks, composite_and_check
from report import summarize_results, make_comparisons


HERE = Path(__file__).resolve().parent


def run_workflow(args):
    if args.run_request:
        request = json.loads(args.run_request.read_text())
    else:
        request = prepare_request(args)
        save_request(request)
        if args.prepare_only:
            print(f"请求已冻结：{args.output_dir / 'request.json'}", flush=True)
            return
    output_dir = Path(request["output_dir"])
    # 无论直接运行还是稍后执行请求，都运行本次冻结的完整源码。
    frozen_script = output_dir / "source" / "workflow.py"
    if Path(__file__).resolve() != frozen_script:
        os.execv(sys.executable, [sys.executable, str(frozen_script),
                                 "--run_request", str(output_dir / "request.json")])
    if (output_dir / "results.jsonl").exists():
        raise ValueError("此请求已执行；新实验请使用新目录")
    # 在导入 torch 前限定物理卡；模型内部始终使用唯一可见的 cuda:0。
    os.environ["CUDA_VISIBLE_DEVICES"] = str(request["gpu_index"])
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    sys.path.insert(0, str(output_dir / "source"))
    from model import FluxEnhancer
    enhancer = FluxEnhancer(Path(request["model_dir"]), Path(request["transformer_file"]),
                            request["prompt"], request["config"])
    records = enhance_batch(enhancer, request)
    summary = summarize_results(request, records, output_dir)
    make_comparisons(request, records, output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_request(args):
    config = json.loads(args.config.read_text())
    if args.pipeline:
        config["pipeline"] = args.pipeline
    if args.guidance_scale is not None:
        config["guidance_scale"] = args.guidance_scale
    if args.processing_size is not None:
        config["processing_size"] = args.processing_size
    if args.resident_model:
        config["resident_model"] = True
    if args.strengths:
        config["strengths"] = args.strengths
    if (not config["strengths"] or len(set(config["strengths"])) != len(config["strengths"])
            or any(not 0 < strength <= 1 for strength in config["strengths"])):
        raise ValueError("strengths 必须互不重复且位于 (0,1]")
    if config["protect_padding_px"] < 0 or config["feather_px"] < 0:
        raise ValueError("mask 边缘参数不能为负")
    scene = json.loads((HERE / "prompts" / "scenes.json").read_text())[args.scene_id]
    prompt_file = args.prompt_file or HERE / "prompts" / scene["prompt_file"]
    source_request = json.loads((args.render_dir / "request.json").read_text())
    rows = {row["model_id"]: row for row in map(json.loads, (args.render_dir / "renders.jsonl").read_text().splitlines())}
    selected = source_request["jobs"][:args.limit] if args.limit is not None else source_request["jobs"]
    if not selected or len({job["model_id"] for job in selected}) != len(selected):
        raise ValueError("输入列表为空或含重复 CAD ID")
    samples = []
    for job in selected:
        row = rows[job["model_id"]]
        if row["status"] != "success" or not row["contact"]["pass"]:
            raise ValueError(f"输入未通过 Step 3 支撑检查：{job['model_id']}")
        read_inputs(row["images"]["rgb"], row["images"]["mask"])
        samples.append({
            "model_id": job["model_id"], "index": job["index"],
            "rgb_file": row["images"]["rgb"], "rgb_sha256": sha256(row["images"]["rgb"]),
            "mask_file": row["images"]["mask"], "mask_sha256": sha256(row["images"]["mask"]),
            "metadata_file": str(args.render_dir / job["model_id"] / "metadata.json"),
            "source": row,
        })
    jobs = [{"job_id": f"{sample['model_id']}/strength_{str(strength).replace('.', '_')}",
             "model_id": sample["model_id"], "strength": strength,
             "seed": config["seed"] + sample["index"]}
            for sample in samples for strength in config["strengths"]]
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "render_dir": str(args.render_dir), "scene_id": args.scene_id,
        "source_request_sha256": sha256(args.render_dir / "request.json"),
        "source_scene_hash": source_request["scene_hash"],
        "prompt": prompt_file.read_text().strip(), "prompt_file": str(prompt_file),
        "prompt_sha256": sha256(prompt_file), "config": config,
        "model_dir": str(args.model_dir), "transformer_file": str(args.transformer_file),
        "gpu_index": args.gpu_index, "output_dir": str(args.output_dir),
        "samples": samples, "jobs": jobs, "command": [sys.executable, *sys.argv],
        "human_review": "pending",
    }


def save_request(request):
    output_dir = Path(request["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=False)
    shutil.copytree(HERE, output_dir / "source", ignore=shutil.ignore_patterns("__pycache__"))
    (output_dir / "prompt.md").write_text(request["prompt"] + "\n")
    snapshot = output_dir / "source"
    request["source_hashes"] = {str(path.relative_to(snapshot)): sha256(path)
                                for path in snapshot.rglob("*") if path.is_file()}
    request["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE, text=True).strip()
    (output_dir / "git.diff").write_bytes(subprocess.check_output(["git", "diff", "HEAD"], cwd=HERE))
    # 只追踪实际使用的权重：GGUF transformer + VAE + 两个文本编码器等。
    model_dir = Path(request["model_dir"])
    model_files = [Path(request["transformer_file"]), model_dir / "model_index.json",
                   model_dir / "transformer" / "config.json"]
    for folder in ("vae", "text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2", "scheduler"):
        model_files.extend(path for path in (model_dir / folder).rglob("*") if path.is_file())
    print("记录模型文件摘要与输入清单……", flush=True)
    request["model_files"] = {str(path): {"bytes": path.stat().st_size, "sha256": sha256(path)}
                              for path in sorted(model_files)}
    request["versions"] = {name: importlib.metadata.version(name)
                           for name in ("torch", "diffusers", "transformers", "accelerate", "gguf", "Pillow", "numpy", "scipy")}
    (output_dir / "request.json").write_text(json.dumps(request, ensure_ascii=False, indent=2) + "\n")


def enhance_batch(enhancer, request):
    output_dir = Path(request["output_dir"])
    config = request["config"]
    samples = {sample["model_id"]: sample for sample in request["samples"]}
    records = []
    with (output_dir / "results.jsonl").open("w") as stream:
        for index, job in enumerate(request["jobs"]):
            print(f"[{index + 1}/{len(request['jobs'])}] {job['job_id']} · seed {job['seed']}", flush=True)
            directory = output_dir / job["job_id"]
            directory.mkdir(parents=True)
            record = dict(job, status="failed")
            started = time.monotonic()
            # 唯一的逐样本异常边界：坏样本保留在分母中，其余输入继续处理。
            try:
                sample = samples[job["model_id"]]
                for kind in ("rgb", "mask"):
                    if sha256(sample[f"{kind}_file"]) != sample[f"{kind}_sha256"]:
                        raise ValueError(f"输入 {kind} 在冻结请求后发生变化")
                rgb, mask = read_inputs(sample["rgb_file"], sample["mask_file"])
                protected, edit_mask, blend_mask = make_masks(
                    mask, config["protect_padding_px"], config["feather_px"])
                protected.save(directory / "protected_mask.png")
                edit_mask.save(directory / "edit_mask.png")
                blend_mask.save(directory / "blend_mask.png")
                generated, inference = enhancer.enhance(rgb, edit_mask, job["strength"], job["seed"])
                generated.save(directory / "generated.png")
                enhanced, checks = composite_and_check(rgb, generated, protected, blend_mask)
                enhanced.save(directory / "enhanced.png")
                record.update(status="success", inference=inference, checks=checks,
                              output_sha256=sha256(directory / "enhanced.png"))
            except Exception as error:
                record["error"] = f"{type(error).__name__}: {error}"
                traceback.print_exc()
            record["elapsed_seconds"] = time.monotonic() - started
            (directory / "metadata.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            records.append(record)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render_dir", type=Path)
    parser.add_argument("--scene_id")
    parser.add_argument("--prompt_file", type=Path, help="用一份固定提示词覆盖该场景的默认文本，用于独立对照实验")
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--model_dir", type=Path)
    parser.add_argument("--transformer_file", type=Path)
    parser.add_argument("--gpu_index", type=int, default=7)
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    parser.add_argument("--pipeline", choices=("kontext", "flux"))
    parser.add_argument("--guidance_scale", type=float)
    parser.add_argument("--processing_size", type=int, choices=(512, 1024))
    parser.add_argument("--resident_model", action="store_true", help="缓存提示词后释放文本编码器，将采样模型常驻 GPU")
    parser.add_argument("--strengths", nargs="+", type=float)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--prepare_only", action="store_true")
    parser.add_argument("--run_request", type=Path)
    args = parser.parse_args()
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.resolve())
    if not args.run_request:
        required = ("render_dir", "scene_id", "output_dir", "model_dir", "transformer_file")
        if any(getattr(args, name) is None for name in required):
            parser.error("新实验需要 --render_dir --scene_id --output_dir --model_dir --transformer_file")
        if args.limit is not None and args.limit <= 0:
            parser.error("--limit 必须大于零")
    summary = run_workflow(args)
    return 0 if summary is None or summary["status"] == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
