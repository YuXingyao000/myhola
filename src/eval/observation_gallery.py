"""Export traceable single-view case galleries from a completed diffusion audit.

Existing metrics are displayed verbatim. STEP rendering is only for the figures;
it does not repair geometry, align shapes, select candidates, or replace metrics.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import multiprocessing as mp
from pathlib import Path
import tempfile
import time

import numpy as np
from PIL import Image

from src.models.diffusion.image_loading import load_image_view
from src.eval.diffusion_audit_report import METHOD_LABELS, STAGE_LABELS
from src.common.io import atomic_json, read_json, sha256_file


def checked_sources(case: dict, row: dict) -> dict:
    """Match manifest paths and saved audit hashes before reading any geometry."""
    if not {"image_path", "gt_step_path", "step_path"}.issubset(row.get("sources", {})):
        raise ValueError("Audit record has incomplete source provenance")
    checked = {}
    for key, source in row.get("sources", {}).items():
        path = Path(source["path"]).resolve()
        if key not in case or Path(case[key]).resolve() != path:
            raise ValueError(f"Manifest and audit source paths differ: {key}")
        current = sha256_file(path) if path.is_file() else None
        if current != source["sha256"]:
            raise ValueError(f"Source hash changed: {path}")
        checked[key] = {"path": str(path), "sha256": current}
    return checked


def _render_child(step: str, resolution: int, deflection: float, directory: str) -> None:
    from src.eval.metrics.observation import legacy_svr_camera, render_step

    destination = Path(directory)
    try:
        depth = render_step(Path(step), legacy_svr_camera(resolution), deflection)
        np.save(destination / "depth.npy", depth)
        atomic_json(destination / "result.json", {"status": "ok"})
    except Exception as error:
        atomic_json(destination / "result.json", {
            "status": "error", "error": f"{type(error).__name__}: {error}",
        })


def bounded_render(step: Path | None, resolution: int, deflection: float, timeout: float) -> dict:
    """A fresh CPU process bounds OCC import, triangulation and rasterization."""
    if step is None:
        return {"status": "missing", "error": "No verified STEP source"}
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="hola-gallery-render-") as directory:
        process = mp.get_context("spawn").Process(
            target=_render_child, args=(str(step), resolution, deflection, directory),
        )
        process.start()
        process.join(timeout)
        if process.is_alive():
            process.terminate()
            process.join(3)
            if process.is_alive():
                process.kill()
                process.join()
            return {"status": "timeout", "seconds": time.monotonic() - started}
        result = read_json(Path(directory) / "result.json")
        if process.exitcode != 0 or not result:
            return {"status": "crash", "exit_code": process.exitcode}
        if result["status"] == "ok":
            result["depth"] = np.load(Path(directory) / "depth.npy", allow_pickle=False)
        result["seconds"] = time.monotonic() - started
        return result


def load_case_records(audit_dir: Path, image_ids: list[str], protocol: dict) -> dict:
    manifest = read_json(audit_dir / "manifest.json")
    methods = protocol["config"]["methods"]
    known = {}
    for case in manifest["cases"]:
        key = (case["image_id"], case["method"])
        if key in known:
            raise ValueError(f"Gallery requires one candidate per method/image: {key}")
        known[key] = case
    missing = set(image_ids) - {key[0] for key in known}
    if missing:
        raise ValueError(f"Requested image IDs are absent from the manifest: {sorted(missing)}")
    records = {}
    for image_id in image_ids:
        records[image_id] = {}
        for method in methods:
            path = audit_dir / "cases" / method / image_id / "audit.json"
            row = read_json(path)
            record = {"audit_path": str(path), "audit_sha256": sha256_file(path) if path.is_file() else None}
            try:
                if not row:
                    raise ValueError("Missing audit record")
                if row.get("audit_fingerprint") != protocol["fingerprint"]:
                    raise ValueError("Audit record uses a different protocol")
                if (row.get("image_id"), row.get("method")) != (image_id, method):
                    raise ValueError("Audit identity differs from the requested case")
                if (image_id, method) not in known:
                    raise ValueError("Missing manifest case")
                record["sources"] = checked_sources(known[(image_id, method)], row)
                record["row"] = row
                record["status"] = "ok"
            except ValueError as error:
                record.update(status="source_error", error=str(error))
            records[image_id][method] = record
    return records


def source_path(record: dict, key: str) -> Path | None:
    if record.get("status") != "ok":
        return None
    source = record.get("sources", {}).get(key, {})
    return Path(source["path"]) if source.get("sha256") else None


def _number(value: float | None, percent: bool = False) -> str:
    if value is None:
        return "—"
    if percent:
        return f"{value * 100:.1f}%"
    if value == 0:
        return "0"
    coefficient, exponent = f"{value:.2e}".split("e")
    return rf"${coefficient}\times10^{{{int(exponent)}}}$"


def plot_case(image_id: str, records: dict, config: dict, output: Path, timeout: float, chinese: bool) -> dict:
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    text = lambda zh, en: zh if chinese else en
    methods = config["methods"]
    size = config["resolution"]
    usable = [record for record in records.values() if record["status"] == "ok"]
    common = usable[0] if usable else {}
    # Shared image and GT must genuinely be shared across methods.
    for record in usable[1:]:
        for key in ("image_path", "gt_step_path"):
            if record["sources"].get(key) != common["sources"].get(key):
                record.update(status="source_error", error=f"Methods have different {key} for {image_id}")
    usable = [record for record in usable if record["status"] == "ok"]
    image_path = source_path(common, "image_path")
    image = None
    image_state = {"status": "missing", "error": "No verified input image source"}
    if image_path is not None:
        try:
            image = load_image_view(image_path, "svr_imgs", 0, read_single_view=True)
            if image.shape[:2] != (size, size):
                image = np.asarray(Image.fromarray(image).resize((size, size), Image.Resampling.NEAREST))
            image_state = {"status": "ok"}
        except Exception as error:
            image_state = {"status": "error", "error": f"Input image: {error}"}
    mask = image.min(axis=2) < config["mask_threshold"] if image is not None else None
    renders = {"gt": bounded_render(source_path(common, "gt_step_path"), size, config["mesh_deflection"], timeout)}
    for method in methods:
        record = records[method]
        renders[method] = bounded_render(source_path(record, "step_path"), size, config["mesh_deflection"], timeout)
        if record["status"] != "ok":
            renders[method] = {"status": "source_error", "error": record["error"]}
    depths = [item["depth"] for item in renders.values() if item["status"] == "ok"]
    lo = min(float(np.nanmin(depth)) for depth in depths) if depths else 0.
    hi = max(float(np.nanmax(depth)) for depth in depths) if depths else 1.
    norm = Normalize(lo, max(hi, lo + 1e-9))
    fig, axes = plt.subplots(2, 2 + len(methods), figsize=(4.1 * (2 + len(methods)), 8.9))
    fig.subplots_adjust(left=.025, right=.94, top=.87, bottom=.15, hspace=.22, wspace=.09)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    def unavailable(ax, item):
        labels = {"missing": text("缺少来源", "Missing source"), "source_error": text("来源校验失败", "Source mismatch"),
                  "timeout": text("渲染超时", "Render timeout"), "error": text("无法渲染", "Not renderable"), "crash": text("渲染进程退出", "Renderer crashed")}
        reason = item.get("error", "").rsplit(": ", 1)[-1]
        ax.text(.5, .55, labels.get(item["status"], item["status"]), ha="center", va="center", transform=ax.transAxes, color="#A74635", fontsize=14)
        ax.text(.5, .4, reason[-85:], ha="center", va="center", transform=ax.transAxes, wrap=True, fontsize=9)

    for row_index in range(2):
        if image is not None:
            axes[row_index, 0].imshow(image if row_index == 0 else mask, cmap="gray", vmin=0, vmax=1)
        else:
            unavailable(axes[row_index, 0], image_state)
    axes[0, 0].set_title(text("输入灰模", "Input SVR image"), fontsize=16, pad=12)
    axes[1, 0].set_title(text("输入前景", "Input foreground"), fontsize=16, pad=12)
    axes[1, 0].set_xlabel(text("固定白底阈值", "Fixed white-background threshold") + f" {config['mask_threshold']}", fontsize=11, labelpad=12)
    depth_artist = None
    for column, key in enumerate(["gt", *methods], 1):
        item = renders[key]
        label = "GT" if key == "gt" else METHOD_LABELS.get(key, key) if chinese else key
        axes[0, column].set_title(f"{label} · " + text("可见深度", "Visible depth"), fontsize=16, pad=12)
        axes[1, column].set_title(f"{label} · " + text("轮廓叠加", "Silhouette overlay"), fontsize=16, pad=12)
        if item["status"] == "ok":
            depth = item["depth"]
            depth_artist = axes[0, column].imshow(np.ma.masked_invalid(depth), cmap="viridis", norm=norm)
            axes[1, column].imshow(image if image is not None else np.ones((size, size, 3)))
            if mask is not None and mask.any() and not mask.all():
                axes[1, column].contour(mask, levels=[.5], colors=["#159A75"], linewidths=1.4)
            foreground = np.isfinite(depth)
            if foreground.any() and not foreground.all():
                axes[1, column].contour(foreground, levels=[.5], colors=["#D65C32"], linewidths=1.1)
        else:
            unavailable(axes[0, column], item)
            unavailable(axes[1, column], item)
        if key == "gt":
            available = [record.get("row", {}).get("observation", {}) for record in usable]
            observation = next((value for value in available if value.get("status") == "ok"), {})
            caption = text("GT 对输入轮廓 IoU", "GT / input silhouette IoU") + " " + _number(observation.get("gt_input_silhouette_iou"), True)
        else:
            row = records[key].get("row", {})
            observation = row.get("observation", {})
            stage = row.get("trace", {}).get("primary_stage", "unknown")
            validity = row.get("validity", "unknown")
            valid_label = {"valid": "合法", "invalid": "非法", "unknown": "未知"}.get(validity, validity) if chinese else validity
            stage_label = STAGE_LABELS.get(stage, stage) if chinese else stage
            caption = f"{valid_label} · {stage_label}\n"
            caption += text("输入轮廓 IoU", "Input silhouette IoU") + " " + _number(observation.get("silhouette_iou"), True) + "\n"
            caption += text("可见表面平方误差", "Visible surface squared CD") + " " + _number(observation.get("visible_face_cd_sq"))
        axes[1, column].set_xlabel(caption, fontsize=11, labelpad=12)
    if depth_artist is not None:
        bar_axis = fig.add_axes([.951, .53, .009, .28])
        fig.colorbar(depth_artist, cax=bar_axis).set_label(text("相机深度", "Camera depth"), fontsize=10)
    fig.suptitle(f"CAD {image_id}  ·  " + text("单视图结果检查", "Single-view case inspection"), x=.025, ha="left", fontsize=23, y=.97)
    fig.text(.025, .92, text("深度共用色阶；绿色为输入轮廓，橙色为重渲染轮廓。", "Shared depth scale; green: input boundary; orange: rerendered boundary."), fontsize=12)
    fig.text(.025, .015, text("标注直接读取已有 audit；仅重新渲染绘图，不修改指标。深度和可见三维误差依赖配对 GT，不能单凭照片确定。", "Labels come from the saved audit; rerendering does not change metrics. Depth and visible 3-D error use paired GT."), fontsize=10, color="#58616A")
    for extension in ("png", "svg"):
        fig.savefig(output / f"{image_id}.{extension}", dpi=180, facecolor="white")
    plt.close(fig)
    return {
        "image_read": image_state,
        "depth_color_limits": [lo, hi],
        "renders": {key: {name: value for name, value in item.items() if name != "depth"} for key, item in renders.items()},
        "records": {method: {**record, "row": {key: record.get("row", {}).get(key) for key in ("validity", "trace", "observation")}} for method, record in records.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--image-ids", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--font", type=Path, help="CJK font file; without this option labels are English")
    parser.add_argument("--render-timeout", type=float, default=60., help="Seconds per STEP, including worker startup")
    args = parser.parse_args()
    if args.render_timeout <= 0 or len(set(args.image_ids)) != len(args.image_ids):
        parser.error("Render timeout must be positive and image IDs must be distinct")
    if any(Path(image_id).name != image_id or image_id in (".", "..") for image_id in args.image_ids):
        parser.error("Image IDs must be single directory names")
    audit, output = args.audit_dir.resolve(), args.output_dir.resolve()
    protocol = read_json(audit / "protocol.json")
    if read_json(audit / "status.json").get("status") != "complete":
        raise ValueError("Gallery requires a completed audit")
    config = protocol["config"]
    for protected in (audit, Path(config["comparison_run"]).resolve()):
        if output.is_relative_to(protected):
            raise ValueError("Gallery output must be outside the source audit and comparison run")
    from src.eval.metrics.observation import renderer_dependencies
    dependencies = renderer_dependencies()
    repository = Path(__file__).resolve().parents[2]
    for relative, key in (("src/eval/metrics/observation.py", "observation_sha256"), ("src/preprocess/extract_visibility.py", "triangulation_sha256")):
        if protocol["source_sha256"].get(str(repository / relative)) != dependencies[key]:
            raise ValueError(f"Renderer source differs from the audit: {relative}")
    if protocol["runtime"]["pythonocc"] != dependencies["pythonocc_version"]:
        raise ValueError("PythonOCC version differs from the audit")
    records = load_case_records(audit, args.image_ids, protocol)
    output.mkdir(parents=True, exist_ok=False)
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager, pyplot as plt
    if args.font:
        font_manager.fontManager.addfont(args.font)
        plt.rcParams["font.family"] = [font_manager.FontProperties(fname=args.font).get_name(), "DejaVu Sans"]
    plt.rcParams.update({"svg.fonttype": "path", "font.size": 11})
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "audit_dir": str(audit), "image_ids": args.image_ids,
        "protocol_sha256": sha256_file(audit / "protocol.json"), "manifest_sha256": sha256_file(audit / "manifest.json"),
        "gallery_source_sha256": sha256_file(Path(__file__)), "renderer_dependencies": dependencies,
        "render_timeout": args.render_timeout, "config": config,
        "font": {"path": str(args.font.resolve()), "sha256": sha256_file(args.font)} if args.font else None,
        "metrics_recomputed": False, "cases": {},
    }
    for image_id in args.image_ids:
        metadata["cases"][image_id] = plot_case(image_id, records[image_id], config, output, args.render_timeout, args.font is not None)
        atomic_json(output / "metadata.json", metadata)
        print(f"Wrote {image_id}.png/.svg", flush=True)


if __name__ == "__main__":
    main()
