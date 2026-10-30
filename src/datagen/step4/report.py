"""按冻结请求汇总，并展示原图与各强度；失败位置不从图中删除。"""

from collections import Counter
import json

from PIL import Image, ImageDraw


def summarize_results(request, records, output_dir):
    counts = Counter(row["status"] for row in records)
    planned = len(request["jobs"])
    summary = {
        "status": "completed" if counts["success"] == planned else "incomplete",
        "scene_id": request["scene_id"], "planned": planned,
        "success": counts["success"], "failed": counts["failed"],
        "not_run": planned - len(records), "human_review": "pending",
        "description": "合成真实感增强；技术通过不等于照片真实性已验收",
        "output_dir": str(output_dir),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return summary


def make_comparisons(request, records, output_dir):
    """每页至多 10 个输入，便于查看 100 张图；五件参数比较保持同一行对齐。"""
    strengths = request["config"]["strengths"]
    samples = request["samples"]
    statuses = {row["job_id"]: row["status"] for row in records}
    tile, label = 256, 24
    columns = 1 + len(strengths)
    gallery = output_dir / "comparisons"
    gallery.mkdir(exist_ok=True)
    for start in range(0, len(samples), 10):
        page = samples[start:start + 10]
        sheet = Image.new("RGB", (columns * tile, len(page) * (tile + label)), "#e9ecef")
        draw = ImageDraw.Draw(sheet)
        for row, sample in enumerate(page):
            y = row * (tile + label)
            with Image.open(sample["rgb_file"]) as original:
                sheet.paste(original.convert("RGB").resize((tile, tile)), (0, y))
            draw.text((6, y + tile + 4), f"{sample['model_id']} / original", fill="black")
            for column, strength in enumerate(strengths, 1):
                job = next(job for job in request["jobs"]
                           if job["model_id"] == sample["model_id"] and job["strength"] == strength)
                x = column * tile
                path = output_dir / job["job_id"] / "enhanced.png"
                status = statuses.get(job["job_id"], "not_run")
                if status == "success":
                    with Image.open(path) as image:
                        sheet.paste(image.convert("RGB").resize((tile, tile)), (x, y))
                draw.text((x + 6, y + tile + 4), f"strength {strength:g} / {status}", fill="black")
        sheet.save(gallery / f"page_{start // 10:02d}.jpg", quality=95)
