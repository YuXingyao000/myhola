"""Package audited presentation artifacts; never recompute research measurements."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import re


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def data_url(path: Path, mime: str) -> str:
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    root = args.root.resolve()
    page = root / "page"
    html = (page / "index.html").read_text()
    identity = "2026-10-13 · 完整容量512维×8层 · 验证最佳epoch42 / step15609"
    html = re.sub(r"<title>.*?</title>", "<title>完整容量 Intersection 失败统计 · 2026-10-13</title>", html, count=1)
    if 'data-experiment-identity="full-capacity-20261013"' not in html:
        html = html.replace("</h1>", '</h1><p class="muted" data-experiment-identity="full-capacity-20261013">' + identity + "</p>", 1)
    (page / "index.html").write_text(html)
    app = (page / "app.js").read_text()
    source_data = (page / "data.js").read_text()
    prefix = "window.PAIRED_CENSUS_DATA="
    if not source_data.startswith(prefix) or not source_data.rstrip().endswith(";"):
        raise ValueError("Unexpected generated page data contract")
    payload = json.loads(source_data[len(prefix):].rstrip()[:-1])
    if "data-coverage-note" not in app:
        context = payload["main"]["failed_face_context"]
        states = context["inspection_states"]
        coverage = (
            f"检查覆盖：{context['counted_final_failed_faces']}个最终失败面中，"
            f"{states.get('inspected', 0)}个完成候选检查、"
            f"{states.get('no_candidate_inspected', 0)}个没有候选可检查、"
            f"{states.get('partial', 0)}个检查不完整、{states.get('unknown', 0)}个未知。"
            "没有可检查候选的面不能视为检查通过。"
        )
        markup = '<br><strong data-coverage-note="true">' + coverage + "</strong>"
        addition = '${c.name==="persistent_face_context_errors"?' + json.dumps(markup, ensure_ascii=False) + ':""}'
        old_caption = "${esc(c.caption)} <a"
        if app.count(old_caption) != 1:
            raise ValueError("Expected a single chart-caption template")
        app = app.replace(old_caption, "${esc(c.caption)}" + addition + " <a")
        (page / "app.js").write_text(app)
    sources = [page / name for name in ("index.html", "app.js", "data.js")]
    for chart in payload["charts"]:
        for key, mime in (("png", "image/png"), ("svg", "image/svg+xml")):
            path = root / chart[key]
            sources.append(path)
            chart[key] = data_url(path, mime)
    for key in ("png", "svg"):
        app = app.replace("../${c." + key + "}", "${c." + key + "}")
    app = app.replace('<a href="${c.png}" target="_blank">', '<a href="${c.png}" download="${c.name}.png">')
    app = app.replace('<a href="${c.svg}">SVG</a>', '<a href="${c.svg}" download="${c.name}.svg">SVG</a>')
    historical_link = '<a href="../../../joint_boundary/legacy_failures_20261011/page/#${id}">查看原HoLa历史3D案例</a>'
    if historical_link not in app:
        raise ValueError("Historical reference markup changed")
    app = app.replace(historical_link, "原HoLa历史图库收录案例（独立资源，离线版未打包）")
    downloads = []
    for name, label, mime in (
        ("NOTE.md", "下载研究判断", "text/markdown;charset=utf-8"),
        ("REPORT.md", "下载中文报告", "text/markdown;charset=utf-8"),
        ("compact_summary.json", "下载紧凑统计", "application/json"),
        ("cases.csv", "下载逐件CSV", "text/csv;charset=utf-8"),
    ):
        path = root / name
        sources.append(path)
        downloads.append(f'<a href="{data_url(path, mime)}" download="{name}">{label}</a>')
    replacement = '<div class="links">' + "".join(downloads) + "</div>"
    html, changed = re.subn(r'<div class="links">.*?</div>', lambda _: replacement, html, count=1)
    if changed != 1:
        raise ValueError("Expected a single presentation navigation block")
    html = html.replace("</h1>", '</h1><p class="muted">自包含离线版；四组筛选、逐件数据及四图均已内嵌，无需HTTP服务。</p>', 1)
    inline_data = prefix + json.dumps(payload, ensure_ascii=False, allow_nan=False).replace("</", "<\\/") + ";"
    source_scripts = '<script src="data.js"></script><script src="app.js"></script>'
    if source_scripts not in html:
        raise ValueError("Expected generated external script tags")
    html = html.replace(source_scripts, "<script>" + inline_data + "</script><script>" + app + "</script>")
    if re.search(r'<script\b[^>]*\bsrc\s*=', html, re.IGNORECASE):
        raise ValueError("Standalone HTML retains an external script")
    destination = page / "standalone.html"
    destination.write_text(html)
    report = {
        "schema": "paired_census_standalone_v1",
        "scope": "Presentation packaging only; scientific measurements remain unchanged.",
        "experiment_identity": identity,
        "sources": [{"path": str(path), "sha256": sha256(path)} for path in sources],
        "output": {"path": str(destination), "sha256": sha256(destination), "bytes": destination.stat().st_size},
        "pair_count": len(payload["pairs"]),
        "main_denominator": payload["main"]["denominator"],
        "embedded_charts": len(payload["charts"]),
        "embedded_downloads": ["NOTE.md", "REPORT.md", "compact_summary.json", "cases.csv"],
        "historical_3d_atlas": "Not embedded; clearly labelled independent historical resource.",
    }
    (root / "standalone_packaging.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("output", "pair_count", "main_denominator", "embedded_charts")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
