"""Serve the existing failure viewer over loopback with per-case loading.

This changes delivery only: source HTML, JavaScript and all scientific payloads
remain untouched. HTTP gzip decoding is handled by the browser, so loading a
case does not depend on the JavaScript DecompressionStream API.
"""
from __future__ import annotations

from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
from urllib.parse import urlsplit


HERE = Path(__file__).resolve().parent
HOST = "127.0.0.1"
PORT = 8765

EMBEDDED_CASE_READER = '''    const packed=window.CASE_DATA[id];
    if(!packed)throw new Error("该案例缺少内嵌几何数据");
    if(typeof DecompressionStream!=="function")throw new Error("此浏览器不支持本地 gzip 解压，请用现代 Chrome 或 Edge 打开。");
    const binary=Uint8Array.from(atob(packed),c=>c.charCodeAt(0));
    const text=await new Response(new Blob([binary]).stream().pipeThrough(new DecompressionStream("gzip"))).text();
    const value=JSON.parse(text);'''

HTTP_CASE_READER = '''    const response=await fetch(`/cases/${encodeURIComponent(id)}.json`);
    if(!response.ok)throw new Error(`案例 ${id} 读取失败（HTTP ${response.status}）`);
    const value=await response.json();'''


def replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise ValueError(f"Expected one delivery substitution: {old[:90]!r}")
    return text.replace(old, new, 1)


@dataclass(frozen=True)
class ViewerAssets:
    routes: dict[str, tuple[str, bytes]]
    case_paths: dict[str, Path]
    sizes: dict[str, int]


def load_assets() -> ViewerAssets:
    """Assemble a small entry page once, without reading the 120 MB artifact."""
    import plotly

    manifest = json.loads((HERE / "data/manifest.json").read_text())
    case_ids = [case["id"] for case in manifest["cases"]]
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("Duplicate case identities in the existing manifest")
    case_paths = {}
    for case_id in case_ids:
        if not re.fullmatch(r"[0-9]{8}", case_id):
            raise ValueError(f"Unexpected case identity: {case_id!r}")
        path = HERE / "data/cases" / f"{case_id}.json.gz"
        if not path.is_file():
            raise FileNotFoundError(path)
        case_paths[case_id] = path

    app = replace_once((HERE / "app.js").read_text(), EMBEDDED_CASE_READER, HTTP_CASE_READER)
    template = (HERE / "template.html").read_text()
    template = replace_once(template, '<meta charset="utf-8">',
                            '<meta charset="utf-8">\n<link rel="icon" href="data:,">')
    template = replace_once(template, "<script>__PLOTLY_JS__</script>",
                            '<script src="/plotly.min.js"></script>')
    template = replace_once(template, "<script>__APP_JS__</script>",
                            '<script src="/app.js"></script>')
    template = replace_once(template, "全部数据已内嵌，可离线查看。",
                            "当前案例按需加载；全部 316 件可浏览，无需一次读取全部几何。")
    substitutions = {
        "__FONT_STYLE__": "",  # Keep the existing system-font fallback stack.
        "__MANIFEST_JSON__": json.dumps(manifest, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c"),
        "__CASE_DATA_JSON__": "{}",
    }
    for old, new in substitutions.items():
        template = replace_once(template, old, new)
    if re.search(r"__[A-Z_]+__", template):
        raise ValueError("Unresolved viewer template placeholder")
    plotly_path = Path(plotly.__file__).parent / "package_data/plotly.min.js"
    routes = {
        "/": ("text/html; charset=utf-8", template.encode()),
        "/app.js": ("text/javascript; charset=utf-8", app.encode()),
        "/plotly.min.js": ("text/javascript; charset=utf-8", plotly_path.read_bytes()),
    }
    routes["/index.html"] = routes["/"]
    sizes = {"entry_bytes": len(routes["/"][1]), "app_bytes": len(routes["/app.js"][1]),
             "plotly_bytes": len(routes["/plotly.min.js"][1]), "case_count": len(case_paths),
             "largest_case_bytes": max(path.stat().st_size for path in case_paths.values())}
    sizes["maximum_initial_bytes"] = sum(sizes[key] for key in
                                         ("entry_bytes", "app_bytes", "plotly_bytes", "largest_case_bytes"))
    if sizes["entry_bytes"] > 1_000_000 or sizes["maximum_initial_bytes"] >= 6_000_000:
        raise ValueError(f"Delivery size budget exceeded: {sizes}")
    return ViewerAssets(routes=routes, case_paths=case_paths, sizes=sizes)


def make_handler(assets: ViewerAssets) -> type[BaseHTTPRequestHandler]:
    class ViewerHandler(BaseHTTPRequestHandler):
        server_version = "FailureViewer/1.0"

        def do_GET(self) -> None:
            self.send_asset(head_only=False)

        def do_HEAD(self) -> None:
            self.send_asset(head_only=True)

        def send_asset(self, head_only: bool) -> None:
            target = urlsplit(self.path)
            if target.scheme or target.netloc or target.query:
                self.send_error(404)
                return
            if target.path in assets.routes:
                content_type, body = assets.routes[target.path]
                compressed = False
            else:
                match = re.fullmatch(r"/cases/([0-9]{8})\.json", target.path)
                case_path = assets.case_paths.get(match.group(1)) if match else None
                if case_path is None:
                    self.send_error(404)
                    return
                content_type, body = "application/json; charset=utf-8", case_path.read_bytes()
                compressed = True
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            if compressed:
                self.send_header("Content-Encoding", "gzip")
            self.send_header("Cache-Control", "no-store" if target.path in ("/", "/index.html") else "private, max-age=3600")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.end_headers()
            if not head_only:
                self.wfile.write(body)

    return ViewerHandler


def main() -> None:
    assets = load_assets()
    server = ThreadingHTTPServer((HOST, PORT), make_handler(assets))
    server.daemon_threads = True
    print(json.dumps({"url": f"http://{HOST}:{PORT}/", **assets.sizes}), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
