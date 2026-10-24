"""Validate loopback delivery without remeasuring or modifying scientific data."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import http.client
import json
from pathlib import Path
import time
from urllib.parse import urlsplit

from playwright.sync_api import sync_playwright

from serve_viewer import HOST, PORT, load_assets


HERE = Path(__file__).resolve().parent
CHROMIUM = "/root/.cache/ms-playwright/chromium-1155/chrome-linux/chrome"
URL = f"http://{HOST}:{PORT}/"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def request(path: str, method: str = "GET") -> tuple[int, dict[str, str], bytes]:
    connection = http.client.HTTPConnection(HOST, PORT, timeout=30)
    try:
        connection.request(method, path)
        response = connection.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        connection.close()


def main() -> None:
    start = time.monotonic()
    source_names = ("app.js", "template.html", "failure_viewer.html", "data/manifest.json",
                    "metrics.csv", "summary.json")
    before = {name: digest(HERE / name) for name in source_names}
    assets = load_assets()
    checks = []
    plotly_sha = json.loads((HERE / "validation/build.json").read_text())["plotly_sha256"]
    for route, (content_type, body) in assets.routes.items():
        status, headers, actual = request(route)
        assert status == 200, (route, status)
        if route == "/plotly.min.js":
            # The existing browser-test Python and serving Python have separate
            # Plotly installations. Check the frozen original bundle's hash.
            assert hashlib.sha256(actual).hexdigest() == plotly_sha, route
            assets.sizes["plotly_bytes"] = len(actual)
        else:
            assert actual == body, route
        assert headers["Content-Type"] == content_type, (route, headers)
    assets.sizes["maximum_initial_bytes"] = sum(assets.sizes[key] for key in
        ("entry_bytes", "app_bytes", "plotly_bytes", "largest_case_bytes"))
    checks.append({"check": "allowed_static_assets_match_generated_bytes", "count": len(assets.routes)})

    case_hashes = {}
    for case_id, source in assets.case_paths.items():
        status, headers, body = request(f"/cases/{case_id}.json")
        assert status == 200 and headers.get("Content-Encoding") == "gzip", (case_id, status, headers)
        assert body == source.read_bytes(), case_id
        value = json.loads(gzip.decompress(body))
        assert value["id"] == case_id, case_id
        case_hashes[case_id] = hashlib.sha256(body).hexdigest()
    checks.append({"check": "all_cases_serve_exact_existing_gzip_and_correct_identity", "count": len(case_hashes)})
    rejected = ("/data/manifest.json", "/README.md", "/cases/", "/cases/00000000.json", "/app.js?x=1",
                "/../README.md", "/%2e%2e/README.md", "/cases/../app.js", "/serve_viewer.py")
    for route in rejected:
        status, _, _ = request(route)
        assert status in (403, 404), (route, status)
    checks.append({"check": "unlisted_paths_and_traversal_rejected", "count": len(rejected)})
    status, headers, body = request("/", "HEAD")
    assert status == 200 and not body and int(headers["Content-Length"]) == assets.sizes["entry_bytes"]

    errors, requests, transferred = [], [], []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=CHROMIUM, headless=True,
            args=["--no-sandbox", "--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
        page = browser.new_page(viewport={"width": 1680, "height": 1120}, device_scale_factor=1)
        page.add_init_script("Object.defineProperty(window,'DecompressionStream',{value:undefined,configurable:true});")
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on("console", lambda message: errors.append(message.text) if message.type == "error" else None)
        page.on("request", lambda request: requests.append(request.url))
        page.on("response", lambda response: transferred.append({"url": response.url,
            "bytes": int(response.headers.get("content-length", "0")), "status": response.status}))
        load_start = time.monotonic()
        page.goto(URL, wait_until="load", timeout=90000)
        page.wait_for_function("window.viewer && !viewer.state.loading", timeout=90000)
        page.evaluate("viewer.ready")
        state = page.evaluate("({id:viewer.state.id,error:viewer.state.error,source:viewer.state.source,n:MANIFEST.cases.length,traceCount:document.getElementById('geometry').data.length,decompress:typeof DecompressionStream})")
        assert state["error"] is None and state["source"] == "fitted" and state["n"] == 316 and state["traceCount"] > 2, state
        assert state["decompress"] == "undefined", state
        initial_case_requests = [url for url in requests if "/cases/" in url]
        assert len(initial_case_requests) == 1, initial_case_requests
        initial_bytes = sum(item["bytes"] for item in transferred)
        assert initial_bytes < 6_000_000, initial_bytes
        checks.append({"check": "browser_initial_load_without_js_decompression", "seconds": time.monotonic() - load_start,
                       "state": state, "case_requests": initial_case_requests, "bytes": initial_bytes})

        first_id = state["id"]
        page.locator("#next").click()
        page.wait_for_function("id=>viewer.state.id!==id && !viewer.state.loading", arg=first_id, timeout=90000)
        second_id = page.evaluate("viewer.state.id")
        assert len([url for url in requests if "/cases/" in url]) == 2
        page.locator("#previous").click()
        page.wait_for_function("id=>viewer.state.id===id && !viewer.state.loading", arg=first_id, timeout=90000)
        assert len([url for url in requests if "/cases/" in url]) == 2
        assert page.evaluate("viewer.cacheSize") == 2
        checks.append({"check": "next_fetches_one_case_previous_reuses_memory_cache", "ids": [first_id, second_id], "case_request_count": 2})

        page.locator("#search").fill("00024146")
        page.wait_for_function("viewer.state.id==='00024146'&&!viewer.state.loading", timeout=90000)
        page.evaluate("viewer.setSelection(6,null,true)")
        assert "丢" in page.locator("#attempt-table").inner_text()
        page.locator("#distribution-hist").click()
        page.wait_for_function("viewer.state.distribution==='hist'")
        assert page.evaluate("viewer.state.error") is None
        checks.append({"check": "known_failure_face_and_histogram_render", "id": "00024146", "face": 6})
        assert all(urlsplit(url).hostname == HOST and urlsplit(url).port == PORT for url in requests), requests
        assert not errors, {"errors": errors, "requests": requests, "responses": transferred}
        browser.close()

    after = {name: digest(HERE / name) for name in source_names}
    assert before == after
    assert all(digest(assets.case_paths[case_id]) == sha for case_id, sha in case_hashes.items())
    report = {"validated_utc": datetime.now(timezone.utc).isoformat(), "url": URL, "passed": True,
              "elapsed_seconds": time.monotonic() - start, "size_budget": assets.sizes, "checks": checks,
              "browser_errors": errors, "browser_request_count": len(requests),
              "browser_case_requests": dict(Counter(urlsplit(url).path for url in requests if "/cases/" in url)),
              "existing_source_sha256_unchanged": after, "case_count_unchanged": len(case_hashes),
              "delivery_source_sha256": {name: digest(HERE / name) for name in ("serve_viewer.py", "validate_http_delivery.py")},
              "scientific_scope": "Existing payload bytes and identities checked; no distances, replay, or evaluation recomputed."}
    output = HERE / "validation/http_delivery.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"passed": True, "report": str(output), "initial_bytes": initial_bytes,
                      "case_count": len(case_hashes), "seconds": report["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
