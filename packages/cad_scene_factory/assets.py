"""搜索素材，将实际文件缓存并复制到模板包；不运行 Blender 或语言模型。"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import sys
import tempfile
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen
import zipfile


PROVIDERS = ("polyhaven", "ambientcg", "sketchfab", "cgbookcase", "local")
KINDS = ("model", "pbr", "hdri")
USER_AGENT = "cad-scene-factory/0.1"
CC0_URL = "https://creativecommons.org/publicdomain/zero/1.0/"


def search_assets(
    provider: str,
    query: str,
    kind: str,
    limit: int = 8,
    catalog: Path | None = None,
) -> list[dict]:
    """只搜索指定来源；None 表示上游没有提供该属性，不凭名称猜尺寸或贴图。"""
    if provider not in PROVIDERS or kind not in KINDS or not 1 <= limit <= 24:
        raise ValueError("Check provider, kind and limit (1–24)")
    if provider == "polyhaven":
        asset_type = {"model": "models", "pbr": "textures", "hdri": "hdris"}[kind]
        data = _get_json("https://api.polyhaven.com/assets?" + urlencode({"type": asset_type}))
        matches = [(key, item) for key, item in data.items() if _matches(query, key, item)]
        matches.sort(key=lambda item: (-item[1].get("download_count", 0), item[0]))
        return [_polyhaven_info(key, item) for key, item in matches[:limit]]
    if provider == "ambientcg":
        asset_type = {"model": "3DModel", "pbr": "Material", "hdri": "HDRI"}[kind]
        data = _ambientcg_query(q=query, type=asset_type, limit=limit)
        return [_ambientcg_info(item) for item in data["foundAssets"]]
    if provider == "sketchfab":
        if kind != "model":
            raise ValueError("Sketchfab integration searches model assets only")
        params = {"type": "models", "q": query, "downloadable": "true", "count": limit}
        data = _get_json("https://api.sketchfab.com/v3/search?" + urlencode(params))
        return [_sketchfab_info(item) for item in data["results"]]

    # CGBookcase 没有在这里实现网页抓取；它与自有素材使用同一个本地目录格式。
    records = _read_catalog(catalog)
    matches = [item for item in records if item["provider"] == provider
               and item["kind"] == kind and _matches(query, item["asset_id"], item)]
    return [_catalog_info(item) for item in matches[:limit]]


def fetch_asset(
    provider: str,
    asset_id: str,
    template_dir: Path,
    cache_dir: Path,
    resolution: str = "2k",
    catalog: Path | None = None,
    max_download_mb: int = 2048,
) -> dict:
    """获取一个素材版本，核对缓存摘要，再更新 build.py 使用的平铺文件清单。

    分辨率仅控制 Poly Haven / ambientCG 的下载选择；Sketchfab 和本地目录
    保留已有文件。缓存不自动更新，已有模板文件也不会被不同内容静默覆盖。
    """
    if provider not in PROVIDERS or not re.fullmatch(r"[A-Za-z0-9_-]+", asset_id):
        raise ValueError("Unknown provider or unsafe asset_id")
    resolution = resolution.lower()
    if not re.fullmatch(r"[1-9][0-9]*k", resolution) or max_download_mb <= 0:
        raise ValueError("resolution must look like 2k; max_download_mb must be positive")
    variant = resolution if provider in {"polyhaven", "ambientcg"} else "original"
    cache_entry = Path(cache_dir).resolve() / provider / asset_id / variant
    template_dir = Path(template_dir).resolve()
    cache_hit = cache_entry.exists()

    # 1. 缓存缺失时下载/复制到临时目录；全部成功后才发布这个缓存版本。
    if not cache_hit:
        cache_entry.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".fetch-", dir=cache_entry.parent) as temporary:
            staging = Path(temporary) / "entry"
            payload = staging / "files"
            payload.mkdir(parents=True)
            metadata = _acquire(provider, asset_id, resolution, payload, catalog,
                                max_download_mb * 1024 * 1024)
            if not metadata.get("license"):
                raise ValueError("Asset license is unknown; add its actual license before use")
            files = [{"path": path.relative_to(payload).as_posix(), "sha256": _sha256(path)}
                     for path in sorted(payload.rglob("*")) if path.is_file()]
            if not files:
                raise ValueError("Asset contains no files")
            metadata["fetched_at"] = datetime.now(timezone.utc).isoformat()
            _write_json(staging / "metadata.json", {"asset": metadata, "files": files})
            staging.rename(cache_entry)

    # 2. 即使缓存命中也重新读文件核摘要，避免把损坏的缓存传播到多个模板。
    cached = json.loads((cache_entry / "metadata.json").read_text(encoding="utf-8"))
    metadata = cached["asset"]
    prefix = Path("assets") / provider / asset_id / variant
    records = []
    for item in cached["files"]:
        source = _inside(cache_entry / "files", item["path"])
        if _sha256(source) != item["sha256"]:
            raise ValueError(f"Cache SHA-256 mismatch: {provider}/{asset_id}/{variant}/{item['path']}")
        relative_path = prefix / item["path"]
        destination = _inside(template_dir, relative_path.as_posix())
        if destination.exists() and _sha256(destination) != item["sha256"]:
            raise ValueError(f"Template file has different content: {relative_path}")
        records.append({
            "path": relative_path.as_posix(), "sha256": item["sha256"],
            "provider": provider, "asset_id": asset_id, "variant": variant,
            "source_url": metadata["source_url"], "license": metadata["license"],
            "license_url": metadata.get("license_url"), "author": metadata.get("author"),
            "author_url": metadata.get("author_url"), "asset_name": metadata.get("name"),
        })

    # 3. 复制真实文件（不用软链接），合并清单，保留其他素材及用户附加字段。
    for item, record in zip(cached["files"], records):
        destination = template_dir / record["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cache_entry / "files" / item["path"], destination)
        if _sha256(destination) != record["sha256"]:
            raise ValueError(f"Copied file SHA-256 mismatch: {record['path']}")
    manifest_path = template_dir / "assets.json"
    existing = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else []
    manifest = {item["path"]: item for item in existing}
    for record in records:
        manifest[record["path"]] = {**manifest.get(record["path"], {}), **record}
    _write_json(manifest_path, sorted(manifest.values(), key=lambda item: item["path"]))
    return {"asset": metadata, "cache_hit": cache_hit, "variant": variant,
            "template_dir": str(template_dir), "files": records}


def _acquire(provider, asset_id, resolution, payload, catalog, byte_budget):
    """供应商分支只负责选择文件；下载、解包与复制规则由下面的公共函数完成。"""
    if provider in {"local", "cgbookcase"}:
        records = _read_catalog(catalog)
        item = next((item for item in records
                     if item["provider"] == provider and item["asset_id"] == asset_id), None)
        if item is None:
            raise ValueError(f"Asset is absent from local catalog: {provider}/{asset_id}")
        metadata = _catalog_info(item)
        if not metadata["license"]:
            raise ValueError("Local catalog must record the actual asset license")
        source_dir = (Path(catalog).resolve().parent / item["source_dir"]).resolve()
        if not source_dir.is_dir():
            raise ValueError("Local catalog source_dir must be an existing directory")
        for source in sorted(source_dir.rglob("*")):
            if source.is_symlink():
                raise ValueError("Local asset directory must contain real files, not symlinks")
            if source.is_file():
                byte_budget -= source.stat().st_size
                if byte_budget < 0:
                    raise ValueError("Local asset exceeds max_download_mb")
                destination = payload / source.relative_to(source_dir)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
        return metadata

    if provider == "polyhaven":
        info = _get_json(f"https://api.polyhaven.com/info/{asset_id}")
        metadata = _polyhaven_info(asset_id, info)
        files = _get_json(f"https://api.polyhaven.com/files/{asset_id}")
        key, format_name = ("hdri", "hdr") if info["type"] == 0 else ("blend", "blend")
        selected = files.get(key, {}).get(resolution, {}).get(format_name)
        if selected is None:
            raise ValueError(f"Poly Haven has no {key}/{resolution}/{format_name} for {asset_id}")
        # include 的相对路径正是 .blend 内引用的依赖路径，必须保留目录结构。
        filename = Path(urlsplit(selected["url"]).path).name
        downloads = [(filename, selected), *selected.get("include", {}).items()]
        if info["type"] == 1:
            metadata["maps"] = [key for key in files if key not in {"blend", "gltf", "mtlx", "usd"}]
    elif provider == "ambientcg":
        found = _ambientcg_query(id=asset_id, limit=1)["foundAssets"]
        if not found:
            raise ValueError(f"ambientCG asset not found: {asset_id}")
        info = found[0]
        metadata = _ambientcg_info(info)
        choices = _ambientcg_downloads(info)
        choices = [item for item in choices if re.search(
            rf"(?:^|_){re.escape(resolution)}(?:[-_.]|$)", item["fileName"], re.IGNORECASE)]
        # 材质优先无损 PNG 包；HDRI/模型保留供应商相应的原始格式。
        choices.sort(key=lambda item: ("PNG" not in item["fileName"].upper(), item["fileName"]))
        if not choices:
            raise ValueError(f"ambientCG has no {resolution} download for {asset_id}")
        selected = choices[0]
        downloads = [(selected["fileName"], {"url": selected["downloadLink"], "size": selected["size"]})]
    else:
        token = os.environ.get("SKETCHFAB_ACCESS_TOKEN")
        if not token:
            raise ValueError("Set SKETCHFAB_ACCESS_TOKEN outside prompts before a Sketchfab download")
        info = _get_json(f"https://api.sketchfab.com/v3/models/{asset_id}")
        metadata = _sketchfab_info(info)
        if not metadata["license"] or not metadata["license_url"]:
            raise ValueError("Sketchfab model has no identifiable license")
        data = _get_json(f"https://api.sketchfab.com/v3/models/{asset_id}/download",
                         headers={"Authorization": f"Bearer {token}"})
        if "gltf" not in data:
            raise ValueError("Sketchfab did not provide a downloadable glTF archive")
        downloads = [("model.zip", data["gltf"])]

    if sum(item.get("size", 0) for _, item in downloads) > byte_budget:
        raise ValueError("Asset exceeds max_download_mb; select a smaller resolution or raise the explicit limit")
    metadata["download_files"] = [name for name, _ in downloads]
    unpacked_limit = byte_budget
    # 下载链接只在内存中使用，尤其不保存 Sketchfab 短期签名 URL。
    for filename, item in downloads:
        destination = _inside(payload, filename)
        size = _download(item["url"], destination, byte_budget)
        byte_budget -= size
        if item.get("size") is not None and size != item["size"]:
            raise ValueError(f"Provider file size mismatch: {filename}")
        if item.get("md5") and _file_digest(destination, "md5") != item["md5"]:
            raise ValueError(f"Provider checksum mismatch: {filename}")
        if destination.suffix.lower() == ".zip":
            _extract_zip(destination, payload, unpacked_limit)
            destination.unlink()
    return metadata


def _polyhaven_info(asset_id, item):
    return {
        "provider": "polyhaven", "provider_name": "Poly Haven", "asset_id": asset_id,
        "name": item["name"], "kind": {0: "hdri", 1: "pbr", 2: "model"}[item["type"]],
        "source_url": f"https://polyhaven.com/a/{asset_id}", "license": "CC0-1.0",
        "license_url": CC0_URL, "author": list(item.get("authors", {})) or None,
        "preview_url": item.get("thumbnail_url"), "maps": None,
        "dimensions": {"values": item["dimensions"], "unit": "unknown"} if item.get("dimensions") else None,
    }


def _ambientcg_info(item):
    previews = item.get("previewImage", {})
    dimensions = [item.get("dimension" + axis) for axis in "XYZ"]
    return {
        "provider": "ambientcg", "provider_name": "ambientCG", "asset_id": item["assetId"],
        "name": item.get("displayName", item["assetId"]),
        "kind": {"Material": "pbr", "HDRI": "hdri", "3DModel": "model"}.get(item["dataType"], item["dataType"]),
        "source_url": item.get("shortLink", f"https://ambientcg.com/a/{item['assetId']}"),
        "license": "CC0-1.0", "license_url": CC0_URL, "author": None,
        "preview_url": previews.get("512-PNG") or next(iter(previews.values()), None),
        "maps": item.get("maps"),
        "dimensions": {"values": dimensions, "unit": "unknown"} if any(dimensions) else None,
    }


def _sketchfab_info(item):
    license_info = item.get("license") or {}
    previews = sorted(item.get("thumbnails", {}).get("images", []),
                      key=lambda image: abs(image.get("width", 0) - 512))
    return {
        "provider": "sketchfab", "provider_name": "Sketchfab", "asset_id": item["uid"],
        "name": item["name"], "kind": "model", "source_url": item["viewerUrl"],
        "license": license_info.get("label"), "license_url": license_info.get("url"),
        "author": item.get("user", {}).get("displayName") or item.get("user", {}).get("username"),
        "author_url": item.get("user", {}).get("profileUrl"),
        "license_requirements": license_info.get("requirements"),
        "preview_url": previews[0]["url"] if previews else None, "maps": None, "dimensions": None,
    }


def _catalog_info(item):
    return {key: item.get(key) for key in (
        "provider", "asset_id", "name", "kind", "source_url", "license", "license_url",
        "author", "author_url", "preview_url", "maps", "dimensions",
    )}


def _read_catalog(catalog):
    if catalog is None:
        raise ValueError("This provider needs --catalog with already downloaded local assets; no online scraping is implemented")
    return json.loads(Path(catalog).read_text(encoding="utf-8"))


def _matches(query, asset_id, item):
    text = " ".join((asset_id, item.get("name", ""), item.get("description", ""),
                     " ".join(item.get("tags", [])))).lower()
    return all(word in text for word in query.lower().split())


def _ambientcg_query(**parameters):
    parameters["include"] = "downloadData,previewData,displayData,dimensionsData,mapData"
    return _get_json("https://ambientcg.com/api/v2/full_json?" + urlencode(parameters))


def _ambientcg_downloads(info):
    return [item for folder in info.get("downloadFolders", {}).values()
            for category in folder.get("downloadFiletypeCategories", {}).values()
            for item in category.get("downloads", [])]


def _open_url(url, headers=None):
    """错误只报告域名和状态，不把授权头或临时下载链接带入日志。"""
    if urlsplit(url).scheme != "https":
        raise ValueError("Remote asset requests must use HTTPS")
    request = Request(url, headers={"User-Agent": USER_AGENT, **(headers or {})})
    try:
        return urlopen(request, timeout=60)
    except HTTPError as error:
        raise RuntimeError(f"{urlsplit(url).hostname}: HTTP {error.code}") from None
    except (URLError, TimeoutError):
        raise RuntimeError(f"{urlsplit(url).hostname}: network request failed") from None


def _get_json(url, headers=None):
    with _open_url(url, headers) as response:
        return json.load(response)


def _download(url, destination, byte_budget):
    destination.parent.mkdir(parents=True, exist_ok=True)
    size = 0
    print(f"下载素材文件：{destination.name}", file=sys.stderr, flush=True)
    with _open_url(url) as response, destination.open("wb") as output:
        for chunk in iter(lambda: response.read(1024 * 1024), b""):
            size += len(chunk)
            if size > byte_budget:
                raise ValueError("Download exceeds max_download_mb")
            output.write(chunk)
    return size


def _inside(root, relative_path):
    """上游依赖路径、ZIP 条目与清单路径都只能落在指定目录内部。"""
    root = root.resolve()
    path = Path(relative_path)
    if not path.parts or path.is_absolute() or "\\" in relative_path or ":" in path.parts[0]:
        raise ValueError(f"Unsafe asset path: {relative_path}")
    destination = (root / path).resolve()
    if not destination.is_relative_to(root):
        raise ValueError(f"Asset path escapes its directory: {relative_path}")
    return destination


def _extract_zip(archive_path, destination, max_bytes):
    with zipfile.ZipFile(archive_path) as archive:
        if sum(item.file_size for item in archive.infolist()) > max_bytes:
            raise ValueError("Unpacked asset exceeds max_download_mb")
        paths = set()
        for item in archive.infolist():
            target = _inside(destination, item.filename)
            mode = stat.S_IFMT(item.external_attr >> 16)
            if mode not in {0, stat.S_IFREG, stat.S_IFDIR}:
                raise ValueError("Archive contains a symlink or special file")
            if target in paths or (target.exists() and not item.is_dir()):
                raise ValueError("Archive contains a duplicate or conflicting file")
            paths.add(target)
        for item in archive.infolist():
            target = _inside(destination, item.filename)
            if item.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(item) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)


def _file_digest(path, algorithm):
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256(path):
    return _file_digest(path, "sha256")


def _write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    search = commands.add_parser("search", help="查询一个来源的候选，不下载素材")
    fetch = commands.add_parser("fetch", help="获取素材，缓存并复制到模板包")
    for subparser in (search, fetch):
        subparser.add_argument("--provider", choices=PROVIDERS, required=True)
        subparser.add_argument("--catalog", type=Path)
    search.add_argument("--query", required=True)
    search.add_argument("--kind", choices=KINDS, required=True)
    search.add_argument("--limit", type=int, default=8)
    fetch.add_argument("--asset_id", required=True)
    fetch.add_argument("--template_dir", type=Path, required=True)
    fetch.add_argument("--cache_dir", type=Path, required=True)
    fetch.add_argument("--resolution", default="2k")
    fetch.add_argument("--max_download_mb", type=int, default=2048)
    args = vars(parser.parse_args())
    command = args.pop("command")
    try:
        result = search_assets(**args) if command == "search" else fetch_asset(**args)
    except (ValueError, RuntimeError, OSError, KeyError, zipfile.BadZipFile) as error:
        print(json.dumps({"error": str(error)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(1)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
