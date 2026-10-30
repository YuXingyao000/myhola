# Asset Tools

`assets.py` is an ordinary Python script. It does not start Blender or call a language model. It has two responsibilities: find candidate assets and put the selected files into a self-contained template package. The Codex agent creating the template remains responsible for scene design, material adjustments, and image inspection.

## 1. Connected sources

| provider | Search and retrieval | Current scope |
| --- | --- | --- |
| `polyhaven` | Official API for models, PBR materials, and HDRIs | No account required. At the requested resolution, fetch the `.blend` file and all texture dependencies for models and PBR materials, or the `.hdr` file for HDRIs. |
| `ambientcg` | Official API for models, PBR materials, and HDRIs | No account required. Use the requested resolution; prefer PNG archives for materials and select other asset types from the files actually available. |
| `sketchfab` | Official model search and Download API | Search does not require login. Downloads require an externally configured OAuth access token; fetch the glTF archive and its textures. |
| `cgbookcase` | Previously downloaded CGBookcase assets registered in a local catalog | **Online search and download are not implemented.** Do not guess undocumented website endpoints. |
| `local` | Owned or previously acquired assets registered in a local catalog | Retrieve files using the source, license, author, and directory recorded in the catalog. |

Each invocation queries only the explicitly selected provider. Provider failures, missing authentication, or an unavailable resolution produce an error. The tool does not silently switch providers or substitute a procedural scene. `search` does not download models or textures.

Verification completed on 2026-10-15: the public search endpoints of the first three providers were exercised. The 1k `aerodynamics_workshop` HDRI was fetched from Poly Haven (1,488,808 bytes), its size and MD5 were checked against the provider's values, and its actual SHA-256 was recorded. With network calls disabled, a second template received the same file from the cache. The 1k `painted_metal_shutter` `.blend` file and four textures were also fetched (5,135,621 bytes in total), with all digests and dependency paths checked. Its material was appended in Blender 5.2.2 and successfully packed by `build.py`. Rebuilding a copied template also succeeded after the original directory was temporarily moved out of the way. Manifest merging, rejection of a corrupted cache, and ZIP extraction were checked with small local fixtures. Online file downloads from ambientCG and authenticated downloads from Sketchfab have **not** been verified with real downloads.

## 2. Search for candidates

The commands below use placeholder paths. Replace them with the script, template, cache, and catalog paths supplied by the caller's runtime configuration. Do not execute the placeholders literally or guess machine-specific paths.

```bash
python /path/to/cad_scene_factory/assets.py search \
  --provider polyhaven --kind pbr --query "concrete" --limit 8

python /path/to/cad_scene_factory/assets.py search \
  --provider sketchfab --kind model --query "used drill" --limit 8
```

`kind` is `model`, `pbr`, or `hdri`. Poly Haven uses keyword matching within its API catalog; the other online providers use their search endpoints. Rankings are not necessarily comparable across providers. English search terms generally work better for these libraries.

The result is a JSON list. Each candidate includes at least:

- `provider`, `asset_id`: stable identifiers used when fetching the asset.
- `name`, `kind`, `source_url`: the name, type, and original asset page.
- `preview_url`: the preview URL supplied by the provider. Returning this URL does not mean the agent has inspected the image.
- `license`, `license_url`, `author`: license and author information supplied upstream.
- `maps`, `dimensions`: texture maps and dimensions known upstream; `null` when unavailable.

Dimensions currently retain the provider's original values and are marked `unit: "unknown"`. Do not interpret these numbers directly as meters in Blender. Scene code should inspect the imported model's actual dimensions and scale it according to the scene's scale convention. Sketchfab search results may provide only a license name; before downloading, the tool reads the model details again to obtain the specific license URL.

Search results are not quality scores. Inspect previews for shape, surface detail, textures, proportions, and the asset's intended role in the final image before deciding to download it. Prefer credible assets for large support surfaces, walls, floors, and prominent props. This does not require downloading a scanned model for every small, distant nut.

## 3. Fetch assets into a template package

```bash
python /path/to/cad_scene_factory/assets.py fetch \
  --provider polyhaven --asset_id painted_metal_shutter \
  --resolution 2k \
  --template_dir /path/to/templates/workshop_003 \
  --cache_dir /path/to/asset_cache
```

This step downloads the actual asset files. The default `--max_download_mb 2048` limits both the total download size of one asset and the extracted size of each ZIP archive. Local assets are also subject to the size limit. This budget can be adjusted explicitly. The tool does not automatically simplify high-polygon models or downsample images.

Processing order:

1. Check the cache. If absent, fetch the files and save that cache version only after every file succeeds.
2. Verify the SHA-256 of every cached file, including on cache hits.
3. Copy the actual files into the template, preserve model-to-texture directory relationships, and verify the copies.
4. Merge `assets.json` at the template root, preserving other assets and user-added fields.

Example directory layout:

```text
workshop_003/
├── build_scene.py
├── build_material.py
├── assets.json
└── assets/
    └── polyhaven/
        └── painted_metal_shutter/
            └── 2k/
                ├── painted_metal_shutter_2k.blend
                └── textures/
                    └── ...
```

`2k` is the asset variant directory. Sketchfab and local assets use `original`, retaining their existing file resolution; `--resolution` does not resample them. Multiple resolutions can coexist.

The template contains actual files, not symlinks to a shared cache. **Room builders must resolve asset paths from the `template_dir` argument passed to `build_scene(template_dir)`**, for example `template_dir / "assets/polyhaven/..."`. Do not use the current working directory, an absolute cache path, or assume the script lives next to the supplied assets. Legacy template builders without this argument resolve paths from `Path(__file__).resolve().parent`. Append objects or materials from Poly Haven `.blend` files rather than creating external library links. Import Sketchfab's `scene.gltf` through Blender. Keep model files, textures, and supporting binary files inside the package.

`assets.json` supports both existing formats:

- A **flat list of per-file records**, used by the original `build.py`. When no manifest exists, `fetch` creates this format.
- A **room manifest object containing a `files` list**, such as `{"reference_id": "...", "source": {...}, "files": [...]}`. `fetch` merges into `files` and preserves every other top-level field; it does not replace the room's source metadata with the newly fetched asset's source.

In either format, merging uses each file's relative `path`. Existing records for other files and custom fields on matching records remain intact. The tool refreshes the matching file's checksum and acquisition fields from the verified cache. Each fetched record includes the actual `sha256`, `provider`, `asset_id`, `variant`, source page, license, and known author. The build stage rechecks file digests and packs resources such as textures actually used by the scene. Do not flatten a room manifest or rewrite its reference/source metadata after fetching an additional asset.

Cached variants do not automatically follow upstream updates. If a template file with the same name has been edited manually, the tool refuses to overwrite it from the cache. Preserve the edited version and record its new digest. A corrupted cache also produces an error. The current script is designed for single-process use: do not run concurrent `fetch` operations against the same template or cache variant.

## 4. Sketchfab authentication and licensing

Complete the initial OAuth authorization externally, then set the resulting access token in the `SKETCHFAB_ACCESS_TOKEN` environment variable inherited by the tool process. Do not put credentials in prompts, the repository, or template packages. Do not print the full environment variable list.

The initial implementation only reads an access token. **Browser login, application registration, and automatic refresh-token renewal are not implemented.** Update the external configuration when the token expires. A website login session or an ordinary API token is not an OAuth access token. No existing account credentials have been read, checked, or used to verify the current code.

The Download API returns short-lived signed URLs. The tool uses them only in memory to download and cache actual files. The manifest records the public asset page, not a temporary download URL or an Authorization header. Network errors report only the domain and status.

“Free to download” does not mean unrestricted modification or redistribution. Check the license of each Sketchfab asset individually. The tool preserves the actual license and does not label the entire platform CC0. Prefer CC0 when creating open template packages. For assets requiring attribution, retain the name, author, source, and license. Before using assets with noncommercial, no-derivatives, share-alike, or similar conditions, determine whether they fit the template and its intended release. The tool does not relicense assets, and a cache hit does not replace a license check.

## 5. Local catalogs and CGBookcase

Register previously downloaded and extracted assets in a JSON list. The example below shows the structure: replace the name, ID, and directory with the actual asset values, and obtain license information from the asset's real source.

```json
[
  {
    "provider": "cgbookcase",
    "asset_id": "my_downloaded_material",
    "name": "Actual name of the downloaded material",
    "kind": "pbr",
    "source_dir": "materials/my_downloaded_material",
    "source_url": "https://www.cgbookcase.com/textures/actual-asset-page",
    "license": "CC0-1.0",
    "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
    "author": "Dorian Zgraggen",
    "tags": ["concrete", "worn"],
    "preview_url": null,
    "maps": ["base_color", "roughness", "normal"],
    "dimensions": null
  }
]
```

`source_dir` may be absolute or relative to the catalog file. The directory must already be extracted and contain only the actual files needed by that asset. The tool does not execute scripts inside it or follow symlinks to copy other directories. Use English letters, digits, underscores, or hyphens in `asset_id`. Use `provider: "local"` for owned assets and record the actual author, license, and source. Leave unknown information as `null`, but establish the license before fetching.

```bash
python /path/to/cad_scene_factory/assets.py search \
  --provider cgbookcase --kind pbr --query concrete \
  --catalog /path/to/local_assets/catalog.json

python /path/to/cad_scene_factory/assets.py fetch \
  --provider cgbookcase --asset_id my_downloaded_material \
  --catalog /path/to/local_assets/catalog.json \
  --template_dir /path/to/templates/workshop_003 \
  --cache_dir /path/to/asset_cache
```

CGBookcase/local requests without a catalog fail explicitly. Online integration can be added later; do not describe it as already implemented.

## 6. Official interfaces and license references

The following interfaces and terms were checked on 2026-10-15:

- [Poly Haven API and access requirements](https://polyhaven.com/our-api): at the time of verification, the API was free and required no key. It requires an identifiable software User-Agent and visible attribution of the asset source to Poly Haven. The tool uses `cad-scene-factory/0.1` and retains source names and links in its results.
- [Poly Haven API terms](https://github.com/Poly-Haven/Public-API/blob/master/ToS.md) and [asset license](https://polyhaven.com/license): assets use CC0; API service terms are separate from asset licensing. Website content such as example renders is not automatically covered by the assets' CC0 license. This tool does not package preview images as training assets.
- [ambientCG API](https://docs.ambientcg.com/api/v2/full_json/) and [license](https://docs.ambientcg.com/license/): use the official interface to retrieve metadata and download links; assets use CC0.
- [Sketchfab OAuth](https://sketchfab.com/developers/oauth), [model downloads](https://sketchfab.com/developers/download-api/downloading-models), and [developer guidelines](https://sketchfab.com/developers/download-api/guidelines): respect user authorization and temporary download URLs, and retain per-asset license information.
- [CGBookcase author's description](https://www.dorianzgraggen.com/projects/cgbookcase/): the PBR texture library uses CC0. This tool currently accesses previously downloaded files through the local catalog.
