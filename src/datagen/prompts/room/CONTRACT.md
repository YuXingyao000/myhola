# Room generation: execution contract

Create a reusable, convincing interior in Blender. This task produces a complete
environment before any CAD object is selected or placed. There is no reserved
central region, prescribed support height, fixed photographic camera, or target
material. Follow this contract, the room guide, the user brief, and the current
stage instructions. The caller supplies concrete paths and tools at runtime.

## Scope and ownership

- Work in the supplied `template_dir`. Read the reference bundle and its images
  before designing. Source projects, reference images, host code, and reports are
  read-only. Do not modify them to pass a check.
- Read `runtime.reference_mode` before choosing how to use the reference:
  - `adapt`: the host supplies a packaged source project and its asset manifest.
    Preserve its mature geometry, UV maps, material maps, and useful asset groups.
    Make meaningful changes to the activity, spatial arrangement, and related
    lighting. Reopening the project and changing cameras alone is insufficient.
    Treat the packaged source files as read-only; change the loaded scene in
    memory through the builder. Do not copy or rehash the base package again.
  - `quality_only`: the reference provides images and quality notes only. Build
    an independent room for the user brief using individual assets and authored
    construction. Do not locate or import the reference's underlying project,
    or use another complete furnished interior as the candidate. Reference layout,
    room type, and lighting design are examples, not requirements to reproduce.
- Use the supplied asset tool and instructions to search, inspect, and acquire
  individual models, PBR materials, and lighting resources. The runtime gives
  `assets_script`, `asset_cache`, and an optional `asset_catalog`. Preserve existing
  `assets.json` provenance and file digests; merge new records through the tool.
  A catalog or authenticated provider is available only when actually configured.
- All production dependencies must be inside the template. A build must work
  after the whole template directory is moved. Resolve paths from `template_dir`;
  do not depend on the original source directory, shared cache, or current
  working directory. External linked libraries must also resolve inside the
  template. Check image paths after loading the base project.
- Use the supplied Blender and tools. Do not install dependencies, start other
  Codex sessions, call other LLMs, or alter host tools. Work within the caller's
  permitted resources and repair budget. Decide ordinary art choices yourself.

## Authored files

Create these files directly in `template_dir`:

| File | Purpose |
| --- | --- |
| `scene_plan.md` | A short account of the activity, layout, asset choices, surfaces, lighting, and reference use. In `adapt`, identify inherited elements and substantive changes. In `quality_only`, explain the independent design and any asset gaps. Update it when the design changes. |
| `build_scene.py` | Rebuild the complete environment from packaged resources. |
| `views.json` | Six reproducible inspection cameras, specified below. |

The builder runs inside Blender and exposes:

```python
def build_scene(template_dir: Path):
    # Load packaged resources and construct or modify bpy.context.scene.
    # The host ignores the return value and saves the resulting scene.
    ...
```

Importing the module must only define functions and constants. Construction,
loading a base `.blend`, material changes, and light changes belong inside the
function. Use clear helpers and object names. Preserve metric scale: one Blender
unit represents one meter, with right-handed coordinates and Z up. The builder
must produce the same intended scene on a fresh invocation; seed any randomness.
Use Cycles with physically based materials. Preserve useful material maps and
choose coherent color management; when adapting a project, change its settings
only to address an identified need. The host saves
the `.blend` and controls render device, samples, seed, and camera outputs.

Write `views.json` as a JSON object with a `views` array. Each view has exactly
these required fields:

```json
{
  "id": "workbench_context",
  "position": [1.2, 1.8, 1.5],
  "target": [0.0, 0.0, 0.9],
  "vertical_fov_degrees": 50.0,
  "resolution": [1024, 768]
}
```

The coordinates above illustrate the format; choose cameras for the actual room.
Use the six IDs `room_overview`, `room_reverse`, `workbench_context`,
`workbench_detail`, `surface_detail`, and `alternate_view`. Include wide room,
medium activity, and close material views. The historical `workbench_*` IDs mean
the principal activity area and its equipment; they do not require a workbench
or permit reducing a factory brief to a repair corner. The reverse and alternate
views must reveal different spatial relationships rather than slight variations
of the same hero shot. Put cameras in plausible accessible positions, away from walls and
opaque objects. Keep the scene intact across views. Do not hide problems through
camera-dependent geometry, excessive blur, darkness, or cropped framing.

## Tools and evidence

The runtime gives the absolute Blender and `room_tools.py` paths. Tool syntax is:

```text
BLENDER -b --threads 8 --python-exit-code 1 --python ROOM_TOOLS -- build --template_dir TEMPLATE --output_dir BUILD
BLENDER -b --threads 8 --python-exit-code 1 --python ROOM_TOOLS -- render --blend BUILD/room.blend --views_file VIEWS --output_dir PREVIEWS --samples 64 --seed SEED --device DEVICE --gpu_index GPU
BLENDER -b --threads 8 --python-exit-code 1 --python ROOM_TOOLS -- inspect --blend BLEND --output_dir INVENTORY
```

Substitute the supplied paths and values. `build` writes `room.blend`,
`build_report.json`, and `scene_inventory.json`. `render` writes one `{id}.png`
and camera record per view, plus `render_report.json`. `inspect` exports a scene
inventory without changing the input `.blend`.

During creation and repair, put working outputs in uniquely named directories
under `template_dir/previews/`. A separate quick views JSON may contain a subset
of the six views. Before handing over a created candidate, actually build, render,
and open at least a room view, an activity view, and a close surface view. Use the
images to correct observed defects. No minimum number of edits or repeated
renders is required. Never substitute descriptions of intended results for
viewing rendered images. If an actual tool failure prevents this, report it.

The host independently rebuilds and renders the complete six-view set, then asks
for read-only inspection. In that stage, do not change or regenerate evidence.
The host decides when to request a separate repair turn. A `ready` review means
ready for human review; human approval remains pending. Report incomplete work
honestly when required resources are missing or the repair budget is exhausted.
If a defining piece of equipment cannot be obtained or represented convincingly,
name that gap. Do not hide it with an invented primitive proxy or silently change
the requested room type to fit the available assets.
