# A working woodworking studio

Create a believable small woodworking studio used for furniture, joinery, or
similar craft work. Show an ordinary working day with a project temporarily
paused. This is a complete, maintained working interior that can be photographed
from several directions. No people are required.

This task uses `quality_only`. The supplied Shed images illustrate convincing
construction, surfaces, objects, spatial relationships, and lighting. Build an
independent studio from individual assets and authored construction; do not locate
or import the reference project or substitute another complete furnished room.
Its layout and furnishings are not requirements. No target CAD component is
selected or placed, and no product-display region should be reserved.

## Activity, scale, and spatial organization

Choose a coherent woodworking activity and describe it briefly in `scene_plan.md`.
Let a recognizable project, its construction stage, and the tools actually
available explain the room. Choose dimensions, equipment, furniture, and their
arrangement yourself. Avoid a generic workshop decorated with unrelated tools
or a lumber showroom without evidence of work.

Use metric, human-scale proportions. Relate stock storage, preparation, active
joinery or assembly, and finished work to a plausible sequence. These functions
can share compact areas. Allow someone to reach tools, hold a workpiece, move
long stock, operate equipment, and open storage. Account for material infeed and
outfeed when choosing machines. Keep circulation and door access usable.

Organize objects into purposeful, uneven groups: tools near the operation they
serve, appropriate clamps near assembly, offcuts separated from usable stock,
and unfinished work supported in a credible position. These are relationships
to consider, not a fixed inventory. Leave usable surfaces and avoid either
uniformly staged rows or random debris everywhere. Room depth should come from
meaningful relationships between working areas, storage, architecture, and access.

Build a continuous enclosure with floor, walls, ceiling or roof structure, doors,
and appropriate windows. Include visible construction details and service routes
where the cameras expose them. Give openings a deliberate exterior or lighting
treatment. All six views must belong to the same complete interior.

## Equipment, stock, and asset choices

Use the supplied asset tools to inspect and acquire individual benches, tools,
machines, storage, PBR surfaces, and lighting resources. Prefer free assets whose
actual licenses permit the intended use and packaging. Record sources, authors,
licenses, and dependencies through the prescribed provenance workflow.
Authenticated providers are available only when access is configured.

Prominent equipment needs credible construction, scale, controls, supports,
working surfaces, and connections. Preserve useful UVs, material maps, fittings,
and attached parts. Verify contact with benches, floors, and racks after import.
Simple architecture, plain stock, offcuts, and secondary objects may be procedural;
major tools should not become decorated boxes. If defining equipment cannot be
obtained or built convincingly, identify the gap rather than silently changing
the requested activity.

Consider how the selected operations produce and collect dust. Powered equipment
that calls for extraction should have a plausible collector, hose connection,
or shared service arrangement, with routes that respect movement and clearance.
Represent only the system the activity requires; do not add disconnected hoses
as decoration. Power leads should also terminate plausibly.

## Materials and evidence of use

Distinguish rough stock, freshly worked wood, older bench surfaces, and finished
pieces. Use believable grain direction and scale; end grain, cut edges, and any
plywood layers should agree with construction. Avoid identical wood textures on
every object or arbitrary grain across adjoining faces. Preserve distinctions
among timber, painted and bare metal, rubber, plastic, masonry, and other finishes.
Inspect imported PBR detail before adding wear.

Show a maintained studio with localized traces of work. Shavings near a planing
position differ from fine sanding dust; cut ends, tool marks, clamp impressions,
small glue residues, and handling wear should follow actual operations. Choose
only the traces supported by the activity. A recently cleared working patch,
cleaner walking route, and modest accumulation in sheltered locations can coexist.
Do not cover the whole floor or every surface with the same dust, scratches,
brown noise, or excessive bump. Large floor, wall, and bench surfaces must hold
up in close inspection as well as wide views.

## Lighting and six-view evidence

Light the actual room through plausible windows, overhead fixtures, and any task
lights required by the work. Visible sources, shadow direction, emitting area,
and reflected highlights must agree. Preserve contact shadows, readable grain,
material separation, and useful differences between work and storage areas.
Avoid camera-facing studio fill, clipped highlights, or darkness and blur hiding
unfinished details. Keep the same geometry and lighting across views.

Choose accessible cameras and sensible fields of view for these IDs:

| View ID | Evidence to show |
| --- | --- |
| `room_overview` | The enclosure, working areas, stock storage, and circulation as a whole. |
| `room_reverse` | A different direction exposing access and relationships absent from the first view. |
| `workbench_context` | The principal woodworking activity, workpiece, tools, and working position. |
| `workbench_detail` | Credible tool or equipment construction and its relationship to work in progress. |
| `surface_detail` | A significant surface exposing grain, texture scale, finish, and localized use. |
| `alternate_view` | Another meaningful spatial relationship with foreground, activity, and background depth. |

Follow the supplied workflow: build, render, actually open the images, and correct
observed defects. Deliver the reusable template, packaged dependencies, provenance,
six views, and unresolved issues. Preserve earlier previews. An Agent's `ready`
means ready for human review, not human acceptance.

## Execution constraints for this batch

Other independent room tasks are running concurrently. Use only the GPU index
provided in your runtime; do not launch other Codex sessions or use other GPUs.

Every asset `fetch` must acquire the same shared cache lock, including fetches
for assets already cached. Substitute the actual runtime paths in this command:

```text
flock -x ASSET_CACHE/.fetch.lock PYTHON ASSETS_SCRIPT fetch ...
```

Here `ASSET_CACHE`, `PYTHON`, and `ASSETS_SCRIPT` mean `runtime.asset_cache`,
`runtime.python`, and `runtime.assets_script`. Keep the lock held until that
fetch command exits. Do not bypass the lock or remove it when another task is
using it. Asset searches and room build/render operations do not need this lock.
All actual dependencies must still be packaged into your own template directory.
