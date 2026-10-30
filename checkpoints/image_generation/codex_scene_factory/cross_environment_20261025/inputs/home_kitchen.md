# An ordinary lived-in home kitchen and dining space

Create a believable domestic kitchen connected to an everyday eating area. It
belongs to a household that cooks and eats here regularly, with an ordinary meal
temporarily interrupted. The room is maintained and reasonably clean, but shows
personal habits and recent activity. No people are required. This is an ordinary
home, not a luxury property advertisement, restaurant, or kitchen showroom.

This task uses `quality_only`. The supplied Shed images illustrate expectations
for construction, material detail, spatial relationships, and illumination. They
do not prescribe an industrial style, dirt, furnishings, or layout. Build an
independent domestic environment from individual assets and authored architecture.
Do not locate or import the reference's underlying project, or substitute another
complete furnished interior.

There is no target CAD object or future product placement at this stage. Do not
reserve a display zone or arrange the room around a single product camera.

## Activity, scale, and layout

Choose a coherent household context and a moment of cooking, serving, or clearing
up. Briefly explain them in `scene_plan.md`. Decide the room dimensions, kitchen
configuration, furnishings, appliance selection, and number of places to eat.
The kitchen and dining functions may share a compact room.

Let actual activity explain the relationships among food storage, washing,
preparation, cooking, eating, and putting things away. Keep usable counter space,
plausible reach, and clear routes between these functions. Relate countertop
heights, appliance sizes, cupboard depths, table height, and seating to the same
human scale. Doors, drawers, chairs, and appliances should have room to operate.
Avoid isolated props in an empty shell or retail-display spacing.

Build the complete enclosure, including floor, walls, ceiling, doors, and any
windows. Include construction visible in the cameras: cabinet toe kicks, worktop
edges, backsplash transitions, frames, reveals, and appropriate sockets or service
connections. Openings need a deliberate exterior or lighting treatment. Reverse
views must belong to the same plausible home.

## Furnishings, appliances, and recent use

Use convincing domestic cabinetry, washing and cooking facilities, storage, and
dining furniture suited to the chosen household. Prominent fixtures and appliances
need recognizable construction, coherent controls, handles, seams, and functional
relationships. Small countertop appliances should connect plausibly to nearby
outlets. A sink needs credible basin depth, a drain, a faucet, and a junction with
its counter.

Use the supplied asset tools to search, inspect, and acquire individual assets.
Prefer free resources with licenses suitable for the intended use and packaging;
record sources, authors, licenses, and dependencies through the provenance
workflow. Inspect imports for dimensions, UVs, materials, and completeness.
Authenticated providers are available only when configured.

Simple architecture, cabinetry, and secondary objects may be procedural when the
visible construction supports it. Prominent furniture, appliances, cookware, and
food require enough detail for the inspection views. If a defining resource cannot
be obtained or represented convincingly, report the specific gap instead of
silently changing the brief or hiding a crude substitute.

Choose connected signs of recent use appropriate to the activity. Arrange purposeful
groups with unequal spacing and sensible orientations. Preserve clean working
patches and circulation. Avoid identical place settings, random clutter, or
filling every surface with food and decorative accessories.

## Materials, maintenance, and light

Choose ordinary domestic finishes and distinguish their responses: painted walls,
wood or laminate, tile and grout, coated cabinetry, stainless steel, ceramic,
glass, plastic, and fabric where present. Preserve good source materials before
adding changes. Check grain direction, texture scale, edge treatment, normal
strength, and roughness in actual images. Large counters, cupboard fronts, floors,
and walls must withstand close inspection.

Use restrained, localized evidence of maintenance and handling. Wiped counter
variation, a few water marks near the sink, small chair scuffs, softened contact
edges, or wrinkles in a used cloth can explain occupation. Different materials
and locations should age differently. Do not import workshop oil, pervasive rust,
uniform brown noise, excessive grease, or exaggerated damage into a maintained
home. Clean surfaces can still have convincing texture and reflections.

Choose a coherent time of day and light the actual room through its windows and
plausible domestic fixtures. Any ceiling, dining, or under-cabinet lighting must
agree with visible sources and their purpose. Retain contact shadows, useful
shadow detail, and varied reflections on cookware, glass, and appliances. Avoid
camera-facing studio fill, uniformly bright surfaces, clipped windows, or darkness
and blur that conceal unfinished construction. Keep lighting and geometry
consistent across views.

## Six-view evidence and delivery

Choose accessible camera positions with sensible fields of view. The historical
`workbench_*` IDs refer here to the principal kitchen activity.

| View ID | Evidence to show |
| --- | --- |
| `room_overview` | The enclosure, kitchen and eating areas, and circulation as a domestic whole. |
| `room_reverse` | A different direction revealing access, storage, and furniture relationships. |
| `workbench_context` | The principal preparation or cooking area with its fixtures and recent activity. |
| `workbench_detail` | Appliance, sink, cookware, or cabinet construction and material response. |
| `surface_detail` | A prominent counter, floor, cabinet, or table surface exposing texture scale and condition. |
| `alternate_view` | A distinct relationship between the eating area, kitchen, and background depth. |

Build, render, actually open the images, and correct observed defects through the
prescribed workflow. Deliver the reusable template, packaged dependencies,
provenance, six views, and remaining issues. Preserve previews and review records.
An Agent's `ready` means ready for human review, not human acceptance.

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
