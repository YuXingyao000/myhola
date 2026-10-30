# An everyday mechanical assembly workshop

Create a believable, occupied small mechanical assembly workshop. Its main
activity is fitting and assembling small mechanical products or subassemblies,
with incoming components, active assembly, checking, and outgoing work forming
a coherent process. This is a complete room that can be photographed from several
positions, during an ordinary working day in an established small business.
No people are required.

This task uses `quality_only`. The supplied Shed images illustrate convincing
construction, material detail, spatial relationships, and illumination. Build
an independent environment from individual assets and authored construction;
the reference does not prescribe its furnishings, architecture, or layout.
There is no supplied workshop project. No target CAD component is selected or
placed at this stage, and no empty product-display region should be reserved.

## Activity, scale, and layout

Choose a plausible assembly activity and explain it briefly in `scene_plan.md`.
It should be visually supported by the actual equipment, parts, tools, and
storage you can represent convincingly. Assembly and fitting should be the main
identity; a machining factory, generic warehouse, or shed with a single vise
would not fulfill this brief. Select the room dimensions, furniture, equipment
count, and arrangement yourself.

Use metric, human-scale proportions. Organize the room around places where
someone could stand or sit to work, reach frequently used tools, access parts,
inspect an assembly, and move completed work. Consider incoming trays or bins,
fastener and component storage, hand or bench tools, a checking area, and work
in progress where appropriate. Functions can share a bench or compact area;
do not turn this list into a fixed inventory or a row of identical stations.

Let activity explain the objects: a partly assembled item can have its relevant
tools and component tray nearby, while finished work and bulk storage occupy
different places. Use varied groups and orientations, clear working patches,
and modest accumulated disorder. Keep usable aisles, chair movement, drawers,
doors, and maintenance access plausible. Avoid random scattering, evenly spaced
duplicate props, empty expanses, or an immaculate product showroom.

## Architecture and equipment

Build a continuous enclosure with floor, walls, ceiling or roof structure,
doorways, and any windows needed by the design. Include visible framing, edges,
skirting, connections, and service routing at the scales the cameras expose.
Openings need a deliberate exterior or lighting treatment. All inspection
directions must belong to the same complete interior.

Use the supplied asset tools to find, inspect, and package credible individual
furniture, tool, storage, and equipment assets. Preserve useful UVs, PBR maps,
attached fittings, handles, controls, and cables. Verify dimensions and contact
with benches, shelves, and floors after importing. Use free assets with recorded
sources and licenses suitable for a reproducible research template; authenticated
providers are available only when configured.

Simple construction, trays, washers, fasteners, and secondary stock can be
procedural when their visible complexity warrants it. Prominent tools, fixtures,
and assemblies need recognizable construction and convincing material detail;
do not represent them as decorated boxes or arbitrary primitive piles. If a
defining asset cannot be obtained or built credibly, report the concrete gap.
Do not silently change the activity to avoid that limitation.

## Surfaces and use

Depict a maintained workshop with several years of use. Choose suitable actual
floor, wall, bench, and storage finishes. Inspect external materials at realistic
texture scale. Preserve distinctions among painted steel, bare metal, timber or
laminate, rubber, plastic, and other surfaces where they belong.

Use existing asset detail before adding wear. Working surfaces may have localized
scratches, polished handling areas, cleaned lubricant traces, or replaceable
protective pads. Storage and quiet corners should age differently from assembly
positions and traffic routes. Wear must follow use rather than a shared layer
of rust, brown noise, or exaggerated bump. Show enough surface variation for
wide views and close inspection, while keeping the workshop maintained and its
tools usable. Newly supplied components may remain clean.

## Lighting and photography

Light the actual room and work. Choose coherent daylight, overhead illumination,
and any local task lamps according to the layout. Source positions, emitting
surfaces, fixture housings, windows, and reflected highlights must agree. Keep
believable contact shadows and useful differences between active benches,
circulation, and storage. Metal should reflect surrounding architecture and
objects, not only an isolated studio light.

Judge exposure and material response in the actual renders. Avoid arbitrary
camera-facing fill, flat uniform illumination, excessive casts, clipped large
highlights, or deep shadows hiding unfinished details. Use the same scene across
views without per-camera object hiding or strong blur that conceals defects.

## Six-view evidence and delivery

Choose accessible viewpoints with sensible fields of view for all required IDs:

| View ID | Evidence to show |
| --- | --- |
| `room_overview` | The enclosure, assembly stations, storage, and circulation as a working whole. |
| `room_reverse` | A substantially different direction revealing access and relationships hidden in the first view. |
| `workbench_context` | The principal assembly activity, working position, component supply, and nearby tools. |
| `workbench_detail` | Convincing tool or fixture construction, work in progress, and local material response. |
| `surface_detail` | A significant bench, equipment, floor, or wall surface that exposes texture scale and use. |
| `alternate_view` | Another spatial relationship with foreground, active work, and background depth. |

Build, render, actually open the images, and correct observed defects using the
prescribed workflow. Deliver the reusable template, packaged dependencies,
provenance, six views, and an honest account of unresolved issues. Preserve prior
previews. The final human review will judge workshop identity, proportions and
layout, surfaces and equipment, illumination, and consistency across views.
The Agent's `ready` decision means ready for human review.
