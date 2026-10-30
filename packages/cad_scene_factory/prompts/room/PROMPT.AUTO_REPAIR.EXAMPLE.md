# An everyday automotive service bay

Create a believable, occupied automotive repair bay in a small independent
garage. Show an ordinary working day with a recognizable maintenance activity,
appropriate equipment, parts handling, and access for the mechanic. The result
should be a complete interior that can be photographed from different positions,
with the material and construction detail of a real, maintained workplace.
No people are required.

This task uses `quality_only`. The supplied Shed images are examples of material
detail, convincing construction, spatial relationships, and illumination. They
do not prescribe a shed layout, a color palette, or a set of furnishings. Build
an independent environment from individual assets and authored construction;
no complete garage project is supplied. No target CAD component is being placed,
and no empty product-display region should be reserved.

## Activity, scale, and layout

Choose a coherent repair or maintenance activity that the available assets can
support, and explain it briefly in `scene_plan.md`. Wheel and brake servicing,
inspection of a removed automotive assembly, or another clearly identifiable
service task can work. The room must read as an automotive workplace through
its equipment, work in progress, and working clearances. A generic bench with
some hand tools and automotive labels would not fulfill the brief.

Select the dimensions, equipment count, furniture, and arrangement yourself.
Use metric, human-scale proportions and account for vehicle or assembly size,
mechanic access, tool movement, storage, doors, and circulation. If a vehicle is
shown, use a credible asset with appropriate bodywork, wheels, glazing, and
interior detail where visible. If the activity centers on a removed assembly,
make that choice visually convincing through the actual work and equipment.
Do not substitute a vaguely car-shaped primitive for a significant vehicle.

Let the work explain the placement of objects. Frequently used tools can be
close to the active task, while replacement parts, removed parts, packaging,
bulk supplies, and completed work occupy suitable places. Consider the physical
relationships among lifting or holding equipment, mobile tool storage, parts
trays, service supplies, and bench work as needed by your chosen activity.
These are functional considerations, not a required inventory or fixed layout.

Show purposeful groups, differing orientations, and modest accumulated disorder
without turning the floor into random clutter. Keep access and work areas
usable. Avoid repeated evenly spaced props, a pristine showroom, or a large
empty shell with only one detailed corner.

## Architecture and equipment

Build a continuous enclosure with floor, walls, roof or ceiling structure, and
credible access openings. Include the construction details visible to the
cameras: door frames and thresholds, structural edges, electrical services,
and any hoses or ventilation appropriate to the activity. Garage openings need
a deliberate exterior or lighting treatment; cameras must not expose an
unfinished backdrop or an empty world outside the room.

Use the supplied asset tools to search for, inspect, and package convincing
individual equipment, tool, storage, and surface assets. Preserve useful UVs,
PBR textures, controls, attached fittings, hoses, and cables. Check dimensions,
orientation, and floor or bench contact after import. Use free assets with
recorded sources and licenses appropriate for a reproducible research template;
authenticated providers are available only when actually configured.

Simple architecture, brackets, secondary stock, fasteners, and packaging may
be procedural. Prominent equipment must have recognizable construction and
credible materials at the views that expose it. Inspect critical imported
assets before committing the scene to them. If a defining asset cannot be
obtained or built convincingly, record the specific limitation instead of
silently presenting a simplified substitute as finished.

## Surfaces and use

Depict a maintained garage with several years of everyday use. Choose finishes
appropriate to the work: floor construction and coatings, painted equipment,
rubber tires or mats, bare metal, plastic fittings, and wall surfaces should
respond differently to light and handling. Use plausible texture scale and
preserve the useful detail already present in external materials.

Wear should follow activity. Traffic areas, frequently handled drawers, tool
contact, and cleaned service spills can differ from sheltered storage. A worn
floor need not be uniformly black or wet. Rubber, painted steel, and machined
surfaces should not all share one roughness or noise pattern. Localized dust,
scuffs, handling polish, and cleaned residue should support the scene rather
than conceal missing construction. New replacement parts may remain clean.

## Lighting and photography

Light the actual bay using plausible daylight access, overhead fixtures, and
any local inspection lighting justified by the activity. Visible fixtures,
emitting surfaces, windows, shadows, and reflections must agree. Preserve a
useful difference between active work, circulation, storage, and the enclosure.
Metal, rubber, and painted surfaces should reflect the surrounding room with
credible contrast rather than an isolated studio lighting setup.

Judge exposure, surface response, and shadow detail in rendered images. Avoid
arbitrary camera-facing fill, uniformly bright illumination, excessive color
casts, and darkness or depth of field that hides unresolved equipment. Use the
same room across all views, without per-camera object hiding.

## Six-view evidence and delivery

Choose accessible viewpoints with sensible fields of view for these exact IDs:

| View ID | Evidence to show |
| --- | --- |
| `room_overview` | The bay, active maintenance task, equipment, enclosure, and circulation as a whole. |
| `room_reverse` | A substantially different direction showing access, storage, and the opposite enclosure. |
| `workbench_context` | The principal repair activity and its relationship to tools, equipment, and working access. |
| `workbench_detail` | Significant vehicle, assembly, or service-equipment construction and material response. |
| `surface_detail` | A major working or equipment surface exposing texture scale and localized use. |
| `alternate_view` | Another convincing relationship with foreground, active work, and background depth. |

The `workbench_*` IDs refer to the core working area, whether or not it is a
literal bench. Build, render, actually open the images, and correct observed
defects using the prescribed workflow. Preserve previous previews and deliver
the reusable template, packaged dependencies, provenance, six views, and honest
remaining limitations. The Agent's `ready` means ready for human review.
