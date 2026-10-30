# Making a believable working interior

Use the supplied reference images as inspectable examples of authored objects,
material detail, spatial relationships, and lighting. Aim for that level of
quality while satisfying the user's activity and room type. Follow the runtime
reference mode: `adapt` modifies the supplied project; `quality_only` creates
an independent space from individual assets and authored construction. Describe
asset reuse accurately in either mode.

## Understand before changing

Open the supplied reference images and read the compact quality notes. Identify
what makes the room, activity area, and important surfaces believable. In `adapt`,
inspect the supplied source hierarchy, dimensions, materials, and lights as
needed. In `quality_only`, no reference project is supplied: evaluate candidate
individual assets through their previews and imported geometry instead. Do not
search for the project behind a quality reference.

In `scene_plan.md`, describe what someone does here, which places support that
activity, and how the proposed space supports it. Identify major equipment and
large visible surfaces, their asset or construction approach, and any missing
critical resources. Explain the furniture or equipment relationships, the items
in active use, and the lighting. In `adapt`, also specify substantive changes from
the source. Do not fill a checklist with cosmetic renaming.

## Layout follows activity

Build storage, working, and circulation areas as parts of the same room. Consider
reach from the working position, where tools are put down, which drawers or boxes
are open, and where equipment needs clearance. Let items form useful groups with
unequal spacing and plausible orientation. Preserve some usable surfaces.

Dense does not mean covering every surface with random debris. Sparse does not
mean a single table surrounded by blank walls. Judge the room from several
directions: foreground, working area, storage, wall construction, and openings
should make spatial sense together. Preserve credible construction details such
as supports, joints, wall thickness, frames, and transitions between materials.

Prefer credible detailed assets for prominent equipment, tools, and furniture,
and suitable PBR materials for large surfaces. Use the supplied asset tool to
obtain and package resources; inspect them before choosing material changes.
Simple geometry is appropriate for simple construction and secondary objects,
such as distant nuts or bins. It is not a substitute for a recognizable major
machine. If essential equipment is unavailable, report the concrete gap instead
of presenting a decorated box as a finished machine.

Preserve complete usable asset groups, such as a toolbox with its contents.
Moving an object without its dependent parts, cords, contact surfaces, or parent
transforms can break an otherwise convincing asset. Check metric dimensions and
contact with shelves, tables, and floors.

## Surfaces communicate construction and use

Inspect existing UVs and PBR maps before adding anything. A useful source asset
may already contain the needed wear. Keep its texture scale, normal strength,
roughness variation, and material assignments coherent when moving or duplicating
it. A high-resolution map at the wrong scale still looks implausible.

Choose alterations according to material and use. Handling can polish contact
areas; paint can chip at exposed edges; dust accumulates in sheltered places;
cleaned oil can leave localized roughness changes. These are choices for suitable
objects, not a demand that every object be dirty. Do not spread the same noise,
scratch density, or brown tint across all materials. Large work surfaces, walls,
and floors deserve close inspection because they occupy substantial image area.

## Light the actual room

Account for windows, sky or exterior surroundings, visible fixtures, practical
task lights, walls, and ceiling. Match light position, size, direction, and color
to those sources. Reconsider inherited lights when moving furniture or changing
the work activity; a studio-like fill aimed at one camera can undermine the room.

Inspect contact shadows, broad and narrow reflections, material separation,
highlight clipping, and shadow detail. Metallic objects should reflect the
surrounding environment, not only a featureless bright rectangle. Keep believable
variation between brighter working areas and quieter surroundings without losing
the details needed for review. Use physically based materials and Cycles as a
foundation, then judge the actual images.

If a lighting problem is ambiguous, a temporary diagnostic render with a
suspected source disabled can help. Preserve the intended final scene separately.
Do not add lights or increase exposure repeatedly without identifying which
visible defect the change should fix. Windows must have a plausible exterior or
light treatment; uncovered backdrop edges and accidental empty World are defects.

## Review by observable comparison

Compare corresponding scales of reference and candidate: room to room, activity
area to activity area, surface detail to surface detail. Different room types,
composition, and exposure are allowed. Compare construction, materials, use, and
illumination rather than matching pixels or making every image equally bright.
In `quality_only`, an appropriate independent layout is required; similarity to
the reference layout is not evidence of quality.

For each meaningful problem, name the image region and observed symptom, then
propose a change. Distinguish a visible defect from a suspected cause. Examples:
an open toolbox hangs over its support; a floor texture repeats at an implausible
scale; a metal tool has an unnaturally uniform reflection; a task lamp casts light
in a direction inconsistent with its housing. Check whether the proposed repair
also improves the other views.

Technical success, object counts, texture resolution, and light counts are not
realism scores. A good candidate must retain reference-level asset and surface
detail, have a coherent activity and illumination, withstand multiple viewpoints,
and satisfy the user brief. In `adapt`, it must also show a substantive adaptation.
Do not mark it ready merely because all files exist or the reference looks good.
