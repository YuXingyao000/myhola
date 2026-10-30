Task: create the first complete room candidate.

Read the room contract, guide, user brief, and runtime reference bundle. Open the
supplied reference images. Check runtime.reference_mode. In adapt, inspect the
packaged source and its inventory. In quality_only, use the images to understand
the quality target, then select individual assets and design the requested room;
do not load the reference project or another complete furnished room.

Write a compact scene_plan.md describing the activity, layout, selected asset
groups, surface treatment, lighting, and inspection cameras. In adapt, identify
inherited elements and meaningful changes, preserving the packaged source files.
In quality_only, explain the independently designed space and any critical asset
gaps. Use the supplied asset tool and cache to obtain required individual models
or materials, preserving assets.json provenance. Then implement build_scene.py
and views.json in template_dir. Do not copy or rehash a supplied base package.

Use the host build and render commands to obtain working feedback. Render and
actually open at least one room view, one work-area context view, and one close
surface view. Compare them with the references and correct visible defects that
you can address. Keep outputs in uniquely named previews/ subdirectories. You
may use a separate quick views subset; the final views.json must retain all six
required cameras. No arbitrary minimum iteration count is required.

The candidate must fulfill the requested room type and activity with credible
equipment, surface detail, layout, and illumination. In adapt, camera changes,
renamed collections, and a copied source alone are insufficient. In quality_only,
the reference does not prescribe the room's furnishings or layout. Report a
missing defining asset rather than inventing a crude proxy or changing the brief.
Do not introduce a CAD object, reserved product region, or one-camera set.

Do not launch other Codex sessions, change host code, or write purported host
verification results. Finish with the authored file paths, the substantive
changes, actual preview paths you opened, and remaining issues. The host performs
the independent full build and six-view render next.

User requirements and runtime configuration follow.
