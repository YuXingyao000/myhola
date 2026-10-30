Task: repair the current room candidate using the supplied evidence.

Read the actual build or render errors and the visual review. Open the affected
images and relevant references before making visual changes. Identify which
observations each change addresses. Preserve aspects that already work, including
mature source geometry, UVs, material detail, and credible spatial relationships.

Edit build_scene.py, views.json, and scene_plan.md as needed inside template_dir.
Keep the six required view roles. Do not fix a visible defect solely by removing
the view that exposed it, cropping it away, hiding geometry per camera, or making
the scene dark or blurred. Keep source references, host tools, and existing host
reports unchanged. In adapt, also keep the packaged base files unchanged and
modify the loaded source scene in memory through the builder. In quality_only,
continue the independent room design; do not import a complete reference room
to bypass an unresolved quality problem. If adding an asset, use the supplied
tool to package it and merge its real provenance. Report missing critical assets
honestly rather than replacing defining equipment with crude proxies.

Build and render updated quick previews using the provided tools, and actually
open the affected views. Check a wider context view when a local change could
alter layout or illumination. Save working outputs in a new previews/ directory.
There is no minimum count of internal edits or renders; use feedback to resolve
the stated defects within the current task.

Return the changed file paths, a concise observation-to-change account, the
preview images you opened, and unresolved issues. Do not claim host or human
approval, rewrite previous reviews, start another Codex session, or request extra
repair rounds. The host rebuilds and inspects the new candidate independently.

Current repair evidence and runtime configuration follow.
