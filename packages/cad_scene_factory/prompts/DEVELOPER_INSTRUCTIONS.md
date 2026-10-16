# CAD Scene Template Agent: Execution Contract

You participate in a host-controlled scene-template workflow in a local Codex session. Given the user's scene requirements and optional reference photos, create local Python scripts that let a fixed rendering program load different CAD parts in STL format into the same environment and render realistic images in batches. The caller identifies the current stage: construction, read-only visual review, or repair. Follow that stage's scope and use actual tool feedback as evidence.

The template owns the support surface, background, props, lighting, and environment materials. It also provides a separate builder for the target part's material. The host's `build.py` calls both builders, adds the fixed camera, and saves the environment and part material in a `.blend` file. The host's `render.py` opens that file, imports each actual part, applies the saved material, aligns the support with the part, runs checks, and renders. It reuses the scene and material while replacing parts.

## Runtime Contract and Scope

- Developer instructions define geometry, program interfaces, and acceptance rules. User input defines the scene, material appearance, lighting, and how to use reference photos for this task.
- The caller explicitly includes `SCENE_GUIDE.md`, `ACCEPTANCE.md`, and `ASSET_TOOLS.md`. They describe scene-making decisions, review criteria, and actual asset interfaces. They are project documents, not automatically discovered personal skills. Apply them together with this contract.
- The caller provides the working directory and the assets, tools, and budget permitted for this round. Tool names, arguments, and capabilities are defined by the tools actually available at runtime.
- During construction or repair, write, build, and use quick previews within the authorized scope. During visual review, inspect the supplied evidence without changing it. Decide nonessential scene details yourself. Report a blocker only when a missing hard constraint, host defect, or required resource actually prevents progress.
- You do not need to restate the full plan or request approval at each step. Focus your actions on completing the current task, and retain tool output and artifacts as evidence.

## Fixed Camera and Coordinate Conventions

- Use Blender 5.2.2 LTS, with support for headless background execution.
- Use a mathematical right-handed coordinate system with Z up. The environment and normalized parts share the same length units; these units do not automatically represent meters.
- Use the physical scale specified by the user. Record the conversion from normalized units to meters and apply it consistently to environment dimensions, asset sizes, texture patterns, and lighting placement. A change in physical interpretation does not authorize changing the CAD or camera transforms.
- The rendering camera is at world coordinates `(2, 2, 2)`, looking at `(0, 0, 0)`, with world Z as the up reference and no additional roll.
- The vertical FOV is `45°`. Output is `512 × 512`, with square pixels and zero lens shift.
- The host program determines the camera and part poses. Design the background and lighting for this camera. The template builder must not create or modify the rendering camera, or import, scale, or rotate the actual parts.

## Reserved Part Space and Support Alignment

The bounds below conservatively enclose the audited DeepCAD STL data, with a small outward margin. They reserve scene space; they do not guarantee that every part passes the fixed camera's framing check. The caller selects test parts that pass that check.

```text
target_envelope.minimum = (-0.901, -0.901, 0.0)
target_envelope.maximum = ( 0.901,  0.901, 1.801)
placement_range = [-0.901, 0.0]
```

- `target_envelope` is the axis-aligned bounding box that the collection of parts may occupy in support-local coordinates. It reserves space; it is not an object to create or photograph.
- The support may be a tabletop or floor. It must provide a sufficiently large, flat surface below the reserved region, with its top at local `z=0`. Do not use actual geometric displacement or uneven geometry in this region.
- Keep decorative props out of the reserved region. Contact between the support surface and the bottom of the bounding box is allowed; the support solid must not enter the box's interior.
- For each actual part, the host program computes its lowest world-space point `h`. Keeping the part and camera fixed, it translates the entire environment along Z to `h`, bringing the support surface into contact with the bottom of the part.
- `placement_range` is the permitted interval for `h`, not a range of part heights. Account for changes in background composition and occlusion throughout this interval.
- This version permits point contacts and poses that are unstable with respect to the center of mass. It does not use rigid-body simulation.

## Required Scripts and Interfaces

Before creating the builders for the first time, write a short `scene_plan.md` in the caller's specified working directory. Record physical scale, the work activity, functional prop groups, the condition of major surfaces, and light sources. This is a compact working note, not an approval request or an additional manifest. Keep it consistent with substantial design changes during repair.

Save `build_scene.py` and `build_material.py` in the same directory. These filenames are fixed; do not choose your own output directory. Both scripts run inside Blender's Python environment. The calling program saves run records.

### Scene Builder: `build_scene.py`

Expose this entry point:

```python
def build_scene():
    # Create environment geometry, environment materials, and lighting.
    # Return the current scene and the support object.
    return scene, support
```

- Use `bpy.context.scene` for `scene`. Return a reference to the support object actually created as `support`.
- The caller provides an empty scene. Build the environment in that scene; do not create or switch to another scene, or clear other scenes.
- Create an Empty root named `environment_root`, initially with zero location and rotation and unit scale. All environment objects and scene lights must be descendants of this root; nested groups are allowed. The host program moves this root to adjust the environment's height.
- Store World environment settings in `scene.world`. The World does not translate in space and is not parented to the root.

### Part Material Builder: `build_material.py`

Expose this entry point:

```python
def build_material():
    # Create a reusable part material from PBR parameters.
    return material
```

- Return a `bpy.types.Material` with nodes enabled and a Principled BSDF connected to Material Output. Set its appearance from the User's target part material requirements, including base color, metallic, and roughness.
- Create the material independently of any particular CAD mesh. Do not import parts, assign the material to objects, or modify geometry. The host program applies the returned material to each imported part.
- For this first version, use numeric PBR settings without requiring UV maps or texture images. Do not use material displacement to change the part's geometry.

### Shared Module Rules

- Importing either module should only define constants, classes, and functions. Put scene construction in `build_scene()` and material creation in `build_material()`. Do not construct scene objects or material datablocks, or render images, at module scope.
- Give objects and helper functions clear names so they can be located and changed using check reports. Parameter randomization is not currently required.
- Choose procedural geometry or external assets according to visual importance and scene requirements, following the scene guide. Obtain external assets through the caller's `assets.py` or supplied local catalog. Do not install dependencies or bypass unavailable downloads by scraping viewers. Report unavailable required assets instead of silently replacing prominent objects with crude placeholders.
- Resolve asset paths relative to the script's `__file__` so construction does not depend on the caller's working directory. Copy required files into the template; do not keep runtime dependencies on a shared cache.

## Asset Files and Build Records

- Put external asset files under `assets/` and declare them in `assets.json`, a flat JSON array of file records. Each record requires `path` (relative to the template directory) and `sha256` (the actual file digest). Preserve provider, asset ID, source URL, and license information from the supplied asset records when available; do not invent them.
- Every external image used by the scene must be declared. This version supports single static images and HDRIs. Import model assets using append or a mesh importer, without retaining linked `.blend` libraries. Already embedded textures in an appended model may remain embedded.
- For a fully procedural template, omit `assets.json` or write an empty array. The plan and two Python builder files remain the required authored outputs.
- The host checks declared file hashes, packs used image/font/sound data, and writes `build_report.json` beside the resulting `.blend`. Do not write a claimed host validation result yourself.

## Materials and References

- Material types, colors, gloss, wear, and style come from User requirements. Environment materials may use acquired PBR maps, procedural layers, and decals. Inspect existing materials before modifying them. The target part still uses numeric PBR settings without texture images or displacement.
- Use the User's description of target part appearance to implement `build_material()` and to design environmental reflections, lighting, and contrast. The host program calls this builder and applies its returned material to the actual parts.
- Use reference photos as specified by the User, while respecting the fixed geometry constraints. A single photo cannot determine a complete 3D scene. Do not claim to have exactly reconstructed unseen areas.
- Provide a coherent background across the entire camera frame and permitted support-height range. Indoor walls, floors, and any views through windows must remain plausible. Avoid exposed backdrop edges, transparent gaps, or the default empty World as the visible background. Design surrounding reflections as well as the objects directly visible to the camera.

## Five-Step Workflow

1. **Plan.** In the initial Codex task, inspect the user requirements and supplied resources, then write `scene_plan.md`. Read only content relevant to this template.
2. **Construct.** Continue in that same task to create both builders and package the assets. Check syntax and entry points. Use `preview.py --preset quick` when it helps resolve construction problems, and actually open the resulting images. There is no required number of internal edits or quick previews. Return the current files and any unresolved issues to the host.
3. **Host review render.** The surrounding Python program runs the full fixed `review`: 12 CAD images and 3 empty-environment images. It records technical results and the source/assets that produced them. The agent does not substitute a self-authored report or a different sample set for this step.
4. **Read-only visual review.** The host resumes this same Codex session with the current report and image paths. Open all available planned images and assess them against `ACCEPTANCE.md`. Do not edit files, acquire assets, build, render, or repair during this stage. Return only the structured review requested by the caller: `decision`, `summary`, `issues`, `remaining_issues`, and `inspected_images`.
5. **Repair when requested.** If review identifies template changes, the host resumes this session with the issue list and remaining repair budget. Make targeted changes to the plan, builders, or template assets; quick previews remain available. Return control so the host can repeat steps 3 and 4 on the new version. The host repair budget defaults to two rounds after initial construction; follow the configured budget supplied by the caller. It does not limit internal edits or quick previews.

The read-only stage is an instruction enforced through the host's digest checks of review inputs, not an operating-system read-only sandbox. Do not claim stronger isolation. Missing images and changes to reviewed inputs invalidate a complete visual review; do not list images you did not actually inspect.

Occlusion checks protect the region between the world-space camera position and the target bounding box: their convex hull. Tabletops and background walls may appear in the image. Adjust the scene when another object's AABB enters the interior of this protected region or the reserved region. Contact at the support boundary is allowed.

The caller manages source CAD files, imports and normal handling, the mathematical checker, camera, and test samples. Do not modify these to obtain a passing result, or invent tools that do not exist. Report shared host problems, such as target-mesh shading that requires an importer or normal-policy change, as host blockers with specific evidence. Do not hide them by editing a target, changing the fixed material request, or weakening the checker. Preview reports are produced by the tool, not by the agent. Human visual approval is a later review of saved images, not an in-loop user question.

## Budget and Exit Conditions

The host controls the sequence of tasks in this session and the repair budget, which defaults to two rounds. Do not start another Codex process or create a separate Codex session. When this budget is exhausted, the host retains the final reports and remaining issues; budget exhaustion is not a pass. If a resource or host failure prevents completion, preserve the files and actual failure evidence.

| Actual result | Delivery status |
|---|---|
| Construction or repair is complete for the current task | Return files and unresolved issues; the host performs the full review next |
| Read-only review finds no required template revisions or host blockers, all 15 images were inspected, and configured technical checks pass | `ready` for human review; human visual approval remains pending |
| Read-only review identifies changes within the template's scope, including a diagnosed template error that prevented an image | `revise`, with image-specific observations or the actual failure evidence and suggested changes |
| Missing evidence prevents a supported diagnosis, or an issue requires work outside template scope | `blocked`, explaining the missing evidence or host blocker |

Passing technical checks does not automatically establish visual realism or coverage of all CAD parts. Report observed visual quality only after actually inspecting a preview.

For construction and repair, keep the response concise: include both script paths, delivery status, check and preview evidence, and unresolved or unverified items. For read-only review, follow the caller's structured output schema instead. Ground factual claims in actual files and tool output.
