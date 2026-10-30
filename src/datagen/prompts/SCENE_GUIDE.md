# Guide to Building Realistic Scene Templates

Use this guide to create reusable scene templates that place different CAD parts in believable everyday surroundings. The outputs are synthetic images, not real photographs.

The execution contract defines the fixed camera, reserved part region, support alignment, and two builder interfaces. The user prompt specifies the scene and target-part appearance for this task. This guide explains how to turn those requirements into a scene that can be inspected and tested.

## 1. Establish How the Space Is Used

Before writing the builders, record a short `scene_plan.md`: physical scale, what work is taking place, the functional prop groups, the condition of major surfaces, and the sources of light. Describe how the space is used and maintained, and what has just happened. Fill in nonessential missing details with consistent choices; do not wait for approval of this note. Planning and initial construction happen in the same Codex task.

- A repair bench used for years might have a repeatedly wiped worktop, chipped paint along its edges, tools temporarily set aside, and a cleaning rag. This does not imply rust or oil on every surface.
- An assembly bench in a new workshop can be tidy while still having credible surface textures, construction details, and objects needed for its work.
- Choose objects and their positions from the activity taking place. Avoid scattering props evenly just to make the scene look busy.

Normalized CAD units do not automatically correspond to meters. Use the user's physical scale and record its conversion in the plan. Establish consistent sizes for the bench, tools, wall textures, and light fixtures; do not assume an asset's default import size is correct. Separate an asset's dimensions from its placement: correcting scale is not a reason to multiply every existing position by the same factor. Record a recognizable reference object's intended physical dimension so the image can be checked against it.

## 2. Block Out the Space from the Actual Camera

Start with the support surface, main background, large props, and primary light. Add detail afterward. A workbench includes its worktop, edges, and any framing that may enter the image. The floor, walls, and any exterior view should form a complete, coherent space.

- The host pipeline places the part and camera. Do not move the camera, rescale the part, or modify the checker to accommodate the scene.
- Keep the support region flat. Keep props outside the reserved part region and the protected volume between the camera and part. Surface materials may still show wear within the support region.
- The entire environment moves vertically to match each part's lowest point. Account for changes to the visible bench edge, background, and lighting at different heights.
- Cover the full image without exposed wall ends, transparent gaps, or implausible empty backgrounds. Visible windows also need a credible exterior view.
- Silver-colored parts reflect surroundings outside the image. Provide appropriate walls, light and dark areas, and light sources beyond the camera frame when needed.
- Design for the actual close-up implied by the camera and physical scale. A small field of view cannot show an entire workshop. Nearby props may be partly cropped by the image, and groups outside the protected region may overlap one another. Use relationships and height differences rather than isolated, evenly spaced decorative objects.
- Identify which surfaces occupy most of the visible background. A room full of off-camera objects does not enrich a frame dominated by one uniform tabletop. Give those visible surfaces appropriate construction details and spatially coherent material variation while keeping the reserved support region flat.

Use the `quick` preset in `preview.py` when needed to inspect the actual composition. Resolve support, occlusion, and scale issues before investing in detailed materials, while designing light sources and reflection surroundings together with the layout.

## 3. Choose Assets by Their Contribution to the Image

Make decisions for meaningful objects or groups, such as a drill, a workbench, or a shelving unit. There is no need to fill out a separate table for every screw.

| Contribution to the image | Preferred approach |
|---|---|
| Simple, small, or distant objects, such as a few nuts | Simple procedural geometry with appropriate PBR materials |
| Large surfaces such as worktops, floors, and walls | Simple geometry with convincing external PBR materials; inspect texture scale and repetition |
| Prominent, structurally complex objects such as drills, toolboxes, and equipment | External models with credible geometry and materials; avoid substituting crude primitives for visible detail |
| Off-camera objects affecting reflections, shadows, or indirect light | Allocate detail according to their actual lighting contribution, not only whether they appear directly in the image |

Use the provided `assets.py` tool to search for and obtain assets; read `--help` first. Sources include Poly Haven, ambientCG, CGBookcase, Sketchfab, and local catalogs. Which sources can actually be searched or downloaded depends on the tool's current configuration and responses.

The asset tool handles caching, copying, and provenance records. Do not bypass unconfigured authentication, install dependencies yourself, or place credentials in scripts, prompts, or the template package. If no suitable asset is available, choose another available candidate, change the prop selection, or report the limitation. Never invent asset provenance.

High polygon counts and high texture resolutions do not guarantee good results. Inspect candidate previews, check material completeness and suitability for the actual camera angle, and select the detail needed for the final 512 × 512 image.

## 4. Inspect Existing Materials Before Changing Them

Inspect important surfaces under the scene's actual lighting. What structure, color, roughness, relief, and wear does the original asset already provide? Does it match the requested state of use? Leaving a suitable material unchanged is a valid decision.

- Check texture connections, color spaces, normal-map orientation, UV or projection scale, and visible seams, stretching, or tiling repetition.
- Do not treat shadows in a base-color texture as a substitute for scene lighting. Check for obvious conflicts with the scene's light sources.
- Roughness, metallic response, and surface relief should match the material. Plastic, paint, and exposed metal should not all reflect light in the same way.
- When the source material already has enough detail, first adjust scale, tint, and roughness. Do not conceal problems beneath a uniform noise layer.
- Inspect the actual contribution of large surfaces at 512 × 512. Useful variation can include construction joints, worn edges, localized wiped patches, and differences between exposed and sheltered areas. Choose what fits the requested use history; more dirt, more texture resolution, and higher normal-map strength are not quality goals by themselves.

When changes are needed, choose the method according to the cause of the wear or variation:

| Appearance to represent | Possible method |
|---|---|
| Subtle roughness variation or small surface irregularities | Procedural nodes with amplitude and scale appropriate to the material |
| Edge wear, polished contact areas, or localized dust | Spatially constrained masks with irregular detail added within them |
| Local oil stains, wiping marks, labels, or scratches | Decals or masked blending; inspect edges and reflection behavior |
| A specific surface appearance unavailable in the source material | Consider texture generation or editing only if the caller provides a generative asset tool |

Wear should have a reason to occur in its location. Exposed steel may rust; a plastic grip should not receive the same rust layer. Dirt should not cover every object uniformly. Different areas may be used and cleaned at different rates.

Generative textures are optional. If used, inspect UV boundaries, seams, consistency across PBR maps, and any baked-in directional lighting or shadows. The availability of a generation tool is not a reason to replace an already suitable material.

The target CAD material still comes from its separate builder and follows the user prompt. At this stage, it is constructed from parameters, requires no UVs or texture images, and must not alter the target geometry. Material displacement must also leave the reserved support region flat.

## 5. Make the Lighting Belong to the Space

Identify a plausible source for the primary light, such as a window on one side, ceiling fixtures, or a task lamp. Then choose its direction, size, intensity, and color temperature. The source does not have to appear in the image, but it should be consistent with the layout.

- Use light direction and size to control tonal transitions, shadow softness, and the shapes of metallic highlights.
- Use walls, the worktop, the ceiling, and off-camera surroundings to produce reflections and indirect light. Add fill light for a specific reason; do not default to studio lights on every side.
- When using an HDRI, check its orientation and brightness against the room's openings. Do not blindly combine an HDRI with additional key lights that produce conflicting illumination.
- Retain contact shadows and useful tonal variation while keeping holes, grooves, thin walls, and the outer silhouette readable.
- Adjust exposure and color management after the materials and lighting are broadly established. A strong local highlight on metal is not automatically a failure, but major structural features must not remain lost in highlights or shadows.

Do not solve every problem by increasing brightness. A washed-out, plastic-like, or flat appearance may come from the material, reflection environment, light size, or exposure. Form a specific explanation before changing the scene. Keep the intended target finish fixed when the user requests it; do not obscure target-mesh shading defects with dirt, blur, or different material settings. Problems requiring changes to the host's import or normal handling belong in the host-blocker report.

## 6. Use Actual Images to Guide Revisions

Read `preview.py --help`. During construction and host-requested repair, use `quick` for useful feedback; the host runs the full `review` afterward. The host configuration supplies the samples, camera, and seeds. Do not replace the fixed review set with parts selected because they look better.

After running a preview, open the images with an available image-viewing tool. Creating a file, reading its path, or inspecting tool logs does not mean the image has been viewed. Judge the final 512 × 512 result; use enlarged views to locate problems.

For each revision, follow: **observed issue → possible cause → targeted change → inspect the new output**. For example, “the drill grip looks like smooth plastic” is an observation, while “it lacks suitable roughness variation” is an explanation to test. Do not report a hypothesis as an established cause.

Address the most consequential specific problems first: background gaps, obvious scale errors, crude prominent props, distorted large surfaces, or lighting that hides structure. Do not repeatedly add detail to invisible small objects. If a technical check fails, first fix the object and rule identified in the report.

When the host resumes the session for visual review, inspect the current 12 CAD images and 3 empty-environment images without editing, downloading, building, or rendering. Use the empty views to assess scale references, functional grouping, dominant surfaces, and illumination; use the CAD views to assess contact, reflections, and feature readability. Return the structured `ready`, `revise`, or `blocked` decision described in `ACCEPTANCE.md`. Record unavailable or uninspected images honestly. Leave changes for a subsequent host-requested repair task.

After repairs, old images and reports no longer represent the final version. The host reruns full review and resumes visual inspection. Its repair budget defaults to two rounds, independently of how many internal edits or quick previews were useful. There is no required number of rendering iterations, and a review decision is not a guarantee of photographic realism.

## 7. Deliver a Reusable Template Package

Deliver `scene_plan.md` and both builders, together with the `assets/` files and `assets.json` actually used. Resolve asset paths relative to the template and include the required source files in the package. A machine-local cache path or temporary download URL is not a substitute for those files.

Retain the build reports, previews, and failure records produced by the tools. In the final response, state which checks were performed, which images were viewed, and what issues remain. Explicitly mark checks that were not run as unverified.

Human visual review takes place after the host workflow ends. Do not wait for item-by-item approval during generation, and do not claim that the template has received human approval.
