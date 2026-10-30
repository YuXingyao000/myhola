# Scene Request: A Maintained Machining Workshop

Create one reusable workshop scene template for photographing different CAD components, one at a time, on a workbench. The images should resemble ordinary photographs taken in a working shop, with believable surfaces, tools, lighting, and spatial context. The photographed component is a freshly machined aluminum part.

This request specifies the desired scene and appearance. Use the caller's execution contract for the camera, reserved part volume, support alignment, builder interfaces, output locations, and preview configuration. Use `SCENE_GUIDE.md` for production decisions, `ASSET_TOOLS.md` for the available asset interfaces, and `ACCEPTANCE.md` for evidence and review. Do not redefine those interfaces in the template.

## 1. Scene Identity and Use History

The setting is a small indoor machining and maintenance workshop that has been used daily for several years. People assemble components, check dimensions, and carry out minor repairs here. Equipment is maintained, and work surfaces are periodically cleaned.

For this scene, imagine a repair and measurement task still in progress. A local area has been cleared for the photographed component, while the tools, measuring equipment, containers, and consumables needed for the work remain nearby. Some are in use, others are temporarily set aside or stored within reach. The bench has persistent signs of repeated use, with differences between handled areas, locally cleaned patches, and less disturbed surroundings.

Express this history through the choice and placement of objects and through material condition. The workshop should feel occupied and functional. It does not need to be dilapidated or covered in dirt, and its contents should not look arranged for a product display.

## 2. Priorities and Creative Latitude

Use these priorities when making tradeoffs:

1. Preserve the fixed geometry contract and a clear view of the target part. Its silhouette, visible holes, grooves, steps, and thin walls should remain readable.
2. Make the workbench and other large visible surfaces convincing, with coherent scale, materials, contact shadows, and reflections.
3. Give prominent props enough structural and material detail to withstand inspection in the final image.
4. Add smaller contextual details only where they contribute to the scene at the final output resolution.

The workshop setting, metal workbench, maintained but visibly used environment, and target material below are requirements. You may choose the exact room layout, tool selection, storage arrangement, window and fixture locations, and nonessential colors. The examples of props below are alternatives, not a checklist of objects that must all be included.

The result should work for the caller's different CAD parts. Do not tailor the layout to one selected mesh or improve composition by moving, rotating, scaling, or editing the target part or camera.

## 3. Workbench and Support Area

Use a substantial metal workbench with a visibly thick tabletop, plausible edges, and a frame that appears capable of carrying workshop equipment. The tabletop should be dark gray steel, with broader, weaker reflections than the target aluminum part. The frame should be dark blue-gray painted metal.

Place the reserved part area on a continuous, flat region of the tabletop. Keep this region and the camera-to-part protected volume free of props. Geometric clearance does not require a visually uniform surface: this local working area should retain believable traces of its use.

Desired wear includes scratches at plausible scales, overlapping wiping patterns, localized polishing where hands or objects frequently contact the surface, and residual marks from cleaned oil where the work would produce them. Exposed frame edges may show paint wear. Make the relevant differences readable at the final image resolution, with varied location, orientation, scale, and intensity. Do not cover the entire bench or every prop with the same dirt or rust layer.

Within the support region, express surface detail through shading without geometric displacement, raised decals, debris, holes, or uneven support geometry. Preserve the host's support alignment behavior, including its allowance for unstable part poses.

The tabletop edge and part of the frame may be visible if they help establish depth and scale. Ensure that the environment remains complete when the host changes its height for different parts.

## 4. Surroundings, Props, and Spatial Scale

Provide a coherent workshop interior around the bench. A warm light-gray painted wall and a blue-gray tool rack or storage area are suitable background elements. Use a maintained concrete floor with believable grain, mild discoloration, and occasional cleaning or scuff marks wherever the floor contributes to the image or lighting.

Arrange functional groups that make the ongoing repair understandable. Examples include an open tool case with tools, a measuring instrument near the working area, a tray of fasteners, and a wiping cloth near appropriate supplies. Choose compatible groups rather than including every example. Vary height and orientation; objects may overlap one another outside the protected region, and nearby items may be partially cropped by the frame. Avoid evenly spaced decorative clutter, isolated props around a large blank center, or repetition used only to increase object count.

If a drill or another mechanically complex tool is prominent, use an asset with convincing geometry and materials. Small or distant nuts and similarly simple items may be created procedurally. If a suitable asset is unavailable, choose a different plausible prop rather than presenting a crude placeholder as finished work.

Keep one CAD component as the subject. Do not include people, additional display components, conspicuous brand logos, or large blocks of text. Secondary objects should provide context without obscuring the part or competing with its main features.

The intended components have a longest dimension of **15–20 cm**. For this first template, use a fixed nominal **17.5 cm**, without scale randomization. A normalized longest dimension of approximately `1.8` therefore represents `0.175 m`: **one scene unit represents `0.175 / 1.8 m` (approximately `0.097222 m`)**. Keep the normalized CAD, camera, support-alignment convention, and reserved region unchanged. Convert physical environment dimensions into these shared scene units; changing a Blender unit label alone is insufficient.

Record this conversion and the physical size of a recognizable reference prop in `scene_plan.md`. Set asset dimensions and placement deliberately; do not simply enlarge all existing coordinates. The fixed camera now photographs a small workbench close-up, roughly 28 cm wide at the origin depth. Build convincing nearby context rather than trying to fit an entire room into that frame. Keep the wider room coherent where it affects the background, shadows, or reflections.

## 5. Environment Materials and Asset Selection

Use the following appearance targets to guide asset selection and adaptation:

| Element | Material and condition | Visible cues to preserve |
|---|---|---|
| Tabletop | Used dark steel, locally cleared for ongoing work | Spatially varied scratches, wiping and contact marks, restrained reflections, believable edge treatment |
| Bench frame and rack | Maintained blue-gray painted metal | Painted surfaces distinct from exposed metal, localized edge wear where justified |
| Wall | Warm light-gray matte paint over a plausible wall surface | Consistent texture scale and local condition; structural details where they contribute to the close-up |
| Floor | Used, periodically cleaned concrete | Grain, mild wear, localized scuffs or cleaning traces without uniformly heavy staining |
| Tools and storage | Materials appropriate to their construction | Distinct plastic grips, paint, bare metal, rubber, or fabric; existing asset detail retained where suitable |

Prefer suitable external PBR materials for large visible surfaces and detailed external models for prominent complex props. Simple structural geometry and minor props may be procedural. Use the asset tools and local catalog actually supplied by the caller.

Inspect the original material before changing it. Keep existing detail when it already matches the scene. Adjust mapping scale, color, roughness, or localized wear only when the actual preview indicates a mismatch. Procedural layers, constrained wear masks, and decals are appropriate for specific changes; uniform noise across every object is not a substitute for material design.

Keep wear consistent with the material and its use. For example, an exposed steel edge may show oxidation, while a plastic handle may show polishing or ingrained dirt. Do not apply the same rust layer to unrelated materials. Avoid conspicuous texture seams, stretched maps, repeated stains, or lighting baked into a texture that conflicts with the scene illumination.

Use free assets whose terms permit the intended use and inclusion in the template package. Prefer CC0 assets; preserve the actual license, author, source, and any required attribution for other suitable assets. Use cached files where available. Choose texture resolution and mesh detail according to their effect in the final image rather than downloading the largest version by default. Do not purchase assets or introduce additional paid services.

Generative texture creation is optional and may be used only if the caller explicitly provides that capability. This scene does not require it. Missing authentication for an optional source should lead to another available candidate or a different optional prop, with any material limitation reported.

## 6. Target CAD Material: Fixed for This Request

The target is a newly machined, clean silver-gray aluminum component. It should have metallic reflections without a mirror finish, and no visible corrosion, oil contamination, dents, or coating.

Implement the target material in `build_material.py` using a Principled BSDF connected to Material Output. Use these fixed numeric socket values:

| Parameter | Value |
|---|---|
| Base Color | `(0.67, 0.70, 0.74, 1.0)` |
| Metallic | `1.0` |
| Roughness | `0.28` |
| IOR | `1.45` |
| Alpha | `1.0` |

Keep this target material constant while improving the environment. It must not require UV maps or texture images, and must not add surface wear layers, geometric displacement, or changes to the part itself. The host imports the actual part and applies the returned material.

## 7. Lighting, Reflections, and Exposure

Use a coherent daytime lighting setup appropriate to the workshop. A broad side window providing the main daylight, with weaker illumination from workshop ceiling fixtures, is a suitable starting point. You may adjust the arrangement to the chosen room layout; there is no required number of lights.

The main light should have a plausible spatial source. Its direction, size, intensity, and color should agree with the window or fixture that motivates it, whether that source is visible or outside the frame. Prefer a clear primary direction with useful reflected fill over uniformly illuminating the subject from every side.

Use the surrounding wall, tabletop, ceiling, and other relevant surfaces to create credible indirect light and reflection patterns. Include the off-camera environment needed by the reflective aluminum part. If an HDRI is used, align its illumination with room openings and other light sources; do not add incompatible dominant light directions.

Preserve contact shadows, depth, and a natural range of bright and dark areas. Metallic highlights may be strong locally, but the important visible geometry should remain readable across the fixed preview group. Avoid broad loss of detail from exposure, crushed cavities, featureless gray lighting, excessive colored illumination, and obvious rendering artifacts.

Design the light for the actual close-up and its dominant surfaces. Let useful highlights, shadows, and reflected dark/bright regions reveal the support and target geometry without flattening every object with uniform fill. A credible workshop need not be uniformly bright, but a dark artistic mood must not hide the component's features. Fog, floating dust, heavy blur, and theatrical effects are not required.

Judge materials and lighting together in actual renders before making final exposure and color-management adjustments. Do not compensate for an implausible reflection environment merely by adding more lights. Keep one coherent lighting setup for the reusable template rather than tuning it separately for each preview part.

## 8. Complete Background and Reference Images

The entire frame should belong to the same plausible workshop. Extend visible surfaces and necessary surroundings far enough to avoid exposed backdrop edges, missing walls, transparent gaps, or an unexplained empty background throughout the permitted support-height range. A visible window needs a plausible exterior.

Use any reference images attached to this request to guide spatial relationships, dominant surfaces, maintenance state, and light direction. Adapt the arrangement to the fixed camera and reserved part area. Do not reproduce people or obstructions that conflict with the task, paste the photo onto a flat backdrop to stand in for the whole scene, or claim to have recovered unseen geometry exactly.

If no images are attached, this written description is sufficient. Make consistent choices for unspecified details and briefly report the main assumptions with the delivery.

## 9. Preview and Delivery

First write the short `scene_plan.md`, then construct the template in the same task. Follow the supplied guide and use `quick` for useful feedback during construction or a host-requested repair. The host runs the complete `review` and resumes this session for dedicated read-only image inspection. In that stage, actually open the current CAD and empty-environment images and return the structured decision from `ACCEPTANCE.md`; leave edits for a subsequent repair task. Do not replace difficult test parts or present results from an earlier version as validation of the final files.

For this scene, pay particular attention to the 17.5 cm component-to-tool proportions, functional relationships visible in the close-up, the condition of the dominant tabletop and background surfaces, background coverage after height changes, and the readability of the fixed aluminum material under the lighting. Cite image-specific evidence. More objects, dirt, or larger textures do not by themselves establish realism.

Deliver the following in the caller's template directory:

- `build_scene.py` and `build_material.py`, using the required entry points.
- `scene_plan.md`, recording the scale conversion, work activity, prop groups, major surfaces, and light sources, kept consistent with the implemented scene.
- The external files actually used under `assets/`, with the tool-maintained `assets.json` containing source, license, and digest records. Resolve dependencies relative to the template so it does not require the shared cache.
- The build and preview artifacts produced by the supplied tools. Preserve their actual reports and any failed or missing samples; do not author a replacement report claiming checks passed.
- A concise delivery note with file and preview locations, the main design and scale assumptions, the images actually inspected, observed remaining issues, and any checks not performed.

Human visual approval happens after the host workflow ends. Report the technical outcome and visual observations separately, and leave human visual approval pending. If preview execution is unavailable, preserve the template and explicitly identify what remains unverified.
