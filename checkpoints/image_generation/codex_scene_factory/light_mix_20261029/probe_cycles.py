"""Record actual path-tracing limits after the frozen renderer configuration."""
import json
from pathlib import Path
import sys

import bpy

source = Path('/mnt/d/data/cad_scene_enhancements/lut_sweep_20261029_001')
destination = Path('/mnt/d/data/cad_scene_enhancements/light_mix_20261029_pilot_001')
request = json.loads((source / 'request.json').read_text())
package = Path(request['source_package'])
sys.path[:0] = [str(package/'step2'), str(package)]
from preview import configure_render

records = []
fields = ['max_bounces', 'diffuse_bounces', 'glossy_bounces', 'transmission_bounces',
          'volume_bounces', 'transparent_max_bounces', 'min_light_bounces',
          'min_transparent_bounces', 'sample_clamp_direct', 'sample_clamp_indirect',
          'light_sampling_threshold', 'use_light_tree', 'use_guiding',
          'caustics_reflective', 'caustics_refractive', 'blur_glossy',
          'use_fast_gi', 'ao_bounces', 'ao_bounces_render',
          'use_denoising', 'denoiser', 'denoising_use_gpu',
          'use_adaptive_sampling', 'adaptive_threshold', 'adaptive_min_samples',
          'samples', 'seed', 'use_animated_seed', 'device']
for scene_id in ('office', 'factory', 'assembly'):
    path = source/'raw'/scene_id/'prepared_scene.blend'
    bpy.ops.wm.open_mainfile(filepath=str(path))
    scene = bpy.context.scene
    original = {name: getattr(scene.cycles, name) for name in fields if hasattr(scene.cycles, name)}
    configure_render(dict(request['render_config'], samples=64, threads=8, seed=8))
    scene.cycles.use_denoising = True
    scene.cycles.use_adaptive_sampling = False
    tree = bpy.data.node_groups.new('settings_probe', 'CompositorNodeTree')
    output = tree.nodes.new('CompositorNodeOutputFile')
    output.format.media_type = 'IMAGE'
    output.format.file_format = 'OPEN_EXR'
    output.format.color_depth = '32'
    record = {'scene_id': scene_id, 'source_blend': str(path),
              'original_cycles': original,
              'export_cycles': {name: getattr(scene.cycles, name) for name in fields if hasattr(scene.cycles, name)},
              'film': {name: getattr(scene.render, name) for name in
                       ('film_transparent', 'filter_size', 'threads_mode', 'threads',
                        'compositor_device', 'compositor_denoise_device')},
              'cycles_film': {name: getattr(scene.cycles, name) for name in
                              ('film_transparent_glass', 'film_transparent_roughness')
                              if hasattr(scene.cycles, name)},
              'color_management': {'view_transform': scene.view_settings.view_transform,
                                   'look': scene.view_settings.look, 'exposure': scene.view_settings.exposure,
                                   'gamma': scene.view_settings.gamma,
                                   'display_device': scene.display_settings.display_device,
                                   'exr_linear_colorspace': output.format.linear_colorspace_settings.name},
              'scene_color_properties': [prop.identifier for prop in scene.bl_rna.properties if 'color' in prop.identifier],
              'view_color_properties': [prop.identifier for prop in scene.view_settings.bl_rna.properties if 'color' in prop.identifier]}
    records.append(record)
result = {'status': 'read_only_probe', 'rendered': False, 'blender': bpy.app.version_string,
          'scope': 'Freshly open the same prepared_scene.blend and apply the same frozen configure_render helper; no rendering or source writes.',
          'linearity_limit': 'Sum(groups)=Combined only validates the original weight-one decomposition. Finite bounce limits truncate transport but do not alone break linearity of fixed transport. Path clamping, sampling thresholds, and denoising mean large light-weight changes are not guaranteed equivalent to a newly traced physical scene.',
          'scenes': records}
(destination/'cycles_settings.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
