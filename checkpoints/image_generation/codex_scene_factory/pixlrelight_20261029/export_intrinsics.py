"""Blender-only PIXLRelight inputs: fixed RGB source and actual lighting AOVs.

Identity preserves original illumination. key_source_01 changes source intensity
using the previously fixed coefficients, then traces paths again. It never edits
the source .blend, CAD, camera, surface BRDFs, or production code.
"""

import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time

import bpy
from mathutils import Matrix
import numpy as np


SOURCE_REQUEST = Path('/mnt/d/data/cad_scene_enhancements/lut_sweep_20261029_001/request.json')
SOURCE_GROUPS = Path('/mnt/d/data/cad_scene_enhancements/light_mix_20261029_sources_001/office/00301161/metadata.json')
PRESETS = Path('/mnt/d/data/cad_scene_enhancements/light_mix_20261029_sources_001/presets.json')
DEFAULT_LIGHT_TOOLS = Path(__file__).parent.parent / 'light_mix_20261029' / 'export_lights.py'
PASSES = {
    'diffuse_color': 'Diffuse Color', 'diffuse_direct': 'Diffuse Direct',
    'diffuse_indirect': 'Diffuse Indirect', 'glossy_color': 'Glossy Color',
    'glossy_direct': 'Glossy Direct', 'glossy_indirect': 'Glossy Indirect',
    'emission': 'Emission', 'environment': 'Environment',
    'combined_noisy': 'Noisy Image', 'combined_denoised': 'Image',
    'normal': 'Normal', 'denoising_normal': 'Denoising Normal',
    'denoising_albedo': 'Denoising Albedo', 'depth': 'Depth',
}


def import_helpers(path):
    spec = importlib.util.spec_from_file_location('light_export_helpers', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def private_node_groups(tree):
    """A copied material still shares nested groups; isolate before scaling."""
    for node in tree.nodes:
        if node.type == 'GROUP':
            node.node_tree = node.node_tree.copy()
            private_node_groups(node.node_tree)


def scale_emission(tree, factor, prefix=''):
    """Multiply emission strength only, preserving its existing texture/control."""
    changes = []
    for node in list(tree.nodes):
        name = prefix + node.name
        socket = None
        if node.type in {'EMISSION', 'BACKGROUND'}:
            socket = node.inputs['Strength']
        elif node.type == 'BSDF_PRINCIPLED':
            socket = node.inputs['Emission Strength']
        if socket is not None:
            if socket.is_linked:
                link = socket.links[0]
                source_node, source_socket = link.from_node.name, link.from_socket.name
                multiplier = tree.nodes.new('ShaderNodeMath')
                multiplier.operation = 'MULTIPLY'
                multiplier.inputs[1].default_value = factor
                tree.links.new(link.from_socket, multiplier.inputs[0])
                tree.links.new(multiplier.outputs[0], socket)
                changes.append({'node': name, 'socket': socket.name, 'factor': factor,
                                'linked_source': [source_node, source_socket], 'method': 'insert Multiply'})
            else:
                before = socket.default_value
                socket.default_value = before * factor
                changes.append({'node': name, 'socket': socket.name, 'factor': factor,
                                'before': before, 'after': socket.default_value})
        if node.type == 'GROUP':
            changes.extend(scale_emission(node.node_tree, factor, name + '/'))
    return changes


def apply_lighting(grouping, weights, helper):
    """Copy light/material datablocks so shared assets cannot cross-contaminate."""
    changes = []
    for item in grouping['lights']:
        obj = bpy.data.objects[item['name']]
        factor = weights[item['group']]
        before = obj.data.energy
        obj.data = obj.data.copy()
        obj.data.energy = before * factor
        changes.append({'kind': 'light_energy', 'object': obj.name, 'group': item['group'],
                        'factor': factor, 'before': before, 'after': obj.data.energy})
    for item in grouping['emissive_material_objects']:
        obj = bpy.data.objects[item['object']]
        names = {material['material'] for material in item['materials']}
        factor = weights[item['group']]
        for slot in obj.material_slots:
            if slot.material is None or slot.material.name not in names:
                continue
            original = slot.material
            material = original.copy()
            private_node_groups(material.node_tree)
            before = helper.node_description(original.node_tree)
            edits = scale_emission(material.node_tree, factor)
            # Object overrides do not mutate a shared mesh's material slots.
            slot.link = 'OBJECT'
            slot.material = material
            changes.append({'kind': 'mesh_emission', 'object': obj.name, 'group': item['group'],
                            'factor': factor, 'source_material': original.name,
                            'private_material': material.name, 'edits': edits,
                            'original_nodes': before, 'modified_nodes': helper.node_description(material.node_tree)})
    scene = bpy.context.scene
    scene.world = scene.world.copy()
    private_node_groups(scene.world.node_tree)
    changes.append({'kind': 'world_emission', 'factor': weights['world'],
                    'edits': scale_emission(scene.world.node_tree, weights['world'])})
    return changes


def add_intrinsic_outputs(folder):
    scene, layer = bpy.context.scene, bpy.context.view_layer
    for name in ('diffuse_color', 'diffuse_direct', 'diffuse_indirect',
                 'glossy_color', 'glossy_direct', 'glossy_indirect',
                 'emit', 'environment', 'normal', 'z', 'object_index'):
        setattr(layer, 'use_pass_' + name, True)
    layer.cycles.denoising_store_passes = True
    layer.update_render_passes()
    # The frozen office scene has no compositor; preserve its direct Image path.
    assert scene.compositing_node_group is None
    tree = bpy.data.node_groups.new('intrinsic_outputs', 'CompositorNodeTree')
    scene.compositing_node_group = tree
    scene.render.use_compositing = True
    tree.interface.new_socket(name='Image', in_out='OUTPUT', socket_type='NodeSocketColor')
    layers = tree.nodes.new('CompositorNodeRLayers')
    layers.layer = layer.name
    output = tree.nodes.new('NodeGroupOutput')
    tree.links.new(layers.outputs['Image'], output.inputs['Image'])
    for name, socket_name in PASSES.items():
        output = tree.nodes.new('CompositorNodeOutputFile')
        output.directory = str(folder)
        output.file_name = name + '_'
        output.format.media_type = 'IMAGE'
        output.format.file_format = 'OPEN_EXR'
        output.format.color_mode = 'RGBA'
        output.format.color_depth = '32'
        output.format.exr_codec = 'ZIP'
        output.save_as_render = False
        output.file_output_items.new('RGBA', 'Image')
        tree.links.new(layers.outputs[socket_name], output.inputs[0])
    mask = tree.nodes.new('CompositorNodeIDMask')
    mask.inputs['Index'].default_value = 1
    mask.inputs['Anti-Alias'].default_value = False
    tree.links.new(layers.outputs['Object Index'], mask.inputs[0])
    output = tree.nodes.new('CompositorNodeOutputFile')
    output.directory = str(folder)
    output.file_name = 'mask_'
    output.format.media_type = 'IMAGE'
    output.format.file_format = 'PNG'
    output.format.color_mode = 'BW'
    output.format.color_depth = '8'
    output.save_as_render = False
    output.file_output_items.new('FLOAT', 'mask')
    tree.links.new(mask.outputs[0], output.inputs[0])
    return [socket.name for socket in layers.outputs]


def run(output_dir, condition, light_tools):
    helper = import_helpers(light_tools)
    request = json.loads(SOURCE_REQUEST.read_text())
    group_metadata = json.loads(SOURCE_GROUPS.read_text())
    grouping = group_metadata['light_groups']
    presets = json.loads(PRESETS.read_text())
    weights = {name: 1.0 for name in grouping['groups']} if condition == 'identity' else \
        presets['presets'][condition]['weights']
    source_package = Path(request['source_package'])
    sys.path[:0] = [str(source_package/'step3'), str(source_package/'step2'), str(source_package)]
    from part import load_part, make_material
    from preview import configure_render, make_camera

    original = Path(request['output_dir'])/'raw'/'office'
    prior_file = original/'00301161'/'metadata.json'
    prior = json.loads(prior_file.read_text())
    source_image = original/'00301161'/'rgb.png'
    source_scene = original/'prepared_scene.blend'
    folder = output_dir/'inputs'/condition
    folder.mkdir(parents=True, exist_ok=False)
    shutil.copy2(source_image, folder/'source.png')
    bpy.ops.wm.open_mainfile(filepath=str(source_scene))
    scene = bpy.context.scene
    config = dict(request['render_config'], samples=64, threads=8, seed=8)
    configure_render(config)
    scene.cycles.use_denoising = True
    scene.cycles.use_adaptive_sampling = False
    part = load_part(str(Path(request['stl_dir'])/'00301161.stl'), 30)
    part.matrix_world = Matrix(prior['geometry']['model_to_world'])
    part.data.materials.clear()
    part.data.materials.append(make_material(prior['material']))
    matrix = Matrix(prior['camera']['camera_to_world_blender'])
    position = matrix.translation
    camera = make_camera({'camera_position_world': list(position),
                          'look_at_world': [position[0],position[1],position[2]-1]}, config)
    camera.matrix_world = matrix
    for obj in bpy.data.objects:
        obj.pass_index = 0
    part.pass_index = 1
    bpy.context.view_layer.update()
    changes = [] if condition == 'identity' else apply_lighting(grouping, weights, helper)
    sockets = add_intrinsic_outputs(folder)
    scene.render.filepath = str(folder/'target.png')
    record = {'status': 'prepared', 'condition': condition, 'scene_id': 'office', 'model_id': '00301161',
              'source_image': str(source_image), 'source_image_sha256': helper.sha256(source_image),
              'source_scene': str(source_scene), 'source_scene_sha256': helper.sha256(source_scene),
              'source_metadata': str(prior_file), 'source_metadata_sha256': helper.sha256(prior_file),
              'source_request_sha256': helper.sha256(SOURCE_REQUEST),
              'group_metadata_sha256': helper.sha256(SOURCE_GROUPS), 'preset_sha256': helper.sha256(PRESETS),
              'export_script_sha256': helper.sha256(__file__), 'helper_sha256': helper.sha256(light_tools),
              'blender': bpy.app.version_string, 'device': 'CPU', 'threads': 8, 'samples': 64, 'seed': 8,
              'resolution': [512,512], 'passes': PASSES, 'actual_sockets': sockets,
              'source_group_mapping': grouping, 'weights': weights, 'lighting_changes': changes,
              'geometry': prior['geometry'], 'camera': prior['camera'], 'material': prior['material'],
              'color_management': group_metadata['original_settings']['color_management'],
              'cycles': {name:getattr(scene.cycles,name) for name in
                         ('max_bounces','diffuse_bounces','glossy_bounces','transmission_bounces',
                          'volume_bounces','transparent_max_bounces','sample_clamp_direct',
                          'sample_clamp_indirect','light_sampling_threshold','blur_glossy')},
              'semantics': 'EXRs are scene-linear float32 Blender Cycles passes. Diffuse/glossy color are BSDF-color passes, '
                           'not camera sRGB. Direct/indirect passes are color-separated lighting. source.png is the '
                           'unchanged original office RGB in both conditions; target.png is this condition re-rendered '
                           'through original AgX. This supplies target illumination, not a synthetic-to-real ground truth.'}
    helper.write_json(folder/'metadata.json',record)
    started = time.monotonic()
    bpy.ops.render.render(write_still=True)
    for name in PASSES:
        path, = folder.glob(name+'_*.exr')
        path.rename(folder/(name+'.exr'))
    path, = folder.glob('mask_*.png')
    path.rename(folder/'mask.png')
    mask = helper.read_pixels(folder/'mask.png')[:,:,0]
    old_mask = helper.read_pixels(original/'00301161'/'mask.png')[:,:,0]
    statistics = {}
    for name in PASSES:
        array = helper.read_pixels(folder/(name+'.exr'))[:,:,:3]
        statistics[name] = {'finite':bool(np.isfinite(array).all()),
                            'min':float(array.min()),'max':float(array.max()),
                            'mean_rgb':array.mean(axis=(0,1),dtype=np.float64).tolist()}
    record.update(status='completed',render_seconds=time.monotonic()-started,
                  mask_changed_pixels=int(np.count_nonzero(mask!=old_mask)),pass_statistics=statistics,
                  source_scene_unchanged=helper.sha256(source_scene)==record['source_scene_sha256'],
                  actual_camera_matrix=[list(row) for row in camera.matrix_world],
                  actual_model_to_world=[list(row) for row in part.matrix_world])
    helper.write_json(folder/'metadata.json',record)
    print(json.dumps({key:record[key] for key in ('condition','status','render_seconds','mask_changed_pixels')}),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output_dir',type=Path,required=True)
    parser.add_argument('--condition',choices=('identity','key_source_01'),required=True)
    parser.add_argument('--light_tools',type=Path,default=DEFAULT_LIGHT_TOOLS)
    arguments = parser.parse_args(sys.argv[sys.argv.index('--')+1:])
    run(arguments.output_dir.resolve(),arguments.condition,arguments.light_tools.resolve())
