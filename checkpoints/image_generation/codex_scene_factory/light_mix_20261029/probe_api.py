"""Read-only Blender API/source probe for experimental light grouping."""
import bpy
import json
from pathlib import Path

root = Path('/mnt/d/data/cad_scene_enhancements/lut_sweep_20261029_001/raw')
for scene_id in ('office', 'factory', 'assembly'):
    bpy.ops.wm.open_mainfile(filepath=str(root / scene_id / 'prepared_scene.blend'))
    scene = bpy.context.scene
    layer = bpy.context.view_layer
    print('SCENE', scene_id)
    print('GROUP_API', hasattr(layer, 'lightgroups'), hasattr(scene.world, 'lightgroup'),
          hasattr(next(iter(scene.objects)), 'lightgroup'))
    print('LIGHTS', json.dumps([{'name': o.name, 'type': o.data.type, 'energy': o.data.energy,
                               'position': list(o.matrix_world.translation), 'hidden': o.hide_render}
                              for o in scene.objects if o.type == 'LIGHT']))
    emission_materials = set()
    for m in bpy.data.materials:
        if not m.use_nodes:
            continue
        for n in m.node_tree.nodes:
            if n.type == 'EMISSION' or (n.type == 'BSDF_PRINCIPLED' and
                    (n.inputs['Emission Color'].is_linked or
                     max(n.inputs['Emission Color'].default_value[:3]) > 0) and
                    (n.inputs['Emission Strength'].is_linked or
                     n.inputs['Emission Strength'].default_value > 0)):
                emission_materials.add(m.name)
    print('EMISSION', json.dumps([{'name': o.name, 'materials': [m.name for m in o.data.materials if m]}
                                  for o in scene.objects if o.type == 'MESH' and
                                  any(m and m.name in emission_materials for m in o.data.materials)]))
    print('COMPOSITOR', scene.render.use_compositing,
          [(n.name, n.bl_idname) for n in scene.compositing_node_group.nodes] if scene.compositing_node_group else None)
    bpy.ops.scene.view_layer_add_lightgroup(name='probe_daylight')
    layer.use_pass_normal = True
    layer.cycles.denoising_store_passes = True
    scene.world.lightgroup = 'probe_daylight'
    layer.update_render_passes()
    tree = bpy.data.node_groups.new('probe', 'CompositorNodeTree')
    scene.compositing_node_group = tree
    node = tree.nodes.new('CompositorNodeRLayers')
    print('SOCKETS', [(s.name, s.identifier) for s in node.outputs])
    print('OUTPUT_ITEMS', [p.identifier for p in bpy.types.CompositorNodeOutputFile.bl_rna.properties])
