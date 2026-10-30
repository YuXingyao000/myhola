"""Read source STEP, frozen Blender import code, and scene on CPU; render nothing.

Run with /mnt/d/anaconda3/envs/torch/bin/python. Blender executes this same file
with --blender-inspection to inspect its native data, without saving the scene.
"""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
RENDER = Path('/mnt/d/data/cad_scene_renders/office_step3_support_fix_20261026_001')
TEMPLATE = Path('/mnt/d/data/cad_scene_templates/office_20261023_001')
BLENDER = '/mnt/d/natural/blender-5.2.2-linux-x64/blender'
IDS = ['00182999', '00301161', '00269315']


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inspect_blender():
    import bpy
    import bmesh
    import numpy as np
    from mathutils import Matrix, Vector

    request = read(RENDER / 'request.json')
    bpy.ops.wm.open_mainfile(filepath=request['base_scene'])
    scene = bpy.context.scene
    scene.cycles.device = 'CPU'
    sys.path.insert(0, str(RENDER / 'source/step3'))
    from part import load_part, make_material

    floor = bpy.data.objects['500 mm carpet tile 5 5']
    material = floor.data.materials[0]
    nodes = material.node_tree.nodes
    shader = next(node for node in nodes if node.type == 'BSDF_PRINCIPLED')
    noise = next(node for node in nodes if node.type == 'TEX_NOISE')
    bump = next(node for node in nodes if node.type == 'BUMP')
    result = {
        'blender_version': bpy.app.version_string,
        'scene_file': request['base_scene'],
        'recorded_scene_sha256': request['scene_hash'],
        'color_management': {key: getattr(scene.view_settings, key)
                             for key in ['view_transform', 'look', 'exposure', 'gamma']},
        'world_background_strength': scene.world.node_tree.nodes['Background'].inputs['Strength'].default_value,
        'lights': [{'name': obj.name, 'type': obj.data.type, 'energy': obj.data.energy,
                    'color': list(obj.data.color), 'location': list(obj.location)}
                   for obj in scene.objects if obj.type == 'LIGHT'],
        'material_override': str(bpy.context.view_layer.material_override),
        'compositor_present': scene.compositing_node_group is not None,
        'floor': {
            'object': floor.name, 'material': material.name,
            'roughness': shader.inputs['Roughness'].default_value,
            'sheen_weight': shader.inputs['Sheen Weight'].default_value,
            'noise_scale': noise.inputs['Scale'].default_value,
            'bump_strength': bump.inputs['Strength'].default_value,
            'bump_distance': bump.inputs['Distance'].default_value,
            'semantic_material': 'procedural contract carpet tile, not vinyl',
        },
        'parts': [],
    }
    for model_id in IDS:
        metadata = read(RENDER / model_id / 'metadata.json')
        part = load_part(Path(metadata['stl_file']), metadata['smooth_angle_degrees'])
        part.matrix_world = Matrix(metadata['geometry']['model_to_world'])
        part.data.materials.clear()
        part.data.materials.append(make_material(metadata['material']))
        bpy.context.view_layer.update()
        mesh = part.data
        bmesh_data = bmesh.new()
        bmesh_data.from_mesh(mesh)
        topology = {
            'source_signed_volume': bmesh_data.calc_volume(signed=True),
            'boundary_edges': sum(edge.is_boundary for edge in bmesh_data.edges),
            'nonmanifold_edges': sum(not edge.is_manifold for edge in bmesh_data.edges),
        }
        bmesh_data.free()
        corners = np.array([normal.vector[:] for normal in mesh.corner_normals])
        deviations = []
        # The large visible front/back planes in these three samples face Y.
        for polygon in mesh.polygons:
            if abs(polygon.normal.y) > 1 - 1e-6:
                for index in polygon.loop_indices:
                    cosine = np.clip(corners[index] @ np.array(polygon.normal), -1, 1)
                    deviations.append(float(np.degrees(np.arccos(cosine))))
        camera = Vector(np.array(metadata['camera']['camera_to_world_blender'])[:3, 3])
        visible = []
        for polygon in mesh.polygons:
            center = part.matrix_world @ polygon.center
            normal = (part.matrix_world.to_3x3() @ polygon.normal).normalized()
            view = (camera - center).normalized()
            if normal.dot(view) > 0.05:
                visible.append((polygon.area * normal.dot(view), polygon, center, normal, view))
        rays = []
        for _, polygon, center, normal, view in sorted(visible, key=lambda row: row[0], reverse=True)[:6]:
            direction = (2 * normal.dot(view) * normal - view).normalized()
            hit, _, _, _, obj, _ = scene.ray_cast(
                bpy.context.evaluated_depsgraph_get(), center + normal * 0.00001,
                direction, distance=100)
            rays.append({'polygon': polygon.index, 'normal': list(normal),
                         'direction': list(direction), 'hit_object': obj.name if hit else None})
        shader = part.data.materials[0].node_tree.nodes['Principled BSDF']
        result['parts'].append({
            'model_id': model_id, 'dimensions_mm': [value * 1000 for value in part.dimensions],
            'vertices': len(mesh.vertices), 'triangles': len(mesh.polygons), **topology,
            'has_custom_normals': mesh.has_custom_normals,
            'sharp_edges': sum(edge.use_edge_sharp for edge in mesh.edges),
            'y_plane_corner_normal_max_error_degrees': max(deviations),
            'reflection_ray_samples': rays,
            'material': {**metadata['material'], 'normal_input_linked': shader.inputs['Normal'].is_linked,
                         'anisotropic': shader.inputs['Anisotropic'].default_value},
        })
        bpy.data.objects.remove(part, do_unlink=True)
    print('INPUT_AUDIT_JSON ' + json.dumps(result))


def run_audit():
    import numpy as np
    from PIL import Image
    import OCC
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.BRepCheck import BRepCheck_Analyzer
    from OCC.Core.Bnd import Bnd_Box
    from OCC.Core.BRepBndLib import brepbndlib
    from OCC.Core.GProp import GProp_GProps
    from OCC.Core.BRepGProp import brepgprop
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_SOLID

    parts = []
    for model_id in IDS:
        metadata = read(RENDER / model_id / 'metadata.json')
        step = Path('/mnt/d/data/deepcad_v6') / model_id / 'normalized_shape.step'
        reader = STEPControl_Reader()
        status = reader.ReadFile(str(step))
        reader.TransferRoots()
        shape = reader.OneShape()
        box = Bnd_Box()
        brepbndlib.AddOptimal(shape, box, False, False)
        bounds = np.array(box.Get())
        props = GProp_GProps()
        brepgprop.VolumeProperties(shape, props)
        counts = {}
        for name, kind in [('faces', TopAbs_FACE), ('solids', TopAbs_SOLID)]:
            explorer, count = TopExp_Explorer(shape, kind), 0
            while explorer.More():
                count += 1
                explorer.Next()
            counts[name] = count
        world_bounds = metadata['geometry']['world_bounds']
        low, high = np.array(world_bounds['minimum']), np.array(world_bounds['maximum'])
        center = (low + high) / 2
        positions = np.stack([center.copy(), center.copy()])
        positions[:, 1] = [low[1], high[1]]
        camera = np.c_[positions, np.ones(2)] @ np.array(metadata['camera']['world_to_camera_cv']).T
        uv = camera[:, :3] @ np.array(metadata['camera']['K']).T
        uv = uv[:, :2] / uv[:, 2:3]
        mask = np.asarray(Image.open(RENDER / model_id / 'mask.png')) > 0
        rgb = np.asarray(Image.open(RENDER / model_id / 'rgb.png'))
        parts.append({
            'model_id': model_id, 'step_file': str(step), 'step_sha256': sha(step),
            'step_read_status': int(status), 'step_valid': BRepCheck_Analyzer(shape).IsValid(),
            'step_counts': counts, 'step_bounds': bounds.tolist(), 'step_volume': props.Mass(),
            'physical_step_dimensions_mm': ((bounds[3:] - bounds[:3]) * metadata['geometry']['scale'] * 1000).tolist(),
            'y_thickness_projection_at_center_px': float(np.linalg.norm(uv[1] - uv[0])),
            'target_pixels': int(mask.sum()), 'target_rgb_median': np.median(rgb[mask], axis=0).tolist(),
            'target_rgb_std': rgb[mask].std(axis=0).tolist(),
            'stl_file': metadata['stl_file'], 'stl_sha256': sha(metadata['stl_file']),
            'stl_matches_manifest': sha(metadata['stl_file']) == metadata['source']['stl_sha256'],
            'mesh_conversion': metadata['source']['conversion'],
            'selected_mesh_source': metadata['source']['selected_source'],
        })
    process = subprocess.run(
        [BLENDER, '--background', '--factory-startup', '--threads', '2', '--python', str(Path(__file__)),
         '--', '--blender-inspection'],
        env=dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1'),
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True)
    result, = [json.loads(line.removeprefix('INPUT_AUDIT_JSON '))
               for line in process.stdout.splitlines() if line.startswith('INPUT_AUDIT_JSON ')]
    request = read(RENDER / 'request.json')
    frozen = {key: {'sha256': sha(RENDER / 'source' / key), 'matches_request': sha(RENDER / 'source' / key) == value}
              for key, value in request['source_hashes'].items()
              if key.endswith(('part.py', 'capture.py', 'preview.py'))}
    output = {
        'purpose': 'Read-only source geometry, shading and reflection diagnosis; no new render',
        'script': str(Path(__file__).resolve()), 'script_sha256': sha(__file__),
        'python': sys.executable, 'pythonocc_version': OCC.VERSION,
        'frozen_step3_source_checks': frozen, 'source_parts': parts, 'blender_inspection': result,
        'scene_build_source': str(TEMPLATE / 'build_scene.py'),
        'scene_build_source_sha256': sha(TEMPLATE / 'build_scene.py'),
        'prompt_mismatch': {
            'prompt_file': str(HERE / 'krea_office_matte.md'),
            'prompt_sha256': sha(HERE / 'krea_office_matte.md'),
            'prompt_floor_description': 'Matte gray vinyl flooring',
            'actual_floor': 'procedural contract carpet tiles',
            'consequence': 'The prompt requests a different material; floor smoothing is not evidence of restored carpet realism.',
            'experiment_handling': 'Keep this prompt unchanged for the current spatial-mechanism comparison; any later correction is a separate version.',
        },
        'limitations': [
            'Reflection rays sample ideal mirror directions at six triangle centers; rough BRDF integration, indirect bounces and light transport contributions were not measured.',
            'Projected Y thickness is measured through the bounding-box center, not a full per-pixel visible-side thickness estimate.',
            'Valid STEP, matching bounds and closed imported meshes do not certify full tessellation fidelity.',
            'Planar normal checks address these three samples, not the entire dataset; some curved facets appropriately have interpolated corner normals.',
            'No source geometry, material, camera, image or prompt was modified, and no GPU rendering was used.',
        ],
    }
    (HERE / 'input_material_audit.json').write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
    print(HERE / 'input_material_audit.json')


if __name__ == '__main__':
    if '--blender-inspection' in sys.argv:
        inspect_blender()
    else:
        run_audit()
