"""Blender 小场景：共用网格/材质的非支撑对象不能被一起修改。"""

import hashlib
import json
from pathlib import Path
import sys

import bpy

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "packages/cad_scene_factory/step2"))
from scene import prepare_support_scene
sys.path.insert(0, str(ROOT / "packages/cad_scene_factory/step3"))
from part import check_support_contact

output = Path(sys.argv[sys.argv.index("--") + 1])
output.mkdir(parents=True, exist_ok=False)
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.mesh.primitive_cube_add()
support = bpy.context.object
support.name = "support"
other = bpy.data.objects.new("shared_mesh_other", support.data)
bpy.context.collection.objects.link(other)
material = bpy.data.materials.new("shared_displacement")
material.use_nodes = True
material.displacement_method = "DISPLACEMENT"
support.data.materials.append(material)
bpy.ops.wm.save_as_mainfile(filepath=str(output / "original.blend"))
original_hash = hashlib.sha256((output / "original.blend").read_bytes()).hexdigest()
points = [tuple(vertex.co) for vertex in support.data.vertices]
report = prepare_support_scene([{"support_object_id": "support"}], output / "prepared.blend")
assert support.data == other.data
assert support.material_slots[0].material.displacement_method == "BUMP"
assert other.material_slots[0].material == material and material.displacement_method == "DISPLACEMENT"
assert points == [tuple(vertex.co) for vertex in support.data.vertices]
assert original_hash == hashlib.sha256((output / "original.blend").read_bytes()).hexdigest()
bpy.ops.wm.open_mainfile(filepath=str(output / "prepared.blend"))
assert bpy.data.objects["support"].material_slots[0].material.displacement_method == "BUMP"
assert bpy.data.objects["shared_mesh_other"].material_slots[0].material.displacement_method == "DISPLACEMENT"
support = bpy.data.objects["support"]
support.location.z = 1.75
bpy.context.view_layer.update()
validation = {"rendered_height_m": {"median": .75}, "tolerance_m": .002}
assert check_support_contact(support, validation)["pass"]
validation["rendered_height_m"]["median"] = .75485
try:
    check_support_contact(support, validation)
except ValueError:
    pass
else:
    raise AssertionError("实际底端必须拒绝 4.85 mm 的嵌入")
(output / "checks.json").write_text(json.dumps({"status": "pass", "shared_material_isolated": True,
                                                "mesh_unchanged": True, "original_scene_unchanged": True,
                                                "prepared_material_survives_reload": True,
                                                "contact_accepts_aligned_and_rejects_buried": True}, indent=2) + "\n")
print("SUPPORT_MATERIAL_CHECK_PASSED")
