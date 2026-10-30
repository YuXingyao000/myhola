"""用小型确定性场景检验 Step 2 的几何语义和真实 CPU 渲染；不调用 LLM。"""

import json
from pathlib import Path
import sys

import bpy
from mathutils import Vector

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / "packages/cad_scene_factory/step2"))
from scene import apply_moves, measure_region, open_scene, validate_moves
from spatial import SceneGeometry, proxy_bounds, camera_spec, project_point
from preview import preview_placement

OUTPUT = Path(sys.argv[sys.argv.index("--") + 1]).resolve()
OUTPUT.mkdir(parents=True, exist_ok=True)
CONFIG = json.loads((ROOT / "packages/cad_scene_factory/step2/config.json").read_text())
CONFIG.update(resolution=[96, 96], frame_margin_px=3, samples=4, threads=2)


def box(name, center, dimensions):
    bpy.ops.mesh.primitive_cube_add(size=1, location=center)
    item = bpy.context.object
    item.name, item.dimensions = name, dimensions
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    return item


def reset():
    open_scene(OUTPUT / "fixture.blend")


def write(name, data):
    (OUTPUT / name).write_text(json.dumps(data, indent=2) + "\n")


bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.context.scene.unit_settings.system = "METRIC"
box("Table", (0, 0, 0.75), (1.4, 1.0, 0.1))
body = box("Loose box", (0, 0, 0.83), (0.10, 0.08, 0.06))
label = box("Loose box label", (0, 0, 0.862), (0.07, 0.06, 0.004))
matrix = label.matrix_world.copy()
label.parent = body
label.matrix_world = matrix
# 一个合并外壳：AABB 包含整个拍摄区域，但实际墙/地不应阻挡桌面。
mesh = bpy.data.meshes.new("Shell mesh")
mesh.from_pydata([(-2,-2,0),(2,-2,0),(2,2,0),(-2,2,0),(-2,2,3),(2,2,3)], [],
                 [(0,1,2,3),(3,2,5,4)])
shell = bpy.data.objects.new("Merged shell", mesh)
bpy.context.scene.collection.objects.link(shell)
lamp = bpy.data.objects.new("Window light", bpy.data.lights.new("Window light", "AREA"))
bpy.context.scene.collection.objects.link(lamp)
lamp.location, lamp.data.energy, lamp.data.shape, lamp.data.size = (0, -1, 3), 250, "DISK", 2
lamp.rotation_euler = (Vector((0,0,.8))-lamp.location).to_track_quat('-Z','Y').to_euler()
world = bpy.data.worlds.new("World")
world.use_nodes = True
world.node_tree.nodes["Background"].inputs["Strength"].default_value = .3
bpy.context.scene.world = world
bpy.context.view_layer.update()
bpy.ops.wm.save_as_mainfile(filepath=str(OUTPUT / "fixture.blend"))

region = {"region_id":"r1", "support_object_id":"Table", "view_id":"table", "image_region_px":[15,15,81,81],
          "reason":"occupied work surface", "movable_groups":[{"id":"box", "members":["Loose box","Loose box label"],
          "support_object_id":"Table", "reason":"loose complete box"}]}
proposal = {"region_id":"r1", "support_point_world":[0,0,.8], "camera_position_world":[.45,-.55,1.2],
            "look_at_world":[0,0,.9], "moves":[{"group_id":"box","pivot_world":[0,0,.8],
            "translation_xy_m":[.25,.05],"yaw_degrees":15}], "retain_in_frame":["box"], "reason":"clear center"}
view = {"id":"table", "position":proposal["camera_position_world"], "target":proposal["look_at_world"],
        "vertical_fov_degrees":45, "resolution":[96,96]}
write("views.json", {"views":[view]})
write("region.json", region)
write("proposal.json", proposal)
results = {}
geometry = SceneGeometry(CONFIG)
assert len(geometry.items["Merged shell"]["parts"]) == 2
measurement = measure_region(geometry, region, view, CONFIG)
assert measurement["counts"]["movable"] > 0 and measurement["counts"]["free"] > 0
results["occupied_and_free_measured"] = measurement["counts"]

before, after = apply_moves(region, proposal)
expected = Vector((.25,.05,0))
assert (after["Loose box"].translation - before["Loose box"].translation - expected).length < 1e-6
relative_before = before["Loose box"].inverted() @ before["Loose box label"]
relative_after = after["Loose box"].inverted() @ after["Loose box label"]
assert max(abs(a-b) for ra,rb in zip(relative_before,relative_after) for a,b in zip(ra,rb)) < 1e-6
reset()
_, repeated = apply_moves(region, proposal)
assert (repeated["Loose box"].translation - after["Loose box"].translation).length < 1e-6
results["whole_group_and_base_relative"] = True

reset()
incomplete = {**region, "movable_groups":[{**region["movable_groups"][0], "members":["Loose box"]}]}
assert any("incomplete_parent" in reason for reason in validate_moves(incomplete, proposal, CONFIG))
results["missing_child_rejected"] = True

reset()
no_move = {**proposal, "moves":[], "retain_in_frame":[]}
occupied = preview_placement(region, no_move, CONFIG, OUTPUT)
assert not occupied["numeric_pass"] and "proxy_collision" in occupied["reasons"]
results["occupied_proxy_rejected"] = occupied["reasons"]

reset()
# 落点就在桌边：中心有支撑，但 24 cm 占地越界，应在渲染前拒绝。
edge = preview_placement(region, {**proposal, "support_point_world":[.68,0,.8]}, CONFIG, OUTPUT)
assert not edge["numeric_pass"] and "proxy_support" in edge["reasons"]
results["support_edge_rejected"] = edge["reasons"]

reset()
bpy.data.objects["Loose box"].location.x = .6
bpy.context.view_layer.update()
off_table = preview_placement(region, proposal, CONFIG, OUTPUT)
assert not off_table["numeric_pass"] and not off_table["moves"][0]["support_after"]["pass"]
results["moved_group_off_support_rejected"] = True

reset()
mesh = bpy.data.meshes.new("Holed support mesh")
vertices, faces = [], []
for x0,y0,x1,y1 in [(-.3,-.3,-.025,.3),(.025,-.3,.3,.3),(-.025,-.3,.025,-.025),(-.025,.025,.025,.3)]:
    offset = len(vertices)
    vertices.extend([(x0,y0,.8),(x1,y0,.8),(x1,y1,.8),(x0,y1,.8)])
    faces.append(tuple(range(offset,offset+4)))
mesh.from_pydata(vertices,[],faces)
surface = bpy.data.objects.new("Holed support",mesh)
bpy.context.scene.collection.objects.link(surface)
bpy.context.view_layer.update()
hole = SceneGeometry(CONFIG).supported("Holed support",proxy_bounds([0,0,.8],CONFIG))
assert not hole["pass"] and hole["unsupported_samples"] > 0
results["interior_support_hole_rejected"] = hole

reset()
apply_moves(region, proposal)
occluder = box("Center occluder", (.225,-.275,1.05), (.02,.02,.02))
bpy.context.view_layer.update()
blocked = SceneGeometry(CONFIG).blockers(proxy_bounds([0,0,.8],CONFIG), camera_spec(view["position"],view["target"],CONFIG))
assert "Center occluder" in blocked
results["central_occluder_rejected"] = blocked

reset()
result = preview_placement(region, proposal, CONFIG, OUTPUT)
write("preview_report.json", result)
assert result["numeric_pass"], result
assert result["visibility"][0]["visible_pixels"] > 0
assert len(result["images"]) == 2
results["real_render_and_visible_group"] = result["visibility"]

reset()
box("Offscreen prop", (2,0,.83), (.1,.08,.06))
bpy.context.view_layer.update()
extra = {"id":"offscreen", "members":["Offscreen prop"], "support_object_id":"Table", "reason":"visibility test"}
mask_dir = OUTPUT / "offscreen"
mask_dir.mkdir(exist_ok=True)
hidden = preview_placement({**region,"movable_groups":[*region["movable_groups"],extra]},
                           {**proposal,"retain_in_frame":["box","offscreen"]},CONFIG,mask_dir)
assert not hidden["numeric_pass"] and "offscreen: no_visible_pixels" in hidden["reasons"]
results["offscreen_retained_group_rejected"] = True
write("checks.json", results)
print("STEP2_CHECKS_PASSED", json.dumps(results))
