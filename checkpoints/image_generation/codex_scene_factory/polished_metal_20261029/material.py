"""只比较粗糙度：原值、素材均值、素材贴图。"""

import bpy


def make_material(part, base, profile):
    material = bpy.data.materials.new("polished_" + profile["name"])
    material.use_nodes = True
    nodes, links = material.node_tree.nodes, material.node_tree.links
    shader = nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = base["base_color_linear"]
    shader.inputs["Metallic"].default_value = base["metallic"]
    shader.inputs["Roughness"].default_value = profile["roughness"]

    if profile.get("roughness_map"):
        coordinate = nodes.new("ShaderNodeTexCoord")
        frequency = nodes.new("ShaderNodeVectorMath")
        frequency.operation = "MULTIPLY"
        frequency.inputs[1].default_value = [
            axis.length / profile["tile_width_m"] for axis in part.matrix_world.to_3x3().col
        ]
        links.new(coordinate.outputs["Object"], frequency.inputs[0])
        texture = nodes.new("ShaderNodeTexImage")
        texture.image = bpy.data.images.load(profile["roughness_map"], check_existing=False)
        texture.image.colorspace_settings.name = "Non-Color"
        texture.projection = "BOX"
        texture.projection_blend = .2
        texture.interpolation = "Linear"
        texture.extension = "REPEAT"
        links.new(frequency.outputs["Vector"], texture.inputs["Vector"])
        links.new(texture.outputs["Color"], shader.inputs["Roughness"])

    assert not shader.inputs["Normal"].is_linked
    assert not nodes.get("Material Output").inputs["Displacement"].is_linked
    return material
