"""仅改变反射分布：不连接法线、位移、底色或透明度。"""

import bpy


def make_material(part, base, profile):
    material = bpy.data.materials.new("reflectance_" + profile["name"])
    material.use_nodes = True
    nodes, links = material.node_tree.nodes, material.node_tree.links
    shader = nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = base["base_color_linear"]
    shader.inputs["Metallic"].default_value = base["metallic"]
    shader.inputs["Roughness"].default_value = base["roughness"]

    if profile["roughness_amplitude"]:
        # STL 无UV。物体坐标乘实际缩放后变为毫米，再定义各方向纹理尺度。
        coordinate = nodes.new("ShaderNodeTexCoord")
        frequency = nodes.new("ShaderNodeVectorMath")
        frequency.operation = "MULTIPLY"
        frequency.inputs[1].default_value = [
            axis.length * 1000 / wavelength
            for axis, wavelength in zip(part.matrix_world.to_3x3().col, profile["wavelengths_mm"])
        ]
        links.new(coordinate.outputs["Object"], frequency.inputs[0])
        noise = nodes.new("ShaderNodeTexNoise")
        noise.noise_dimensions = "3D"
        noise.inputs["Scale"].default_value = 1
        noise.inputs["Detail"].default_value = 1
        links.new(frequency.outputs["Vector"], noise.inputs["Vector"])
        variation = nodes.new("ShaderNodeMath")
        variation.operation = "MULTIPLY_ADD"
        variation.inputs[1].default_value = 2 * profile["roughness_amplitude"]
        variation.inputs[2].default_value = base["roughness"] - profile["roughness_amplitude"]
        links.new(noise.outputs["Fac"], variation.inputs[0])
        links.new(variation.outputs[0], shader.inputs["Roughness"])

    if profile["anisotropy"]:
        # 表面切线 T = normalize(N × (D × N))，D为局部X轴的世界方向。
        direction = part.matrix_world.to_3x3().col[0].normalized()
        geometry = nodes.new("ShaderNodeNewGeometry")
        cross = nodes.new("ShaderNodeVectorMath")
        cross.operation = "CROSS_PRODUCT"
        cross.inputs[0].default_value = direction
        links.new(geometry.outputs["Normal"], cross.inputs[1])
        tangent = nodes.new("ShaderNodeVectorMath")
        tangent.operation = "CROSS_PRODUCT"
        links.new(geometry.outputs["Normal"], tangent.inputs[0])
        links.new(cross.outputs["Vector"], tangent.inputs[1])
        normalize = nodes.new("ShaderNodeVectorMath")
        normalize.operation = "NORMALIZE"
        links.new(tangent.outputs["Vector"], normalize.inputs[0])
        links.new(normalize.outputs["Vector"], shader.inputs["Tangent"])

        # N平行D时各向异性归零；不需要任意切换另一根轴。
        dot = nodes.new("ShaderNodeVectorMath")
        dot.operation = "DOT_PRODUCT"
        dot.inputs[1].default_value = direction
        links.new(geometry.outputs["Normal"], dot.inputs[0])
        square = nodes.new("ShaderNodeMath")
        square.operation = "MULTIPLY"
        links.new(dot.outputs["Value"], square.inputs[0])
        links.new(dot.outputs["Value"], square.inputs[1])
        strength = nodes.new("ShaderNodeMath")
        strength.operation = "MULTIPLY_ADD"
        strength.use_clamp = True
        strength.inputs[1].default_value = -profile["anisotropy"]
        strength.inputs[2].default_value = profile["anisotropy"]
        links.new(square.outputs[0], strength.inputs[0])
        links.new(strength.outputs[0], shader.inputs["Anisotropic"])

    assert not shader.inputs["Normal"].is_linked
    assert not nodes.get("Material Output").inputs["Displacement"].is_linked
    return material
