"""固定六件实验的纯程序材质；构造后返回，不改网格、变换或材质槽。"""

import copy

import bpy


def _node_metadata(node):
    """记录实际写入 Blender 的值，保留浮点精度与插槽 identifier。"""
    result = {"name": node.name, "type": node.bl_idname, "inputs": []}
    for property_name in (
        "operation", "use_clamp", "noise_dimensions", "noise_type", "normalize",
        "distribution", "subsurface_method",
    ):
        if hasattr(node, property_name):
            result[property_name] = getattr(node, property_name)
    for socket in node.inputs:
        entry = {
            "name": socket.name, "identifier": socket.identifier,
            "linked": socket.is_linked,
        }
        if hasattr(socket, "default_value"):
            value = socket.default_value
            entry["default_value"] = (
                list(value) if hasattr(value, "__len__") and not isinstance(value, str)
                else value
            )
        result["inputs"].append(entry)
    return result


def make_finish(part: bpy.types.Object, config: dict) -> tuple[bpy.types.Material, dict]:
    """返回新材质与 JSON 可序列化的参数记录；调用者负责明确替换材质槽。

    Step3 将一世界坐标单位约定为一米。Object 纹理坐标处于 STL 局部空间，
    因此逐轴乘 matrix_world 的列长度及 1000 才得到毫米。这里不 apply scale。
    """
    mode = config["surface_mode"]
    if mode not in {"uniform", "directional"}:
        raise ValueError(f"Unsupported fixed surface mode: {mode}")
    matrix = part.matrix_world.to_3x3()
    scales = [column.length for column in matrix.col]
    axes = [column.normalized() for column in matrix.col]
    mm_per_local_unit = [value * 1000.0 for value in scales]
    world_axis = list(axes[0])

    material = bpy.data.materials.new(config["name"])
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    links = material.node_tree.links

    def add_node(node_type, name, location):
        node = nodes.new(node_type)
        node.name = name
        node.label = name
        node.location = location
        return node

    shader = add_node("ShaderNodeBsdfPrincipled", "Finish BSDF", (650, 100))
    shader.distribution = "MULTI_GGX"
    for socket_name, config_key in {
        "Base Color": "base_color_linear", "Metallic": "metallic",
        "Roughness": "roughness", "Anisotropic": "anisotropic",
        "Anisotropic Rotation": "anisotropic_rotation", "IOR": "ior",
        "Specular IOR Level": "specular_ior_level", "Specular Tint": "specular_tint_linear",
        "Alpha": "alpha", "Subsurface Weight": "subsurface_weight",
        "Coat Weight": "coat_weight", "Sheen Weight": "sheen_weight",
        "Transmission Weight": "transmission_weight", "Emission Strength": "emission_strength",
        "Thin Film Thickness": "thin_film_thickness",
    }.items():
        shader.inputs[socket_name].default_value = config[config_key]
    output = add_node("ShaderNodeOutputMaterial", "Material Output", (1000, 100))
    links.new(shader.outputs["BSDF"], output.inputs["Surface"])

    if mode == "directional":
        noise_config = config["noise"]
        coordinate = add_node("ShaderNodeTexCoord", "Local Object Coordinate", (-1000, 500))
        to_mm = add_node("ShaderNodeVectorMath", "Local Coordinate to Millimetres", (-800, 500))
        to_mm.operation = "MULTIPLY"
        to_mm.inputs[1].default_value = mm_per_local_unit
        links.new(coordinate.outputs["Object"], to_mm.inputs[0])
        frequency = add_node("ShaderNodeVectorMath", "Millimetres to Grain Coordinate", (-600, 500))
        frequency.operation = "DIVIDE"
        frequency.inputs[1].default_value = noise_config["wavelengths_mm"]
        links.new(to_mm.outputs["Vector"], frequency.inputs[0])
        noise = add_node("ShaderNodeTexNoise", "Directional Grain", (-350, 500))
        noise.noise_dimensions = noise_config["dimensions"]
        noise.noise_type = noise_config["noise_type"]
        noise.normalize = noise_config["normalize"]
        for socket_name, config_key in {
            "Scale": "scale", "Detail": "detail", "Roughness": "roughness",
            "Lacunarity": "lacunarity", "Distortion": "distortion",
        }.items():
            noise.inputs[socket_name].default_value = noise_config[config_key]
        links.new(frequency.outputs["Vector"], noise.inputs["Vector"])
        roughness = add_node("ShaderNodeMath", "Roughness Centre and Variation", (50, 500))
        roughness.operation = "MULTIPLY_ADD"
        roughness.inputs[1].default_value = 2.0 * config["roughness_amplitude"]
        roughness.inputs[2].default_value = config["roughness"] - config["roughness_amplitude"]
        links.new(noise.outputs["Fac"], roughness.inputs[0])
        links.new(roughness.outputs[0], shader.inputs["Roughness"])

        geometry = add_node("ShaderNodeNewGeometry", "Unmodified Surface Normal", (-1000, -100))
        dot = add_node("ShaderNodeVectorMath", "Normal Dot World Grain Axis", (-800, -100))
        dot.operation = "DOT_PRODUCT"
        dot.inputs[1].default_value = world_axis
        links.new(geometry.outputs["Normal"], dot.inputs[0])
        projection = add_node("ShaderNodeVectorMath", "Normal Component of Grain Axis", (-550, -100))
        projection.operation = "SCALE"
        links.new(geometry.outputs["Normal"], projection.inputs[0])
        links.new(dot.outputs["Value"], projection.inputs["Scale"])
        tangent = add_node("ShaderNodeVectorMath", "Projected Grain Axis", (-300, -100))
        tangent.operation = "SUBTRACT"
        tangent.inputs[0].default_value = world_axis
        links.new(projection.outputs["Vector"], tangent.inputs[1])
        normalize = add_node("ShaderNodeVectorMath", "Normalized Grain Tangent", (-50, -100))
        normalize.operation = "NORMALIZE"
        links.new(tangent.outputs["Vector"], normalize.inputs[0])
        links.new(normalize.outputs["Vector"], shader.inputs["Tangent"])

        square = add_node("ShaderNodeMath", "Squared Normal Axis Alignment", (-550, -400))
        square.operation = "MULTIPLY"
        links.new(dot.outputs["Value"], square.inputs[0])
        links.new(dot.outputs["Value"], square.inputs[1])
        attenuation = add_node("ShaderNodeMath", "One Minus Squared Alignment", (-300, -400))
        attenuation.operation = "SUBTRACT"
        attenuation.inputs[0].default_value = 1.0
        links.new(square.outputs[0], attenuation.inputs[1])
        anisotropy = add_node("ShaderNodeMath", "Attenuated Anisotropy", (-50, -400))
        anisotropy.operation = "MULTIPLY"
        anisotropy.inputs[1].default_value = config["anisotropic"]
        links.new(attenuation.outputs[0], anisotropy.inputs[0])
        links.new(anisotropy.outputs[0], shader.inputs["Anisotropic"])

    metadata = {
        "blender_version": bpy.app.version_string,
        "material_name": material.name,
        "profile": copy.deepcopy(config),
        "part_name": part.name,
        "matrix_world": [list(row) for row in part.matrix_world],
        "matrix_column_lengths": scales,
        "mm_per_local_unit": mm_per_local_unit,
        "world_grain_axis": world_axis,
        "coordinate_contract": "Object local coordinate * matrix column lengths * 1000; world unit is metre",
        "tangent_contract": "normalize(D - dot(N_surface, D) * N_surface)",
        "anisotropic_contract": "anisotropic * (1 - dot(N_surface, D)^2)",
        "roughness_contract": "roughness + roughness_amplitude * (2 * Noise.Fac - 1)",
        "procedural_contracts_active": mode == "directional",
        "nodes": [_node_metadata(node) for node in nodes],
        "links": [
            {
                "from_node": link.from_node.name, "from_socket": link.from_socket.identifier,
                "to_node": link.to_node.name, "to_socket": link.to_socket.identifier,
            }
            for link in links
        ],
    }
    return material, metadata
