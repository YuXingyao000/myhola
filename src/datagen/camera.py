"""Blender 拍摄适配；当前只实现相机配置。"""

import math

import bpy
from mathutils import Matrix, Vector

from geometry import PerspectiveCamera


def configure_camera(
    scene: bpy.types.Scene,
    camera_object: bpy.types.Object,
    profile: PerspectiveCamera,
    *,
    near_clip: float,
    far_clip: float,
) -> None:
    """将数学相机应用到已有 Blender 相机，近远裁剪与世界坐标同单位。

    camera_object 是装配器创建的专用相机，无父级、约束或动画。
    同时配置 scene 的完整画幅；不创建对象、不移动 CAD 或环境。
    """
    forward = (Vector(profile.target) - Vector(profile.position)).normalized()
    right = forward.cross(Vector(profile.up_reference).normalized())
    if right.length <= 1e-12:
        raise ValueError("Camera direction and up_reference must define a frame")
    right.normalize()
    up = right.cross(forward)

    # Blender 镜头沿局部 -Z 看，矩阵三列依次为右、上、后。
    transform = Matrix((right, up, -forward)).transposed().to_4x4()
    transform.translation = Vector(profile.position)
    camera_object.matrix_world = transform

    camera = camera_object.data
    camera.type = "PERSP"
    camera.sensor_fit = "VERTICAL"
    camera.sensor_height = 32.0
    camera.lens = camera.sensor_height / (2 * math.tan(math.radians(profile.vertical_fov_degrees) / 2))
    camera.shift_x = 0.0
    camera.shift_y = 0.0
    camera.clip_start = near_clip
    camera.clip_end = far_clip
    camera.dof.use_dof = False

    render = scene.render
    render.resolution_x = profile.width
    render.resolution_y = profile.height
    render.resolution_percentage = 100
    render.pixel_aspect_x = 1.0
    render.pixel_aspect_y = 1.0
    render.use_border = False
    render.use_crop_to_border = False
    scene.camera = camera_object
