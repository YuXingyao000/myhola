"""用场景 AABB 检查相机到目标包围盒之间的有限保护区域。"""

from collections.abc import Mapping

from .geometry import BoundingBox, PerspectiveCamera, Vector3


def find_blockers(
    target_bounds: BoundingBox,
    camera: PerspectiveCamera,
    scene_bounds: Mapping[str, BoundingBox],
) -> list[str]:
    """返回进入保护区域内部的对象名称，保持输入顺序。

    输入均为世界坐标，目标盒三轴有正长度；scene_bounds 排除目标自身，
    包含支撑物。边界接触允许，面／线／点进入内部也算占用。
    这里只使用相机位置，画幅与 FOV 由 geometry.check_bbox_framing 检查。
    """
    return [
        name for name, bounds in scene_bounds.items()
        if _intersects_protected_volume(bounds, target_bounds, camera.position)
    ]


def _intersects_protected_volume(
    bounds: BoundingBox, target: BoundingBox, camera_position: Vector3,
) -> bool:
    # K = union_t [C+t*(Bmin-C), C+t*(Bmax-C)]，0 <= t <= 1。
    # 场景 AABB 必须与某个 t 对应的开盒相交，才会进入 K 内部。
    t_min, t_max = 0.0, 1.0
    for axis in range(3):
        c = camera_position[axis]
        # 每轴两条严格不等式，统一写成 slope*t < offset。
        constraints = (
            (target.minimum[axis] - c, bounds.maximum[axis] - c),
            (c - target.maximum[axis], c - bounds.minimum[axis]),
        )
        for slope, offset in constraints:
            if slope > 0:
                t_max = min(t_max, offset / slope)
            elif slope < 0:
                t_min = max(t_min, offset / slope)
            elif offset <= 0:
                return False

    # 忽略仅由浮点舍入留下的极小区间；单点接触不算进入内部。
    return t_min + 1e-12 < t_max
