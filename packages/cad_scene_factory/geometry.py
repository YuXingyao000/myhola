"""支撑摆放与相机投影，输入由数据准备层保证有效。

世界坐标右手系、Z 向上；位置和裁剪距离同单位，相机前方深度为正。
"""

from dataclasses import dataclass
from itertools import product
import math


Vector3 = tuple[float, float, float]


def _subtract(a: Vector3, b: Vector3) -> Vector3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _dot(a: Vector3, b: Vector3) -> float:
    return sum(x * y for x, y in zip(a, b))


def _cross(a: Vector3, b: Vector3) -> Vector3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _unit(vector: Vector3) -> Vector3:
    length = math.hypot(*vector)
    if length == 0:
        raise ValueError("Cannot normalize a zero-length camera direction")
    return tuple(v / length for v in vector)


@dataclass(frozen=True)
class BoundingBox:
    """轴对齐包围盒；允许退化盒，CAD 合法性由准备层另行检查。"""

    minimum: Vector3
    maximum: Vector3

    def corners(self) -> tuple[Vector3, ...]:
        """返回八角点；退化轴上的重复点仍保留。"""
        return tuple(product(*zip(self.minimum, self.maximum)))


@dataclass(frozen=True)
class PerspectiveCamera:
    """Look-at 相机，方形像素、零镜头偏移，FOV 明确为垂直方向。

    近远裁剪不设默认值，由入镜检查的调用者显式提供。
    """

    position: Vector3 = (2.0, 2.0, 2.0)
    target: Vector3 = (0.0, 0.0, 0.0)
    up_reference: Vector3 = (0.0, 0.0, 1.0)
    vertical_fov_degrees: float = 45.0
    width: int = 512
    height: int = 512


@dataclass(frozen=True)
class SupportPlacement:
    """只有环境根需要移动，CAD 与相机的世界坐标不变。"""

    environment_translation: Vector3
    target_bounds_in_support_frame: BoundingBox


def plan_support_placement(target_bounds: BoundingBox) -> SupportPlacement:
    """模板支撑面为局部 z=0，按当前 CAD 的最低点 h 平移环境。

    支撑局部坐标 q=p-(0,0,h)。输入必须是实际 CAD 的紧包围盒，
    不能用集合预留包络代替；底部对齐本身不证明实际接触或稳定性。
    """
    h = target_bounds.minimum[2]
    translation = (0.0, 0.0, h)
    local_bounds = BoundingBox(
        _subtract(target_bounds.minimum, translation),
        _subtract(target_bounds.maximum, translation),
    )
    return SupportPlacement(translation, local_bounds)


def _camera_axes(camera: PerspectiveCamera) -> tuple[Vector3, Vector3, Vector3]:
    forward = _unit(_subtract(camera.target, camera.position))
    up_reference = _unit(camera.up_reference)
    right_unnormalized = _cross(forward, up_reference)
    # 平行向量经归一化可能留下极小叉积，不能把舍入噪声放大为相机方向。
    if math.hypot(*right_unnormalized) <= 1e-12:
        raise ValueError("up_reference must not be parallel or nearly parallel to the view direction")
    right = _unit(right_unnormalized)
    up = _cross(right, forward)
    return right, up, forward


def _camera_coordinates(point: Vector3, camera: PerspectiveCamera) -> Vector3:
    offset = _subtract(point, camera.position)
    return tuple(_dot(offset, axis) for axis in _camera_axes(camera))


def _pixel_coordinates(
    x: float, y: float, depth: float, camera: PerspectiveCamera,
) -> tuple[float, float]:
    half_angle = math.radians(camera.vertical_fov_degrees) / 2
    focal_pixels = camera.height / (2 * math.tan(half_angle))
    return (
        camera.width / 2 + (x / depth) * focal_pixels,
        camera.height / 2 - (y / depth) * focal_pixels,
    )


def project_point(point: Vector3, camera: PerspectiveCamera) -> tuple[float, float, float]:
    """世界点 → (u, v, depth)；左上原点、右／下为正、前方深度为正。

    返回连续像素边界坐标，512 方幅中心为 (256,256)。不自动裁剪到
    画幅；位于镜头平面或后方的点抛出 ValueError。此函数不检查近远裁剪。
    """
    x, y, depth = _camera_coordinates(point, camera)
    if depth <= 0:
        raise ValueError("Point must be in front of the camera")
    u, v = _pixel_coordinates(x, y, depth, camera)
    return u, v, depth


def check_bbox_framing(
    bounds: BoundingBox,
    camera: PerspectiveCamera,
    *,
    near_clip: float,
    far_clip: float,
    margin_pixels: float = 0.0,
) -> bool:
    """八角点均在视锥内才返回 True；False 不代表实际 CAD 必被裁切。

    裁剪距离是沿相机前向轴的深度，需与世界坐标同单位。margin_pixels
    是四边相同的余量，必须小于短边的一半。边界包含在内；画幅边界使用
    1e-9 像素绝对容差，裁剪边界使用 1e-12 相对容差以吸收舍入误差。
    不处理道具遮挡、有限支撑面或网格合法性，不改变输入来适配相机。
    """
    pixel_tolerance = 1e-9
    for corner in bounds.corners():
        x, y, depth = _camera_coordinates(corner, camera)
        if not near_clip * (1 - 1e-12) <= depth <= far_clip * (1 + 1e-12):
            return False
        u, v = _pixel_coordinates(x, y, depth, camera)
        if not (
            margin_pixels - pixel_tolerance <= u <= camera.width - margin_pixels + pixel_tolerance
            and margin_pixels - pixel_tolerance <= v <= camera.height - margin_pixels + pixel_tolerance
        ):
            return False
    return True
