"""模板包入口；坐标右手系、Z-up，使用共同的归一化单位，不默认是米。"""

from dataclasses import asdict, dataclass
import json
from pathlib import Path

from .geometry import BoundingBox, PerspectiveCamera


@dataclass(frozen=True)
class FloatParameter:
    """连续参数，约定 minimum <= default <= maximum；本层不负责采样。"""

    default: float
    minimum: float
    maximum: float


@dataclass(frozen=True)
class TemplateManifest:
    """可复用布景模板的入口清单，对应模板包中的 template.json。

    清单描述模板的适用范围、构建入口和可变参数。具体 CAD、某次采样
    得到的参数值及渲染图片属于拍摄实例，不保存在这里。坐标和长度沿用
    模板与 CAD 共用的归一化单位；三个文件／目录引用都相对于 JSON 的
    所在目录，移动整个模板包后仍使用同样的相对路径。

    Attributes:
        template_id: 模板标识，例如 workbench_001；同一模板的不同版本
            沿用该标识，便于追踪修改和复用。
        version: 模板内容版本号，与 template_id 一起标识具体版本；
            不是 JSON 格式的版本号。
        target_envelope: 不同零件允许占用的支撑局部包围盒，支撑面为
            局部 z=0。若 CAD 世界最低点为 h，则局部点 q=p-(0,0,h)。
            这是模板为零件集合预留的空间，不是某个 CAD 的实际包围盒。
        placement_range: 当前姿态 CAD 的世界最低点 h=zmin 所允许的
            闭区间 [h_min, h_max]。外围程序据 h 沿 Z 平移整个环境，
            CAD 和相机保持不动。该区间不表示零件本身的高度范围。
        camera_profile: 模板制作／检查所用的世界坐标相机预设，记录
            位置、朝向参考、垂直 FOV 和画幅；由外围程序应用到相机。
        support_object: 构建程序输出的环境支撑对象名称，例如桌面对象。
            后续检查用它定位支撑物；仅记录名称不证明接触或支撑有效。
        builder: 环境构建程序的相对路径，例如 build_scene.py；该程序
            负责背景、支撑物和灯光，目标 CAD 与相机由外围程序管理。
        material_recipes: 零件 PBR 材质配方 JSON 的相对路径；这里保存
            文件引用，具体配方由后续材质模块读取。
        assets_directory: 模板所依赖的背景网格、贴图等素材的相对目录。
        parameters: 构建参数名到 FloatParameter 的映射，描述默认值
            和允许范围；可为空。参数之间的关联及采样方式由后续流程处理。
    """

    template_id: str
    version: int
    target_envelope: BoundingBox
    placement_range: tuple[float, float]
    camera_profile: PerspectiveCamera
    support_object: str
    builder: str
    material_recipes: str
    assets_directory: str
    parameters: dict[str, FloatParameter]


def load_template(path: Path) -> TemplateManifest:
    """读取 JSON，将坐标数组和嵌套字典恢复为模板清单使用的 Python 类型。

    这里只载入元数据，不执行构建程序或读取配方／素材；文件引用保持为
    相对路径字符串，包内文件和实际场景的检查由后续构建流程完成。
    """
    data = json.loads(path.read_text(encoding="utf-8"))

    bounds = data["target_envelope"]
    target_envelope = BoundingBox(
        minimum=tuple(bounds["minimum"]),
        maximum=tuple(bounds["maximum"]),
    )
    placement_range = tuple(data["placement_range"])

    camera = data["camera_profile"]
    camera_profile = PerspectiveCamera(
        position=tuple(camera["position"]),
        target=tuple(camera["target"]),
        up_reference=tuple(camera["up_reference"]),
        vertical_fov_degrees=camera["vertical_fov_degrees"],
        width=camera["width"],
        height=camera["height"],
    )

    parameters = {
        name: FloatParameter(**parameter)
        for name, parameter in data["parameters"].items()
    }

    # 集中写回需要类型转换的字段，其余标量和文件引用保持原值。
    data["target_envelope"] = target_envelope
    data["placement_range"] = placement_range
    data["camera_profile"] = camera_profile
    data["parameters"] = parameters
    return TemplateManifest(**data)


def save_template(template: TemplateManifest, path: Path) -> None:
    """保存到新文件；父目录由调用者创建，已有清单不会被覆盖。"""
    content = json.dumps(asdict(template), ensure_ascii=False, indent=2, allow_nan=False)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(content + "\n")
