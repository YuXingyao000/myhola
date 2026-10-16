"""在当前 Blender 空场景中执行模板的环境构建程序。"""

from collections.abc import Mapping
from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys

import bpy

from ..contracts import TemplateManifest, load_template, resolve_parameters


@dataclass(frozen=True)
class BuiltTemplate:
    """一次构建的输入记录与场景对象，供后续摆放、检查和导出使用。

    Attributes:
        template: 本次读取的模板清单，保留其适用范围和相机等约定。
        parameters: 传给构建函数的完整参数；记录输入值，不从场景反推。
        seed: 传给构建函数的随机种子。
        environment_root: 环境的 Empty 根节点。模板在其下放置网格、
            灯光及分组；后续平移此节点即可整体调整支撑高度。
        support_object: 清单指定的支撑对象，位于根节点的后代层级中；
            这里只确认对象归属，接触与遮挡检查由后续步骤完成。

    对象引用仅在当前 Blender 会话中有效，不能作为跨进程文件使用。
    """

    template: TemplateManifest
    parameters: dict[str, float]
    seed: int
    environment_root: bpy.types.Object
    support_object: bpy.types.Object


def build_template(
    template_path: Path,
    *,
    seed: int,
    overrides: Mapping[str, float] | None = None,
) -> BuiltTemplate:
    """读取模板包，在当前空场景中构建环境并返回其根节点和支撑对象。

    template_path 指向 template.json；构建文件和素材目录相对于它解析。
    seed 显式交给模板，overrides 经 resolve_parameters 补全为实际参数。
    当前场景必须为空，避免构建时混入旧对象；本函数不会主动删除对象。

    模板文件作为 Python 模块加载，可定义常量、类和辅助函数。
    实际创建对象的代码放在以下入口函数内：
        build_scene(*, scene, environment_root, assets_directory, parameters, seed)

    scene 是 bpy.context.scene，与 bpy.ops 当前操作的场景一致。
    environment_root 初始变换为单位矩阵；模板保持该变换，将所有环境
    对象挂在其下，并以局部 z=0 为支撑面。World 通过 scene.world 设置。
    assets_directory 是绝对 Path；parameters 是独立字典；随机过程由
    模板用 seed 控制。目标 CAD 和拍摄相机随后由外围程序加入。

    这是可信模板的内部执行入口，尚不提供作者代码隔离或完整场景验收。
    文件、参数和构建错误直接抛出，发生构建错误后由调用者废弃该场景。
    """
    template_path = template_path.resolve()
    template = load_template(template_path)
    parameters = resolve_parameters(template.parameters, overrides)
    builder_path = template_path.parent / template.builder
    assets_directory = template_path.parent / template.assets_directory

    scene = bpy.context.scene
    if scene.objects:
        raise ValueError("Template construction requires an empty scene")

    # 构建与摆放分开：模板先在支撑局部坐标下创建环境，之后整体平移。
    environment_root = bpy.data.objects.new("environment_root", None)
    scene.collection.objects.link(environment_root)

    # 按路径加载模块；先登记到 sys.modules，供 dataclass 等解析所属模块。
    spec = importlib.util.spec_from_file_location("cad_scene_template", builder_path)
    builder = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = builder
    spec.loader.exec_module(builder)
    builder.build_scene(
        scene=scene,
        environment_root=environment_root,
        assets_directory=assets_directory,
        parameters=parameters.copy(),
        seed=seed,
    )
    bpy.context.view_layer.update()

    # 允许模板按功能分组；只在这个环境的后代中定位支撑对象。
    environment_objects = {obj.name: obj for obj in environment_root.children_recursive}
    support_object = environment_objects[template.support_object]
    return BuiltTemplate(
        template=template,
        parameters=parameters,
        seed=seed,
        environment_root=environment_root,
        support_object=support_object,
    )
