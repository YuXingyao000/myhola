# 第六实现切片：在 Blender 中构建模板环境

## Task

把已批准的清单与参数接口接到模板 Python 程序，构建可整体平移的环境层级。当前只完成构建调用，不导出环境、加载 CAD、渲染图片或接入 Codex。

## Assumptions

沿用[模板协议](TEMPLATE_CONTRACT_V0.md)的支撑局部坐标：支撑面 z=0，环境根节点初始变换为单位矩阵。构建在专用 Blender 进程的当前空场景进行；此次使用手写、可信的测试模板。执行作者代码的隔离与后续场景验收尚未实现。

## Inputs

`build_template(template_path, *, seed, overrides=None)` 接受入口清单路径、显式随机种子与参数覆盖。包内 builder 和 assets_directory 相对于清单目录解析，不依赖进程工作目录。模板是一个独立 Python 模块，定义 `build_scene(*, scene, environment_root, assets_directory, parameters, seed)`；顶层可放置导入、常量、类和辅助函数，实际构建放在入口函数内。

## Outputs

`BuiltTemplate` 保存清单、传给构建程序的完整参数、种子、环境根节点和支撑对象引用。对象仍在当前 Blender 场景中；结果是内存构建状态，不是已验收的模板或可跨进程传递的文件。

## Processing Protocol

先读取清单和确定参数，确认当前场景无对象，再创建 Empty 根节点。通过 importlib 按路径创建 builder 模块，在 sys.modules 中登记后执行模块代码，随后调用 builder.build_scene，传入绝对素材路径及参数副本。专用构建进程使用模块名 cad_scene_template；再次构建时创建新模块并替换此条目。模板负责把环境网格、灯光及其分组挂在根节点下，允许多级父子关系；World 设置在 scene.world 上。构建结束更新依赖图，从根节点后代中按名称取得支撑对象，然后返回结果。失败直接传播，调用者废弃失败场景，本函数不自动清场或重试。

## Metrics

检查实际 Blender 对象、材质取值、素材路径、固定种子的可重复行为和环境整体平移。不评价图像真实感、遮挡、包络与接触质量；声明支撑对象名称并不证明它位于 z=0 或满足支撑要求。

## Decision Rule

参数有效、空场景、构建函数正常返回且指定支撑对象位于环境根节点下，才返回 BuiltTemplate。模板仍须在后续切片中经过几何和内容检查。

## Edge Cases

非空场景拒绝，不删除已有对象。参数名称／取值错误在执行模板前拒绝；文件、JSON、函数及支撑名缺失沿用 Python 异常。空参数字典有效。函数不重设全局 RNG，模板应使用传入 seed 建立自己的随机生成器；同种子可重复仍取决于模板实现与运行环境。无批量、排名、失败分母或文件输出。模板须保持根节点单位变换并将全部环境对象纳入层级；本片只定位支撑对象，不冒充已检查所有作者行为。

## Validation / Sanity Checks

使用临时手写模板，通过本机 Blender 4.2.0 检查实际节点材质、相对素材路径、固定种子、嵌套层级及整体平移。检查非法参数不执行模板、非空场景不被修改、层级外支撑对象不被误认。所有测试数据在临时目录，不依赖真实 CAD、外部素材或模型调用。运行命令：

```bash
PYTHONPATH=packages/cad_scene_factory/src /mnt/d/natural/blender-4.2.0-linux-x64/blender --background --factory-startup --python-use-system-env --python-exit-code 1 --python packages/cad_scene_factory/tests/blender/test_build.py
```

2026-10-13：Blender 4.2.0（构建 a51f293548ad）中 6 项集成测试全部通过；Python 3.12.7 中原有 29 项纯 Python 测试全部通过。测试验证的是构建接口和手写模板行为，未产生渲染图片，未验证任意生成程序的可重复性或真实感。

同日按用户意见改为 importlib 模块加载，测试模板使用带延迟类型注解的 dataclass 描述桌面材质；6 项 Blender 测试再次通过，确认模块登记及重复构建正常。纯 Python 模块未改动，本次未重复运行其测试。

## Implementation Constraints

仅添加 blender/build.py 及对应 Blender 测试，沿用现有合同。不增加 BuildContext、自定义注册表、CLI 或进程控制器。普通 Python 模块仍不导入 bpy。中文注释说明参数、坐标和调用顺序；不增加重复的字段类型检查。importlib 负责动态加载单文件模块，不构成安全隔离；尚不执行模型生成的程序。

## Deliverables

[build.py](../../../packages/cad_scene_factory/src/cad_scene_factory/blender/build.py)、[test_build.py](../../../packages/cad_scene_factory/tests/blender/test_build.py)、本说明及 [CODE_LAYOUT.md](CODE_LAYOUT.md) 状态更新。本片完成后留待用户审阅。
