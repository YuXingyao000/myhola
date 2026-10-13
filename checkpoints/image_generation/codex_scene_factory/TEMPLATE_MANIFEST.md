# 第四实现切片：模板入口数据

## Task

落实用户确认的“普通目录＋template.json 入口”：只定义模板清单及 JSON 读写，让读者能看清包内数据如何组织。不生成模板程序、场景数据或图片。

## Assumptions

这是当前草案的最小数据结构，尚不承担完整模板验收。内部字段遵守约定，不重复添加逐项类型／数值校验。坐标右手系、Z-up、共用归一化单位；模板支撑局部 z=0，不默认单位为米。正式包络、物理尺寸和首次制作预算仍待后续数据准备确定。

## Inputs

`TemplateManifest` 保存 `template_id`、`version`（模板内容版本）、`target_envelope`（支撑局部包围盒）、`placement_range`（实际 CAD 世界最低点 h 的闭区间）、`camera_profile`、`support_object`、`builder`、`material_recipes`、`assets_directory`、`parameters`。相机是制作／检查所用的预设，仍由外围程序控制。

每个 `FloatParameter` 保存 `default`、`minimum`、`maximum`，约定有限数值且 minimum ≤ default ≤ maximum。当前只表示连续数值参数；类别、相关采样与真实场景参数的解释留到有具体模板时实现。

## Outputs

`load_template(path)` 返回类型明确的清单；`save_template(template, path)` 写一个新的 JSON 文件。文件引用均相对于 template.json 所在目录，清单里不保存本机绝对路径。示例只展示文件格式，不是可执行模板包。

## Processing Protocol

读入 JSON 后显式恢复 BoundingBox、PerspectiveCamera、tuple 坐标及 FloatParameter；其余字段直接传入数据类。写入时用 dataclasses.asdict 转换，以 UTF-8、缩进 JSON 保存，不另写通用序列化框架。不会导入或执行 builder，也不会自动寻找或读取素材。

## Metrics

只验证字段解释及往返保存是否保持数据。不评价可渲染性、真实感、素材完整性或模板适用率。

## Decision Rule

载入成功只表示清单已解析；不代表通过几何或视觉验收。保存目标必须不存在，避免覆盖已有清单；保存不等于完成模板冻结，完整版本／素材记录在后续工作流实现。

## Edge Cases

空参数映射允许。缺文件、非法 JSON 或必需字段缺失直接抛出原始异常，不做 fallback；目标目录由调用者创建。输入数值、区间顺序与路径约定由调用者保证，本片不做通用合法性判断；写出时禁止 NaN／Inf。素材／builder 是否存在、支撑对象是否真实可用、字段是否符合制作请求，留到构建入口集中检查。本片无批次、重试或失败分母。

## Validation / Sanity Checks

载入真实示例，核对已有数学类型可直接使用；修改预设后保存并重新载入，检查信息不丢失；确认保存不会覆盖已有文件。继续运行纯 Python 测试，不重新运行未修改的 Blender 模块。

2026-10-13：Python 3.12.7 下新增 3 项与原有 20 项测试全部通过。运行命令：

```bash
PYTHONPATH=packages/cad_scene_factory/src python -m unittest discover -s packages/cad_scene_factory/tests -v
```

## Implementation Constraints

一个功能文件 contracts.py、一个示例 JSON、一个测试文件。仅标准库，不加 schema 库、动态类型分派、文件发现、随机采样、数据库或其他工作流模块。

## Deliverables

[contracts.py](../../../packages/cad_scene_factory/src/cad_scene_factory/contracts.py)、[示例清单](../../../packages/cad_scene_factory/examples/template_manifest.json)、[test_contracts.py](../../../packages/cad_scene_factory/tests/test_contracts.py)及本说明。样例中的几何范围仅为格式演示，不能作为数据集归一化合同。2026-10-13：按用户要求补充逐字段注释，并将 load_template 的类型转换与集中写回分开；用户审阅通过。
