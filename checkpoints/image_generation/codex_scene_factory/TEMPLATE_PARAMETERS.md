# 第五实现切片：确定一次构建使用的参数值

## Task

让模板清单里的参数定义落到实际数值：取默认值、应用显式覆盖、检查允许范围，输出后续构建程序可以直接使用的完整参数字典。

## Assumptions

沿用[模板入口](TEMPLATE_MANIFEST.md)的 FloatParameter，定义满足有限数值及 minimum ≤ default ≤ maximum。当前只处理连续数值参数，字段名没有固定枚举；参数含义由具体模板定义。本片不确定概率分布，也不假设多个参数可以独立随机组合。

## Inputs

`resolve_parameters(parameters, overrides=None)`：参数名到 FloatParameter 的映射，以及可选的“参数名 → 本次实际值”映射。数值输入类型由调用者保证，不做字符串转数值等隐式转换。

## Outputs

新的 `dict[str, float]`，包含全部已定义参数，顺序沿用参数定义；未显式覆盖的值来自 default。原始定义和覆盖字典保持不变。

## Processing Protocol

先收集默认值，再逐项应用覆盖，最后统一检查实际使用值。参数名未知时抛出 KeyError；实际值非有限或超出闭区间时抛出 ValueError，并给出名称、取值及范围。不裁剪取值、不回退默认值，不执行随机采样。

## Metrics

只验证参数选择与范围判断是否符合约定，不评价参数组合的物理合理性、模板真实感或生成覆盖率。

## Decision Rule

所有实际值均有限且在对应闭区间内时返回完整字典；端点和 minimum=maximum 的固定值允许。定义本身遵守输入合同，本片不重复逐层验证整个清单。通过此检查不表示参数组合已经通过场景验收。

## Edge Cases

无覆盖和空覆盖均使用默认值。无参数定义时返回空字典；此时提供任何覆盖名仍为未知参数错误。缺省项与未知项分别处理，拼写错误不能静默忽略。NaN／正负 Inf 作为实际取值拒绝。没有文件读取、批量处理、重试或失败分母。

## Validation / Sanity Checks

载入现有清单示例，核对默认值及显式覆盖；检查部分覆盖、输入不变、区间端点、固定值、空参数、未知名称、越界与非有限实际值。仅运行相关纯 Python 测试，不重跑未修改的 Blender 模块。

2026-10-13：Python 3.12.7 下新增 6 项与原有 23 项测试全部通过，共 29 项。运行命令：

```bash
PYTHONPATH=packages/cad_scene_factory/src python -m unittest discover -s packages/cad_scene_factory/tests -v
```

## Implementation Constraints

仅在现有 contracts.py 及 test_contracts.py 中增添一个函数和相关测试，不另拆参数模块。数据类描述参数，resolve_parameters 选择本次实际值；后续由主机或 Blender 构建入口共用。仅用标准库，保留详细中文 docstring 和顺序清晰的三个处理块，不引入配置框架、采样器、注册表或更多场景模块。

## Deliverables

[contracts.py](../../../packages/cad_scene_factory/src/cad_scene_factory/contracts.py) 中的新增函数、[test_contracts.py](../../../packages/cad_scene_factory/tests/test_contracts.py) 中的相关测试及本说明。2026-10-13 用户审阅通过。
