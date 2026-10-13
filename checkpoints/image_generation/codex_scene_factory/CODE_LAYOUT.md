# 布景模板生成：代码组织方案

2026-10-12 架构设计，2026-10-13 更新实现状态；当前已实现[几何内核](GEOMETRY_CORE.md)与 [AABB 保护区域检查](VISIBILITY_CORE.md)，尚未安装依赖或运行模型／Blender。需求以 [模板协议](TEMPLATE_CONTRACT_V0.md) 为准；本文是代码架构说明，不替代该执行协议。

## 独立子项目

项目放在 `packages/cad_scene_factory/`，计划以自己的 `pyproject.toml` 和 `src/cad_scene_factory/` 作为可独立安装、提取开源的项目。当前只有纯数学模块 `geometry.py`、`visibility.py` 与对应测试；安装配置和下面的其他文件按实际实现进度添加，不创建空壳模块。

依据：主仓库 [pyproject.toml](../../../pyproject.toml) 直接依赖 Torch、Lightning、diffusers、torch-scatter 等训练栈。场景工具应能在没有 HoLa 模型和数据目录的情况下使用；核心不导入 `src.brepnet`，不读取 DeepCAD 目录约定。需要兼容旧 NPZ 时，由主项目适配层显式转换通用 manifest。

旧 `src/brepnet/data/datagen/` 保留为历史管线及可复用资产，不移动或删除。新子项目不把旧入口配置漂移带进新协议。研究笔记继续放在 `checkpoints/image_generation/`；实现稳定后，使用说明与接口说明随独立子项目发布。

建议项目内容：

```text
packages/cad_scene_factory/
  pyproject.toml
  README.md
  src/cad_scene_factory/       核心程序
  prompts/                    版本化的作者指令
  configs/                    相机、材质策略、渲染与预算预设
  examples/                   最小请求、手写模板和可分发测试几何
  docs/                       快速开始、架构、协议、模板编写指南
  tests/                      纯 Python 检查与 Blender 集成检查
```

模板、CAD、缓存与生成图片放在用户指定的外部工作目录；不成为源码包的一部分。代码许可与第三方资产许可分别记录；发布素材应核对可分发性，不在本次文档工作中擅自选择代码许可证或发布。

## 十个明确职责

以下为主要职责文件，`__init__.py`、入口薄包装及后续确有需要的辅助文件不算架构层。目标是能按流程阅读，不以文件数量限制合理拆分。

| 文件（相对 `src/cad_scene_factory/`） | 负责什么 | 不负责什么 |
|---|---|---|
| `cli.py` | 解析 `prepare`、`author`、`render` 命令，打印输出与失败位置 | 不放业务判断 |
| `workflow.py` | `author_template()` 和 `render_dataset()` 的顺序步骤、有限修复与状态流转 | 不写 Blender 节点或传输协议细节 |
| `contracts.py` | 请求、模板、材质、渲染实例和报告的类型、校验及 JSON schema | 不导入 OCC、bpy、Codex 或训练框架 |
| `cad.py` | STEP 读取、实体选择、单位／归一化检查、三角化和缓存 | 不做布景或逐图模型调用 |
| `author.py` | Codex 适配、提示组装、受控工具映射、事件与用量收集 | 不掌握最终几何和验收权 |
| `blender_process.py` | 显式启动构建／渲染进程、GPU 选择、超时和结果收集 | 不把所有 GPU 分配给每个进程 |
| `workspace.py` | 任务目录、资产索引、内容摘要、状态／日志、不可变模板保存 | 不掩盖缺失资产或静默猜路径 |
| `blender/build.py` | 独立执行作者环境程序，检查并导出背景网格、灯光及受限材质描述 | 不加载真实生产 CAD 或保留未知回调 |
| `blender/render.py` | 加载受检查环境、导入目标、实例化摆放、固定相机、赋材质、出图 | 不调用 Codex 或重写源 CAD |
| `blender/checks.py` | 接触、穿插、包络、相机、可见性及目标完整性检查 | 不把技术通过叫作真实感通过 |

`blender/render.py` 初版用明确的短函数组织：`load_environment()`、`load_target()`、`place_instance()`、`configure_camera()`、`apply_material()`、`render_outputs()`。未来摆放或材质复杂到影响阅读时，再自然提取 `placement.py`／`materials.py`，不先建立抽象工厂与插件注册系统。

用户修改相机预设到 `configs/`；修改创作指导到 `prompts/`；修改落地规则到 `place_instance()`；修改验收逻辑到 `blender/checks.py`。这些入口在 README 的“我应该改哪里”表中保持同步。

## 两条可直接阅读的流程

`workflow.py` 明文表达步骤，不用事件总线、装饰器注册或隐式回调隐藏顺序。概念伪代码如下，不代表当前存在这些函数实现：

```python
def author_template(request, workspace):
    validate_request(request)
    candidate = create_draft(request, workspace)
    for attempt in range(request.max_repairs + 1):
        report = build_and_preview(candidate, request.development_panel)
        if report.ready_for_acceptance:
            return accept_and_freeze(candidate, request.acceptance_panel)
        if attempt < request.max_repairs:
            candidate = repair_with_codex(candidate, report)
    return record_unfinished(candidate)


def render_dataset(template, prepared_cads, request):
    for cad in prepared_cads:
        instance = plan_instance(template, cad, request)
        result = render_with_blender(instance)
        save_result_including_failures(instance, result)
```

其中 `create_draft()` 是一次初稿任务，`max_repairs` 限制额外修复；调用次数、墙钟和预览预算仍由统一预算检查执行，伪代码不充当完整预算算法。实施时区分技术拒收、未审阅和验收失败，独立验收失败不能自动反馈给作者继续修复。最终接收来自固定控制器，而非 `author.py` 自报。高层读者只需先理解两个工作流；具体数学或节点实现留在对应文件。

命令界面提案（尚不可执行）：`cad-scene prepare` 准备 CAD；`cad-scene author` 制作模板；`cad-scene render` 复用模板批量拍摄。检查和预览工具调用同一套底层函数，不另写一套“Agent 专用渲染器”。已有冻结模板的渲染不要求登录 Codex。

## 主机 Python 与 Blender Python 的边界

- 主机进程运行工作流、配置／合同、Codex 调用和 CAD 准备。OCC 只在 CAD 准备功能导入，不让读取配置或渲染准备好的网格依赖它。
- 只有 `blender/` 运行模块导入 `bpy`。构建作者程序与生产渲染使用不同进程；生产加载受检查的环境数据，不执行生成的 `.blend` 回调或任意作者 Python。
- `blender_process.py` 使用命令参数列表，传绝对请求／结果路径；跨进程传版本化 JSON 和网格／纹理文件，不传 Python 对象或 pickle。
- Blender 入口显式定位随包分发的运行模块和纯 Python 合同代码；不能假定 Blender 自带 Python 能导入主机环境的 OCC／Codex 依赖。包顶层导入无重依赖副作用，相关模块按职责加载。
- 配置分别声明 Blender 可执行文件、CAD 环境和作者客户端，不硬编码机器路径。环境版本、请求摘要、执行结果与错误阶段保留到元数据。
- 命令行／源码目录分离并不构成执行隔离：作者环境构建的实际文件／网络／凭证访问边界沿整体方案实现并验证，不由本布局图冒充已解决。

## 核心源码、模板和运行产物分开

外部工作目录的提案：

```text
<workspace>/
  author_jobs/      每次模型任务的草稿、事件、预览与修复记录
  templates/        通过发布条件的不可变模板版本
  cad_cache/        STEP 派生网格、面对应、单位和变换
  assets/           允许资产及来源／许可／文件摘要
  renders/          每次批量拍摄的请求、图像、报告与失败状态
```

每个模板包含 `template.json`、环境程序、`material_recipes.json`、资产锁定及参数范围。模板版本冻结后不原地改写；新修复生成新版本。作者只写任务草稿，核心库与已发布模板不交给模型随意修改。所有路径由一个显式 workspace 根解析，不从当前目录或用户机器名推断。

run 清单记录成功、拒收、失败、超时和未审阅。输入摘要不一致不能仅凭同名文件跳过；图像与模板版本／CAD／相机／实际参数可追溯。八卡调度以后包在渲染请求分发外层，先完成单任务正确性，不改变核心合同。

## 可读性与开源交付

1. README 首屏解释项目输入、输出和三个入口；提供一个无需 Codex、无需下载大型资产的手写模板演示，使读者先理解渲染端。
2. `docs/architecture.md` 给模块职责与阅读顺序；`docs/protocol.md` 给相机、包络、支撑和材质合同；`docs/template_authoring.md` 说明如何写模板及使用检查工具。
3. 默认中文说明，代码标识符清晰英文；公开发布时补英文入口。关键函数 docstring 交代坐标系、单位、输入／输出和失败行为，解释为什么必须保护这些条件，而非逐行复述代码。
4. 不使用 `utils.py`／`manager.py` 容纳所有功能，不写吞异常的 fallback，不以脚本文件名偷偷改变协议。配置统一解析并保存 resolved 版本。
5. 使用显式类型和一个合同定义源；如需 JSON schema，从同一实现生成／核对，不手工维护两份漂移字段。初版不强加特定第三方配置框架。
6. 程序材质和模板可扩展靠文档化函数／文件接口；当前只实现一个 Codex 后端，不提前维护多个 provider 或插件系统。
7. 用户明确要求研究代码优先易读：内部计算信任约定输入，不逐层重复类型、维数和参数范围校验。保留影响数据／几何正确性的判断；外部数据与模型输出的必要检查集中在进入程序的边界，不包装每个数学运算。

测试聚焦真实边界：坐标与投影保持、不同最低点的正确接触、形状越界和外物遮挡、跨进程 schema／错误返回、失败不丢失。纯 Python 测试不要求 bpy；Blender 集成检查使用少量可分发几何，明确需要运行环境。无模型调用的假后端只验证控制流，不宣称验证 Codex 制作质量。

## 实施顺序与当前状态

建议先完成合同与配置、CAD 准备、一个手写模板及固定渲染，再接 Codex 制作／有限修复，最后扩批量与八卡。这样制作器和渲染器可以独立定位问题，每一步都有可阅读的入口和具体产物。

用户要求从底层逐步实现、先审阅最核心部分。第一切片 `geometry.py` 已通过审阅，负责包围盒、相机投影、支撑高度及保守入镜判断，见 [GEOMETRY_CORE.md](GEOMETRY_CORE.md)。第二切片 `visibility.py` 已通过审阅，负责场景 AABB 是否进入相机到目标包围盒的保护区域，见 [VISIBILITY_CORE.md](VISIBILITY_CORE.md)。两者各有一个测试文件，是供后续 CAD／Blender 适配共用的数学辅助层。其余职责尚未实现；没有执行训练、渲染或新的模型制作任务。
