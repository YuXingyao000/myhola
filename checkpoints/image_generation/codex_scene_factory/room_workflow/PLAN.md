# 独立环境生成与验收（2026-10-19）

## Task

实现可从 Python／命令行调用的 Codex 环境生成工作流，并实际产出接近 The Shed 质量的可复用场景。首个任务是基于成熟资产的实质性改编，不把导入原工程当作生成成功。当前不插入 CAD、不接 FLUX。

## Assumptions

- 数据：The Shed 原工程、贴图与已有本机参考图可用，CC0 来源有下载清单；保留源工程。
- 任务：用户允许参考驱动的派生场景。先证明改编质量，未来独立新空间迁移是后续能力，不把本次改编称作原创从零生成。
- 实现：沿用 ChatGPT 登录的 Codex 调用（gpt-6-astra / ultra），不使用 API key。实际模型服务仍经 Codex CLI 调用；公开 Python 函数作为本工程 API。
- 场景：米制、Z 向上、无目标中心预留区。环境灯光、材质与布局统一制作，保持独立房间。

## Inputs

用户 Markdown 场景需求、显式 reference JSON（源工程、可复制素材目录、参考图与审计说明）、Blender 路径、GPU、seed、渲染采样和返修预算。输入源码与数据目录只读，产物独立目录。

## Outputs

模板包含 `build_scene.py`、`views.json`、`scene_plan.md`、`assets/`、`assets.json`，外围保存 `room.blend`、逐视角 PNG、构建／渲染报告、参考对照、Codex 调用记录和 workflow_result.json。

`build_scene.py` 暴露 `build_scene(template_dir: Path)`，在 Blender 内构建场景；可打开模板内的派生基础 blend。所有生产依赖位于模板目录内，参考工程及贴图由外围复制。`views.json` 为 `{"views": [...]}`；每项含 `id`、`position`、`target`、`vertical_fov_degrees`、`resolution: [width,height]`。首轮六个视角：room_overview、room_reverse、workbench_context、workbench_detail、surface_detail、alternate_view。至少全景／中景／近景三种尺度，非六张近似英雄机位。

## Processing Protocol

1. 核对工具、参考与路径，复制参考素材并保存摘要；生成请求不混入旧 CAD 布景协议。
2. 一个制作 Agent 检查参考、写计划／脚本／观察机位，调用 Blender 快速反馈并实际查看图片。任务要求有功能意义的布局／场景状态变化，保持成熟资产。
3. 外围从脚本重新构建，渲染六个环境视角；当前版本技术失败则提供实际错误返修。
4. 技术通过后，Agent 只读查看当前图与参考图，按布局、表面、光照和改编实质性给出具体判定；不使用未经校准的总分。
5. 有问题则有限返修。保留所有版本／失败，预算耗尽仍不合格则 incomplete。完成后独立检查源工程未变、模板搬移后可重建、实际图片质量。

## Metrics

记录预定／成功／失败视角、资源缺失／外部依赖、每轮时间和调用用量、实际变更清单、逐视角视觉问题。布局密度及物件数只用于诊断，不定义真实感阈值。完成证据为真实生成调用、重建与迁移验证，以及对照多视角图片的独立审阅。

## Decision Rule

technical pass 要求所有视角渲染成功、尺寸有效、依赖齐全且位于包内；视觉 pass 要求无未解决的显著尺度、结构穿插、表面／光照假感，保留与参考相称的细节，并证实有实质改编。Agent ready 表示可交付审阅；human_review 保留 pending，不伪称用户批准。根 Agent 实际查看产物并独立记录结论，不能仅以子 Agent ready 判断任务完成。

## Edge Cases

缺失参考或认证失败显式记录；不更换付费认证。畸形视角／回复、缺文件、构建异常、损坏贴图、UDIM 缺瓦片和中断都保留阶段与日志。多组件和合并建筑不套用旧 CAD AABB 禁入规则。预算内无法达到质量则保持 incomplete 并分析原因。

## Validation / Sanity Checks

先检查构建／渲染工具真实 Blender 行为，再实际 Codex 生成；六视角覆盖与图片内容独立确认；源文件摘要前后相同；移动完整模板后从脚本重建并渲染代表视角。固定版本、seed 与视角，记录会影响复现的软硬件版本。不声称像素级跨硬件一致。

## Implementation Constraints

独立工作树 `/mnt/d/python_room_generation`、分支 `codex/room-generation`，不改训练工作区。新增清晰的 `room_workflow.py`（普通 Python）、`room_tools.py`（Blender）、`prompts/room/`（Markdown／schema）。复用 `call_llm.run_codex_turn` 传输，不使用旧 prepare_request 注入 CAD 合同。提示词不写在 Python 中。无新框架依赖；不提交数据或生成二进制。

## Deliverables

可读源码、完整示例命令与 Python API、外部模板包、实际多视角预览、调用及验收记录、研究 NOTE／紧凑 RESULTS。首轮通过后继续以实际质量证据审计完整目标，不能以脚本可运行代替高质量产物。
