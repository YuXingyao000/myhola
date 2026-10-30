# Shed 派生房间：实际制作与验收记录

记录核对：2026-10-23；实际结果保存于 2026-10-20 13:32 UTC。首版制作完成，外围重建和六图渲染技术通过；Agent 审阅为 `ready`，主控已另行逐张看图并认可其作为接近 Shed 质量的派生首例。完整模板在同一 Linux 环境搬移后的重建与六图渲染也已通过。工作流为 `ready_for_review`，`human_review` 仍为 `pending`。

紧凑结果见 [RUN_RESULTS.json](RUN_RESULTS.json)，外部模板的完整交付记录见 [delivery_report.json](/mnt/d/data/cad_scene_templates/workshop_room_20261019_001/workflow/delivery_report.json)。本次文档收尾只核对已有记录，没有新增模型调用或渲染。

2026-10-23 主控重新打开六张最终原图与两张参考图，复核上述定性结论；并行审计重算了 14 个工具／提示词文件、421 个源素材及副本、454 项审阅证据和 425 个迁移文件的摘要，均匹配对应记录。五阶段交互流程图也通过 736px／320px 下的逐阶段操作与布局检查。

**任务与方法**

本次实现并执行 [PLAN.md](PLAN.md) 的首个 Shed 派生房间案例：使用已有高质量工程、参考图和审计资料，让 Codex 编写可重建场景程序，依据真实 Blender 预览改进，再由外围独立重建与验收。控制流程见 [ALGORITHM.md](ALGORITHM.md)。

目标活动是下午的台钳保养与手工具整理。已有房间、家具、工具、UV 和材质贴图来自 The Shed，`assets.json` 记录 Poly Haven 来源和 CC0 许可。新内容是场景程序、布局与活动安排、相关照明及少量次要物件；这是参考驱动的实质改编，没有训练 LLM 参数，也不是从零制作所有资产或证明跨场景泛化。

**本次输入、版本与预算**

| 项目 | 实际配置／记录 |
| --- | --- |
| 工作区 | `/mnt/d/python_room_generation`，基底 Git commit `e6babb259c94033412c56a26d23972d09d6b2559`；本轮新增／修改文件尚未以该 commit 固化，具体版本以任务快照为准 |
| 模板与全部运行证据 | `/mnt/d/data/cad_scene_templates/workshop_room_20261019_001/` |
| 模型调用 | ChatGPT 账号的 Codex CLI，`gpt-6-astra`、`ultra`，`danger-full-access`；未切换为 API key |
| 固定会话 | `01a1537a-84c6-7442-963d-c2a59e711b91`；创建、恢复和审阅都显式指定该会话 |
| 渲染 | Blender 5.2.2 LTS、Cycles、OPTIX；本次使用 GPU index 2、seed 8；制作预览 64 samples，外围验收 128 samples，不代表该卡当前空闲 |
| 视角与预算 | 六个固定角色视角，1024×768；首版后最多三次外围返修，最终 `repairs_used=0`；制作内部三组预览不计入外围返修次数 |
| 源工程 | `assets/the_shed/` 内的只读基础工程与素材；清单 421 个文件，源 blend SHA256 为 `fe6294a84839e2225f2d9702a05ff5bc4472ff78743daabbcc60a7a95761c6e1` |
| 版本迁移 | `workflow/recovery_20261020/migration.json` 与 `workflow/host_png_fix_20261020/migration.json` 保留旧／新摘要及变更记录；当前主控 SHA256 前缀 `4c9dd890cf0f`，调用层 `4b7e377d3b14`，Blender 工具 `4f8801f45f77` |

**实际过程与失败**

1. 原 `workflow/create/` 已读取参考并检查部分源工程，但遇到账号额度限制；`result.json=failed`、退出码 1，未交付首版。原事件与失败记录保留。
2. 恢复前归档原 `request.json`、工作流结果及旧代码快照。新增终止失败创建的同会话恢复、锁内状态更新、简洁服务错误记录和恢复提示词；通过 `migration.json` 显式更新任务的 `host_files`，没有静默跳过摘要检查。
3. `create_resume_01` 开始制作后，主控发现 GPU 0／1 各约占用 22,073 MiB，GPU 2 约 15 MiB。主控只向本任务制作进程发送 SIGINT，确认终止且没有残留子进程，保留已写的三份制作文件，再把设备改为 GPU 2。原请求副本、原因和会话 ID 保存在 `gpu_change.json`。
4. `create_resume_02` 在原会话继续，最终 `result.json=completed`、退出码 0。`build_scene.py`、`views.json`、`scene_plan.md` 已交付；制作预览及失败尝试均保留。

**制作中具体改变了什么**

以下依据作者计划、实际脚本和制作事件记录，尚不替代外围独立看图结论。房间形成 L 形工作与工具归还区：复用完整柜体实例，移动台钳及夹持灯，让工具箱及关联工具成组布置；调整凳子、梯子及长柄工具以留下使用空间。新增布料、零件盘、垫圈和螺栓是次要程序物件，主要家具和工具仍保留来源资产的细节。

初次预览因源工程多层 EXR 的 `media_type` 与 PNG 不兼容，报告六图均未成功；当时由作者在模板脚本内修复，未改主控。之后三组预览 `creation_06_render`、`creation_08_render`、`creation_10_render` 均为 6/6 成功，64 samples，报告耗时分别约 86.48、86.60、87.42 秒。它们是制作反馈成本的一部分，不是端到端耗时。

作者看图后移除了实心木板墙后缺乏实际开口支持的“窗口灯”，弱化布料的斑纹，按实际网格顶点修正扳手与支撑面的接触，并把螺栓改为侧放；灯具发光位置与反向视角的暗部也作了调整。最终制作预览为 `previews/creation_10_render/`，构建产物为 `previews/creation_09_build/room.blend`。作者报告的剩余局限包括来源台钳 1K 贴图的极近景限制、布料／零件盘细节较简单，以及复用柜体具有相同磨损图案。

外围审阅完成后，主控把 `media_type="IMAGE"` 统一放入 `room_tools.py` 的 PNG 输出准备，避免每个构建脚本重复处理同一兼容问题。以 `MULTI_LAYER_IMAGE` 微型场景重跑的 [23 项真实 Blender 回归](TOOLS_RESULTS_20261020.json) 全部通过；`workflow/host_png_fix_20261020/migration.json` 显式记录此次仅工具两行改动与新快照，原制作版本继续保留，原始模板、源工程和贴图未改。随后使用更新后的工具完成完整模板搬移验证，结果如下。

**当前验收证据**

| 层次 | 当前结果与边界 |
| --- | --- |
| 工作流实现 | [19 项本地检查通过](workflow_validation.json)，Codex／预览为模拟；原 16 项记录保存在 [旧记录](workflow_validation_20261019.json) |
| 调用层实现 | [13 项检查通过](transport_validation_20261020.json)，模拟子进程加真实 CLI 帮助解析，没有模型调用 |
| Blender 工具 | [23 项最新回归通过](TOOLS_RESULTS_20261020.json)，多层 EXR 初始状态覆盖 PNG 修正，并复查依赖、UDIM、相机、部分失败和微型模板搬移；不能据此推断当前完整房间质量 |
| 真实首版制作 | `create_resume_02` 完成，三份制作文件与最终六张制作预览存在；这是实际模型调用证据 |
| 外围技术验收 | `workflow/round_00/preview/preview_report.json=pass`，planned=6、success=6、failed=0；无缺失或包外依赖 |
| Agent 视觉审阅 | `workflow/round_00/review.json` 为 `ready`，问题列表为空；同一制作会话逐张对照六图和两张参考，结论包含布局、表面、照明与实质改编 |
| 主控独立观察 | 已逐张查看六图，认可其作为接近 Shed 质量的派生首例；这是视觉判断，没有转换成未经校准的数值分数 |
| 完整模板迁移 | [relocation_result.json](/mnt/d/data/cad_scene_templates/workshop_room_20261019_001_relocated/relocation_result.json) 为 `pass`；旧模板路径不可用时从副本重建，planned=6、success=6、failed=0，原路径随后恢复 |
| 源素材保留 | 最终交付记录确认源工程及打包基础资产摘要未变；没有覆盖原 Shed 工程 |
| 用户批准 | `human_review=pending`；本笔记未替用户批准 |

**已知用量**

本次 CLI 返回的 `usage` 是**会话累计值**。已核对本会话 rollout 最后一个 `token_count.info.total_token_usage`，与 `workflow/round_00/inspect/result.json` 完全一致：input tokens **5,836,143**（含 cached input **5,466,112**），output tokens **28,830**（含 reasoning output **10,919**），total tokens **5,864,973**。该累计记录时间为 2026-10-20 13:24:36 UTC。

创建结束的 `create_resume_02/result.json` 曾报告 5,364,221 input、26,737 output，属于同一会话的较早累计快照，**不能与审阅结束值相加**。缓存输入和推理输出也各自已经包含在对应总数内。这些计数包含重复上下文处理，不是唯一文本长度，不用于估算费用。

原 `create` 和被中断的 `create_resume_01` 仍为 `usage=null`，各失败回合的分段用量未知；后续 rollout 已有整个会话的累计记录。保留这两层事实，不把失败分段未知改写成零，也不再把最终累计值误称为仅审阅回合用量。

阶段耗时另见 [RUN_TIMING.json](RUN_TIMING.json)：从实际 rollout 的开始／终止事件追溯，四个 Codex 回合分别为 137.508、252.025、602.445、155.907 秒，合计约 19 分 8 秒，包含失败与中断回合及其内部工具、制作预览；不再重复累加作者渲染时间。外围验收和迁移的工具内部渲染各约 127 秒，另列记录。完整端到端时长没有直接采集，保留为未知，不把额度等待间隔或单次渲染时间当成完整生成成本。

**完整模板迁移与结果边界**

迁移由 [validate_relocation.py](validate_relocation.py) 完成：复制模板代码、机位、计划、素材清单及全部基础资产到 `workshop_room_20261019_001_relocated/`，暂时移走原模板目录，在新路径重新构建与渲染，最后恢复原目录。报告明确记录 `source_template_unavailable_during_check=true` 与 `source_template_restored=true`。这验证的是同机 Linux 目录搬移，没有完成 Windows 或跨硬件验证。

六张迁移图均有极小像素差异，8-bit 像素平均绝对差范围约 `0.000081–0.000356`，并非逐像素完全相同；逐图差值和 SHA256 见 [RUN_RESULTS.json](RUN_RESULTS.json)。技术通过、作者审阅与 [主控定性审阅](/mnt/d/data/cad_scene_templates/workshop_room_20261019_001/workflow/root_visual_review.json) 分别保留，用户批准仍未发生。

此案例支持这套输入、成熟资产、工具和预算下的 Shed 派生首例。重复柜体磨损、次要程序布料／零件盘的简化，以及来源台钳贴图分辨率仍是可见局限。无成熟资产参考的新房间、其他空间类型、CAD 插入、FLUX、实拍基准及全面碰撞／物理验证均未在本阶段完成，不据此宣称通用高质量生成或独立原创资产制作能力。
