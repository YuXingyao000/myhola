# 工厂独立生成：接口与首轮实验

## 当前状态

2026-10-23：接口解耦和素材接入已实现并验证；实际工厂任务已完成，状态 `ready_for_review`，人工验收仍为 `pending`。输出目录为 `/mnt/d/data/cad_scene_templates/factory_20261023_001`，会话 `01a1677a-e314-7e02-a721-9eb70a5831cd`。Agent 独立选择素材、制作和内部修正场景；主控没有手工修改生成的布景。

启动前两次查询 NVIDIA 状态，八卡均为 0%／15 MiB，计算进程列表为空；本次显式使用 GPU 0。这里描述启动时资源状态，不表示该卡以后始终空闲。实际入口和固定参数见 [PLAN.md](PLAN.md) 及包内 README。

## 实际生成结果

- Agent 选择了 6.8 × 5.8 米的小批量钻孔、沉孔和钳工作业空间：三台钻床、工件储存架、转运车、检查工位及完整建筑外壳；没有车床／铣床，也不宣称具备这些工序。
- 11 个外部资产、61 个文件，共 156,778,605 字节；来源为 Poly Haven 和 ambientCG，素材清单记录许可均为 CC0。没有使用 Sketchfab 或基础厂房工程。
- 内部三次六图预览后提交首个候选；外围 `round_00` 独立重建与六图渲染通过，正式 1200 × 900、128 samples、seed 8、OPTIX GPU 0。外围返修使用 0／3 次，不能把三次内部预览说成三次外围返修。
- 外围重建约 29.9 秒，六图渲染约 35.6 秒。Agent 制作与审阅耗时另计；用量也不能把同一会话的累计值逐轮相加。
- 只读审阅给出 `ready`，源码、素材和证据在审阅前后保持一致；这只是 Agent 的意见。
- 额外把四个交付文件及 `assets/` 复制到 `/mnt/d/data/cad_scene_templates/factory_20261023_001_portability`，不复制原 `.blend` 或缓存，用 CPU 构建重新得到场景。迁移构建通过，缺失／外部依赖均为零；本项未再次渲染。

[FINAL_AUDIT.json](FINAL_AUDIT.json) 核对宿主 18、参考 4、审阅证据 95、素材 61 项摘要全部一致，`authored_versions/version_002` 的四文件与正式首轮证据匹配。[PORTABILITY.json](PORTABILITY.json) 保留换目录重建结果。

## 用量与时间

会话元数据记录创建 1,587.706 秒（约 26 分 28 秒），审阅 331.452 秒（约 5 分 31 秒）；这包括 Agent 内部工具使用。外围重建／渲染另计。42 个唯一 response 去重合计 input 3,562,912，其中 cached input 3,308,928；output 35,787，其中 reasoning output 13,153。总 token 3,598,699，缓存和推理均为相应总量的子集，不额外相加。

创建 input/output 为 3,102,230／32,503；审阅增量为 460,682／3,284。CLI 审阅结果中的 usage 是会话累计，不能再与创建结果求和。这是 ChatGPT 账号调用的 token 遥测，不是 API 付费账单，也不直接等价于账号额度百分比。

正式图片与总览位于模板 `workflow/round_00/preview/`，技术报告为其中的 `preview_report.json`，Agent 意见为 `workflow/round_00/review.json`。本轮没有 CAD 放置或 FLUX 编辑，不能由环境图通过推断图像—CAD 几何对应已经通过。

## 主控看图意见（不是人工结论）

已逐张查看正式六图并与 Shed 工作台参考对照。钻床、台钳和工具车的材质、结构明显比简化几何有说服力，整体能识别为钻孔／钳工作业车间；六视角确实展示了空间正反方向。仍然明显的差距是：收纳盒与工位排列重复、墙面和建筑细节较少、背景活动密度较低，自制道具与成熟素材的细节水平不统一。日光与顶灯有实际来源，但光照较均匀，配合规整布局仍有较强的人工布景感。

因此，这一轮证明了仅图片质量参考的独立搭建流程能跑通；不据此宣称已经达到 Shed 的场景真实感，更不能把 Agent 的 `ready` 当成人工通过。是否接受这种规模与细节水平，由用户对这组原图判断。

## 与 Shed 实验的区别

`generate_room(..., reference_mode="quality_only", asset_cache=..., asset_catalog=None)` 只使用图片／说明；默认 `adapt` 继续复制和保护基础工程。`room_reference.py --quality_reference_file` 复制质量图片并写简短说明，去掉基础工程、原布局、原相机参数和原审计路径。实际参考包位于 `/mnt/d/data/cad_scene_references/the_shed/quality_reference/`。

素材工具和操作说明现在明确注入 room runtime。`assets.py` 支持房间对象的 `files` 和历史平铺清单，保留既有来源元数据；外围构建前检查清单文件的实际 SHA256。共享缓存只用于复用下载，最终依赖仍在模板内。

制作、六视角重建、只读审阅及有限返修的状态机不变；Blender 工具未修改。工厂提示要求完整机加工作业空间，设备、工位及布局由 Agent 决定。Shed 图片仅用于判断表面、建筑细节与光照的质量，没有提供可导入的基础工程。

这是“接口解耦后的相同工作流”实验，不能记录为原封不动只替换用户提示词。旧 Shed 产物与源码快照保留，没有用当前代码覆写旧结果。

## 实现验证

- [VALIDATION.json](VALIDATION.json)：原状态机 19 项、新接口 11 项、素材桥接 7 项通过。Agent／Blender 使用模拟；文件复制、缓存和摘要检查使用真实临时文件。
- [NETWORK_ASSET_CHECK.json](NETWORK_ASSET_CHECK.json)：真实 HTTPS 下载一个 1k HDRI（1,488,808 字节），合并房间清单并检查摘要；没有渲染，测试素材未预置进工厂场景。
- 实际 dry run 核对三份指导文件、仅图片参考、共享缓存配置和完整工具摘要；不调用模型、不复制源工程。

素材验证脚本为 [validate_assets.py](validate_assets.py)，接口验证为 [validate_workflow.py](validate_workflow.py)。这些结果证明实现行为，不代表工厂图像已达到人工质量标准。

## 留存与验收

本次固定工具和提示词复制到模板 `workflow/host_source/`，同时留存原 Git 状态和已跟踪文件差异。`request.json` 保留模型、设备、seed、采样与返修预算；各回合事件和结果、每轮图片和报告单独保存。

运行中核对发现，原流程虽保留每轮 `.blend`、机位、图像与报告，却不逐版复制生成源码。本次增加 [capture_versions.py](capture_versions.py) 作为旁路观测器，不改正在运行的冻结工具或 Agent 文件；从 06:56:11 UTC 起把四个交付文件的稳定内容复制到模板 `workflow/authored_versions/`。快照时间是观测时间。更早的 `build_01` 脚本与机位从创建事件 `item_38` 的字面量 heredoc 恢复到 `recovered_build_01/`，没有重新执行命令；不能把后来观测的源码冒充首版。后续正式轮次可用 `evidence.json` 的摘要匹配源码版本。

宿主快照另补入 `call_llm.py` 导入时读取的 `prompts/DEVELOPER_INSTRUCTIONS.md`，来源与摘要记在 `supplemental_dependencies.json`；room 实际使用的 developer instructions 仍是请求里保存的房间指南。这些是留存改进，不是场景质量调整。

制作期间第一次额外草稿使用 48 samples，偏离计划的快速预览值；其后 `render_02` 和 `render_03` 均为 64 samples、六图成功、seed 8。正式外围仍固定 128 samples。创建期间曾出现 WebSocket 断线，客户端切换 HTTPS 后在原会话继续；失败事件保留，没有另开制作会话。

已交付六视角原图及总览；Agent 的 ready 与技术通过分开记录，用户批准保持 pending。人工反馈后另开修订轮并保留初次交付结果；尚未收到反馈时不填通过结论。
