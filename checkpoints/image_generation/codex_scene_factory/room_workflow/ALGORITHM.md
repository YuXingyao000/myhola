# 房间生成工作流：算法、接口与证据边界

后续 `quality_only` 独立工厂实验新增的参考解耦与素材接口见 [工厂记录](../factory_generalization/NOTE.md)。本文保留首个 Shed 改编运行的算法与证据口径。

本文对应 `packages/cad_scene_factory/` 的房间流程，记录核对日期为 2026-10-23；下述实际制作与验收完成于 2026-10-20 UTC。目标是从 The Shed 成熟工程及参考图制作可复用的派生房间，再用真实 Blender 构建、六视角渲染和看图反馈检查质量。本阶段不插入 CAD，也不调用 FLUX；产物是合成场景及渲染图。

这里的 Agent 指一个能读文件、写脚本、调用工具和查看图片的 Codex 会话。Python 主控决定何时制作、验收、返修和停止；Agent 决定具体布局、材质和灯光改动。它没有固定的“文字直接变成房间”数学解码器，而是在工具反馈中编写并修正场景程序。

**1. 四个文件如何分工**

| 文件 | 运行位置与输入 | 主要输出／职责 |
| --- | --- | --- |
| `room_reference.py` | 普通 Python；本机 The Shed 下载目录、已有审计与预览 | `reference.json`、逐文件摘要、语义组和灯光摘要、两张已有参考图；不启动 Blender |
| `room_workflow.py` | 普通 Python；用户需求、参考 JSON、输出目录、Blender 路径及预算 | 复制资产、调用 Codex、调度 Blender、检验报告、反馈返修，持续写入 `workflow_result.json` |
| `room_tools.py` | **Blender 内部 Python**；模板脚本或 `.blend`、机位与渲染参数 | 构建、依赖检查、渲染和场景概要；直接使用 `bpy`，不能当普通 Python 脚本运行 |
| `call_llm.py` | 普通 Python；任务配置、阶段提示、图片、可选会话 ID／回复 schema | 启动 `codex exec` 或指定会话的 `resume`，保存事件、回复、用量和失败信息 |

Agent 编写的 `build_scene.py` 同样运行在 Blender 内。它暴露 `build_scene(template_dir: Path)`，通过 `bpy` 修改当前场景；外围忽略返回值并保存场景。导入模块时只定义函数和常量，实际加载基础工程、移动物件、修改材质和灯光都在函数中完成。

**2. 输入必须明确到哪些内容**

用户用 Markdown 描述房间活动、布局、表面、照明及质量要求。程序还需要显式参考包与运行参数：

| 输入 | 内容与约束 |
| --- | --- |
| `reference_file` | `reference.json`；含源 `.blend`、可复制资产目录、来源／许可证、参考图片和说明文件路径、资产清单与摘要 |
| `output_dir` | 独立模板目录；新任务要求为空，已有任务通过恢复接口继续 |
| `blender` | 实际 Blender 可执行文件；工具路径、普通 Python 路径由主控写入运行配置 |
| 模型与资源 | 默认 `gpt-6-astra`、`ultra`；渲染默认 `OPTIX`、GPU 0、128 samples、seed 8，可显式覆盖 |
| 返修预算 | `max_repairs=3` 默认表示首版之后最多三次外围返修；是整个任务的累计上限 |

公开 Python 入口为：

```python
generate_room(user_prompt, reference_file, output_dir, blender,
              model="gpt-6-astra", reasoning_effort="ultra",
              sandbox="workspace-write", device="OPTIX", gpu_index=0,
              samples=128, seed=8, max_repairs=3, dry_run=False) -> dict
resume_room(output_dir, *, feedback=None, max_repairs=None) -> dict
preview_room(request, output_dir) -> dict
```

`generate_room` 新建任务并返回实际状态；`dry_run=True` 只写请求，不复制资产或调用模型。`resume_room` 读取已有请求与状态，延续明确的会话；`feedback` 可传入外部审阅意见。若创建阶段已终止但失败，恢复要求调用记录含明确会话 ID 和退出码，用 `resume_create.md` 在原会话继续首版，另存 `create_resume_*`，不重新复制资产；没有终止证据或会话 ID 时拒绝另开会话。`preview_room` 不调用模型，独立重建并渲染当前模板。

命令行入口为 `python -m packages.cad_scene_factory.room_workflow`，新任务使用 `--user_prompt_file --reference_file --output_dir --blender`，恢复使用 `--resume --output_dir`。恢复仍受原请求的 `host_files` 摘要约束，代码改动不会被自动接受；接口具备恢复分支，也不意味着旧任务可以跨版本直接续跑。

**3. 先整理参考，再冻结本次任务**

1. `prepare_reference(reference_dir, output_dir)` 读取下载清单和既有审计，核对源 `.blend` 与下载记录的 SHA256。逐个记录源资产的相对路径、字节数与 SHA256；摘要用于识别文件是否变化，不是质量分数。
2. 参考包保留两张已有预览、语义物件组、组合关系、工作台平面、灯光和成像摘要。物件 AABB 只说明空间范围，不能充当接触面或精确碰撞体；不同类型灯的能量数值也不能直接比较图像贡献。
3. 参考包自身不复制大资产。正式任务由主控复制到 `assets/the_shed/`，对源文件和副本逐项校验，生成带来源记录的 `assets.json`，并另存原始资产记录。
4. `request.json` 保存用户需求、模型和运行配置、显式注入的提示词及 `host_files` 摘要。冻结对象包括主控、Blender 工具、调用层、相关渲染辅助脚本和 `prompts/room/`；后续阶段检查这些文件未变，同时复查源工程、资产清单、源素材和复制的基础资产。

复制的基础工程也保持只读。Agent 从它加载场景，在 Blender 内存里通过构建脚本实施改编；新增资源须打包并追加来源。改变相机、重命名物件或原样导入 Shed 都不满足要求：必须在活动、家具／工具关系及相关照明上形成有功能意义的变化，同时保留成熟几何、UV 和材质细节。

**4. 一轮完整控制流程**

1. **制作首版。** 主控把 `CONTRACT.md`、`GUIDE.md` 和运行路径组成显式任务指令，再把 `create.md`、用户需求和参考图交给新 Codex 会话。房间流程直接复用 `run_codex_turn`，不调用旧 `prepare_request`，因此不会注入旧 CAD 摆放协议。
2. **Agent 编写与自查。** 产出 `scene_plan.md`、`build_scene.py`、`views.json`。提示词要求它真实构建、渲染并打开至少一张房间图、一张活动中景和一张表面近景；快速预览放入独立的 `previews/` 子目录。这是制作要求，不能仅凭回复中的声明认定已执行。
3. **外围重新构建。** 主控检查交付文件和六个视角 ID，启动新的后台 Blender。`build` 从空场景导入构建脚本、执行 `build_scene(template_dir)`，检查实际依赖，再保存 `room.blend`、`build_report.json` 和 `scene_inventory.json`。
4. **外围拍摄六图。** `render` 一次打开保存的场景，用指定 GPU／CPU、采样数、seed 和六个相机拍摄；逐图记录成功或失败、相机矩阵及错误。主控再读取 PNG，核对画幅，生成拼图和 `preview_report.json`。
5. **技术失败进入返修。** 缺文件、构建错误、缺贴图、外部依赖、坏相机或部分渲染失败都形成具体反馈。失败视角仍计入预定六张的分母；该轮不进入视觉审阅。
6. **技术通过后看图。** 主控把六张当前图和参考图交给同一会话，要求按 `inspect.md` 只读审阅，输出符合 `review.schema.json` 的 JSON。审阅前后核对模板、资产、图片和报告摘要，拒绝审阅时改证据；也检查回复准确列出全部图片名，且 `ready` 不含未解决问题。
7. **按判断继续或停止。** `revise` 且预算尚有余额时，用 `repair.md` 和实际问题续接原会话，更新脚本后回到外围重建；`ready` 输出 `ready_for_review`；`blocked`、预算耗尽或异常输出 `incomplete`。每轮目录与失败记录保留。

六个视角固定为 `room_overview`、`room_reverse`、`workbench_context`、`workbench_detail`、`surface_detail`、`alternate_view`。`views.json` 格式为 `{"views": [...]}`，每项包含 `id`、米制 `position`／`target`、`vertical_fov_degrees`、`resolution: [width, height]`。坐标为右手系、Z 向上；机位需要覆盖全景、中景与近景，反向机位应揭示不同空间关系。程序检查名称与可渲染性，机位是否有意义仍要看图判断。

返修预算约束外围发出的返修次数，不是单个 Agent 回合内部命令数、渲染次数或总耗时的硬上限。首版可以自行修正观察到的问题；没有强制最少迭代次数。每个模板有进程锁，避免两个主控同时操作同一任务。

**5. Blender 工具究竟检查了什么**

`room_tools.py` 提供 `build --template_dir --output_dir`、`render --blend --views_file --output_dir` 和 `inspect --blend --output_dir`，均通过 `BLENDER --background --python ROOM_TOOLS -- ...` 执行。主控还传入 `--python-exit-code 1`，使 Python 异常对应非零退出。

构建检查图像、链接库、字体、音视频、缓存和体积数据块中的文件依赖；已打包内容可通过，外部文件必须存在且位于模板内。UDIM 按登记瓦片逐项检查。历史素材库的 weak reference 不等于实际渲染依赖。保存前先按来源工程解析绝对路径，保存到新位置后再相对化，避免相对路径的基准漂移。

渲染显式选择设备，使用 Cycles，关闭相机景深，按垂直视场角设置相机，保存世界／逆变换与投影矩阵；输出 PNG 前统一设置 `media_type="IMAGE"`，避免继承源工程多层 EXR 模式。保留场景调色，同时移除可能向外部路径写文件的合成输出节点。`inspect` 只读导出对象、材质、灯光、成像设置及依赖概要，不改变输入 `.blend`。

这些检查不枚举任意 Python 读文件行为，也没有覆盖所有模拟目录、外部 OSL 脚本或视频逐帧依赖。因此“包内依赖通过”仍需与实际模板搬移后的重建和渲染一起解释。

**6. Codex 调用与清洁上下文**

`run_codex_turn(request, prompt, turn_dir, session_id=None, images=None, output_schema=None)` 是本工程的普通 Python 接口。底层运行已登录账号的 `codex exec`，强制 ChatGPT 登录，移除子进程环境里的 `CODEX_API_KEY` 和 `OPENAI_API_KEY`。它不是用 API key 直接发送 HTTP 请求；额度与额外 credits 由账号侧决定。

首次调用建立新会话；后续制作、审阅和返修使用事件里的精确 `session_id`，不选择“最近会话”，也不因等待较久另开制作会话。调用层关闭个人配置／规则、项目文档自动注入、记忆、自动 skills、插件和 hooks；**清洁上下文仍包含 Codex 内置指令、工具说明、本任务显式指令及工具结果**，续接时也保留本任务历史。

每次调用保存 `prompt.txt`、`events.jsonl`、`stderr.log`、`reply.txt`（若有）和 `result.json`。只有进程成功退出、出现唯一匹配的会话 ID 与 `turn.completed`、没有 `turn.failed` 且最终回复非空，才标记调用 `completed`。这只表示该回合完成，不表示场景验收通过。失败也先留下调用结果，再交由主控记录 `incomplete`。

本次实际核对发现 CLI 的 `usage` 对应**会话累计值**，与该会话 rollout 的 `token_count.info.total_token_usage` 一致。不同阶段的 `usage` 不能相加；缓存输入已包含在输入总数中，推理输出已包含在输出总数中。

**7. 如何读取结果**

| 记录／状态 | 能支持的结论 |
| --- | --- |
| `request.json`、`assets.json` | 本次输入、运行配置、代码摘要和基础资源来源 |
| `workflow/round_*/preview/` | 每版真实构建、六图渲染、图片与报告；`pass` 仅表示这些技术检查通过 |
| `workflow/round_*/review.json` | 对布局、表面、光照、实质改编的观察及具体问题；不使用未经校准的总分 |
| `workflow/root_visual_review.json` | 主控另行查看六图与参考的定性结论；不等于用户批准或盲评 |
| [RUN_RESULTS.json](RUN_RESULTS.json) | 交付汇总、实际报告路径、完整模板迁移及图像差异、会话累计用量和适用范围 |
| [RUN_TIMING.json](RUN_TIMING.json) | 从真实调用事件追溯的阶段耗时；区分 Agent 内部预览与外围渲染，完整端到端耗时未知 |
| `workflow_result.json: ready_for_review` | 模型认为当前候选可交付审阅；`human_review` 仍为 `pending` |
| `workflow_result.json: incomplete` | 某阶段失败、受阻或返修预算耗尽，不能称为生成成功 |

视觉审阅沿用制作会话，不是独立模型评价。外围重建独立于 Agent 的快速预览；最终质量仍需另行查看六图、核对实质改编、检查源工程未变及完整模板搬移重建。不能只凭模型 `ready` 或技术 `pass` 宣称完成高质量房间。

**8. 截至核对日期，哪些已经验证**

既有 [工具验证](TOOLS_NOTE.md) 在 Blender 5.2.2 LTS、CPU、1 sample 的微型场景上覆盖了两瓦片 UDIM、缺失／外部依赖拒绝、部分失败计数、横竖画幅和相机投影；复制模板并使旧路径不可用后，验证了保存工程渲染与脚本重建。还真实调用外围六视角接口并只读检查 Shed 源工程。这证明相应实现行为，不证明完整房间的可迁移性、真实感或 GPU 成本。

当前 [工作流验证记录](workflow_validation.json) 的 19 项检查使用模拟 Codex 与模拟预览覆盖状态转换、摘要检查、审阅字段和恢复分支；[调用层验证](transport_validation_20261020.json) 的 13 项检查覆盖模拟子进程和真实 CLI 帮助解析。它们不能当作真实制作或模型视觉能力证据；完整模板另行进行了下述实际搬移、重建和六图渲染。

真实创建目录为 `/mnt/d/data/cad_scene_templates/workshop_room_20261019_001/`。首轮因账号额度失败；2026-10-20 恢复时发现 GPU 0／1 被占用，主控中断本任务的制作进程，保存已写脚本后切到 GPU 2，继续原会话 `01a1537a-84c6-7442-963d-c2a59e711b91`。`workflow/recovery_20261020/` 保留原请求／结果、原代码和新代码快照、`migration.json` 与 `gpu_change.json`；代码版本和 GPU 的变化均显式记录，没有静默绕过冻结检查。

`workflow/create_resume_02/result.json` 已为 `completed`，三份制作文件已交付。制作阶段有三组六图、64 samples 的成功预览；根据脚本、计划和调用记录，实际修正涉及 Blender 5.2 输出格式兼容、移除实心墙后的虚构窗口灯、减弱布料斑纹、扳手接触及螺栓侧放。审阅结束后的会话累计为 5,836,143 input tokens（其中 5,466,112 cached）及 28,830 output tokens，总计 5,864,973；创建结束时的较小累计值不能再加一次。失败回合的分段用量未知，最终累计值已有整会话记录，但不据此估算费用。

最终 [交付记录](RUN_RESULTS.json) 保存于 2026-10-20 13:32 UTC：外围六图技术 `pass`，构建未报告缺失或外部依赖；Agent 审阅为 `ready`，工作流为 `ready_for_review`，外围返修次数为 0。主控已逐张另行查看六图，认可其作为接近 Shed 质量的派生首例；这是非盲定性判断，`human_review` 仍为 `pending`。PNG 兼容处理已统一收进工具，含多层 EXR 初始状态的 [23 项真实 Blender 回归](TOOLS_RESULTS_20261020.json) 全部通过，并显式记录工具版本迁移。

完整模板已复制到 `workshop_room_20261019_001_relocated/`，在原模板路径暂时不可用时从脚本重建，并以相同配置完成六图渲染；原目录随后恢复，源工程与打包基础资产摘要未变。迁移后六图均非逐像素完全相同，8-bit 像素平均绝对差为约 `0.000081–0.000356`，逐图记录见 [RUN_RESULTS.json](RUN_RESULTS.json)。这是同一 Linux 环境的目录迁移证据，没有验证 Windows 或跨硬件运行，也不承诺像素级一致。

详见 [运行笔记](NOTE.md)。本例复用成熟几何、UV 和贴图，没有训练 LLM 参数，仅证明这套输入、资产和预算下的 Shed 派生首例；其他空间、原创资产生成、CAD 插入、FLUX 及实拍对照评估均未在本阶段验证。
