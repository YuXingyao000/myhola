# Step 2：选位、挪物、试拍

输入是 Step 1 已有的 `room.blend`、与它对应的 `views.json` 和预览图。
输出是若干「支撑点 + 相机 + 局部移动方案」，暂不装入真实 CAD。
每个方案独立读取基础场景，只用 20 cm 灰方体试拍，原场景文件不回写。

## 建议阅读顺序

| 文件 | 职责 |
|---|---|
| `workflow.py` → `run_workflow()` | 外围主循环；选区一次，之后最多三轮提议/预览/评议 |
| `scene.py` | 支撑面副本、读取与索引、选区测量、完整组平移/旋转 |
| `spatial.py` | 真实网格求交、支撑采样、AABB、投影与保护体检查 |
| `preview.py` → `preview_placement()` | 核对动作、检查终态、同机位前后渲染、可见像素掩码 |
| `support.py` | 支撑面 ID 掩码、渲染深度反投影与高度验收 |
| `tools.py` | Blender 子进程入口；describe / prepare_supports / measure / preview |
| `config.json` | 尺寸、摄影规格、移动及返修预算；默认 CPU 八线程 |
| `prompts/` | 中文工作合同、三个阶段提示词、对应 JSON schema |

`run_workflow()` 串联下面的节点。F 是外围 Python，A 是 Agent，T 是 Blender 工具：

| 流程图节点 | `workflow.py` 中的函数 | 输入 → 返回 |
|---|---|---|
| F 准备场景证据 | `main()` 中 describe / annotate_views | 运行输入 → 场景索引、编号图 |
| A 选择候选区域 | `ask_agent(..., "select_regions", ...)` | 场景证据 → 区域及可移动组 |
| T 准备支撑面副本 | `prepare_supports()` | 支撑对象 → 新场景、材质改动及摘要 |
| T 测量表面与占用 | `run_tool(..., "measure", ...)` | 区域 → 实测支撑和占用 |
| F 建立候选集 | `run_workflow()` 中 candidates / measurements | 测量结果 → 活跃候选，不适用项记录原因 |
| A 提议整理与构图 | `ask_agent(..., "propose_setup", ...)` | 活跃候选、上轮反馈 → 落点/机位/移动方案 |
| T 在副本执行并检查 | `run_tool(..., "preview", ...)` | 方案 → 数值报告、前后图及渲染表面验收 |
| A 审阅前后图与报告 | `ask_agent(..., "inspect_setup", ...)` | 报告与图片 → keep/revise/drop 意见 |
| F 外围判定 | `run_workflow()` 中 reviews 循环 | 数值与视觉意见、预算 → 下一轮待修候选 |
| F 汇总全部结果 | `export_placements()` | 全部候选与尝试 → 结果清单 |

`active` 只包含本轮需要处理的位置；`state` 保留所有候选、历史尝试和同一个 Agent
会话。循环里显式完成“提议 → 执行 → 审阅 → 判定”，下一轮重新筛选 `active`；
通过项和淘汰项退出循环，剩余项才返修。准备阶段也在同一入口，`--prepare_only` 在此返回。

`workflow.py` 复用上层 `call_llm.run_codex_turn()`，不用当前 GUI 对话。
首轮创建一个新摆放会话，随后显式续接该会话。Agent 只返回数据，外围调用 Blender。
典型上限是 1 次选区 + 3 ×（提议 + 评议）= 7 个外围 turn；不是七次底层模型推理。
通过的候选立刻退出循环。无合格位置也是正常结果，不偷偷换输入或无限重试。

## 支撑面与渲染保持一致

选区使用原场景概览。选区后，外围 `prepare_supports()` 调用 Blender 的
`scene.prepare_support_scene()`：只对选中的支撑对象复制位移材质，使用对象级材质槽，
将几何 Displacement 改成 BUMP。颜色、粗糙度、Normal 和微表面凹凸保留；其他共用
材质/网格的对象不受影响。另存 `support_scene/prepared_scene.blend`，原场景不回写。
宏观起伏需要事先做成可测量的网格；本函数不烘焙任意位移、不推断物理支撑。

后续测量、局部整理、代理试拍及 Step 3 都使用这个副本。`request/state/ready` 中的
`base_scene` 与 `scene_hash` 指向副本；`source_scene/source_scene_hash` 保留原工程来源。

`after.png` 同次输出指定支撑对象的 ID 掩码及 Z EXR。`support.py` 将代理周围可见
表面的深度反投影成世界高度，与 `support_point_world.z` 对比，使用既有 2 mm 容差。
没有可见支撑像素或高度不符都不能通过。`ready.placements[].support_validation`
保存数值与原始证据路径；这仍不证明被遮挡部分的连续几何或物理稳定性。
Step 3 要求这项验收存在且通过，旧 ready 必须重新验收后再使用。

## 运行

在仓库根目录使用下面的形式。`overview_dir` 中每张 PNG 按 `views.json` 的视图 ID
命名，且必须对应同一版基础场景。选区使用原图像素；编号图保持原分辨率。

```bash
python packages/cad_scene_factory/step2/workflow.py \
  --base_scene /path/to/build/room.blend \
  --views_file /path/to/views.json \
  --overview_dir /path/to/renders \
  --output_dir /path/to/new_placement_run \
  --blender /mnt/d/natural/blender-5.2.2-linux-x64/blender \
  --model gpt-6-astra --reasoning_effort ultra \
  --sandbox danger-full-access
```

也可以 `python -m packages.cad_scene_factory.step2 ...`。
添加 `--prepare_only` 只做场景索引和编号图，不调用 Agent、不渲染。正式运行使用新输出
目录。可选 `--user_prompt_file /path/request.md` 补充拍摄偏好；阈值通过独立 `--config`
JSON 显式设置。没有恢复/锁/自动修复配置层；故障直接抛出，日志及已完成记录留在目录里。

汽车场景应使用 `auto_repair_20261025_001/workflow/human_edit_01/build/room.blend`。
其 `human_edit_01/views.json` 和 `human_edit_01/renders/` 是删盒子后重新渲染的一张
surface_detail，可用于单视角工具检查。搜索整个房间时需要提供该版本完整多视角，不能
把旧版含盒子的图当作新版证据。

目前渲染端针对本机 **Blender 5.2.2** 实测；Object Index 节点与文件输出使用该版本 API。
默认 CPU，隐藏 CUDA 设备，Cycles/合成/降噪都显式选择 CPU；不占用训练 GPU。

## 产物

```text
new_placement_run/
  request.json                 输入、提示词、固定配置
  scene_context.json           对象/层级索引、单位、视图和编号位置
  context/                     编号图、Blender 命令与日志
  select_regions/              Agent 回复、原始事件、会话和用量
  support_scene/               prepared_scene.blend、材质处理记录；后续共用此副本
  measurement/                 全部局部采样结果
  round_00/                    后续 round_01、round_02 同结构
    propose_setup/             本轮完整方案
    candidate_00/              job/report、before/after、物件/支撑掩码、支撑深度、工具日志
    inspect_setup/             看图意见及调用记录
  placements.json              全部候选、全部尝试、失败原因
  ready.json                   副本路径/摘要、原场景来源、变换、机位、图片及支撑面验收
```

每个候选终态是 `proxy_ready`、`rejected` 或 `incomplete`。
总体 `completed` 只表示有限流程完成，可能零项通过；`human_review` 始终独立待定。
若进程/工具/模型调用失败，当前文件保留 `incomplete`，异常退出；不是几何不合格。
用量保留每个 turn 的原始字段，不把可能累计的会话数字直接求和。

## 第一版边界

- 米制、Z 向上、近水平的单一支撑对象；`support_object_id` 指实际渲染网格。
  多个网格拼出的复合台面、堆叠/悬挂物件不自动推断。表面有 2 mm 以上起伏或细小孔洞时
  可能拒绝或被网格采样遗漏；记录的是 1 cm 离散检查，不是连续几何证明。
  碰撞/遮挡包络略去底部 2 mm 接触带，避免把薄刮痕/贴花当成必须清走的物件；完整
  入镜仍检查整个方体。这是明确允许的接触误差，不是严格零穿入。
- Agent 在选区时定义可移动组，工具检查父子完整和有限变换；语义归属与是否固定仍是
  Agent 假设。首版不移动集合实例、带骨骼/连接约束的物件。
- 只检查移动前后支撑和移动终点碰撞，不计算推移路径、重心、摩擦或物理稳定。
  组内原有相互位置保持；其他物件保守使用 AABB。大网格按共面片区拆盒，仍可能误拒。
- 所有被挪组都要求在照片里有实际 Object Index 像素；是否可辨认、整理是否自然仍需
  Agent 看图及后续人工审阅。该通道不能证明透明/折射物件的感知可见性。
- 模型返回格式交给输出 schema；本地只补候选对应、文件尺寸和必要几何检查。
  不提供通用不可信脚本执行器、任务恢复系统或额外规划 Agent。
- 原 blend 摘要用于溯源与检测意外修改，提示词不是文件系统权限隔离。
  真实 CAD 的接触、实际轮廓、孔槽可见性和材质检查留给下一阶段。

验证记录见 `checkpoints/image_generation/codex_scene_factory/placement_workflow/NOTE.md`。
