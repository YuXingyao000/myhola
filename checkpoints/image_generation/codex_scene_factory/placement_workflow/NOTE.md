# Step 2 首版实现与验证

日期：2026-10-26。工作树 `/mnt/d/python_room_generation`，分支 `codex/room-generation`；
实现基于 HEAD `e6babb259c94033412c56a26d23972d09d6b2559`，本次修改尚未提交。

## 实现

用户要求分模块、方便亲自审阅，采用耦合的实验输入/输出，避免防御性编程。
新增 `packages/cad_scene_factory/step1.py` 作为原房间生成入口的转发，保留旧模块的位置，
避免破坏历史任务续跑时的宿主文件校验。新代码集中在 `packages/cad_scene_factory/step2/`。

主循环是 `workflow.run_workflow()`：选区一次 → 测量 → 逐轮批量提议 → 串行工具预览 →
批量看图 → 保存通过项、返修其余项。默认最多初稿加两轮返修，一个新会话，最多七个
外围 turn。没有新增规划 Agent；当前聊天不参与运行时循环。新代码没有 try/except
包装或通用任务框架；故障直接退出，阶段开始前已保存的状态保持 incomplete。

`scene.py` 管理支撑测量与完整物件组；`spatial.py` 缓存求交网格、构造碰撞盒及相机计算；
`preview.py` 检查终态并真实渲染前后图、物件 Object Index 掩码；`tools.py` 是 Blender
JSON 入口。提示词中文独立 MD、schema 和参数独立 JSON。原场景、材质、灯光不改写。

首版取舍：只处理近水平单一支撑网格、完整小道具组的 XY 平移/Z 旋转，不处理骨骼、
约束物件或集合实例的移动。语义分组由 Agent 提议，工具检查层级和变换；不是对所有
物件可动性的自动证明。已有全景图作为选区输入，必须来自同版场景。

## 已运行验证

全部使用 CPU；没有占用训练 GPU，没有调用真实 LLM。本轮验证是工程行为验证，不能
证明 Agent 已经会在八个场景中选出好位置，更不能当作真实 CAD 拍摄验收。

1. **真实 Blender 小场景**：`validation/check_blender.py`，Blender 5.2.2，96×96、4 samples、
   seed 8、CPU 两线程。构造有松散盒子、子对象标签和合并房间外壳的桌面。
   - 同一表面测得空位 4,588 个、有可移动占用的落点 1,088 个，1 cm 网格。
   - 父子物件整体移动且相对关系不变；重新打开基础场景重做方案不累加位移。
   - 缺子对象、代理碰撞、越过桌边、移动终点无支撑、底面内部孔洞、中心遮挡均被拒绝。
   - 合并外壳拆为面片碰撞盒，没有把整个房间当实心障碍。
   - 合格方案输出真实 before/after；被挪物件掩码有 289 个可见像素。
   - 指定保留但完全离开画面的物件，实际掩码为零并拒绝。
   - 产物：`/mnt/d/data/cad_scene_placements/step2_validation_20261026_final/`。

2. **外围有限循环**：`validation/check_loop.py`，替换 Agent 和 Blender 边界。
   三个候选分别一轮通过、二轮通过、持续数值失败。验证真实循环只尝试 1/2/3 次、
   使用同一明确会话 ID、不会重做通过项，Agent 错误返回 keep 也不能覆盖数值失败。
   工具异常保留 incomplete。产物：`/mnt/d/data/cad_scene_placements/step2_loop_validation_20261026_final/`。

3. **接口集成**：运行真实 host → Blender describe/measure/preview → export，Agent 回复用
   固定 JSON 替代并用三个输出 schema 校验。一处占用区域经移动后通过，真实 CPU 产出
   前后图与掩码，ready.json 包含基础场景、变换和机位。
   产物：`/mnt/d/data/cad_scene_placements/step2_integration_20261026/`。
   此处 keep 是测试桩；不代表真正的 Agent 视觉审阅。

4. **已有汽车场景**：读取删盒子后的 `human_edit_01/build/room.blend` 和对应单张
   surface_detail，生成对象索引与编号图：1,137 个渲染几何条目，其中 46 个是实例。
   对预先指定的一块真实钢台面完成局部求交，测得高度约 0.9465 m；1,080 个支持落点
   均被道具占用，因为这次工具检查未给可移动白名单，记录 fixed_or_unknown，未误报空位。
   - 初次检查将程序化刮痕薄网格也列为障碍。改为沿用 2 mm 接触高度容差后，刮痕不再
     被列为需清走的障碍，工具箱/撬胎棒/喷罐仍保留。
   - 这个容差略去碰撞/遮挡包络底部的 2 mm，不是宣称严格零穿入；完整入镜仍用全包络。
   - 产物：`/mnt/d/data/cad_scene_placements/auto_repair_step2_prepare_20261026_v2/`，
     最终局部测量在 `bench_measurement_contact_band/report.json`，约 9.8 秒。
   - 未渲染该真实场景的新机位，未让 Agent 选位，也未修改其基础工程。

代码编译、两个命令行入口、JSON schema 定义与差异空白检查通过。新代码对 Blender 5.2
的 Object Index 输出名称、ID Mask 输入插口和文件输出媒体类型做了实际调用验证；
未宣称兼容其他 Blender 版本。

## 审阅与后续

从 `step2/workflow.py:run_workflow()` 阅读，再看 `preview.py:preview_placement()`。
README 中有运行方式与第一版限制。可先 `--prepare_only` 检查输入和编号图；正式运行
用新目录。此轮未消耗 Codex 账号额度，未运行八个场景的自动位置搜索。

代理位置通过之后仍需验证真实 CAD 的支撑、姿态、轮廓/孔槽和材质；人工审阅单独记录。
1 cm 离散支撑检查、保守 AABB、2 mm 接触带和 Object Index 可见像素分别有明确误差边界，
不能把 proxy_ready 描述成物理稳定或照片真实性的证明。

## 2026-10-26：按流程图整理主循环

用户审阅指出主循环混入了摘要组装、路径和记录细节，难以对应流程图。此次只重构外围代码：
`run_workflow()` 移到文件开头，直接串联准备证据、选区、测量、建立候选集，以及循环中的
提议、执行预览、审阅、外围判定，最后导出。每个节点有同名职责的独立函数，定义按执行
顺序排列；循环返回的 `active` 明确表示还需要返修的位置。README 增加节点与函数对照。

准备证据也纳入统一入口；内部 Python 调用改为 `run_workflow(request, state,
prepare_only=False)`，CLI 参数不变。没有增加框架、类或异常捕获，也未改提示词、几何
检查和渲染行为。

验证：更新并运行已有循环检查，初轮通过、返修后通过、数值否决、预算终止和工具故障
均通过。与重构前已有记录按输出目录归一化比较，`placements.json`、`ready.json` 和
工具故障状态完全一致。通过真实 Blender 执行 `--prepare_only`，成功导出场景索引与
编号图；没有渲染或真实 Agent 调用。编译、CLI 与 `git diff --check` 通过。
本轮临时检查目录：`/tmp/cad_step2_readability_u2hw6og4/`。
