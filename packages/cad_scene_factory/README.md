# CAD 布景模板的构建与拍摄

## 四个阶段

- **Step 1：生成场景。** `step1.py` 调用现有 `room_workflow.py`。原实现与提示词路径保留，
  避免改变历史任务的宿主校验和续跑入口。
- **Step 2：选位、局部挪物、方体试拍。** 新实现集中在 [`step2/`](step2/README.md)，从
  `step2/workflow.py` 的 `run_workflow()` 阅读主循环。提示词在 `step2/prompts/`，参数在
  `step2/config.json`。本阶段暂不导入真实 CAD。
- **Step 3：替换零件并批量摄影。** [`step3/`](step3/README.md) 读取 Step 2 的位置与相机，
  从 STL 清单抽样，循环导入、摆放、赋予 PBR 材质、渲染，输出 RGB / mask / depth 和溯源记录。
  不调用 Agent。
- **Step 4：增强环境真实质感。** [`step4/`](step4/README.md) 使用 RGB 与零件 mask，
  按场景固定提示词执行本地 FLUX 图生图，再合成回原始零件像素。保存参数、模型摘要、
  原始生成结果和对照图；输入输出均为 512×512，无逐图 Agent 调用。

四步独立启动；后一步读取前一步的文件，不依赖当前 GUI 对话。

## 当前入口：先制作独立环境

`room_workflow.py` 是新环境制作入口：读参考图片与可选基础工程，Codex 编写场景，
Blender 从脚本重新构建并渲染六个环境视角，再由 Codex 对照审阅、有限返修。
`reference_mode=adapt` 改编基础工程，`quality_only` 从单体素材搭建独立空间。
此阶段没有目标 CAD、中央预留区或固定产品机位。下面保留的 `workflow.py`
属于旧零件特写流程，两套合同不能混用。

代码分工：`room_reference.py` 准备紧凑参考包；`room_workflow.py` 管理任务；
`room_tools.py` 在 Blender 内构建／检查／渲染。提示词位于 `prompts/room/`。
首个示例基于 The Shed 的 CC0 工程进行实际改编，明确记录继承的资产与修改，
不把导入原场景或更换机位当作独立生成能力。

阅读代码从 `room_workflow.py` 的 `generate_room()` 开始，再看
`_execute_locked()` 中的制作、重建、审阅与返修循环。
[算法说明](../../checkpoints/image_generation/codex_scene_factory/room_workflow/ALGORITHM.md)
解释各文件和数据的关系；
[实际运行记录](../../checkpoints/image_generation/codex_scene_factory/room_workflow/NOTE.md)
包含首个场景的预览位置、验收证据和已知限制。

```bash
python packages/cad_scene_factory/room_reference.py \
  --reference_dir /mnt/d/data/cad_scene_references/the_shed \
  --output_dir /mnt/d/data/cad_scene_references/the_shed/room_reference

python packages/cad_scene_factory/room_workflow.py \
  --user_prompt_file packages/cad_scene_factory/prompts/room/PROMPT.EXAMPLE.md \
  --reference_file /mnt/d/data/cad_scene_references/the_shed/room_reference/reference.json \
  --output_dir /mnt/d/data/cad_scene_templates/workshop_room_001 \
  --blender /mnt/d/natural/blender-5.2.2-linux-x64/blender \
  --model gpt-6-astra --reasoning_effort ultra \
  --device OPTIX --gpu_index 0 --samples 128 --max_repairs 3 \
  --sandbox danger-full-access
```

`danger-full-access` 是本机 bwrap 无法创建 namespace 时使用的显式选择，
具有当前用户权限；只读评审依赖指令与文件摘要检查，并非 OS 隔离。
调用沿用 Codex 的 ChatGPT 登录，移除 API key 环境变量，不切换成按 API key 认证。
扣额／额外 credits 规则仍由账号侧决定。个人规则、记忆、插件与自动 skill 不载入；
Codex 内置指令、工具说明、当前任务和参考资料属于实际上下文。

从仓库根目录也可以调用 Python API：

```python
from pathlib import Path
from packages.cad_scene_factory.room_workflow import generate_room

result = generate_room(
    user_prompt=Path("packages/cad_scene_factory/prompts/room/PROMPT.EXAMPLE.md").read_text(),
    reference_file=Path("/mnt/d/data/cad_scene_references/the_shed/room_reference/reference.json"),
    output_dir=Path("/mnt/d/data/cad_scene_templates/workshop_room_001"),
    blender=Path("/mnt/d/natural/blender-5.2.2-linux-x64/blender"),
    sandbox="danger-full-access",
)
```

模板包含 `build_scene.py`、`views.json`、`scene_plan.md`、`assets/` 和 `assets.json`。
各轮实际 `.blend`、PNG、相机与检查报告在 `workflow/round_XX/preview/`；
调用原始记录、会话 ID、反馈与用量一并保留。`workflow_result.json` 的
`room_blend`、`last_report` 指向通过版本。`ready_for_review` 仍保留 human_review=pending。
脚本可重建不自动证明视觉质量；最终应查看全景、中景和近景实际图片。

如外部审阅发现问题，保存文字反馈后继续明确的原会话：

```bash
python packages/cad_scene_factory/room_workflow.py \
  --output_dir /mnt/d/data/cad_scene_templates/workshop_room_001 \
  --resume --feedback_file /path/to/feedback.md --max_repairs 4
```

返修上限是任务累计次数；记录不覆盖。独占任务锁防止同一模板同时启动两个外围进程。
遇到长时间调用先确认原进程状态，不因为观察超时就重启。若原进程异常退出而其子进程
仍活跃，应先处理原子进程，再恢复；锁本身不构成跨进程树的作业管理系统。

## 工厂测试：只给需求、质量图片和素材工具

先从现有参考导出纯图片包，去掉基础工程、原场景布局和审计路径：

```bash
python packages/cad_scene_factory/room_reference.py \
  --quality_reference_file /mnt/d/data/cad_scene_references/the_shed/room_reference/reference.json \
  --output_dir /mnt/d/data/cad_scene_references/the_shed/quality_reference
```

确认 GPU 0 空闲后，以新的输出目录启动。Agent 会自行调用 Blender，不能在
所有卡都运行训练时提前启动制作阶段。

```bash
python packages/cad_scene_factory/room_workflow.py \
  --user_prompt_file packages/cad_scene_factory/prompts/room/PROMPT.FACTORY.EXAMPLE.md \
  --reference_file /mnt/d/data/cad_scene_references/the_shed/quality_reference/reference.json \
  --reference_mode quality_only \
  --asset_cache /mnt/d/data/cad_scene_assets \
  --output_dir /mnt/d/data/cad_scene_templates/factory_20261023_001 \
  --blender /mnt/d/natural/blender-5.2.2-linux-x64/blender \
  --model gpt-6-astra --reasoning_effort ultra \
  --device OPTIX --gpu_index 0 --samples 128 --seed 8 --max_repairs 3 \
  --sandbox danger-full-access
```

Python API 使用同名 `reference_mode`、`asset_cache` 和可选 `asset_catalog` 参数。
`quality_only` 接受含 `id`、`images` 和可选 `documents` 的参考 JSON；每项图片／说明
提供 `path`，制作时冻结其摘要。这个模式不接受带基础工程字段的旧参考包。
若使用已下载的本地素材目录，可通过 `--asset_catalog` 传入素材工具支持的 JSON。

素材工具、操作指南和缓存路径会显式交给 Agent；下载内容必须复制进模板，
在外围构建前核对清单摘要。机床等关键资源不可用时保留具体缺口，不能用简陋替代物
宣称完成。原 Shed 示例仍使用默认 `adapt`。修改后的工具与提示词会生成新快照，
旧任务继续保留原版本，不能静默以新代码续跑。

工厂协议与运行记录见 [factory_generalization](../../checkpoints/image_generation/codex_scene_factory/factory_generalization/)。
最终六图交给用户判断，`human_review` 保持 pending；后续意见写入独立反馈文件，
通过 `--resume --feedback_file` 追加修订，不覆盖首次候选与原审阅。

另外提供三份 `quality_only` 场景需求，可替换上面命令的 `--user_prompt_file`，
并为每个任务指定新输出目录：

| 场景 | 用户需求示例 |
| --- | --- |
| 机械装配车间 | [PROMPT.ASSEMBLY.EXAMPLE.md](prompts/room/PROMPT.ASSEMBLY.EXAMPLE.md) |
| 电子维修工作室 | [PROMPT.ELECTRONICS.EXAMPLE.md](prompts/room/PROMPT.ELECTRONICS.EXAMPLE.md) |
| 普通办公室 | [PROMPT.OFFICE.EXAMPLE.md](prompts/room/PROMPT.OFFICE.EXAMPLE.md) |
| 汽车维修工位 | [PROMPT.AUTO_REPAIR.EXAMPLE.md](prompts/room/PROMPT.AUTO_REPAIR.EXAMPLE.md) |
| 木工工作室 | [PROMPT.WOODWORKING.EXAMPLE.md](prompts/room/PROMPT.WOODWORKING.EXAMPLE.md) |
| 家庭餐厨空间 | [PROMPT.HOME_KITCHEN.EXAMPLE.md](prompts/room/PROMPT.HOME_KITCHEN.EXAMPLE.md) |

三场景批次的实际输入、设备分配和命令见
[运行记录](../../checkpoints/image_generation/codex_scene_factory/cross_environment_20261023/NOTE.md)。
并发运行时使用不同 GPU，共享缓存的 `fetch` 需通过同一 `flock` 串行化；
该批次的具体要求保存在其 `EXECUTION.md` 和实际输入中。

新增汽车、木工与家庭餐厨环境的参数和实际状态见
[扩展批次记录](../../checkpoints/image_generation/codex_scene_factory/cross_environment_20261025/NOTE.md)，
[八场景清单](../../checkpoints/image_generation/codex_scene_factory/cross_environment_20261025/SCENES.json)
区分已经有完整六图的场景与尚在生成的任务；列入清单不等于已经验收。

## 旧入口：固定零件特写模板

先用普通 Python 执行 `workflow.py`：Codex 制作模板，外围程序预览，Codex 看图审阅，
有问题则有界返修，最后提交给人检查。原 `call_llm.py` 命令行入口转到同一流程。
模板依然是 `build_scene.py` 和 `build_material.py`；构建与拍摄使用 Blender 5.2.2 LTS。
示例中的 `blender` 需在 PATH 中，或替换为实际可执行文件路径；本机使用
`/mnt/d/natural/blender-5.2.2-linux-x64/blender`。

| 文件 | 职责 | 输出 |
|---|---|---|
| `prepare_stl.py` | 合并源模型编号、转换已有 PLY／STEP、核对既有 split | 带编号的 STL、逐件记录、清单副本和汇总 |
| `workflow.py` | 制作、外围预览、只读审阅、返修与交付的主循环 | 模板包、逐轮证据、`workflow_result.json` |
| `call_llm.py` | 组装上下文，新建或按明确ID续接Codex会话 | 每回合提示词、事件、回复及用量 |
| `prompts/` | 模型读取的协议、指南、运行说明和场景需求示例 | 可独立审阅的 Markdown 文本 |
| `prompts/stages/` | 制作、只读审阅、返修三个阶段的任务模板 | `create.md`、`inspect.md`、`repair.md` |
| `prompts/review.schema.json` | 只读审阅回复的结构 | `ready`／`revise`／`blocked`及具体证据字段 |
| `assets.py` | 检索、缓存、复制素材；保留来源与许可 | `assets/`、`assets.json` |
| `preview.py` | 从当前模板重建，执行固定 quick / review | 图片总览、逐件结果、技术检查报告 |
| `review_models.json` | 固定 12 个训练零件，按 STEP 面数排序 | 两档预览共同使用的样本与摘要 |
| `build.py` | 执行两个模板脚本，添加固定相机，保存环境和零件材质 | `.blend` 场景文件 |
| `render.py` | 打开场景，逐件导入 STL/OBJ、应用材质、检查并渲染 | PNG、任务输入及逐图记录 |
| `camera.py` | 将固定相机参数应用到 Blender | 相机和画幅设置 |
| `geometry.py` | 包围盒、支撑关系、相机投影及入镜检查 | 数学计算结果 |
| `visibility.py` | AABB 与相机到零件之间的保护区域相交检查 | 遮挡物名称 |

根目录保留程序和本说明，模型提示词与回复格式集中放在 `prompts/`，不写在 Python 中。
`build.py` 使用 `camera.py` 和 `geometry.py`；`render.py` 使用 `geometry.py` 和
`visibility.py`。任务规则在 `prompts/DEVELOPER_INSTRUCTIONS.md`，场景需求示例在
`prompts/PROMPT.EXAMPLE.md`；运行工具说明在 `prompts/RUNTIME.md`。
生成的模板、场景和图片使用独立的输出目录。

## 先读哪些文件

先读 `workflow.py` 的 `run_workflow()`，它直接表达五步的执行顺序。
`create_template()` 包含方案与制作，`inspect_template()` 只看图，
`repair_template()` 修改当前模板。需要了解子进程参数时再读 `call_llm.py`。

- [DEVELOPER_INSTRUCTIONS.md](prompts/DEVELOPER_INSTRUCTIONS.md)：固定接口、坐标和工作流职责。
- [SCENE_GUIDE.md](prompts/SCENE_GUIDE.md)：英文布景指南，解释场景、素材、使用痕迹和光照如何共同构成照片感。
- [ACCEPTANCE.md](prompts/ACCEPTANCE.md)：哪些检查由代码完成，哪些需要真正看图，什么时候才能交付。
- [ASSET_TOOLS.md](prompts/ASSET_TOOLS.md)：素材来源、实际接口、缓存及 Sketchfab 授权配置。
- [PROMPT.EXAMPLE.md](prompts/PROMPT.EXAMPLE.md)：一份有人使用过的车间需求，目标铝件仍保持干净。
- [RUNTIME.md](prompts/RUNTIME.md)：工具路径和本次配置的运行说明；[阶段模板](prompts/stages/)定义每回合任务，[审阅格式](prompts/review.schema.json)定义回复字段。

调用时显式读取所需文档和阶段模板，Python只负责传入路径、配置与上一轮结果，
不需要安装 skill。个人 skills、记忆和旧聊天仍不载入。公共请求保存在 `request.json`，
各阶段实际发送的提示词保存在对应调用目录的 `prompt.txt`。

## 准备渲染网格

在已有 trimesh、NumPy 的普通 Python 环境中执行；需要 STEP 回退时，该环境还需
PythonOCC。路径均显式提供，输出必须使用新目录。

```bash
python packages/cad_scene_factory/prepare_stl.py \
  --data_roots /path/to/deepcad_v6 /path/to/deepcad_v6_cond \
  --training_list src/preprocess/splits/deduplicated_deepcad_training_7_30.txt \
  --validation_list src/preprocess/splits/deduplicated_deepcad_validation_7_30.txt \
  --testing_list src/preprocess/splits/deduplicated_deepcad_testing_7_30.txt \
  --output_dir /path/to/deepcad_stl_total \
  --workers 16
```

每个源编号对应一个 `{model_id}.stl`，保留前导零。不属于指定 split 的源模型也保留，
记录为未分配。`splits/` 原样保存三份源清单；重复、缺失、冲突和失败写入
`manifest.jsonl` 与 `summary.json`，不改变既有划分。字节相同的导出 STL 另存分组，
这不等于完成近似几何去重。

已有 `mesh.ply` 直接导出 STL，保留坐标、三角形及绕序；无有效 PLY 时才使用
`normalized_shape.step`。STEP 回退使用源坐标单位的 0.001 线性细分容差和 0.1 弧度
角度容差。不读取 `post_processed_shape.step`，不额外归一化、旋转或修复模型。
导出网格可用不等于其已适配当前相机；布景范围及入镜仍由后续渲染阶段检查。

## 生成模板

在普通 Python 环境中运行，使用已登录的本地 Codex：

```bash
python packages/cad_scene_factory/workflow.py \
  --user_prompt_file packages/cad_scene_factory/prompts/PROMPT.EXAMPLE.md \
  --output_dir /path/to/template \
  --model gpt-6-astra --reasoning_effort ultra \
  --blender /path/to/blender \
  --stl_dir /path/to/deepcad_stl_total \
  --asset_cache /path/to/asset_cache \
  --max_repairs 2
```

Developer 协议自动从 `prompts/DEVELOPER_INSTRUCTIONS.md` 读取。
`--user_prompt_file` 指向本次场景需求；也可以用 `--user_prompt "场景描述"` 直接传入文本。
有参考照片时添加 `--image_file /path/to/reference.jpg`。

`--output_dir` 是模板目录。Codex 在其中写出 `scene_plan.md`、`build_scene.py`、
`build_material.py` 和所需素材。不同布景使用新目录；已有模板或工作流记录不被覆盖。
增加 `--dry_run` 只组装和保存输入，不启动 Codex，适合先检查提示词内容。

提供 `--blender` 与 `--stl_dir` 后，实际流程如下：

1. 新建一个Codex会话，先记录尺度与布景决定，再制作模板；内部可自主quick预览。
2. 外围调用 `preview_template(review)`，重新构建并生成固定15张图。
3. 技术失败直接安排返修；技术通过则将15张原图附到同一会话，进行只读审阅。
4. 审阅返回 `ready`、`revise` 或 `blocked`。需要返修才修改模板并返回第2步。
5. 默认首次制作加最多两轮外围返修；可用 `--max_repairs` 调整，0表示仅检查首版。

Codex续接使用本次明确会话ID，不使用“最近一次会话”。每次续接仍使用显式的账号、
模型与个人上下文隔离配置。阶段调用会消耗账号额度；Blender本身不消耗模型额度。
返修轮数不限制单次Agent任务的token、耗时或内部工具调用数。

只读审阅由提示词要求，并由前后摘要检查：模板、素材、源STL、固定测试组、外围代码、
预览图片和构建产物发生变化时，不接受本轮交付。它不提供操作系统级只读沙箱。
`inspected_images` 必须覆盖全部预览，但清单正确不证明Agent的视觉判断正确。

`workflow_result.json` 的状态为：`ready_for_review`（待人工审阅）、`incomplete`
（仍有失败／阻塞／预算耗尽）或 `generated_unverified`（仅生成）。人工视觉状态始终
保留 `pending`，不在Agent Loop里弹出逐项确认。`incomplete` 时CLI以非零状态退出。
登录／CLI异常、畸形审阅回复及只读阶段证据改动会停止并保留记录，不自动重试。

记录位置：`request.json` 是完整初始输入；`workflow/create/` 保存首次调用；
`workflow/round_00/preview/` 保存本轮图片、`.blend` 和技术报告，`inspect/` 保存
审阅调用，`review.json` 保存审阅决定，必要时 `repair/` 保存返修调用。后续版本使用
`round_01/`、`round_02/`，旧图片和日志保留。每次调用都保存 `prompt.txt`、
`events.jsonl`、`stderr.log`、`reply.txt` 和 `result.json`。模板素材不逐轮复制。

不提供这两个路径时仍可只生成脚本，但仅检查脚本存在，不能宣称模板技术验收通过。
Python中的旧 `call_llm()` 函数保留一次生成包装，不负责外围检查；完整流程使用
`prepare_request()` 和 `run_workflow()`。旧CLI命令则直接进入完整流程。
本机的 `workspace-write` 受 bwrap namespace 问题影响时，可显式添加
`--sandbox danger-full-access`；它使用当前系统用户的完整权限，不会自动降级或切换。

## 固定预览

```bash
python packages/cad_scene_factory/preview.py \
  --template_dir /path/to/template \
  --blender /path/to/blender \
  --stl_dir /path/to/deepcad_stl_total \
  --panel_file packages/cad_scene_factory/review_models.json \
  --preset quick
```

quick 使用低、中、高面数各一件、64 samples；改为 review 则使用全部 12 件、256 samples，
另拍支撑高度为 -0.901、-0.4505、0 的三个空场景。两档均为 512×512，同一零件的 seed
保持一致。`--device` 和 `--gpu_index` 选择渲染设备，默认 OPTIX GPU 0。

样本沿用 workshop_002 首轮看过的 12 个训练零件，按原始 STEP 的 B-Rep 面数重排
（7 到 28 面），并记录 STL、STEP 和训练清单摘要。该组用于模板回归检查，不代表全库
复杂度分布；不要为了让某模板通过而换掉失败样本。

每次预览输出到模板下新的 `previews/<时间>_<档位>/`，也可显式指定新的 `--output_dir`。
先看 `contact_sheet.png`，再看单张原图；`preview_report.json` 保留全体计划样本、失败
原因、源码及素材摘要。构建失败和未生成的图片不会从分母中消失。现有几何检查是 AABB
检查，不等于实际表面接触、像素可见性或真实感证明；未执行的迁移检查明确列为 not run。

## 构建场景

```bash
blender --background --python-exit-code 1 --python packages/cad_scene_factory/build.py -- \
  --scene_script /path/to/template/build_scene.py \
  --material_script /path/to/template/build_material.py \
  --output_blend /path/to/template/scene.blend
```

`build.py` 不导入零件。它将返回的零件材质标记为 fake user，使暂时没有使用者的
材质也能被保存，并在场景属性中记录材质和支撑物名称。已加载的外部素材会打包进
`.blend`。两个源脚本和素材清单的 SHA-256 同样保存在场景中。构建后还写入
`build_report.json`，记录实际使用的版本、源码摘要、素材和打包结果。

有外部素材时，文件放进模板的 `assets/`，并在同级 `assets.json` 中逐文件声明：

```json
[
  {
    "path": "assets/wall/base_color.png",
    "sha256": "填写实际文件的 SHA-256",
    "provider": "填写实际来源",
    "source_url": "填写实际素材地址",
    "license": "填写来源提供的许可"
  }
]
```

程序会核对所有声明文件的摘要，拒绝包外素材路径和未声明的外部贴图。当前支持单张
静态图片／HDRI，模型资产应使用 append 或网格导入；已有内嵌贴图可保留，外部 linked
`.blend` 库会被拒绝。图像、字体和声音完成打包后再次核对是否确实内嵌。
纯程序布景可省略清单，或使用空数组。检索／下载使用 `assets.py`，具体来源支持情况
和授权配置见 [ASSET_TOOLS.md](prompts/ASSET_TOOLS.md)。

模板使用素材时，应相对于自己的 `__file__` 定位文件。
相机固定为 `(2, 2, 2)` 看向原点、垂直 FOV `45°`、`512 × 512`。

## 批量拍摄

```bash
blender --background --python-exit-code 1 --python packages/cad_scene_factory/render.py -- \
  --scene_file /path/to/template/scene.blend \
  --model_files /path/to/part_a.stl /path/to/part_b.obj \
  --output_dir /path/to/new_render_batch \
  --device OPTIX --gpu_index 0 --samples 64 --seed 0
```

输入网格应已归一化，使用右手坐标系、Z 向上，与相机及环境共享长度单位。
导入时保留网格坐标和姿态；OBJ 内的多个网格对象作为一个零件处理。
STL 导入后按相邻面夹角生成着色法线：小夹角平滑，大于 30° 的边保持硬边，
共面三角形仍呈平面。阈值在 `render.py` 的 `STL_SMOOTH_ANGLE_DEGREES` 中定义，
并写入渲染 `request.json`。这不移动顶点或改变三角形，也不改变轮廓；OBJ 沿用
自身导入法线。STL 缺少 B-Rep 曲面语义，小于阈值的设计折线可能被平滑，低细分
轮廓也不会因此变圆，不能据此宣称精确恢复 CAD 法线。模板 Agent 不得修改源 CAD
或用材质变化掩盖共享渲染器的问题。
每个零件仅改变环境根节点的高度，使支撑顶面接触零件的最低点。
入镜、支撑面高度／覆盖范围和 AABB 遮挡检查失败时默认停止，不写入该零件的图片。
预览工具使用 `--continue_on_error` 完成后续样本，同时保留错误并最终以失败状态退出。
仅对声明的支撑面采用 `1e-6` 长度单位的接触容差，吸收 Blender float32 世界变换的
舍入误差；其他物体仍按原始 AABB 检查，模型几何不变。

输出目录必须是新目录，避免混入上次任务的图片。图片名称包含顺序编号；
`request.json` 保存完整模型列表、相机、模板来源和渲染参数，`renders.jsonl`
保存每项的成功／失败状态、模型路径和 seed，以及成功检查所得包围盒和环境高度。
默认单张 seed 为基础值加序号；固定预览通过 `--model_seeds` 使用冻结的逐件 seed。

默认使用一张 OptiX GPU；`--gpu_index` 选择该进程检测到的 GPU 序号。
支持 `--device CUDA` 或 `--device CPU`。多个任务可以使用不同 GPU，但本轮未实现任务调度。
构建与批量拍摄均不调用 Codex；技术报告由预览程序生成，视觉审阅和后续修改分开进行。

## 本机下一轮 Example

使用英文 `prompts/PROMPT.EXAMPLE.md`。本轮固定名义零件长边17.5cm，以维修过程中局部清理
的工作台为例，明确功能性物件组、主要表面状态与光照；目标材质仍保持原数值。
它按场景用途、空间与支撑、环境材质、主体材质、光照、参考图及交付要求组织需求。
复用时修改对应场景内容；相机、接口和检查协议由 Developer
约定提供，不必在 User 提示词里重新填写。先只组装输入，查看模板目录里的 `request.json`：

```bash
scene_factory=/mnt/d/python/packages/cad_scene_factory
scene_template=/mnt/d/data/cad_scene_templates/workshop_workflow_001
scene_blender=/mnt/d/natural/blender-5.2.2-linux-x64/blender

/mnt/d/anaconda3/bin/python "$scene_factory/workflow.py" \
  --user_prompt_file "$scene_factory/prompts/PROMPT.EXAMPLE.md" \
  --output_dir "$scene_template" \
  --model gpt-6-astra --reasoning_effort ultra \
  --blender "$scene_blender" \
  --stl_dir /mnt/d/data/deepcad_stl_total \
  --asset_cache /mnt/d/data/cad_scene_assets \
  --max_repairs 2 \
  --sandbox danger-full-access \
  --dry_run
```

准备好后去掉最后的 `--dry_run`，才会启动 Codex 并使用账号额度。这里使用本机已确认
可执行的完整权限模式；在支持工作区沙箱的机器上使用默认模式即可。
Sketchfab 下载需事先配置授权；不配置也可使用已连接的其他素材来源。
