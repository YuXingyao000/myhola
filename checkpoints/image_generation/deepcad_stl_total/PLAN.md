# DeepCAD 渲染网格整理：冻结执行规格

## Task

合并用户指定的 `deepcad_v6` 与 `deepcad_v6_cond` 模型编号，建立独立的 `deepcad_stl_total` 渲染输入目录，核对三份既有 train/validation/test 清单。保留源数据和源 split；不在本任务中重新划分、近似几何去重或生成图片。

## Assumptions

- 已观察：原始目录有 102,904 个模型编号，条件目录有 55,448 个编号；样本含 `normalized_shape.step` 和 `mesh.ply`，未观察到已有 STL。
- 数据假设：`mesh.ply` 是当前资产中已有的 CAD 三角化结果；直接导出 STL，保留其坐标、三角形和绕序。此事实不等同于逐件认证 PLY 与 STEP 完全一致。
- 边界假设：整理两个目录的全部编号；不属于三份 split 的模型也保留，标注 `unassigned`。
- 实现假设：使用已有 `torch` 环境中的 trimesh、NumPy、PythonOCC，不安装新依赖。采用一个 Python 进程和可配置 I/O 工作线程。
- 不确定项：完整文件库存、缺失模型和同编号冲突由全库审计确定，不凭抽查补写结论。

## Inputs

- 显式传入两个数据根目录。
- `src/brepnet/data/list/deduplicated_deepcad_{training,validation,testing}_7_30.txt`，不读取 `bak/` 副本。
- 模型目录中已有 `mesh.ply`、`normalized_shape.step`；不以重建产物 `post_processed_shape.step` 替代源 CAD。

## Outputs

- `/mnt/d/data/deepcad_stl_total/{model_id}.stl`，保留编号中的前导零。
- `manifest.jsonl`：每个源编号或 split 请求编号的状态、来源、源文件摘要、split、三角形数量、包围盒与 STL 摘要。
- `summary.json`：根目录重叠、split 内部／跨 split 重复、缺失、转换结果与运行配置。
- `duplicate_stl_groups.json`：字节相同的导出 STL 分组，不宣称近似几何去重。
- `splits/`：原三份清单原样保存，另保存源模型中未分配 split 的编号。

## Processing Protocol

1. 读取根目录的模型编号和指定清单，统计重叠、重复与缺失。
2. 按编号确定源网格；同编号存在不同源内容时记录冲突，不静默选择覆盖。
3. 优先将已有 PLY 导出为二进制 STL，不执行额外归一化、旋转、修复或重网格化。
4. 无有效 PLY 时，使用该编号的 `normalized_shape.step`，明确记录回退原因；线性细分容差为 0.001、角度容差为 0.1 弧度，均为源坐标单位。
5. 验证网格非空、坐标有限、索引有效，核对导出三角形与源网格在 float32 精度下逐项相同；失败保留编号和原因。
6. 保存逐件记录及紧凑汇总，核对导出文件编号、数量和大小。

## Metrics

源编号并集／交集、各 split 行数和唯一编号数、重复编号、请求但缺失的编号、成功／失败／冲突数、二进制 STL 精确重复组和跨 split 精确重复组。每件保存 AABB、顶点数、三角形数、源／输出摘要。

## Decision Rule

全部源编号应获得明确状态。源 split 完整性以原清单为分母，不因失败缩小分母。只有三份清单的模型均成功、且不存在编号重复或来源冲突时，标记编号层面的渲染准备完成；精确 STL 跨 split 重复另行报告，不自动删除或重划 split。

## Edge Cases

空清单保留并报告；空白行不作为编号；缺失根目录报错；无源几何、损坏 PLY/STEP、空网格、NaN/Inf、转换失败与同编号冲突均进入 manifest。重复源编号只导出一个文件，重复清单仍原样保存。不同编号的相同 STL 保留双方。输出使用新目录，避免覆盖已有派生数据。

## Validation / Sanity Checks

先用少量真实 PLY、STEP 回退、缺失编号和冲突来源检查转换及统计；再运行全库。独立读取代表性输出 STL，比对坐标／三角形及 AABB。最后核对全库输出编号、文件长度和 manifest 统计。不以这些检查声称照片真实感或全库 B-Rep 合法性。

## Implementation Constraints

一个可直接阅读的 `prepare_stl.py`；路径、split 和线程数均通过 CLI 提供。状态与回退显式记录，不修改训练／渲染程序。数据与完整逐件报告写到数据目录，代码与紧凑运行结果写在本研究主题中。

## Deliverables

转换脚本、完整 STL 目录、原 split 副本、审计记录和本目录的运行说明。另独立下载并校验 Blender 5.2 LTS，记录真实版本；素材获取、模板包和背景覆盖仅讨论，本轮不修改它们。
