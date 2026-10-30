# Step 3：STL 批量摄影执行约定

## Task

在已通过人工初看、Step 2 已导出的办公室机位中替换真实 STL，先验证少量，
再启动随机 100 件、每件一张的批量拍摄。无需 Agent；FLUX 留给下一阶段。

## Assumptions

- 数据：`deepcad_stl_total/manifest.jsonl` 的成功行含来源、split、STL 摘要；
  实际历史转换为 PLY→STL，不能声称本次重新由 STEP 三角化。
- 边界：使用已有 5 个 `proxy_ready` 位置及对应场景，不重新选位。
  保留源 STL 的 XYZ 姿态，仅统一尺度与平移；不计算重心或物理稳定。
- 实现：CPU 8 线程，场景按机位加载一次，同机位循环替换零件。
  默认固定缎面金属参数，不用贴图；本轮首先验证数据接口。
- 无需等待确认的假设：100 件指 100 个不同 CAD ID、每件一图，而非 100×5 图。
  最长边均匀采样 0.15–0.20 m；机位均衡分配。当前没有必须等待用户回答的事项。

## Inputs

Step 2 `ready.json`、其匹配的 `room.blend`，STL 目录及 manifest，
Blender 路径、配置、输出目录、抽样数和 seed。
支撑修复后，ready 必须含通过的 `support_validation`，场景必须是同一次验收的副本。

## Outputs

- `request.json`：完整抽样清单、源行、机位、配置、输入/代码摘要与运行命令。
- 每件 `rgb.png`、`mask.png`、`depth.exr`、`metadata.json`。
- `renders.jsonl`、`summary.json` 和 Blender 日志；失败与未执行样本保留在分母内。
- 自动生成缩略图总览，供后续查看；不通过图像生成工具改动渲染结果。

## Processing Protocol

1. 从 training 成功行按 STL 字节摘要去重，排序后固定 seed 抽样。保留源 split，
   不为当前实验另划 train/test；这不是几何等价去重，也不修正源库跨 split 重复。
2. 将样本均衡分配到 ready 机位；冻结每件尺寸和渲染 seed。
3. 打开原场景，按 Step 2 的绝对矩阵应用挪物结果，恢复原相机。
4. 导入单个 STL。保持顶点/三角形，30° smooth-by-angle 仅改变着色法线。
   令包围盒 XY 中心对齐支撑点，Z 最低点对齐支撑高度；最长边落在代理尺寸内。
5. 赋予固定 Principled BSDF 金属，使用原场景灯光和色彩管理。
6. 同一次 Cycles 渲染输出 RGB、Object Index 二值可见掩码和 Z 深度。
   本机 Blender 5.2.2 已实测 Z 是相机轴向深度，单位米，含完整场景。
   初稿的射线距离假设在验证中被推翻，见 NOTE；不按旧假设读取深度。
7. 写逐件记录，删除当前模型，再渲染下一件。全部位置完成后汇总和拼图。

## Metrics

记录 planned/success/failed/not_run，包围盒是否包含于原代理、实际零件投影、
掩码像素数/边界、图像尺寸、实际耗时。记录相机 K、camera-to-world、
world-to-camera（CV 约定：X 右/Y 下/Z 前）、model-to-world。

## Decision Rule

每件必须在已验收代理内、底端与独立验收的渲染表面齐平、完整入镜并有可见掩码。代理复用不宣称每件孔槽可辨、
姿态物理稳定或 STEP 保真。没有 Agent 视觉评分；人工质量判断与技术完成分开。

## Edge Cases

空机位、抽样数超过候选数、输入摘要不符或已有输出目录：开始前报错。
损坏 STL、空/非有限/零尺度网格：记该件失败，不补抽。一个逐件异常边界允许继续
后续样本；工具崩溃时已完成记录保留，其余记 not_run。缺少源字段不静默猜测。

## Validation / Sanity Checks

检查可重复抽样、无重复摘要、机位数量均衡；小批真实 Blender 运行验证模型替换
不会累积、变换可逆、原姿态不旋转、接触高度、像素掩码和深度读回；检查实图。
验证通过后后台启动 100 件，不人工监看或改动运行中的拍摄方案。

## Implementation Constraints

独立 `step3/`，以可读主循环组织，配置用 JSON；不修改 Step 1/2 行为。
复用 Step 2 摄影配置和矩阵设置；不引入通用任务框架、自动返修、LLM 或新依赖。
单 Blender 进程，CPU，不占用训练 GPU。新运行新目录，源场景只读。

## Deliverables

代码、中文说明、验证记录，以及固定 100 件的请求清单与批量运行产物。
状态区分已实现、已验证、已启动、已完成；不将部分输出报告为全量成功。
