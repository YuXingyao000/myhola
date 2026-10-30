# 手机摄影质感探索：自适应 3D LUT 分支

## Task
实际试用官方 Image-Adaptive-3DLUT 的 sRGB 预训练，对照当前 Blender 图是否能获得更自然的摄影影调。它是修图候选，不预设其能解决材质、光照或道具造型问题；也不是 iPhone ISP 的复现。

## Assumptions
输入为 Step 3 已显示变换的 8-bit RGB PNG，按 sRGB 图像输入模型，不重复施加 AgX/线性曝光。图像可整体调色，包含 CAD；保持采样位置和尺寸，但不再宣称 CAD RGB 不变。仅 CPU 推理，不触碰 GPU。

## Inputs
固定五件：`office_step3_support_fix_20261026_001` 中 00301161、00489180、00495463、00182999、00269315；另加入 `office_step3_100_support_fix_20261027_001/00320493` 原图，作为此前 diffusion 明确失败案例的同源对照。均为训练集开发样本，非独立测试集。

## Outputs
新目录保存冻结请求、输入/权重/代码摘要、paired/unpaired sRGB 结果及原图对照、每件时耗与颜色/裁剪描述统计。每种模型固定 mix=0.5 和 1.0，共 6×2×2=24 张；原图不覆盖。

## Processing Protocol
锁定官方仓库 commit `b491f6df64a588864739a157db271e5c848e1805`；下载 sRGB 四份小权重及必要源码作为来源证据。先复核 classifier 结构与原始三线性插值规则，再用纯 PyTorch 推理。输出为原图与 LUT 调色的确定性 RGB 混合；无随机生成、无空间变形、无逐图调参。

## Metrics
记录 24 张计划覆盖率；输出有限性/尺寸；模型推理秒数；RGB MAE、黑白端裁剪占比，均为描述指标，不称真实感指标。逐图看图记录色偏、暗部丢失、亮部损失、CG 感是否改善；人工验收 pending。

## Decision Rule
先确认插值与官方实现等价，再判断视觉结果。保持全部候选，不根据像素变化大小选择“更真实”。若只让画面更暗/更艳却无自然感改善，明确判为本轮未解决目标，不扩大批量。

## Edge Cases
固定清单不得为空；源文件缺失、非法尺寸、NaN/Inf 直接失败并保留运行记录，不自动补图或换样本。无分母的统计写 null，不忽略失败。新运行不覆盖已有目录。

## Validation / Sanity Checks
使用随机 LUT、颜色边界、identity LUT 与独立直接三线性公式核对，特别检查通道与 B/G/R 轴顺序、官方 1.0001 分箱宽度；比较真实权重输出。严格加载 state_dict；实际读取全部 24 张尺寸与摘要并看对照图。

## Implementation Constraints
代码在 `step4/retouch/`，实验记录在本目录。复用现有 Torch/Pillow/NumPy，无安装旧 CUDA/C++ 扩展，无 GPU，无大模型。保留官方 LICENSE 与来源信息。逻辑保持输入→预测 LUT→调色→混合→保存，避免防御性框架。

## Deliverables
可独立运行的 CPU 修图脚本、固定输入配置、24 张结果及对照图、数值核对报告、实际看图的结论与限制、来源/hash/时耗。此分支不改 diffusion 主代码或已有结果。
