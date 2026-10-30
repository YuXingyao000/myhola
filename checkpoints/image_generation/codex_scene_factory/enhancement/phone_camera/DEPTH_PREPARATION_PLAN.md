# Depth ControlNet 准备协议（2026-10-27）

## Task
为更强的摄影重绘提供背景布局条件。下载专用Depth ControlNet，固定当前6件源图，
从Step3渲染深度构建近白远黑的控制图；本步骤不运行diffusion，也不占GPU。

## Assumptions
输入元数据声明为相机轴Z、米制、全场景首表面；本步骤沿用该契约，不重新定义几何。
使用渲染1/Z形成相对深度条件，不声称与训练使用的Depth Anything V2输出分布完全一致。
ControlNet是软约束，不保证细小道具、纹理或CAD几何保真。

## Inputs
`samples.json`固定6条；每条`source.images.depth`为512×512 EXR，RGB和mask均保留。
模型固定Shakker-Labs/FLUX.1-dev-ControlNet-Depth，revision
`2fdfbcf28432e544b41151b1064a1666fd0d5b47`。

## Outputs
模型目录`/mnt/d/model/Flux1_ControlNet_Depth`，保存权重/config/README/许可来源及SHA。
控制图目录`/mnt/d/data/cad_scene_enhancements/office_phone_depth_inputs_20261027_001`，
每件`depth.png`（RGB三通道、512×512）、有效区域mask、metadata，另含冻结请求与总预览。

## Processing Protocol
有效像素为finite且Z>0。对有效像素取d=1/Z，然后按本帧有效d最小/最大值做min-max，
四舍五入到0–255；近为白、远为黑；无效区域固定0。不做百分位裁剪、不平滑边界、不挪动像素。
记录Z/d范围、有效数量、输入及输出摘要。优先用隔离目录的小型OpenEXR读取原Z通道；
不能使用时才以现有Blender CPU读取，记录读取方式，避免改动训练Python环境。

## Metrics
6件计划覆盖率；控制图尺寸/通道/动态范围/有效比例；输入和模型完整SHA；制作耗时。
预览确认桌面、屏幕、打印机、柜体、地面和CAD与源RGB对应。上述检查非真实感指标。

## Decision Rule
全部摘要和图像对应通过后交付外围diffusion流程。没有有效深度或深度常数时明确失败，
不默默用空白图替代；失败保留在6件分母中。模型下载完成须校验整个权重文件，不以文件存在代替。

## Edge Cases
NaN/Inf/非正像素视为无效且输出黑；全无效、常数深度、尺寸不符直接失败。
缺文件报错；新输出目录不覆盖旧产物；原EXR不修改。

## Validation / Sanity Checks
独立核对归一化式、近/远极值映射、RGB三通道一致、有效mask、摘要及全部六张布局。
不读取或修改GPU训练状态；下载单连接或极低并发，不运行渲染任务。

## Implementation Constraints
只新增`prepare_depth.py`与本准备文档；不改模型/比较器/配置文件。
只保存透明可读的实验逻辑，不增加自动重试工作流或通用插件层。

## Deliverables
本地完整且验证的ControlNet、可复现深度转换脚本、6份控制图与预览、输入输出摘要和解释。
