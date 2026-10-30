# 制造表面与既有背景增强的组合

## Task
检验目标CAD均匀偏平的外观能否通过程序化金属表面改善。保留已经验证的空背景
Canny增强与相对明暗合成，比较两种明确的表面假设，不再重复轻量修图或扫描diffusion参数。

## Assumptions
数据：沿用既有六件training开发输入及实际配对图。当前STL只通过matrix_world缩放，
局部坐标不是米；scale乘1000后才得到mm。相机、姿态、尺寸、网格与法线均冻结。
任务：允许改变目标材质和由此产生的真实渲染光传输，不添加锈迹、划痕、孔槽、bevel、
displacement或Bump。已有薄片不足1px的侧面不能靠材质恢复。
实现：仅GPU7；CPU合成；没有生成模型/LLM调用。既有Canny空背景E逐字节复用。
两种材质参数是待验证的外观假设，并非测量的具体铝合金。没有需要用户确认的接口缺口。

## Inputs
`office_phone_pairs_20261028_001/request.json`及其冻结source、六组with_part/empty；
`office_phone_plate_canny_20261028_001/`的六张enhanced_empty与上一轮relative_darkening；
场景依赖列表沿用pair_capture_audit_20261028.json，运行前重新核验35个文件摘要。

## Outputs
每件两种新材质各保存RGB、二值mask、深度和scene-linear EXR；材质参数/节点约定、
网格/角法线摘要、相机/模型矩阵、接触、渲染时间和设备。CPU保存增强背景上的合成与指标。
三列对照固定为上一轮原材质、均匀银色、轻微定向加工；另存目标局部同位置放大。
所有旧数据不覆盖；12个预定候选均保留，不根据外观剔除失败样本。

## Processing Protocol
1. 准备阶段冻结旧pair请求与source、输入摘要、场景依赖、两份研究脚本与profiles。
2. 对每件从旧场景恢复原placement/camera/model；渲染seed沿用8–12/17，256 samples。
3. A：Base Color linear=[.80,.81,.82,1]，Metallic=1、Roughness=.32、Anisotropic=0。
4. B：同A基底；Anisotropic最大.30，Roughness=.32+.02*(2*N-1)。
   N来自Noise(3D,Scale=1,Detail=1,Roughness=.5,Distortion=0)，
   输入q=(x_mm/20,y_mm/.25,z_mm/.25)。没有随机偏移或逐件调参。
   Tangent为局部X轴D在当前表面的投影；各向异性乘(1-dot(N_surface,D)^2)
   在加工轴与法线平行处降到0，避免依赖不存在的UV切线。
5. 两者Coat/Sheen/Transmission/Emission=0、Alpha=1，不使用Bump或位移。
6. 不重生成背景；以新有件图I、同seed旧empty B、固定enhanced_empty E，调用上一轮
   冻结相对明暗合成，保护mask+2px逐字节复制新I。每种材质的差分各自重新计算。
7. 全部六件实际审图：亮度变化、表面方向性、金属可辨性、接触、孔槽、是否出现条纹。

## Metrics
计划12/成功/失败/缺失；mask逐像素与旧pair相同；深度最大绝对误差与不一致数；
mesh/角法线摘要前后相同；相机/模型矩阵误差；接触；源文件SHA；每次渲染耗时。
合成重新计算，目标/保护区误差0、裁剪与恒等指标。图像差分用于定位，不作真实感评分。

## Decision Rule
只有变亮或更蓝不等于更像照片；定向表面如果不可见，就如实报告没有明显收益。
网格/角法线/相机/接触/mask变化属于实现失败，不能用真实感掩盖。
不通过加厚薄片、黑线假刻槽或掩盖孔洞来提升立体感；不对六件挑不同获胜材质。
保留统一候选，最终人工状态pending，不自动扩展100件。

## Edge Cases
依赖摘要不符在渲染前直接失败。逐候选只保留一个异常边界，失败仍在12分母。
复用已有合成的输入/非有限检查；不加入重试、fallback或通用配置框架。

## Validation / Sanity Checks
Blender5.2.2 CPU节点构造检查与真实尺度检查；shader连接前后网格和corner normals摘要；
实际PNG/EXR重读核对mask/depth；所有12张合成独立复算。源scene只读且不save回磁盘。

## Implementation Constraints
独立研究目录，不改Step3/Step4生产默认。提示词/参数不塞入Python，参数在profiles.json。
复用冻结capture/render/composite函数；研究代码只负责本轮材质替换与证据保存。

## Deliverables
最小材质构造、固定12次渲染、CPU合成与全量对照、技术审计、实际视觉判断和结论。
本轮所有结果仍是合成真实感图像；没有独立实拍评估或下游重建增益结论。
