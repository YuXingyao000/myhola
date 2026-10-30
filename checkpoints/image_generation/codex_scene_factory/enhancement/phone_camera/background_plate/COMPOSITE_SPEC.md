# 空场景增强与配对差值合成（2026-10-27，实跑前冻结）

## Task

把 CAD 从扩散输入中移除：先增强空场景，再转移配对渲染中的 CAD 及其对环境的影响。检验是否减少旧方案的伪底座和分离接触影。固定比较两种合成公式，不扫描强度。

## Assumptions

I 是带 CAD 渲染，B 是空场景渲染，E 是增强后的空场景。I/B 必须共享相机、几何布局、材质、灯光、采样设置、色管；B 中 CAD 对所有射线不可见，不能只关闭 camera visibility。固定 seed 仍不保证两次 Monte Carlo 噪声及去噪误差完全抵消。

三者为 sRGB 编码的显示用 RGB PNG。对 PNG 做 sRGB EOTF 后只得到 **display-linear** 值；AgX 的 tone mapping/look 已烘焙，不能称为 scene-linear 辐射值。差值和比例均为显示域近似。空场景几何和光照被生成器改变后，旧渲染的接触、遮挡和反射可能失配。

## Inputs

同尺寸的三张 8 位 RGB PIL 图像 `with_part_rgb`、`empty_rgb`、`enhanced_empty` 和非空二值 CAD `mask`，本轮 512×512。彩色输入必须为 `mode="RGB"`，不隐式转换浮点、灰度或带 alpha 图像。模式为 `additive` 或 `relative_darkening`。背景增强使用空场景的深度，不得沿用带 CAD 深度。固定边界保护为 mask 外扩 2 px，不采用旧的大保护圈。

## Outputs

`compose(I, B, E, mask, mode) -> (protected_image, metrics, diagnostics)`。主图的 CAD+2 px 原像素严格回贴；诊断包含未回贴 raw、凸包保护对照图、保护 mask、差值热图、边缘重合图和浮点绝对差。所有结果仍是合成图像。

## Processing Protocol

1. sRGB EOTF 解码得到 i、b、e。保存 d=i-b，不做阈值删差值或平滑；微小差值可能是光照，也可能是渲染噪声。
2. 加法：`c = e + d`。
3. 相对暗化：`n=min(d,0)`、`p=max(d,0)`，`r=1+n/max(b,1e-4)`，`q=n-b*(r-1)`，`c=r*e+p+q`。r 限制在 [0,1]（输入为非负 RGB），只传递暗化比例；p 是正差，不冒称纯反射；q 是近黑分母保护的恒等性修正。两模式在 E=B 时均严格重建 I（浮点容差内）。
4. 记录 c 裁剪前范围、负值和溢出，再裁剪为 [0,1]、sRGB 编码和量化。
5. 主图仅在 CAD mask+2 px 区域逐字节恢复 I。孔隙和凹形空区允许显示增强背景，不填成物体，也不强制恢复大片旧背景。额外凸包保护图只作为诊断，不作为主输出。
6. 差值影响范围原样显示；以原空图/增强空图强边缘的重合及新增边缘定位潜在支撑面/道具位移。信号与叠图只辅助人工定位，不当作几何门槛；低纹理区域没有边缘也不能证明未移动。

## Metrics

两模式分别记录恒等重建最大浮点/8bit误差、CAD+2px误差、裁剪前最小/最大值及越界通道比例、原差值在目标外的分布、原空图边缘在增强图 2px 内的支持比例、增强图未匹配边缘比例。边缘指标在全图及 CAD 邻域（bbox+64px）分别报告，无阈值自动通过。

## Decision Rule

固定六件两模式全部保存。工程要求为 E=B 可复原 I、保护区域字节一致、无 NaN/Inf。视觉上需确认接触影贴合、没有伪底座、背景没有新增遮挡或移动支撑边界，且增强确有照片收益。工程通过不替代视觉/几何验收。若背景更换支撑平面、光源方向或 BRDF 后残差失配，明确标为不可直接合成，不试图用该公式自动矫正。

## Edge Cases

尺寸不符、空/全图/非二值 mask、非 8 位 RGB 彩色输入直接报错；失败由外围保留。浮点图（包括 NaN/Inf）不属于输入契约，不能先转成 8 位图掩盖原值。合成核心仍检查输出有限性。不使用恒等测试的相等分支作弊。近黑底图使用 q 维持恒等性。外扩可能覆盖细小孔边缘，记录保护像素数。若无边缘，支持指标为 null 而不是完美分数。输出裁剪可能损失暗部/高光，因此报告裁剪前值。

## Validation / Sanity Checks

CPU 合成样例验证 E=B 恒等、零差值、带阴影/正反射的解析值、近黑分母、裁剪报警、CAD精确回贴、孔隙采用新背景和边缘位移信号。仅使用 NumPy/Pillow/OpenCV/SciPy，不加载模型。

## Implementation Constraints

简单函数，无 class/重试框架，不修改共享 step4 模型或渲染驱动，不占 GPU。本轮不将 AgX PNG 逆变换称为真实辐射恢复；以后若改 EXR，需保留线性场景值和 OCIO 元数据，并另行处理生成背景没有 HDR/实际光照的问题。

## Deliverables

独立 `composite.py`、CPU 验证脚本/JSON及外围保存的六件输出。理论参考：[Debevec 1998 原论文，第6节](https://pauldebevec.com/Research/IBL/debevec-siggraph98.pdf)以原背景加有物体/无物体渲染差值转移环境影响，成立依赖局部几何及照明近似。这里使用显示域 PNG，范围比原辐射域方法弱。[Blender 色彩空间说明](https://docs.blender.org/manual/en/latest/render/color_management/color_spaces.html)和[图像格式说明](https://docs.blender.org/manual/en/latest/files/media/image_formats.html)区分场景线性中间文件和带显示变换的 PNG；sRGB EOTF 不会消除 AgX。
