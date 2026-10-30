# 抗锯齿前景与背景光影分层：固定十件

## Task
执行用户批准的十件管线，去除旧mask外扩2px硬回贴造成的背景色圈。
几何、拓扑、姿态和相机固定；允许光照变化。只生成合成真实感图像。

## Assumptions
继续办公室场景。既有六件用于前后对照，另按原100件清单顺序补四件，
均为training开发数据。原材质不变。非CAD环境以holdout提取前景，保留间接照明、
反射及原有遮挡；不用简单隐藏所有环境几何。保护的是几何，不要求最终CAD RGB逐字节不变。

## Inputs
六件沿用phone_camera/samples.json；补00985631、00062376、00121033、00199547，
来自office_step3_100_support_fix_20261027_001。原相机、模型矩阵、材质、render seed随metadata。
增强seed为既有六件8–13，新增四件14–17。512×512、Cycles256 samples、GPU7。
空背景FLUX Krea Q8/Union Pro2 Canny沿用已验证配置：strength .65、guidance4.5、
40步时间表、control .7/end .8、Canny100/200，沿用office surfaces提示词。

## Outputs
新目录保留十件I完整渲染、B空背景、F透明前景RGBA、G相机不可见CAD的背景交互、
S仅shadow射线可见CAD的背景干预；RGB PNG、线性RGBA EXR、mask/depth与元数据。
增强背景E、主合成final.png、旧硬边合成对照、RGB交互合成诊断、影子/间接光增益图、
逐件完整对照和边缘放大、完整十件清单、耗时/失败/验收记录。

## Processing Protocol
1. 冻结输入、脚本、配置和场景外部素材摘要；十件全部按原位姿恢复，不改mesh/法线。
2. 拍I/B/F/G/S。F使用Blender真实覆盖alpha，透明film、非CAD几何holdout；
   G仅关闭CAD camera visibility；S再关闭diffuse/glossy/transmission/volume，保留shadow。
   S是射线可见性干预，不声称是全部物理阴影；G-S包含间接遮挡、反射和色溢，不称纯反射。
3. 仅B进入生成模型。六件复用已有E，核对B与旧输入最大RGB差不超过1/255并保存
   最大值、MAE和变化像素数；超出则中止，不静默复用。新增四件固定seed单次增强。
   修订依据：001首件预检中旧/新B仅47像素差1级、MAE0.0000598/255，I也仅44像素差1级；
   这是记录中的量化级差异，不声称逐字节输入相同。002在运行其余九件前冻结此容差。
   新旧合成比较仍共享同一当前I/B与E，不额外抽样选优。
4. PNG RGB用sRGB EOTF进入display-linear；AgX已烘焙，不当作scene-linear。
   PNG F为straight RGBA，alpha不做gamma，不再次除alpha；EXR留作独立线性分解核对。
5. 主方案保留E色度，仅转移光影亮度。Y为Rec709亮度：
   shadow_gain=Y(S)/max(Y(B),1e-6)，indirect_gain=(Y(G)-Y(S))/max(Y(B),1e-6)，
   H=E*(shadow_gain+indirect_gain)。两个增益单独保存，无额外平滑/阈值/逐件强度调节。
   这是中性色调转移近似，有意不搬运旧背景彩色光影；E=B时不承诺RGB精确还原G。
6. final=alpha*F+(1-alpha)*H，再编码PNG。没有2px扩张，没有向CAD内部羽化，
   alpha=1的区域背景贡献严格为0。阴影/反射不作为前景RGB的一部分叠加。
7. 固定诊断：使用原relative_darkening(G,B,E)作为H的RGB转移版本；
   另用原compose(I,B,E,mask)复算旧硬边版本。两项都不是额外diffusion抽样。

## Metrics
计划10件的渲染/增强/合成覆盖率，source mesh/normal/camera/transform摘要，原二值mask差异、
接触高度、alpha面积/部分覆盖像素、EXR有限性和线性F-over-G对I重建差异。
记录前景不透明区背景污染误差、alpha边界旧/新对照、增益范围、裁剪比例和耗时。
这些是实现核对，不是自动真实感评分。

## Decision Rule
先完成一件五层读取及alpha/颜色约定验证，再执行剩余九件；原固定分母始终10。
完整查看十件，比较色圈、薄壁/孔槽、接触影、背景设备、色斑和剪贴感。
若仍有边缘或光影问题明确保留，不把运行通过当作人工验收，也不自动扩大100件。

## Edge Cases
缺失输入、非法尺寸、空/全覆盖前景或NaN直接报错；仅批量入口保留逐件失败。
近黑分母用1e-6并记录极端增益，禁止静默截断掩盖；输出裁剪单独报告。
没有被原mask覆盖但由抗锯齿产生的边界像素不自动判为几何变形；核对coverage支持域。

## Validation / Sanity Checks
实际PNG alpha是否有中间值、opaque部分颜色与I是否一致、holdout是否保留原遮挡；
scene-linear前景premult/straight约定通过EXR和简单解析合成核对；新合成在alpha=0/1时
分别精确退化为H/F。旧方案保留同一输入/背景，避免把新采样当作边缘修复收益。

## Implementation Constraints
代码放本独立研究目录，复用Step3和Step4，不覆盖生产默认、旧数据或原场景。
提示词/配置独立md/json。少量直接函数，单进程，只GPU7，无LLM调用、无付费API。
不引入IC-Light、不改几何/粗糙度/材质，不尝试全局模糊遮盖色圈。

## Deliverables
十件主图与旧/新对照、独立审核、中文NOTE及可复现命令；失败和局限保留在分母与记录中。
