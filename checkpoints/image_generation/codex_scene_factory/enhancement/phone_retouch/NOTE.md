# 2026-10-27：自适应 3D LUT 真实运行结果

## 结论

**已实际运行并逐图查看，但当前 LUT 修图没有解决“像 iPhone 随手拍”的目标。**
它增加对比度、改变颜色，不补充材质细节、不改变反射构成，也不能解决 CAD 曲面法线或道具造型问题。
mix0.5比较温和；mix1常把桌面/零件染得更黄，并压暗黑色外壳。没有依据把这种变化称为真实感提升。
24张均保留，不扩为100张，不据此筛掉原图；人工验收仍 pending。

这是修图方向的一个明确候选结果，不代表所有小模型都无效。该预训练学习 FiveK 照片修图，
与当前已显示变换的合成渲染有域差异；并未针对 iPhone 或 render-to-real 训练。

## 输入、方法与产物

代码：`packages/cad_scene_factory/step4/retouch/{workflow,model,validate}.py`。
实际输出：`/mnt/d/data/cad_scene_enhancements/office_phone_lut_20261027_001`。
固定清单见 [inputs.json](inputs.json)：此前办公室5件 pilot 的原图，加 full100 的00320493原图。
不是处理 diffusion 已产生伪影的图；00320493只作为同源对照，因此不能说 LUT 修复了生成伪影。

输入输出512×512；整图调色，含CAD像素。使用paired/unpaired sRGB各一份，mix0.5/1.0。
每件输出两行三列 `comparison.jpg`：原图、mix0.5、mix1；上行paired，下行unpaired。
保存原图摘要、完整来源字段、实际代码副本、Git信息、权重来源/hash、24条结果和汇总。

CPU4线程；6×2=12次模型推理共 **0.279秒**（两种mix复用一次推理），
包括模型读取、保存和制图的阶段为 **4.99秒**；不含Python启动和请求准备。
未使用GPU、未训练、未安装新依赖。

## 六件实际看图记录

| CAD | 观察 |
|---|---|
| 00301161 | paired1与unpaired1使前景金属板和木桌明显偏黄，键盘/盒子黑色更重；模型的平整CG表面依然存在。0.5较柔和。 |
| 00489180 | 圆盘前面仍近乎均匀米色，LUT未生成真实金属反射层次；桌边更黄、机箱更黑。 |
| 00495463 | 打印机暗部和桌边对比加强，但打印机造型、过于一致的木纹和零件弱可辨性未改善。 |
| 00182999 | paired1明显提亮地面并加强柜下暗部，零件仍像均匀灰色十字；unpaired变化偏小，未消除合成感。 |
| 00269315 | 咖啡和便签颜色更浓，黑色机箱压暗；弧形CAD的反射和场景材质仍未达到手机摄影观感。 |
| 00320493 | 笔和便笺位置/形状保留，未出现此前Krea的半透明环；这是非空间颜色映射的预期性质，非真实感成功证明。 |

以上为 Agent 对24张候选的视觉观察，尚无独立人工评分或下游CAD结果。

## 验证与描述统计

官方仓库commit：`b491f6df64a588864739a157db271e5c848e1805`；
文件及SHA在 `/mnt/d/model/Image_Adaptive_3DLUT/source.json` 和每次请求中。
仅下载sRGB四份权重，合计约4.76MB；没有使用XYZ权重。
权重加载strict，paired参数593,516、unpaired593,068（含三张LUT）。

旧demo依赖不可直接使用的扩展，本实现用 `grid_sample` 替代。坐标明确采用
`models.py` 的 `1.0001/(33-1)`（`models_x.py`另用1.000001，未混淆）。
LUT维度是输出通道/B/G/R，grid坐标是R/G/B。

[validation.json](validation.json)：官方分类器定义与本实现输出最大差0；真实LUT与独立
NumPy三线性公式最大差2.38e-7；随机LUT最大差3.22e-6；identity边界偏移符合官方约定。
因此变化有限并非classifier结构或RGB轴弄错的已知实现错误。

[output_validation.json](output_validation.json)：24/24读取成功，无重复/缺失；所有源图摘要未变，
所有输出512×512×3且摘要一致。该审计不验证真实感和每一处几何可辨性。

| 候选 | 平均RGB绝对变化 /255 | 输出通道落在0的比例 |
|---|---:|---:|
| paired 0.5 | 7.42 | 0.163% |
| paired 1.0 | 14.65 | 1.391% |
| unpaired 0.5 | 7.35 | 0.032% |
| unpaired 1.0 | 14.65 | 0.826% |

这些是颜色变化/暗端裁剪描述，不是真实感指标。mix1暗部损失值得注意，
不能因没有像素位移就认定标注证据完全无损。

## 依据

- [官方仓库与sRGB权重](https://github.com/HuiZeng/Image-Adaptive-3DLUT)
- [原论文](https://arxiv.org/abs/2009.14468)：学习图像自适应LUT以改变照片颜色/影调，支持paired/unpaired。
- 原始分类器、demo及CPU三线性源码已按固定commit保存于本地模型目录；Apache-2.0 LICENSE随源码保留。
