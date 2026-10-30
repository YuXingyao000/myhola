# 2026-10-28：真实覆盖alpha与背景光影，十件实验

## 目的与当前状态

用户确认空背景增强路线，并指出零件周围色差。此前诊断确认：旧主图将CAD mask
外扩2px并硬回贴，带回旧桌面/地毯形成色圈；更宽色斑还涉及旧RGB光影转移。
本轮禁止修改几何、拓扑、姿态与相机，允许光照变化；保持原材质。

独立合成算术检查4/4通过。001完成首件五层预检，002已完成正式十件运行：
渲染10/10、增强合成10/10、独立产物核验10/10，无失败或缺失。
首次六件复用既有增强背景，新增四件各生成一次；没有按效果重抽。
用户后续反馈只有十字件较好，部分图片周围道具扭曲；本批不能记为整体视觉通过。
没有升级生产默认或扩到100件。运行JSON中的 `human_review: pending` 是交付当时状态，
后续反馈在本文追加，保留原始运行报告。

## 方法与代码

- `prepare.py`：固定此前六件，再按100件清单顺序补四件；冻结源码、输入、35个外部素材。
- `render_layers.py`：原图I、空背景B、透明前景F、背景交互G、shadow-ray干预S五次渲染。
  F使用非CAD几何holdout保留真实遮挡、间接光与反射，PNG为straight RGBA；
  同时保存scene-linear premultiplied EXR和未去噪EXR。
- `compose_layers.py`：先转移背景明暗，再用F真实alpha叠加。主背景增益为Y(G)/Y(B)，
  S用于分开保存shadow干预和其余间接作用；G-S不冒称纯反射。
  保留增强背景色度，有意不转移旧场景彩色光影；这会损失真实彩色反射，是当前取舍。
- `enhance.py`：六件复用既有Canny背景，新增四件仅一次固定seed增强；保存新主图、
  旧硬边合成、完整RGB光影转移诊断及全量前后对照。
- `validate_composite.py`：CPU独立解析核对alpha、相邻透明像素、亮度增益与裁剪记录。

PNG中的AgX仍已烘焙，sRGB EOTF只得到display-linear。此处不是场景线性重光照，
不承诺任意新背景均可物理正确融合。几何标签不变仍不代替最终图像的可辨性验收。

## 首件预检与协议修订

001首件00301161五层成功，原二值mask变化0，真实部分覆盖alpha像素1073。
PNG alpha与EXR alpha最大误差0.0019608，符合8bit四舍五入。
未去噪线性F-over-G相对I：全图MAE2.55e-5，不透明区MAE3.87e-8，
边界MAE0.00622；独立采样/过滤使边界不能要求严格恒等。
去噪后的对应全图MAE0.000207，不透明区MAE0.001799；不把去噪差异藏成精确重现。

原设计要求新B与旧B字节完全一致，但首件B仅47像素差1级灰度，MAE0.0000598/255；
I仅44像素差1级。002在执行其余九件前明确冻结max<=1的复用容差，并逐件记录差异。
不是静默fallback，也不按画面质量改seed。001作为预检保留：1件成功、9件未执行，
不是十件成功运行。新旧合成对照使用同一当前I/B/E，避免重复抽样混入合成比较。

## 运行目录与命令

预检：`/mnt/d/data/cad_scene_enhancements/office_alpha_layers_10_20261028_001`。
正式：`/mnt/d/data/cad_scene_enhancements/office_alpha_layers_10_20261028_002`。
每件输出512×512，原render seed和增强seed分开记录；仅GPU7，旧停止进程不动。

```bash
/mnt/d/anaconda3/envs/generate/bin/python prepare.py --output_dir /path/to/NEW_RUN

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=8 \
/mnt/d/natural/blender-5.2.2-linux-x64/blender --background --factory-startup \
  --python-exit-code 1 --threads 8 --python /path/to/NEW_RUN/render_layers.py \
  -- --request /path/to/NEW_RUN/request.json

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=8 \
/mnt/d/anaconda3/envs/generate/bin/python /path/to/NEW_RUN/enhance.py \
  --request /path/to/NEW_RUN/request.json
```

## 十件结果与证据

固定顺序为00301161、00489180、00495463、00182999、00269315、00320493、
00985631、00062376、00121033、00199547，均为训练开发样本，不是独立测试集。
所有最终图为512×512；每件目录包含 `final.png`、alpha、五层渲染、增强空背景、
旧合成对照、RGB光影转移对照、全图和边缘放大对照。全部十件均展示，未只选成功子集。

- [十件总览](/mnt/d/data/cad_scene_enhancements/office_alpha_layers_10_20261028_002/gallery.png)
- [生成覆盖率](/mnt/d/data/cad_scene_enhancements/office_alpha_layers_10_20261028_002/enhancement_summary.json)
- [完整数值核验](/mnt/d/data/cad_scene_enhancements/office_alpha_layers_10_20261028_002/audit_results.json)
- [逐件Agent视觉审阅](visual_review.json)
- [十字薄片边缘对照](/mnt/d/data/cad_scene_enhancements/office_alpha_layers_10_20261028_002/00182999/edge_comparison.png)

独立CPU核验确认：十件源二值mask差异均为0；F在目标区域的深度与I逐值相同；
PNG alpha与EXR alpha量化一致，真实部分覆盖像素为601–1073个。最终图、不透明前景、
透明背景及全部合成诊断均与冻结公式重算一致，未发生越界裁剪。原2px圈内alpha为0的
像素每件132–287个，全部等于新背景，没有强制回贴原图。六张缓存E像素未变，
对应新B与旧B最大差均不超过1灰度级。源STL等文件及冻结脚本摘要匹配。

渲染器记录几何、法线、模型变换前后摘要一致，模型变换对原记录差为0，相机矩阵
浮点恢复误差不超过2.39e-7；这是渲染器内检查，审计未独立重建几何。
六件深度与此前256采样渲染相同。与原64采样图比较时，深度数值和独立去噪RGB
不保证逐像素一致；不能将“保留几何”写成“最终CAD颜色与原图逐像素相同”。

Agent逐件查看结果后，观察到旧浅色描边明显减轻，十字薄片和拱形件最明显；
未见明显新增孔槽、封孔或薄壁变厚。主版本木桌接触阴影偏灰，RGB转移对照更暖；
两者都减轻主要色圈，中性化本身不证明物理正确。CAD表面仍偏平、偏干净，
本轮解决合成边缘问题，不支持“整体照片感已解决”的结论。

50次渲染累计183.0秒（各次render计时，不含完整进程启动与装配）；增强合成阶段
65.7秒（该阶段计时，不是端到端总耗时）。仅使用GPU7，结束时已核对为0%利用率、
15MiB显存占用，未改动旧停止进程。001预检和002正式输出均保留。

## 当前验证边界

像素、mask、深度及几何记录核对不代替最终照片感评估。中性增益避免搬运旧色度，
也会丢失彩色间接光，且不能修复增强背景改变了支撑面、光源方向、道具或遮挡的问题。
本轮为单一办公室场景中的十件开发实验，不能推广为全部场景的质量结论。
这些输出是合成真实感图像；数值通过不代表视觉通过。

## 交付后的用户反馈与问题定位

用户指出只有十字件较好，其他部分样本周边存在扭曲物品。只读对照确认：
00489180的小主机、纸笔，00301161的显示器支柱与键盘，00062376的小主机/杯柄，
00121033的打印机结构，已经在 `enhanced_empty.png` 中被改写，最终合成沿用了这些变化。
此问题出现在背景生成阶段，修复旧2px回贴圈没有解决它。

实际调用为FLUX Krea Q8 + Union Pro 2 Canny，输入是移除CAD后的完整场景图，
edit mask为全白，strength=0.65，control scale=0.7，control end=0.8。
“空背景”仍包含全部周围道具；全白mask让它们都可被重画。背景全量参与合成，
本轮没有使用此前的0.5混合；strength也不是原图/生成图混合权重。
Canny是条件引导，没有逐道具几何保留约束。六件复用了先前增强结果、四件各新生成一次，
均无背景道具变形拦截或自动视觉返修。执行成功与视觉可用必须分别报告。

原CAD前景继续使用原场景照明，没有针对生成背景重新照明；这还可能造成整体融合不一致。
本次仅定位和说明问题，未修改生成参数、合成代码或重跑图像。
