# 2026-10-27：局部修图、分区域采样与提示词范围

后续2026-10-28已实现并验证[空背景增强与零件光影合成](background_plate/NOTE.md)。
下文“下一步/待验证”是本轮结束时的历史状态，不覆盖后续结果。

本轮推进了两条路线，但尚未达到可统一用于批量数据的手机照片质感。
更值得保留的候选是“只描述背景摄影效果与实际材质”的深度约束增强；
DeepLPF局部修图和大范围保护圈均没有解决主要质量问题。

## 实际完成与判断

沿用原六件开发输入、相机和seed8–13，没有重选样本或抽种子。
DeepLPF增加12个修图候选，diffusion三组各6次，共18次采样；
每次另存raw和保护合成，spatial还存材质合成，共42个diffusion图像文件。
所有输出均512×512，原数据、上一轮候选和原100件结果均保留。

| 方案 | 具体改变 | 实际结果 |
|---|---|---|
| DeepLPF与平滑局部增益 | 真实空间变化的摄影滤镜；CPU、无需训练 | 12/12完成。主要改变冷暖、明暗与反差，未明显改善CG材质。不能据此否定所有轻量模型。 |
| spatial | 扩大凸包周边保护区；低噪阶段才释放CAD内部 | 6/6完成。新增物移到圈外，木纹出现接缝；圆盘细槽被改变，材质改善微弱。 |
| generic surfaces | 原matte配置，只移除零件、纸张、铅笔等物件要求 | 6/6完成。旧底座、棱条、纸笔消失，但地面被改成镜面。 |
| actual office surfaces | 与上一行同实现，只明确低光泽木面、哑光墙、低绒地毯 | 6/6完成。哑光地毯恢复，旧底座/棱条未重现；接触影与背景设备结构问题仍在。 |

这些结论来自完整六件实际看图，不是自动质量评分。最后一组中，
00301161、00495463、00269315有值得讨论的背景改善；00182999仍有
与零件分离的模糊暗条/接触影，00489180存在叠层设备与双支撑疑点，
00320493设备局部也被改成不明块状。六件均未被自动接受为训练数据。

## 影响下一步判断的证据

1. **修图与材质生成不是同一能力。** DeepLPF确实具备空间变化能力，
   不是再次测试全局LUT；但其摄影滤镜没有恢复输入缺少的反射/材质证据。
   详情见[局部修图记录](../phone_retouch_local/NOTE.md)。
2. **薄片不能靠生成模型“变立体”。** 原STEP十字厚度2.16mm，当前投影
   仅0.92px；网格/主要平面法线未发现压扁错误。其主要镜面方向命中灰地毯，
   解释了灰片外观。块和拱主要反射木桌。原场景材质是地毯而非旧prompt所写vinyl。
   [实际STEP、Blender与射线证据](INPUT_AUDIT.md)。
3. **背景编辑的提示词不应要求另生成前景类别。** 原matte文字同时主动要求
   金属零件、纸张和铅笔。移除这一组物件描述后，同seed的新增物明显减少。
   这是整份任务描述范围的证据，不能单独归因某个词，更不证明完全无幻觉。
4. **扩大保护圈不能代替结构约束。** 在本次spatial方案中，新增物被推到
   圈外，且原木纹与增强木纹衔接可见。材质分支从第20个去噪步骤开放，共7步；
   十字和拱没有安全的8×8内部单元，实际未更新材质，不能将其计为材质试验成功。

## 代码、协议与验证

- 新增`spatial_sampling.py`及[冻结协议](SPATIAL_SPEC.md)。公共
  `step4/model.py`只增加可选的逐步mask回调；默认路径保持关闭。
- `compare_diffusion.py`仍是一个模型加载后遍历固定清单的研究驱动。
  spatial输出与严格原像素输出分开，前者不借用后者的几何保证。
- 提示词与配置仍独立存md/json：`krea_existing_surfaces.md`、
  `krea_office_surfaces.md`、`profiles.json`；没有逐图LLM调用。
- [CPU采样时序验证](spatial_sampling_validation.json)、
  [spatial独立技术审计](spatial_technical_audit.json)、
  [两组提示词对照审计](continuation_technical_audit.json)均完成。
  技术通过仅证明输入/输出、实际采样时序、原像素保护和合成实现，不证明真实感。
- 相对旧matte，源码快照多了未启用的spatial分支，不能称整个代码快照相同；
  配置与有效采样路径仅提示词改变。generic与actual两组源码逐字一致。
- 视觉记录：[spatial](spatial_visual_review.json)、
  [独立第二审图](spatial_second_review.json)、
  [generic](surfaces_visual_review.json)、
  [actual materials](office_surfaces_visual_review.json)。

## 运行与产物

完整六件两页对照和旧/新失败例：
`/mnt/d/data/cad_scene_enhancements/office_phone_comparison_20261027_002/`。
`preview.jpg`展示00301161和00182999；`comparison_00/01.jpg`保留六件原顺序，
包含原图、DeepLPF和最后背景增强，不只展示改善样本。

最后一组目录：
`/mnt/d/data/cad_scene_enhancements/office_phone_depth_office_surfaces_20261027_001`。
独立新目录复现命令：

```bash
OMP_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
/mnt/d/anaconda3/envs/generate/bin/python \
 /mnt/d/python_room_generation/checkpoints/image_generation/codex_scene_factory/enhancement/phone_camera/compare_diffusion.py \
 --profile depth_office_surfaces \
 --output_dir /mnt/d/data/cad_scene_enhancements/office_phone_NEW_RUN
```

只使用GPU7，三个任务均已退出，该卡恢复15MiB/0%；未修改0–6训练作业。
CPU局部模型/投影约3.33秒，diffusion具体时间见[性能记录](continuation_performance.json)，
不含下载/散列/加载/制图。当前production默认配置未升级，不扩大100件。

## 尚未解决

背景会重画原有设备，且接触阴影与受保护CAD的关系仍可能错误。当前没有统一
六件通过的配置，不能声称手机照片目标完成。下一步应针对这些机制设计改动：
例如把无零件背景的增强与CAD/接触光传输合成分开，或用场景自身的材质和照明
改善目标外观；这两项在本记录中均是待验证方案，不是已完成效果。
