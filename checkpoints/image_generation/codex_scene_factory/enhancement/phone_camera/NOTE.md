# 2026-10-27：手机摄影质感比较

本文件记录首轮72候选；后续局部修图、分区域采样与提示词范围对照见
[续接记录](CONTINUATION.md)。原结果与失败均保留，尚无统一验收的配置。
2026-10-28新增[空背景增强与零件光影合成](background_plate/NOTE.md)，
属于已实际运行的后续机制改动；下文判断保留为首轮结论。

当前尚未达到稳定的手机摄影质感增强。LUT收益很小；较强的深度约束生成已带来可见背景摄影化，
但仍会在原零件附近新增结构，且原CAD材质与增强环境存在落差。
不能将完成技术运行、变得更锐或 RGB 变化增大报告成真实感提升。最终人工审阅 pending。

## 固定对照

六张输入详见 samples.json：原五件修正接触面的办公室 pilot，加100件中00320493原图。
它们全是 training split 的开发样本，非独立测试。使用同一图像与 seed8–13；所有源图保留。
LUT对整图调色；Kontext整图重新生成；Krea使用背景mask采样。后两路同时保留raw和
原CAD回贴版。raw尺寸512；1024是内部处理尺寸，不是输出分辨率。

## 已获得的证据

| 方案 | 技术覆盖 | 实际观察 |
|---|---:|---|
| 自适应3DLUT paired/unpaired × mix0.5/1 | 24/24 | 增加暖色和反差；1.0偏黄、压黑。CG材料/照明逻辑未改善。 |
| 标准Kontext，1024、28步、guidance2.5 | 6/6 | 多数改动小；圆盘新增黑孔；回贴后两例有亮边。没有稳定摄影收益。 |
| Krea，1024、guidance2.5、strength0.25/0.35 | 12/12 | 桌纹和设备细节被重画；十字延长、拱形开放区新增面板，回贴不能清除。 |
| Krea + DepthControlNet，全图重绘，control0/0.5 | 12/12 | 背景摄影化明显；raw全部改形/换物/移除CAD。深度保住部分大形但不能保孔槽。 |
| Depth + 目标凸包保护，512 | 6/6 | 目标主体和开放区域保留；仍有外部棱条/底座与半透明道具。 |
| 同类保护候选，内部1024/.5 | 6/6 | 大型新增结构减少，木桌更均匀CG；出现镜面地面、拱端连线。 |
| 512保护版 + 明确低光泽材料提示词 | 6/6 | 四件背景较自然、仍待人审；薄片旁新条与十字下新底座两件明确失败。 |

LUT分支详见 [修图记录](../phone_retouch/NOTE.md)。两组生成结果逐件记录分别在
[kontext_visual_review.json](kontext_visual_review.json) 与
[krea_visual_review.json](krea_visual_review.json)。以上是实际看图的Agent判断，不替代用户验收。

输出目录统一在 `/mnt/d/data/cad_scene_enhancements/`：
- `office_phone_lut_20261027_001`
- `office_phone_kontext_20261027_001`
- `office_phone_krea_20261027_001`
- `office_phone_depth_inputs_20261027_001`：已完成六件深度条件，并非生成结果。

## 代码与入口

- `packages/cad_scene_factory/step4/retouch/workflow.py`：CPU小模型修图；不调用diffusion/LLM。
- `packages/cad_scene_factory/step4/model.py`：复用权重加载，增加标准Kontext整图和DepthControlNet分支。
- 本目录 `compare_diffusion.py`：冻结配置/输入/源码 → 一次加载 → 六件循环 → 保存raw/回贴版 → 比较图。
- `profiles.json` 和两份 `.md`：候选参数与固定英文提示词。
- `prepare_depth.py`：渲染米制Z → 单帧minmax逆深度 → 近白远黑控制图；记录来源hash与归一化。
- `inspect_outputs.py`：CAD局部四栏图、双向边缘提示、亮暗截断和位移信号。不会自动验收几何。

运行候选示例（只用GPU7；新输出目录，不覆盖旧结果）：

```bash
OMP_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
/mnt/d/anaconda3/envs/generate/bin/python \
 /mnt/d/python_room_generation/checkpoints/image_generation/codex_scene_factory/enhancement/phone_camera/compare_diffusion.py \
 --profile kontext_phone \
 --output_dir /mnt/d/data/cad_scene_enhancements/office_phone_kontext_NEW_RUN
```

可选profile为`kontext_phone`、`krea_phone`、`depth_off`、`depth_on`、`depth_guard`、
`depth_guard_1024`、`depth_guard_matte`。以上均已完成端到端运行；Depth本地模型与
六件控制图已验证，目录写在profiles.json。`--prepare_only`仅冻结请求；
之后以输出目录中的 `compare_diffusion.py --run_request <request.json>` 执行实际冻结代码。

## 已明确的限制

1. 锁住原CAD像素不等于锁住结构：孔槽/拱形空隙、原轮廓外新增部分仍可被背景生成覆盖。
2. 原轮廓边缘支持率也不能单独验收：圆盘外轮廓保留，却新增内部黑孔。双向边缘仅用于提示。
3. 分辨率影响FLUX动态sigma shift，1024/.25与512/.25不是等噪声实验。实际initial_sigma另存。
4. 对已有显示映射PNG不重复AgX；LUT权重是sRGB而非XYZ/HDRNet RAW输入。这里没有复现iPhone ISP。
5. 现有生成输出没有稳定、值得转移的摄影光色，因此暂不实施“从生成图拟合颜色变换”的第三组合。
6. Depth模型在估计深度上训练；真实1/Z是合理控制候选，不声称条件分布完全等价。

## 原始来源

- [Image-Adaptive-3DLUT](https://github.com/HuiZeng/Image-Adaptive-3DLUT)
- [FLUX.1-Kontext模型](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev)
- [Krea模型](https://huggingface.co/black-forest-labs/FLUX.1-Krea-dev)
- [Shakker专用DepthControlNet](https://huggingface.co/Shakker-Labs/FLUX.1-dev-ControlNet-Depth)
- [Diffusers FLUX接口](https://huggingface.co/docs/diffusers/v0.37.1/en/api/pipelines/flux)

## 交付与判断

本轮固定6件共产生72个候选：24修图，48次diffusion（每次另存raw和protected两版）。
[独立技术审计](final_audit.json)确认72/72完整，120个输出文件尺寸/摘要/源码/输入/seed/
合成检查一致，0失败/缺失/重复。**这是技术完整性，不是72件质量通过。**
六件实际mask无封闭二维背景孔；凸包对拱形空隙的保护实际验证，封闭孔则另做合成掩码检查，
不能以空真检查推断所有CAD孔洞均经实测。原面内部新黑孔在完整生成时也已真实出现。

最后一组低光泽材料提示词的逐件判断见 [matte_visual_review.json](matte_visual_review.json)：
00301161、00489180、00269315、00320493的环境更自然，但原CAD依然较平，尚待人审；
00495463仍新增透明/金属棱条，00182999仍新增底座/板片，不接受这些结构破坏。
这里的“四件”是这组开发样本的观察，不是总体通过率，不自动纳入训练集。

完整对照：`/mnt/d/data/cad_scene_enhancements/office_phone_comparison_20261027_001/`，
两页comparison_00/01.jpg保留六件顺序；preview.jpg并列普通收益例与失败例。
最终diffusion候选在`office_phone_depth_guard_matte_20261027_001`；没有覆盖旧100件。

成本见 [performance.json](performance.json)。LUT12次CPU推理累计0.279秒；48次diffusion
纯推理约641.7秒，末版每件约7.55秒、峰值分配15.74GiB，所有输出512×512。
下载/文件摘要/模型加载/制图不含在纯推理时间中。仅用GPU7，全部本轮任务已经退出，
该卡回到15MiB/0%利用率；没有调用逐图LLM，没有修改GPU0–6作业。

本轮决定：小模型LUT暂不作为主线；继续研究“较强生成 + 几何/接触约束”，保留其真实纹理收益。
不能靠仅锁CAD像素验收，也不能继续单纯扫strength。剩余关键问题是目标周围新增结构、
原CAD与环境材质/光色一致性及可用于自动过滤的几何证据。当前没有统一可推荐的六件全通过配置，
手机照片质感目标保持未完成，不扩大100件，不宣称已复现iPhone ISP。
