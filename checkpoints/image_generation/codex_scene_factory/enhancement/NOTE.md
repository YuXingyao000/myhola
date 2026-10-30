# Step 4 实现与试验记录

最新的2026-10-28分层实验见[背景底图与光影合成](phone_camera/background_plate/NOTE.md)：
固定六件比较Depth/Canny背景增强，保留CAD及渲染光影；已减少旧底座/脱离阴影问题，
目标表面仍有渲染感。下面100件为既有保守基线，不代表新方案已批量采用。

## 当前交付结果

Step 4 独立批处理已实现，八份场景提示词已准备。办公室100件全部完成增强，
0失败、0缺失；保持512×512，保护区域最大像素差0。结果位于
`/mnt/d/data/cad_scene_enhancements/office_100_krea_20261027_001`，
包含 `preview.png`、十页 `comparisons/`、逐件生成/合成图与mask、冻结源码/请求、日志和报告。

本轮采用Krea Q8、原生512、摄影caption、strength0.25、guidance4.5；
40时间表步数实际执行10步。100张纯推理278.69秒，逐件阶段含读写共318.64秒，
不含加载/摘要准备及最后汇总；单张中位2.79秒，PyTorch峰值allocated12.74GiB、reserved13.23GiB。
GPU7已释放。源100张Blender图以及全部探索结果保留。

**结论：可复现的保守增强基线已完成，尚不能认定明显的照片真实性提升。**
低强度主要改变纹理、影调与少量细节；高强度虽变化更大，却重画背景甚至添加零件外结构。
人工质量验收保持pending；技术像素保护不覆盖掩码以外的幻觉、阴影或反射一致性。
此次只处理办公室，其他七场景的prompt未实测。没有下游CAD重建增益结论。

独立复核见 [batch_validation.json](batch_validation.json)：100/100技术通过。
按运行前记录的每位置最先两件计划，抽查10/100，发现2件需标记：
`00320493`的笔变成半透明环/片状物（明确伪影）；`00269315`的笔缩短、桌面新增小亮点
（较轻变化待人工复核）。其他90件未视觉审阅，其余8件未见同等级问题也不代表照片感提升。
详见 [batch_visual_review.json](batch_visual_review.json)。本批标记为
`not_accepted_candidate_batch`；不能把100技术成功转述为100张合格训练数据。
对照图 `review_preview.png` 同时展示批次首件与明确失败例，全部100件保留。

## 2026-10-27：输入纠正与首轮增强

目标是背景照片质感增强，输入输出固定 512×512。每场景一份固定英文提示词，
不逐件调用 LLM。原始 CAD 可见像素保持不变。完整约定见 [PLAN.md](PLAN.md)。

### 已确认的输入事实

旧 `office_step3_100_20261026T104030Z/result` 请求包含 100 件，但只完成 34 件，
且使用修复前的位移支撑面，旧进程仍暂停，不能把目录名当作 100 张可用图。

用户释放 GPU 7 后，用相同 jobs/尺寸/seed/相机/移动矩阵、新版支撑场景重拍：
`/mnt/d/data/cad_scene_renders/office_step3_100_support_fix_20261027_001`。
100/100 完成，图像与 mask 尺寸正确，contact/framing 全通过，旧数据保留。
验证见 `../batch_capture/gpu_rerender_validation.json`。这是办公室五个位置的 100 件，
不能描述为八个环境均已实验。GPU0–6 的训练未调用或修改。
实际来源均为 training split；五个摆放位置各20件，mask占图面积约0.40%–11.55%。
本轮用于生成管线开发，不作为独立测试集或真实照片能力评估。

### 模型与实现依据

本轮开始时已有本地 FLUX.1 Kontext dev 完整组件及 Q8 GGUF；其任务是图像编辑，
本轮先验证它能否在保持布局的条件下改善环境质感。选用本地模型不等于证明最优。
随后增加 Krea 对照。两种模型使用各自 Transformer 与配置，不混用权重。

- [官方 Kontext 模型说明](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev)
- [Diffusers 0.37.1 FLUX / Kontext 接口](https://huggingface.co/docs/diffusers/v0.37.1/en/api/pipelines/flux)
- [GGUF 与 offload](https://huggingface.co/docs/diffusers/v0.37.1/en/quantization/gguf)
- [Krea 官方说明](https://huggingface.co/black-forest-labs/FLUX.1-Krea-dev)

Kontext 对照使用 `FluxKontextInpaintPipeline`，原图既是带有限噪声的采样起点，也是参考。
关闭 `_auto_resize` 并明确 `max_area=512**2`，避免默认升至约 1024 档。
mask 在模型内部二值化，羽化只用于最后的像素合成。实际步数从 pipeline 读取，
不把时间表步数当成每张实际执行步数。CLIP 分支按其 77-token 上限截取，T5 分支
接收完整场景提示词（上限 512 tokens）；这是模型双编码器的输入差异。

运行环境：`/mnt/d/anaconda3/envs/generate/bin/python`；模型目录
`/mnt/d/model/Flux1_Kontext_dev`，Transformer 为
`/mnt/d/model/Flux1_Kontext_dev_GGUF/flux1-kontext-dev-Q8_0.gguf`。
完整依赖版本与实际使用文件 SHA-256 写入每次 request。

### 首轮协议及状态

修正后的固定五件位于 `office_step3_support_fix_20261026_001`。
首轮对比输出：`/mnt/d/data/cad_scene_enhancements/office_pilot_20261027_001`。
五件 × 三档 strength（0.25 / 0.45 / 0.65），40 时间表步数，guidance 3.5；
每件不同 strength 使用同一 seed，始终从原始渲染图开始。
保护外扩 2 px，外侧羽化 4 px。没有改变 CAD 本身的材质。

已通过 CPU 测试：真实五件 mask 与人工构造的细边/孔洞 mask，合成保护像素差为零；
八份 prompt 与场景注册表一一对应。

首轮 15/15 完成，独立审计通过；实际 10/18/26 步，累计推理 248.47 秒，
allocated 显存峰值 12.52 GiB。背景变化轻微，没有足够证据认定照片真实感明显提升。
不能将这个技术成功当作模型选型完成。

| 输出目录后缀 | 对照 | 当前结果 |
|---|---|---|
| `office_pilot_20261027_001` | v1 prompt，0.25/0.45/0.65，五件 | 15/15 技术通过；视觉变化很小 |
| `office_pilot_20261027_002` | 同 v1 prompt，0.85/1.0，五件 | 10/10 技术通过；仍以局部变化为主 |
| `office_pilot_20261027_003` | 更明确摄影化的 v2 prompt，0.65/1.0，五件 | 10/10 技术通过；printer机位强度1有明显影调变化，其余仍弱，尚无一致的增益 |
| `office_diagnostic_color_20261027_001` | 单件，明确要求桌面变蓝，强度1 | 1/1；桌面确实变蓝，排除编辑通路整体失效；仅诊断，不作为数据集增强 |
| `office_diagnostic_resident_20261027_001` | 同蓝色诊断，模型常驻 GPU | 原始生成及最终合成均与 offload 逐像素相同；40步33.9→16.0秒；随后通过GC释放编码器循环引用，空闲常驻分配降至12.15GiB |
| `office_pilot_20261027_004` | 内部1024准备请求 | 未执行；保留请求和not_run说明，因修正编码器释放后另起005 |
| `office_pilot_20261027_005` | 内部1024→最终512，v2，强度1，五件 | 5/5 技术完成，累计302.1秒，峰值14.53GiB；无一致真实感增益，地面十字出现浅色轮廓，不能据像素保护通过认定视觉合格 |

v1 的全部八份提示词保存在前两轮的 source 副本中；随后八份 `.md` 曾整体改为
更明确的摄影转换描述 v2，现已进一步迁移到下述摄影 caption v3。每个 run 内始终一份固定场景提示词，无逐件改写。
原生512的蓝桌面诊断成功，说明 mask/采样/提示词调用通路有效；这并不证明摄影化任务有效。
Kontext 在强度1时仍接收原图参考条件，不是无条件从零生成。

### Krea 对照与批量配置冻结

Krea Q8 已下载并验证 SHA-256：
`0d085b1e3ae0b90e5dbf74da049a80a565617de622a147d28ee37a07761fbd90`。
量化仓库为 `QuantStack/FLUX.1-Krea-dev-GGUF`，revision
`b17b5df266ee23dcdea70e027173002aac36a1c8`；组件见
`/mnt/d/model/Flux1_Krea_dev/components.json`，下载来源见 GGUF 目录的 `source.json`。
下载期间磁盘阻塞导致9次传输截断，全部续传恢复、最终0失败分块；不计入图像推理失败。

普通 `FluxInpaintPipeline`，512原生、guidance4.5、40时间表步数：

| 输出目录 | 固定五件对照 | 结论 |
|---|---|---|
| `office_krea_pilot_20261027_001` | v2编辑式prompt；0.25/0.45/0.65 | 15/15技术通过；推理73.88秒。0.25变化克制；0.45开始重画家具细节；0.65新增/替换道具，零件外还出现附加结构 |
| `office_krea_pilot_20261027_002` | v3摄影caption；0.25/0.35/0.45 | 15/15独立技术审计通过，推理57.80秒；0.25最保守，0.35/0.45的地面出现淡色伪影。没有证据证明caption相对v2带来普遍质量提升 |

选择 **Krea Q8、native512、strength0.25、guidance4.5、40时间表/10实际步数**，
保护外扩2px、仅外侧羽化4px不变。选择时的输入、prompt摘要、理由见 [selection.json](selection.json)，
该文件在100件运行前冻结；不依据100件结果重新挑参数或删除难例。
选择依据是较少明显布局和接缝问题，不能描述为显著照片级提升或模型最优性证明。

八份场景prompt现为摄影caption v3；office与002实测文件逐字节相同。
其余七份仅迁移了文本格式、保留各实际场景材料及维护程度，尚无生成质量结果。
每次run固定一份场景prompt，没有逐件调用LLM或自动改写。

完整100件增强已在独立目录完成：
`/mnt/d/data/cad_scene_enhancements/office_100_krea_20261027_001`。
不覆盖原始渲染或探索阶段输出，最终质量留待人工审阅。

### 验收边界

mask 不覆盖接触阴影或周围倒影。即使零件像素不变，背景的结构或光照改变仍可能
导致拼贴感；必须查看实际图。背景变化 MAE 只记录改动程度，不能判定真实性。
增强后全场景深度不再保证与原深度对应。人工验收保持 pending，不能以成功计数替代。
