# 摄影增强后的轻量修图：固定六图验证

## Task

检验 DeepLPF 是否适合放在已完成的 `depth_office_surfaces` 后面，作为最后的摄影影调处理。
只跑官方模型和既有受限局部增益各一次，不调参、不挑图、不新增模型。

## Assumptions

已有事实：此前直接修原 CG 图没有明确照片化收益。此次输入已有生成式摄影纹理，可能更接近
FiveK 修图训练域；这只是待验证假设。sRGB PNG 已做显示变换，不是 RAW，不重复 tone mapping。
两路都处理 CAD 像素；局部增益不移动像素，但可能压暗孔槽，不能称为几何保证。

## Inputs

固定顺序：00301161、00489180、00495463、00182999、00269315、00320493。
输入为 `/mnt/d/data/cad_scene_enhancements/office_phone_depth_office_surfaces_20261027_001/`
各件 `strength_0.65/protected.png`，512×512；mask 沿用 `phone_camera/samples.json`。
官方代码、权重、CPU 适配器及投影公式完全沿用本目录 `run.py` 和 `PLAN.md`。
权重：`/mnt/d/model/DeepLPF`，commit `4faeef6597f299d38acc597df9cfe970f8d90ca2`。

## Outputs

新目录 `office_phone_deeplpf_refinement_20261027_001` 保存六件×两路，共 12 张结果、增益场、
逐件并列图、输入/代码/权重 hash、成本、描述统计与全部六件看图记录。原 enhanced 图不覆写。

## Processing Protocol

1. 冻结六图 enhanced 输入与原 mask 的路径和 hash；不修改固定样本顺序。
2. 复用官方 DeepLPF CPU 推理，4 线程，输入 sRGB[0,1]。
3. 投影只提取平滑局部线性 RGB 增益：sigma24px、epsilon0.01、范围[0.5,2]，不改参数。
4. 对 input / official / projected 同尺寸并列，并实际查看全部六件及目标局部。
5. 独立记录曝光、冷暖色偏、黑白截断、目标孔槽/边界可辨以及整体观感；不把更鲜艳等同照片化。

## Metrics

12张覆盖、尺寸/hash、CPU 推理/投影时间。全图与 CAD mask 内分别记录亮度分位数、RGB均值、
任一通道及全部通道为0/255的像素比例、近黑/近白比例、相对输入RGB误差。
亮度使用显示域 sRGB 的加权值，仅作描述；没有中性灰实拍参照，RGB均值不是白平衡准确性。
输入目标内部强边缘处的输出梯度均值比仅辅助定位细节变化，不用于验收拓扑。

## Decision Rule

若只是冷暖/反差变化，没有稳定的摄影收益，或暗部/孔槽退步，则不接为默认步骤。
仅技术完成不代表成功，不按六件结果挑有利图、扫描参数或扩批量。

## Edge Cases

缺图、尺寸错误、空mask或数值异常直接报错并保留现场，不替换样本；失败留在12张分母。
没有足够内部梯度的样本把该描述标为无有效边缘，不以零除或假通过处理。

## Validation / Sanity Checks

核对输入 hash 不变、输出 hash/尺寸、既有投影恒等与空间变化检查；重算保存增益到输出的误差。
全六件看图，不读取主审结论替代独立观察；保留生成图已有伪影，不能把它们算作修图新造。

## Implementation Constraints

仅CPU4线程；不改共享 step4/model.py，不安装依赖，不下载新权重，不使用GPU。
复用已有 run.py，仅新增输入清单、轻量评估与本轮记录，保持研究代码可读。

## Deliverables

冻结规格、12张新结果、六件并列和局部对照、成本/描述统计、逐件视觉结论。
明确这是对合成真实感图像的后处理，不是实拍、手机ISP复现或配对几何安全证明。
