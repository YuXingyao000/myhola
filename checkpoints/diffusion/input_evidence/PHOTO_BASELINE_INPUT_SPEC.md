# 旧合成照片：输入检查与 RGB 基线讨论

2026-10-25 后续进展：本文记录前一次输入讨论；随后用户授权长跑，已完成全部 train + validation100 的 flux / 原 AE cache 校验并启动训练，见 [训练协议与状态](../photo_baseline/README.md)。下文“尚未训练”保留其当时语境。

## Task

响应用户“先用已有较差合成照片尝试训练，并建立 baseline；先直观解释图片如何处理”的请求。当前阶段确认资产、解释与冻结输入基线建议；本检查不启动正式训练，不改模型。

## Assumptions

- 目标域按单张照片处理；来源是合成真实感图像。
- 候选资产为 `/mnt/d/data/deepcad_v7_cond/{id}/real_photo.npz` 的 `flux`。原图与 GT CAD 仍需核对，不凭 ID 相同认证配对。
- 现有源 train/validation split 保留；这里只检查存在性与确定性小样本内容，不依此做全库质量过滤或研究效果结论。
- 建议的 baseline 为原版 1024 维、24 层 concat 网络，从原始 SVR 权重初始化，只做照片 RGB 条件的 DDPM 微调；DPO、新几何 loss、深度／法线条件均不属于第一版输入基线。该选择是建议，尚未训练。

## Inputs

- 源 split：`src/brepnet/data/list/deduplicated_deepcad_{training,validation}_7_30.txt`。
- 条件目录：`/mnt/d/data/deepcad_v7_cond`。
- 原版 AE 重新编码的 cache 候选：`checkpoints/diffusion/preference_data/20261020T134930Z/gt_cache`。

## Outputs

- `photo_asset_check.json`：split 摘要、照片与 cache 的存在数、缺失 ID；种子 42 确定性抽取的 4 个 validation 对象，各 NPZ 字段形状、dtype、有限性、数值范围与源摘要。
- `photo_preview_{id}.png`：抽样 `flux` 原数组的无损 PNG，仅用于观察数据，不改输入。

## Processing Protocol

按源清单逐条检查 `real_photo.npz` 与原版 cache 的存在性；区分独立存在和共同存在。只对固定抽样解压 `real_photo.npz` 的已有数组，不加载模型、不读 test 个例、不使用 GT 来裁剪照片。抽样失败保留原因，不补抽。

## Metrics

只记录资产覆盖与样本数组属性；不计算或推断几何合格率、照片质量分数、重建性能。

## Decision Rule

存在性检查通过只表示可进一步准备训练。正式基线需完整验证被接受样本可解码、cache 与原版 AE 同源、视图／旋转对应、无跨 split ID 重叠；几何配对质量单独记录。现有历史质量报告不能替代这些条件。

## Edge Cases

空清单、重复 ID、缺文件、无 `flux`、非 RGB、非 uint8、空数组、非有限值、ZIP/NPY 读取错误均显式记录。单张 `[H,W,3]` 不解释为 24 视图；不生成伪视图，不静默回退到灰模。原始资产保持只读。

## Validation / Sanity Checks

核对实际路径和 NPZ 键；检查 train/validation ID 交集；查看抽样原图；确认报告没有将存在性计数写成有效配对数量。baseline 后续训练/validation 均明确使用照片。

## Implementation Constraints

单进程、Python 标准库 + NumPy/Pillow；只写本主题目录，不覆盖源图，不新增训练依赖、不启动 GPU。

## Deliverables

检查报告、原图预览和中文直观输入方案。正式训练配置、真实小批量验证与训练运行另属下一实施阶段，不把本次讨论称为已启动训练。
