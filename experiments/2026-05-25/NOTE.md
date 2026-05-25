# 2026-05-25 实验笔记

## 今日目标

修复 `real_photo_ratio` 配置失效 bug，确认 topology bias 在**真实照片**条件下是否有效（而非仅在白模上好看）。

## 必做

- [ ] **质量评估 (IoU)** — 确认 FLUX 数据确实存在且质量合格，否则后续训练无意义
- [ ] **Topo bias + 白模 (corrected)** — 用修正后的代码重跑，确认之前结果可复现（之前 config 没生效，实际可能混入了 20% 真实照片）

## 可选

- [ ] **Topo bias + 真实照片 (ratio=1.0)** — 核心实验，但前提是 FLUX 数据已生成完毕。如果 `single_view.npz` 不存在，会被 filter 掉导致无数据
- [ ] **Baseline + 真实照片 (ratio=1.0)** — 对照组，证明 topo bias 确实有帮助。和上面同一个前置条件
- [ ] **Blender 重渲染** — 相机参数已修（FOV=45°, scale=0.9），但如果不急着重新生成数据，可以后面再做
- [ ] **IoU + DINO + CLIP 全指标** — 有 GPU 空闲时跑，用于论文 dataset 章节的数量报告

## 依赖 / 前置条件

| 条件 | 状态 | 说明 |
|------|------|------|
| `deepcad_v6_cond/{model_id}/imgs.npz` | ✅ 应该存在 | OCC 白模/灰模渲染 |
| `deepcad_v6_cond/{model_id}/single_view.npz` | ❓ 需确认 | FLUX 生成图，ratio=1.0 实验的前提 |
| AE checkpoint `1119_deepcad_aug1_11k.ckpt` | ✅ | 已有 |
| Latent cache `ae_cache/1119_deepcad_aug1_11k/` | ✅ | 已有 |

## 预期结果

- **如果 FLUX IoU > 0.70 (大部分模型)**：数据质量 OK，可以直接用于 ratio=1.0 训练
- **如果 FLUX IoU < 0.50**：数据质量差，需要重新调 prompt 或加 quality filter 再训练
- **如果 topo bias (real) loss 比 baseline (real) 低**：证明拓扑感知在真实照片上有效 → 论文贡献成立
- **如果 topo bias (real) ≈ baseline (real)**：拓扑信息在真实照片场景下帮助不大，需要配合 intrinsic decomposition 一起用

## 今日修复的 Bug

1. `dataset.py` 的 `real_photo_ratio` 从未被代码读取（硬编码 0.2）→ 已修
2. Blender 相机 FOV 和归一化比例与 OCC 不一致 → 已修
3. `DataGenerationRefactored` 重命名为 `datagen`，Hydra 化配置
