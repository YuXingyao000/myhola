# 目标服务器项目级验证（2026-10-17）

验证目标是确认迁移后的新环境可以执行当前研究代码，不启动正式训练，也不依据此次结果判断模型质量。现有环境的依赖、8 张 RTX 5090、NCCL 和 PythonOCC 验证见相邻 `environment/`。

## 固定验证范围

- 仅使用原 DeepCAD **training** split 中的 `00000061`、`00000070`，核验清单成员和实际输入摘要；不读取 ABC 最终评估数据。
- 使用当前灰模训练配置、原 AE 与本地 DINO 权重、rotation 0 的 mean/std 缓存；完整 24 层 diffusion、batch 2、BF16 前向和反向，验证图像投影梯度以及冻结 DINO/AE 无梯度。
- 原 AE 在可信 checkpoint 下执行 batch 1 前向、反向和 cached mean 几何解码；检查输出有限、post/eval/OCC 入口可导入。
- 新 `generate` 环境完整加载 Q8 GGUF、CLIP/T5、VAE，使用 training 灰模图进行 256px、2 步推理，只判断权重与运行时是否贯通。
- 不更新 optimizer、不保存训练权重，不复制或加载源机仍在训练的 diffusion checkpoint。diffusion 验证使用随机初始化主干。

## 执行

在目标 `/mnt/d/python` 下，开启离线 Hugging Face 读取即可使用已迁移缓存：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=/mnt/d/python \
  /mnt/d/anaconda3/envs/torch/bin/python \
  checkpoints/server_migration/deployment_20261016/project_validation/validate_project.py training
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=/mnt/d/python \
  /mnt/d/anaconda3/envs/generate/bin/python \
  checkpoints/server_migration/deployment_20261016/project_validation/validate_project.py flux
```

运行时产物保存在目标 `/mnt/d/migration/project-validation/`。紧凑的 `training.json` / `flux.json` 结果复制回本目录后记录结论；生成图片留在运行目录，不作为研究数据或质量证据。

## 当前状态

验证脚本已准备，待主任务完成源码同步后执行。此前基础环境检查不等同于本页的项目级验证。
