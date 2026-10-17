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

## 当前状态与完成 hook 交接

2026-10-17，按用户要求停止持续轮询，把后续检查交给传输完成 hook。

**已完成：**

- 真实 training loader batch 2，完整 24 层 diffusion 的 500/500 参数张量、图像投影的 6/6 张量均得到非零有限梯度；冻结 DINO/AE 无梯度。图像条件形状 `[2,1,257,1024]`，latent stats 为 `[2,30,64]`。
- 原 AE 可信 checkpoint 严格加载，training batch 1 的 704/704 参数张量得到非零有限梯度；cached mean 解码的面和边坐标均有限。
- FLUX Q8 GGUF + CLIP/T5/VAE 全部加载（119 秒），真实 training 条件图完成两步推理，输出 tensor 有限值断言通过。最终图片导出因验证脚本直接将 BF16 tensor 转 NumPy 失败；生产 `Img2BrepPipeline.__call__` 已有 uint8 转换，不是该生产路径故障。验证脚本现已按相同方式修复，未重跑。
- 主仓 scene_factory 的 base Python 3.13.9 入口帮助检查均通过；原 base 的 rich/Pillow/NumPy 保留，仅补充 `trimesh==5.1.1`（不安装或升级其他依赖）。原机和目标 base 均无 PythonOCC，STEP 回退使用 `torch` 环境。旧 data-generation 工作树的三个纯标准库核心模块导入通过。
- Blender 4.2/5.2 后台启动通过，5.2 实际识别 8 张 RTX 5090 的 OptiX 设备，未执行研究渲染。详见 [scene-base.json](scene-base.json) 和 [blender-optix.json](blender-optix.json)。
- 新 `torch` / `generate` 已分别用 `pip install --no-deps --no-build-isolation -e /mnt/d/python` 完成项目可编辑安装，之后两个环境的 `pip check` 均通过。安装没有更改已验证依赖版本。

**待完成 hook 接管：**

1. 确认环境 agent 的 torchvision bundled libjpeg SONAME 冲突修复已完成。首次训练检查在最后的 `post.construct_brep -> OCC.IGESControl` 导入失败；原因是 torchvision 自带 libjpeg 的 SONAME 与 Conda libjpeg 冲突，导致 libtiff 找不到 `jpeg12_write_raw_data`。修复前证据保留在 [training-before-occ-fix.json](training-before-occ-fix.json)，不得将其整体标为 passed。
2. 源码与 `.git` 同步完整后，按上文命令重跑 `training`（首次约 34 秒）。脚本还增加真实 `UnifiedTrainer._init_model_from_checkpoint` 的可信 AE 加载验证，以覆盖 PyTorch 2.8 的 `weights_only` 兼容修复。首次检查时 Git 元数据仍在传输，因此 JSON 明确记录 `git_metadata_ready=false`；重跑应记录实际 HEAD。
3. 按上文命令重跑 `flux`（首次总计约 196 秒），确认修复后的 uint8 导出产生 256px 图片及 `status=passed`。修复前证据保留在 [flux-before-export-fix.json](flux-before-export-fix.json)。该图片仅为环境验证，不纳入研究数据、不据此评估质量。
4. 将目标 `/mnt/d/migration/project-validation/training.json` 和 `flux.json` 复制回本目录，并据实际结果更新本页。GPU 验证进程已退出，没有后台正式训练。

所有首次报告均原样保留，没有把后续修复追写成当时已通过。检查未触碰 ABC 最终评估子集，没有移动源机仍在训练的 diffusion 权重，也没有改动源机 Python 环境。
