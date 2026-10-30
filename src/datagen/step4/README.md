# Step 4：场景真实质感增强

普通 Python 批处理；输入 Step 3 的 `rgb.png` 和 `mask.png`，输出 512×512 增强图。
不调用 LLM、不需要 ComfyUI。每个场景一份固定英文提示词，模型加载一次、文本编码一次，
逐件从原图加有限噪声后去噪。支持本地 Kontext 或普通 FLUX（如 Krea）的 GGUF 权重，
分别通过 `--pipeline kontext/flux` 显式选择相应管线。

## 从哪里读

| 文件 | 职责 |
|---|---|
| `workflow.py → run_workflow()` | 冻结输入 → 加载模型 → 批处理 → 汇总 |
| `workflow.py → enhance_batch()` | 读取一件 → 构造 mask → 增强 → 合成 → 检查与保存 |
| `model.py → FluxEnhancer` | 加载 Kontext / 普通 FLUX、缓存提示词、执行带 mask 的图生图 |
| `masks.py` | 保护区外扩、编辑 mask、仅外侧羽化、原像素合成 |
| `report.py` | 保留失败位置的汇总及分页对照图 |
| `prompts/scenes.json` 与八份 `.md` | 实际场景 ID → 固定提示词 |
| `config.json` | 去噪强度、步数、引导、边缘处理、seed |

`run_workflow()` 会保存完整源码副本并重新执行该副本，长任务不受随后工作区修改影响。
不提供自动重试或覆盖；比较另一个模型/参数时用新目录。
`--run_request` 完全使用已冻结请求，不会应用同时传入的其他参数。

## 本机运行

输入必须来自支撑高度修正后的 Step 3。本轮 100 件仍是办公室一个场景，八份提示词
用于后续其他场景，不能把本轮结果描述为覆盖八个环境。

```bash
/mnt/d/anaconda3/envs/generate/bin/python \
  /mnt/d/python_room_generation/src/datagen/step4/workflow.py \
  --render_dir /mnt/d/data/cad_scene_renders/office_step3_100_support_fix_20261027_001 \
  --scene_id office_20261023_001 \
  --output_dir /mnt/d/data/cad_scene_enhancements/office_100_NEW_RUN \
  --model_dir /mnt/d/model/Flux1_Krea_dev \
  --transformer_file /mnt/d/model/Flux1_Krea_dev_GGUF/flux1-krea-dev-Q8_0.gguf \
  --pipeline flux --guidance_scale 4.5 --processing_size 512 \
  --strengths 0.25 --gpu_index 7
```

`--strengths 0.25 0.45 0.65` 对同一输入/seed 比较强度；`--limit 5` 显式限制输入数量。
当前默认采用 Krea Q8、原生512、strength=0.25，40时间表步数实际执行10步。
这是在办公室五件开发样本上选定的保守增强配置，视觉改善有限，其他七个场景的提示词仅已准备，
尚未验证效果。办公室100张全量技术通过，但固定抽查10张中发现2张背景道具变化/伪影，
本批是尚未验收的候选图，不直接视为合格训练数据；详见 [试验记录](../../../checkpoints/image_generation/codex_scene_factory/enhancement/NOTE.md)。

加 `--prepare_only` 只检查输入并保存请求，不初始化 CUDA。之后执行：

```bash
/mnt/d/anaconda3/envs/generate/bin/python \
  /mnt/d/python_room_generation/src/datagen/step4/workflow.py \
  --run_request /path/to/run/request.json
```

本机已存在所需环境与权重，程序只从本地读取。已验证环境为 Torch 2.6.0+cu124、
Diffusers 0.37.1、Transformers 5.5.3、Accelerate 1.13.0、GGUF 0.18.0。
其他依赖为 Pillow、NumPy、SciPy。实际版本、模型文件 SHA-256 保存到请求中。
GPU 通过 `CUDA_VISIBLE_DEVICES` 隔离，内部使用唯一可见的 `cuda:0`；不要并发启动其他
使用同一卡的任务。加载阶段用 CPU offload 编码提示词；默认随后释放文本编码器，
将 Transformer/VAE 常驻 GPU，减少逐件传输。配置 `resident_model=false` 可保留
逐模块 offload。确切峰值与耗时由真实运行记录给出。

## 参数与保护范围

- `strength` 决定使用采样时间表的哪一段；它不是固定的“改动百分比”。
  Kontext 还持续接收原图参考，普通 FLUX 只将原图作为 img2img 起点；
  两种管线相同 strength 不代表相同改动程度。
- `num_inference_steps` 是完整时间表步数，实际执行步数通常更少，记录为 `actual_steps`。
- 明确关闭 Kontext 自动调整尺寸；默认内部 512。`--processing_size 1024` 显式启用
  内部 1024 对照，RGB 上采样、mask 最近邻放大，生成后缩回 512 再合成原始保护区域。
  `generated.png` 在此模式下也是缩回 512 后的模型输出。实际内部尺寸写入记录。
- 原零件 mask 白色是零件；编辑 mask 白色是允许生成的背景。
- 保护区向外扩 2 像素；采样使用硬 mask。最终合成在保护区外侧 4 像素渐变。
- 合成覆盖原始保护区，要求该区域最大像素差为零。模型原始输出也保留，便于判断
  生成本身与最终合成的区别。羽化不进入零件内部。
- mask 不包括桌面上的接触阴影和零件倒影。提示词要求保留原照明，但这不是几何保证；
  人工仍须检查接缝、阴影、环境反射和背景物体是否变形。
- 全场景深度暂不参与模型控制，也不能视为增强后背景的准确深度标签。

## 输出

```text
run/
  request.json            样本、场景、输入摘要、提示词、模型摘要、版本、seed
  prompt.md               本次实际使用的完整提示词
  source/                执行源码与配置副本
  git.diff
  results.jsonl           每件/每档结果；失败留在分母
  summary.json            planned / success / failed / not_run；人工审阅状态
  comparisons/page_*.jpg  原图与增强图，最多十个输入一页
  <model_id>/strength_0_45/
    generated.png        合成前的512输出；内部1024模式下已经缩小
    enhanced.png         合成原零件后的最终图
    protected_mask.png
    edit_mask.png
    blend_mask.png
    metadata.json        像素检查、实际步数、耗时、显存、输出摘要
```

源 RGB/mask/CAD 来源/机位/材质在冻结请求中引用并保存摘要；原始 Step 3 目录保留。
输出属于**合成真实感图像**。技术成功只保证输入/尺寸/像素保护契约，不代表真实感提升。
背景平均变化量是描述指标，不是质量分数；人工判断与后续模型训练效果分别记录。

## 手机摄影质感探索（2026-10-27）

新增 [CPU自适应3DLUT修图](retouch/README.md)，以及标准Kontext整图编辑、Krea1024和
DepthControlNet的机制比较。实验入口、固定六件和全部候选见
[研究记录](../../../checkpoints/image_generation/codex_scene_factory/enhancement/phone_camera/NOTE.md)。
它们是独立研究候选，不更改本页旧100件的默认配置；目前没有获验收的手机摄影增强版本。
