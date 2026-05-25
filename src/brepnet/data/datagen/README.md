# Data Generation Pipeline

Real-photo CAD dataset 生成流水线。基于 FLUX.1-Kontext 将白模渲染转为逼真照片。

## Quick Start

```bash
# 从项目根目录运行 (cd /mnt/d/python 或 D:\myhola)

# 1. Blender 渲染 (24 视角, PBR 材质, FOV=45° 对齐 OCC)
python -m src.brepnet.data.datagen.run_blender

# 2. FLUX 生成 (动态 prompt, per-model 随机材质)
python -m src.brepnet.data.datagen.run_flux

# 3. 打包为训练用 npz
python -m src.brepnet.data.datagen.run_pack

# 全流程一键
python -m src.brepnet.data.datagen.run_pipeline
```

## Configuration

所有参数通过 Hydra config 管理 (`configs/datagen/`)，支持 CLI override：

```bash
# 覆盖路径
python -m src.brepnet.data.datagen.run_blender paths.model_root=/mnt/e/data

# 修改 GPU 数量
python -m src.brepnet.data.datagen.run_blender blender.num_gpus=4

# 切换 prompt 模式 (fixed = 固定 / dynamic = 随机材质)
python -m src.brepnet.data.datagen.run_flux datagen/prompt=fixed

# 修改 FLUX 参数
python -m src.brepnet.data.datagen.run_flux flux.num_steps=28 flux.guidance_scale=4.0

# 查看完整 resolved config
python -m src.brepnet.data.datagen.run_blender --cfg job
```

### Config 文件结构

```
configs/
├── datagen.yaml              # 主入口 (defaults composition)
└── datagen/
    ├── paths.yaml            # 数据路径、模型权重路径、输出目录
    ├── blender.yaml          # Blender: GPU数量、采样数、材质
    ├── flux.yaml             # FLUX: 步数、guidance scale、seed
    └── prompt/
        ├── fixed.yaml        # 固定 prompt (pre_encode_text=True, 快)
        └── dynamic.yaml      # 动态 prompt (per-model 随机, 多样性高)
```

## Pipeline 阶段

```
Stage 1: Blender Render
  mesh.stl → 24 视角 PBR 渲染图 (1024×1024, Metal010 材质)
  相机: eye=(2,2,2), FOV=45°, 归一化 scale=0.9 (对齐 OCC 渲染器)

Stage 2: FLUX.1-Kontext Generation
  白模渲染图 + prompt → 逼真照片 (保持形状, 改变材质/背景/光照)
  模型: Q8_0 GGUF 量化, num_steps=40, guidance_scale=3.5

Stage 3: Pack
  Blender + FLUX PNG → single_view.npz (训练时 dataset.py 读取)
```

## Prompt 模式

### Fixed (`datagen/prompt=fixed`)
- 所有模型使用相同 prompt
- `pre_encode_text=True`: text encoder 只跑一次，节省显存
- 适合大规模批量生成

### Dynamic (`datagen/prompt=dynamic`)
- 每个 model_id 基于 SHA256(seed:model_id) 确定性选择材质描述
- 从 materials × finishes × processes × conditions 组合
- `pre_encode_text=False`: 每张图重新编码 prompt
- 生成结果更多样

## 目录说明

```
datagen/
├── config.py          # 项目根目录检测 + 路径解析
├── compat.py          # 向后兼容 (run_lists.py 使用)
├── prompts.py         # Prompt 构建逻辑 (fixed / dynamic)
├── utils.py           # 共享工具: resolve_rank, load_model_ids, split_for_gpus
├── launcher.py        # 多进程启动器 (rank 并行)
├── run_blender.py     # [Hydra] Blender 渲染入口
├── run_flux.py        # [Hydra] FLUX 生成入口
├── run_pack.py        # [Hydra] NPZ 打包入口
├── run_pipeline.py    # [Hydra] 全流程编排
├── run_lists.py       # [Legacy] model list 采样
├── split_model_lists.py  # [Legacy] 多机器分割
├── mask_flux_with_blender.py  # 背景 mask (Blender 轮廓 → FLUX 输出)
├── external/generation/       # FLUX pipeline (DO NOT MODIFY)
│   ├── config.py              #   Img2BrepConfig
│   └── pipeline.py            #   Img2BrepPipeline (GGUF loading)
├── blender_scripts/           # Blender 无头渲染脚本
│   ├── render_cube24.py       #   24 视角 (cube rotation group)
│   └── render_single_view.py  #   单视角多材质
├── flux_scripts/              # FLUX worker 脚本 (被 run_flux.py 调度)
│   ├── generate_cube24_dynamic.py   # 动态 prompt 模式
│   └── generate_single_view.py      # 固定 prompt 模式
└── materials/                 # PBR 材质纹理 (Metal010 等)
```

## 注意事项

- **FLUX 接口脆弱**: `external/generation/pipeline.py` 中的 GGUF 加载、`in_channels=64` hack、推理参数组合均不可修改
- **相机对齐**: Blender 的 FOV=45°、normalize scale=0.9 已对齐 OCC 渲染器 (`extract_imgs.py`)
- **Model List**: 使用 `src/brepnet/data/list/` 下的共享列表，不再单独维护
