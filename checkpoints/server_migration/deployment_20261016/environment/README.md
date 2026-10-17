# 新服务器环境部署（2026-10-16）

本次为新 Ubuntu 22.04 / RTX 5090 服务器重新安装环境，没有复制源服务器的 Conda prefix、CUDA 扩展或 Python site-packages。源服务器训练与环境没有改动。

## 版本选择

- Anaconda Distribution 2025.12-2（Conda 25.11.1）安装到 `/mnt/d/anaconda3`，关闭自动进入 base，初始化 Bash/Zsh，并提供 `/usr/local/bin/conda`。默认 channel 为 conda-forge、strict priority。
- 研究环境名保留 `torch`；生成环境 `generate` 从这次新建的 `torch` 环境复制，再独立使用。
- Python 3.10.22；PyTorch 2.8.0 + CUDA 12.8、torchvision 0.23.0、torchaudio 2.8.0。官方 PyTorch 自 2.7 起提供 Blackwell 支持，2.8/cu128 又有对应 PyG 扩展轮子，不按 `nvidia-smi` 显示的驱动 CUDA 上限选择包。
- PythonOCC 保留源机实际组合：`pythonocc-core==7.8.1.1`、`occt==7.8.1`。用户所指 7.8.1 API 对应这个组合，两项写入 Conda pin。
- 保留 PyG 2.6.1 与 Lightning 2.5.0.post0 的 API 代际；scatter/sparse 等扩展使用针对 PyTorch 2.8/cu128 的官方 PyG wheel。
- NumPy 1.26.4 遵守仓库 `<2.0` 声明。FLUX 使用 diffusers 0.37.1、transformers 4.57.6，并显式补充 GGUF、SentencePiece、Accelerate 与 Hydra。
- 系统原有 CUDA 13.2 不卸载；新增 CUDA 12.8 的 nvcc/headers 用于编译匹配 PyTorch 的 CUDA 扩展，驱动不变。Conda 激活设置 `CUDA_HOME=/usr/local/cuda-12.8`、`TORCH_CUDA_ARCH_LIST=12.0`。

## 安装和验证入口

```bash
bash install_environment.sh
bash configure_conda.sh
bash install_extensions.sh
/mnt/d/anaconda3/envs/torch/bin/python validate_environment.py
OMP_NUM_THREADS=1 /mnt/d/anaconda3/envs/torch/bin/torchrun --standalone --nproc-per-node=8 validate_nccl.py
```

脚本由此次部署任务传入目标机执行，软件日志持久保存到 `/mnt/d/migration/logs/`。`install_environment.sh` 验证 Anaconda 安装包 SHA256；本次下载 torch 核心 wheel 后也对照官方 wheel 索引 SHA256 核验，再经 SSH 传到目标机。下载文件与新环境不是旧环境迁移。

目标首次访问 Ubuntu 源较慢，将 Ubuntu apt 地址切到阿里镜像，仍使用 Ubuntu 原始签名链；原配置保存在 `/mnt/d/migration/logs/apt-sources.original.list`。Anaconda、Conda、PyTorch、PyG 包的来源仍为各自官方仓库；普通 PyPI 依赖通过阿里 PyPI 镜像下载，以缓解目标直连大 wheel 速度低的问题。PyTorch、torchvision、torchaudio 三个 wheel 都核对了官方索引 SHA256。

日常使用：

```bash
source /mnt/d/anaconda3/etc/profile.d/conda.sh
conda activate torch
cd /mnt/d/python
```

本目录验证仅检查依赖、GPU 运算、拓扑库和入口，不运行正式训练，不读取 ABC 最终评估子集。环境部署与验证已完成。详细结果见 [environment-validation.json](environment-validation.json) 和 [nccl-validation.json](nccl-validation.json)。

- 8 张 RTX 5090 均通过 FP32 反向、BF16 矩阵乘法、scatter mean、sparse matmul、GATv2 反向；PyTorch wheel 原生包含 `sm_120`。
- 本机编译 `chamferdist==1.0.3`，CUDA 距离与反向通过；在 `torch` 和 `generate` 均安装。
- PythonOCC box 构造、STEP 写入/读回及 BRepCheck 合法性检查通过。
- FluxKontextPipeline、FluxTransformer2DModel、T5/CLIP/DINO、Hydra、GGUF、SentencePiece 实际导入通过；Q8 的 CUDA 反量化检查通过，最大绝对误差 0.003933。
- 8 卡 NCCL all-reduce 结果均为 36；两个环境 `pip check` 均通过。`generate` 的独立 prefix、Torch/CUDA、OCC 与 FLUX 导入也已检查。
- 完整包版本见 `torch-conda-explicit.txt`、`torch-pip-freeze.txt` 及对应 `generate-*` 文件；本机编译 wheel 和下载缓存留在目标 `/mnt/d/migration/`，不纳入 Git。

`conda run -n torch` 以及 `conda run -n generate` 可用于非交互命令。激活环境时 CUDA 编译路径固定到 12.8；退出时恢复之前的 CUDA 环境变量。

### Checkpoint 兼容性边界

[实际 AE 加载检查](checkpoint-compatibility.json) 显示：原 AE 在 PyTorch 2.8 默认 `torch.load` 下因 OmegaConf DictConfig 被拒绝；显式 `weights_only=False` 可正常读取 704 个 state entries。项目内部可信 checkpoint 的 `torch.load` 调用需要明确加载策略，不设置全局强制关闭 weights-only 的环境变量。已检查 Lightning 2.5.0.post0 的标准 resume 路径，其内部 `_load` 已显式使用 `weights_only=False`。

本报告交付时目标研究源码还在准备上传，因此未宣称仓库入口、完整 FLUX 权重加载或实际研究批次已验证；这些检查需在源码/模型到齐后进行。没有启动正式训练或评估 ABC 最终子集。

## 官方依据

- [PyTorch 2.7 Blackwell / CUDA 12.8 支持](https://pytorch.org/blog/pytorch-2-7/)
- [PyTorch 历史版本安装组合](https://docs.pytorch.org/get-started/previous-versions/)
- [PyG 2.7 安装说明及 PyTorch 2.8/cu128 矩阵](https://pytorch-geometric.readthedocs.io/en/2.7.0/install/installation.html)
- [NVIDIA Blackwell 兼容性指南](https://docs.nvidia.com/cuda/blackwell-compatibility-guide/)
- [Anaconda 安装包与 SHA256](https://repo.anaconda.com/archive/)
- [chamferdist 源码](https://github.com/krrish94/chamferdist)
