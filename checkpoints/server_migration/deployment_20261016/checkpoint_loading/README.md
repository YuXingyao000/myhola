# 新环境 checkpoint 加载兼容检查

2026-10-17：为新机 PyTorch 2.8 显式指定当前主线的反序列化合同，不改变网络结构、state key 或严格匹配策略。

- `src/brepnet/train.py`：项目自有、可信 Lightning 快照含 OmegaConf 配置元数据，初始化时显式 `weights_only=False`，与 VAE 现有加载方式一致。
- `src/brepnet/diffusion/topology.py`：对应历史 toy writer 保存 tensor、primitive args 和已转为 Python float 的 metrics，显式 `weights_only=True`。
- `src/brepnet/diffusion/image_encoder.py`：Depth Anything V2 的纯 state dict 显式 `weights_only=True`。

[PyTorch 2.8 序列化说明](https://docs.pytorch.org/docs/2.8/notes/serialization.html#torch-load-with-weights-only-true)确认从 2.6 起默认使用 `weights_only=True`。本机 Lightning 2.5.0.post0 的 `lightning_fabric.utilities.cloud_io._load` 已显式默认 `weights_only=False`，此次未改 resume 逻辑或设置全局加载开关。

## 已完成验证与边界

源机 Python 3.10 / PyTorch 2.4.1，`CUDA_VISIBLE_DEVICES=''`，仅 CPU 临时合成快照：

1. 含 OmegaConf 元数据的 Lightning 快照在 restricted load 下报错，统一 trainer 的显式完整加载成功，参数一致。
2. diffusion 快照缺少参数时仍严格失败，没有放宽 key 匹配。
3. toy predictor 的 tensor + primitive checkpoint 经实际调用加载成功。
4. Depth Anything encoder 经模拟 backbone 的实际调用加载纯 state dict 成功。

另独立复跑数据入口的现有 14 项 CPU fixture，全部通过（12.31 秒）；覆盖 Hydra 配置、两个 worker 的固定／动态 prompt 和 ABC CLI 路径保护。未读取 ABC 实际几何，未执行 FLUX 推理，未改源机环境。新机 PyTorch 2.8 的实际权重与 GPU 验证由部署环境检查另行记录，以上不替代该项验收。
