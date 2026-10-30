# 服务器运行记录

状态：interrupted；退出码：1。
耗时：54422.1 秒；checkpoint 最大 step：40656。
最佳 validation loss：0.05018790438771248；最后 validation loss：0.05741097405552864。
显存为 nvidia-smi 每秒采样的峰值（MiB）：{'0': 29356.0, '1': 28880.0, '2': 32098.0, '3': 31460.0, '4': 31364.0, '5': 30988.0, '6': 30292.0, '7': 29400.0}。不是 CUDA allocator 精确峰值。

仅训练与 validation；未运行 test 或 OCC 合法率评估。完整配置、命令和 Git 信息见 metadata/invocations。
回传方式：开发机通过已有 SSH 通路主动拉取并校验；READY 标记只表示可传，不表示已回传。
计算节点保留所有权重，开发机验收前不删除。

日志测速区间：step 1099 → 40899，54333.380 秒；0.733 steps/s，93.762 samples/s（全局 batch 128）。
测速按 TensorBoard wall_time 差计算；跳过日志 step 0，区间内若含 validation／权重保存，其耗时也计入。
