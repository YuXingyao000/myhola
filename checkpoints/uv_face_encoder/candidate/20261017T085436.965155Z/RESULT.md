# 服务器运行记录

状态：failed；退出码：1。
耗时：1595.5 秒；checkpoint 最大 step：1089。
最佳 validation loss：0.40685710310935974；最后 validation loss：0.40685710310935974。
显存为 nvidia-smi 每秒采样的峰值（MiB）：{'0': 26478.0, '1': 23100.0, '2': 31058.0, '3': 30440.0, '4': 28960.0, '5': 24182.0, '6': 31582.0, '7': 29352.0}。不是 CUDA allocator 精确峰值。

仅训练与 validation；未运行 test 或 OCC 合法率评估。完整配置、命令和 Git 信息见 metadata/invocations。
回传方式：开发机通过已有 SSH 通路主动拉取并校验；READY 标记只表示可传，不表示已回传。
计算节点保留所有权重，开发机验收前不删除。

日志测速区间：step 49 → 1149，1472.763 秒；0.747 steps/s，95.603 samples/s（全局 batch 128）。
测速按 TensorBoard wall_time 差计算；跳过日志 step 0，区间内若含 validation／权重保存，其耗时也计入。
