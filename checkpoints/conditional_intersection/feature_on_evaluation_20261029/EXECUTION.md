# 执行记录

本次保留 SPEC 中冻结的指标、输入、checkpoint选择和后处理算法。实际资源调整：best415 在142物理GPU7，last5395 在物理GPU6，并行各自单CAD导出；两者完成即释放GPU给新训练。187后处理每组32 workers，分组导出完成后各自启动，允许重叠，总64 workers，每worker单线程。启动前187有128逻辑CPU，load约16，可用内存约942GiB。

每件整个子进程180秒墙钟预算（包括几何测量）不变。并发资源变化不用于速度优劣结论，也不修改算法容差、尝试次数或合法性标准。指标与生产后处理源码均与历史协议一致。

| 权重 | step / epoch | SHA256 |
|---|---|---|
| 本轮validation total-loss best | 415 / 0 | ff2df58d6ddbe91cdf3cc46463abd3c7ec461bb1a2ea218d1c511412784bdf6b |
| 停训前最后完整保存快照 | 5395 / 12 | ac27e75060b4435a9cab3d3ba13ec9c50546cce5a7e2e5cf4e30e4b7a935de52 |
| 历史91%来源 | 42471 / 116 | ce749416c1635dda1563161b971bd4260d064cfdb0c31f1fa9d06ad0e70961ff |

last取不可变的 stopped-last-complete-epoch=0012-step=00005395.ckpt，而非仍可能被写入的last路径。两种feature开启权重均不按OCC结果挑选或再训练。

实际存储修复：187的/mnt/d发生ext4 journal阻塞，两次初始OCC停在依赖预加载，均0 CAD尝试。后续将17477个输入/历史文件（2.57GB）、原Python环境109594个普通文件（12.15GB）及734个源码文件（152.98MB）逐SHA复制至`/root/hola_eval/conditional_feature_on_20261029/`。未重新安装环境、未改变任何生产post文件；通过预加载核验后才派发CAD任务。原环境和源码保留，未修改挂载选项。实际执行脚本已同步到本目录；完整环境与大产物保留根盘，不进入Git。后处理逻辑data_root仍标记原数据位置，scheduler仅将GT读取物理路径映射至逐SHA相同的根盘副本。
