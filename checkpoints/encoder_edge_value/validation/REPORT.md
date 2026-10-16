# Encoder 边内容消息：实现验证

日期：2026-10-15。仅证明实现与训练接口可用，不是几何质量或 solid 提升证据。

- 相关回归测试 **112 passed**，含新增25项：attention/输出零初始化等价、直接边值传递、self-loop mean、空边、排列等变、packed隔离、梯度、严格迁移和resume清理。
- 已评估 conditioned 源模型的 **736个旧tensor精确保留**；仅新增20个tensor、**51,680参数**，总参数182,017,661。原5层32维单头GAT与8层512维Intersection容量保留。
- 真实数据2件，FP32与BF16各核验74个输出/中间状态/loss tensor，和原GAT对照**最大绝对差0**。
- 真实数据BF16两步Adam：首步所有新末层有非零梯度，上游MLP首步零梯度、第二步非零；face/edge CNN、posterior、decoder均有有限梯度。没有冻结Encoder。
- 真实batch16：面数 `[7,14,14,12,20,14,13,13,18,7,14,26,18,18,19,8]`，完整前向/反向/Adam峰值分配显存 **7.363 GiB**。这是该批次测量，不能保证任意复杂度批次的显存上限。
- 统一8卡入口已完成2步、每卡16，有限输出且五个新末层均更新。有限validation只检查入口，不参与正式checkpoint选择。
- 随后8卡从step2完整恢复并更新至step3。组件初始化路径故意设为不存在，确认resume跳过迁移并恢复完整状态。首次恢复将max_epochs设为2，被短检查的epoch进度立即停止，未推进；提高epoch安全上限、保持max_steps=3后成功续训，见 [恢复记录](resume_check.json)。
- 独立审查另以double精度逐目标展开softmax、mean自环和求和，前向误差2.22e-16，输入梯度与finite-difference gradcheck通过。
- 顺序启动器使用无训练的子进程检查：完整两组、首组失败取消后组、停止取消后组、退出0但步数不足、来源变化拒绝后组、重复启动拒绝、过期PID不发信号均通过；checkpoint步数读取也使用CPU真实torch文件检查。

检查时采用控制CUDA数值模式核验初始化等价；正式两组沿用原trainer数值设置，不声称跨运行逐位可复现。小批次运行的loss涨跌不用于研究判断。

明细：[real-data result](result.json)、[测试](tests.json)、[8卡检查](ddp_check.json)。验证脚本不保存训练权重；统一入口检查的临时权重位于忽略目录，与正式运行分开。
