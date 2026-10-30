# 训练中断与回传记录

本记录补充启动时快照；已封存run文件保持不变。

训练于北京时间2026-10-17 17:24:10结束，退出码1，耗时26分35秒。错误为 `ValueError: face_latents must be finite.`，触发于 `src/brepnet/models/blocks.py:1525` 的decoder输入检查。可确认潜变量含NaN或Inf；日志没有区分类型，不能定位最早来源。未发现OOM。

最后控制台进度为epoch3的71/363，推算执行约1160 optimizer steps；最后完整checkpoint确认到step1089/epoch2，788项Adam状态均step1089且权重与optimizer有限。三次完整validation loss依次为1.330961、0.627364、0.406857；最佳与最后均为0.406857。保存3份best和last，共4份完整Lightning checkpoint。显存采样峰值31582MiB（30.84GiB），不是allocator精确峰值。

训练/监督进程已退出，8卡资源已释放。本次未改训练参数、未重启、未执行test或OCC评估；后续研究判断交回开发机。

2026-10-17 18:52:57（北京时间）已在开发机yuxingyao-187-187启动独立拉取进程PID2613744。SSH连接双向均已实测，不复制私钥。拉取仅按既定46项文件清单传输；计算节点先校验43项payload SHA256，开发机再逐文件核对并原子发布到 `/mnt/d/python/checkpoints/uv_face_encoder/candidate/20261017T085436.965155Z`。

北京时间2026-10-17 18:56:13，46个文件、8,747,818,374字节已回传；源端及接收端全部SHA256验证通过，状态为`returned_and_verified`。目标同名目录原先不存在，已从独立staging原子发布；计算节点所有权重保留。验收记录保存于开发机 `/mnt/d/python/checkpoints/uv_face_encoder/candidate/20261017T085436.965155Z.return_receipt.json`。
