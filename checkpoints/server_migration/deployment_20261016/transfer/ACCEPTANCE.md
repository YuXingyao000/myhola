# 资产迁移验收（进行中）

截至 **2026-10-17 12:00（Asia/Shanghai）**：35 组必要资产中 30 组 rsync 成功，ABC 原始 STEP 正在传输，后面还有 4 组 ABC 图像目录；没有失败组。此时 STEP 已传约 807 GB。**本次迁移尚未全部完成。** 动态状态以 `/mnt/d/migration-transfer-logs/status.json` 为准，Git 中的 [状态快照](status_snapshot.json) 只表示记录时刻。

DeepCAD 原始数据、条件、24-view features、visibility、近期研究资产、两套 AE、DINOv2、精选 FLUX 和 Blender 已完成对应传输组。源机的 diffusion 运行目录按用户要求暂不迁移，之后仍需单独同步稳定权重。

## 已执行的独立验收

[验收明细](asset_verification.json) 记录源端与目标端共 521 个路径检查，未发现差异或新增断链：

- 原清单中的 122 个保留 checkpoint 软链接均保留目标字符串，且在目标端可解析；DINOv2 的 2 个模型软链接也可解析。124 个链接最终指向的路径全部位于本次所选资产根目录内。
- 36 份保留的稳定研究 checkpoint 文件大小与当前源端一致；这部分是元数据检查，不是逐权重全文件哈希。
- 所有已完成组的显式根路径存在，包括 64 个开发样本的先行复制路径。
- 从原 training split 用固定 seed `20261017` 抽取 8 个 CAD；其原始几何文件、6 种条件文件、rotation 0/23 的 latent 与 visibility，共 104 份文件 SHA-256 一致。不据此宣称完整数据集逐文件校验通过，也未重新解释旋转语义。
- ABC AE 的完整 SHA-256 一致；DeepCAD AE 的既有完整哈希记录见 [ae_checksum.json](ae_checksum.json)。没有查看 ABC 原始几何内容、评估结果或失败样本。

原清单中的 82 个链接按既定边界排除：17 个全局 W&B 调试缓存链接、57 个浏览器 profile 链接、8 个暂缓迁移的 diffusion 目录链接。它们没有作为应迁移的科学资产进行补拷。原 visibility 缺少的 3 个 CAD 仍属于源端既有缺口，本次搬运没有生成替代数据。

## 为什么仍需要时间

[实时 I/O 采样](step_io_diagnosis.json) 显示，STEP 传输的一个 30 秒窗口中，源 `/mnt/d` 是机械盘，利用率约 98%，读速约 36 MiB/s，平均读取等待 12.8 ms。rsync 在 60 次进程采样中 49 次等待磁盘，CPU 只占一个核的约 17%；压缩后送入 SSH 的流量约 4.7 MiB/s。随后目标接收进程大部分时间等待输入。这个窗口的主要限制是源盘的小文件读取，不能把累计显示的约 50 MB/s 未压缩有效数据速率当作网络吞吐或局域网带宽上限。

已经完成的 24-view cache 只有约 4.23 GB 文件内容，却包含 126 万个文件及父目录，实际用了 1 小时 50 分钟。文件量和目录操作是此前“纯网络耗时”估计未覆盖的成本。空目标上的新文件没有现存基础文件可做差分，增加 `--whole-file` 不会绕过这些读取，也没有为此中断当前队列。

在上述记录时刻，按 STEP 剩余约 160–170 GB、后续图像约 250 GB 及已观测速率粗估，可能仍需约 **3–5 小时**。这不是完成承诺；盘上文件分布、其他读写与图像组的目录开销会改变耗时。不能使用 rsync 当前增量文件列表显示的 `99%` 代表整个 ABC 库已经完成 99%。

## 尚待确认

只有剩余 5 组都返回成功，队列总状态变为 `completed`，才能声明所选资产传输完成。之后更新状态快照并检查最终返回码与日志异常。代码、Git 工作树、新环境运行验证由主迁移流程验收；本报告仅覆盖资产，不替代它们，也不是研究质量评估。

轻量监视器 `summarize_progress.py --watch 60` 每分钟只读取队列状态和已完成日志末尾，不扫描数据目录。最新摘要在 `/mnt/d/migration-transfer-logs/verification_summary.json`，仅发生组切换、失败、状态停更或完成时追加 `/mnt/d/migration-transfer-logs/verification_events.jsonl`；它不发送外部消息，也不能自行唤醒对话。逐组实际逻辑传输量、文件数、耗时和 rsync 发送字节的定格记录见 [group_transfer_summary.json](group_transfer_summary.json)。
