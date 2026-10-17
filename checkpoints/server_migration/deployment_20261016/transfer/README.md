# 必要资产传输队列

2026-10-16 按本次用户补充要求执行。目标为 `root@172.31.224.142:12222`，保留资产的原绝对路径。源机训练继续运行；本队列不迁移旧 Python 环境，也不迁移活跃的 `checkpoints/diffusion_layerwise_svr/`。

`manifest.json` 固定本次 35 组选择，来源是 `../../20261017/{data,weights,generation}.json`。代码及 Git 由主迁移流程处理，本队列排除两处 checkpoints 中已跟踪的文件以避免覆盖。生成历史目录中未受主仓 Git 管理的小脚本、split 和来源日志按原清单保留。

先迁 DeepCAD AE、DINOv2 和 training split 前 64 个开发样本（几何、完整条件和 rotation0 `features.npy`），使目标环境能够进行实现验证；随后迁 Blender、字体、精选 FLUX、ABC AE、必要历史重放输入和来源记录、完整 DeepCAD、近期研究资产，最后迁 ABC 大库。64 个样本列表见 `development_sample_ids.txt`，不是新的研究 split 或评估协议。

## 执行与恢复

仅运行一个 rsync，避免多路随机读取源 HDD。队列每 5 秒原子更新状态，数据读取完成后按 rsync 返回码记录结果。所有动态日志与状态都位于 `/mnt/d/migration-transfer-logs/`，不进入 Git。

```bash
# 当前进度（status.json 包含每组开始/结束时间、PID、进度、退出码与日志路径）
cat /mnt/d/migration-transfer-logs/status.json

# SSH master 必须可用；无认证信息写入脚本、清单或日志
ssh -S /tmp/hola-migration-ssh-%C -o BatchMode=yes -p 12222 root@172.31.224.142 true

# 首次启动或中断后继续；锁保证重复启动不会多开队列
nohup /mnt/d/anaconda3/bin/python /mnt/d/python/checkpoints/server_migration/deployment_20261016/transfer/run_transfer.py \
  </dev/null >>/mnt/d/migration-transfer-logs/queue.log 2>&1 &
```

如需暂停，对 `status.json` 的队列 `pid` 发送 `SIGTERM`。这只中断当前 rsync；临时文件保存在目标 `.rsync-partial/`，源文件和训练进程不受影响。之后重启同一脚本：跳过已完成组，重试未完成组，rsync 复用目标现有内容。网络断开会定时重试；原 SSH master 失效时需重新建立可用连接。源机重启后需要重新启动队列，未安装系统开机服务。

传输错误 10/12/30/35/255 持续每 30 秒重试；其他错误每组最多三次，然后留下 `failed` 状态并继续独立资产。最终 `needs_attention` 表示仍有失败项；只有全部组 rsync 成功才记为 `completed`。单组完成不等于已执行全量独立哈希核验；部署加载、稳定权重哈希与科学软链接闭包仍由验收步骤判断。

## 选择边界

- `rsync -aR` 保留权限、时间和软链接；不跟随软链接，不使用 `--delete`，不删除或移动源资产。
- 24-view cache 仅传 `features.npy` 及其父目录；同目录的历史 `data.npz` 不迁移。
- ABC 只传原始解包 STEP、`abc_v2_new_imgs` 和三个选中的解包条件目录，不传 `.tar` / `.tar.gz` 归档。
- 两处近期 checkpoints 排除清单中的 29 份临时权重，保留有效 validation split、报告、原始预测、后处理缓存及科学结果链接。
- 排除 `.git`、Python 缓存、浏览器 profile、认证文件，以及指向全局 W&B 调试缓存的链接。`server_migration/` 自身由主流程处理，避免复制正在变化的迁移状态。
- FLUX 仅传清单中的六个组件目录与五个独立文件；DINOv2 仅传模型目录，保留其 snapshots→blobs 软链接。

初始验证：AE 单文件 dry-run 成功；后台实际传输在前 20 秒已传约 725 MB，rsync 报告约 36 MB/s。该值仅代表初始大文件，不能用于承诺含数百万小文件的整体完成时间。

## 吞吐与压缩测量

两端 rsync 3.2.7 都支持 zstd。共享 SSH master 上的 256 MiB 零数据传输在资产队列同时运行时得到 18.56 MiB/s，master CPU 约占一个核的 11.4%；初始单资产约 37.8 MiB/s。这更符合共享链路吞吐约束，未见源端 SSH 加密占满一个核。探针细节见 `network_probe.json`。

仅 ABC `data_step` 启用 `--compress-choice=zstd --compress-level=1`。按文件系统枚举顺序取 64 个 STEP，只传输字节、不查看几何或评估结果：17.13 MB 输入，普通方式发送 17.15 MB、zstd-1 发送 2.41 MB（14.0%），分别用时 1.29 秒和 0.44 秒；临时目标已清理，源文件未动。见 `step_compression_probe.json`。这是小规模便利样本，不能保证全库同样比例；PNG、NPZ、模型权重不压缩。

以首批容量和初始速率估算，不压缩的纯网络耗时约 12–13 小时；若 ABC 压缩比例接近该样本，纯网络耗时可能约 5–6 小时。源盘顺序读取、数百万小文件的目录遍历、共享链路与实际压缩率均会影响总时长，因此这些是传输量参考，不是完成时间承诺。

启动中的旧 worker 会在当前组自然完成后切换到支持压缩的新版本；新版本在每组开始前重读该组清单设置，不需要中断正在传输的资产。动态状态的 `runner_version=20261016-zstd1` 表示新版本已经生效。

AE 源端和目标 SHA-256 一致，见 `ae_checksum.json`。其他已完成组的 rsync 返回码、后续状态及错误应以运行时 JSON 为准。
