# UV Face Encoder 训练产物拉取

本脚本运行在**开发机**，通过已有 SSH 公钥连接计算节点。依赖 Python 3.10+ 标准库、`ssh`、`rsync`；目标开发机使用 Linux，以 `renameat2(RENAME_NOREPLACE)` 原子发布并保证不覆盖已有同名运行目录。计算节点使用已有 `sh`、`sha256sum`、`rsync`。脚本不复制认证文件、不修改 SSH 设置、不删除计算节点产物。

当前计算节点没有向开发机登录的私钥或转发 agent；已有部署记录验证的是开发机向 `root@172.31.224.142:12222` 登录。因此必须在开发机启动本脚本。仅在计算节点保存脚本，不等于已经部署了自动拉取。

## 启动

本轮正式 run 为 `20261017T085436.965155Z`。以下命令全部在**开发机**执行；先用已有 SSH 下载工具，之后独立后台等待并拉取：

```sh
ssh -o BatchMode=yes -p 12222 root@172.31.224.142 true

mkdir -p /mnt/d/python/checkpoints/uv_face_encoder/validation/server_preflight_20261017/return_tools
rsync -rt --ignore-existing --protect-args \
  -e 'ssh -o BatchMode=yes -p 12222' \
  root@172.31.224.142:/mnt/d/python/checkpoints/uv_face_encoder/validation/server_preflight_20261017/return_tools/ \
  /mnt/d/python/checkpoints/uv_face_encoder/validation/server_preflight_20261017/return_tools/

nohup python3 /mnt/d/python/checkpoints/uv_face_encoder/validation/server_preflight_20261017/return_tools/pull_run.py \
  --run-dir /mnt/d/python/checkpoints/uv_face_encoder/candidate/20261017T085436.965155Z \
  --destination-root /mnt/d/python/checkpoints/uv_face_encoder/candidate \
  > /tmp/uv-face-return-20261017T085436.965155Z.log 2>&1 < /dev/null &
echo $!
```

启动后记录 shell 返回的 PID。可对该 PID 发送 `SIGTERM` 或 `SIGINT` 中断等待或传输；临时文件保留。SSH 断开不影响 `nohup` 进程，但主机重启后需重新执行同一命令；脚本没有安装开机服务。

默认每 60 秒检查 `metadata/READY_FOR_RETURN.json`。网络尚未恢复时继续等待并记录错误；标记就绪后，清单错误、校验失败或 rsync 错误会以非零状态退出。`completed`、`failed`、`interrupted` 三种训练状态均可拉取；回传验收通过不代表训练正常完成。

## 计算节点交付协议

训练 supervisor 必须先停止写入交付内容、关闭日志、完成指标与结果导出，再按以下顺序发布：

1. `RETURN_FILES.txt`：每行一个规范化 UTF-8 相对文件路径，包含全部交付文件，且必须包含 `SHA256SUMS`、`RETURN_FILES.txt` 和 `metadata/READY_FOR_RETURN.json` 本身。
2. `SHA256SUMS`：使用 `sha256sum` 常规文本格式，**精确覆盖清单中除上述三个控制文件之外的所有文件**。三个控制文件不加入此摘要清单，以免形成循环摘要。
3. 最后原子写入 `metadata/READY_FOR_RETURN.json`，例如：

```json
{
  "status": "completed",
  "run_dir": "/mnt/d/python/checkpoints/uv_face_encoder/candidate/<UTC运行目录>",
  "manifest_sha256": "<SHA256SUMS 文件的 SHA256>",
  "files_sha256": "<RETURN_FILES.txt 文件的 SHA256>"
}
```

标记发布后全部交付文件保持不变。文件名不得含反斜杠、控制字符、空路径分段、`.` 或 `..` 分段；所有路径均须为相对路径，源端与接收端均拒绝软链接。清单不得包含数据集、环境目录或认证信息。

## 校验、失败恢复与结果

脚本先校验远程控制文件与 READY 的摘要一致，检查远程路径没有软链接，再执行远程 `sha256sum --check --strict SHA256SUMS`。随后只按 `RETURN_FILES.txt` 拉取到开发机目标根目录下的 `.<UTC运行目录>.return-staging/`，并核对文件集合、所有交付文件的 SHA256、两个清单及 READY 的完整内容。远程 READY 在重试间改变会直接报错，保留原 staging。

全部通过后原子发布为 `<UTC运行目录>/`，在同一父目录生成 `<UTC运行目录>.return_receipt.json`。验收记录包含两端 SHA256 校验结果、训练结束状态、文件数和字节数。验收记录由开发机生成，不包含在计算节点的原始清单内。最终目录或验收记录已经存在时脚本立即拒绝覆盖。

状态和重试命令保存在 `.<UTC运行目录>.return-state.json`；互斥锁为 `.<UTC运行目录>.return.lock`。失败时查看状态文件和启动日志，修复连接或可核实的传输问题后重新执行原命令，脚本会复用 staging。不要通过改写清单或删除远程权重掩盖校验错误。如果原子发布已成功但验收记录写入失败，状态会明确提示最终目录已存在，需人工检查；重新运行不会覆盖该目录。

只有脚本返回 0 且验收记录为 `returned_and_verified`，才能报告“已回传”。此工具开发时仅做本地临时文件验证，**没有执行实际回传，也没有在开发机启动守护进程**。

本地验收：`--help` 通过；17 项临时文件检查通过，覆盖路径逃逸、完整清单、损坏文件、额外文件、软链接、带空格/单引号/美元符号的路径，以及原子发布对已有空目录的拒绝。远程校验 shell 的逻辑只在本地临时目录执行过；这些检查不构成 SSH 传输或开发机可达性验收。
