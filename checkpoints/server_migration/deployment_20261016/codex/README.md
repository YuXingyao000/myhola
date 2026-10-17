# 目标 Codex 配置与验收（2026-10-17）

目标安装官方 Node.js v24.22.0、npm 11.19.1、Codex CLI 0.160.1；Node 下载已核对官方 SHA256。Codex CLI 与源机版本一致。

## 项目上下文

- 主仓 `.codex/config.toml` 设置 `gpt-6-astra`、`ultra` 和 `pragmatic`，`project_doc_max_bytes=65536` 容纳嵌套研究指南。
- 自定义数据研究技能随项目放在 `.agents/skills/spec-research-data-code`，不恢复已删除的旧实验技能。
- 目标 `~/.codex/config.toml` 信任本次四个项目工作树；没有复制其他项目聊天、SQLite 会话库、插件运行状态或全局缓存。
- Codex 登录信息通过独立 SSH 输入传入目标权限 0600 的认证文件，没有写入仓库或日志。`codex login status` 显示 ChatGPT 登录。

## 实际连接验证

只检查本地登录状态不足以确认服务可用。首次直接连接官方模型服务超时；目标使用源机现有、目标也可达的局域网代理后，真实 CLI 请求成功。

目标 `/etc/profile.d/hola-network.sh` 为登录 shell 配置现有代理；`/usr/local/bin/codex` 启动器也读取它，确保非交互 SSH 启动 Codex 时能够联网。已有显式代理变量优先，`HOLA_DISABLE_PROXY=1` 可跳过默认注入。代理服务是外部运行依赖；它停机或变更时应更新目标网络配置。迁移数据使用直接 SSH，不经该代理。

验收使用临时只读对话，禁止调用工具，仅要求根据已加载的项目指南回答三条规则；推理等级临时设为 low，项目默认 ultra 没有改变。实际答复为：

> 新实验记录放在 `checkpoints/`；FLUX 编辑图片称为“合成真实感图像”；后处理规则开发使用 validation 划分。

这同时验证账号可实际请求模型、项目配置可解析、项目指南进入新对话。验收使用 `--ephemeral`，没有创建需要迁移的历史聊天。运行日志保存在目标 `/mnt/d/migration/logs/codex-context-check-v2.*`，不提交认证文件或完整会话日志。

## 开启新对话

```bash
cd /mnt/d/python
codex
```

桌面端仍需在使用者自己的客户端添加目标 SSH 连接并打开 `/mnt/d/python`；服务器安装不会自动改变客户端的连接列表。本次没有将源机其他项目的聊天转移到目标。

依据：[官方 Codex CLI](https://learn.chatgpt.com/docs/codex/cli)、[认证说明](https://learn.chatgpt.com/docs/auth)、[项目配置](https://learn.chatgpt.com/docs/config-file/config-basic)、[项目指令](https://learn.chatgpt.com/docs/agent-configuration/agents-md)、[技能目录](https://learn.chatgpt.com/docs/build-skills)。
