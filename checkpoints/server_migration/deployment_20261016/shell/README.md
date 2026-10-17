# 新服务器命令行与字体（2026-10-17）

用户已指定 Oh My Zsh、命令建议、语法高亮和 Powerlevel10k；具体命令行配色和布局尚未指定。本次只建立可用的基础配置，不把临时外观视为用户最终选择。

## 安装结果

目标 `172.31.224.142:12222` 的 root 登录 shell 已从 Bash 切换为 `/usr/bin/zsh`（5.8.1）。Oh My Zsh 安装于 `/root/.oh-my-zsh`，启用内置 `git` 及两个额外插件：

- `zsh-autosuggestions`：根据历史输入提供命令建议。
- `zsh-syntax-highlighting`：输入命令时高亮语法。

主题为 `powerlevel10k/powerlevel10k`，配置在 `/root/.p10k.zsh`。暂用 ASCII 模式和基本的主机/目录/Git/环境信息，不依赖 Nerd Font 图标，不自动弹出配置向导；后续可按用户审美修改，也可手动运行 `p10k configure`。Oh My Zsh 自动更新关闭，避免登录时改动或交互提示；可按需手动更新。

保留原 `.zshrc` 的 Conda 初始化块，新增独立标记的 shell 配置块。目标网络代理在交互 zsh 中沿用原部署设置，Codex、Node/npm 路径保持可用。原配置备份在 `/mnt/d/migration/backups/shell-20261017T041819Z/`，其中保存原登录 shell。新设置对下次 SSH 登录生效，现有会话可用 `exec zsh`。

安装仓库的精确提交、字体来源/哈希和备份路径见 [installation.json](installation.json)。只修改目标服务器；本机 shell 不变。

## 字体分工

| 用途 | 用户偏好 | 容器内已安装 |
|---|---|---|
| 标题、UI | Jost | 正体/斜体可变字体 |
| 正文、内容 | Archivo | 正体/斜体可变字体 |
| 中文 | 思源黑体 | Source Han Sans SC Regular / Bold |

字体及许可证放在 `/usr/local/share/fonts/hola/`，Fontconfig 已识别三族，供容器生成网页、图片、报告时显式选择。没有强制修改全局 sans-serif/monospace 回退或现有报告的字体路径。

SSH 终端中的字形由客户端字体决定，容器安装字体不会改变本机终端外观。上述拉丁字体为比例字体，不默认用作代码终端的等宽字体；若后续需要 P10k 图标，需在本机终端选择具备相应字形的字体。用户本地的终端字体、字号、配色和排版留待后续指定。

## 验证

- zsh 配置语法检查通过；交互登录环境中 Powerlevel10k、建议、高亮插件均已加载。已实际通过 SSH 分配终端启动登录 shell，P10k 的 gitstatusd 初始化成功并显示正常提示符。
- `conda activate torch` 后 Python 正确指向 `/mnt/d/anaconda3/envs/torch/bin/python`，Codex 为 `/usr/local/bin/codex`，代理变量可用。
- 非交互 SSH 只输出指定内容，无主题/插件噪声，保持 rsync 等命令的输入输出协议。
- 字体文件对照官方仓库 Git blob SHA1 验证，并额外记录 SHA256；Fontconfig 能匹配 Archivo、Jost 和 Source Han Sans SC。

来源：[Oh My Zsh](https://github.com/ohmyzsh/ohmyzsh)、[Powerlevel10k](https://github.com/romkatv/powerlevel10k)、[命令建议](https://github.com/zsh-users/zsh-autosuggestions)、[语法高亮](https://github.com/zsh-users/zsh-syntax-highlighting)、[Archivo](https://github.com/google/fonts/tree/main/ofl/archivo)、[Jost](https://github.com/google/fonts/tree/main/ofl/jost)、[思源黑体](https://github.com/adobe-fonts/source-han-sans)。
