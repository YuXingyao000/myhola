# 照片310000步：完整偏好数据

用户于2026-10-26授权重新生成照片偏好对，并同时覆盖完整validation。固定协议见[PLAN.md](PLAN.md)，输入检查见[INPUT_AUDIT.json](INPUT_AUDIT.json)，实现验收见[VALIDATION.json](VALIDATION.json)。这是合成真实感照片数据，后续DPO训练从当前照片baseline起步。

## 数据与规模

| 阶段 | CAD数 | 每图候选 | 候选总数 | 偏好清单 |
|---|---:|---:|---:|---|
| validation，先运行 | 3505 | 8 | 28040 | validation_pairs.jsonl |
| train，自动接续 | 46627 | 8 | 373016 | pairs.jsonl |
| 合计 | 50132 | 8 | 401056 | 各图至多一对，无pair也保留原因 |

训练与validation的CAD ID交集为0；两份清单分开，validation不进入训练manifest。即便validation没有合格偏好对也继续生成train，避免只完成部分请求或依据validation放宽标签规则。测试集不参与。

模型为照片 `step310000`，SHA256 `d8ba2f83fa6989ad235c021d2a41ca23d79505076a522954086597d575caa745`。原HoLa checkpoint提供冻结DINO/AE，其SHA256为 `9dcafb192529c1ddf65942392c25b409fbbf2bce407805f34a460d0b44bb4839`。只读 `real_photo.npz[flux]`，uint8[512,512,3] → PIL缩放224 → ImageNet归一化；不读灰模或masked照片。

候选保留原始30×32 latent，不重新排列或编码修复后的CAD。选择规则与旧实验相同：优先合法胜确认非法且raw几何未超过退步上限；否则从两个合法候选中按三轮post几何一致改善选对。CD退步上限2%、改善门槛5%，F1退步上限0.01、改善门槛0.02。unknown不当负例。几何测量失败但native检查已经完成时，保留已知有效性，缺失post指标不补零。

## 运行

本轮正式run：`/mnt/d/python/checkpoints/diffusion/photo_preference_data/20261026T055801Z`。启动后实时状态在run下 `status.json`，每阶段结束保存 `metadata/{validation,train}_summary.json`；全部完成后顶层status为complete且两份正式manifest均发布。运行实际启动核验另记[LAUNCH.json](LAUNCH.json)。

新运行命令（不要对正在运行的任务重复执行）：

```bash
bash /mnt/d/python/checkpoints/diffusion/photo_preference/RUN_GENERATION.sh \
  /mnt/d/python/checkpoints/diffusion/photo_preference_data/20261026T055801Z
```

中断后的显式恢复（会严格核对配置、源码、checkpoint和已有产物，不重抽中断候选）：

```bash
bash /mnt/d/python/checkpoints/diffusion/photo_preference/RUN_GENERATION.sh \
  /mnt/d/python/checkpoints/diffusion/photo_preference_data/20261026T055801Z \
  --resume
```

8张4090，每卡2个独立worker，CPU32、有界队列64，FP32/batch1、原生DDPM1000。两个同卡worker的真实完整采样已经对照已完成baseline c00：两件latent及全部raw字段逐位相同。实际32个候选的生成→OCC→选对完成，恢复后394个候选文件摘要和偏好标签不变。这些只验证实现，不是照片DPO效果证据。

已有同结构灰模双worker约2.76候选/秒的八卡短窗口仅作预算参考，40.1万候选约40小时；实际全量耗时受CPU后处理、I/O及零件复杂度影响，预计约两天量级。正式启动后会记录本轮实测速率。

本次不自动启动DPO训练。已有PreferenceDataset/PreferenceDiffusion训练入口仍需下一步显式适配照片数据协议与照片policy/reference初始化，不能直接套用灰模训练配置。

## 来源与版本

本轮改动放在 `data/preference_source.py`（显式条件/模型来源）、`preference_generate.py`（共享调度、输入错误保留分母）、`preference_io.py`（协议与源快照），照片模型加载只增加显式DINO repo参数。原模型结构、采样器、偏好阈值、VAE和OCC算法保持既有实现。

旧run保留不变；修改后的源码不能伪装成旧run的原始源码，严格resume会检测版本差异。新run记录完整源快照和Git dirty diff，覆盖工作区现有未提交依赖；仅提交本轮改动，不纳入其他DPO/几何研究工作。运行数据、checkpoint与大量候选均不提交Git。
