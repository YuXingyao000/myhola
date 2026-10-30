# Diffusion 两条对话的上下文与运行核对

2026-10-25，北京时间。用户确认此次问题是“DPO 加／不加几何辅助 loss”，随后要求补齐另一条对话的历史，检查是否因上下文混淆跑错程序。

已通过 `read_thread` 分页读取 [Diffusion Branch (2)](thread://01a14582-ef44-7802-9218-3ed599112aa8?hostId=remote-ssh-discovered%3AVCC187) 的关键讨论、训练和评估历史。聊天记录用于恢复意图；下列运行事实同时核对磁盘协议、训练日志、加载记录及结果文件。本次没有修改模型、配置、权重或运行中的评估，没有启动新训练或重复评估。

## 结论与需纠正的语境

没有发现此次几何对照加载了错误架构、错误 checkpoint、错误偏好数据或错误 VAE 的证据。此前“74%→54%”属于10月17日逐层图像条件架构对照，不能解释10月25日几何辅助 loss 的效果。

“几何条件”在本轮确切指训练辅助监督：原去噪损失＋DPO＋0.0226×几何损失。它没有在推理时增加GT几何输入，也没有改变主干结构。

## 实验关系

| 实验 | 比较对象 | 已有结果与边界 |
|---|---|---|
| 10月17日单图架构对照 | 旧HoLa 1024维concat，与768维逐层图像cross-attention | 固定100件，合法solid为74与54。训练量、编码与增强也不同，不是本轮几何loss消融。 |
| 10月18–19日无条件实验 | 旧1024模型与新768模型 | 独立无条件生成实验，不是后续DPO的初始化来源。 |
| 10月20–23日原DPO | 原HoLa SVR初始化，旧偏好标签，训练5000步 | 完整3505件，原SVR2724件合法（77.72%），DPO3093件合法（88.25%）。 |
| 10月24日观测标签DPO | 同原SVR初始化，改偏好标签，训练5000步 | 独立完整评估已经完成。旧标签DPO3093/3505，新标签DPO3083/3505（87.96%）；有效率差值95%区间跨零。两者均未加入几何辅助loss。 |
| 10月25日几何loss对照 | 从旧标签DPO5000初始化，两组各额外5000步，一组加几何loss | 3505件/组；仍在后处理评估。比较的是继续DPO与DPO＋几何项。 |

10月24日部分说明仍写“尚未评估”，属于未更新的历史状态；实际以 `preference_eval/observation_v1_evaluation/status.json` 及报告为准。不能拿它的结果冒充几何辅助loss结果。

## 训练来源核对

几何实验两组共用初始权重：

`checkpoints/diffusion_preference_svr/20261023T030527Z/weights/steps/step=00005000.ckpt`

SHA256：`57754104ccfc92f3b43f991ce43bccda94d0e3ed86d11ec99b2f81793da5caee`。

| 项目 | 继续DPO | DPO＋几何项 |
|---|---|---|
| run | `diffusion_preference_continue/20261025T054547.727170Z` | `diffusion_preference_geometry/20261025T062543.945628Z` |
| 选定权重 | `weights/steps/step=00005000.ckpt` | `weights/steps/step=00005000.ckpt` |
| 权重SHA256 | `0c42e7a522e8fe8d3e1a5b838ed70a3709f7870b4285aea00e57a21007a1c0a1` | `6765b42a00743a2990fe4a406a823f39f196269e35accdbfea5f9f367b09f870` |
| 几何开关／权重 | false／0 | true／0.0226 |
| 新run global_step／Adam step | 5000／5000 | 5000／5000 |
| 恢复旧优化器 | 否 | 否 |

两组保存的命令、配置和训练日志均指向 `PreferenceDiffusion`／`LegacySVRDenoiser`：1024维、24层concat架构。没有使用768维逐层条件模型。

两组使用同一旧42037对训练清单，SHA256为 `454cd972b6acb2e036cf7e7d0b0446116067e33b28354b53e29d2fb1705b1911`；没有切换为observation重标注数据。每组均从旧DPO5000初始化、创建新Adam并额外训练5000步，不能将新run的step5000误读为累计只训练5000步。

训练预算一致：8卡×batch8、FP32、seed42、Adam lr=1e-5。等步数不等于等计算成本。

实际几何训练W&B汇总记录有非零几何项：`geometry_loss=0.02347729`、`geometry_weighted_loss=0.000530587`、有效噪声阶段比例约`0.497429`；控制组未启用此项。加权比例核对为0.0226000075，三项相加与总loss仅相差约6.2×10⁻⁸。这些是训练诊断，不是重建质量指标。最后一次W&B验证在step4368，不是step5000的独立重建评估。

## 评估实际加载核对

输出：`checkpoints/diffusion/preference_eval/geometry_v1_evaluation/`。

- GPU0/2/4/6的实际worker加载继续DPO checkpoint；GPU1/3/5/7加载几何checkpoint。
- 8个worker的strict load、model config equality、reference/DINO与原模型一致性记录均通过；几何组冻结decoder逐tensor等于原SVR内置VAE。
- 原SVR为 `/mnt/d/ckpt/HoLa/Diffusion_svr_sq30_1500k.ckpt`，SHA256 `9dcafb192529c1ddf65942392c25b409fbbf2bce407805f34a460d0b44bb4839`。没有接入后来UV／shared-boundary等VAE。
- worker元数据顶层`global_step=1500000`表示原SVR底座；真正本轮policy的路径和步数位于嵌套`preference`字段。顶层字段不表示评估忽略了微调权重。
- 内部目录`baseline`＝继续DPO；`dpo_5000`＝DPO＋几何项。名称沿用工具槽位，报告映射明确，不能据目录名解释为原SVR与旧DPO5000。
- 两组各3505件生成全部成功。本轮从头生成两组，没有复用旧SVR/DPO评估输出。冻结评估源码摘要与检查时源码一致。
- 固定validation、view0、K=1、seed42、DDPM1000、FP32、batch1、无补抽；原VAE和recover_v1后处理一致，原生有效性检查一致。train/validation CAD identity交集为0。
- 后处理几何三轮是同一STEP的三次采样测量，不是三次生成／修复后选最好的一次。

证据：[评估协议及严格检查](EVALUATION_PREFLIGHT.json)、[启动与两组实际输出核查](EVALUATION_LAUNCH.json)、[梯度实现验收](CPU_VALIDATION.json)。本次还读了实际run metadata和训练日志，没有重做全部权重计算或重新训练。

## 本对话刚才的快照是什么

北京时间18:59:25–18:59:26读取的共同已完成集合为2283件，完整结果尚未生成：

| 指标 | 继续DPO | DPO＋几何项 |
|---|---:|---:|
| 合法solid | 2043/2283（89.49%） | 2035/2283（89.14%） |
| 后处理前面平方Chamfer ↓ | 6.381×10⁻³ | 6.429×10⁻³ |
| 后处理前边平方Chamfer ↓ | 1.470×10⁻² | 1.487×10⁻² |
| 面匹配F1 ↑ | 90.32% | 90.37% |

47件从合法变非法，39件从非法变合法；有效率差为−0.350个百分点，配对95%区间[−1.139，+0.438]。8项配对统计均经冻结快照独立复算，全部区间跨零。Post共同合法且三轮几何测量完整的集合为1901件，单列于[快照报告](interim_comparison_20261025/README.md)。

快照保留全部3505件状态，pending没有算成失败。完成顺序可能偏向较快对象，不能把它当作最终验证集结论；bootstrap区间不消除这种选择偏差。目前只能说“这份快照尚未显示几何项的明确收益”，不能说上一版已被证明更好。

本次检查过程中主评估仍正常推进，最新一次读取为5489/7010个任务完成，观测阶段尚未启动。后续以运行状态和最终报告为准，不以本文动态进度作为完成凭证。

## 后续解释边界

当前证据排除了已检查的路线、权重、数据和方法标签混用；不等于证明所有历史程序完全没有bug。几何项通过冻结decoder回传到policy的实现已验收，但并不直接监督推理时的连接分类、闭合边界或OCC实体合法性。是否值得保留，仍需完整3505件结果和可见一致性结果，不能仅依据训练loss或中途均值决定。

另有一个明确的溯源记录缺口：训练启动时`preference_geometry.py`尚未受Git跟踪，因此没有包含在run保存的`git.diff`中。其当前内容及另外五个实现文件的SHA256均与训练前`CPU_VALIDATION.json`一致；这提供了额外核对证据，但不能把Git commit／diff单独描述成完整训练源码快照。此次没有重载GB级checkpoint或逐步记录正式GPU梯度；权重／Adam的结论依据已保存严格preflight与运行记录交叉核验。
