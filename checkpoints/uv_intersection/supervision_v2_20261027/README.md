# UV监督第二阶段

用户授权新增曲线监督并从上一轮继续全量训练。**已按用户要求于2026-10-28 15:54（北京时间）在142停训；固定51,460步完整checkpoint完成3495件validation评估。** 曲线CD²较UV30k降低70.09%，仍为旧XYZ64的6.84倍；面CD²较UV30k退步10.89%。完整结果见 [评估报告](evaluation_20261028/results_v1/REPORT.md)，判断与边界见 [本次研究记录](evaluation_20261028/NOTE.md)。启动信息保留于 [LAUNCH](LAUNCH.json)，曲线见 [W&B](https://wandb.ai/whatseven777/hola-brep/runs/c41fedf5a922)。

用户随后决定将后续研究基线恢复为 [32维显式共享XYZ版本](../../vae_baseline/xyz32_20261028/README.md)。本实验保持停止并保留全部资产；没有启动新的32维训练。

| 项目 | 状态 |
|---|---|
| 网络/固定曲面 | 保持30k UV模型全部结构和参数形状，170,699,647参数全量训练 |
| 新loss | 已实现直接UV、XYZ差分走向、GT尺度相对XYZ；绝对XYZ改L1 |
| 局部验证 | loss/网络26项、dataset10项、adapter11项通过；真实两件与两卡完整resume通过 |
| GT UV缓存 | 49,975件全部生成并逐文件审计通过，UV有效点99.94075%，无样本丢弃 |
| 正式训练 | 已停；固定保存步数30000→51460，增加21460次更新，保留完整模型/Adam |
| 下游评估 | 固定validation 3495/3495成功；三方几何/拓扑比较完成；OCC合法率未评估 |

入口配置：`configs/train_vae_uv_intersection_supervision_v2.yaml`。实际系数见 [架构与目标](ARCHITECTURE.md)，标签协议见 [SPEC](SPEC.md)，研究判断见 [PLAN](PLAN.md)。

源checkpoint：`/mnt/d/python/checkpoints/uv_intersection/joint64/20261025T144428Z/weights/last.ckpt`，step30000、epoch72、SHA256 `c4f409b1a904d431cf82cf4754001ca590b59dc44abbaf987181819977a32318`。已实际读取：Adam有617个参数状态，全部step30000、lr2e-5。

续训使用显式`trainer.resume_from_checkpoint`保留完整模型、Adam及global_step；`trainer.run_dir`指向原run的`stages/supervision_v2_<UTC>/`，追加30k至60000。新阶段独立W&B与best分数；旧run/权重不覆盖。现有run管理支持该方式，无需修改通用trainer。阶段溯源记录源路径、摘要和起止步数。

数据保持train46480、validation3495的原accepted清单、原增强、7卡×每卡2件×累积8（112件/更新）。正式生成规则只用固定validation前32件验收；test不参与。UV失效mask只屏蔽UV项，不丢CAD/边，其XYZ、切向和拓扑监督仍保留。

W&B中的`curve_xyz`继续表示旧MSE诊断；本轮绝对目标名为`curve_xyz_l1`。所有实际loss有`weighted_`贡献，避免原始值与加权值混淆。新增face_normal只拆出诊断，不重复加权；KL只加一次。

验收记录：[标签全量覆盖](LABEL_COVERAGE.json)、[32件真实投影](LABEL_PREFLIGHT_32.json)、[合成几何](LABEL_SYNTHETIC.json)、[独立数学审计](PROJECTION_AUDIT.json)、[真实模型/Adam检查](REAL_BATCH_VALIDATION.json)、[两卡Lightning续训](DDP_VALIDATION.json)。原始30k权重全部逐tensor保留；两卡检查的617份Adam状态从30000到30002。该两步仅是验收，正式训练仍从原始30000开始。

标签目录为`/mnt/d/data/uv_curve_targets/fixed_bspline_supervision_v2_20261027`，固定参数SHA256 `86f05342ed4b50ab4f97bdb3d9a10b96b5c1a64eefa567b935e6c3b2c9cc3121`。投影代码为`de61a2a8`，后续只收紧异常配置/负reason flag校验，数值算法和默认参数未变。6起点、17×17粗网格、20步有界GN、4步长线搜索、轨迹DP；残差阈值1e-3、连续UV步长阈值0.35，其余设置完整写在每份sidecar与报告。7分片最大耗时104.59秒；独立读取全部sidecar审计15.48秒。

46,108,032个UV点中46,080,715有效；完整单侧轨迹2,877,130/2,881,752。零有效CAD=0、零有效双侧边=0；单侧零有效轨迹1,278照常保留XYZ/切向目标。无效点主要是多解/参数跳变，39点残差超阈值。有效点残差全局P99为1.57e-7。全缓存审计将源SHA与生成日志对照，没有重算全部raw SHA；训练loader逐件重新核对原始文件摘要。

正式阶段目录：`/mnt/d/python/checkpoints/uv_intersection/joint64/20261025T144428Z/stages/supervision_v2_20261027T081450Z`。启动代码commit `16b7129afb9d9b4b2c0dfae94629780a8d9b80f3`，GPU1–7，Adam lr2e-5，BF16，seed42，原计划累计步数30000→60000；实际固定评估保存至51460。启动日志确认7个rank、0冻结参数、完整checkpoint恢复；16:16:42启动验收时至少完成19次新更新。停训后原训练进程组全部退出，W&B正常结束；没有重启训练。见 [停止记录](evaluation_20261028/STOP.json)。后续文档提交不改变本次训练代码。
