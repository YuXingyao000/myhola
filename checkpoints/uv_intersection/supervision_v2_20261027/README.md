# UV监督第二阶段

用户授权新增曲线监督并从上一轮继续全量训练。当前实现与验收进展见下表；正式启动信息将在完成真实批次和标签覆盖验收后补入。

| 项目 | 状态 |
|---|---|
| 网络/固定曲面 | 保持30k UV模型全部结构和参数形状，170,699,647参数全量训练 |
| 新loss | 已实现直接UV、XYZ差分走向、GT尺度相对XYZ；绝对XYZ改L1 |
| 局部验证 | 新11项+原15项，共26项通过；真实数据/续训验证待完成 |
| GT UV缓存 | 按当前GT固定曲面离线反求，生成及覆盖验收待完成 |
| 正式训练 | 待启动，不能把实现通过视为已训练或效果改善 |

入口配置：`configs/train_vae_uv_intersection_supervision_v2.yaml`。实际系数见 [架构与目标](ARCHITECTURE.md)，标签协议见 [SPEC](SPEC.md)，研究判断见 [PLAN](PLAN.md)。

源checkpoint：`/mnt/d/python/checkpoints/uv_intersection/joint64/20261025T144428Z/weights/last.ckpt`，step30000、epoch72、SHA256 `c4f409b1a904d431cf82cf4754001ca590b59dc44abbaf987181819977a32318`。已实际读取：Adam有617个参数状态，全部step30000、lr2e-5。

续训使用显式`trainer.resume_from_checkpoint`保留完整模型、Adam及global_step；`trainer.run_dir`指向原run的`stages/supervision_v2_<UTC>/`，追加30k至60000。新阶段独立W&B与best分数；旧run/权重不覆盖。现有run管理支持该方式，无需修改通用trainer。阶段溯源记录源路径、摘要和起止步数。

数据保持train46480、validation3495的原accepted清单、原增强、7卡×每卡2件×累积8（112件/更新）。正式生成规则只用固定validation前32件验收；test不参与。UV失效mask只屏蔽UV项，不丢CAD/边，其XYZ、切向和拓扑监督仍保留。

W&B中的`curve_xyz`继续表示旧MSE诊断；本轮绝对目标名为`curve_xyz_l1`。所有实际loss有`weighted_`贡献，避免原始值与加权值混淆。新增face_normal只拆出诊断，不重复加权；KL只加一次。
