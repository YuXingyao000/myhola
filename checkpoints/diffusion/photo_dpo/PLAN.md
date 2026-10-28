# 照片 baseline 的首轮 DPO（2026-10-28）

## 1. Task

用户已授权从完整照片偏好数据训练一版DPO，并启动后台运行。检验照片310k baseline经离线偏好微调后是否改善下游合法性与几何；本轮交付为实现验证、配置溯源和实际启动，不提前宣称质量提升。

## 2. Assumptions

数据：FLUX合成真实感图像，不是实拍。任务：沿用此前首轮DPO的5,000 optimizer steps，GT DDPM + DPO，无新增几何辅助loss。实现：保留用户要求空出的GPU 7，用0–6、每卡8、全局56，因此样本曝光量与旧8卡实验不同；只启动一版DPO，不额外启动普通续训对照。无需另行确认。

## 3. Inputs

生成run `checkpoints/diffusion/photo_preference_data/20261026T055801Z`：42,283 train pairs、3,160 validation pairs，原train/validation CAD分别46,627/3,505，K8。policy及reference均从 `photo_baseline/evaluation_20261026/weights/photo_latest.ckpt` 的310,000步严格初始化；原SVR权重仅提供同源冻结DINO/AE。训练GT mean和DINO CLS复用照片baseline `cache_v1`，校验该cache与生成协议记录的训练来源一致。

## 4. Outputs

独立 `checkpoints/diffusion_preference_photo/<UTC>/`；weights、logs、wandb、metadata分目录。每1,000步及结束保存完整policy/reference/DINO/Adam/步数。照片配置、短跑检查、完整运行命令和代码快照可追溯。

## 5. Processing Protocol

GT与pair独立均匀抽样。GT mean随机复制和排列为[30,32]；生成候选原始[30,32]不再编码、不重排。冻结DINO的照片CLS[1024]复用baseline缓存，可训练img_fc保留。winner/loser共享timestep与噪声，偏好分支关闭dropout，reference全程冻结。Adam lr=1e-5、FP32、beta=20、lambda=1、clip norm=1、seed42，5000步，共280,000次GT及pair抽样。validation使用完整3,505件GT，在线读取同一flux照片与生成run的GT cache；原baseline validation仅缓存100件，不能假定覆盖完整validation。3,160 validation pairs保持独立，预留冻结模型的偏好诊断，不参加optimizer。本轮不增加第二种验证目标。

## 6. Metrics

记录train DDPM/DPO loss、相对reference的preference logit/accuracy及winner/loser MSE；完整validation GT去噪loss仅作训练监测。7卡验证sampler补齐到3,507条，含2次重复，不作重建效果结论。下游质量须另用固定3,505件、K1、FP32 DDPM1000、原AE及既有post口径比较照片baseline。

## 7. Decision Rule

实际batch前后向有限、policy梯度正常、reference/DINO冻结、初始DPO约log(2)、照片checkpoint逐参数匹配、7卡短跑及严格resume通过后启动正式5,000步。保存固定阶段，不能把最低去噪validation loss称为最佳重建模型。

## 8. Edge Cases

缺文件、协议/hash/split不符、重复pair、跨split、同一winner/loser、非有限值或错误latent形状均报错；不回退灰模、不默默换checkpoint。validation的无pair图片仍参加GT监测。恢复只接受原run和原数据来源，恢复保存的reference，不重新初始化。其他未提交研究改动保留。

## 9. Validation / Sanity Checks

单元验证照片与灰模入口分离、pair来源/分割、缓存映射、冻结边界、特征与在线DINO语义、旧路径回归、严格加载/恢复；真实FP32 batch8检查loss及梯度；短7卡训练验证checkpoint和Adam恢复。正式启动检查7卡进度、有限loss、可读日志与GPU7未被占用。

## 10. Implementation Constraints

沿用现有PreferenceDiffusion及统一trainer，不改变24层主干、采样器、DPO公式、VAE或后处理。照片数据逻辑独立模块；新增源码和配置提交，既有未提交改动不混入提交，并额外保存运行源码快照。

## 11. Deliverables

照片DPO配置、输入/模型适配、针对性验证记录、后台运行及完整命令、run路径和日志。DPO效果评价与继续训练对照尚未执行。
