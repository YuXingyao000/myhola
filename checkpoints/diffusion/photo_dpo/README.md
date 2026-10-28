# 照片310k的首轮DPO

2026-10-28用户授权使用新生成的照片偏好对启动一版DPO。固定协议见[PLAN.md](PLAN.md)。本轮使用FLUX合成真实感图像，检验离线偏好微调的作用，未增加条件架构或几何辅助loss。

policy和冻结reference均严格初始化到照片baseline的310,000步；原SVR checkpoint仅提供同源冻结DINO/AE，不作为本轮reference。新Adam、新DPO步数，不恢复baseline optimizer。原24层网络和DPO公式不变。

## 数据与训练

生成run为 `checkpoints/diffusion/photo_preference_data/20261026T055801Z`。训练使用全部46,627件CAD作为GT流，独立均匀抽取42,283对候选；未能选出pair的4,344件仍参加GT训练。候选latent保持原始[30,32]，不重编码、排列或按pair类型重采样。

训练复用照片baseline `cache_v1` 的冻结DINO CLS和GT mean；可训练img_fc仍更新。已核对全部46,627件train及原缓存100件validation的照片/原始GT摘要与新候选生成输入一致，见[CACHE_BINDING.json](CACHE_BINDING.json)。这一选择保留baseline训练GT的数值，避免把重新编码后约1e-6的差异引入训练目标。

完整3,505件validation只监测GT DDPM loss，读取生成run的GT cache和原flux RGB；baseline的validation特征缓存只有100件，未将其冒充完整validation。3,160对validation偏好保持独立，供之后冻结模型的偏好诊断，不进入optimizer。7卡分布式验证补齐2件重复，只作训练监测；正式重建评价仍须固定单次采样和完整分母。

| 项目 | 本轮设置 |
|---|---|
| 设备 | 物理GPU 0–6；GPU 7继续留空 |
| batch / 预算 | 每卡8、全局56；5,000 optimizer steps，GT和pair各280,000次抽样 |
| 目标 | GT epsilon MSE + DPO，beta=20、lambda=1；无几何辅助loss |
| 优化器 | Adam lr=1e-5，FP32，clip norm=1，seed42 |
| 保存 | 每1,000步及最后状态；另保留val-loss top3/last |
| 比较边界 | 与旧灰模8卡DPO的320,000次曝光量不同；本轮未另跑普通续训对照 |

## 实现与验收

照片数据读取在 `src/brepnet/data/photo_preference_dataset.py`；共享PreferenceDiffusion只增加严格照片初始化与冻结特征输入。统一trainer核对pair来源、原DINO/AE、照片步数，并在resume时保持已保存的reference和目标。配置是 `configs/train_diffusion_preference_photo.yaml`。

51项相关测试通过，包含照片/灰模输入分离、跨split/同候选拒绝、cached/online loss等价、冻结边界及旧observation/geometry路径回归。真实FP32 batch8连续3次更新全部policy梯度有限，初始DPO为0.69314718，初始policy/reference及DINO与照片310k逐参数相同，更新后reference/DINO未变。实测峰值allocated约5.77GiB，在线照片DINO与baseline缓存最大绝对差1.204e-5，在预定容差内。见[MODEL_PREFLIGHT.json](MODEL_PREFLIGHT.json)。这些仅为实现验证，不是DPO重建效果证据。

7卡短跑先完成4步，再把原SVR与照片初始化checkpoint路径均设为不存在，从完整DPO checkpoint恢复到6步；全部Adam参数状态step=6，reference/DINO逐参数未变，policy继续更新，数据provenance一致，TensorBoard标量均有限。短跑validation仅1个batch；其通过不代替正式完整validation。检查见[DDP_VALIDATION.json](DDP_VALIDATION.json)。DataLoader没有保存epoch内精确采样/RNG轨迹，不声称与不中断训练逐位一致。

运行保留resolved config、命令、Git commit/dirty diff，另保存实际源码快照，覆盖已有未提交依赖。提交仅包含本轮照片适配；既有geometry/observation等工作留在工作区。

## 启动

使用[RUN_TRAINING.sh](RUN_TRAINING.sh)指定新的绝对run路径。脚本固定使用GPU 0–6，拒绝覆盖已有run；已启动任务不可重复执行。实际run和后台会话将在启动核验后写入 `RUN_STATUS.json`。从断点继续必须使用原run及完整checkpoint，不能重新初始化reference。

本次只启动DPO。下游对照使用照片310k的81.68%完整validation baseline及原评价协议；未经完整重建评估，不把训练loss或偏好accuracy当作合法率提升。
