# 原版 HoLa 照片 RGB baseline

## Task
2026-10-25 用户明确授权启动长时间后台 baseline，今晚训练、明天再讨论条件架构。只用已有合成照片 RGB，保留原版 1024/24 层 concat 网络。

## Assumptions
从原始 SVR 的 1,500,000-step checkpoint 初始化，新 Adam/new run；不算从零训练或精确续训。沿用用户此前 10,000,000 optimizer steps 上限；新增照片训练至少允许达到原 SVR 的 150 万步，不设置早停。原 checkpoint 记录 2 卡×256、lr=1e-4、bf16；本轮 8 卡×64 保持全局 batch512。数据只有单图 identity，不沿用原多旋转增强。

## Inputs
`/mnt/d/data/deepcad_v7_cond/{id}/real_photo.npz[flux]`，完整 RGB；原始 SVR `/mnt/d/ckpt/HoLa/Diffusion_svr_sq30_1500k.ckpt`；原 AE 同源的 `preference_data/20261020T134930Z/gt_cache`。train 46,627；固定已有 validation100 用于训练期间去噪监测，不声称全量下游评价。

## Outputs
独立 RGB baseline 模型／数据入口、显式训练配置、紧凑数据准备报告与源摘要、CPU/GPU/恢复验证报告；后台训练 run 的配置、Git/source 快照、日志、W&B 与权重。

## Processing Protocol
完整 RGB → PIL Resize(224,224) → ToTensor → DINO 原均值/方差归一化；固定原 checkpoint 中冻结 DINO，FP32 编码照片 CLS，仅缓存冻结输出，不缓存可训练 img_fc。每件验证 flux 为 uint8 RGB、尺寸与数值合法，读取原 cache mean32/std32；父 protocol 绑定 checkpoint hash，逐对象 source.json 校验 precision、layout、features hash。固定单图 rotation0。缓存记录图像／latent 摘要，任何失败均报错，不回退灰模、不静默删样本。正式训练只读完成的缓存；无增强使冻结特征缓存合理。缓存阶段允许独立 GPU worker 分片，正式训练由 Lightning DDP 统一管理。

## Metrics
训练/验证 epsilon MSE、有限梯度、optimizer step、实际吞吐、GPU 利用率／显存、checkpoint 落盘及 resume。这里不宣称照片重建质量或有效率提升。

## Decision Rule
严格旧权重迁移、缓存来源、真实 batch 的 loss/gradient、短 DDP 和 resume 检查通过后启动长跑；确认所有 GPU 活跃、步数推进且 checkpoint 已生成再交付状态。保持全局 batch512，不为显存占用率修改研究预算。保存定期快照并限制数量，避免长跑耗尽磁盘；best/last 保留。

## Edge Cases
缺失／损坏图像、cache 来源不符、split 重叠、重复 ID、非有限值、超过30面、feature维度不符均拒绝；失败列表保留。新 run 不覆盖现有结果，resume 校验缓存指纹、模型结构并恢复 optimizer/step。

## Validation / Sanity Checks
FP32 原 diffuse 与复用 denoiser 输出一致；冻结特征离线/在线误差记录；真实照片 batch 反传；所有可训练参数获梯度，DINO冻结；完整缓存检查；8-GPU smoke 与严格 resume；正式 checkpoint 写出，确认缓存输入不是灰模。

## Implementation Constraints
只修改相关模块，保留已有未提交工作；共享原 LegacySVRDenoiser，网络不增层、不变宽；新模型不加载无用 reference 或训练 VAE/OCC。不做 depth/normal 条件、DPO、几何 loss、SAM裁剪或质量阈值筛选。已有图片质量局限保留在实验解释中。原验证100仅作监测，其他 validation / test 不参与本次参数选择。

## Deliverables
已运行的 tmux 后台长训练、可复制的训练／恢复命令、实时状态路径与预算说明。若实际吞吐不足，优先修复供数／日志／通信开销，不悄悄改变网络或有效 batch。
