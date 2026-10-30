# 冻结原 decoder 的预测采样点几何监督

## 1. Task

按用户 2026-10-25 授权，实现原 DPO + 几何损失及同起点继续 DPO 控制组，给出手动命令和准确 loss/架构说明；不启动正式训练，不新增预测头或改变推理。

## 2. Assumptions

继续使用旧 concat SVR、原内置 VAE、灰模 view0、原 GT cache 和旧 pair。几何梯度作用于 GT 去噪分支，不为 winner/loser 配置伪 GT。GT 面顺序由生成 cache 时的原始 data.npz 保证，通过 source.json 摘要核验。

## 3. Inputs

共同初始化：`diffusion_preference_svr/20261023T030527Z/weights/steps/step=00005000.ckpt`。原 SVR `/mnt/d/ckpt/HoLa/Diffusion_svr_sq30_1500k.ckpt` 提供冻结 decoder/reference/DINO；旧 `preference_data/20261020T134930Z` 提供 pair/cache，原 DeepCAD v7 提供 GT XYZ/有向半边 owner pair。源 train/validation 清单不变。

## 4. Outputs

可选几何 loss、显式数据对应、严格初始化/恢复/推理加载、两份训练配置、针对性测试和 CPU 真实数据检查记录。新 run/新 Adam，不覆盖旧 run；各额外 5000 steps。

## 5. Processing Protocol

保留原 random-duplicate padding 的随机调用，额外返回原始 N 个真实面的 slot 位置；每面只取其原始副本一次，不把复制面当独立 GT，不平均预测 latent。epsilon 预测按 `z0_hat=(zt-sqrt(1-alpha_bar_t)*eps_hat)/sqrt(alpha_bar_t)` 还原，不 clip、不 detach。

仅 SNR(t)>=1 的 GT 样本参与几何项。冻结原 AE 为 eval；按原 decoder 的 face attention、surface head、inter 和 curve head 计算全局 XYZ。曲线按 GT 有向 owner pair 取特征；不使用二值分类筛选，不加拓扑损失，不走 inference/OCC/NumPy。GT pair 仅训练可用，推理仍原生预测连接。

## 6. Metrics

`CD2(P,Q)=mean_p min_q ||p-q||² + mean_q min_p ||q-p||²`。每个 CAD 分别对真实面和 GT 非 self 有向半边求均值，再相加。只监督 XYZ，不监督 normal/bbox 参数本身。空边的 curve 项为 0。

`L_geo=(1/B)*sum_i 1[SNR_i>=1]*(L_surface_i+L_curve_i)`，分母为整个本卡 GT batch，非有效样本数，便于 DDP 等权。总 loss 为 `L_ddpm + 1.0*L_dpo + lambda_geo*L_geo`；DPO 保留 beta20、shared t/noise 与 `-beta/2` 的现有定义。验证 loss 仍为 GT epsilon MSE，不增加几何计算。

## 7. Decision Rule

两组仅几何项开关不同；继续使用同一旧 pair/reference、8卡×batch8、FP32、lr1e-5、seed42、clip1、fresh Adam。几何权重在固定 train 首8件、固定 seed 和分层 t=[62,187,312,437,562,687,812,937] 上一次校准：使 weighted geometry 对 epsilon 预测的梯度范数约为 DDPM 的 10%，数值写入配置后固定。该比率是初始化尺度检查，不是研究最优值，也不声称参数空间梯度比相同。正式效果仍由固定完整 validation 判断。

## 8. Edge Cases

原始/cache 摘要变化、几何 shape/索引/非有限值、缺失字段显式报错；过滤 self-owner 与原编码一致。不丢弃失败 CAD。全 batch 高噪声时 geo 为有限 0、DDPM/DPO 正常反传。冻结参数不更新，几何计算保留 latent 梯度。新增 decoder state 必须严格恢复；旧 checkpoint 初始化只允许显式新增原冻结 AE，禁止宽松加载。

## 9. Validation / Sanity Checks

检查 loss 数学/方向、门限/全 batch 归一化、空边、padding/排列、所有 loss 组成、冻结边界、新旧初始化与 resume、推理加载保留原行为。CPU 真实小批量核对 GT/cache 形状与摘要、原 decoder 对应路径数值、有限且非零 latent/policy 梯度；不以接口检查宣称效果提升。

## 10. Implementation Constraints

改动限定 preference.py、preference_dataset.py、必要的 trainer/legacy 严格加载边界、独立几何数学模块与相关配置/测试。保留原 policy/reference/DINO 参数名称和计算，不新增可训练参数、不换新版 VAE、不改采样/后处理。原几何外默认配置行为保持不变。

## 11. Deliverables

可运行代码、两组手动命令、固定权重/损失/架构说明、CPU 验证报告。完成后向用户确认实际 loss 与架构变化，正式训练由用户启动。
