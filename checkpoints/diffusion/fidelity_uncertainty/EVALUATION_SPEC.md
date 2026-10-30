# 新旧 DPO 完整 validation 对照：执行契约

## 1. Task

为已完成的新观测标签 DPO step5000 准备完整重建评估，复用旧结果、只生成新模型候选。用户延续手动启动要求；本次实现与 CPU 验证后提供运行命令，不启动 GPU 或正式全量评估。

## 2. Assumptions

已核实旧完整评估及其观测 audit 完成，固定 split 与影响生成/测量的源码未改变。原 SVR、旧 DPO、新 DPO 的比较保留同一输入信息、采样及修复预算。训练 loss 不作为重建效果判断依据。调度与报表改动可版本化，指标本身不变。

## 3. Inputs

旧完整比较 `checkpoints/diffusion/preference_eval/20261023T090834Z`；旧观测 audit `checkpoints/diffusion/failure_analysis/runs/20261023T105637Z_full`；新权重 `checkpoints/diffusion_preference_svr_observation/20261024T040949.398883Z/weights/steps/step=00005000.ckpt`；原 SVR `/mnt/d/ckpt/HoLa/Diffusion_svr_sq30_1500k.ckpt`。固定 `deduplicated_deepcad_validation_7_30.txt` 3,505 个 CAD、与 train 无重叠，不读取 test。

## 4. Outputs

独立新 comparison run、仅新 DPO 的 observation audit、跨 run 的三方法摘要（原 SVR/旧 DPO/新 DPO）、旧新 DPO 配对差值和置信区间、逐 CAD CSV、图与中文解释。原 run/audit 不修改。新增输出路径显式指定，续跑需同协议。

## 5. Processing Protocol

原采样：K=1、seed42、原生 DDPM1000、FP32、TF32 off、batch1、c00、不补抽。沿用原 VAE 与 recover_v1 / drop0 / 200(adaptive600) / CPU180秒 / 三轮 post 测量。

扩展 `preference_compare --reuse-methods baseline`，按(method,CAD)复用原 SVR；复用前核验原 checkpoint、采样/post/源码、输入和输出摘要及终态记录。只对被复用方法要求 checkpoint 相同；新 DPO checkpoint 可不同。原 SVR 全量复用时全部指定 GPU 仅分配新 DPO，不加载原 SVR 做重复推理。CPU 后处理仅处理未复用项。

接着现有 `diffusion_audit --methods dpo_5000` 仅测新 DPO，旧 audit 只读。最后跨 run 核验比较口径、同 CAD seed 与输入摘要、audit 来源，再生成旧新 DPO 报告。按三个顺序阶段组成一个可恢复的手动入口。

## 6. Metrics

固定分母：推理成功、STEP 产出、native valid/invalid/unknown、post 完整率、观测可测率和耗时。准确性：raw/post 面/边 CD²、面 F1、raw 面数误差；观测使用现有224² view0 的轮廓 IoU、可见表面 CD²、轮廓边界距离、可测深度 MAE。Post 要求三轮齐全，先在 CAD 内平均，再跨 CAD 平均。

## 7. Decision Rule

固定比较新旧 step5000，不按本次 validation 选择训练 checkpoint。报告新减旧的同 CAD 差值，CAD 配对 bootstrap 10,000 次、seed42、95%百分位区间。合法率以全3505为分母；几何/观测同时列各自可测集与共同可测集，缺测不填零。不以单个指标或单个 p 值自动决定改进；同时检查合法性、可见忠实度与完整几何取舍，评估后才进入几何训练。

## 8. Edge Cases

失败、unknown、缺输出、超时保留在预定分母，终态失败不借复用重试。源摘要、seed、split、checkpoint、指标协议变化拒绝复用/汇总；不将旧 DPO 冒充原 SVR。空共同可测集返回 null 和 n=0；NaN/Inf 视为缺测且报告。显式中断留下未完成状态，可恢复；并发运行同一输出目录必须拒绝或明确不支持。

## 9. Validation / Sanity Checks

CPU fixture 检查部分方法复用、不同新 checkpoint 的允许范围、复用源污染拒绝、GPU/CPU任务分配、旧全方法复用兼容、固定分母、共同集、配对CI方向和可重复性。真实旧完整结果做只读接口验证；不会以小样本假装新模型质量结果。手动启动前验证源权重/清单与旧 run 协议。

## 10. Implementation Constraints

复用既有生成、后处理、观测和配对统计函数，不改变采样/指标公式。保持旧 CLI 默认行为；新能力显式配置。每次记录配置、Git/代码及输入摘要。冻结 source run 保持只读，不向旧 symlink 路径写入。CPU 验证禁用 CUDA；不把自动执行正式评估作为代码检查副作用。

## 11. Deliverables

按方法复用的比较入口、跨 run 对照报告工具、可恢复的手动三阶段启动脚本、CPU 验证记录及运行说明。完整新模型评估结果由用户启动后产生。
