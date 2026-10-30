# 几何辅助训练对照评估（2026-10-25）

## 1. Task

用户确认两组运行完成并要求开始测试。核验固定 checkpoint 后直接启动完整 validation 对照。

## 2. Assumptions

比较继续 DPO 与增加几何项两组，各自新 run step5000（旧 DPO5000 + 额外5000）。使用当前冻结的严格加载器，同时生成两组；不复用旧 loader 协议下的预测，不重跑第三个模型。沿用 spec-research-data-code 技能记录数据协议。

## 3. Inputs

控制组：`checkpoints/diffusion_preference_continue/20261025T054547.727170Z/weights/steps/step=00005000.ckpt`。
几何组：`checkpoints/diffusion_preference_geometry/20261025T062543.945628Z/weights/steps/step=00005000.ckpt`。
原 SVR `/mnt/d/ckpt/HoLa/Diffusion_svr_sq30_1500k.ckpt` 提供不变的 decoder；固定 `src/brepnet/data/list/deduplicated_deepcad_validation_7_30.txt` 3505 件。原 DeepCAD v7 raw/cond 根目录不变。

## 4. Outputs

独立 comparison（两组 raw、post、合法率、完整几何与配对区间）和 observation（可见指标、缺测、失败阶段与配对区间）。协议记录两组权重摘要、真实标签、训练来源；后台入口保存日志和阶段状态。

## 5. Processing Protocol

扩展现有 preference_compare，允许 baseline 槽显式加载第二个 PreferenceDiffusion checkpoint。内部目录名 baseline/dpo_5000 保留兼容，但协议与报告明确映射为继续 DPO / DPO+geometry；独立 pair 协议版本禁止误复用原 SVR baseline。

8 GPU 分给两组各4卡；每卡 batch1、每图 K=1、c00、seed42、原生 DDPM1000、FP32、TF32 off、不补抽。全部生成完成后，16 CPU workers 按 recover_v1/drop0/200(adaptive600)/180秒/三轮 post 评估；随后16 workers按原224²视角0、120秒观测协议评估两组。不修改采样、后处理、指标公式。

## 6. Metrics

固定3505分母的生成覆盖率、STEP产出、native valid/invalid/unknown；raw/post 面/边 CD²、面F1、raw面数绝对误差；输入轮廓IoU、轮廓边界距离、可见面CD²、深度MAE及overlap coverage。后处理三轮完整时先CAD内平均。

## 7. Decision Rule

固定step5000，无checkpoint挑选。差值为几何组减控制组；CAD配对bootstrap10000次、seed42、95%区间。分别报告各自可测集与共同可测集，不用训练loss宣称重建提升。只用validation，不读取test。

## 8. Edge Cases

所有失败/缺失/超时留在固定分母；缺测为null，不填零。中断不补抽，不覆盖旧产物；恢复需相同源码、配置和输入摘要。冻结reference/DINO/decoder必须通过原严格加载器的逐tensor相等检查。

## 9. Validation / Sanity Checks

已读取两组checkpoint：global_step与所有Adam step均5000，初始权重SHA、旧pair SHA、训练预算一致，geometry为false/0与true/0.0226。补充CPU测试验证baseline确实加载控制checkpoint、协议不能与原SVR混用、报告标签正确；启动后核验全部GPU worker与首批真实输出。

## 10. Implementation Constraints

只改评估入口的模型选择、协议/报告标签及启动脚本，不改模型训练、采样、后处理或测量公式。不委派agent，不启动新的训练。

## 11. Deliverables

可复现的两组评估入口、冻结来源记录、已启动且首批生成正常的后台任务，以及输出路径和真实进度。
