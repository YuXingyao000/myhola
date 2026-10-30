# 原 SVR 与 DPO 5,000 步：validation100 快速复核

## 1. Task
回答 DPO 微调是否损伤或改善单图重建，实际运行固定100件的配对比较；不以去噪loss替代下游结果。

## 2. Assumptions
仅本次灰模view0、K=1开发验证。最终5,000步权重在查看本次结果前选定，不按结果挑中间checkpoint。
保留原HoLa的1024/24 concat及原AE；8卡可用。用户授权本次100件评估，不扩到完整validation。

## 3. Inputs
- 清单：checkpoints/diffusion/svr_comparison_20261017/validation100.txt。
- 原权重：/mnt/d/ckpt/HoLa/Diffusion_svr_sq30_1500k.ckpt。
- DPO：checkpoints/diffusion_preference_svr/20261023T030527Z/weights/steps/step=00005000.ckpt。
- 图片：/mnt/d/data/deepcad_v7_cond/{id}/imgs.npz 的 svr_imgs[0]。
- GT：/mnt/d/data/deepcad_v7/{id}/data.npz、normalized_shape.step。

## 4. Outputs
新独立run，保存双checkpoint/source/input哈希、冻结清单、逐件seed、原始latent/NPZ、
后处理STEP与诊断、逐件指标、固定分母汇总、配对CSV及静态PNG/SVG图。

## 5. Processing Protocol
两模型均重新生成100件，每件仅c00；seed=candidate_seed(42,id,0)，不重试/GT挑选。
原生DDPM1000，FP32、TF32关闭、batch1、原AE解码/去重；DPO只替换可训练五模块。
严格检查DPO的reference/DINO等于原权重，模型配置及源checkpoint hash一致。
复用eval.preference.evaluate_candidate：recover_v1、drop0、优化200/自适应600、
CPU几何，单候选整个评估进程180秒，post采样3次，双方共用GT采样缓存及采样seed。

## 6. Metrics
固定分母100：推理成功、STEP产出、native合法solid、invalid/unknown及错误/超时数量。
raw：面/边CD²（两个方向均值之和）、Hungarian面F1（CD²阈值0.1）、面数绝对误差。
post：合法且三轮几何均可计算的面/边CD²及面F1；先在每件内平均三轮，再对CAD平均。
两模型同时报告各自可计算数量/均值与共同可计算集合上的配对均值及差值。
图中CD²显示×1000，F1显示百分数；未知/不可计算不填零，失败保留在100件分母。
本次快速复核不另算顶点/FE/EV/LFD；不能将这三项几何指标称为完整论文指标。

## 7. Decision Rule
报告最终DPO相对原SVR的变化及合法性转移（救回/退步/双方有效/双方无效或未知）。
几何配对差值可用按CAD重采样bootstrap（seed42，10,000次）给区间；不据此自动调参或继续训练。

## 8. Edge Cases
缺输入、推理异常、OCC失败、超时、post采样失败分别记录，不偷偷重抽或筛除。
合法性已判定但几何采样失败的对象仍保留其合法性；post均值明确成功子集。
未知运行故障不能假定几何非法。两者有效率分母始终100。

## 9. Validation / Sanity Checks
严格加载单元检查；真实单对象DPO native接口与PreferenceDiffusion.sample_latents逐值等价，
并核对原SVR c00与先前已存候选相同。确认原AE、冻结DINO/reference及归一化图片保持一致。
汇总验证每模型恰100条终态、对象不重复、共同集合一致、均值按对象而非repeat加权。

## 10. Implementation Constraints
复用现有采样、后处理、指标；不修改模型或训练，不覆盖历史产物。
加载器在diffusion，比较编排/汇总在eval，简短注释和显式参数。

## 11. Deliverables
可复用评估入口、最小接口检查、运行记录、逐件CSV/JSON和比较图；完成本次100件后报告实际结果。
