# Task

测试用户中断后的 intersection refinement VAE，判断相对原 HoLa light 的几何、预测连接和后处理 solid 是否改善。先固定验证集最佳权重，再读取测试结果；本轮不续训、不改网络/loss/阈值、不按 test 挑 checkpoint。

# Assumptions

训练 run 为 `frozen/20261011T123836.485257Z`。本轮为旧模型初始化后的微调效果评估，不是等预算架构消融。冻结 encoder/posterior 必须与原版精确相同；核对后可重放旧固定比较保存的输入 latent，避免重复编码带来数值差异。

采用已审计 2417 个 test CAD；原始 2424 件的 7 个合同不支持样本保留总体分母。原版既有后处理仅在清单、raw SHA、采样设置、预算和全部 post 算法文件 hash 匹配时复用，保留其来源和原协议，不伪造为本次运行。

# Inputs

- 新权重：`checkpoints/intersection_refinement/frozen/20261011T123836.485257Z/weights/epoch=0010-step=00003993-val_loss=0.039476.ckpt`；由 checkpoint callback 的 val/loss 最优规则固定。
- 原权重：`/mnt/d/data/ae_checkpoints/1119_deepcad_aug1_11k.ckpt`。
- 输入来源：`checkpoints/joint_boundary/comparison_20261011/raw_predictions/{posterior_mean,archived_latents}/legacy/{id}/data.npz` 中已保存的 float32 face_latents 与输入哈希。
- 数据：`/mnt/d/data/deepcad_v7`；清单 `checkpoints/joint_boundary/validation/test_contract_audit/accepted.txt`，SHA256 `2864849e8240134c6033cc6cd64efa19337f67fea94b89a493690ae84355b7b3`。
- 原版后处理记录：`checkpoints/joint_boundary/comparison_20261011/{post,metadata/post}/{fit_only,optimized}/`。

# Outputs

本目录下保存训练审计、协议、输入/权重/代码来源、两个输入模式的逐对象预测和 raw 指标、两种 posterior-mean 后处理结果、简明中文报告和指标图。大文件与日志放既有 ignored 运行目录；紧凑报告受 Git 管理。方法名明确为 legacy/intersection，不能将新模型冒称 joint/shared-boundary。

# Processing Protocol

1. 审计 best/last 元数据与 val 历史，选定上述 best；原始 val 使用 posterior sample，test 使用固定 mean，两种 loss 不直接比数值。
2. 严格加载新模型，核对 encoder/posterior 权重、配置与原版一致。推理采用原评估 FP32、关闭 TF32、seed42、逐对象独立解码；GPU 仅按固定 ID 分片，不使用补齐 sampler。
3. 两个输入模式分别重放完全相同的既存 latent：posterior mean 为 identity B-Rep 重建；archived latent 为之前固定 diffusion 输出的 decoder 替换诊断。后者不重新采样、不宣称完整照片流程已改善。
4. 新模型使用预测 classifier 概率阈值 >0.5，保留有向半边，不使用 GT connectivity 筛边，也不额外去重面。原版保持历史 no-dedup 输出。
5. raw 指标复用既有几何/拓扑公式并显式参数化 candidate 方法名，检查两方法 input_latent_sha256 一致。
6. posterior mean 对新模型执行 fit_only 与 optimized；CPU、drop_num=0、名义优化200次（原算法可延至600）、整个对象子进程超时180秒、逐ID固定seed。原版符合协议的逐对象终态只读复用，否则重新运行所需部分并说明原因。
7. 原读取容差的 TopAbs_SOLID+BRepCheck 为主有效性；tolerance=.1 的历史兼容口径单列。STEP存在、success marker与solid有效性分开。可读原生STEP采样用于几何评估。

# Metrics

面全局XYZ对应点MAE、面/边双向平方CD、unordered面连接P/R/F1、单向pair及互反曲线差异；archived模式不借用GT面顺序计算MAE/拓扑F1。post报告有效solid/2417与/2424、失败/超时/缺失、各自可计算集合及共同集合的原生STEP几何。补充逐对象成功转换：原失败→新成功、原成功→新失败、两者均成功/失败。

# Decision Rule

分别判断几何、连接与solid变化，不以val总loss平台代替下游结论。不能把所有变化都归因于新增反馈模块，因为还共同改变了训练损失和候选规则。posterior重建与固定diffusion latent重放结论分开，不修改test前固定的checkpoint/阈值/修复预算。

# Edge Cases

缺失、非法形状、NaN/Inf、输入哈希不一致、错误索引、不可读STEP、几何采样失败、超时均显式留在固定分母。不把空预测或不可算CD当零。保存每ID状态；不同协议不得静默混用或覆写旧产物。

# Validation / Sanity Checks

验证新模型严格加载、冻结权重相等、若干identity mean与缓存latent一致；对方法参数化运行既有指标/后处理测试及intersection有向半边行为测试；小批导出通过当前raw合同检查后执行完整清单。复用baseline前核对协议、算法hash及全体raw文件hash。

# Implementation Constraints

复用已有metric/post算法，仅调整调度和明确的方法命名；旧legacy/joint调用默认行为保留。所有新增记录在checkpoints，不写experiments。禁止选择GT best-of-K，不启动正式训练。

# Deliverables

可复现命令、训练状态审计、2417件完整raw/post对比、固定归档latent诊断、直观指标图、中文结论和Git提交。明确任何尚未完成部分，不用小批结果冒充全集结论。

## 数值复核门槛修订（完整指标计算前）

首轮 mean 数值复核使用 atol=1e-6、rtol=1e-5，四个分片被拦下。随后对固定的24个审计ID各重复5次：原legacy自身GPU重算最大差2.89e-6，与自身旧缓存最大差4.29e-6；新旧实现间最大差3.34e-6。两件真实CPU案例逐bit一致，243个encoder/posterior张量也全等。原legacy和新实现各自仅77/120次通过原严格门槛，表明该门槛会误拒绝原模型本身。

因此将审计 atol 改为1e-5、rtol不变，两者均120/120通过；仅修正数值复核，不改模型、输入latent、SHA校验、预测阈值或指标。首次部分输出/协议/日志完整保存在 `metadata/initial_mean_audit_failed/`，最终两模式使用统一修订协议全量重跑。详见 `numerical_audit.json`；不得将首轮成功子集混入最终统计。
