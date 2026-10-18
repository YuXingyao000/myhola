# UV 边界监督：实现协议（2026-10-18）

## Task
为已采用的 CNN + legacy intersection + refinement + curve conditioning 增加显式边—面 UV 标签、辅助 UVPathHead 和固定对应几何监督。代码在 VAE 独立工作树；完成后交付 172.31.224.142 上直接运行的前台命令，用户手动启动和报告完成。无自动唤醒、正式训练或全量生成副作用。

## Assumptions
- 现有 `data.npz` 的面网格和有向曲线采样顺序是训练合同；不能把原生 STEP 遍历顺序直接当成该编号。
- 每面网格为 `[v_index,u_index,xyz]`。UV 使用同一原曲面的采样范围归一化，旋转 XYZ 不改变 UV。
- 单条 native edge 可读取 pcurve；旧数据中合并/重新拟合的边可能只获得近似投影。二者明确区分。
- 最终 XYZ 仍由旧几何头产生；新 UV 是辅助输出，不能宣称保证曲线在两面交线上。

## Inputs
- 显式 `--data-root`、`--step-root`（默认与 data root 相同）、`--output-root`、split 清单。
- 每对象 `data.npz`、`normalized_shape.step`。源文件摘要、代码版本、生成参数进入记录。
- 已评估的 conditioned refinement checkpoint；保留 CNN、冻结 encoder/posterior，与采用基线一致。

## Outputs
- `/mnt/d/data/deepcad_v7_uv/v1/{model_id}/edge_uv.npz`，根 `dataset.json`、逐对象 `manifest.jsonl`、汇总报告。
- `schema_version`，`source_sha256`，原 `edge_face_connectivity [R,3]`，`edge_uv [R,2,16,2]`，`edge_uv_valid [R,2,16]`，`edge_uv_residual [R,2,16]`，`edge_uv_method [R,2,16]`（0无效/1原生/2投影），`face_uv_bounds [F,4]`，面对应关系与对象状态。float32 训练字段，原始参数范围可 float64；无 pickle。
- batch 输出 `edge_uv_targets` 字典：连接行对应的 `uv`、`valid`、`residual`、`method`。源连接行过滤与标签同步，组 batch 后行顺序一致。
- 解码器辅助输出 `edge_uv [P,2,16,2]`，只对 GT 正 pair 监督；无 GT 输入推理。

## Processing Protocol
1. 验证输入、摘要、采样形状；按完整面网格坐标做 STEP→NPZ 一一匹配，拒绝不确定对应。
2. 对同一条旧 edge 的16个 XYZ，优先匹配原生边参数和 pcurve；不能精确匹配时，在已确认的两个原曲面上构造投影标签。
3. 周期面保留原参数图分支与点序；不以普通圆柱的“两环+seam”假设重建边界。遇到不可判定的多个 UV 解显式标为无效，不逐点任意选分支。
4. 保存代回原曲面的残差、方法与有效掩码。生成失败记录在原清单分母中；能保留原几何的样本输出全无效标签及原因。损坏原几何/缺失源文件明确失败，训练不静默丢样本。
5. 原子写文件，版本目录配置不可静默混用；resume 校验源摘要与生成参数，不覆盖不同来源产物。
6. UVPathHead：concat(E64,Fi32,Fj32)→Linear128/SiLU，加16点 t 位置 MLP(1→32→128)，2×pre-LN TransformerEncoderLayer(128,4 heads,FFN256,GELU,dropout0)，LayerNorm→Linear4→reshape/permute。原 inter 保持8层512维。
7. 损失：有效标签的归一化 UV L1；固定 GT UV 读取预测面，与预测曲线比较。扣除 `stopgrad(C_gt - bilinear(S_gt,U_gt))` 的插值残差。保留所有原 XYZ/局部/bbox/分类/贴合/twin 目标，不新增对预测 UV 的自由滑动贴合项。
8. 初始化仅允许新增 UV head 参数，旧参数严格保留；完整 checkpoint/resume 仍严格加载。默认开关关闭，禁用时行为不变。

## Metrics
- 标签：输入总数、成功/部分/失败数，原生/投影/无效点数量，原曲面回代残差、面匹配误差；无效点不计成零误差。
- 模型：raw/weighted UV 与固定对应几何 loss、有效标签覆盖率、原有全部 loss、有限梯度和旧输出等价性。
- 当前只验实现；合法率改进以后在固定 validation 协议评估。

## Decision Rule
- 初始源面/原生边匹配容差 `1e-5`、投影有效残差阈值 `1e-3`（数据归一化世界坐标），均显式保存；不依据 test 调整。
- 有效标签必须有唯一可追溯图分支、有限 UV、合法参数域和合格回代残差；其余 mask=false，仍保留对象与旧几何监督。
- 新 loss 初始权重 UV=0.01、固定对应几何=0.1，独立日志，属于待评估研究设置。
- 真实小批量前反向、迁移严格性、标签对应和旋转一致性检查通过后交付运行命令；不以此宣称研究收益。

## Edge Cases
- 空清单拒绝；重复 ID 拒绝或跨多个 split 合并去重并记录；路径逃逸拒绝。
- 缺失、损坏、NaN/Inf、非法编号、多个面匹配、周期歧义、退化边、合并边、全无效标签显式记录。
- 零有效点 loss 返回与输出相连的零，不产生 NaN；DDP 不因掩码缺失而漏梯度。
- 配置启用监督但标签缺失/摘要错/shape错应报错；全无效的合法 sidecar 允许但记录覆盖率。
- 不复制条件图像，不重分 split，不改变后处理。

## Validation / Sanity Checks
- 构造平面/B-spline 已知对应，证明固定 UV 监督能检测面内漂移；验证残差修正使 GT 零损失。
- 真实 validation 少量对象检查 STEP 对应、两侧 UV、失败状态与恢复行为；测试集不参与调试。
- loader 非连续 edge ID、反向连接、两对象 batch 偏移、旋转不变 UV、hash mismatch。
- 模型形状、GT 不进入 forward、UV/shared feature 梯度、原模型严格等价迁移、resume 清除初始化来源。

## Implementation Constraints
- 生成实现放 `src/brepnet/data/`；模型放现有 `models/blocks.py`、`models/vae.py`；复用 `train.py` 和 dataset 类。
- 默认单进程，支持可恢复前台运行与可见进度；不自动启动全量生成/训练，不安装 hook。
- 本次在 `codex/vae-uv-boundary` 分支，保留之前 UV encoder 评估的所有未提交修改，提交仅包含本次文件。
- 数据不进 Git，研究说明与紧凑验证记录放 `checkpoints/uv_boundary/`。

## Deliverables
生成器、sidecar 校验与 loader、可选 UV head/loss/严格初始化、训练配置、针对性测试、README 和直接在 224.142 前台运行的生成/训练命令。注明本地/远端实际验证边界与部署版本。
