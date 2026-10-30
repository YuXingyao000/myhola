# 观测一致性 DPO 数据重选：执行契约

## 1. Task

实现第一阶段：复用原 K=8 候选，补算观测指标、重选每图至多一对。用户已授权实现，但最新指示改为提供命令由用户手动运行；本次只做 CPU 验证，不启动全量生成、DPO 或几何训练。

## 2. Assumptions

数据：源 run 已完成，使用原灰模 view0、原 latent、native 有效性与 raw 指标。边界：不更换 VAE、policy、后处理，不补抽候选。实现：沿用现有 CPU OCC/NumPy 观测实现；主要优化是跨 CAD 并行、同 CAD 共用 GT 渲染及 mask，无需加载 GPU 模型。阈值沿用已同意的 PLAN；不是已证实的最优阈值。

## 3. Inputs

显式 `--source-run`，读取其 `metadata/protocol.json`、完成状态、固定 train/validation ID、候选 inference/evaluation、STEP、原始 latent、GT cache、源图像与 GT STEP。核对源协议 fingerprint 与已记录文件摘要。只接受原 `svr_preference_generation_v1`；不读取 test。

## 4. Outputs

独立 `--output-dir`，协议版本 `svr_observation_preference_v1`。输出 `pairs.jsonl`、`validation_pairs.jsonl`、每 CAD 的 8 候选观测结果和完整选择报告、旧新 pair 保留/替换/翻转/弃权/新增统计、状态及 provenance。pair 的绝对路径引用源生成 latent；GT cache 显式引用源 run，不复制、不重新编码。原 run 不修改。

## 5. Processing Protocol

冻结协议与代码摘要后运行。每 CAD 一次读取输入、渲染 GT，各候选独立渲染。CAD 子进程超时/崩溃隔离，已完成候选原子写入后可复用；并发数显式配置，限制 BLAS/OpenMP 线程避免超订阅。中断后 `--resume`；协议或源摘要变化拒绝续跑。每次正式输出按冻结 ID 顺序重建 manifest，不能追加重复 pair；完整状态才能用于训练。小样本 `--limit` 独立目录明确标记 partial，不能伪装完整数据。

## 6. Metrics

V=保存的 native validity；S=输入轮廓 IoU；C=双向可见表面平方 Chamfer；B=归一化轮廓边界距离。沿用 `eval/metrics/observation.py` 的 224² view0 相机、mask 阈值245和三角化。C 使用训练 GT，并非部署时评分；B 包含前景/背景孔洞边界，不声称覆盖所有内部槽线。raw 面/边 CD² 与面 F1 仅保护合法/非法对。

## 7. Decision Rule

全部观测保护：S(W)>=S(L)-0.01、C(W)<=1.02*C(L)、B(W)<=1.02*B(L)。明确改善任一：S 增加>=0.02，或 C/B 降低>=5%且严格降低。合法/非法：保护+raw面/边CD最多退步2%、面F1最多下降0.01；双合法：保护+明确改善；其他弃权。优先合法/非法，否则双合法；排序后用 seed42+image_id 确定性均匀抽一对。不用完整 GT 在双合法之间打破平局。pair_type 保持 validity/geometry，后者另标明 observation 依据，兼容训练接口。

## 8. Edge Cases

unknown、缺 STEP、渲染失败、空几何、NaN/Inf、超时均保留在分母，缺测不得自动判输。零对零不算改善；乘法比较不除零。测量失败不改变旧合法性标签。源记录摘要不匹配是完整性错误，终止运行，不当作普通无 pair。损坏/不符形状的 latent 不能入选。源候选本就 interrupted/missing 时记录缺失，不补生成。源 split 重复、重叠、空或非法 ID 拒绝。

## 9. Validation / Sanity Checks

CPU 测试配对规则的改善/冲突/缺测/零基准与确定性；验证 source/protocol 变化被拒绝、原子输出/续跑不重复、固定分母及 partial 禁止训练。使用少数真实 validation CAD 验证 OCC 到观测指标到 pair manifest，全程禁用 CUDA。小样本仅验证实现与记录速度，不能证明 DPO 效果或预测全量耗时。

## 10. Implementation Constraints

Python 3.10+，不新增依赖；模块职责清晰，不更改旧选择器行为。CPU 子进程禁用 CUDA；不通过占满显存假装提高吞吐。源数据只读、输出目录加独占锁，错误信息包含 split/CAD/candidate。运行记录命令、Git/diff、代码与输入摘要、运行参数；失败仍可审计。

## 11. Deliverables

数据重选 CLI、独立选择器、兼容新协议的 dataset provenance、相关测试、中文运行/续跑说明与 CPU 验证报告。全量数据和后续训练由下一步运行产生，本次不宣称已完成。
