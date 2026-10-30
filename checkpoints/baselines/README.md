# Baseline 文献口径（2026-10-29）

用途：投稿对比前先核对每篇论文"合法率/准确率"到底量的是什么。以下均为**文献结论**，来自论文正文、附录或官方代码的阅读，未在本机复现。我们的主口径是：保存 STEP → `read.precision.mode=0` 重读 → `TopAbs_SOLID` 且 `BRepCheck_Analyzer` 通过，失败与超时都留在分母里。

## 口径一览

| 方法 | 我们关心的数字 | 实际含义 | 能否直接比较 |
|---|---|---|---|
| HoLa（我们，arXiv 2504.14257，TOG 44(4)） | VAE 消融 Table 5 的 Full 行：Val **94.18%** | 2026-10-30 已核对 arXiv v1/v2/v3 原文：§5.5 写 "ablation study on the DeepCAD test set"，四行依次为 94.18/91.63/88.37/79.62%。仓库中没有任何复现这张表的记录 | 同一 test 集。**2026-10-11 用 `1119_deepcad_aug1_11k`（light）+ 本仓库后处理得到 2048/2417（84.73%），0.1 兼容口径也是 2048**。差距原因见下节 |
| HiFi-BRep（CVPR 2026，arXiv 2608.16485） | 重建 Valid 95.2%、Face Acc 100%、Edge Acc 99.5%、Adj Acc 97.5% | 见下节，口径明显更宽松 | **已决定排除**，不作为 VAE 重建的对比对象 |
| BrepARG（CVPR 2026，arXiv 2601.16771） | 重建 Valid 87.5%（附录 Table 6，codebook 4096） | 从 DeepCAD test 随机抽 3000 件，经其后处理后判定合法；判定标准没写清楚；没有几何或拓扑误差 | 只能当作量级参考 |
| AutoBrep（arXiv 2512.03018） | 只有 FSQ 的逐点 RMSE（ABC-1M） | 没有重建合法率，没有拓扑指标 | 不可比 |
| DualBrep（arXiv 2606.31579） | ABC 点云→B-Rep：HoLa-BRep 73.98%，DualBrep-recon 76.34% | 任务是点云重建，不是 VAE 自编码；ABC 10–100 面，测试 4k | 不可比 VAE；但它在点云条件任务上把 HoLa 当 baseline |

无条件生成的 Valid（BrepARG DeepCAD 87.6%、AutoBrep ABC 70.8%、HiFi-BRep DeepCAD 72.2%）都是各自的协议，彼此不同口径，不要横向拼表。

## HiFi-BRep 为什么能到 95%

依据官方代码 `github.com/1nnoh/HiFi-BRep`（`eval/vae_reconstruction.py`、`src/eval/validity.py`、`src/utils/cad_utils.py`）：

1. **重建评估把 GT 面数、GT 边数喂给 decoder**：`use_ground_truth_counts()` 把预测的 `num_face_logits`/`num_edge_logits` 替换为 GT 的 one-hot（`docs/TRAINING.md`："decodes posterior means with ground-truth face and edge counts"）。边数是 oracle 输入。
2. **Face/Edge Acc 是数量分类的准确率**，不是几何精度。所以 100% 并不说明几何准确。
3. **Adj Acc 是 E×F 关联矩阵逐元素的准确率**：每行 F 个元素里只有 2 个为 1，大量真负例会把数字抬高。
4. **合法判定宽松**：只检查所有 shell 都满足 `Closed()`，且体积大于 1e-6；全仓库没有调用 `BRepCheck_Analyzer`，也不写 STEP 重读。`ShapeFix_Face` 的最大容差设为 0.1。
5. **数据**：DeepCAD 只有 train 83,612 和 val 6,709，没有 test；VAE 的 variant 是 `4-30` 面（包含 4–6 面的简单件）。论文没写 Table 2 用的是哪个 variant 或哪个 split。
6. 无条件生成的 DeepCAD Valid 72.2% 是两个独立模型按等量样本合并的结果：`4-30` 为 91.1%，`7-30` 为 53.3%。

## 我们当前最好的 VAE（截至 2026-10-29）

分支 `codex/uv-intersection-training`/`codex/vae-uv-boundary` 中的 shared curve + UV best42471：DeepCAD validation 3495 件，主口径 3180 件合法（90.99%）。raw 面 CD² 2.63e-5，边 CD² 5.97e-5，face-pair micro F1 99.86%，face-pair 完全正确 3393/3495。见 `checkpoints/conditional_intersection/feature_on_evaluation_20261029/REPORT.md`（在该分支上）。尚未在 test 上评估，也没有算 HoLa Table 5 的 Acc/Compl 指标。

## HoLa Table 5 的 94.18% 与 84.73% 复现之间的差距（2026-10-30，未解决）

源码事实（初始 commit `27017eb5`，`src/brepnet/eval/legacy/check_valid_legacy.py`）：
- 重读 STEP 后，先用 `ShapeFix_ShapeTolerance.SetTolerance(shape, 0.1)` 把容差放到 0.1，再跑 `BRepCheck_Analyzer`。
- `Valid rate = valid_count / len(folders)`：分母是输出目录里实际存在的文件夹数。没有产出文件夹的样本（崩溃、超时）不进入分母。

候选原因：
1. 权重不同：复现用的是 `/mnt/d/data/ae_checkpoints/1119_deepcad_aug1_11k.ckpt`，发布的 DeepCAD AE 是 `/mnt/d/ckpt/HoLa/AE_deepcad_1100k.ckpt`。
2. 分母口径不同（输出文件夹数，还是固定清单）。
3. 后处理版本或 `drop_num` 不同。
4. 消融使用的 test 子集不同。

容差这一项单独不能解释差距：10-11 复现中 0.1 兼容口径和严格口径都是 2048。

下一步：用 `AE_deepcad_1100k.ckpt` 在 2417 件 test 上重建，并分别报告：原始脚本口径、固定分母下的 0.1 口径、我们的严格口径。这样能把差距拆成"权重"和"口径"两部分。需要时询问论文一作当时的评估流程。

## 投稿前待办

- 查清 HoLa 论文 94.18% 和本仓库复现 84.73% 之间的差距（见上节）。
- 冻结的 VAE 在 test 上只评估一次：同时报告主口径、0.1 兼容口径，以及 HoLa Table 5 的 V/E/F Acc/Compl。
- 如果确实需要和 HiFi-BRep 对比，就用它的口径（GT 面/边数 + closed + volume）另外跑一张表，并在表中注明，不要和我们的主口径混排。
