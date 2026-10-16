# 当前训练产物的解释边界

**现有证据更支持“替换关键边特征模块后仍在适配，并伴随部分目标相互牵制”，而不是已经证实“末期过拟合”或“局部注意力架构无效”。** 前半句仍是研究解释，直接事实是：总训练/验证目标持续下降，边局部重建在较长平台后才开始改善，部分面/贴合目标同时在训练集和验证集上回升；选定checkpoint的下游质量明显差于上一版。

本审查只读现有CSV、审计、训练配置与源码，没有运行新模型或改动配置/模型/post。数据来源：[training_history.csv](/mnt/d/python/checkpoints/intersection_local/evaluation_20261012/training_history.csv)、[training_audit.json](/mnt/d/python/checkpoints/intersection_local/evaluation_20261012/training_audit.json)。

## 1. 这次具体训练了什么：源码与配置事实

- **新 local inter 是新初始化，不是旧 inter 权重的微小修改。** 迁移只排除 `decoder.inter.*`，其余state键必须完整匹配并严格加载；新 inter 保留构造时的新权重。因此decoder面特征Transformer、面/边几何头、bbox头、classifier、已训练refiner都继承上一版，但新模块起初并不保证输出落在这些下游模块熟悉的特征分布上。见[初始化边界](/mnt/d/python/src/brepnet/models/intersection_vae.py:114)、[保留非inter权重并装入fresh inter](/mnt/d/python/src/brepnet/models/intersection_vae.py:133)、[新模块初始化](/mnt/d/python/src/brepnet/models/local_intersection.py:74)。最后一项“分布适配”是机制动机，不是已测量的分布结果。
- **只冻结encoder/posterior，整个decoder都继续更新。** 冻结梯度及强制eval见[模型冻结](/mnt/d/python/src/brepnet/models/intersection_vae.py:107)和[train/encode/sample](/mnt/d/python/src/brepnet/models/intersection_vae.py:156)。配置的 `trainable_scope=all` 使 trainer 不再按 `trainable_module_prefixes` 缩小范围；因此配置里虽仍留有 `inter/classifier` 名称，它们不是本轮唯一可训练模块。见[scope处理](/mnt/d/python/src/brepnet/train.py:195)。审计已验证encoder/posterior实际权重与来源一致，不只依赖配置声明。
- **这是新optimizer的初始化run，非恢复旧optimizer继续训练。** 实际 `resume_from_checkpoint=null`，Adam对所有 `requires_grad` 参数使用同一个lr=2e-5；8卡×16、bf16 mixed。见[运行配置](/mnt/d/python/checkpoints/intersection_local/frozen/20261012T073904.575690Z/metadata/hydra/config.yaml:1)、[Adam参数选择](/mnt/d/python/src/brepnet/train.py:277)。源码没有为新inter和已迁移模块分设学习率。
- 新inter使用 **2层、hidden128、4头** 的单query cross-attention；两面CNN特征和预测XYZ都**没有detach**。边相关loss因此有直接路径更新面CNN及面几何支路。见[实际结构配置](/mnt/d/python/checkpoints/intersection_local/frozen/20261012T073904.575690Z/metadata/hydra/config.yaml:85)、[memory梯度路径](/mnt/d/python/src/brepnet/models/local_intersection.py:126)、[decoder接入](/mnt/d/python/src/brepnet/models/intersection_decoder.py:271)。本次不只是给原网络追加一个不干扰旧输出的残差分支。
- 上一版inter配置为hidden512、8层，每层8头、FFN2048；新分支FFN256。因此本轮同时改变了信息来源和网络容量，并非仅将相同规模注意力的K/V换成局部token。[旧结构](/mnt/d/python/src/brepnet/models/blocks.py:300)、[旧运行配置](/mnt/d/python/checkpoints/intersection_refinement/frozen/20261011T123836.485257Z/metadata/hydra/config.yaml:68)。不能据此判定小容量就是本次失败原因。
- 从两个实际checkpoint读取 `model.decoder.inter.*`：旧149个状态张量共33,683,008个元素，新37个共311,616个元素（约0.925%）。这只是 **inter 子模块** 的比较，不是整个模型缩小到这个比例；新模块还读取既有面CNN的局部特征。原始计数见 [intersection_capacity.json](intersection_capacity.json)。

## 2. edge_feature到底在逼近什么：源码事实

`edge_feature = L1(E_coarse[GT对应有向候选], stopgrad(E_encoder[GT边]))`，权重1。teacher来自冻结edge CNN直接读取GT六通道边采样，不是旧intersection的输出，也不是refinement后的E；GT对应关系仅用于loss映射，未筛选decoder候选。见[encoder边特征来源](/mnt/d/python/src/brepnet/models/holistic_encoder.py:48)、[GT有向映射](/mnt/d/python/src/brepnet/models/intersection_losses.py:110)、[feature loss及权重](/mnt/d/python/src/brepnet/models/intersection_losses.py:310)。

teacher feature显式detach，加上当前encoder在no_grad下运行，因此该目标不推动teacher跟着新inter一起移动。**它约束新inter及其上游decoder去恢复已有边特征表示。** 同时，当前同一个边几何头也解码teacher feature，并通过 `encoder_edge_local/bbox` 接受GT几何监督；这两个辅助loss会更新共享边几何头，不能误认为“teacher分支整体冻结”。见[teacher几何经过当前共享头](/mnt/d/python/src/brepnet/models/intersection_vae.py:178)。

`edge_local`是六通道局部采样的逐元素L1，`edge_bbox`是center三坐标+scale的L1；它们不是同一种几何距离，也不能把0.285直接解释成XYZ误差、全零曲线或特定激活饱和。[重建项](/mnt/d/python/src/brepnet/models/intersection_losses.py:221)、[L1定义](/mnt/d/python/src/brepnet/models/intersection_losses.py:47)。实际目标还有refined global XYZ、GT正边到两面三角网格的incidence、twin一致性；coarse权重.25、refined重建1、incidence/twin各.1。[权重求和](/mnt/d/python/src/brepnet/models/intersection_losses.py:299)、[incidence定义](/mnt/d/python/src/brepnet/models/intersection_losses.py:230)。

## 3. 训练曲线直接说明什么

下表直接取CSV epoch行（epoch为0起始，因此0–12是前13轮）：

| 验证指标 | epoch0 | epoch12 | epoch13 | epoch19 |
|---|---:|---:|---:|---:|
| total loss | 1.75955 | 0.972008 | 0.813182 | 0.507484 |
| edge_feature | 1.13281 | 0.478438 | 0.454245 | 0.362728 |
| refined_edge_local | 0.285134 | 0.284433 | 0.185235 | 0.0512937 |
| refined_edge_bbox | 0.0603287 | 0.0202236 | 0.0204590 | 0.0160111 |
| encoder_edge_local | 0.00148097 | 0.00126598 | 0.00173904 | 0.00209693 |
| refined_face_local | 0.00467374 | 0.00371100 | 0.00395608 | 0.00418643 |
| incidence（未乘.1） | 0.0635323 | 0.0148239 | 0.0172754 | 0.0158359 |

- 前13轮refined_edge_local只下降约 **0.246%**；与此同时feature loss、bbox、global edge XYZ已有下降。**不同目标的改善不同步**是事实；“前13轮完全没学东西”不成立。
- epoch12→19边局部L1下降 **81.97%**，最后一轮还从0.0612309降至0.0512937。20个记录epoch的train总目标和val总目标都逐轮下降；末轮是验证最优，best与last文件完全相同。记录没有证明已达配置上限或收敛。
- 相对epoch12，末轮val face_local增加 **12.81%**、train face_local增加 **12.01%**；val incidence增加 **6.83%**、train incidence增加 **5.37%**。incidence在epoch14–15附近更高，随后逐步回落，**不能描述为末轮仍在持续上升**。
- 末轮共享边头从encoder feature解码的local L1为0.00209693，而refined候选路径为0.0512937（约24.46倍）；bbox分别0.00260082和0.0160111（约6.16倍）。这表明同一头在两种输入特征路径上差距仍大；teacher直接读取GT边，不能据此要求两路径天然同等容易。
- edge_feature占val总loss的数值比例：epoch0 **64.38%**、epoch12 **49.22%**、epoch19 **71.48%**。这说明它仍是标量目标大项，**不能仅按loss占比断言它的梯度占比或压制了其他任务**。

## 4. 哪些机制解释有依据，哪些还没有被证明

**与事实相容的研究假设：** 新inter需要重新学会已有边表示及旧几何头的输入约定；在适配期间，已迁移的共享几何头、面CNN、面特征和refiner同时移动。epoch13边local开始下降时，teacher-edge辅助重建、face_local和incidence同时变差，随后部分恢复，与多目标权衡及已有几何能力受干扰相容。新inter新增的面CNN/XYZ梯度路径为这类相互影响提供了结构上的可能性。

数值上，epoch12→19总loss下降0.464524，其中edge_feature、refined_edge_local、coarse_edge_local分别下降0.115711、0.233140、0.057652；相反face_local、face_xyz及加权incidence的增加仅约0.000475、0.000240、0.000101。**所以上述几何子项回升与总loss下降完全可以同时发生。** 这只是目标数值分解，不是梯度冲突测量。

**未证实的机制：** 缺少中间激活/梯度/早期几何快照，不能把0.285平台定性为ReLU死亡、注意力塌缩、曲线均值化或latent尺度错误；也不能说epoch13发生了某个已定位的结构事件。共享头teacher误差变大支持“参数移动”，但尚不能单独定位是哪组参数或哪项loss造成。

## 5. 当前可下的结论

1. **典型“train继续变好、val总体持续变坏”的过拟合证据不足。** 两者总目标都下降；部分面/贴合项的退化也发生在训练集。局部泛化问题仍可能存在，但不能用“末期过拟合”概括这次失败。
2. **当前训练产物确实明显劣于上一版，不能用未收敛的可能性淡化实测。** 固定latent下边CD、面几何、连接和solid均退化，详见[主报告](/mnt/d/python/checkpoints/intersection_local/evaluation_20261012/REPORT.md)及[分布审计](/mnt/d/python/checkpoints/intersection_local/evaluation_20261012/diagnostics/RAW_DISTRIBUTION.md)。
3. **这还不是对“局部注意力一般无效”的识别实验。** 本轮同时改变inter容量/结构和初始化，并在既定训练目标下共同微调全部decoder；没有保持已迁移模块功能不变，也不是等预算对照。曲线与源码更提示需把“表示迁移/联合优化是否成功”与“局部几何信息有没有价值”分开判断；现有数据不能承诺继续训练就会修复。
4. 验证使用posterior采样（`vae_validation_inference=false`），测试使用固定posterior mean；train epoch均值又覆盖一轮内变化的参数。不能直接用train/val总loss高低或val/test数值差判断泛化。[validation调用](/mnt/d/python/src/brepnet/train.py:394)、[posterior采样](/mnt/d/python/src/brepnet/models/holistic_encoder.py:72)。当前记录也没有每epoch的raw/solid轨迹，不能断言epoch12的下游质量优于epoch19。
