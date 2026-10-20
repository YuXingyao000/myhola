# 单图候选偏好对选择 v2：供审核的完整算法

状态：2026-10-20，研究提议；未实现、未生成候选或pair、未启动训练。用户要求明确选样本算法及实际数据。本稿将此前仅有效性配对的建议扩展为两类配对；下面的90%分位、5%改善等是可审核的设计默认，不是实测最佳值或论文标准。冻结实现前需接受这组规则并完成validation校准。

## 1. Task

使用最初HoLa-BRep的单图SVR和配套VAE，收集同图生成候选，通过固定 `recover_v1` 后处理和GT指标建立偏好训练对。

两个明确目标：

- `validity`：生成合法实体，同时原始几何不退步。
- `geometry`：两个均合法的候选中，更准确重建配对GT的面和边，同时原始面F1不退步。

第二类目标会比较GT背面，不能宣称保留所有合理补全。它优化的是数据集配对GT重建，不是单图全部合理三维解释的覆盖率。拓扑、顶点、多样性等仍需单独评估，不能保证它们自动改善。

## 2. Assumptions

- 基础checkpoint为 `/mnt/d/ckpt/HoLa/Diffusion_svr_sq30_1500k.ckpt`，1024维、24层、原全局图像concat。
- 使用旧checkpoint内最初HoLa-BRep的AE；此前审计与 `/mnt/d/ckpt/HoLa/AE_deepcad_1100k.ckpt` 的704个tensor一致。当前UV改进VAE不直接接入本轮旧latent空间。
- 图像为灰模view0；训练latent取原cache的mean32。cache来源、rotation和输入需进一步内容验收，文件存在不是兼容证明。
- 每图 `K=8`，固定候选seed与DDPM协议，无按结果追加采样。
- 后处理显式 `recover_v1`、optimized、drop=0、名义200/自适应600步；每候选180秒整子进程预算，所有内部修复共享，不读GT。
- 本稿优先有效性pair，每图最多一对。该优先级是有意的研究选择，不是两类pair自动平衡。

## 3. Inputs

### 实际已有资产

| 内容 | 路径/合同 |
|---|---|
| GT几何与连接 | `/mnt/d/data/deepcad_v7/{id}/data.npz` |
| GT实体 | `/mnt/d/data/deepcad_v7/{id}/normalized_shape.step` |
| 灰模图片 | `/mnt/d/data/deepcad_v7_cond/{id}/imgs.npz`，`svr_imgs[0]` |
| 原版GT latent | `/mnt/d/data/ae_cache/1119_deepcad_aug1_11k_24/{id}_0/features.npy`，`[N,64]=mean32+std32` |
| train列表 | `src/brepnet/data/list/deduplicated_deepcad_training_7_30.txt`，46,627个唯一ID |
| validation列表 | `src/brepnet/data/list/deduplicated_deepcad_validation_7_30.txt`，3,505个唯一ID |
| test列表 | `src/brepnet/data/list/deduplicated_deepcad_testing_7_30.txt`，2,424个唯一ID |

本轮只读核对三份列表：内部无重复、相互CAD ID交集为0；上述四种数据文件在各列表对应路径全部存在。抽查train首项图片头为 `[24,224,224,3] uint8`，latent为 `[7,64] float32`。没有对全量内容/几何重复重新验收，也没有读取test样本内容。split无ID重叠不等于重新证明无几何泄漏；DeepCAD test曾参与历史开发。

### 逻辑数据集

| 名称 | 一条记录是什么 | 当前状态/用途 |
|---|---|---|
| `D_GT` | `(image, GT mean latent, GT CAD id)` | 已有train数据；原DDPM训练始终使用 |
| `C_train` | `(image id, candidate id, seed, original latent, raw, post, metrics, status)` | 未生成；train图片的K候选 |
| `D_valid` | `(image, winner latent, loser latent, validity evidence)` | 未生成；有效性偏好 |
| `D_geom` | `(image, winner latent, loser latent, geometry evidence)` | 未生成；几何偏好 |
| `C_cal` | 固定旧模型在完整validation 3,505张图、每图预先指定seed的K=1结果 | recover_v1新协议下未生成/验收；用于阈值计算，不训练 |
| validation/test评估记录 | 固定模型的K=1主结果及另列的K候选分布 | 不参与梯度；test在方案冻结后使用 |

`C_train`、`D_valid`、`D_geom`均为DeepCAD train的派生数据，不是新增外部数据集。全train × K8为373,016个请求，不是373,016个pair；本算法最多46,627个pair，实际更少。正式运行范围/资源预算需写入manifest，本稿不启动全量收集。

本版建议用完整validation建立校准分布，新增预算是3,505次K=1生成及后处理，不是直接复用以前100件就完成。已有100件只作历史诊断；旧legacy后处理分数不能混入recover_v1校准表。

不使用real_photo/FLUX、实拍图、ABC、多视图或无条件样本构造本轮pair。此前100件SVR对照来自validation且K=1，不能移入偏好训练集。

## 4. Outputs

- `candidates`：每候选原始去重前 `[30,32]` latent；去重映射、raw预测、post STEP路径与hash；图像、GT、模型及协议版本。
- `metrics`：六个标签字段、其余评估指标、每项可计算状态及异常原因。
- `calibration`：固定validation清单、seed、分位数计算集合、重采样记录、阈值数值与版本。
- `pairs`：`image_id,winner_id,loser_id,pair_type,rule_version`，两侧完整metric及逐条件判定。
- `selection_summary`：所有请求状态、每图合法数、各条件拒绝数、两类pair及无pair图片数、复杂度分布。

偏好训练直接使用原始生成latent，不重新编码生成CAD、不将去重结果重新padding替代原始latent、不按GT面顺序重排两侧latent。

## 5. Processing Protocol

1. 验证图像/GT/cache/VAE合同，冻结模型、种子、后处理和指标协议。
2. 用旧模型在validation生成固定K=1校准结果，计算第7节阈值；该过程不得调用test。
3. 对声明的train清单每图生成K=8，保留全部请求，不按成功率重抽。
4. 原版VAE解码、固定recover_v1后处理；逐候选计算六个标签字段，保存其他诊断指标。
5. 每图枚举所有候选组合，按第7节选择一对或零对。
6. 保存固定 `D_valid` / `D_geom` manifest。两者的图片ID在该优先规则下不重叠。
7. 训练时独立采样 `D_GT` 与 `D_valid ∪ D_geom`；原数据完整保留。偏好集按图片均匀、每对等权，不按分差大小加权。
8. 固定旧模型作reference；训练模型从同权重初始化，VAE/DINO冻结。实际更新可训练diffusion及原可训练图像投影；需要改变冻结范围时另列配置。
9. validation检查K=1和固定K候选；冻结模型、规则及预算后再做test。

## 6. Metrics

只有以下六个字段参加本版标签算法：

| 符号 | 字段 | 作用 |
|---|---|---|
| `v` | recover_v1后原生有效性，取valid/confirmed-invalid/unknown | 区分两种pair及排除运行故障 |
| `r_f` | raw face CD² | 原始面几何保护 |
| `r_e` | raw half-edge CD² | 原始边几何保护 |
| `F` | raw face F1 | 漏面/重复面等的保护 |
| `p_f` | post face CD² | 正样本质量门槛与几何改善 |
| `p_e` | post edge CD² | 正样本质量门槛与几何改善 |

- CD²为双向最近平方距离均值之和，不取根号、不除以2，存原单位。
- raw使用固定原网格。F为面primitive Hungarian匹配、CD²<0.1计TP，不是点级F1。
- post每面1024点（面内按三角面积采样）、每边128点（原curve参数均匀），坐标不重新对齐/归一化，固定采样seed。
- 同一图片的全部候选共用同一版本GT几何和同一份GT采样点；GT采样seed由image ID和评价协议确定，不随候选seed变化。校准重采样时，每一轮也共用该轮固定GT点，原始GT文件和采样cache均记录hash。
- 任意比较只做raw对raw、post对post，禁止用合法候选post CD与无效候选raw CD直接比较。
- 原生检查使用STEP保存的容差，不在评价阶段强制改为0.1；保留最大存储容差作诊断，不将其解释成统一工程精度认证。

post primitive F1、vertex指标、FE/EV、数量、LFD、耗时、历史0.1检查仅报告，不生成标签。现有FE/EV没有按几何距离删除差对应；现有visible metric不是真正重投影，因此本版不使用它们作标签。LFD不作为“越大越好”的奖励。

## 7. Decision Rule

### 7.1 正样本筛选与共同保护

定义A通过正样本筛选：

```text
Q(A) =
    v(A) == valid
    and p_f(A) <= tau_f
    and p_e(A) <= tau_e
    and F(A) >= tau_F
```

定义A的raw指标不差于B：

```text
R(A, B) =
    r_f(A) <= r_f(B)
    and r_e(A) <= r_e(B)
    and F(A) >= F(B)
```

raw固定网格和确定性计算，本版采用逐项严格不退步，不设置允许退步的经验百分比。这个规则比较保守，可能降低pair覆盖率；不可因结果数量少而临时放松。

### 7.2 有效性pair

```text
validity_pair(A, B) =
    same_image_and_protocol(A, B)
    and all_required_raw_metrics_available(A, B)
    and Q(A)
    and v(B) == confirmed_invalid
    and R(A, B)
```

这里A为winner、B为loser。B不要求可计算post几何，不能用不可计算post值填成无穷大来赢得配对。A的post指标缺失则不接受。

### 7.3 几何pair

```text
geometry_pair(A, B) =
    same_image_and_protocol(A, B)
    and all_required_raw_and_post_metrics_available(A, B)
    and Q(A)
    and v(B) == valid
    and R(A, B)
    and p_f(B) - p_f(A) >= max(rho * p_f(B), delta_f)
    and p_e(B) - p_e(A) >= max(rho * p_e(B), delta_e)
    and p_f(A) < p_f(B)
    and p_e(A) < p_e(B)
```

建议 `rho=0.05`：面和边post CD²都至少下降5%，且超过各自重采样波动门槛；额外严格小于条件排除两个零距离的相同结果。5%是可审核的初始设计选择，不是经验证的可感知精度阈值。该版本要求两个距离都改善，比“至少一项变好”更保守。

这里B也是合法实体；“负”表示相对于A重建GT较差。两个都合法但仅背面不同，仍可能因GT指标形成pair；不能承诺不处罚合理隐藏补全。

### 7.4 一张图的最终选择

```python
# 指标均已离线计算；以下是算法伪代码，不是新增可执行实现。
valid_pairs = enumerate_validity_pairs(candidates)
if valid_pairs:
    return seeded_uniform_choice(valid_pairs), "validity"

geometry_pairs = enumerate_geometry_pairs(candidates)
if geometry_pairs:
    return seeded_uniform_choice(geometry_pairs), "geometry"

return None  # 图片仍保留在原始GT DDPM训练集。
```

- 同时可产生两类pair时，有效性优先；这会影响两类信号比例，需报告。
- 每图最多一对，不能把28个组合全部当独立图片放大权重。
- 候选合格对内均匀选择，不挑差距最大的极端对；seed由固定run seed和image ID确定。
- K8全合法可进入几何分支；全无效、差异不足、指标冲突、质量门槛未过则无pair。

### 7.5 阈值的具体生成方式

下面是本版建议的可执行校准规则；尚未运行，数值结果不存在。

1. `C_cal`取完整validation清单的旧SVR K=1，采样seed在看到结果前固定，后处理必须同为recover_v1。
2. 保留原生合法且全部六个标签字段可测的校准候选构成集合S，同时报告S/3505。若S不足100，不生成正式阈值，先审查数据/协议，不静默改用旧legacy结果。
3. 使用固定线性插值分位数：
   - `tau_f = Q90({p_f(x): x in S})`
   - `tau_e = Q90({p_e(x): x in S})`
   - `tau_F = Q10({F(x): x in S})`
4. 对S中按固定hash顺序选出的100件，固定STEP几何，仅使用5组固定seed重新采样预测/GT表面点并评价。每件的10组两两绝对分差合并：
   - `delta_f = Q95(|p_f^(a) - p_f^(b)|)`
   - `delta_e = Q95(|p_e^(a) - p_e^(b)|)`
   边参数采样确定时delta_e可以为0，仍需通过5%及严格改善条件。
5. `rho=5%`控制期望的相对改善；delta控制已观察的测量波动。95%分位只是经验噪声尺度，不是每个pair有95%正确概率，也不是几何可接受标准。
6. 冻结阈值文件及校准来源后才建立正式train标签；在validation人工审核典型接受/拒绝样本与复杂度分布。若审核后改规则，升级版本、重建全部pair，不按个例临时豁免。

Q90/Q10只是排除旧模型合法输出中指标较差的尾部，不证明通过者语义正确，也不保证联合保留90%。可能偏向简单CAD，因此必须报告面数分层覆盖；若要引入工程公差或可见孔槽正确性，需要另行制定具有该语义的标准，不能从分位数推出。

## 8. Edge Cases

- confirmed-invalid仅指固定预算流程结束后确认几何/构造失败；OOM、缺文件、未知timeout、程序异常为unknown，留在请求分母但不当负样本。
- recover_v1内部多次路径属于统一预算；不能给某候选额外修复或为收集负样本故意减少修复。
- 无面/无边、NaN、不可采样或指标缺失：相关metric为N/A，不填0/无穷大、不以原点代替，不参加依赖该项的pair。
- 正样本不是GT本身；两侧都是当前固定旧模型真实生成的完整latent。GT另走DDPM训练路径。
- 没有偏好pair的图片仍使用GT DDPM；不从其他图片借负例，不人为破坏latent制造负例。
- 全部请求留在原分母；连续metric仅在可测集合汇总并披露覆盖率；合法子集与全体分开。
- 多样性按同图合法且可测LFD子集计算，少于2个为N/A，报告m/K；不将高LFD当奖励，不静默删除离群值。
- 部署不能使用GT选样；GT best-of-K只作独立oracle诊断。

## 9. Validation / Sanity Checks

- 原版AE/cache/view0和旧SVR严格加载验收；不以目录名称或shape代替内容证据。
- 用固定metric表核验接受/拒绝案例：合法但错误、面好边差、全合法且明确改善、同分、全无效、运行失败。
- 检查raw/post不混比、所有pair同图、原始latent未经重编码/重排、每图至多一对。
- 检查阈值只来自validation，pair只来自train；样本生成与后处理不读取GT条件。
- 输出完整拒绝统计及按GT面数分层的覆盖；新recover_v1基线独立计算，不拿旧74/100估算pair数量。
- 正式验证同时报告K=1和K候选分布；不会因训练目标使用某metric就省略其他下游指标。

## 10. Implementation Constraints

- 本稿是研究算法审核，不写生产代码或启动收集。
- 复用现有raw/post metric内核，显式传统一采样与后处理配置；FE/EV等历史指标不改名覆盖。
- 偏好样本等权，metrics只生成winner/loser标签，不直接通过OCC反传，也不按分差当loss倍数。
- 训练目标建议：
  `L = E[D_GT](L_DDPM) + lambda * E[D_valid ∪ D_geom](L_DiffusionDPO)`。
  两路独立采样；偏好并集每图最多一条，均匀采样，不额外平衡两类。分别记录两类loss/数量；lambda、DPO时间权重和训练预算在训练协议中另行冻结，不能称已确定。
- 冻结reference和训练模型均来自同一旧SVR。匹配实际epsilon参数化、噪声调度，不直接套用别的模型x0公式。
- 旧VAE/旧cache与UV新版迁移分开；本轮不因UV训练状态改变已冻结标签。

## 11. Deliverables

本轮交付本算法稿及实际数据资产核对。下一阶段需要冻结候选清单和预算、产出calibration文件、候选metric表、pair表和拒绝统计，完成接口验证后才提供正式训练命令。

参考：

- [原SVR/AE与数据溯源](../svr_comparison_20261017/report/provenance.md)
- [原指标协议与结果](../svr_comparison_20261017/README.md)
- [优化后处理](../../postprocess_refinement/README.md)
- [CADCrafter](https://arxiv.org/html/2504.04753v1#S3.SS5)：编译有效性偏好先例，未提供本稿的几何规则或阈值。
- [Diffusion-DPO](https://arxiv.org/abs/2311.12908)：偏好训练目标的来源，不是本项目标签算法有效的证据。
