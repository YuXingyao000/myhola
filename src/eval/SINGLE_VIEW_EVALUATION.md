# 单视图重建评估与失败分析

这些工具读取已经完成的 diffusion 对照结果，统计失败停在哪一步、预测几何是否自洽、
结果是否符合输入视图，以及候选之间有多少变化。它们不重新生成、不修复CAD、不按GT挑选候选。

单张图不能确定所有隐藏几何。因此，完整GT重建误差、输入视图一致性和实体合法性分别报告。
完整GT误差较大但轮廓相近的模型需要进一步检查，不能直接判为错误，也不能直接认定是合理多解。

## 选择入口

| 任务 | 入口 |
|---|---|
| 已保存的原模型／DPO完整对照：失败溯源、内部几何检查、单视图指标、重复度 | `python -m src.eval.diffusion_audit` |
| 已有逐件audit结果：重新汇总或调整图表字体 | `python -m src.eval.diffusion_audit_report` |
| 已有原模型／DPO指标汇总：导出独立的中文对照图 | `python -m src.eval.preference_figures` |
| 同一张图片已有多个候选：计算候选间差异与raw指纹重复度 | `python -m src.eval.metrics.conditional_diversity` |
| 普通STEP结果目录：原有condition、validity、complexity等指标 | `python -m src.evaluate`，见[通用说明](README.md) |

从仓库根目录运行，使用项目已有环境。STEP读取和三角化需要PythonOCC；数值部分使用
NumPy、SciPy，图表使用Matplotlib。公开入口与主体代码位于`src/eval/`，
目前仍复用`tools/vae_compare_post.py`、`tools/legacy_failure_geometry.py`的文件、进程及
STEP检查函数；因此需要完整仓库，尚不能只复制`eval/`目录作为独立安装包使用。

## 1. 已完成对照的完整分析

```bash
python -m src.eval.diffusion_audit \
  --comparison-run checkpoints/diffusion/preference_eval/20261023T090834Z \
  --output-dir checkpoints/diffusion/failure_analysis/runs/full_validation_audit \
  --methods baseline dpo_5000 \
  --workers 16 \
  --timeout 120 \
  --resolution 224 \
  --mesh-deflection 0.002 \
  --mask-threshold 245 \
  --distance-threshold 0.01
```

中文图表可增加`--font /path/to/SourceHanSansSC-Regular.otf`。输出目录必须是新目录。
`--timeout`限制本次每件分析的CPU子进程时间，不改变原生成／后处理预算。

原对照的`status.json`必须是`complete`。如果它还在计算原有指标，可增加
`--wait-seconds 3600`，等待最多一小时；不会把正在评估的样本算作失败。
等待失败或输入协议不满足时程序报错，不自行启动或补跑原对照。

中断后使用相同命令加`--resume`。恢复要求源码、对照协议、参数和已记录输入hash一致，
已完成记录直接复用；已有超时、异常等终态也不会被自动重试。改变预算或代码重新分析时，
使用新的输出目录，保留旧结果。

### 对照适配器需要什么

`comparison_cases()`读取`metadata/protocol.json`中的`identities`作为固定分母，
读取同一协议的`config.raw_data_root`和`config.condition_data_root`定位GT与输入图。
它不会用当前已有输出目录反推样本清单。

```text
comparison_run/
  metadata/protocol.json
  status.json
  raw_predictions/{method}/{image_id}/
    inference.json
    data.npz
  evaluations/{method}/{image_id}/
    metrics.json
    post/c00/
      recon_brep.step
      post_diagnostics.json
```

当前适配器针对每图单次生成`c00`；报告明确拒绝同一方法／图片有多个候选，K>1走第4节的独立入口。
方法名由`--methods`显式传入。推理记录中的raw与
输入hash、评估记录中的STEP hash在存在时必须匹配。额外保存实际读取文件的路径与hash。
数据接口不同的模型应提供新的明确适配器，不在路径缺失时猜测另一个run。

raw内部几何检查要求HoLa原有半边表示：`pred_face[F,U,V,3]`、
`pred_edge[E,P,3]`、`pred_edge_face_connectivity[R,3]`，连接行是`[edge_id, face_a, face_b]`。
显式标为其他表示的NPZ不会被当成半边模型解释。

### 输出结构

```text
audit_output/
  protocol.json                    # 配置、原对照协议hash、分析源码hash
  launch.json                      # 命令、Git提交、启动时间
  git.diff                         # 启动时工作区差异
  manifest.json                    # 固定的全部method/image/candidate记录
  status.json                      # 等待、分析进度或complete
  cases/{method}/{image_id}/
    audit.json                     # 来源、阶段证据、连续指标、状态与错误
    worker.log                     # 隔离CPU子进程日志
  gt_render_cache/                 # 新建的GT深度缓存，不修改原GT或预测
  report/
    summary.json
    cases.csv
    README.md
    01_failure_stages.png/.svg      # 有失败记录时生成
    02_internal_consistency.png/.svg
    03_observation_consistency.png/.svg
```

完整几何CD/F1等原指标继续保留在原对照结果中；`audit.json.raw_metrics`保留对应raw指标。
新可见指标使用独立名称，不能替换或混入原完整几何指标。

只重新生成统计表和图，不重新读取STEP进行几何计算：

```bash
python -m src.eval.diffusion_audit_report \
  --audit-dir checkpoints/diffusion/failure_analysis/runs/full_validation_audit \
  --font /path/to/SourceHanSansSC-Regular.otf
```

原有`preference_report`生成的`summary.json`可另导出4张独立对照图：合法率、
后处理前面／边误差、后处理后面／边误差、面匹配F1。适用于固定100件或完整validation，
每栏的共同样本数取自原汇总，不重新计算指标：

```bash
python -m src.eval.preference_figures \
  --summary /path/to/comparison_report/summary.json \
  --output-dir /path/to/new_presentation_figures \
  --font /path/to/SourceHanSansSC-Regular.otf
```

输出高清PNG、可缩放的SVG、`plotted_values.csv`与来源hash。几何误差直接显示
`5.72×10⁻³`这样的原始数量级；F1和合法率显示百分比。CSV保留未缩放均值、各自共同分母和
原汇总中的配对差值95%区间。不要将raw与post两个不同共同集合的均值当作后处理收益。

## 2. 失败位置与几何诊断

`trace.primary_stage`每件只取一个值，类别总数等于固定请求数。已经确认合法的实体，
不会因为后续某个几何指标报错而改记为非法。

| 记录值 | 含义 |
|---|---|
| `valid_solid` | 原评估已经确认合法solid |
| `generation_failure` | 推理记录明确失败 |
| `pending` | 推理／评估尚未形成完整终态记录 |
| `raw_input_failure` | 原评估无法读取或处理raw几何 |
| `budget_exhausted` | 原评估／后处理达到预算 |
| `incomplete_trimmed_faces` | 最后一条恢复路径仍有面未成功裁剪 |
| `no_solid` | 最后一条恢复路径未构造出solid |
| `native_readback_or_boundary_rejected` | solid／STEP重读或边界检查拒绝 |
| `execution_error` | 原流程发生执行或读取异常 |
| `unresolved` | 已有记录不足以定位到上述阶段 |

所有恢复路径另存`path_summaries`；`furthest_recovery_stage`表示各路径曾到达的最远阶段，
不一定等于最后一条路径的失败位置。`stage_evidence`指出具体诊断路径，方便回到逐件记录核对。

与互斥阶段不同，以下观察可以同时出现，因此比例不能相加当作总失败率：

- 半边只有一个方向、同一方向重复、自连接、非法连接索引、孤立面、面图不连通。
- 两条互反曲线不重合；曲线偏离其所属面。
- 后处理曾出现边界环组装失败、开环、面修复减少边界环、裁剪拒绝、STEP重读拒绝等。

内部几何检查不读取GT。曲线到面距离使用raw曲面网格形成的有限三角形；
互反曲线按弧长取64点，到对方原折线计算距离。默认用每条边／所属面的距离P95，
以及互反两方向P95的较大值，与归一化XYZ长度`0.01`比较；同时保存连续距离和具体边／面编号。
它不是平方Chamfer，也不是毫米或工程容差。

实际阈值从每件记录读取并显示，混有不同阈值的记录会报错。`raw_distance_distributions`
另按全部／valid／invalid分组：先在每件内对边／所属面或互反pair的P95取均值，再汇总图片分布，
避免只有二值异常率而丢失偏差大小。

这些是离散几何近似和异常共现证据。面图连通不保证能构成实体，边到网格较远也不能单独证明
失败由某个神经网络模块导致。没有保存的拟合残差、优化位移和缝合内部轨迹记为
`not_recorded`；仅凭这次输出不能把责任划分给diffusion或VAE。

本次分析自身的超时、崩溃、来源hash变化另记在`audit.status`与`errors`中，
不要与原生成／后处理的失败阶段混为一谈。
保存STEP的独立检查与旧valid/invalid结论不一致时单列；旧结论为unknown而本次能读取检查的对象
另记`unknown_resolved`数量，不计作前后判定冲突，也不静默覆盖原来的validity。

## 3. 单视图观测指标

当前实现是**原HoLa灰模SVR的view 0协议**，不是通用的任意照片相机估计器：

| 参数 | 值 |
|---|---|
| 图像 | `imgs.npz["svr_imgs"][0]` |
| 相机位置／目标／参考up | `(2,2,2)`／`(0,0,0)`／`(2,2,3)` |
| 垂直视场角 | 45°，透视投影 |
| 默认分辨率 | 224×224 |
| 坐标约定 | 相机x向右、y向上、z向前；图像行号向下 |
| 默认三角化偏差 | 0.002，沿用已有归一化XYZ坐标 |

预测STEP和GT STEP分别三角化**已裁剪的面**，用相同相机独立做z-buffer；
透视深度使用逆深度插值。每个前景像素的最近表面反投影成一个三维点。
不使用ICP、旋转选优或逐件缩放，也不将所有raw面与可见GT面直接比较。
可读取、可渲染的非法STEP也能计算这些指标，合法性另外报告。

输入图前景固定取`min(R,G,B) < 245`，需要时以最近邻缩放二值mask。
这适用于当前白底灰模；有复杂背景的实拍图不能直接沿用该mask规则。

| 指标 | 定义与解释 |
|---|---|
| `silhouette_iou` ↑ | 预测渲染前景与输入图前景的交集／并集；直接使用输入图 |
| `silhouette_boundary_distance` ↓ | 两个mask边界之间，两方向最近像素距离的均值之和，再除以图像对角线；不平方、不除2 |
| `visible_face_cd_sq` ↓ | 预测可见点集与GT可见点集之间，两方向最近**平方**距离均值之和；另存两个方向 |
| `depth_mae` ↓ | 双方共同前景像素上的相机z深度绝对误差均值；无交集时为null |
| `overlap_coverage` ↑ | 共同前景像素数／GT前景像素数；与深度误差一起读，避免只比较很小的交集 |
| `gt_input_silhouette_iou`、`gt_input_silhouette_boundary_distance` | 将GT自身重渲染后与输入图比较，检查相机／光栅化的基准差异 |

可见平方CD使用归一化坐标的平方单位；深度误差使用归一化长度。二者均不是毫米。
原OCC灰模中的轮廓线和当前三角渲染存在差异，因此GT自身的轮廓指标不必等于完美值。
不能自动剔除这些对象以提高模型成绩。

可见三维误差仍使用**配对GT**提供的三维深度，它不是单张输入图直接测得的量。
隐藏表面不直接进入可见点集，但新几何如果遮挡了原可见部分，仍会改变渲染结果并被计入。
这允许单独检查可见证据，不等于验证隐藏补全已经正确。

## 4. 重复度与同图多候选差异

完整对照每图K=1，只能检查不同输入间是否反复生成相似输出，不能测量同一张图的多解性。
同图多候选需要明确列出每张图的全部预定候选，包括缺失输出：

```json
{"method":"baseline","image_id":"example","candidate_id":"c00","raw_path":"/path/to/c00/data.npz","validity":"valid"}
{"method":"baseline","image_id":"example","candidate_id":"c01","raw_path":"/path/to/c01/data.npz","validity":"invalid"}
```

上述是JSONL的两行；也接受JSON数组或`{"records": [...]}`。相对`raw_path`以manifest所在
目录为基准。`validity`使用已保存的`valid`、`invalid`、`unknown`，工具不重新构造solid。

```bash
python -m src.eval.metrics.conditional_diversity \
  --manifest /path/to/intended_candidates.jsonl \
  --output-dir /path/to/new_diversity_report \
  --workers 8 \
  --quantization-steps 0.0001 0.001 0.01
```

输出`manifest.json`、`diversity.json`、`fingerprints.json`以及
`candidates.csv`、`pairs.csv`、`images.csv`。输出目录必须是新目录。

**raw网格指纹：** 面点与边点分别按`floor(x/h+0.5)`量化、去重、排序，然后组合hash，
固定`h=1e-4/1e-3/1e-2`。不单独对齐或归一化；点顺序、面顺序和完全重复点不改变指纹。
分别报告不同指纹数／可测样本数、重复组成员数／可测样本数、重复pair数／可测pair数，
保留重复组ID和缺失名单，并另列valid子集。

相同指纹只说明采样点占据相同量化格子，不证明严格的B-Rep几何／拓扑等价。
没有相同指纹也不能排除近似重复：量化边界、采样密度和未裁剪raw表示都会影响结果。

**同图候选差异：** 对同一`(method,image_id)`中每个候选对，以全部原始面点／边点分别计算
双向平方Chamfer；每图先取pair均值，再对图片等权平均。每图K=8有28对，K=1输出null，
不会将“无法测量”写成“差异为0”。全部候选与双方valid的候选对分别汇总。
缺失数据的组标为partial，同时报告测得pair数和仅完整组的均值。

距离较大可能来自合理补全，也可能来自坏几何或偏离输入；必须结合合法性与可见一致性分析。
原SVR已有100图K=8可作单独附录，但DPO没有对应K=8时不能推断DPO使多样性提高或降低。

可导入接口为`geometry_fingerprints()`、`summarize_fingerprints()`、
`evaluate_conditional_diversity()`。数值命名为`mean_pairwise_cd_sq`：
[MPC官方TMD实现](https://github.com/rundiwu/Multimodal-Shape-Completion/blob/master/evaluation/total_mutual_diff.py)
采用不同缩放，数值为相同pair均值的K倍，不能直接混报。
[HoLa论文](https://arxiv.org/html/2504.14257v1#S5.SS1)中的最近训练集LFD用于novelty，
也不是这里的同图候选差异。

## 5. 分母和结果读取

先看固定请求数、合法／非法／未知数量及分析覆盖率，再看指标均值。
缺失、超时、异常、不可渲染、无共同前景分别保留；不填零，不静默移出请求分母。

双方法可见指标同时给各自可测子集和共同可测图片的配对比较；每个指标的共同集合可能不同。
失败阶段以固定全集计数；异常比例以明确列出的可测子集为分母，valid/invalid/unknown分组另列。
不能把成功子集上较小的误差当作全量质量提高。

分析读取原文件，只在新的audit目录写报告与缓存。若需要修复规则或模型改进，先根据validation
提出假设，再另建固定预算的实验；此工具不会利用GT替候选修复或选择结果。

## 6. 按指定 ID 查看单视图失败案例

`observation_gallery`读取**已经完成**的audit及其manifest，只绘制显式指定的图片，
不会自动挑选表现好的案例，也不会重算或改写已有指标：

```bash
python -m src.eval.observation_gallery \
  --audit-dir /path/to/completed_audit \
  --image-ids 00000178 00000894 00003406 \
  --output-dir /path/to/new_case_gallery \
  --font /path/to/SourceHanSansSC-Regular.otf \
  --render-timeout 60
```

每件输出PNG、SVG，依次展示输入灰模、GT可见深度、各方法预测深度及轮廓叠加。
同一件的所有深度图共用数值色阶；绿色轮廓来自输入图，橙色来自STEP重渲染。
图中合法性、失败阶段、轮廓IoU和可见表面平方误差直接读取保存的audit。
没有指定`--font`时使用英文标签，避免依赖本机未安装的中文字体。

读取前核对manifest路径、逐件源文件hash、渲染源码和OCC版本；来源变化不会继续当成
原评估结果展示。渲染在独立CPU进程中执行，单个STEP超时、缺失或无法三角化时保留该列
并明确标注。它不修复、不ICP、不调整相机，也不把新渲染覆盖到原run。

输出目录必须为新的独立目录，不能放在源audit或原对照run内部。
`metadata.json`记录指定ID、源路径和hash、所用audit/manifest/字体/源码hash、
渲染状态、共享色阶和原指标；这类人工指定案例仅用于检查，不能替代全量统计。
