# Conditional Intersection：feature开启首轮的完整validation评估

**来源32维模型原生合法率为3180/3495（90.99%）；本轮best为3171/3495（90.73%），停训前last为3172/3495（90.76%）。** 两个checkpoint预先固定并全部报告；没有根据OCC结果挑权重。

本轮validation total loss未下降，但原始几何确实改善。last相对来源的面CD²下降27.52%，边下降17.85%。面在3071/3495件上更准，边在2885/3495件上更准；这不是少量离群对象拉低均值。合法率没有提高。在共同合法且双方可测的2928件中，last最终STEP边CD²下降约1.75%，面基本持平（+0.028%）。raw与STEP采用不同采样方式及可测集合，不能用这两个降幅之差归因后处理抹掉了多少收益。

## 完整结果

CD²为两个方向最近点平方距离均值之和，下表以×10⁻⁵显示，越小越好；不是欧氏距离或毫米。

| checkpoint | 原始面CD² | 原始边CD² | face-pair完全正确 | micro F1 | 原生合法solid |
|---|---:|---:|---:|---:|---:|
| 来源32D best42471 | 2.633866 | 5.969163 | 3393/3495 | 0.99860422 | 3180/3495（90.99%） |
| Conditional best415 | 2.014066 | 5.645720 | 3397/3495 | 0.99860935 | 3171/3495（90.73%） |
| Conditional last5395 | 1.908992 | 4.903703 | 3397/3495 | 0.99855926 | 3172/3495（90.76%） |

last的完全正确face-pair对象数增加，但micro F1略降：FP从179到199，FN从99到88，不能概括为拓扑所有指标都改善。

| 相对来源32D | 共同合法 | 救回 | 新增失败 | 共同失败 | 净变化 |
|---|---:|---:|---:|---:|---:|
| best415 | 3137 | 34 | 43 | 281 | -9件（-0.258个百分点） |
| last5395 | 3130 | 42 | 50 | 273 | -8件（-0.229个百分点） |

全部3495件都保留在合法率分母里。原始3505件中10件属于历史数据合同排除，沿用原清单；对应原3505分母合法率如下。

| checkpoint | /原3505 | STEP存在 | STEP可读 | STEP几何可测 | 合法且几何可测 |
|---|---:|---:|---:|---:|---:|
| shared_curve_uv | 90.73% | 3495 | 3494 | 3308 | 3052 |
| conditional_best415 | 90.47% | 3495 | 3495 | 3319 | 3055 |
| conditional_last5395 | 90.50% | 3495 | 3494 | 3316 | 3056 |

## 共同合法且双方几何可测的STEP比较

仅比较同一组对象。每个候选对应的共同可测集合不同，因此表中两行来源均值也可能不同。CD²仍以×10⁻⁵表示；STEP采样与raw采样不同，二者数值不能直接相减。

| 候选 | 配对对象数 | 来源面 | 候选面 | 来源边 | 候选边 |
|---|---:|---:|---:|---:|---:|
| best415 | 2930 | 47.515704 | 47.277941 | 9.690827 | 9.073555 |
| last5395 | 2928 | 47.411049 | 47.424378 | 9.683820 | 9.514483 |

## 固定口径与验证

同一DeepCAD validation 3495件，清单SHA256 `ae607bfe5499e919d1e68a5110c0c211daf1158aedbe57a4756eda99f6f3f0ed`，未使用test。142各模型用自身encoder重编码、posterior mean、FP32、无autocast/TF32、identity、不增强、概率>0.5、不去重。decoder未读取GT拓扑/UV。这是CAD输入的VAE重建，不是照片重建评估。

OCC沿用来源模型的11个生产post文件，逐SHA相同；`recover_v1`、optimized、优化200步/既有自适应600、drop_num=0、每件整个子进程180秒预算不变。主合法率是在保存STEP后，`read.precision.mode=0`重读，as-read为SOLID且BRepCheck通过；成功标记、Compound、文件存在和0.1兼容容差检查都不替代它。合法确认后的几何缺测/超时仍保留为合法。

两个导出各3495/3495成功，所有内部几何/拓扑状态均ok，输入CAD SHA与来源逐件一致，回传raw SHA逐件一致。最终额外独立重读两组STEP，检查每件native合法性与runner记录一致。配对四格、完整终态、合法数与几何可测集合由独立聚合重新计算。

187的`/mnt/d`在评估开始时发生ext4 journal阻塞。两次最初启动停在依赖预加载，均尚未派发任何CAD，没有消耗单件尝试。为避免存储故障污染180秒预算，将raw、GT STEP、来源缓存/终态和原始Python环境逐SHA复制到根盘；保持PyTorch2.4.1/cu124、PythonOCC7.8.1.1，无重新安装。预加载检查证实没有从`/mnt/d`映射共享库。实际使用128可用逻辑CPU，两组各32 workers并行，每worker单线程；CPU负载记录保留。执行存储与并发资源变化不能用于速度比较。详见EXECUTION与validation记录。

## 结论边界

这轮仍是`edge_feature=1`版本。总loss包含特征匹配，不能据total loss上升认定曲线几何未改善；训练历史审计显示从epoch0到12，去feature后的validation loss下降约2.44%，最终边XYZ项下降约4.38%。实际旧run的`metadata/hydra/config.yaml:18`与rank0启动日志均确认`vae_validation_inference:false`：W&B validation使用posterior采样z，本报告固定用posterior mean。训练时loss的监督/聚合方式也不同，因此不能把两个口径的数值或降幅直接对应。

来源best42471历史上冻结encoder，本轮Conditional则全模型解冻。本次比较同时改变注意力结构、encoder是否训练及训练预算，不能将全部变化唯一归因某一loss或attention。关闭feature后的新run属于另一个实验，本报告不预判其结果；当前结果也不足以声称大幅突破边面可裁剪性瓶颈。

- 冻结协议：[SPEC.md](SPEC.md)
- 原始几何：[raw_summary.json](raw_summary.json)
- 完整汇总：[summary.json](summary.json)
- 逐CAD压缩表：[per_cad.csv.gz](per_cad.csv.gz)
- 独立STEP审计及输入核验：`validation/`
- 完整运行产物：`/root/hola_eval/conditional_feature_on_20261029/`
