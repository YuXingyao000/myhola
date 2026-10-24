# 关闭 edge_feature 的 64D VAE：固定 validation OCC 验证

## Task
验证固定 30,000-step checkpoint 的原生 OCC 合法率、逐对象救回与退步及几何保真。对照为初始化来源 shared-curve + UV 32D best42471；复用 2026-10-20 已完成流程，不改后处理规则，不启动训练。

## Assumptions
训练后64D encoder必须自行编码同一CAD；不能复用32D latent。此前 mean 几何指标改善并不预设 OCC 改善。结论限于 seed=42 这一组运行。3495件为历史合同接受集，原3505件中的10件排除继续披露。

## Inputs
- 固定 split：`checkpoints/joint_boundary/validation/val_contract_audit/accepted.txt`；3495唯一ID，SHA256 `ae607bfe5499e919d1e68a5110c0c211daf1158aedbe57a4756eda99f6f3f0ed`。
- CAD `/mnt/d/data/deepcad_v7`；UV `/mnt/d/data/deepcad_v7_uv/v1`。
- 候选：142 `checkpoints/latent_expansion/d64_no_edge_feature/20261023T034042Z/weights/last.ckpt`，global_step=30000，SHA256 `4262f4998fe047e19c3ca21a835d74c183b7430f7a656c709ec8ed045262ada4`。
- 对照：`checkpoints/shared_curve_uv/evaluation_20261020/best`；checkpoint SHA256 `ce749416c1635dda1563161b971bd4260d064cfdb0c31f1fa9d06ad0e70961ff`；原生合法3180/3495。

## Outputs
本目录保存SPEC、运行脚本、命令/配置/源码与checkpoint指纹、逐件原始预测、OCC终态、CSV、汇总JSON、整体统计图及中文REPORT。大体积预测/STEP/缓存/日志不提交Git。

## Processing Protocol
1. 142运行严格加载的候选encoder + posterior mean，identity输入，FP32，无autocast/TF32，seed42，邻接threshold=0.5，不做face去重；逐件导出预测。对原始几何指标与此前已完成3495件mean诊断核验。
2. 187原评估环境 CPU 运行旧 `tools/vae_compare_post.py`；实验局部wrapper仅新增明确方法名 `no_edge_feature64`，允许 `shared_curve_uv` 为baseline。生产post源码11文件与基线逐一核验一致。
3. 固定 optimized/recover_v1，max_optimize_iter=200，既有adaptive上限600，drop_num=0，use_cuda=False，32 workers，每worker单线程；每件整个子进程包括几何测量硬预算180秒。无GT修复/选择，无新增重试。
4. 基线raw、终态及GT采样cache只读复用，校验raw SHA与协议fingerprint。候选独立输出，不覆盖既有产物。
5. raw失败测量复用旧函数与签名 `704227ad75de74bd531ae8641c4060818dcfac02089bf6f43053eb95b9c4c77a`；阈值0.001/0.002/0.006/0.01/0.08，主阈值0.006。最多8 CPU workers测量。

## Metrics
- 主指标：保存STEP后read.precision.mode=0原生重读，类型为TopAbs_SOLID且BRepCheck通过。STEP存在、可读、success及兼容0.1容差合法另列。
- 3495和原3505两个分母；source32→candidate共同合法/救回/退步/共同失败四格。
- 最终失败阶段互斥统计、最远到达阶段；raw owner三角网格P95 gap、twin、端点代理、face-pair错误与GT网格误差，作为可重叠观察而非已证因果。
- raw全量CD²使用此前诊断并核验导出一致；post几何使用双方共同原生合法且双方几何可测同一集合，face/edge CD² mean/median/P90，并报告缺测。另列共同可测集合。
- STEP采样沿用每面1024面积加权mesh点（deflection=.01/angular=.1）、每边128原生参数均匀点；无对齐/缩放，双向最近邻平方距离均值之和。退化边用真实顶点，不以原点替代。
- 原生内部最大存储容差及耗时/超时单列；内部容差不等于几何精度，历史运行耗时不能推导严格速度差异。

## Decision Rule
报告全量合法率差值和逐件救回/退步，再检查共同合法集合的几何变化。可判断该固定模型是否改善下游合法性；不能仅凭成功子集、raw loss或单seed推导架构无问题，也不能将32D对64D比较归因为feature loss唯一因素。

## Edge Cases
失败、缺失、异常、超时保留在预定分母；合法确认后的几何测量超时仍计合法、几何缺测。unknown不归为阴性。无静默fallback、删除失败、自动改容差或按结果选checkpoint。test不使用。

## Validation / Sanity Checks
核验split/checkpoint/源码hash、3495唯一覆盖和完整终态；source323180基准可复现；四格和3495，互斥阶段和失败总数；合法与几何可测状态分离；复用缓存签名匹配。先进行真实小批量导出验证和原指标核对，再全量执行。

## Implementation Constraints
只在本实验目录新增适配代码，保留用户dirty工具及其他研究工作；不修改生产模型/loss/post。使用已安装环境，不抢占187的diffusion GPU。冻结测量定义，不根据validation个例改本轮规则。每阶段记录命令、Git commit/dirty来源、时间与版本。

## Deliverables
完整3495件OCC结果、配对CSV、汇总与核验JSON、总体图、中文REPORT及明确研究判断；报告实际完成状态与局限，不以启动运行代替完成。
