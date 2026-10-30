# 完整 validation 对照入口：实现与 CPU 验证

2026-10-24。状态：代码完成、预检通过；新模型正式生成/评估尚未启动。本次没有使用 GPU，没有进入几何训练。

## 实现

- `src/brepnet/eval/preference_compare.py` 增加 `--reuse-methods`。省略时仍复用两方法；显式 `baseline` 时只要求原 SVR checkpoint 相同，新 DPO checkpoint 可以不同。逐 CAD 在 CPU 核验 seed、输入、预测、评估协议、STEP 摘要后建立只读引用。全部原 SVR 已复用时，全部 GPU 分配新 DPO，CPU 后处理只处理新 DPO。
- `src/brepnet/eval/preference_ablation_report.py` 汇总两个 comparison run 和两个 audit。核验完整清单、原 checkpoint、采样/后处理/指标代码、输入与来源；输出三个模型的固定分母统计及新减旧的配对差值和 CAD bootstrap CI。缺测不填零，同时报告各自可测集、共同可测集和单侧缺测。
- `src/brepnet/eval/preference_ablation.py` 顺序运行 comparison → observation → report；记录协议、命令、Git diff、阶段日志和文件摘要。独占锁防止同目录并发；中断时终止子进程组。完成阶段只通过 CPU 校验后跳过。仅初始化为空的阶段保留到 `interrupted_initializations/` 后重新开始；已有产物但缺协议时拒绝猜测恢复。
- `run_evaluation.sh` 固定本次源路径、step5000、3505 validation、seed42、8 GPU、16 CPU workers 和原测量参数，供用户手动启动。

## 自动验证

使用源机现有 `/mnt/d/anaconda3/envs/torch/bin/python`，显式关闭 CUDA。最终 **65 项通过（3.62 秒）**：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /mnt/d/anaconda3/envs/torch/bin/python -m pytest -q \
  src/brepnet/test/test_preference_compare_reuse.py \
  src/brepnet/test/test_preference_ablation_report.py \
  src/brepnet/test/test_preference_ablation_workflow.py \
  src/brepnet/test/test_preference_report.py \
  src/brepnet/test/test_diffusion_audit.py
```

覆盖选择性复用、全卡分配、失败终态、来源变更拒绝、原报告兼容、全分母和共同集、CI 方向及可重复性、非有限值缺测、check-only 不创建产物、输出锁、跨 workflow receipt 拒绝、完成阶段跳过、源/输出篡改、中断和空初始化恢复，以及短 CPU 子进程的进程组清理。shell 语法与 `git diff --check` 通过。

## 真实资产只读验证

1. 旧完整 comparison 的 3505 清单及源协议可复用；抽查普通样本 `00000178` 和链式引用旧 100 件 run 的 `00938923`，输入/输出/STEP 摘要均通过。
2. 对真实旧 comparison/audit 执行“旧模型对自身”的全量汇总与实际导出，bootstrap 10000 次：3505 CAD、三方法标签共 10515 CSV 行、66603 个来源文件；全部配对差值和 CI 均为 0。PNG 2340×846 实际查看、SVG 解析、README、JSON 和来源指纹核验通过。完整导出耗时 57.77 秒，只写 `/tmp/preference_ablation_old_self_report_20261024T104500.151215Z`。这是接口自检，不是新 DPO 效果结果，也没有重新计算几何指标。
3. 手动脚本加 `--check-only`，对真实新 checkpoint、原 checkpoint、完整 split、训练 CAD 不重叠、旧 run/audit、当前测量源码与运行库进行检查，全部通过。checkpoint 使用 CPU mmap 只读取元数据；推理入口的完整权重/冻结参数严格加载仍留在正式运行时执行。

紧凑证据保存在 [EVALUATION_PREFLIGHT.json](EVALUATION_PREFLIGHT.json)，包含权重/源码摘要、检查范围和实际输出目录未创建的状态。当前环境是源机 torch2.4.1+cu124 / Python3.10.13 / PythonOCC7.8.1.1；未套用第二台服务器环境。

## 结论边界

可以手动开始固定协议的完整对照。65 项 CPU 检查与旧数据自检证明调度、复用和汇总接口可用；不能据此宣称新模型合法率或可见重建质量提升。保持 K=1，不能评估同图多样性。完成正式对照后再决定几何辅助训练。
