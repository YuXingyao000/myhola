# 运行状态

2026-10-14（Asia/Shanghai）。代码及验证已完成，正式8小时训练已启动，已观察到第14个训练step、loss有限；尚未完成首个epoch或训练后评估。

- 工作树：`/mnt/d/python_curve_eval`，分支 `codex/curve-point-feedback`。
- 初始化：上一轮 conditioned epoch157/step57354；新 optimizer，不延续验证脚本的临时更新。
- 本次仅新增最终曲线逐点几何反馈，27,139参数；原encoder冻结，Intersection容量、已有conditioning、loss权重不变。
- 配置入口：`train_vae_intersection_point_feedback`；8 GPU × batch16、bf16-mixed、Adam lr2e-5、seed42，8小时Timer。
- [150项自动检查、真实数据与多卡验证](validation/REPORT.md)全部通过。
- 正式源代码提交：`04406ac11b1d094e37fc5e250da8ac7705d93e7e`；运行metadata记录启动时工作树干净。
- run：`/mnt/d/python_curve_eval/checkpoints/intersection_refinement/point_feedback/20261014T153723.959456Z`。
- 启动：2026-10-14 23:37:24（北京时间），正式训练约23:37:54开始；预计10月15日07:38左右达到8小时预算，随后完成验证及保存。
- tmux会话：`hola-point-feedback-20261014`，训练主进程PID38284；终端日志在run的 `logs/console.log`，结束状态由启动器写入 `metadata/exit.json`。
- [W&B运行](https://wandb.ai/whatseven777/hola-brep/runs/72b7bf3883c4)，每epoch保存验证最佳top3与last。此处不是训练完成报告，也没有几何或solid提升结论。

## 本次边界处理

真实batch16检查确认原bbox头在BF16下偶发精确零scale，且可能是GT正候选。新反馈仅在预测scale精确0时不施加增量，以保持原bbox与local/global坐标合同；没有clamp正常的小尺度。记录见验证报告。此次没有同时把原bbox头改为FP32，也没有改变其loss；该数值现象可作为后续独立研究问题。
