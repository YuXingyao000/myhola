# 运行状态

2026-10-15（Asia/Shanghai）。正式训练已达到8小时上限后正常退出，退出码0，best与last均已保存。本轮完整评估及失败诊断已完成，冻结协议见 [evaluation_20261015/PLAN.md](evaluation_20261015/PLAN.md)，结果与研究判断见 [evaluation_20261015/NOTE.md](evaluation_20261015/NOTE.md)。

- 工作树：`/mnt/d/python_curve_eval`，分支 `codex/curve-point-feedback`。
- 初始化：上一轮 conditioned epoch157/step57354；新 optimizer，不延续验证脚本的临时更新。
- 本次仅新增最终曲线逐点几何反馈，27,139参数；原encoder冻结，Intersection容量、已有conditioning、loss权重不变。
- 配置入口：`train_vae_intersection_point_feedback`；8 GPU × batch16、bf16-mixed、Adam lr2e-5、seed42，8小时Timer。
- [150项自动检查、真实数据与多卡验证](validation/REPORT.md)全部通过。
- 正式源代码提交：`04406ac11b1d094e37fc5e250da8ac7705d93e7e`；运行metadata记录启动时工作树干净。
- run：`/mnt/d/python_curve_eval/checkpoints/intersection_refinement/point_feedback/20261014T153723.959456Z`。
- 启动：2026-10-14 23:37:24（北京时间）；结束：10月15日07:38:33，包含到期后完整验证与保存。
- last：epoch58／step21274；末轮训练完成220/363个batch后由Timer停止，随后完成55个验证batch。不能将末轮部分训练均值视为完整epoch。
- 验证最佳：epoch34／step12705，val/loss=0.03722964599728584。只使用此验证选择权重做本轮test；不按测试结果换权重。
- 历史tmux会话 `hola-point-feedback-20261014` 与主进程PID38284均已退出；日志在run的 `logs/console.log`，结束状态为 `metadata/exit.json`。
- [W&B运行](https://wandb.ai/whatseven777/hola-brep/runs/72b7bf3883c4)，每epoch保存验证最佳top3与last。

## 完整评估结论

固定2417件，直接比较上一轮conditioned与本轮point_feedback：optimized原生合法solid由2111降到2104（87.34%→87.05%），救回50、新增失败57；fit-only由1047到1048。原refinement历史结果为2113件。原始面XYZ MAE下降7.51%，边平方CD中位数下降7.55%，但边均值上升12.97%，存在少量严重长尾退步。当前不将本轮升级为默认checkpoint。

313失败和50救回均完成双侧完整诊断；失败末端为缺面50、拼接无solid255、保存检查失败7、重读不合法1。失败中229件无向面邻接全对；新增失败57件中49件邻接全对。几何贴合超阈值也常见于救回组，不能当成充分失败原因。见 [评估报告](evaluation_20261015/REPORT.md)、[全集fail census](failure_census_20261015/REPORT.md)、[离线图表与逐ID详情](failure_census_20261015/page/standalone.html)。两版额外训练曝光不同，不将变化单独归因于新模块。

## 本次边界处理

真实batch16检查确认原bbox头在BF16下偶发精确零scale，且可能是GT正候选。新反馈仅在预测scale精确0时不施加增量，以保持原bbox与local/global坐标合同；没有clamp正常的小尺度。记录见验证报告。此次没有同时把原bbox头改为FP32，也没有改变其loss；该数值现象可作为后续独立研究问题。
