# 最终曲线逐点反馈：配置与运行

2026-10-14 新增；研究假设、固定范围和比较规则见 [PLAN.md](PLAN.md)。实现验证已通过，正式运行状态见 [NOTE.md](NOTE.md)，尚无训练效果结论。

## 新增代码与配置

| 位置 | 本次新增或调整 |
|---|---|
| `src/brepnet/models/blocks.py` | 提取原17维读取为无参数helper；新增 `CurvePointFeedback`；仅最终曲线使用逐点XYZ增量，local/global输出一起更新 |
| `src/brepnet/models/vae.py` | 新增 `init_curve_feedback_checkpoint` 严格迁移，保留已训练的conditioning及全部旧权重 |
| `configs/model/vae_intersection.yaml` | `curve_point_feedback` 默认关闭，hidden_dim128 / feature_dim64 |
| `configs/train_vae_intersection_point_feedback.yaml` | 本轮装配配置；启用已有conditioning及新反馈，保持原训练参数 |
| `configs/trainer/default.yaml`、`src/brepnet/train.py` | 通用可选 `max_time`（默认null）；resume清除新初始化字段，无新增模型专属trainer |

27,139个新增参数（6个tensor），全模型从181,965,981增至181,993,120。新增末层为零；关闭开关不增加state keys。输入仅有预测几何/特征，无GT前向依赖。训练仍更新原decoder与新增分支，encoder/posterior冻结。

## 新运行

以下路径明确指向本线程隔离工作树，避免其他任务切换 `/mnt/d/python` 分支影响运行：

```bash
cd /mnt/d/python_curve_eval
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 /mnt/d/anaconda3/envs/torch/bin/python -m src.brepnet.train \
  --config-name train_vae_intersection_point_feedback \
  trainer.output_dir=/mnt/d/python_curve_eval/checkpoints \
  trainer.devices=8 trainer.num_workers=2 \
  "model.init_curve_feedback_checkpoint='/mnt/d/python/checkpoints/intersection_refinement/curve_conditioning/20261013T083729.740439Z/weights/epoch=0157-step=00057354-val_loss=0.038204.ckpt'" \
  dataset.raw_data_root=/mnt/d/data/deepcad_v7 \
  dataset.train_dataset=/mnt/d/python_curve_eval/checkpoints/joint_boundary/validation/train_contract_audit/accepted.txt \
  dataset.val_dataset=/mnt/d/python_curve_eval/checkpoints/joint_boundary/validation/val_contract_audit/accepted.txt \
  dataset.test_dataset=/mnt/d/python_curve_eval/checkpoints/joint_boundary/validation/test_contract_audit/accepted.txt
```

默认8小时预算由Lightning Timer控制，允许当前step、验证和保存结束；不是强杀进程。每epoch保存val/loss最佳3份及last。完整新模型加载使用 `model.checkpoint`，恢复完整训练状态使用 `trainer.resume_from_checkpoint`，后者清除初始化来源并严格恢复。若续训需要额外8小时，应显式调整预算并核对Timer累计时间语义。

真实数据检查和运行状态会记录在本目录的 `validation/`、`NOTE.md`；权重/日志/metadata保留本地并由Git忽略。
