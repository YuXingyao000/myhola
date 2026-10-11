# Intersection refinement 测试

本次评估固定 epoch 10 / step 3993 的验证最佳权重，与旧 HoLa light 对照。协议见 [PLAN](PLAN.md)，训练状态见 [training_audit.json](training_audit.json)。完整结果见 [REPORT.md](REPORT.md)；以下命令没有启动训练，也不依据 test 更换 checkpoint。

## 可复现命令

在 `/mnt/d/python` 使用本机已验证的 `torch` 环境。原始运行输出保留，完整重跑应选新的 comparison root。需要相同的原版参考目录和已挂载资产。

```bash
conda activate torch
cd /mnt/d/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
```

推理入口如下。两个 mode 分别为 `posterior_mean` 和 `archived_latents`；每个 mode 对 shard-index=0…7 / cuda:0…7 并行运行，等待所有分片完成后进入指标步骤。固定ID分片没有 distributed sampler 补齐。

```bash
python -m tools.intersection_evaluate_inference \
  --checkpoint 'checkpoints/intersection_refinement/frozen/20261011T123836.485257Z/weights/epoch=0010-step=00003993-val_loss=0.039476.ckpt' \
  --legacy-checkpoint /mnt/d/data/ae_checkpoints/1119_deepcad_aug1_11k.ckpt \
  --reference-root checkpoints/joint_boundary/comparison_20261011 \
  --data-root /mnt/d/data/deepcad_v7 \
  --split-list checkpoints/joint_boundary/validation/test_contract_audit/accepted.txt \
  --output-root checkpoints/intersection_refinement/evaluation_20261011 \
  --mode posterior_mean --num-shards 8 --shard-index 0 --device cuda:0
```

`raw_predictions/<mode>/legacy` 为原比较中同名目录的只读引用（本次使用目录 symlink），不是重新生成或改写的原版预测。新入口只写 intersection；同时逐件核对原协议、latent和原输入source哈希。相同协议中断续跑加 `--resume`；协议变化必须保留旧产物并使用新目录，不绕过检查。

对两个 mode 分别计算 raw 指标：

```bash
python -m tools.vae_compare_metrics \
  --candidate-method intersection \
  --comparison-root checkpoints/intersection_refinement/evaluation_20261011 \
  --data-root /mnt/d/data/deepcad_v7 \
  --split-list checkpoints/joint_boundary/validation/test_contract_audit/accepted.txt \
  --mode posterior_mean --workers 16
```

只对 posterior mean 执行两种后处理，mode 分别为 `fit_only`、`optimized`。旧 legacy 终态以原协议只读复用；脚本逐件核对 raw SHA，并检查全部后处理算法文件、采样、预算和清单。新模型严格使用相同后处理算法。

```bash
python -m tools.vae_compare_post \
  --candidate-method intersection --methods intersection \
  --reference-root checkpoints/joint_boundary/comparison_20261011 \
  --comparison-root checkpoints/intersection_refinement/evaluation_20261011 \
  --data-root /mnt/d/data/deepcad_v7 \
  --split-list checkpoints/joint_boundary/validation/test_contract_audit/accepted.txt \
  --mode optimized --workers 32 --timeout 180 \
  --max-optimize-iter 200 --original-test-count 2424
```

所有结果完整后：

```bash
python -m tools.intersection_evaluate_report \
  --comparison-root checkpoints/intersection_refinement/evaluation_20261011
```

报告包含 raw 几何/连接、fit-only/optimized solid、两方法共同可计算的 STEP 几何、原失败→新成功和原成功→新失败计数，以及固定 diffusion latent 的单独结果。图表在 `visualizations/`，每件状态与协议在 `metadata/`；失败和超时留在固定分母。

## 实现验证

39 项推理/指标/后处理测试和 16 项报告合同测试通过，见 [validation_summary.json](validation_summary.json)。测试覆盖新权重加载和原输入复用、方法名参数化、有向半边诊断、原版后处理复用条件与拒绝混用。真实两件 GPU 导出也通过。数值审计或批量执行中的中断会保留记录并在最终报告中说明；这些检查不代替完整测试结果。

逐件 CSV 的独立重算审阅见 [independent_review.json](independent_review.json)，核对输入哈希、固定分母、配对统计、实体状态、超时阶段及极端案例对均值的影响。
