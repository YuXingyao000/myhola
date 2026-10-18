# 历史 HoLa 无条件生成基线

目的：先测指定旧无条件模型，在统一采样和后处理规则下建立参照；后续与真正接受无条件训练的新版模型比较。

## 旧模型的实际计算

每面latent仍是32维。它先经两层MLP投影到768维，再拼接256维零条件，成为1024维特征；
加上1024维时间嵌入后，进入24层Transformer。拼接后的1024维全部参与后续计算，不存在永久不参与计算的256维。

![旧无条件架构](report/architecture.png)

- 权重：`/mnt/d/ckpt/HoLa/Diffusion_uncond_1100k.ckpt`。
- 实际保存：1,000,000 steps / epoch324；完整checkpoint SHA256见[冻结协议](report/protocol.json)。
- 配置：epsilon prediction、L2、1000步linear DDPM、beta范围`[1e-4, 0.02]`、fixed_small variance。
- 无条件结构：1024维、16 heads、24层，FFN2048；每次从`[1,30,32]`高斯噪声开始。
- 使用checkpoint自带的冻结AE；该AE已在CPU逐值核对，与当前1119 AE的704个tensor相同。
- 旧代码`condition=null`在恢复构造时显式转换为空列表，表示不创建任何条件编码器；完整加载仍要求strict，不跳过未使用的classifier参数。

## 执行规则

完整协议见[PLAN](PLAN.md)。固定1000个编号和seed见[seeds.csv](report/seeds.csv)；每件一次采样，不按成功、面数或视觉效果补样。
FP32、TF32关闭、原生DDPM与去重逻辑；后处理为legacy、drop=0、每件180秒、16个CPU worker。
主有效率按STEP保存精度读取，检查solid类型及BRepCheck；0.1容差检查单列。
原论文§5.1使用过滤与不同有效性容差，不能将论文82.68%直接作为本轮应复现的数字。

## 状态与结果

完整严格加载、CPU前向、AE一致性和正式首件完整采样均已通过，见[恢复检查](report/runtime_validation.json)。
评估脚本的9项检查通过，见[评估检查](report/evaluation_validation.json)。8卡全量生成及CPU后处理正在执行；完成后在此报告全部1000件的结果。

## 复现命令

```bash
/mnt/d/anaconda3/envs/torch/bin/python \
  checkpoints/diffusion/uncond_baseline_20261018/run.py --phase prepare

/mnt/d/anaconda3/envs/torch/bin/python \
  checkpoints/diffusion/uncond_baseline_20261018/run.py \
  --phase all --gpus 0 1 2 3 4 5 6 7
```

推理与后处理可分开调用`--phase infer`、`--phase post`、`--phase summarize`。
`--limit 1`仅限制本次执行首件，预定分母仍为1000；同协议已完成样本只读取，算法失败不重试。
指定`--root /absolute/path/to/new_run`可另存完整运行，协议和seed保持相同。完成汇总后生成图表：

```bash
/mnt/d/anaconda3/envs/torch/bin/python \
  checkpoints/diffusion/uncond_baseline_20261018/visualize.py
```

## 结果的适用范围

本轮关注生成覆盖、实体有效性和面边数量分布。随机图库使用预定前24件，包括失败格子。
数量分布和图库不能代替正式形状覆盖率、新颖性或工程精度评价；本轮没有逐件GT，也不计算单图重建误差。
“原生合法solid”按STEP中已经保存的容差读取；旧后处理写出的容差本身可能较宽，该指标不代表高制造精度。
本轮没有训练新版无条件模型，不能给出新旧架构的性能结论。

参考：[HoLa原论文](https://arxiv.org/html/2504.14257v1)、[前次SVR比较](../svr_comparison_20261017/README.md)。
