# 完整 validation：原 SVR 与 DPO

用户在100件对照后要求跑完剩余validation。冻结协议见[PLAN.md](PLAN.md)，
实现检查见[VALIDATION.json](VALIDATION.json)。本轮只扩大样本范围，不修改模型或评估。

## 输入与样本量

完整清单：`src/brepnet/data/list/deduplicated_deepcad_validation_7_30.txt`，
SHA256 `b6403d9b235db7ebeaac76ba3a9d076440f149863ac692ae938129e20404ee32`。

- 3505个唯一CAD；与46627件train清单交集0。
- 原100件全部包含；它们是固定ID哈希排序前100，不依据结果筛选。
- 新增3405件、每模型每图1次，共6810次新生成；完整集每模型3505件。
- 全部3505件的`imgs.npz`、`data.npz`及`normalized_shape.step`均存在。
  这是文件存在性核对，实际解压、生成及评估中的错误仍须记录。

原100件run为`checkpoints/diffusion/preference_eval/20261023T062400Z`，保持原样。
新run只读引用其原始输出及评估。旧100件两方法各4件几何评估错误也原样保留，
不能借扩展重新尝试。双方权重、图片/GT根目录、seed、采样和后处理合同及影响输出的源码
保持一致，编排与汇总改动单独记录。仍使用原HoLa的VAE和最终DPO 5000步。

## 运行

目标run：`/mnt/d/python/checkpoints/diffusion/preference_eval/20261023T090834Z`。
启动脚本会生成完毕后自动做CPU后处理与指标计算，再汇总完整3505件。
GPU 0/2/4/6运行原模型；1/3/5/7运行DPO；后处理CPU并发16。

```bash
bash checkpoints/diffusion/preference/validation_full_20261023/run.sh \
  /mnt/d/python/checkpoints/diffusion/preference_eval/20261023T090834Z
```

这是新run命令，已有run不得重复执行。进程启动状态另记于同目录`LAUNCH.json`。
运行中查看`status.json`及`logs/`，总日志为run路径加`.log`，退出码加`.exit_code.txt`。
完整结果将写入run的`report/summary.json`、`report/per_object.csv`、`report/comparison.png`。
完成前不把旧100件结果当作完整集结果。

## 验证

10个汇总检查通过，真实旧100件重新计算的均值、置信区间、合法性转移完全一致。
真实复用检查覆盖400个记录引用，重复初始化保留相同引用；checkpoint或指标源码变化
会被拒绝。新的bootstrap按批分配内存，保持原10,000次配对CAD抽样序列。
新run的耗时仅对应本轮新增生成/评估及完整汇总，不代表3505件从零运行的耗时。
