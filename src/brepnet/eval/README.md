# Evaluation (`src/brepnet/eval/`)

## 入口

**唯一入口是 `run.py`**：

```bash
python -m src.brepnet.eval.run \
    --pred-root <后处理输出目录> \
    --gt-root <GT数据目录> \
    --split-list <model list文件> \
    --metrics <逗号分隔的指标>
```

## 常用命令

### 几何重建指标 (Condition: CD + F1)

```bash
# 标准评估：GT 固定使用 identity-first cube24 的 0 号旋转
python -m src.brepnet.eval.run \
    --pred-root /mnt/d/data/results/post_output \
    --gt-root /mnt/d/data/deepcad_v6 \
    --split-list src/brepnet/data/list/test.txt \
    --metrics condition \
    --use-ray --num-cpus 16
```

### Validity (STEP 文件是否合法)

```bash
python -m src.brepnet.eval.run \
    --pred-root /mnt/d/data/results/post_output \
    --metrics validity \
    --use-ray --num-cpus 16
```

### 全部指标一起跑

```bash
python -m src.brepnet.eval.run \
    --pred-root /mnt/d/data/results/post_output \
    --gt-root /mnt/d/data/deepcad_v6 \
    --split-list src/brepnet/data/list/test.txt \
    --metrics condition,validity,complexity \
    --use-ray --num-cpus 16
```

### Uniqueness (不需要 GT)

```bash
python -m src.brepnet.eval.run \
    --pred-root /mnt/d/data/results/post_output \
    --metrics unique \
    --use-ray
```

### FLUX 数据集质量 (单独脚本)

```bash
cd /tmp && PYTHONPATH=/mnt/d/python python -m src.brepnet.eval.quality_metrics \
    --condition-root /mnt/d/data/deepcad_v6_cond \
    --model-list /mnt/d/python/src/brepnet/data/list/test.txt \
    --sam2-checkpoint /mnt/d/model/sam2.1_hiera_large.pt \
    --compute-dino --compute-clip \
    --output quality_report.json
```

注意：必须从 `/tmp` 运行（避免 configs/ 和 SAM2 Hydra 冲突）。

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--pred-root` | 后处理输出目录 (每个模型一个子文件夹, 含 recon_brep.step) | **必填** |
| `--gt-root` | GT 数据目录 (每个模型含 normalized_shape.step) | condition 必填 |
| `--split-list` | model list 文本文件 | 空=扫描 pred-root 下所有文件夹 |
| `--metrics` | 逗号分隔: condition, validity, complexity, unique, lfd | condition |
| `--use-ray` | 启用 Ray 并行 | 关 |
| `--num-cpus` | Ray CPU 数 | 16 |
| `--from-scratch` | 忽略已有结果，全部重算 | 关 |
| `--only-valid` | condition 统计时只包含 valid solid | 关 |

## Rotation 注意事项

运行时已统一为 identity-first cube24：`rotation_id=0` 就是 identity。评估入口不再提供
`--rotation-policy` 或 `--rotation-id`，condition 指标固定按 identity GT 计算。

## 输出文件

评估完成后 `pred-root/` 下会生成：

```
pred-root/
├── eval_condition.csv           # 每个模型的 CD/F1 指标
├── eval_condition_summary.json  # 汇总统计
├── eval_validity.csv
├── eval_validity_summary.json
├── eval_complexity.csv
├── eval_complexity_summary.json
├── eval_unique_summary.json
├── eval_summary.json            # 全部指标合并
└── {model_id}/
    ├── eval_condition.npz       # 单模型详细结果
    ├── eval_validity.npz
    └── eval_error.txt           # 如果出错
```

## pred-root 期望的目录结构

```
pred-root/{model_id}/
├── recon_brep.step         # 后处理生成的 STEP 文件 (主要输入)
├── recon_brep.stl          # valid solid 时才有
└── success.txt             # valid solid 标记
```

这些文件由 `python -m src.brepnet.post.construct_brep` 生成。

## 完整流水线 (Inference → Post → Eval)

```bash
# 1. Inference: 生成 raw npz
python -m src.brepnet.train --config-name ... trainer.evaluate=true \
    trainer.resume_from_checkpoint=/path/to/last.ckpt \
    trainer.test_output_dir=/path/to/inference_output

# 2. Post-processing: npz → STEP
python -m src.brepnet.post.construct_brep \
    --data_root /path/to/inference_output \
    --out_root /path/to/post_output \
    --use_ray --use_cuda --num_cpus 16

# 3. Evaluation: STEP vs GT
python -m src.brepnet.eval.run \
    --pred-root /path/to/post_output \
    --gt-root /mnt/d/data/deepcad_v6 \
    --split-list src/brepnet/data/list/test.txt \
    --metrics condition,validity,complexity \
    --use-ray --num-cpus 16
```

## 目录结构

```
src/brepnet/eval/
├── run.py                  # ← 唯一入口
├── protocol.py             # EvalSample, iter_eval_samples
├── parallel.py             # Ray wrapper
├── io.py                   # save/load helpers
├── metrics/
│   ├── condition.py        # CD, F1 (Face/Edge/Vertex) + rotation handling
│   ├── validity.py         # STEP solid validity (BRepCheck)
│   ├── complexity.py       # Face/Edge/Vertex count, cyclomatic complexity
│   ├── uniqueness.py       # VF2 graph isomorphism
│   ├── lfd.py              # Light Field Descriptor
│   └── point_cloud_set.py  # MMD, COV, JSD (分布指标)
├── quality_metrics.py      # FLUX 数据集质量 (SAM2 IoU, DINO, CLIP)
├── adapters/               # Point2CAD, ComplexGen, NVDNet 格式适配
├── legacy/                 # ⚠️ 废弃代码, 仅作备份, 不要使用
├── METRICS_TAXONOMY.md     # 指标分类与论文对应关系
└── README.md               # 本文件
```

## 不要使用

- `legacy/` 下的所有脚本 — 接口不匹配当前 pipeline
- `eval_cd.sh` / `eval_condition.sh` — 旧 shell 脚本，路径硬编码
- `sample_points.py` / `sample_points_gt.py` — 旧点云采样，路径硬编码
