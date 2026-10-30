#!/usr/bin/env bash
# 原始发布代码 /mnt/d/HoLa-Brep 的后处理与合法性检查，输入为 2026-10-11 legacy posterior-mean 预测（2417 件）。
set -u
PY=/mnt/d/anaconda3/envs/torch/bin/python
RAW=/mnt/d/python/checkpoints/joint_boundary/comparison_20261011/raw_predictions/posterior_mean/legacy
OUT=/mnt/d/python/checkpoints/baselines/hola_table5_20261030
cd /mnt/d/HoLa-Brep
for DROP in 0 3; do
  echo "=== drop_num=$DROP start $(date -u +%FT%TZ)"
  $PY -m construct_brep --data_root $RAW --out_root $OUT/post_drop$DROP --use_ray --num_cpus 120 --drop_num $DROP --from_scratch > $OUT/logs/construct_drop$DROP.log 2>&1
  echo "=== drop_num=$DROP construct done $(date -u +%FT%TZ) rc=$?"
  $PY -m eval.check_valid --data_root $OUT/post_drop$DROP > $OUT/logs/check_drop$DROP.log 2>&1
  echo "=== drop_num=$DROP check done $(date -u +%FT%TZ) rc=$?"
  tail -3 $OUT/logs/check_drop$DROP.log
done
