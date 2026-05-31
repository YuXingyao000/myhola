# 防读取当前目录的configs
mkdir -p /tmp/qeval && cd /tmp/qeval && PYTHONPATH=/mnt/d/python \
    python -m src.brepnet.eval.quality_metrics \
    --condition-root /mnt/d/data/deepcad_v7_cond \
    --model-list /tmp/deepcad_v7_cond_all.txt \
    --sam2-checkpoint /mnt/d/data/checkpoints/sam2.1_hiera_large.pt \
    --use-ray --num-gpus 8 --actors-per-gpu 2 \
    --compute-dino --compute-clip \
    --topk-vis-dir /mnt/d/data/quality_topk_vis --topk 10 \
    --output /mnt/d/python/experiments/2026-5-30/dataset_evaluation/deepcad_v7_cond_quality_report.json