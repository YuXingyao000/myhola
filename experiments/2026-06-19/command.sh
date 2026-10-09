cd /mnt/d/python

# 1. Train the real-photo baseline diffusion model without topology mask.
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m src.brepnet.train \
  --config-name train_diffusion_baseline_real \
  trainer.devices=8 \
  trainer.batch_size=64 \
  trainer.num_workers=8 \
  trainer.max_steps=100000 \
  trainer.output_dir=experiments/baseline \
  trainer.exp_name=diffusion_baseline_real_v7_cube24 \
  trainer.wandb.enabled=true \
  trainer.wandb.project=hola-brep \
  trainer.wandb.name=diffusion_baseline_real_v7_cube24 \
  dataset.raw_data_root=/mnt/d/data/deepcad_v7 \
  dataset.cached_latent_root=/mnt/d/data/ae_cache/1119_deepcad_aug1_11k_24 \
  dataset.condition_data_root=/mnt/d/data/deepcad_v7_cond \
  dataset.train_dataset=src/brepnet/data/list/deduplicated_deepcad_training_7_30.txt \
  dataset.val_dataset=src/brepnet/data/list/deduplicated_deepcad_validation_7_30.txt \
  dataset.test_dataset=src/brepnet/data/list/deduplicated_deepcad_testing_7_30.txt \
  dataset.real_photo_ratio=1.0 \
  dataset.validation_real_photo_ratio=1.0 \
  dataset.is_aug=false \
  dataset.is_overfit=false \
  dataset.load_topology=false \
  dataset.epoch_scale_factor=200 \
  model.autoencoder.checkpoint=/mnt/d/data/ae_checkpoints/1119_deepcad_aug1_11k.ckpt \
  model.topology_bias.enabled=false \
  hydra.job.chdir=false

cd /mnt/d/python

# 2. Generate full test predictions from the trained baseline checkpoint.
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m src.brepnet.train \
  --config-name train_diffusion_baseline_real \
  trainer.devices=1 \
  trainer.batch_size=8 \
  trainer.num_workers=8 \
  trainer.evaluate=true \
  trainer.resume_from_checkpoint=experiments/baseline/diffusion_baseline_real_v7_cube24/checkpoints/last.ckpt \
  trainer.output_dir=experiments/baseline \
  trainer.exp_name=diffusion_baseline_real_v7_cube24 \
  trainer.wandb.enabled=false \
  eval.enabled=true \
  eval.output_dir=experiments/baseline/diffusion_baseline_real_v7_cube24/test_outputs \
  dataset.raw_data_root=/mnt/d/data/deepcad_v7 \
  dataset.cached_latent_root=/mnt/d/data/ae_cache/1119_deepcad_aug1_11k_24 \
  dataset.condition_data_root=/mnt/d/data/deepcad_v7_cond \
  dataset.test_dataset=src/brepnet/data/list/deduplicated_deepcad_testing_7_30.txt \
  dataset.real_photo_ratio=1.0 \
  dataset.validation_real_photo_ratio=1.0 \
  dataset.is_aug=false \
  dataset.is_overfit=false \
  dataset.load_topology=false \
  model.autoencoder.checkpoint=/mnt/d/data/ae_checkpoints/1119_deepcad_aug1_11k.ckpt \
  model.topology_bias.enabled=false \
  hydra.job.chdir=false

cd /mnt/d/python

# 3. Post-process baseline predictions and run standard condition evaluation.
python -m src.brepnet.experiments.condition_debug.post_eval \
  --data-root experiments/baseline/diffusion_baseline_real_v7_cube24/test_outputs \
  --post-root experiments/baseline/diffusion_baseline_real_v7_cube24/test_post \
  --gt-root /mnt/d/data/deepcad_v7 \
  --list src/brepnet/data/list/deduplicated_deepcad_testing_7_30.txt \
  --num-cpus 32 \
  --timeout 180 \
  --force

cd /mnt/d/python

# 4. Run baseline validity and visible-face condition evaluation.
python -m src.brepnet.eval.run \
  --pred-root experiments/baseline/diffusion_baseline_real_v7_cube24/test_post \
  --gt-root /mnt/d/data/deepcad_v7 \
  --split-list src/brepnet/data/list/deduplicated_deepcad_testing_7_30.txt \
  --metrics validity,visible_condition \
  --visibility-root /mnt/d/data/deepcad_v7_visibility \
  --visibility-view-id 0 \
  --visible-threshold 0.05 \
  --use-ray \
  --num-cpus 32 \
  --timeout 180 \
  --from-scratch

cd /mnt/d/python

# 5. Run the two-step pipeline: 06-18 CVAE topology cache -> baseline diffusion.
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 python experiments/2026-06-19/cvae_diffusion_two_step.py \
  --diffusion-checkpoint experiments/baseline/diffusion_baseline_real_v7_cube24/checkpoints/last.ckpt \
  --cvae-checkpoint experiments/2026-06-18/outputs_cvae_visibility/best.pt \
  --output-dir experiments/baseline/cvae_visibility_diffusion_outputs \
  --topology-cache-dir experiments/baseline/cvae_visibility_topology_cache \
  --split test \
  --raw-root /mnt/d/data/deepcad_v7 \
  --condition-root /mnt/d/data/deepcad_v7_cond \
  --cached-latent-root /mnt/d/data/ae_cache/1119_deepcad_aug1_11k_24 \
  --ae-checkpoint /mnt/d/data/ae_checkpoints/1119_deepcad_aug1_11k.ckpt \
  --test-list src/brepnet/data/list/deduplicated_deepcad_testing_7_30.txt \
  --batch-size 2 \
  --num-workers 8 \
  --topology-bias-scale 2.0

cd /mnt/d/python

# 6. Post-process two-step predictions and run standard condition evaluation.
python -m src.brepnet.experiments.condition_debug.post_eval \
  --data-root experiments/baseline/cvae_visibility_diffusion_outputs \
  --post-root experiments/baseline/cvae_visibility_diffusion_post \
  --gt-root /mnt/d/data/deepcad_v7 \
  --list src/brepnet/data/list/deduplicated_deepcad_testing_7_30.txt \
  --num-cpus 32 \
  --timeout 180 \
  --force

cd /mnt/d/python

# 7. Run two-step validity and visible-face condition evaluation.
python -m src.brepnet.eval.run \
  --pred-root experiments/baseline/cvae_visibility_diffusion_post \
  --gt-root /mnt/d/data/deepcad_v7 \
  --split-list src/brepnet/data/list/deduplicated_deepcad_testing_7_30.txt \
  --metrics validity,visible_condition \
  --visibility-root /mnt/d/data/deepcad_v7_visibility \
  --visibility-view-id 0 \
  --visible-threshold 0.05 \
  --use-ray \
  --num-cpus 32 \
  --timeout 180 \
  --from-scratch
