# Train
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m src.brepnet.train --config-name train_diffusion_white \
    trainer.check_val_every_n_epoch=1 \
    trainer.num_workers=32 \
    trainer.batch_size=64 \
    trainer.precision=bf16-mixed \
    dataset.name=Diffusion_dataset \
    dataset.latent_root=/mnt/d/data/ae_cache/1119_deepcad_aug1_11k \
    dataset.condition_root=/mnt/d/data/deepcad_v6_cond \
    dataset.train_dataset=/mnt/d/python/src/brepnet/data/DataGeneration/test_list/train.txt \
    dataset.val_dataset=/mnt/d/python/src/brepnet/data/DataGeneration/test_list/train.txt \
    dataset.test_dataset=/mnt/d/python/src/brepnet/data/DataGeneration/test_list/train.txt \
    dataset.cached_condition=false \
    dataset.is_aug=1 \
    dataset.max_faces=30 \
    dataset.scale_factor=200 \
    trainer.devices=8 \
    trainer.wandb.enabled=true \
    trainer.exp_name=0411_deepcad_natural_blender_align \
    model.denoiser.hidden_dim=768 \
    model.padding.max_faces=30 \
    model.autoencoder.checkpoint=/mnt/d/data/ae_checkpoints/1119_deepcad_aug1_11k.ckpt \
    model.autoencoder.in_channels=6 \
    model.noise.beta_schedule=squaredcos_cap_v2 \
    hydra.job.chdir=false

# Flux
cd /mnt/d/python/src/brepnet/data/DataGeneration
DG="/mnt/d/python/src/brepnet/data/DataGeneration"
PY="${PYTHON:-python3}"
mkdir -p "${DG}/output_flux_single_view" "${DG}/logs_flux"
for rank in $(seq 0 7); do
  (
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    export CUDA_VISIBLE_DEVICES="${rank}"
    export RANK="${rank}"
    exec "${PY}" "${DG}/generate_flux_kontext_from_blender.py" \
      --render-root "${DG}/output_single_view" \
      --output-root "${DG}/output_flux_single_view" \
      --model-list-dir "${DG}/render_lists" \
      --transformer-path "/mnt/d/model/Flux1_Kontext_dev_GGUF/flux1-kontext-dev-Q8_0.gguf" \
      --base-model-path "/mnt/d/model/Flux1_Kontext_dev" \
      --num-steps 28 \
      --guidance-scale 3.5 \
      --true-cfg-scale 1.0 \
      --seed 42 \
      --skip-existing
  ) >"${DG}/logs_flux/rank_${rank}.log" 2>&1 &
done
wait
echo "done; logs under ${DG}/logs_flux/"

# Compact
python /mnt/d/python/src/brepnet/data/DataGeneration/pack_natural_npz.py \
  --render-root /mnt/d/python/src/brepnet/data/DataGeneration/output_flux_single_view \
  --target-root /mnt/d/data/deepcad_v6_cond \
  --model-list /mnt/d/python/src/brepnet/data/DataGeneration/render_lists/all.txt \
  --num-workers 8 \
  --overwrite

# Train
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m src.brepnet.train --config-name train_diffusion_white \
    trainer.check_val_every_n_epoch=1 \
    trainer.num_workers=32 \
    trainer.batch_size=64 \
    trainer.precision=bf16-mixed \
    dataset.name=Diffusion_dataset \
    dataset.latent_root=/mnt/d/data/ae_cache/1119_deepcad_aug1_11k \
    dataset.condition_root=/mnt/d/data/deepcad_v6_cond \
    dataset.train_dataset=/mnt/d/python/src/brepnet/data/DataGeneration/test_list/train.txt \
    dataset.val_dataset=/mnt/d/python/src/brepnet/data/DataGeneration/test_list/train.txt \
    dataset.test_dataset=/mnt/d/python/src/brepnet/data/DataGeneration/test_list/train.txt \
    dataset.cached_condition=false \
    dataset.is_aug=1 \
    dataset.max_faces=30 \
    dataset.scale_factor=200 \
    trainer.devices=8 \
    trainer.wandb.enabled=true \
    trainer.exp_name=0413_deepcad_natural_blender_align \
    model.denoiser.hidden_dim=768 \
    model.padding.max_faces=30 \
    model.autoencoder.checkpoint=/mnt/d/data/ae_checkpoints/1119_deepcad_aug1_11k.ckpt \
    model.autoencoder.in_channels=6 \
    model.noise.beta_schedule=squaredcos_cap_v2 \
    hydra.job.chdir=false
