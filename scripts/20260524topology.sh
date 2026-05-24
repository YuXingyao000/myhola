CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  /mnt/d/miniconda3/envs/img2brep/bin/python -m src.brepnet.train \
    --config-name train_diffusion_oracle_topology \
    trainer.devices=8 \
    trainer.num_workers=32 \
    trainer.batch_size=64 \
    trainer.learning_rate=1e-4 \
    trainer.precision=bf16-mixed \
    trainer.num_sanity_val_steps=0 \
    trainer.check_val_every_n_epoch=1 \
    trainer.output_dir=./outputs/diffusion_oracle_topology \
    trainer.exp_name=oracle_topology_soft_bias \
    trainer.wandb.enabled=true \
    dataset.data_root=/mnt/d/data/deepcad_v6 \
    dataset.latent_root=/mnt/d/data/ae_cache/1119_deepcad_aug1_11k \
    dataset.condition_root=/mnt/d/data/deepcad_v6_cond \
    dataset.train_dataset=src/brepnet/data/list/deduplicated_deepcad_training_7_30.txt \
    dataset.val_dataset=src/brepnet/data/list/deduplicated_deepcad_validation_7_30.txt \
    dataset.test_dataset=src/brepnet/data/list/deduplicated_deepcad_testing_7_30.txt \
    model.autoencoder.checkpoint=/mnt/d/data/ae_checkpoints/1119_deepcad_aug1_11k.ckpt \
    model.topology_bias.mode=soft_bias \
    hydra.job.chdir=false


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  /mnt/d/miniconda3/envs/img2brep/bin/python -m src.brepnet.train \
    --config-name train_diffusion_oracle_topology \
    trainer.devices=8 \
    trainer.num_workers=32 \
    trainer.batch_size=64 \
    trainer.learning_rate=1e-4 \
    trainer.precision=bf16-mixed \
    trainer.num_sanity_val_steps=0 \
    trainer.check_val_every_n_epoch=1 \
    trainer.output_dir=./outputs/diffusion_oracle_topology \
    trainer.exp_name=oracle_topology_hard_mask \
    trainer.wandb.enabled=true \
    dataset.data_root=/mnt/d/data/deepcad_v6 \
    dataset.latent_root=/mnt/d/data/ae_cache/1119_deepcad_aug1_11k \
    dataset.condition_root=/mnt/d/data/deepcad_v6_cond \
    dataset.train_dataset=src/brepnet/data/list/deduplicated_deepcad_training_7_30.txt \
    dataset.val_dataset=src/brepnet/data/list/deduplicated_deepcad_validation_7_30.txt \
    dataset.test_dataset=src/brepnet/data/list/deduplicated_deepcad_testing_7_30.txt \
    model.autoencoder.checkpoint=/mnt/d/data/ae_checkpoints/1119_deepcad_aug1_11k.ckpt \
    model.topology_bias.mode=hard_mask \
    hydra.job.chdir=false