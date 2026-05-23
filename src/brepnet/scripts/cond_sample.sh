# Example: ./src/brepnet/cond.sh /mnt/d/cond_results/pc_gaussian_24 pc_pure_2.4m.ckpt pc 0 ae0925
# Example: ./src/brepnet/cond.sh /mnt/d/cond_results/pc_gaussian_24 pc_gaussian_2.4m.ckpt pc 1e-6 ae0925_gaussian

# ./src/brepnet/cond.sh /mnt/d/gen_test_v2/cond_results/mv_gaussian_24 mv_gaussian_2.4m.ckpt multi_img 1e-6 ae0925_gaussian
# ./src/brepnet/cond.sh /mnt/d/gen_test_v2/cond_results/mv_pure_24 mv_pure_2.4m.ckpt multi_img 0 ae0925
# ./src/brepnet/cond.sh /mnt/d/gen_test_v2/cond_results/pc_gaussian_24 pc_gaussian_2.4m.ckpt pc 1e-6 ae0925_gaussian
# ./src/brepnet/cond.sh /mnt/d/gen_test_v2/cond_results/pc_pure_24 pc_pure_2.4m.ckpt pc 0 ae0925
# ./src/brepnet/cond.sh /mnt/d/gen_test_v2/cond_results/sv_pure_24 sv_pure_2.4m.ckpt single_img 0 ae0925

dir=$1
ckpt=$2
condition=$3
gaussian_weights=$4
dataset=$5

test_list=${TEST_LIST:-/mnt/d/deepcad_test_v6}
cond_root=${COND_ROOT:-/mnt/d/deepcad_v6_cond}
condition_group=${condition}
if [ "${condition}" = "pc" ]; then
    condition_group=point_cloud
elif [ "${condition}" = "txt" ]; then
    condition_group=text
fi

python -m src.brepnet.train --config-name train condition=${condition_group} \
    trainer.resume_from_checkpoint=/mnt/d/cond_checkpoints/${ckpt} \
    trainer.evaluate=true \
    trainer.precision=bf16-mixed \
    trainer.batch_size=16 \
    trainer.test_output_dir=${dir} \
    model.noise.prediction_type=sample \
    model.padding.type=zero \
    model.autoencoder.gaussian_weights=${gaussian_weights} \
    model.autoencoder.sigmoid=false \
    dataset.name=Diffusion_dataset \
    dataset.train_dataset=${test_list} \
    dataset.val_dataset=${test_list} \
    dataset.test_dataset=${test_list} \
    dataset.latent_root=/mnt/d/v6_4m/${dataset} \
    dataset.condition_root=${cond_root}

python -m src.brepnet.post.construct_brep --data_root ${dir} --out_root ${dir}_post --use_ray --use_cuda --num_cpus 16
