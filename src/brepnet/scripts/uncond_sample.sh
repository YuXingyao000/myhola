# Example: ./src/brepnet/uncond.sh /mnt/d/uncond_results/uncond_gaussian_l2_epsilon_random_1m uncond_gaussian_l2_epsilon_random_1m.ckpt epsilon random 1024 4096
dir=$1
ckpt=$2
diffusion_type=$3
padding_type=$4
batch_size=$5
sample_size=$6

python -m src.brepnet.train --config-name train condition=none dataset=dummy \
    trainer.resume_from_checkpoint=/mnt/d/uncond_checkpoints/${ckpt} \
    trainer.evaluate=true \
    trainer.precision=16-mixed \
    trainer.batch_size=${batch_size} \
    trainer.test_output_dir=${dir} \
    model.noise.prediction_type=${diffusion_type} \
    model.padding.type=${padding_type} \
    model.autoencoder.gaussian_weights=1e-6 \
    model.autoencoder.sigmoid=false \
    dataset.length=${sample_size}

python -m src.brepnet.post.construct_brep --data_root ${dir} --out_root ${dir}_post --use_ray --use_cuda --num_cpus 16
