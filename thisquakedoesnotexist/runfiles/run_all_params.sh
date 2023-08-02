#!/bin/bash

gpus=1
model_name="GAN 1D"
data_file="thisquakedoesnotexist/data/japan/waveforms.npy"
attr_file="thisquakedoesnotexist/data/japan/attributes.csv"
model_file="thisquakedoesnotexist/models/gan.py"
lt=1000
dt=0.05
noise_dim=100
learning_rate=1e-4
n_critic=12
gp_lambda=10
beta1=0.0
beta2=0.9
sample_rate=20
n_cond_bins=20
epochs=2
batch_size=128
frac_train=0.9
output_dir="thisquakedoesnotexist/data/output"
print_freq=400
save_freq=1
plot_format='pdf'
tracking_uri="/home/rworreby/thisquakedoesnotexist/mlruns/"
experiment_name="Unnamed Experiment"


for epochs in 2; do
    python thisquakedoesnotexist/amplitudes/amplitudes.py \
        --gpus "$gpus" --model_name "$model_name" --data_file "$data_file" \
        --attr_file "$attr_file" --model_file "$model_file" -lt "$lt" \
        -dt "$dt" --noise_dim "$noise_dim" --learning_rate "$learning_rate" \
        --n_critic "$n_critic" --gp_lambda "$gp_lambda" --beta1 "$beta1" \
        --beta2 "$beta2" --sample_rate "$sample_rate" \
        --n_cond_bins "$n_cond_bins" --epochs "$epochs" \
        --batch_size "$batch_size" --frac_train "$frac_train" \
        --output_dir "$output_dir" --print_freq "$print_freq" \
        --save_freq "$save_freq" --plot_format "$plot_format" \
        --tracking_uri "$tracking_uri" --experiment_name "$experiment_name"
done
