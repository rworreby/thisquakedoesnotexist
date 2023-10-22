#!/bin/bash

gpus=1
experiment_name="Amplitudes"
batch_size=128
noise_dim=100
sample_rate=20
frac_train=0.9
critic_iter=12
lr=1e-5
dt=0.05
model_file="thisquakedoesnotexist/models/gan.py"
plot_format='pdf'

for epochs in 200; do
    python thisquakedoesnotexist/thisquakedoesnotexist.py --gpus "$gpus" \
        --experiment_name "$experiment_name" --batch_size "$batch_size" \
        --noise_dim "$noise_dim" --sample_rate "$sample_rate" \
        --frac_train "$frac_train" --n_critic "$critic_iter" -lr "$lr" \
        --epochs "$epochs" --model_file "$model_file" -dt "$dt" \
        --plot_format "$plot_format" --no_vs30_bins
done