#!/bin/bash

gpus=1
name="Amplitudes Test"
batch_size=128
noise_dim=100
sample_rate=20
frac_train=0.9
critic_iter=12
lr=1e-4
dt=0.05
model_file="thisquakedoesnotexist/models/gan.py"

for epochs in 1; do
    python thisquakedoesnotexist/amplitudes/amplitudes.py --gpu "$gpus" \
        --name "$name" --batch_size "$batch_size" --noise_dim "$noise_dim" \
        --sample_rate "$sample_rate" --frac_train "$frac_train" \
        --n_critic "$critic_iter" --lr "$lr" --epochs "$epochs" \
        --model_file "$model_file" --dt "$dt"
done
