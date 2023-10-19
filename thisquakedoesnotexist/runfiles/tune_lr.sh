#!/bin/bash

gpus=1
experiment_name="Tune LR"
epochs=80

for lr in 1e-3 1e-4 1e-5; do
    python thisquakedoesnotexist/amplitudes/amplitudes.py \
        --gpu "$gpus" --experiment_name "$experiment_name" --epochs "$epochs" \
        -lr "$lr"
done