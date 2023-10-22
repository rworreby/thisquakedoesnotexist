#!/bin/bash

gpus=1
experiment_name="Pipeline_Test"
epochs=2

python thisquakedoesnotexist/thisquakedoesnotexist.py \
    --gpu "$gpus" --experiment_name "$experiment_name" --epochs "$epochs"