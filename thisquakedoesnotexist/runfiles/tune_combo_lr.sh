#!/bin/bash

gpus=1
name="Combo_LR_Tuning"
epochs=1000

for lr in 1e-3 1e-4 1e-5; do
    python thisquakedoesnotexist/condensed_code/combo.py --gpu $gpus --name $name --epochs $epochs --lr $lr
done