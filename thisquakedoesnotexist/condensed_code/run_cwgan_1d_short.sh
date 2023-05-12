#!/bin/bash

gpus=1
name="Pipeline_Test"
epochs=3

python thisquakedoesnotexist/condensed_code/wgan_cond_1d_eval.py --gpu $gpus --name $name --epochs $epochs