#!/bin/bash

gpus=1
name="Pipeline_Test"
epochs=1

python thisquakedoesnotexist/condensed_code/wgan_eval_d1.py --gpu $gpus --name $name --epochs $epochs