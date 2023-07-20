#!/bin/bash

gpus=1
name="Pipeline_Test"
epochs=1

python thisquakedoesnotexist/condensed_code/main.py --gpu $gpus --name $name --epochs $epochs